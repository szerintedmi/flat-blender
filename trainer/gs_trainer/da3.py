"""Depth Anything 3, pose-conditioned, over windows of frames: the network half of the depth fusion (P4).

The LiDAR is accurate where it reads, but it reads only to about 5 m, not on glass, mirrors, dark or thin
things, and not at all in the stills. DA3 predicts dense depth for every pixel of every image. Given our
poses and intrinsics (`extrinsics` world-to-camera OpenCV, `intrinsics` at the image's size), it
conditions on them and aligns its own pose estimate to ours with a Sim(3) (Umeyama), dividing its depth by
that scale (`align_to_input_ext_scale=True`). So its depth comes out in metres and consistent across a
window, unlike the scale-free depth the thesis had to fuse.

How it runs here:
- Sequences: frames grouped by session and image size (DA3 centre-crops a batch to its smallest image, which
  would shift the field of view), ordered by capture time `t`. Stills sit among the video frames of their
  session, so they share windows with frames the LiDAR saw.
- Windows of `window` frames (default 16), `overlap` frames shared with the next (default half). A frame
  near a window's edge sees less context on one side, so each frame keeps the prediction of the window in
  which it sits most centrally; with half-window overlap that is always within the middle half of a
  window, except at a sequence's two ends. Picking one window rather than averaging keeps each depth map
  internally consistent (two windows can disagree on scale by a few per cent).
- Output, per frame, at DA3's processing resolution (`process_res` on the long side, rounded to its 14 px
  patch, e.g. 504x378 for 1920x1440): `prior/da3/<stem>.npz` with `depth` float16 (metres, z-depth) and
  `conf` float16 (DA3's confidence, >= 1, higher is surer; not a probability). `prior/da3/meta.json`
  records the model and window settings; a change means everything is predicted again. Frames already
  done are skipped, so an interrupted run resumes.

Measured on scan 3 (DA3-BASE, Mac MPS): the per-frame scale against the LiDAR is within ~2 % in rooms with
texture and baseline, but off by up to 30 % in some windows (and meaningless on a blank wall at 0.7 m), so
`gs_trainer.fusion` re-aligns each frame's scale to the LiDAR and measures DA3's error per frame before
trusting it.

`depth_anything_3` is imported only when a model is loaded, so this module (and the tests, with a fake
model) work without it. Weights: DA3-BASE and DA3-SMALL are Apache-2.0; DA3-GIANT/LARGE are CC BY-NC (for
comparison only, never for anything we keep).

    python -m gs_trainer.da3 --data data/<project>/splat/datasets/default
        [--model depth-anything/DA3-BASE] [--window 16]
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .data import Frame, Scene

VERSION = 1
DEFAULT_MODEL = "depth-anything/DA3-BASE"
MIN_POSED = 3  # windows with fewer frames run without poses


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str = DEFAULT_MODEL, device: str | None = None) -> Any:
    """A DA3 model from the Hugging Face hub on `device` (default: CUDA, else MPS, else CPU), in eval mode.

    Tests replace this function with a fake whose `inference` has the same signature.
    """
    try:
        from depth_anything_3.api import DepthAnything3  # pyright: ignore[reportMissingImports]
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Depth Anything 3 is not installed: install the trainer's `da3` extra "
            "(see trainer/README.md) or `pip install --no-deps git+https://github.com/ByteDance-Seed/Depth-Anything-3`"
        ) from e
    return DepthAnything3.from_pretrained(name).to(device or default_device()).eval()


def windows(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """[start, stop) windows over n items: `size` long (n if shorter), consecutive ones sharing `overlap`.

    The last window ends exactly at n (it may overlap its predecessor more), so every item is covered.
    """
    if n <= 0:
        return []
    size = min(size, n)
    stride = max(1, size - max(0, overlap))
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] != n - size:
        starts.append(n - size)
    return [(s, s + size) for s in starts]


def central_window(n: int, wins: Sequence[tuple[int, int]]) -> list[int]:
    """For each of n items, the index of the window it sits most centrally in (ties: the earlier one)."""
    best = [-1] * n
    dist = [float("inf")] * n
    for k, (a, b) in enumerate(wins):
        centre = (a + b - 1) / 2
        for j in range(a, b):
            d = abs(j - centre)
            if d < dist[j]:
                best[j], dist[j] = k, d
    return best


def capture_times(data: Path) -> dict[str, float]:
    """stem -> capture time `t` from transforms.json and split.json's extrap frames (missing: absent)."""
    tf = json.loads((data / "transforms.json").read_text())
    entries = list(tf["frames"])
    split_path = data / tf.get("split_file_path", "split.json")
    if split_path.is_file():
        entries += json.loads(split_path.read_text()).get("extrap_frames", [])
    return {Path(e["file_path"]).stem: float(e["t"]) for e in entries if "t" in e}


def sequences(scene: Scene, times: dict[str, float] | None = None) -> list[list[int]]:
    """Frame indices grouped by (session, image size), each group ordered by capture time, then stem."""
    times = times or {}
    groups: dict[tuple[str, int, int], list[int]] = {}
    for i, f in enumerate(scene.frames):
        groups.setdefault((f.session, f.w, f.h), []).append(i)
    order = lambda i: (times.get(scene.frames[i].stem, float("inf")), scene.frames[i].stem)  # noqa: E731
    return [sorted(ix, key=order) for _, ix in sorted(groups.items())]


def camera_arrays(frames: Sequence[Frame]) -> tuple[np.ndarray, np.ndarray]:
    """(world-to-camera (N, 4, 4) OpenCV, K (N, 3, 3) at each image's own size), float32, as DA3 takes them."""
    w2c = np.stack([np.linalg.inv(f.c2w) for f in frames]).astype(np.float32)
    k = np.stack([f.K for f in frames]).astype(np.float32)
    return w2c, k


def predict(
    model: Any,
    scene: Scene,
    indices: Sequence[int] | None = None,
    window: int = 16,
    overlap: int | None = None,
    process_res: int = 504,
    times: dict[str, float] | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (frame index, depth (h, w) float32 metres, conf (h, w) float32) for every frame, once each.

    `indices` limits which frames are yielded; their windows still take their neighbours as context.
    Memory stays at one window: each frame is yielded by the window it sits most centrally in, right after
    that window ran.
    """
    overlap = window // 2 if overlap is None else overlap
    wanted = set(range(len(scene.frames)) if indices is None else indices)
    for seq in sequences(scene, times):
        wins = windows(len(seq), window, overlap)
        owner = central_window(len(seq), wins)
        for k, (a, b) in enumerate(wins):
            mine = [j for j in range(a, b) if owner[j] == k and seq[j] in wanted]
            if not mine:
                continue
            frames = [scene.frames[seq[j]] for j in range(a, b)]
            w2c, ks = camera_arrays(frames)
            # A Sim(3) from fewer than 3 camera centres has no defined scale: run those unposed (scale-free;
            # the fusion re-aligns every frame's scale anyway).
            posed = len(frames) >= MIN_POSED
            pred = model.inference(
                [str(f.image_path) for f in frames],
                extrinsics=w2c if posed else None,
                intrinsics=ks if posed else None,
                align_to_input_ext_scale=True,
                process_res=process_res,
            )
            depth = np.asarray(pred.depth, np.float32)
            conf = np.asarray(pred.conf, np.float32) if pred.conf is not None else np.ones_like(depth)
            for j in mine:
                yield seq[j], depth[j - a], conf[j - a]


def da3_path(data: Path, stem: str) -> Path:
    return Path(data) / "prior" / "da3" / f"{stem}.npz"


def load(data: Path, stem: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(depth float32, conf float32) at DA3's processing resolution, or None if not predicted."""
    p = da3_path(data, stem)
    if not p.is_file():
        return None
    z = np.load(p)
    return z["depth"].astype(np.float32), z["conf"].astype(np.float32)


def run(
    data: Path,
    model: str = DEFAULT_MODEL,
    window: int = 16,
    overlap: int | None = None,
    process_res: int = 504,
    device: str | None = None,
) -> dict:
    """Predict DA3 depth for every frame of the dataset lacking one; write `prior/da3/`; return the meta."""
    data = Path(data)
    started = time.monotonic()
    overlap = window // 2 if overlap is None else overlap
    scene = Scene.load(data)
    out = data / "prior" / "da3"
    out.mkdir(parents=True, exist_ok=True)
    settings = {"version": VERSION, "model": model, "window": window, "overlap": overlap}
    settings["process_res"] = process_res
    meta_path = out / "meta.json"
    if meta_path.is_file():
        old = json.loads(meta_path.read_text())
        if any(old.get(k) != v for k, v in settings.items()):
            for p in out.glob("*.npz"):
                p.unlink()
    todo = [i for i, f in enumerate(scene.frames) if not (out / f"{f.stem}.npz").is_file()]
    load_s = infer_s = 0.0
    shape: list[int] | None = None
    if todo:
        t0 = time.monotonic()
        net = load_model(model, device)
        load_s = time.monotonic() - t0
        t0 = time.monotonic()
        for i, depth, conf in predict(net, scene, todo, window, overlap, process_res, capture_times(data)):
            stem = scene.frames[i].stem
            tmp = out / f"{stem}.tmp.npz"
            np.savez_compressed(tmp, depth=depth.astype(np.float16), conf=conf.astype(np.float16))
            os.replace(tmp, out / f"{stem}.npz")  # whole files only: a killed run resumes cleanly
            shape = list(depth.shape)
        infer_s = time.monotonic() - t0
    meta = settings | {
        "frames": len(scene.frames),
        "computed": len(todo),
        "skipped": len(scene.frames) - len(todo),
        "shape": shape,
        "seconds": round(time.monotonic() - started, 2),
        "model_load_s": round(load_s, 2),
        "s_per_frame": round(infer_s / len(todo), 3) if todo else None,
    }
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gs_trainer.da3", description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="dataset v2 directory")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default {DEFAULT_MODEL})")
    ap.add_argument("--window", type=int, default=16, help="frames per DA3 call (default 16)")
    ap.add_argument(
        "--overlap", type=int, default=None, help="frames shared by neighbouring windows (default window/2)"
    )
    ap.add_argument("--process-res", type=int, default=504, help="DA3's long side in pixels (default 504)")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: the best available)")
    args = ap.parse_args(argv)
    meta = run(args.data, args.model, args.window, args.overlap, args.process_res, args.device)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
