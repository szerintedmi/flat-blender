"""msplat training with LiDAR free-space pruning; run by msplat's own Python, not the pipeline's.

    <msplat python> msplat_train.py --input <dataset> --output <splat.ply> --views <views.npz> [...]

Same training as `msplat-train` (same defaults, `step=` lines, the same evaluation block), plus: every
`--prune-every` steps up to `--prune-until`, the splats in LiDAR free space are removed (see freespace.py)
and training continues, so the splats on the real surfaces take over the colour the removed ones carried.

msplat's Python API has no access to the splats, so pruning goes through a checkpoint: save, drop the
rows of every per-splat tensor (parameters and Adam moments), load. Format (msplat 1.1.4, little-endian):
"MSPL", u32 version, u32 step, u32 splat count, u32 SH degree, 36 bytes of settings, then tensors as
u32 ndim, u64 dims[ndim], u64 byte count, float32 data. Positions (the first tensor) are in msplat's
internal frame, a scaled and shifted copy of the dataset's (even with keep_crs, which only undoes it on
export); the transform is fitted from the cameras.

Only numpy and msplat: freespace.py is imported from this file's directory.
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import freespace

HEADER = 56


def read_checkpoint(path: Path) -> tuple[bytes, int, list[np.ndarray]]:
    raw = path.read_bytes()
    if raw[:4] != b"MSPL":
        raise ValueError(f"{path}: not an msplat checkpoint")
    n = struct.unpack_from("<I", raw, 12)[0]
    pos, tensors = HEADER, []
    while pos < len(raw):
        ndim = struct.unpack_from("<I", raw, pos)[0]
        dims = struct.unpack_from(f"<{ndim}Q", raw, pos + 4)
        nbytes = struct.unpack_from("<Q", raw, pos + 4 + 8 * ndim)[0]
        start = pos + 12 + 8 * ndim
        tensors.append(np.frombuffer(raw, "<f4", nbytes // 4, start).reshape(dims))
        pos = start + nbytes
    if not tensors or tensors[0].shape != (n, 3):
        raise ValueError(f"{path}: unexpected layout (first tensor {tensors[0].shape if tensors else None})")
    return raw[:HEADER], n, tensors


def write_checkpoint(path: Path, header: bytes, tensors: list[np.ndarray], n: int) -> None:
    with path.open("wb") as f:
        f.write(header[:12] + struct.pack("<I", n) + header[16:])
        for t in tensors:
            f.write(
                struct.pack("<I", t.ndim) + struct.pack(f"<{t.ndim}Q", *t.shape) + struct.pack("<Q", t.nbytes)
            )
            f.write(np.ascontiguousarray(t).tobytes())


def internal_frame(
    dataset: Path, cameras: np.ndarray, test_every: int, eval_mode: bool
) -> tuple[float, np.ndarray]:
    """(scale, offset) with internal = scale * dataset + offset, from the training cameras' positions:
    `cameras` (n, 3) as msplat holds them, in its order (sorted by path, every `test_every`-th held out)."""
    frames = sorted(
        json.loads((dataset / "transforms.json").read_text())["frames"], key=lambda f: f["file_path"]
    )
    train = [f for i, f in enumerate(frames) if not (eval_mode and i % test_every == 0)]
    a = np.array([np.array(f["transform_matrix"])[:3, 3] for f in train[: len(cameras)]])
    b = cameras[: len(a)]
    ac, bc = a - a.mean(0), b - b.mean(0)
    scale = float((ac * bc).sum() / (ac * ac).sum())
    offset = b.mean(0) - scale * a.mean(0)
    residual = float(np.abs(scale * a + offset - b).max())
    if residual > 1e-4 * max(1.0, float(np.abs(b).max())):
        raise RuntimeError(f"cameras don't match msplat's ({residual:.2e}): dataset order assumption broken")
    return scale, offset


def main() -> None:
    ap = argparse.ArgumentParser(description="msplat training with LiDAR free-space pruning")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--views", type=Path, help="depth views .npz (depthviews.py); without it, no pruning")
    ap.add_argument("--num-iters", type=int, default=30000)
    ap.add_argument("--downscale-factor", type=float, default=1.0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--prune-every", type=int, default=1500)
    ap.add_argument("--prune-until", type=int, default=15000)
    ap.add_argument("--min-free", type=int, default=3)
    a, rest = ap.parse_known_args()
    import msplat  # pyright: ignore[reportMissingImports]  # msplat's env only

    a.output.parent.mkdir(parents=True, exist_ok=True)

    ds = msplat.Dataset(str(a.input), a.downscale_factor, a.eval, a.test_every)
    print(f"Loaded {ds.num_train} train cameras" + (f", {ds.num_test} test cameras" if a.eval else ""))
    cfg = msplat.TrainingConfig(
        iterations=a.num_iters, keep_crs=True, downscale_factor=a.downscale_factor, output=str(a.output)
    )
    tokens = [t for arg in rest for t in arg.split("=", 1)]  # msplat-train's options: --sh-degree 2 / =2
    for key, value in zip(tokens[::2], tokens[1::2], strict=True):
        name = key.removeprefix("--").replace("-", "_")
        if not hasattr(cfg, name):
            raise SystemExit(f"unknown msplat option {key}")
        setattr(cfg, name, type(getattr(cfg, name))(value))
    trainer = msplat.GaussianTrainer(ds, cfg)
    views = freespace.load_views(a.views) if a.views else None
    cameras = np.array([np.asarray(ds.camera_pose(k))[:3, 3] for k in range(ds.num_train)])
    scale, offset = internal_frame(a.input, cameras, a.test_every, a.eval) if views else (1.0, np.zeros(3))
    log: list[dict[str, int]] = []
    checkpoint = a.output.with_suffix(".prune.ckpt")
    started = time.monotonic()
    while trainer.iteration < a.num_iters:
        stats = trainer.step()
        it = trainer.iteration
        if it % 100 == 0:
            print(f"step={it:>6}  splats={stats.splat_count:>8,}  ms={stats.ms_per_step:.1f}", flush=True)
        if views is not None and it % a.prune_every == 0 and it <= a.prune_until:
            trainer.save_checkpoint(str(checkpoint))
            header, n, tensors = read_checkpoint(checkpoint)
            centres = (tensors[0].astype(np.float64) - offset) / scale
            radius = np.minimum(np.exp(tensors[1].astype(np.float64)).max(axis=1) / scale, 0.10)
            drop = freespace.drop_mask(*freespace.votes(centres, radius, *views), a.min_free)
            keep = ~drop
            write_checkpoint(
                checkpoint, header, [t[keep] if t.shape[0] == n else t for t in tensors], int(keep.sum())
            )
            trainer.load_checkpoint(str(checkpoint))
            checkpoint.unlink()
            log.append({"step": it, "splats": n, "removed": int(drop.sum())})
            print(f"Pruned at step {it}: {int(drop.sum()):,} of {n:,} splats in LiDAR free space", flush=True)
    trainer.export_ply(str(a.output))
    print(f"Saved {a.output}")
    prune = {"views": len(views[0]) if views else 0, "removed": sum(e["removed"] for e in log), "rounds": log}
    a.output.with_name("prune.json").write_text(json.dumps(prune, indent=1) + "\n")
    if a.eval:
        m = trainer.evaluate()
        print(f"\n=== Evaluation ({m['num_test']} test views) ===")
        print(f"  PSNR:  {m['psnr']:.4f}")
        print(f"  SSIM:  {m['ssim']:.4f}")
        print(f"  L1:    {m['l1']:.4f}")
        print(f"  Gaussians: {m['num_gaussians']:,}")
    print(f"Training took {time.monotonic() - started:.1f} s")
    msplat.cleanup()


if __name__ == "__main__":
    main()
