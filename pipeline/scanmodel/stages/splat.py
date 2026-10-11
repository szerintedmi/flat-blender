"""Stage 7, splat: export a nerfstudio-format dataset and (optionally) train it with msplat, Brush or gsplat.

Dataset v2 in data/<project>/splat/datasets/default/ (the contract is in trainer/README.md):
  transforms.json   per-image intrinsics + camera_to_world (OpenGL camera axes, scene frame, z up), the
                    camera ("video"/"still"), session and time; frames sorted by file name (so every
                    trainer holds out the same every-8th view). Train + interp frames only
  split.json        interp (every 8th of transforms.json) and extrap (the closing lap session, whose
                    frames are only listed here, so msplat and Brush never train on them)
  images/           stills (downscaled to the training size, blurry ones dropped, brightness matched to
                    the video: `brightness_gain` in their entry) + video frames, the sharpest per ~12 cm /
                    6° of camera movement (or every `every_s` with `select="time"`); LED flicker bands
                    divided out (`deband`, `scanmodel.deband`) in sessions that have them
  selection.json    how the images were chosen: thresholds, counts, sharpness, gains, dropped stems + reason
  depth/            LiDAR depth of video frames as recorded: 256x192 uint16 mm, holes are 0
  confidence/       ARKit's confidence of that depth (0/1/2), same size
  points.ply        fused points, 2 cm voxels (float xyz + uchar rgb), as the initial splat positions
  views.npz         LiDAR depth views for free-space pruning (`scanmodel.depthviews`, votes in `scanmodel.freespace`)
  mesh.ply          the TSDF mesh (float32 vertices, int32 triangles): gsplat's depth and normal priors
  test_poses.json   each test frame's pose re-solved against the training frames (`scanmodel.testposes`), so
                    the eval can tell a test frame's pose error from splat quality
transforms.json's `pose_source` and `mesh_pose_source` say which poses the cameras and the fused points + mesh
come from: "arkit-posegraph" (`scanmodel poses`), "arkit" (pose graph off) or "refine" (`scanmodel refine`).

msplat and Brush read only transforms.json (intrinsics, poses, `ply_file_path`), no depth. msplat runs
through `scanmodel/msplat_train.py` in msplat's own Python, which prunes splats in LiDAR free space during
training with `views.npz`; Brush runs as is; gsplat runs on a cloud GPU through `gs-trainer cloud train`
(trainer/). Training writes data/<project>/splat/runs/<name>/splat.ply in the scene frame (metres, z up).

Named views for side-by-side renders live in the project's scene/views.json (`scanmodel view add` from a viewer URL).

Cropping per object (objects/<id>.spz) is phase 4.
"""

import json
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
import open3d as o3d

from .. import deband as db
from .. import progress, testposes
from ..cache import console
from ..config import from_repo, rel
from ..depthviews import write_views
from ..geom import GL_CV, is_rigid
from ..io import VideoReader, read_jsonl
from ..models import bundle as b
from ..models.bundle import Intrinsics
from ..models.common import F64, mat
from ..models.derived import PosedImage, Poses

TRAINERS = ("msplat", "brush", "gsplat")
DEFAULT_TRAINER = "msplat"
# msplat keeps every training image in memory as float32 RGB, twice on the CPU plus a GPU copy, so
# 1482 images at 1920x1440 would need ~100 GB. Train at a downscale that keeps the sum of image pixels
# under this (~3 GB per copy); on a 24 GB Mac that is 1/4 resolution for ~1500 images.
MSPLAT_MAX_PIXELS = 260_000_000
# Every 8th frame of transforms.json (index % 8 == 0) is the `interp` test set: the views msplat
# (`--test-every 8`) and Brush (`--eval-split-every 8`) hold out themselves, so all trainers score alike.
INTERP_EVERY = 8
DATASET_VERSION = 2


def frame_entry(
    im: PosedImage, k: Intrinsics, file: str, depth: str | None = None, confidence: str | None = None
) -> dict[str, object]:
    c2w_gl = mat(im.camera_to_world) @ GL_CV
    c2w_gl[3] = [0.0, 0.0, 0.0, 1.0]
    e: dict[str, object] = {
        "file_path": file,
        "transform_matrix": c2w_gl.tolist(),
        "fl_x": k.fx,
        "fl_y": k.fy,
        "cx": k.cx,
        "cy": k.cy,
        "w": k.width,
        "h": k.height,
        "camera": "still" if im.kind == "still" else "video",  # two cameras: colour correction per camera
        "session": im.session,
        "t": im.t,
    }
    if depth:
        e["depth_file_path"] = depth
    if confidence:
        e["confidence_file_path"] = confidence
    return e


def stem(frame: dict[str, object]) -> str:
    """`images/<stem>.jpg` -> `<stem>`: the frame's name in split.json, depth/ and confidence/."""
    return Path(str(frame["file_path"])).stem


def extrap_session_of(sessions: Iterable[str], choice: str = "auto") -> str | None:
    """The session held out of training as the `extrap` test set.

    "auto": the closing lap (a session whose name ends in `-lap`), which walks through every room once
    more along a path no training view took, so it shows the needles and doubles an every-8th view hides.
    Auto keeps a sole session in training even if its name ends in `-lap`.
    "none": no extrap set. Anything else names the session.
    """
    names = sorted(set(sessions))
    if choice == "none":
        return None
    if choice == "auto":
        return next((s for s in names if s.endswith("-lap")), None) if len(names) > 1 else None
    if choice not in names:
        raise ValueError(f"no session {choice!r} to hold out; sessions: {', '.join(names)}")
    if len(names) == 1:
        raise ValueError(f"cannot hold out the only session {choice!r}; no training sessions would remain")
    return choice


def split(
    frames: list[dict[str, object]], extrap_session: str | None
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """(transforms.json frames, split.json) from all exported frames.

    Frames are sorted by file name: msplat sorts its cameras by path before taking every nth for eval and
    Brush splits in file order, so sorting here makes both hold out exactly the `interp` frames. The extrap
    session's frames are left out of transforms.json (msplat and Brush read nothing else).
    """
    frames = sorted(frames, key=lambda f: str(f["file_path"]))
    train = [f for f in frames if f["session"] != extrap_session]
    extrap = [f for f in frames if f["session"] == extrap_session]
    return train, {
        "interp": [stem(f) for f in train[::INTERP_EVERY]],
        "extrap": [stem(f) for f in extrap],
        "extrap_session": extrap_session,
        "extrap_frames": extrap,
    }


def _resize(img: np.ndarray, k: Intrinsics, max_side: int) -> tuple[np.ndarray, Intrinsics]:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    if scale == 1.0:
        return img, k.scaled(w, h)
    nw, nh = round(w * scale), round(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), k.scaled(nw, nh)


def write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary PLY with float32 x/y/z and uchar red/green/blue: what COLMAP-style splat inits read.

    Open3D writes doubles and normals, which msplat reads but other loaders may not.
    """
    n = len(xyz)
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    rec = np.empty(n, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"] = xyz
    rec["c"] = rgb
    with path.open("wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[2])
            if line == "end_header":
                break
    return 0


def write_mesh_ply(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    """Binary PLY with float32 x/y/z vertices and int32 triangles, nothing else.

    fuse/dense.ply carries doubles, normals and colours (370 MB for scan 3); the trainer only raycasts it
    for depth and normals, so this keeps the upload to half that.
    """
    n, m = len(vertices), len(triangles)
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {m}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    faces = np.empty(m, dtype=[("n", "u1"), ("v", "<i4", 3)])
    faces["n"] = 3
    faces["v"] = triangles
    with path.open("wb") as f:
        f.write(header.encode())
        f.write(np.ascontiguousarray(vertices, dtype="<f4").tobytes())
        f.write(faces.tobytes())


# --------------------------------------------------------------------------- input quality (P6 a1, a2)

SELECT_MODES = ("motion", "time")
SHARP_SIDE = 480  # sharpness is measured on a grey image downscaled to this long side
LOCAL_WINDOW_S = 1.0  # relative sharpness: against the median of the frames within ± this
BLUR_WINDOW_S = 2.0  # blur drop, `blur_ref="picked"`: against the median of the picked frames within ± this
BLUR_REFS = ("neighbours", "picked")
STILL_VIDEO_WINDOW_S = 1.0  # a still against the video frames within ± this
GAIN_CLIP = 0.25  # brightness gains stay within 1 ± this
WELL_EXPOSED = (16, 240)  # 8-bit range counted as well exposed for the gains


def _shrink(img: np.ndarray, side: int) -> np.ndarray:
    """`img` downscaled (area) so its long side is at most `side`."""
    h, w = img.shape[:2]
    scale = side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def sharpness(img: np.ndarray, side: int = SHARP_SIDE) -> float:
    """Variance of the 4-neighbour Laplacian of the grey image downscaled to long side `side`.

    The same measure as the app's stills (BundleKit `Sharpness.swift`), at a smaller size so that every
    video frame can be scored cheaply. Texture-dependent: compare only within a place.
    """
    img = _shrink(img, side)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(grey.astype(np.float32), cv2.CV_32F, ksize=1).var())


def video_sharpness(video: Path, cache: Path, side: int = SHARP_SIDE) -> tuple[np.ndarray, float]:
    """Sharpness of every frame of `video` (index = frame index, NaN past the decodable end) and the
    seconds spent decoding (0 when read from `cache`, a .npy with a .json describing the video next to it).

    One sequential decode: seeking HEVC is slow and inexact.
    """
    st = video.stat()
    meta = {"video": rel(video), "bytes": st.st_size, "mtime": st.st_mtime, "side": side}
    meta_file = cache.with_suffix(".json")
    if cache.is_file() and meta_file.is_file() and json.loads(meta_file.read_text()) == meta:
        return np.load(cache), 0.0
    started = time.monotonic()
    reader = VideoReader(video)
    n = reader.frame_count()
    values: list[float] = []
    with progress.bar("frames") as bar:
        task = bar.add_task(f"sharpness {video.parent.name}", total=n)
        while True:
            ok, img = reader.cap.read()
            if not ok:
                break
            values.append(sharpness(img, side))
            bar.advance(task)
        bar.update(task, total=len(values))
    reader.close()
    out = np.full(max(n, len(values)), np.nan, np.float32)
    out[: len(values)] = values
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    meta_file.write_text(json.dumps(meta) + "\n")
    return out, time.monotonic() - started


def _window_median(t: np.ndarray, values: np.ndarray, at: np.ndarray, half: float) -> np.ndarray:
    """Median of `values` (sorted by time `t`) within `at` ± `half`, for each time in `at`."""
    lo = np.searchsorted(t, at - half, side="left")
    hi = np.searchsorted(t, at + half, side="right")
    return np.array([np.median(values[a:b]) if b > a else np.nan for a, b in zip(lo, hi, strict=True)])


def relative_sharpness(t: np.ndarray, s: np.ndarray, half: float = LOCAL_WINDOW_S) -> np.ndarray:
    """Sharpness over the median of the frames within ± `half` s: texture differences between places
    cancel, so a plain wall and a bookshelf can be compared."""
    med = _window_median(t, s, t, half)
    return s / np.maximum(med, 1e-9)


def select_time(t: np.ndarray, every_s: float) -> list[int]:
    """The first frame of every `every_s` (the export before P6): indices into `t`."""
    picked: list[int] = []
    for i, ti in enumerate(t):
        if not picked or ti - t[picked[-1]] >= every_s:
            picked.append(i)
    return picked


def rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the rotation between two 3x3 rotations, degrees."""
    c = (float(np.trace(a.T @ b)) - 1) / 2
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def motion_slots(
    t: np.ndarray, c2w: np.ndarray, min_move: float, min_turn: float, max_gap_s: float
) -> list[np.ndarray]:
    """Consecutive runs of frames (indices into `t`, in time order): a new slot opens when the camera has
    moved ≥ `min_move` m or turned ≥ `min_turn` degrees since the first frame of the current slot, or
    `max_gap_s` has passed (so slow pans and standing still are still sampled)."""
    starts = [0] if len(t) else []
    for i in range(1, len(t)):
        a = starts[-1]
        if (
            t[i] - t[a] >= max_gap_s
            or float(np.linalg.norm(c2w[i, :3, 3] - c2w[a, :3, 3])) >= min_move
            or rotation_deg(c2w[a, :3, :3], c2w[i, :3, :3]) >= min_turn
        ):
            starts.append(i)
    bounds = [*starts, len(t)]
    return [np.arange(bounds[k], bounds[k + 1]) for k in range(len(starts))]


def drop_blurry(
    picked: list[int],
    t: np.ndarray,
    s: np.ndarray,
    rel_blur: float,
    max_gap_s: float,
    ref: np.ndarray | None = None,
    half: float = BLUR_WINDOW_S,
) -> tuple[list[int], dict[int, str]]:
    """Drop picked frames (indices into `t`, time order) with sharpness < `rel_blur` x their reference:
    `ref[i]` (the median of all frames around frame i, `blur_ref="neighbours"`) or, without `ref`, the
    median of the picked frames within ± `half` s (`"picked"`; texture changes between the picks then
    look like blur). Unless that leaves no frame for more than 2 x `max_gap_s` (the session's first and
    last frames count as edges): never leave an area unseen.
    Returns (kept, {dropped or rescued index: reason})."""
    if not picked:
        return [], {}
    pt = t[picked]
    ps = s[picked]
    med = ref[picked] if ref is not None else _window_median(pt, ps, pt, half)
    what = (
        f"of all frames within ±{LOCAL_WINDOW_S:g} s" if ref is not None else f"of picked within ±{half:g} s"
    )
    kept: list[int] = []
    notes: dict[int, str] = {}
    for j, i in enumerate(picked):
        if not ps[j] < rel_blur * med[j]:
            kept.append(i)
            continue
        prev_t = t[kept[-1]] if kept else t[0]
        next_t = pt[j + 1] if j + 1 < len(picked) else t[-1]
        if next_t - prev_t > 2 * max_gap_s:
            kept.append(i)
            notes[i] = f"kept: blurry ({ps[j]:.0f} < {rel_blur:g} x {med[j]:.0f}) but the only view"
        else:
            notes[i] = f"blurry: {ps[j]:.0f} < {rel_blur:g} x median {med[j]:.0f} {what}"
    return kept, notes


def select_motion(
    t: np.ndarray,
    c2w: np.ndarray,
    s: np.ndarray,
    min_move: float,
    min_turn: float,
    max_gap_s: float,
    rel_blur: float,
    blur_ref: str = "neighbours",
    all_t: np.ndarray | None = None,
    all_s: np.ndarray | None = None,
) -> tuple[list[int], dict[int, str], int]:
    """P6 a1: movement-based slots, the sharpest frame of each (relative to the local median), then the
    blur drop against `blur_ref`: "neighbours" = the median of every frame within ± LOCAL_WINDOW_S
    (`all_t`/`all_s`, time-sorted: also the frames that aren't candidates; default the candidates),
    "picked" = the median of the picked frames within ± BLUR_WINDOW_S.
    Returns (kept indices into `t`, notes from `drop_blurry`, number of slots)."""
    if blur_ref not in BLUR_REFS:
        raise ValueError(f"blur_ref must be one of {', '.join(BLUR_REFS)}, got {blur_ref!r}")
    slots = motion_slots(t, c2w, min_move, min_turn, max_gap_s)
    rel = relative_sharpness(t, s)
    picked = [int(slot[np.argmax(rel[slot])]) for slot in slots]
    ref = None
    if blur_ref == "neighbours":
        at, av = (all_t, all_s) if all_t is not None and all_s is not None else (t, s)
        ref = _window_median(at, av, t, LOCAL_WINDOW_S)
    kept, notes = drop_blurry(picked, t, s, rel_blur, max_gap_s, ref)
    return kept, notes, len(slots)


def blurry_still(s: float, room_median: float, video_median: float, rel_blur: float) -> bool:
    """A still is dropped when it is blurrier than `rel_blur` x its room's median still AND blurrier than
    the video frames around it (NaN: no video, the room decides): a soft still that is still sharper
    than the video at that spot adds detail."""
    return s < rel_blur * room_median and not s >= video_median


def brightness_gain(
    img: np.ndarray, ref: np.ndarray, clip: float = GAIN_CLIP, exposed: tuple[int, int] = WELL_EXPOSED
) -> np.ndarray:
    """Per-channel gains (in the images' channel order) that match `img`'s brightness to `ref`'s.

    Both are resized to the smaller one's size; the gain per channel is the ratio of the medians over the
    pixels well exposed (all channels within `exposed`) in both, clipped to 1 ± `clip`. Global, so the
    slightly different views of a still and its nearest video frame don't matter.
    """
    h, w = min(img.shape[0], ref.shape[0]), min(img.shape[1], ref.shape[1])
    a = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if img.shape[:2] != (h, w) else img
    r = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA) if ref.shape[:2] != (h, w) else ref
    lo, hi = exposed
    ok = np.all((a >= lo) & (a <= hi), axis=2) & np.all((r >= lo) & (r <= hi), axis=2)
    if ok.sum() < 0.01 * h * w:
        return np.ones(a.shape[2])
    gain = np.median(r[ok].astype(np.float64), axis=0) / np.maximum(
        np.median(a[ok].astype(np.float64), axis=0), 1
    )
    return np.clip(gain, 1 - clip, 1 + clip)


def apply_gain(img: np.ndarray, gain: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float32) * gain.astype(np.float32), 0, 255).round().astype(np.uint8)


def _percentiles(x: Iterable[float]) -> dict[str, float]:
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if not len(a):
        return {}
    p = np.percentile(a, [10, 50, 90])
    return {
        "n": len(a),
        "p10": round(float(p[0]), 1),
        "median": round(float(p[1]), 1),
        "p90": round(float(p[2]), 1),
    }


def _frame_idx(im: PosedImage) -> int:
    return int(im.file.split("#")[1])


def export(
    poses_dir: Path,
    fuse_dir: Path,
    out: Path,
    every_s: float = 0.5,
    max_side: int = 1920,
    extrap_session: str = "auto",
    select: str = "motion",
    min_move: float = 0.12,
    min_turn: float = 6.0,
    max_gap_s: float = 1.0,
    rel_blur: float = 0.6,
    blur_ref: str = "neighbours",
    sharpness_dir: Path | None = None,
    deband: str = "auto",
    mains_hz: float = db.MAINS_HZ,
    deband_dir: Path | None = None,
    pose_source: str | None = None,
    mesh_pose_source: str | None = None,
    test_poses: bool = True,
) -> dict[str, object]:
    """Write dataset v2 to `out` (see the module docstring and trainer/README.md).

    Video frames: `select` "motion" (P6 a1: a slot per `min_move` m / `min_turn` degrees / `max_gap_s`, the
    sharpest frame of each, then frames below `rel_blur` x their reference dropped unless they're the only
    view; `blur_ref` "neighbours": every frame within ±1 s, "picked": the picked frames within ±2 s) or "time" (the first frame of every `every_s`). Stills (P6 a2): dropped when below `rel_blur` x
    their room's median sharpness and below the video around them; brightness matched to their nearest
    video frame. Per-frame sharpness is cached in `sharpness_dir` (default `out`/../sharpness).
    `deband`: divide LED flicker bands out of the frames and stills (`scanmodel.deband`) of every session ("on"),
    none ("off"), or the sessions `scanmodel.deband.detect` finds banded ("auto", cached in `deband_dir`, default
    `out`/../deband); `mains_hz` sets the flicker frequency (twice this).
    `selection.json` records counts, thresholds and every dropped image with its reason.
    `pose_source` / `mesh_pose_source` (e.g. "arkit-posegraph", "refine"): where the camera poses and the
    poses the fused points and mesh were made from came from, stamped into transforms.json and selection.json.
    `test_poses`: re-solve the test frames' poses against the training frames into test_poses.json.
    """
    if select not in SELECT_MODES:
        raise ValueError(f"select must be one of {', '.join(SELECT_MODES)}, got {select!r}")
    if blur_ref not in BLUR_REFS:
        raise ValueError(f"blur_ref must be one of {', '.join(BLUR_REFS)}, got {blur_ref!r}")
    if deband not in db.DEBAND_MODES:
        raise ValueError(f"deband must be one of {', '.join(db.DEBAND_MODES)}, got {deband!r}")
    sharpness_dir = sharpness_dir or out.parent / "sharpness"
    deband_dir = deband_dir or out.parent / "deband"
    light_hz = 2 * mains_hz
    poses = Poses.model_validate_json((poses_dir / "poses.json").read_text())
    raw = {s.session: from_repo(s.raw_dir) for s in poses.sessions}
    held_out = extrap_session_of(raw, extrap_session)
    if out.exists():
        shutil.rmtree(out)
    for d in ("images", "depth", "confidence"):
        (out / d).mkdir(parents=True)
    frames: list[dict[str, object]] = []

    stills = list(read_jsonl(poses_dir / "stills.jsonl", PosedImage))
    all_frames: dict[str, list[PosedImage]] = {}  # every posed frame: times for the stills' neighbourhoods
    by_session: dict[str, list[PosedImage]] = {}  # the candidates: tracking normal, with depth
    for im in read_jsonl(poses_dir / "frames.jsonl", PosedImage):
        all_frames.setdefault(im.session, []).append(im)
        if im.tracking in (None, b.TrackingState.NORMAL) and im.depth:
            by_session.setdefault(im.session, []).append(im)

    # Sharpness of every video frame (cached), then the selection per session.
    sharp: dict[str, np.ndarray] = {}
    decode_s: dict[str, float] = {}
    for sid in sorted(by_session):
        video = raw[sid] / b.VIDEO_FILE
        if video.is_file():
            sharp[sid], decode_s[sid] = video_sharpness(video, sharpness_dir / f"{sid}.npy")

    # LED flicker: which sessions to deband.
    debands: dict[str, dict[str, object]] = {}
    for sid in sorted(sharp):
        if deband == "off":
            continue
        found = db.detect_cached(raw[sid] / b.VIDEO_FILE, deband_dir / f"{sid}.json", light_hz)
        debands[sid] = {"detected": found, "applied": deband == "on" or bool(found["banded"])}
        console.print(
            f"  banding {sid}: {100 * float(found['amplitude']):.1f} %"  # type: ignore[arg-type]
            f" -> {'deband' if debands[sid]['applied'] else 'leave'}",
            highlight=False,
        )

    picked: dict[str, list[PosedImage]] = {}
    sessions: dict[str, dict[str, object]] = {}
    dropped: dict[str, str] = {}  # stem -> reason
    rescued: dict[str, str] = {}
    sharp_before: list[float] = []
    sharp_time: list[float] = []
    sharp_after: list[float] = []
    for sid, ims in by_session.items():
        if sid not in sharp:
            continue
        ims = sorted(ims, key=lambda im: im.t or 0)
        sv = sharp[sid]
        ims = [im for im in ims if _frame_idx(im) < len(sv) and np.isfinite(sv[_frame_idx(im)])]
        if not ims:
            continue
        t = np.array([im.t or 0 for im in ims])
        s = np.array([sv[_frame_idx(im)] for im in ims], dtype=np.float64)
        by_time = select_time(t, every_s)
        if select == "time":
            kept, notes, n_slots = by_time, {}, len(by_time)
        else:
            c2w = np.array([im.camera_to_world for im in ims], dtype=np.float64)
            every = sorted(
                (f.t or 0, float(sv[_frame_idx(f)]))
                for f in all_frames[sid]
                if _frame_idx(f) < len(sv) and np.isfinite(sv[_frame_idx(f)])
            )
            all_t, all_s = (np.array(x, dtype=np.float64) for x in zip(*every, strict=True))
            kept, notes, n_slots = select_motion(
                t, c2w, s, min_move, min_turn, max_gap_s, rel_blur, blur_ref, all_t, all_s
            )
        picked[sid] = [ims[i] for i in kept]
        for i, note in notes.items():
            (rescued if note.startswith("kept") else dropped)[f"{sid}_{_frame_idx(ims[i]):06d}"] = note
        sharp_before.extend(s.tolist())
        sharp_time.extend(s[by_time].tolist())
        sharp_after.extend(s[kept].tolist())
        sessions[sid] = {
            "candidates": len(ims),
            "time_every_s": len(by_time),
            "slots": n_slots,
            "dropped_blurry": sum(not n.startswith("kept") for n in notes.values()),
            "kept_only_view": sum(n.startswith("kept") for n in notes.values()),
            "video": len(kept),
            "sharpness_candidates": _percentiles(s),
            "sharpness_time_every_s": _percentiles(s[by_time]),
            "sharpness_video": _percentiles(s[kept]),
            "decode_s": round(decode_s[sid], 1),
        }

    # Stills (P6 a2): sharpness at the training size, compared with the room and the video around them.
    raw_stills = {sid: {st.id: st for st in read_jsonl(raw[sid] / b.STILLS_FILE, b.Still)} for sid in raw}
    still_sharp: dict[str, float] = {}
    for im in stills:
        img = cv2.imread(str(raw[im.session] / im.file))
        if img is not None:
            still_sharp[im.id] = sharpness(_resize(img, im.intrinsics, max_side)[0])
    room_median = {
        sid: float(np.median([v for k, v in still_sharp.items() if k.startswith(f"{sid}/")]))
        for sid in {im.session for im in stills if im.id in still_sharp}
    }
    kept_stills: list[PosedImage] = []
    nearest: dict[str, int] = {}  # still id -> video frame idx for its brightness
    still_notes: dict[str, dict[str, object]] = {}
    for im in stills:
        name = f"{im.session}_s{im.id.split('/s')[-1]}"
        if im.id not in still_sharp:
            dropped[name] = "unreadable"
            continue
        sv_still = still_sharp[im.id]
        video_med = math.nan
        posed = all_frames.get(im.session, [])
        if im.session in sharp and posed and im.t is not None:
            ft = np.array([f.t or 0 for f in posed])
            near = [_frame_idx(f) for f in posed if abs((f.t or 0) - im.t) <= STILL_VIDEO_WINDOW_S]
            vals = [float(sharp[im.session][i]) for i in near if i < len(sharp[im.session])]
            vals = [v for v in vals if math.isfinite(v)]
            video_med = float(np.median(vals)) if vals else math.nan
            raw_still = raw_stills[im.session].get(int(im.id.split("/s")[-1]))
            nf = raw_still.nearest_frame if raw_still else None
            nearest[im.id] = nf if nf is not None else _frame_idx(posed[int(np.argmin(np.abs(ft - im.t)))])
        room = room_median[im.session]
        still_notes[name] = {"sharpness": round(sv_still, 1), "room_median": round(room, 1)}
        if math.isfinite(video_med):
            still_notes[name]["video_median"] = round(video_med, 1)
        if blurry_still(sv_still, room, video_med, rel_blur):
            dropped[name] = (
                f"blurry still: {sv_still:.0f} < {rel_blur:g} x room median {room:.0f}"
                f" and < video median {video_med:.0f} within ±{STILL_VIDEO_WINDOW_S:g} s"
            )
            continue
        kept_stills.append(im)

    n_video = sum(map(len, picked.values()))
    rule = (
        f"every {every_s:g} s"
        if select == "time"
        else f"motion {min_move:g} m / {min_turn:g}° / {max_gap_s:g} s, blur < {rel_blur:g}x dropped"
    )
    console.print(
        f"  {len(kept_stills)}/{len(stills)} stills + {n_video} video frames ({rule}, with depth)"
        f" from {len(picked)} sessions, max side {max_side}px; extrap: {held_out or 'none'}",
        highlight=False,
    )

    refs: dict[str, np.ndarray] = {}  # still id -> its nearest video frame, small, for the gains
    still_fields: dict[str, tuple[db.Field, PosedImage]] = {}  # still id -> its nearest frame's flicker
    with progress.bar("images") as bar:
        task = bar.add_task("video", total=len(kept_stills) + n_video)
        for sid, ims in picked.items():
            bar.update(task, description=sid)
            want: dict[int, PosedImage | None] = {_frame_idx(im): im for im in ims}
            ref_of: dict[
                int, list[str]
            ] = {}  # video frame idx -> the stills it is the brightness reference of
            for st in kept_stills:
                if st.session == sid and st.id in nearest:
                    ref_of.setdefault(nearest[st.id], []).append(st.id)
                    want.setdefault(nearest[st.id], None)
            posed = {_frame_idx(f): f for f in all_frames.get(sid, []) if f.t is not None}
            fix = bool(debands.get(sid, {}).get("applied"))
            n_fixed, n_skipped, amps = 0, 0, list[float]()
            for idx, img, nb in _frames_with_neighbours(
                raw[sid] / b.VIDEO_FILE,
                sorted(want),
                {i: f.t for i, f in posed.items() if f.t is not None} if fix else {},
                light_hz,
            ):
                if fix and nb is None:
                    n_skipped += 1  # no usable neighbours: left as it is
                elif fix and nb is not None:
                    cur = _view(posed[idx], img)
                    img, field = db.deband(cur, [_view(posed[j], nb[j]) for j in sorted(nb)], light_hz)
                    n_fixed += 1
                    amps.append(100 * field.amplitude)
                    for still_id in ref_of.get(idx, []):
                        still_fields[still_id] = (field, posed[idx])
                for still_id in ref_of.get(idx, []):
                    refs[still_id] = _shrink(img, SHARP_SIDE)
                im = want[idx]
                if im is None:
                    continue
                img, k = _resize(img, im.intrinsics, max_side)
                name = f"{sid}_{idx:06d}"
                cv2.imwrite(str(out / f"images/{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                # Depth and confidence as recorded (256x192, uint16 mm / uint8): the trainer upsamples and
                # masks on the GPU, and can switch confidence levels during training.
                assert im.depth is not None
                shutil.copyfile(raw[sid] / im.depth, out / f"depth/{name}.png")
                conf = None
                if im.confidence:
                    conf = f"confidence/{name}.png"
                    shutil.copyfile(raw[sid] / im.confidence, out / conf)
                frames.append(frame_entry(im, k, f"images/{name}.jpg", f"depth/{name}.png", conf))
                bar.advance(task)
            if sid in debands:
                debands[sid] |= {
                    "frames": n_fixed,
                    "skipped": n_skipped,
                    "field_amplitude": _percentiles(amps),
                }

        bar.update(task, description="stills")
        gains: list[np.ndarray] = []
        for im in kept_stills:
            bar.advance(task)
            img = cv2.imread(str(raw[im.session] / im.file))
            if img is None:
                continue
            img, k = _resize(img, im.intrinsics, max_side)
            name = f"{im.session}_s{im.id.split('/s')[-1]}"
            if im.id in still_fields:
                field, frame = still_fields[im.id]
                gain = db.still_log_gain(
                    field, _view(frame, img), _view(im, img), (img.shape[1], img.shape[0])
                )
                img = db.apply(img, gain)
                still_notes[name]["debanded"] = True
            gain_bgr = brightness_gain(img, refs[im.id]) if im.id in refs else np.ones(3)
            gains.append(gain_bgr[::-1])
            cv2.imwrite(
                str(out / f"images/{name}.jpg"), apply_gain(img, gain_bgr), [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            e = frame_entry(im, k, f"images/{name}.jpg")
            e["brightness_gain"] = [round(float(g), 4) for g in gain_bgr[::-1]]
            frames.append(e)
            still_notes[name]["brightness_gain"] = e["brightness_gain"]
        bar.update(task, description="images", total=bar.tasks[task].completed)  # a video ended early

    train_frames, split_json = split(frames, held_out)
    console.print("  initial points: fused points at 2 cm voxels", highlight=False)
    pc = o3d.io.read_point_cloud(fuse_dir / "points.ply").voxel_down_sample(0.02)
    xyz = np.asarray(pc.points)
    rgb = (
        (np.asarray(pc.colors) * 255).round().astype(np.uint8)
        if pc.has_colors()
        else np.full((len(xyz), 3), 128, np.uint8)
    )
    write_points_ply(out / "points.ply", xyz, rgb)
    console.print("  mesh: fuse/dense.ply as float32 vertices + int32 triangles", highlight=False)
    mesh = o3d.io.read_triangle_mesh(fuse_dir / "dense.ply")
    triangles = np.asarray(mesh.triangles)
    write_mesh_ply(out / "mesh.ply", np.asarray(mesh.vertices), triangles)
    del mesh
    n_views = write_views(poses_dir, out / "views.npz")
    console.print(f"  pruning: {n_views} LiDAR depth views -> views.npz", highlight=False)
    (out / "split.json").write_text(json.dumps(split_json, indent=1) + "\n")
    transforms = {
        "version": DATASET_VERSION,
        "camera_model": "OPENCV",
        "depth_unit_scale_factor": 0.001,
        "ply_file_path": "points.ply",
        "mesh_file_path": "mesh.ply",
        "split_file_path": "split.json",
        **({"pose_source": pose_source} if pose_source else {}),
        **({"mesh_pose_source": mesh_pose_source} if mesh_pose_source else {}),
        "frames": train_frames,
    }
    (out / "transforms.json").write_text(json.dumps(transforms, indent=1) + "\n")
    g = np.array(gains) if gains else np.ones((0, 3))
    n_stills = sum(f["camera"] == "still" for f in frames)
    n_video_written = len(frames) - n_stills
    selection = {
        "select": select,
        "pose_source": pose_source,
        "mesh_pose_source": mesh_pose_source,
        "thresholds": {
            "every_s": every_s,
            "min_move_m": min_move,
            "min_turn_deg": min_turn,
            "max_gap_s": max_gap_s,
            "rel_blur": rel_blur,
            "sharpness_side": SHARP_SIDE,
            "blur_ref": blur_ref,
            "local_window_s": LOCAL_WINDOW_S,
            "blur_window_s": BLUR_WINDOW_S,
            "still_video_window_s": STILL_VIDEO_WINDOW_S,
            "gain_clip": GAIN_CLIP,
            "well_exposed": list(WELL_EXPOSED),
        },
        "counts": {
            "video_candidates": len(sharp_before),
            "video_time_every_s": len(sharp_time),
            "video": n_video_written,
            "stills_posed": len(stills),
            "stills": n_stills,
            "dropped": len(dropped),
            "kept_only_view": len(rescued),
        },
        "sharpness": {
            "candidates": _percentiles(sharp_before),
            "time_every_s": _percentiles(sharp_time),
            "video": _percentiles(sharp_after),
            "stills": _percentiles(still_sharp.values()),
        },
        "brightness_gain": {
            "stills": len(g),
            "median_rgb": [round(float(x), 4) for x in np.median(g, axis=0)] if len(g) else None,
            "p10_rgb": [round(float(x), 4) for x in np.percentile(g, 10, axis=0)] if len(g) else None,
            "p90_rgb": [round(float(x), 4) for x in np.percentile(g, 90, axis=0)] if len(g) else None,
            "clipped": int(np.sum(np.any(np.abs(g - 1) >= GAIN_CLIP - 1e-9, axis=1))),
        },
        "deband": {
            "mode": deband,
            "light_hz": light_hz,
            "auto_threshold": db.AUTO_THRESHOLD,
            "sessions": debands,
            "stills": len(still_fields),
        },
        "sessions": sessions,
        "stills": still_notes,
        "dropped": dict(sorted(dropped.items())),
        "kept_only_view": dict(sorted(rescued.items())),
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    tp_summary = testposes.write(out)["summary"] if test_poses else None
    return {
        "select": select,
        "video": n_video_written,
        "stills": n_stills,
        "dropped": len(dropped),
        "images": len(frames),
        "train": len(train_frames) - len(train_frames[::INTERP_EVERY]),
        "interp": len(train_frames[::INTERP_EVERY]),
        "extrap": len(frames) - len(train_frames),
        "extrap_session": held_out,
        "with_depth": sum("depth_file_path" in f for f in frames),
        "debanded": sum(int(d.get("frames", 0)) for d in debands.values()),  # type: ignore[call-overload]
        "init_points": len(xyz),
        "mesh_triangles": len(triangles),
        "views": n_views,
        "pose_source": pose_source,
        "mesh_pose_source": mesh_pose_source,
        "test_poses": tp_summary,
    }


def _view(im: PosedImage, img: np.ndarray) -> db.View:
    return db.View(img, im.t or 0.0, mat(im.camera_to_world), im.intrinsics)


def _frames_with_neighbours(
    video: Path, want: list[int], times: dict[int, float], light_hz: float
) -> Iterable[tuple[int, np.ndarray, dict[int, np.ndarray] | None]]:
    """(idx, image, {neighbour idx: image} or None) for each wanted frame index (ascending), in one
    sequential decode. Neighbours (`scanmodel.deband.neighbours`) are only looked for among `times` (frame index ->
    time); with empty `times` none are read."""
    nbs = {i: db.neighbours(i, times, light_hz) if times else None for i in want}
    needed = sorted({*want, *(j for n in nbs.values() if n for j in n)})
    reader = VideoReader(video)
    buf: dict[int, np.ndarray] = {}
    pending = list(want)

    def ready(i: int) -> tuple[int, np.ndarray, dict[int, np.ndarray] | None]:
        n = nbs[i]
        return i, buf[i], ({x: buf[x] for x in n} if n and all(x in buf for x in n) else None)

    try:
        for j in needed:
            img = reader.read(j)
            if img is None:
                break
            buf[j] = img
            while pending and max([pending[0], *(nbs[pending[0]] or ())]) <= j:
                i = pending.pop(0)
                if i in buf:
                    yield ready(i)
            # frames still needed: the next few pending ones and their neighbours (within ±2)
            low = min([j + 1, *(x for i in pending[:3] for x in (i, *(nbs[i] or ())))])
            for x in [x for x in buf if x < low]:
                del buf[x]
        for i in pending:  # the video ended before a neighbour
            if i in buf:
                yield ready(i)
    finally:
        reader.close()


def msplat_downscale(dataset: Path, max_pixels: int = MSPLAT_MAX_PIXELS) -> float:
    """Smallest downscale factor (>= 1) that keeps msplat's in-memory images under `max_pixels`."""
    frames = json.loads((dataset / "transforms.json").read_text())["frames"]
    pixels = sum(int(f["w"]) * int(f["h"]) for f in frames)
    return max(1.0, math.ceil(math.sqrt(pixels / max_pixels) * 4) / 4)


MSPLAT_SCRIPT = Path(__file__).parent.parent / "msplat_train.py"
TRAINER_PROJECT = Path(__file__).resolve().parents[3] / "trainer"  # gs-trainer: its own uv project


def script_python(exe: str) -> str:
    """The Python interpreter of a console script (msplat-train's shebang): msplat lives in its own env."""
    first = Path(exe).read_bytes()[:512].split(b"\n")[0].decode(errors="replace")
    if not first.startswith("#!") or "python" not in first:
        raise RuntimeError(
            f"{exe} is not a Python console script; set [tools] msplat in settings.toml to msplat-train from `uv tool install`"
        )
    return first[2:].strip().split()[0]


def downscaled_side(dataset: Path, downscale: float | None) -> int | None:
    """The longest image side at 1/`downscale` resolution (Brush's --max-resolution, gsplat's --max-side)."""
    if not downscale or downscale <= 1:
        return None
    frames = json.loads((dataset / "transforms.json").read_text())["frames"]
    return round(max(max(int(f["w"]), int(f["h"])) for f in frames) / downscale)


def command(
    trainer: str,
    exe: str,
    dataset: Path,
    out: Path,
    iterations: int,
    eval_every: int,
    downscale: float | None,
    extra: list[str],
    views: Path | None = None,
    name: str | None = None,
    project: str | None = None,
) -> list[str]:
    """The trainer's command line. Each writes `out/splat.ply` in the dataset's (scene) frame. msplat runs
    through MSPLAT_SCRIPT (pruning in LiDAR free space when `views` is given); gsplat through
    `gs-trainer cloud train` (`exe` is uv), which trains on a cloud GPU and downloads the result into `out`.
    gsplat ignores `eval_every`: it scores the dataset's split.json."""
    side = downscaled_side(dataset, downscale)
    if trainer == "msplat":
        cmd = [
            script_python(exe),
            str(MSPLAT_SCRIPT),
            "--input",
            str(dataset),
            "--output",
            str(out / "splat.ply"),
        ]
        cmd += ["--num-iters", str(iterations)]
        if views is not None:
            cmd += ["--views", str(views)]
        if downscale and downscale > 1:
            cmd += ["--downscale-factor", f"{downscale:g}"]
        cmd += ["--eval", "--test-every", str(eval_every)] if eval_every > 0 else []
    elif trainer == "brush":
        # Brush v0.3: training runs headless when given a dataset path without --with-viewer.
        cmd = [exe, str(dataset), "--total-steps", str(iterations)]
        cmd += ["--export-path", str(out), "--export-name", "splat.ply", "--export-every", str(iterations)]
        if side:
            cmd += ["--max-resolution", str(side)]
        if eval_every > 0:  # Brush prints no metrics headless: save the renders, score them in eval_renders
            cmd += ["--eval-split-every", str(eval_every), "--eval-every", str(iterations)]
            cmd += ["--eval-save-to-disk"]
    elif trainer == "gsplat":
        cmd = [exe, "run", "--project", str(TRAINER_PROJECT), "gs-trainer", "cloud", "train"]
        if project:
            cmd += ["--project", project]
        cmd += ["--data", str(dataset), "--out", str(out), "--name", name or trainer]
        cmd += ["--steps", str(iterations)]
        if side:
            cmd += ["--max-side", str(side)]
        return [*cmd, "--", *extra] if extra else cmd  # the rest goes to gs_trainer.train on the GPU
    else:
        raise ValueError(f"unknown trainer {trainer!r}; pick one of {', '.join(TRAINERS)}")
    return [*cmd, *extra]


def run_logged(cmd: list[str], cwd: Path, log: Path, total: int = 0, label: str = "training") -> None:
    """Run the trainer with a progress bar (steps, elapsed, ETA), keeping its output in `log`."""
    noisy = re.compile(r"^(Densified|WARNING: per-tile overflow)")  # every 100 steps: log only
    code = progress.run(cmd, cwd, log, total, label, show=lambda line: not noisy.match(line))
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def parse_metrics(log: str) -> dict[str, float]:
    """PSNR/SSIM on the held-out views, as the trainer printed them (last value wins)."""
    out: dict[str, float] = {}
    for key in ("psnr", "ssim"):
        found = re.findall(rf"\b{key}\b\W+([0-9]+\.[0-9]+)", log, flags=re.IGNORECASE)
        if found:
            out[key] = float(found[-1])
    return out


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM (11x11 Gaussian window, sigma 1.5, per channel), images in [0, 1]."""

    def g(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    c1, c2 = 0.01**2, 0.03**2
    ma, mb = g(a), g(b)
    va, vb, cov = g(a * a) - ma * ma, g(b * b) - mb * mb, g(a * b) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))).mean())


def eval_renders(dataset: Path, renders: Path) -> dict[str, float]:
    """Mean PSNR/SSIM of held-out renders (`<image stem>.png`) against the photos, at the render size."""
    psnr: list[float] = []
    ssim: list[float] = []
    for r in sorted(renders.glob("*.png")):
        img, photo = cv2.imread(str(r)), cv2.imread(str(dataset / "images" / f"{r.stem}.jpg"))
        if img is None or photo is None:
            continue
        a = img.astype(np.float64) / 255
        b = cv2.resize(photo, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA) / 255
        psnr.append(float(-10 * np.log10(max(float(((a - b) ** 2).mean()), 1e-10))))
        ssim.append(_ssim(a, b))
    return {"psnr": round(float(np.mean(psnr)), 4), "ssim": round(float(np.mean(ssim)), 4)} if psnr else {}


def train(
    trainer: str,
    exe: str,
    dataset: Path,
    out: Path,
    iterations: int,
    eval_every: int = 8,
    downscale: float | None = None,
    extra: list[str] | None = None,
    prune: bool = False,
    name: str | None = None,
    project: str | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Train the dataset with msplat, Brush or gsplat into `out/splat.ply`. Returns a summary for the
    manifest.

    `downscale` None: msplat picks one from MSPLAT_MAX_PIXELS, Brush trains at the dataset's resolution,
    gsplat at gs-trainer's default size. `prune`: msplat removes splats in LiDAR free space during training
    with the dataset's views.npz (Brush can't: no API; gsplat prunes on its own). `name`: the run's name
    on the cloud (gsplat).
    """
    path = shutil.which(os.path.expanduser(exe))
    if path is None:
        hint = {
            "msplat": "uv tool install --python 3.13 'msplat[cli]', or set [tools] msplat = \"/path/to/msplat-train\"",
            "brush": 'get it from https://github.com/ArthurBrussee/brush/releases into /Applications/brush_app, or set [tools] brush = "/path/to/brush_app"',
            "gsplat": "install uv (https://docs.astral.sh/uv/)",
        }.get(trainer, "")
        raise FileNotFoundError(
            f"{trainer} not found ({exe!r}). {hint}{' in settings.toml' if trainer != 'gsplat' else ''}"
        )
    if trainer == "msplat" and downscale is None:
        downscale = msplat_downscale(dataset)
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("splat.ply", "prune.json", "metrics.json"):
        (out / stale).unlink(missing_ok=True)
    shutil.rmtree(out / f"eval_{iterations}", ignore_errors=True)
    views: Path | None = None
    if trainer == "msplat" and prune:
        views = dataset / "views.npz"
        if not views.is_file():
            raise FileNotFoundError(f"{views} missing: re-export the dataset with `scanmodel splat --force`")
    cmd = command(
        trainer, path, dataset.resolve(), out.resolve(), iterations, eval_every, downscale, extra or [],
        views.resolve() if views else None, name, project,
    )  # fmt: skip
    # gs-trainer downloads the remote train.log into `out`, so its own output goes to cloud.log.
    log = out / ("cloud.log" if trainer == "gsplat" else "train.log")
    started = time.monotonic()
    run_logged(cmd, out, log, iterations, trainer)  # cwd=out: Brush writes its autotune cache to ./target
    wall = time.monotonic() - started
    ply = out / "splat.ply"
    if not ply.is_file():
        raise FileNotFoundError(f"{trainer} finished without writing {ply}; see {log}")
    summary: dict[str, object] = {
        "trainer": trainer,
        "dataset": rel(dataset, root),
        "command": [rel(c, root) if os.path.isabs(c) else c for c in cmd],
        "wall_s": round(wall, 1),
        "iterations": iterations,
        "downscale": downscale,
        "splats": ply_vertex_count(ply),
        "ply_mb": round(ply.stat().st_size / 1e6, 1),
    }
    if trainer == "gsplat":
        metrics = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").is_file() else {}
        run = json.loads((out / "run.json").read_text()) if (out / "run.json").is_file() else {}
        summary |= {k: metrics[k] for k in ("psnr", "ssim", "lpips") if k in metrics}
        summary |= {"gpu_wall_s": metrics.get("wall_s"), "run": run.get("run_id", run.get("id"))}
        return summary
    prune_log = json.loads((out / "prune.json").read_text()) if (out / "prune.json").is_file() else None
    return {
        **summary,
        "pruned": prune_log["removed"] if prune_log else None,
        **parse_metrics(log.read_text()),
        **(
            eval_renders(dataset, out / f"eval_{iterations}") if trainer == "brush" and eval_every > 0 else {}
        ),
    }


# --------------------------------------------------------------------------- named views

VIEW_SIZE = (960, 720)
VIEWER_FOV = 70.0  # the viewer's vertical field of view without ?fov=


def look_at(eye: F64, target: F64) -> F64:
    """Camera-to-world, OpenCV axes (+z forward, +y down), z-up world: the viewer's `?cam=` + `?look=`."""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r = r / np.linalg.norm(r)
    m = np.eye(4)
    m[:3, :3] = np.stack([r, np.cross(f, r), f], axis=1)
    m[:3, 3] = eye
    return m


def view_from_url(
    name: str, url: str, note: str = "", size: tuple[int, int] = VIEW_SIZE
) -> dict[str, object]:
    """A named view from a viewer URL (or just its query string).

    The viewer takes `?pose=` (16 numbers, row-major camera-to-world with OpenGL axes in the scene frame, a
    nerfstudio `transform_matrix`) or `?cam=x,y,z&look=x,y,z`, and `?fov=` (vertical, degrees). The view
    keeps the pose in OpenCV axes, like everything else in the pipeline.
    """
    query = urlsplit(url).query if "?" in url else url
    q = {k: v[-1] for k, v in parse_qs(query).items()}

    def numbers(key: str, n: int) -> F64 | None:
        if key not in q:
            return None
        try:
            v = np.array([float(x) for x in q[key].split(",")])
        except ValueError:
            v = np.empty(0)
        if len(v) != n or not np.all(np.isfinite(v)):
            raise ValueError(f"?{key}= needs {n} comma-separated numbers, got {q[key]!r}")
        return v

    pose, cam, look = numbers("pose", 16), numbers("cam", 3), numbers("look", 3)
    if pose is not None:
        c2w = pose.reshape(4, 4) @ GL_CV
        if not is_rigid(c2w):
            raise ValueError("?pose= is not a rigid camera-to-world matrix")
    elif cam is not None and look is not None:
        c2w = look_at(cam, look)
    else:
        raise ValueError("the URL needs ?pose= or ?cam= and ?look= (copy it from the viewer)")
    c2w[3] = [0.0, 0.0, 0.0, 1.0]
    fov = float(q.get("fov", VIEWER_FOV))
    if not 0 < fov < 180:
        raise ValueError(f"?fov= must be between 0 and 180 degrees, got {fov}")
    return {
        "name": name,
        "c2w": (np.round(c2w, 6) + 0.0).tolist(),  # + 0.0: no -0.0 in the file
        "fov_y_deg": round(fov, 3),
        "w": size[0],
        "h": size[1],
        "note": note,
    }


def view_from_image(
    name: str, im: PosedImage, note: str = "", size: tuple[int, int] = VIEW_SIZE
) -> dict[str, object]:
    """A named view at a captured frame's or still's pose: its vertical field of view, centred, at `size`
    (scaled to the image's aspect: a 4:3 frame stays 4:3)."""
    k = im.intrinsics
    c2w = mat(im.camera_to_world)
    c2w[3] = [0.0, 0.0, 0.0, 1.0]
    w = size[0]
    h = max(round(w * k.height / k.width), 1)
    return {
        "name": name,
        "c2w": (np.round(c2w, 6) + 0.0).tolist(),
        "fov_y_deg": round(math.degrees(2 * math.atan(k.height / 2 / k.fy)), 3),
        "w": w,
        "h": h,
        "note": note,
    }


def find_image(poses_dir: Path, ref: str) -> PosedImage:
    """A frame or still of `poses_dir` by id (`<session>/<idx>`, `<session>/s<n>`) or dataset image stem
    (`<session>_000342`, `<session>_s44`, as in `images/`)."""
    if "/" not in ref and "_" in ref:
        session, _, tail = ref.rpartition("_")
        ref = f"{session}/{tail if tail.startswith('s') else int(tail)}"
    for file in ("frames.jsonl", "stills.jsonl"):
        for im in read_jsonl(poses_dir / file, PosedImage):
            if im.id == ref:
                return im
    raise KeyError(f"no frame or still {ref!r} in {poses_dir}")


def view_url(view: dict[str, object]) -> str:
    """The viewer query that shows `view` (the inverse of view_from_url)."""
    gl = np.asarray(view["c2w"], dtype=float) @ GL_CV
    pose = ",".join(f"{x:.6g}" for x in gl.ravel())
    return f"?pose={pose}&fov={view['fov_y_deg']:g}"


def read_named_views(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text())["views"] if path.is_file() else []


def add_named_view(path: Path, view: dict[str, object]) -> bool:
    """Append `view` to the views file, or replace the view of the same name. Returns True if replaced."""
    views = read_named_views(path)
    replaced = any(v["name"] == view["name"] for v in views)
    views = [view if v["name"] == view["name"] else v for v in views] if replaced else [*views, view]
    text = json.dumps({"views": views}, indent=1)
    # One matrix row per line, so the file stays readable and diffs stay small.
    text = re.sub(r"\[\s+([-+0-9.e,\s]+?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m[1]) + "]", text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    return replaced
