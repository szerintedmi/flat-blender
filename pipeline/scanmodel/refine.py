"""Photometric pose refinement (`scanmodel refine`): bundle adjustment of frames and stills on SIFT tracks.

The pose graph (`scanmodel.align`) leaves separate passes past the same surface a few cm apart (scan 4: median
29 mm, 6-12 px at 960 px), while frames of one pass agree to 1-3 mm. A splat trained on that ghosts. This
refines every pose against image features matched across passes, sessions and the lap:

- Views: frames with LiDAR depth, thinned to `MIN_MOVE` / `MIN_TURN` / `MAX_GAP_S`, plus the stills.
  A turning frame's keypoints are noisier (rolling shutter: 14.7 ms readout, ~6 px of shear at 60 deg/s),
  so its observations get a larger sigma (`skew_sigma`) instead of the frame being left out. Optionally
  (`readout_s`) keypoints are moved to the mid-readout time with ARKit's angular velocity; on scan 4 that
  lowered the BA residual but not the misalignment of planar targets, so it is off by default.
- Features: SIFT at 960x720 (RootSIFT, contrast threshold 0.02 for plain walls), each keypoint with its
  LiDAR depth where the depth is smooth. Cached per session in `features/`.
- Pairs: neighbours in time (same pass), and views of the same LiDAR voxels at least `CROSS_DT` apart
  looking the same way (other passes, the lap), the best few per other pass.
- First a coarse round (see "coarse alignment" below): wide windows and a loose gate, drift accepted only
  where many pairs agree on one offset: a trajectory cut where ARKit saw nothing or has a gap in its frames
  and the later sessions inherited the jump ("trajectory cuts"), then per session and bin; if any is found,
  a coarse pose graph moves the start poses first.
- Matching, guided by the current poses: a keypoint may only match near where its LiDAR point projects
  (what `WINDOW_M` of pose error moves it, at least `WINDOW_MIN_PX`) or, without depth, near its epipolar
  line; ratio test and mutual nearest neighbour within
  those windows (so a repeated pattern elsewhere in the image can't match); then a PnP RANSAC on the
  LiDAR points must agree with the current relative pose within `GATE_M` / `GATE_DEG` (mirrors, glass,
  moved things and repeated patterns don't), and where both keypoints have LiDAR depth the two depths
  must agree (`DEPTH_GATE_*`: a mirror's LiDAR sees the glass, its feature lies behind it).
- Tracks: union-find over the matches; tracks with two keypoints in one view are dropped, and so are
  tracks whose LiDAR points scatter by more than 10 cm (moved objects, wrong merges). Only tracks with
  three observations or a cross-pass one are kept.
- Bundle adjustment (gtsam, Levenberg-Marquardt): reprojection factors (Huber) on every observation, a
  weak LiDAR range prior on landmarks (scale, no drift), ARKit's relative motion between consecutive views
  from the raw ARKit poses (tight: a pass stays rigid; x10 looser in the first 5 s of a session that
  started unsettled), gravity per view (loose, see below), a weak prior to the start pose (gauge, and the global frame).
  Stills have their own features and hang off their nearest frame by a looser link. Two rounds: observations off by more than `OUTLIER_PX` after the first are dropped.
- Floor heights (what the pose graph's floor links did): raw ARKit's relative motion drifts vertically
  and its gravity direction is off by up to ~0.3 deg per session; without floors the adjustment followed
  both (scan 3: camera heights tilted by 4.9 mm per metre across the scan, the conservatory 2 cm lower, the
  hallway 1.5 cm above the living room). Each frame view's LiDAR floor cells (`align.floors`: horizontal,
  well below the camera, outside furniture and stairs, per 0.5 m cell; views thinned like the pose graph's
  keyframes) observe one height landmark per cell; only cells seen at least `align.MIN_DT` apart count, at
  most `FLOOR_PER_SESSION` views per cell and session (a pass's views share its depth bias). Error in
  world z only (`FLOOR_SIGMA`, Huber); x and y are left free. Cells next to each other at the same height
  (within 1.5 cm at the pose graph's poses) form a floor level whose cells are held level (`LEVEL_SIGMA`).
  Gravity per view is loose (`GRAVITY_DEG`): a tight one pins each session to ARKit's own slightly tilted
  vertical (on scans 3 and 4 the cross-pass residual and the floors both improve when it is loose; the
  start priors still hold the overall vertical). Afterwards the main floor level is moved back to its
  height before the adjustment (the scene frame: floor at z = 0).
- The corrections (refined @ original^-1) are interpolated in time to every frame (`align.corrector`).
"""

import itertools
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cv2
import gtsam
import numpy as np

from .config import from_repo, rel
from .io import VideoReader, decoders, read_confidence, read_depth
from .models import bundle as b
from .models.bundle import Intrinsics
from .models.common import F64, mat
from .models.derived import PosedImage

WIDTH, HEIGHT = 960, 720  # feature image size
N_FEATURES = 1500
CONTRAST = 0.02  # SIFT contrast threshold: OpenCV's 0.04 finds little on plain walls
MIN_MOVE, MIN_TURN, MAX_GAP_S = 0.05, 4.0, 0.4  # view thinning
READOUT_S = 0.0147  # rolling-shutter readout, top row first (scan 4's LED bands: 985 rows per 10 ms)
MAX_SPIN = 120.0  # deg/s: faster-turning frames are too blurred to be views
RS_LEFT = 0.3  # share of the rolling-shutter skew assumed left after correcting the keypoints for it
VOXEL = 0.2  # pair search: LiDAR voxels seen
CROSS_DT = 6.0  # a cross-pass pair is at least this far apart in time
MIN_OVERLAP = 0.3  # shared voxels / the smaller view's voxels
MAX_VIEW_ANGLE = 50.0  # degrees between viewing directions
PER_PASS, MAX_PASSES = 2, 6  # cross pairs per view: best per other pass, passes
SEQ_STEPS = (1, 2, 3)  # same-pass pairs: the next views in time
WINDOW_M, WINDOW_MIN_PX = 0.08, 30.0  # guided window: what 8 cm of pose error moves the point, at 960 px
EPI_PX = 40.0  # epipolar band (keypoints without depth)
DEPTH_GATE_M, DEPTH_GATE_REL = 0.08, 0.03  # a match's two LiDAR depths agree within this after the pose
RATIO = 0.85
MAX_DESC = 0.6  # Hellinger distance (RootSIFT, L2) of an accepted match at most
PNP_PX = 2.0
MIN_INLIERS = 15
GATE_M, GATE_DEG = 0.25, 3.0  # a pair's PnP pose vs the current relative pose
PIXEL_SIGMA = 1.0  # at 960 px
OUTLIER_PX = 3.0
HUBER = 1.345
CACHE_VERSION = 1
FLOOR_SIGMA = 0.005  # m per floor observation (Huber)
LEVEL_SIGMA = 0.005  # m: a floor cell from its level's height (Huber)
GRAVITY_DEG = 2.0  # per view, to ARKit's gravity direction (see the module docstring)
FLOOR_FREE = 1.0  # m: x and y of a floor observation are barely constrained (it sits at the cell's centre)
FLOOR_HELD = 0.002  # m: a floor observation's point in the camera frame (a variable) held by a prior
FLOOR_PER_SESSION = 8  # views per floor cell and session
FLOOR_MOVE, FLOOR_TURN = 0.15, 10.0  # frame views thinned like the pose graph's keyframes for floor cells


@dataclass
class View:
    """A bundle-adjustment node: a video frame with depth, or a still."""

    image: PosedImage
    trajectory: int
    t: float
    pose: F64  # camera_to_world, scene frame, OpenCV axes
    k: Intrinsics  # at WIDTH x HEIGHT
    omega: F64 = field(default_factory=lambda: np.zeros(3))  # angular velocity, camera axes, rad/s
    sigma: float = PIXEL_SIGMA  # keypoint noise (px at WIDTH): more while turning (rolling-shutter skew)

    @property
    def still(self) -> bool:
        return self.image.kind == "still"


@dataclass
class Features:
    uv: np.ndarray  # (N, 2) float32 pixel positions at WIDTH x HEIGHT, pixel-corner convention, as detected
    desc: np.ndarray  # (N, 128) uint8 SIFT
    depth: np.ndarray  # (N,) float32 LiDAR depth (m), 0 = none
    conf: np.ndarray  # (N,) uint8 LiDAR confidence of that depth


def _rootsift(desc: np.ndarray) -> np.ndarray:
    d = desc.astype(np.float32)
    d /= np.maximum(d.sum(axis=1, keepdims=True), 1e-6)
    return np.sqrt(d)


def select_views(
    frames: list[PosedImage],
    stills: list[PosedImage],
    trajectory: dict[str, int],
    min_move: float = MIN_MOVE,
    min_turn: float = MIN_TURN,
    max_gap: float = MAX_GAP_S,
    readout_s: float = 0.0,
) -> list[View]:
    """Frames with depth (tracking normal) thinned by motion and time, then the stills; with each frame's
    angular velocity from its neighbours at full frame rate (all `frames`)."""
    by: dict[str, list[PosedImage]] = defaultdict(list)
    for f in frames:
        if f.t is not None and f.tracking in (None, b.TrackingState.NORMAL):
            by[f.session].append(f)
    out: list[View] = []
    for sid, fs in by.items():
        fs.sort(key=lambda f: f.t or 0.0)
        t = np.array([f.t or 0.0 for f in fs])
        rot = np.array([mat(f.camera_to_world)[:3, :3] for f in fs])
        last: PosedImage | None = None
        for n, f in enumerate(fs):
            if not f.depth:
                continue
            m = mat(f.camera_to_world)
            if last is not None:
                lm = mat(last.camera_to_world)
                move = float(np.linalg.norm(m[:3, 3] - lm[:3, 3]))
                turn = _angle_deg(lm[:3, :3], m[:3, :3])
                if move < min_move and turn < min_turn and (f.t or 0.0) - (last.t or 0.0) < max_gap:
                    continue
            a, c = max(n - 1, 0), min(n + 1, len(fs) - 1)
            omega = np.zeros(3)
            if c > a and t[c] - t[a] < 0.1:
                omega = _rotvec(rot[a].T @ rot[c]) / (t[c] - t[a])
            if np.degrees(np.linalg.norm(omega)) > MAX_SPIN:
                continue
            last = f
            k = f.intrinsics.scaled(WIDTH, HEIGHT)
            out.append(View(f, trajectory[sid], f.t or 0.0, m, k, omega, skew_sigma(k.fx, omega, readout_s)))
    for s in stills:
        if s.t is not None and s.session in trajectory:
            out.append(
                View(
                    s, trajectory[s.session], s.t, mat(s.camera_to_world), s.intrinsics.scaled(WIDTH, HEIGHT)
                )
            )
    return sorted(out, key=lambda v: (v.trajectory, v.t, v.still))


def skew_sigma(fx: float, omega: F64, readout_s: float = 0.0) -> float:
    """Keypoint noise of a frame turning at `omega`: the rolling shutter shears the image by up to
    fx |omega| READOUT_S / 2 px either side of the middle row (RS_LEFT of that when corrected)."""
    skew = fx * float(np.linalg.norm(omega)) * READOUT_S / 2 * (RS_LEFT if readout_s else 1.0)
    return float(np.hypot(PIXEL_SIGMA, skew))


def _angle_deg(a: F64, c: F64) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ c) - 1) / 2, -1.0, 1.0))))


def _rotvec(r: F64) -> F64:
    return np.asarray(gtsam.Rot3.Logmap(gtsam.Rot3(r)), dtype=np.float64)


def _keypoint_depth(uv: np.ndarray, depth: F64, conf: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """LiDAR depth at each keypoint (median of the 3x3 depth pixels around it), 0 at depth edges."""
    dh, dw = depth.shape
    du = np.clip((uv[:, 0] * dw / WIDTH).astype(int), 1, dw - 2)
    dv = np.clip((uv[:, 1] * dh / HEIGHT).astype(int), 1, dh - 2)
    win = np.stack([depth[dv + a, du + c] for a in (-1, 0, 1) for c in (-1, 0, 1)], axis=1)
    z = np.median(win, axis=1)
    ok = (win.min(axis=1) > 0.2) & (np.ptp(win, axis=1) < 0.02 + 0.02 * z) & (z < 6.0)
    cf = np.full(len(uv), 2, np.uint8)
    if conf is not None:
        cw = np.stack([conf[dv + a, du + c] for a in (-1, 0, 1) for c in (-1, 0, 1)], axis=1)
        cf = cw.min(axis=1).astype(np.uint8)
    ok &= cf >= 1
    return np.where(ok, z, 0.0).astype(np.float32), np.where(ok, cf, 0).astype(np.uint8)


def _detect(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT.create(nfeatures=N_FEATURES, contrastThreshold=CONTRAST)
    kps, desc = cast(tuple[object, object], sift.detectAndCompute(gray, None))
    if not isinstance(desc, np.ndarray) or not isinstance(kps, tuple | list) or not kps:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    uv = np.array([k.pt for k in kps], np.float32) + 0.5  # pixel-corner convention, like the intrinsics
    return uv, np.clip(np.round(desc), 0, 255).astype(np.uint8)


def _features_of(img: np.ndarray, view: View, raw: Path) -> Features:
    gray = cv2.cvtColor(cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    uv, desc = _detect(gray)
    depth = np.zeros(len(uv), np.float32)
    conf = np.zeros(len(uv), np.uint8)
    im = view.image
    if im.depth and len(uv):
        d = read_depth(raw / im.depth)
        c = read_confidence(raw / im.confidence) if im.confidence else None
        depth, conf = _keypoint_depth(uv, d, c)
    return Features(uv, desc, depth, conf)


def _cache_meta(raw: Path) -> dict[str, object]:
    video = raw / b.VIDEO_FILE
    st = video.stat() if video.is_file() else None
    return {
        "version": CACHE_VERSION,
        "video": rel(video),
        "bytes": st.st_size if st else 0,
        "mtime": st.st_mtime if st else 0,
        "size": [WIDTH, HEIGHT],
        "n": N_FEATURES,
        "contrast": CONTRAST,
    }


def _load_cache(path: Path, meta: dict[str, object]) -> dict[str, Features]:
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as z:
        legacy = {**meta, "video": str(from_repo(str(meta["video"])))}  # caches from before relative paths
        if str(z["meta"]) not in (repr(meta), repr(legacy)):
            return {}
        ids = [str(x) for x in z["ids"]]
        off = z["offsets"]
        uv, desc, depth, conf = z["uv"], z["desc"], z["depth"], z["conf"]
        return {
            i: Features(
                uv[off[n] : off[n + 1]],
                desc[off[n] : off[n + 1]],
                depth[off[n] : off[n + 1]],
                conf[off[n] : off[n + 1]],
            )
            for n, i in enumerate(ids)
        }


def _save_cache(path: Path, meta: dict[str, object], feats: dict[str, Features]) -> None:
    ids = sorted(feats)
    sizes = [len(feats[i].uv) for i in ids]
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    def cat(get: Callable[[Features], np.ndarray], shape: tuple[int, ...], dtype: type) -> np.ndarray:
        parts = [get(feats[i]) for i in ids]
        return np.concatenate(parts) if parts else np.zeros(shape, dtype)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        meta=np.array(repr(meta)),
        ids=np.array(ids),
        offsets=off,
        uv=cat(lambda f: f.uv, (0, 2), np.float32),
        desc=cat(lambda f: f.desc, (0, 128), np.uint8),
        depth=cat(lambda f: f.depth, (0,), np.float32),
        conf=cat(lambda f: f.conf, (0,), np.uint8),
    )
    tmp.replace(path)


def features(
    views: list[View], raw: dict[str, Path], cache_dir: Path, progress: Callable[[int], None] | None = None
) -> list[Features]:
    """Features of every view (video frames decoded in one pass per session, sessions in parallel; stills
    from their JPEGs), cached per session in `cache_dir/<session>.npz`."""
    out: list[Features | None] = [None] * len(views)
    by: dict[str, list[int]] = defaultdict(list)
    for n, v in enumerate(views):
        by[v.image.session].append(n)
    workers = max(1, min(8, (cv2.getNumberOfCPUs() or 2) - 1))
    slots = threading.BoundedSemaphore(4 * workers)  # decoded images waiting for SIFT (8 MB each)

    def session(pool: ThreadPoolExecutor, sid: str, idx: list[int]) -> None:
        meta = _cache_meta(raw[sid])
        path = cache_dir / f"{sid}.npz"
        cache = _load_cache(path, meta)
        todo: list[int] = []
        for n in idx:
            f = cache.get(views[n].image.id)
            if f is not None:
                out[n] = f
                if progress:
                    progress(1)
            else:
                todo.append(n)
        if not todo:
            return
        jobs: list[tuple[int, Future[Features]]] = []

        def submit(n: int, img: np.ndarray) -> None:
            slots.acquire()
            job = pool.submit(_features_of, img, views[n], raw[sid])
            job.add_done_callback(lambda _: slots.release())
            jobs.append((n, job))

        frames = sorted((n for n in todo if not views[n].still), key=lambda n: _frame_idx(views[n].image))
        if frames:
            reader = VideoReader(raw[sid] / b.VIDEO_FILE)
            try:
                for n in frames:
                    img = reader.read(_frame_idx(views[n].image))
                    if img is None:
                        break
                    submit(n, img)
            finally:
                reader.close()
        for n in todo:
            if views[n].still:
                img = cv2.imread(str(raw[sid] / views[n].image.file))
                if img is not None:
                    submit(n, img)
        for n, job in jobs:
            f = job.result()
            out[n] = f
            cache[views[n].image.id] = f
            if progress:
                progress(1)
        _save_cache(path, meta, cache)

    with ThreadPoolExecutor(max_workers=workers) as pool, ThreadPoolExecutor(decoders(len(by))) as readers:
        for done in [readers.submit(session, pool, sid, idx) for sid, idx in by.items()]:
            done.result()
    empty = Features(
        np.zeros((0, 2), np.float32),
        np.zeros((0, 128), np.uint8),
        np.zeros(0, np.float32),
        np.zeros(0, np.uint8),
    )
    return [f if f is not None else empty for f in out]


def _frame_idx(im: PosedImage) -> int:
    return int(im.file.split("#")[1])


# --------------------------------------------------------------------------- rolling shutter


def rs_corrected(uv: np.ndarray, view: View, readout: float = READOUT_S) -> np.ndarray:
    """Keypoints moved to where they'd be at the frame's mid-readout time (rotation only: at 0.1 m/s the
    translation in 7 ms is under a mm). Row v is read `readout * (v / H - 0.5)` s after mid-frame; the camera
    has turned by omega * that, so the mid-frame ray is exp(omega dt) times the observed ray."""
    if view.still or not readout or not len(uv) or not np.any(view.omega):
        return uv.astype(np.float64)
    k = view.k
    dt = readout * (uv[:, 1] / HEIGHT - 0.5)
    rays = np.column_stack([(uv[:, 0] - k.cx) / k.fx, (uv[:, 1] - k.cy) / k.fy, np.ones(len(uv))])
    # small angle: exp([w] dt) r ~ r + dt (w x r)
    rays = rays + dt[:, None] * np.cross(view.omega[None, :], rays)
    return np.column_stack([k.fx * rays[:, 0] / rays[:, 2] + k.cx, k.fy * rays[:, 1] / rays[:, 2] + k.cy])


# --------------------------------------------------------------------------- pairs


def view_voxels(
    views: list[View], feats: list[Features], voxel: float = VOXEL
) -> list[set[tuple[int, int, int]]]:
    """LiDAR voxels (`voxel` m) each view sees, from its keypoints with depth; a still borrows the nearest
    frame's (within 1 s)."""
    out: list[set[tuple[int, int, int]]] = []
    for v, f in zip(views, feats, strict=True):
        ok = f.depth > 0
        out.append(
            set(
                map(
                    tuple,
                    np.floor(_world(v, f.uv[ok].astype(np.float64), f.depth[ok]) / voxel)
                    .astype(int)
                    .tolist(),
                )
            )
        )
    frames = [i for i, v in enumerate(views) if not v.still]
    ft = np.array([views[i].t for i in frames])
    for i, v in enumerate(views):
        if v.still and len(ft):
            j = frames[int(np.argmin(np.abs(ft - v.t)))]
            if abs(views[j].t - v.t) < 1.0 and views[j].image.session == v.image.session:
                out[i] = out[j]
    return out


def _world(v: View, uv: F64, z: np.ndarray) -> F64:
    k = v.k
    cam = np.column_stack([(uv[:, 0] - k.cx) / k.fx * z, (uv[:, 1] - k.cy) / k.fy * z, z])
    return cam @ v.pose[:3, :3].T + v.pose[:3, 3]


def pairs(
    views: list[View],
    voxels: list[set[tuple[int, int, int]]],
    per_pass: int = PER_PASS,
    same_pass: bool = True,
) -> list[tuple[int, int, str]]:
    """(i, j, "seq" | "cross"), i < j. Same pass (if `same_pass`): the next SEQ_STEPS views of the trajectory (frames
    and stills alike, within 3 s). Cross: views >= CROSS_DT apart sharing >= MIN_OVERLAP of their voxels and
    looking within MAX_VIEW_ANGLE of each other; per view the `per_pass` best of each other pass (a pass:
    candidates within 3 s of each other), at most MAX_PASSES passes."""
    out: set[tuple[int, int, str]] = set()
    for i in range(len(views) if same_pass else 0):
        for s in SEQ_STEPS:
            j = i + s
            if (
                j < len(views)
                and views[j].trajectory == views[i].trajectory
                and views[j].t - views[i].t < 3.0
            ):
                out.add((i, j, "seq"))
    index: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, vs in enumerate(voxels):
        for c in vs:
            index[c].append(i)
    fwd = np.array([v.pose[:3, 2] for v in views])
    cos_max = np.cos(np.radians(MAX_VIEW_ANGLE))
    seq = {(i, j) for i, j, _ in out}
    for i, vs in enumerate(voxels):
        if not vs:
            continue
        shared: dict[int, int] = defaultdict(int)
        for c in vs:
            for j in index[c]:
                if j != i:
                    shared[j] += 1
        cands: list[tuple[float, int]] = []
        for j, n in shared.items():
            if abs(views[j].t - views[i].t) < CROSS_DT and views[j].trajectory == views[i].trajectory:
                continue
            ratio = n / max(1, min(len(vs), len(voxels[j])))
            if ratio >= MIN_OVERLAP and fwd[i] @ fwd[j] >= cos_max:
                cands.append((views[j].t + 1e4 * views[j].trajectory, j))
        cands.sort()
        passes: list[list[int]] = []
        last = -np.inf
        for t, j in cands:
            if t - last > 3.0:
                passes.append([])
            passes[-1].append(j)
            last = t
        ranked = sorted(passes, key=lambda p: -max(shared[j] for j in p))[:MAX_PASSES]
        for p in ranked:
            for j in sorted(p, key=lambda j: -shared[j])[:per_pass]:
                a, c = min(i, j), max(i, j)
                if (a, c) not in seq:
                    out.add((a, c, "cross"))
    return sorted(out)


# --------------------------------------------------------------------------- matching


@dataclass
class Match:
    """What matching a pair of views gave: the kept keypoint indices (a, b), the candidate count, and the PnP
    relative pose (camera a -> camera b) when one was solved, with how far it is from the current relative
    pose (`dev_m`, `dev_deg`, as the gate measures it). `gated`: solved, but beyond the gate."""

    ia: np.ndarray
    ib: np.ndarray
    candidates: int
    rel: F64 | None = None
    dev_m: float = 0.0
    dev_deg: float = 0.0
    gated: bool = False


@dataclass
class PairMatches:
    i: int
    j: int
    kind: str
    ia: np.ndarray  # keypoint indices in view i
    ib: np.ndarray  # in view j
    candidates: int  # matches before the geometric check
    rel: F64 | None = None  # PnP relative pose i -> j (see Match)
    dev_m: float = 0.0
    dev_deg: float = 0.0
    gated: bool = False


def _k3(k: Intrinsics) -> F64:
    return np.array([[k.fx, 0.0, k.cx], [0.0, k.fy, k.cy], [0.0, 0.0, 1.0]])


def match_pair(
    va: View,
    vb: View,
    fa: Features,
    fb: Features,
    ua: F64,
    ub: F64,
    da: np.ndarray,
    db: np.ndarray,
    window_m: float = WINDOW_M,
    gate: tuple[float, float] = (GATE_M, GATE_DEG),
) -> Match:
    """Guided matching of two views (see the module docstring). `ua`/`ub`: rolling-shutter corrected
    keypoints, `da`/`db`: RootSIFT descriptors; `window_m`: the pose error the guided windows allow, `gate`:
    (m, deg) the PnP pose may differ from the current relative pose."""
    none = np.zeros(0, int)
    if len(ua) < MIN_INLIERS or len(ub) < MIN_INLIERS:
        return Match(none, none, 0)
    rel = np.linalg.inv(vb.pose) @ va.pose  # camera a -> camera b
    r, t = rel[:3, :3], rel[:3, 3]
    ka, kb = _k3(va.k), _k3(vb.k)
    allowed = np.zeros((len(ua), len(ub)), bool)
    has = fa.depth > 0
    if has.any():
        z = fa.depth[has].astype(np.float64)
        pa = np.column_stack([(ua[has, 0] - va.k.cx) / va.k.fx * z, (ua[has, 1] - va.k.cy) / va.k.fy * z, z])
        pb = pa @ r.T + t
        front = pb[:, 2] > 0.1
        pred = np.full((len(pb), 2), 1e9)
        pred[front] = (pb[front, :2] / pb[front, 2:3]) * [vb.k.fx, vb.k.fy] + [vb.k.cx, vb.k.cy]
        d2 = ((pred[:, None, :] - ub[None, :, :]) ** 2).sum(-1)
        radius = np.maximum(WINDOW_MIN_PX, vb.k.fx * window_m / np.maximum(pb[:, 2], 0.1))
        allowed[has] = d2 < (radius**2)[:, None]
    if (~has).any():
        tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
        f = np.linalg.inv(kb).T @ tx @ r @ np.linalg.inv(ka)
        xa = np.column_stack([ua[~has], np.ones((~has).sum())])
        lines = xa @ f.T  # (n, 3) lines in b
        xb = np.column_stack([ub, np.ones(len(ub))])
        dist = np.abs(lines @ xb.T) / np.maximum(np.linalg.norm(lines[:, :2], axis=1, keepdims=True), 1e-12)
        allowed[~has] = dist < EPI_PX
    if not allowed.any():
        return Match(none, none, 0)
    dist = np.sqrt(np.maximum(2.0 - 2.0 * (da @ db.T), 0.0))
    dist[~allowed] = np.inf
    best_b = np.argmin(dist, axis=1)
    d1 = dist[np.arange(len(ua)), best_b]
    dist2 = dist.copy()
    dist2[np.arange(len(ua)), best_b] = np.inf
    d2 = dist2.min(axis=1)
    best_a = np.argmin(dist, axis=0)
    ok = np.isfinite(d1) & (d1 < MAX_DESC) & (d1 < RATIO * d2) & (best_a[best_b] == np.arange(len(ua)))
    ia, ib = np.flatnonzero(ok), best_b[ok]
    n_cand = len(ia)
    if n_cand < MIN_INLIERS:
        return Match(none, none, n_cand)
    v = verify(va, vb, fa, fb, ua, ub, ia, ib, rel, gate)
    if v.keep.sum() < MIN_INLIERS:
        return Match(none, none, n_cand, v.rel, v.dev_m, v.dev_deg, v.gated)
    return Match(ia[v.keep], ib[v.keep], n_cand, v.rel, v.dev_m, v.dev_deg)


@dataclass
class Verified:
    keep: np.ndarray  # inlier mask of the candidate matches
    rel: F64 | None = None  # the PnP relative pose a -> b, if solved
    dev_m: float = 0.0  # its distance from the current relative pose (translation, rotation)
    dev_deg: float = 0.0
    gated: bool = False  # solved but beyond the gate: nothing kept


def verify(
    va: View,
    vb: View,
    fa: Features,
    fb: Features,
    ua: F64,
    ub: F64,
    ia: np.ndarray,
    ib: np.ndarray,
    rel: F64,
    gate: tuple[float, float] = (GATE_M, GATE_DEG),
) -> Verified:
    """Inlier mask of candidate matches: PnP RANSAC of b against a's LiDAR points (whichever side has more
    depth), gated against the current relative pose `rel` (a -> b) by `gate` (m, deg); every match is then
    checked against that pose: by reprojecting its LiDAR point(s), or the epipolar (Sampson) distance
    without depth."""
    keep = np.zeros(len(ia), bool)
    # Solve with the side that has more depth-backed matches; express everything as a -> b.
    za, zb = fa.depth[ia] > 0, fb.depth[ib] > 0
    flip = zb.sum() > za.sum()
    src_v, dst_v = (vb, va) if flip else (va, vb)
    src_u, dst_u = (ub[ib], ua[ia]) if flip else (ua[ia], ub[ib])
    src_z = (fb.depth[ib] if flip else fa.depth[ia]).astype(np.float64)
    has = src_z > 0
    if has.sum() < MIN_INLIERS:
        return Verified(keep)
    k_src, k_dst = src_v.k, dst_v.k
    thr = PNP_PX * max(va.sigma, vb.sigma) / PIXEL_SIGMA
    pts = np.column_stack(
        [
            (src_u[has, 0] - k_src.cx) / k_src.fx * src_z[has],
            (src_u[has, 1] - k_src.cy) / k_src.fy * src_z[has],
            src_z[has],
        ]
    )
    guess = np.linalg.inv(rel) if flip else rel
    rvec0 = cv2.Rodrigues(guess[:3, :3])[0]
    tvec0 = guess[:3, 3].reshape(3, 1).copy()
    try:
        found = cv2.solvePnPRansac(
            pts, dst_u[has].astype(np.float64), _k3(k_dst), None, rvec0.copy(), tvec0.copy(),
            useExtrinsicGuess=True, iterationsCount=300, reprojectionError=thr, confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )  # fmt: skip
    except cv2.error:
        return Verified(keep)
    ok, rvec, tvec, inl = cast(tuple[bool, np.ndarray, np.ndarray, np.ndarray | None], found)
    if not ok or inl is None or len(inl) < MIN_INLIERS:
        return Verified(keep)
    est = np.eye(4)
    est[:3, :3] = cv2.Rodrigues(rvec)[0]
    est[:3, 3] = tvec.ravel()
    rel_est = np.linalg.inv(est) if flip else est  # a -> b
    dev_m = float(np.linalg.norm(est[:3, 3] - guess[:3, 3]))
    dev_deg = _angle_deg(est[:3, :3], guess[:3, :3])
    if dev_m > gate[0] or dev_deg > gate[1]:
        # a consistent but wrong solution (repeated pattern, mirror), a moved object, or drift the poses
        # don't have yet (`coarse_groups` tells them apart by consensus)
        return Verified(keep, rel_est, dev_m, dev_deg, gated=True)
    return Verified(consistent(va, vb, fa, fb, ua, ub, ia, ib, rel_est), rel_est, dev_m, dev_deg)


def consistent(
    va: View, vb: View, fa: Features, fb: Features, ua: F64, ub: F64, ia: np.ndarray, ib: np.ndarray, rel: F64
) -> np.ndarray:
    """Matches that agree with the relative pose `rel` (a -> b) within PNP_PX (x 1.5 for the epipolar test):
    a match is checked through each side's LiDAR point, or by its Sampson distance when neither has one."""
    r, t = rel[:3, :3], rel[:3, 3]
    ok = np.ones(len(ia), bool)
    checked = np.zeros(len(ia), bool)
    for src, dst, fs, us, ud, isrc, idst, m in (
        (va, vb, fa, ua, ub, ia, ib, rel),
        (vb, va, fb, ub, ua, ib, ia, np.linalg.inv(rel)),
    ):
        z = fs.depth[isrc].astype(np.float64)
        has = z > 0
        if not has.any():
            continue
        p = np.column_stack(
            [
                (us[isrc][has, 0] - src.k.cx) / src.k.fx * z[has],
                (us[isrc][has, 1] - src.k.cy) / src.k.fy * z[has],
                z[has],
            ]
        )
        q = p @ m[:3, :3].T + m[:3, 3]
        front = q[:, 2] > 0.1
        proj = np.full((len(q), 2), 1e9)
        proj[front] = q[front, :2] / q[front, 2:3] * [dst.k.fx, dst.k.fy] + [dst.k.cx, dst.k.cy]
        err = np.linalg.norm(proj - ud[idst][has], axis=1)
        # LiDAR depth is noisy (1 % + a few mm): allow its effect on the reprojection too
        allow = PNP_PX * max(va.sigma, vb.sigma) / PIXEL_SIGMA + 0.02 * dst.k.fx * np.linalg.norm(
            m[:3, 3]
        ) / np.maximum(z[has], 0.3)
        sub = ok[has] & (err < allow)
        # 3D: where the other view has depth too, it must agree (mirrors: LiDAR sees the glass, the
        # feature sits behind it; moved things)
        zd = (fb.depth[ib] if src is va else fa.depth[ia])[has].astype(np.float64)
        both = zd > 0
        sub[both] &= np.abs(q[both, 2] - zd[both]) < DEPTH_GATE_M + DEPTH_GATE_REL * zd[both]
        ok[has] = sub
        checked[has] = True
    rest = ~checked
    if rest.any():
        tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
        f = np.linalg.inv(_k3(vb.k)).T @ tx @ r @ np.linalg.inv(_k3(va.k))
        x1 = np.column_stack([ua[ia][rest], np.ones(rest.sum())])
        x2 = np.column_stack([ub[ib][rest], np.ones(rest.sum())])
        fx1, ftx2 = x1 @ f.T, x2 @ f
        num = np.sum(x2 * fx1, axis=1) ** 2
        den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
        ok[rest] = (
            np.sqrt(num / np.maximum(den, 1e-12)) < 1.5 * PNP_PX * max(va.sigma, vb.sigma) / PIXEL_SIGMA
        )
    return ok


def match_all(
    views: list[View],
    feats: list[Features],
    uvs: list[F64],
    candidates: list[tuple[int, int, str]],
    progress: Callable[[int], None] | None = None,
    window_m: float = WINDOW_M,
    gate: tuple[float, float] = (GATE_M, GATE_DEG),
) -> list[PairMatches]:
    descs = [_rootsift(f.desc) for f in feats]
    cv2.setRNGSeed(0)

    def one(p: tuple[int, int, str]) -> PairMatches:
        i, j, kind = p
        m = match_pair(
            views[i], views[j], feats[i], feats[j], uvs[i], uvs[j], descs[i], descs[j], window_m, gate
        )
        if progress:
            progress(1)
        return PairMatches(i, j, kind, m.ia, m.ib, m.candidates, m.rel, m.dev_m, m.dev_deg, m.gated)

    with ThreadPoolExecutor(max_workers=max(1, min(8, (cv2.getNumberOfCPUs() or 2) - 1))) as pool:
        return list(pool.map(one, candidates))


def pair_offsets(views: list[View], p: PairMatches, poses: list[F64] | None = None) -> tuple[F64, F64, float]:
    """Where pair p's PnP relative pose puts each camera given the other, minus where it is (world, m), at
    `poses` (default the views'): (offset of view i, offset of view j, rotation between them in degrees)."""
    assert p.rel is not None
    pi, pj = (poses[p.i], poses[p.j]) if poses is not None else (views[p.i].pose, views[p.j].pose)
    at_i, at_j = pj @ p.rel, pi @ np.linalg.inv(p.rel)
    return at_i[:3, 3] - pi[:3, 3], at_j[:3, 3] - pj[:3, 3], _angle_deg(at_j[:3, :3], pj[:3, :3])


def _tight(p: PairMatches) -> bool:
    return p.dev_m <= GATE_M and p.dev_deg <= GATE_DEG


# --------------------------------------------------------------------------- coarse alignment
#
# The fine round above only finds what the poses already nearly have: its windows allow 8 cm of pose error
# and its gate 25 cm / 3 deg. ARKit can drift much further where it can't see much: scan 5's lap seemed to
# come out 48 cm and 1.5 deg off after sliding along a blank wall, tracking "normal" (2026-10-10: it was the
# living session before it that had drifted, see "trajectory cuts" below; the lap was right); the pose
# graph found no closures for the lap (too few features) and refine's gate threw away every match that
# showed the offset. So before the fine round, a coarse one: cross pairs from 50 cm voxels, matched with wide windows and
# a loose gate (1 m / 6 deg). A loose match alone proves nothing (repeated cabinet doors and stair steps
# give consistent wrong poses ~20 cm off), so drift is accepted only by consensus, per session and 4 s bin
# of the camera being moved: enough loose pairs, most agreeing on one offset (10 cm + 3 % of the distance
# between the cameras) larger than 10 cm, outnumbering the bin's tight pairs 3:1 (tight pairs in such a bin
# agreed with the wrong pose: dropped), from two other sessions or views looking 45 deg apart, and the same
# in at least three bins of the session in a row (aliasing is local to one surface, drift persists); a
# run's neighbours with the same offset join it when their loose pairs merely outnumber the tight ones.
# Where a run is, the session drifted unseen from its last view with a tight pair before it (`Span`):
# ARKit's relative motion over that stretch is loosened by the offset (`stretch_odometry`; with nothing to
# place the drift in it, spreading it evenly is all one can do), and the span's views lose their pull to
# the start pose. Then a coarse pose graph over the views (ARKit's relative motion, gravity, loose priors,
# the pairs' relative poses: Huber first, so a far-off consistent majority can pull, then every loose pair
# re-gated around that solution and solved again with Cauchy) gives the start poses of the fine round and
# the bundle adjustment, which keeps the loosened odometry. Without a run nothing changes.

# coarse pairs: views of the same 50 cm voxels, the best one per other pass
COARSE_VOXEL, COARSE_PER_PASS = 0.5, 1
COARSE_WINDOW_M = 0.6  # guided windows: what 60 cm of pose error moves a point
LOOSE_M, LOOSE_DEG = 1.0, 6.0  # the coarse round's gate
BIN_S = 4.0  # consensus per session and bin of this many seconds
GROUP_MIN = 5  # loose pairs in a bin, at least
AGREE = 0.6  # share of them within TOL_M + TOL_REL x camera distance of their median
TOL_M, TOL_REL = 0.10, 0.03
DRIFT_M = 0.10  # a consensus offset larger than this is drift the fine round can't fix
LOOSE_OVER_TIGHT = 3  # agreeing loose pairs per tight pair in the bin, at least
RUN_BINS = 3  # bins in a row (gaps of one bin allowed) agreeing within 2 TOL_M (neighbours: 1:1 enough)
SPREAD_DEG = 45.0  # with pairs to a single other session: the bin's views must look this far apart
COARSE_PAIR = (1.0, 0.03)  # deg, m: sigma of a pair's relative pose in the coarse solve
COARSE_PRIOR = (3.0, 1.0)  # deg, m: each view's pull to its start pose in the coarse solve (none in a span)
REGATE_M, REGATE_DEG = 0.10, 2.0  # second coarse solve: pairs that agree this well with the first


def coarse_pairs(views: list[View], feats: list[Features]) -> list[tuple[int, int, str]]:
    """Cross pairs of frame views sharing 50 cm voxels (`COARSE_VOXEL`) at the current poses, the best one
    per other pass: coarse voxels still overlap when the poses are off by tens of cm."""
    voxels = view_voxels(views, feats, COARSE_VOXEL)
    return [
        p
        for p in pairs(views, voxels, COARSE_PER_PASS, same_pass=False)
        if not views[p[0]].still and not views[p[1]].still
    ]


@dataclass
class Group:
    """One session's coarse pairs in one time bin, seen from its own cameras."""

    session: str
    bin: int
    t0: float  # s from the session's first view
    loose: int  # pairs matched under the loose gate only
    tight: int  # pairs to other sessions within the fine round's gate
    agree: int  # loose pairs within tolerance of the median offset
    offset: F64  # median offset of the loose pairs (world, m): where the others put this session's camera
    sessions: int  # other sessions among the agreeing pairs
    spread_deg: float  # largest angle between the agreeing pairs' viewing directions in this session
    ok: bool  # passed the bin's tests
    near: bool = False  # agreeing loose pairs outnumber the tight ones, any diversity: enough next to a run
    admitted: bool = False  # ... and is part of a run of RUN_BINS
    rot_deg: float = 0.0  # median rotation of the agreeing pairs' PnP poses vs the current ones
    members: list[int] = field(default_factory=list[int])  # agreeing loose pairs (indices into the matches)
    tight_members: list[int] = field(default_factory=list[int])


def coarse_groups(views: list[View], matches: list[PairMatches]) -> list[Group]:
    """The consensus tests (see above) per session and bin, over the pairs matched under the loose gate."""
    first: dict[str, float] = {}
    for v in views:
        first[v.image.session] = min(first.get(v.image.session, v.t), v.t)
    entries: dict[tuple[str, int], list[tuple[int, int, int, F64, bool, float]]] = defaultdict(list)
    for n, p in enumerate(matches):
        if not len(p.ia) or p.rel is None:
            continue
        oi, oj, rot = pair_offsets(views, p)
        for me, other, off in ((p.i, p.j, oi), (p.j, p.i, oj)):
            s = views[me].image.session
            entries[(s, int((views[me].t - first[s]) // BIN_S))].append((n, me, other, off, _tight(p), rot))
    groups: list[Group] = []
    for (s, k), es in sorted(entries.items()):
        loose = [e for e in es if not e[4]]
        # tight pairs with the session's own other passes don't vote: a session's views drift together
        tight = [e[0] for e in es if e[4] and views[e[2]].image.session != s]
        g = Group(s, k, k * BIN_S, len(loose), len(tight), 0, np.zeros(3), 0, 0.0, False, tight_members=tight)
        groups.append(g)
        if len(loose) < GROUP_MIN:
            continue
        off = np.array([e[3] for e in loose])
        med = np.median(off, axis=0)
        dist = np.array([np.linalg.norm(views[e[1]].pose[:3, 3] - views[e[2]].pose[:3, 3]) for e in loose])
        agree = np.linalg.norm(off - med, axis=1) <= TOL_M + TOL_REL * dist
        good = [e for e, a in zip(loose, agree, strict=True) if a]
        fwd = np.array([views[e[1]].pose[:3, 2] for e in good]).reshape(-1, 3)
        g.agree, g.offset, g.members = len(good), med, [e[0] for e in good]
        g.rot_deg = float(np.median([e[5] for e in good])) if good else 0.0
        g.sessions = len({views[e[2]].image.session for e in good} - {s})
        g.spread_deg = float(np.degrees(np.arccos(np.clip((fwd @ fwd.T).min(), -1.0, 1.0)))) if good else 0.0
        g.near = (
            g.agree >= AGREE * len(loose) and float(np.linalg.norm(med)) > DRIFT_M and g.agree > len(tight)
        )
        g.ok = (
            g.near
            and g.agree >= LOOSE_OVER_TIGHT * len(tight)
            and (g.sessions >= 2 or g.spread_deg >= SPREAD_DEG)
        )
    for run in _runs([g for g in groups if g.ok]):
        if len(run) >= RUN_BINS:
            for g in run:
                g.admitted = True
    # A run's neighbours with the same offset are drifted too if their agreeing loose pairs merely outnumber
    # the tight ones, from whatever sessions: drift persists, and the bins at a run's edge see less (scan 5's
    # lap at 20-28 s: half its pairs to the hallway alone, tight ones among them along a blank wall).
    by = {(g.session, g.bin): g for g in groups}
    grew = True
    while grew:
        grew = False
        for g in [g for g in groups if g.admitted]:
            for k in (g.bin - 1, g.bin + 1):
                h = by.get((g.session, k))
                if (
                    h is not None
                    and not h.admitted
                    and h.near
                    and float(np.linalg.norm(h.offset - g.offset)) <= 2 * TOL_M
                ):
                    h.admitted = grew = True
    return groups


def _runs(groups: list[Group]) -> list[list[Group]]:
    """Groups of one session in a row: bins at most two apart, offsets within 2 TOL_M of the previous."""
    out: list[list[Group]] = []
    for g in sorted(groups, key=lambda g: (g.session, g.bin)):
        last = out[-1][-1] if out else None
        if (
            last is None
            or last.session != g.session
            or g.bin - last.bin > 2
            or float(np.linalg.norm(g.offset - last.offset)) > 2 * TOL_M
        ):
            out.append([])
        out[-1].append(g)
    return out


@dataclass
class Span:
    """Where a session drifted unseen: from its last view with a tight pair before an admitted run of bins
    (else its first view, `t0`) to the end of the run's first bin (`t1`: by then the drift is there), and
    on to its next tight pair after the run (else its last view, `t_end`): the views that are off."""

    session: str
    t0: float
    t1: float
    t_end: float
    offset: F64  # the run's first bin: offset (world, m) and rotation
    rot_deg: float
    trajectory: int | None = None  # a trajectory cut (`Cut`): the stretch of this trajectory, any session

    def holds(self, v: View) -> bool:
        """The view is in the span's stretch from t0 to t1 (its session's, or its trajectory's)."""
        if self.trajectory is not None:
            return v.trajectory == self.trajectory and self.t0 <= v.t <= self.t1
        return v.image.session == self.session and self.t0 <= v.t <= self.t1

    def frees(self, v: View) -> bool:
        """The view is known to be off: no pull to its start pose in the coarse solve."""
        if self.trajectory is not None:
            return v.trajectory == self.trajectory and self.t0 <= v.t <= self.t_end
        return v.image.session == self.session and self.t0 <= v.t <= self.t_end


def drift_spans(views: list[View], matches: list[PairMatches], groups: list[Group]) -> list[Span]:
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    for v in views:
        first[v.image.session] = min(first.get(v.image.session, v.t), v.t)
        last[v.image.session] = max(last.get(v.image.session, v.t), v.t)
    dropped = {n for g in groups if g.admitted for n in g.tight_members}
    seen: dict[str, list[float]] = defaultdict(list)  # times of views with a tight pair to another session
    for n, p in enumerate(matches):
        si, sj = views[p.i].image.session, views[p.j].image.session
        if len(p.ia) and p.rel is not None and _tight(p) and n not in dropped and si != sj:
            seen[si].append(views[p.i].t)
            seen[sj].append(views[p.j].t)
    out: list[Span] = []
    for run in _runs([g for g in groups if g.admitted]):
        s, g = run[0].session, run[0]
        start, end = first[s] + g.bin * BIN_S, first[s] + (run[-1].bin + 1) * BIN_S
        out.append(
            Span(
                s,
                max((t for t in seen[s] if t < start), default=first[s]),
                start + BIN_S,
                min((t for t in seen[s] if t > end), default=last[s]),
                g.offset,
                g.rot_deg,
            )
        )
    return out


def stretch_odometry(views: list[View], odo: list[Odometry], spans: list[Span]) -> tuple[list[Odometry], int]:
    """ARKit's relative motion, loosened where a session drifted unseen: the links of a span from `t0` to
    `t1` together can stretch by its offset and turn by its rotation, in any direction (each gets offset /
    sqrt(links) and rotation / sqrt(links) more sigma; gravity still holds roll and pitch). After `t1`
    ARKit's motion is healthy again (scan 5's lap: 50-53 cm off in every bin from 20 to 48 s) and stays as
    it was. Without this, odometry that is 0.5 m wrong over a few seconds (60 sigma) pulls the session's
    right start onto its drifted end. Variants tried on scan 5: loosening only along the offset, or only
    horizontally, or also after `t1`, bent the other sessions more (4-6 cm instead of 3.7) and made eval no
    better. Returns the links and how many were loosened."""
    out = list(odo)
    loosened: set[int] = set()
    for sp in spans:
        links = [
            n
            for n, o in enumerate(odo)
            if sp.holds(views[o.i])
            and sp.holds(views[o.j])
            and not views[o.i].still
            and not views[o.j].still
            and n not in loosened
        ]
        extra = np.zeros((6, 6))
        if links:
            extra[:3, :3] = np.radians(sp.rot_deg) ** 2 / len(links) * np.eye(3)
            extra[3:, 3:] = float(sp.offset @ sp.offset) / len(links) * np.eye(3)
        for n in links:
            o = out[n]
            out[n] = Odometry(o.i, o.j, o.rel, o.rot_deg, o.trans_m, extra)
            loosened.add(n)
    return out, len(loosened)


def coarse_solve(
    views: list[View], odo: list[Odometry], matches: list[PairMatches], groups: list[Group], spans: list[Span]
) -> tuple[list[F64], int]:
    """The coarse pose graph (see above): start poses for the fine round, and how many pairs the second
    solve kept. Solved with the tight pairs (but those in admitted bins) and the admitted loose ones. The
    views of a drift span have no pull to their start pose (they are known to be off: their priors would
    drag everything along), the others a loose one. Holding the others fixed instead, and releasing them
    only in the bundle adjustment, bent the model more on scan 5 (7 cm instead of 3.7: the adjustment then
    starts from an inconsistent point)."""
    admitted = {n for g in groups if g.admitted for n in g.members}
    dropped = {n for g in groups if g.admitted for n in g.tight_members}
    first = [
        n for n, p in enumerate(matches) if len(p.ia) and p.rel is not None and _tight(p) and n not in dropped
    ]
    free = [any(sp.frees(v) for sp in spans) for v in views]
    huber = gtsam.noiseModel.mEstimator.Huber.Create(HUBER)
    start = _coarse_graph(
        views, odo, matches, sorted({*first, *admitted}), huber, [v.pose for v in views], free
    )
    again = []
    for n, p in enumerate(matches):
        if len(p.ia) and p.rel is not None:
            _, oj, rot = pair_offsets(views, p, start)
            if np.linalg.norm(oj) <= REGATE_M and rot <= REGATE_DEG:
                again.append(n)
    cauchy = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
    return _coarse_graph(views, odo, matches, again, cauchy, start, free), len(again)


def _coarse_graph(
    views: list[View],
    odo: list[Odometry],
    matches: list[PairMatches],
    use: list[int],
    kernel: gtsam.noiseModel.mEstimator.Base,
    init: list[F64],
    free: list[bool],
) -> list[F64]:
    g = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    down = gtsam.Unit3(np.array([0.0, 0.0, -1.0]))
    gravity = gtsam.noiseModel.Isotropic.Sigma(2, np.radians(GRAVITY_DEG))
    for i, v in enumerate(views):
        values.insert(i, _pose3(init[i]))
        if not free[i]:
            g.add(gtsam.PriorFactorPose3(i, _pose3(v.pose), _sigmas(*COARSE_PRIOR)))
        g.add(
            gtsam.AttitudeFactorPose3(
                i, down, gravity, gtsam.Unit3(v.pose[:3, :3].T @ np.array([0.0, 0.0, -1.0]))
            )
        )
    for o in odo:
        g.add(gtsam.BetweenFactorPose3(o.i, o.j, _pose3(o.rel), o.noise()))
    noise = gtsam.noiseModel.Robust.Create(kernel, _sigmas(*COARSE_PAIR))
    for n in use:
        p = matches[n]
        assert p.rel is not None
        g.add(gtsam.BetweenFactorPose3(p.i, p.j, _pose3(np.linalg.inv(p.rel)), noise))
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    result = gtsam.LevenbergMarquardtOptimizer(g, values, params).optimize()
    return [result.atPose3(i).matrix() for i in range(len(views))]


# --------------------------------------------------------------------------- trajectory cuts
#
# The vote above asks, per session, whether a session's cameras sit off the other sessions'. That misses
# drift inside one ARKit trajectory that later sessions inherit: sessions that continue the previous one's
# ARSession (Next room, the closing lap) carry its drift on, so after it everything agrees with everything
# again, and the disagreement only shows between the blocks before and after. Scan 5: the living session
# drifted ~0.5 m between t ~110 and ~156 s, camera held high on plain walls and ceiling (nothing matched
# there; ARKit's tracking "normal" all along, briefly relocalised at 157.9 s and jumped back: ingest's
# "excursion"), and the hallway, kitchen and lap all continued its ARSession. The vote saw only the lap's
# bins disagreeing with three other sessions and moved the lap; the living room's south wall stayed doubled
# 40 cm apart.
#
# So first: a trajectory may be cut where ARKit had nothing to hold on to, and only there (aliasing is
# local to a surface and finds no cut to explain it): stretches of `UNSEEN_S` or more between views
# anchored by a coarse pair (matched under the loose gate) to a view `ANCHOR_S` or more away along the
# trajectory or to another trajectory (a camera that only sees again what it saw in the last half minute
# drifts with it: scan 5's living session at 128 / 144 s), and gaps in its frames (ingest's dropped windows: excursions,
# `drop_frames`; tracking that wasn't normal). A cut is admitted when the pairs crossing it (one view before
# its stretch, one after, same trajectory) tell one story: enough agreeing loose pairs, `AGREE` of them on
# one offset larger than `DRIFT_M`, outnumbering the crossing pairs that agree with the current poses (tight)
# `LOOSE_OVER_TIGHT`:1, from `RUN_BINS` bins or more on each side, and from two sessions or views
# `SPREAD_DEG` apart on the later side. The later side is moved (the trajectory's start holds the scene
# frame) by one rigid transform (yaw and a translation, from the agreeing pairs), spread linearly over the
# stretch; ARKit's relative motion over it is loosened for the coarse solve and the bundle adjustment
# (`Span` with the trajectory). The admitted cut that makes most coarse pairs consistent goes first; the
# others are re-tested at the moved poses. Then the coarse pairs are matched again and the vote runs as
# before. Without a cut nothing changes.

UNSEEN_S = 2 * BIN_S  # an unseen stretch is at least this long
ANCHOR_S = 30.0  # a view is anchored by a pair to a view this far away along its trajectory (or another one)
FRAME_GAP_S = 0.25  # a gap between a session's frames (60 Hz) this long is a discontinuity


@dataclass
class Cut:
    """A place a trajectory may have jumped: from `t0` (its last anchored view, or the last frame before a
    gap) to `t1` (the next one), and how the pairs crossing it voted at the poses it was tested at."""

    trajectory: int
    t0: float
    t1: float
    kind: str  # "unseen" (no anchored view in between) or "gap" (no frames in between)
    loose: int = 0
    tight: int = 0
    agree: int = 0
    bins_before: int = 0
    bins_after: int = 0
    sessions_after: int = 0
    spread_deg: float = 0.0
    offset: F64 = field(default_factory=lambda: np.zeros(3))  # median: where the earlier side puts the later
    yaw_deg: float = 0.0
    jump: F64 = field(default_factory=lambda: np.eye(4))  # world transform for the views after t1
    gain: int = 0  # coarse pairs consistent after the jump minus before
    admitted: bool = False
    members: list[int] = field(default_factory=list[int])  # agreeing crossing pairs


def _consistent(views: list[View], p: PairMatches, poses: list[F64]) -> bool:
    """The pair's PnP pose agrees with the poses within the consensus tolerance (and the loose gate's turn)."""
    _, oj, rot = pair_offsets(views, p, poses)
    dist = float(np.linalg.norm(poses[p.i][:3, 3] - poses[p.j][:3, 3]))
    return float(np.linalg.norm(oj)) <= TOL_M + TOL_REL * dist and rot <= LOOSE_DEG


def _yaw(r: F64) -> float:
    return float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))


def cut_candidates(
    views: list[View],
    matches: list[PairMatches],
    frame_gaps: list[tuple[int, float, float]] = [],  # noqa: B006
) -> list[Cut]:
    """Unseen stretches and frame gaps of each trajectory (see above), in time order. `frame_gaps`:
    (trajectory, last frame time before, first after) for gaps in a session's frames."""
    anchored: dict[int, set[float]] = defaultdict(set)
    for p in matches:
        if not len(p.ia) or p.rel is None:
            continue
        a, c = views[p.i], views[p.j]
        if a.trajectory != c.trajectory or abs(a.t - c.t) >= ANCHOR_S:
            anchored[a.trajectory] |= {a.t, c.t} if a.trajectory == c.trajectory else {a.t}
            if a.trajectory != c.trajectory:
                anchored[c.trajectory].add(c.t)
    out = [Cut(k, t0, t1, "gap") for k, t0, t1 in frame_gaps]
    for k, ts in anchored.items():
        tt = sorted(ts)
        out += [Cut(k, a, c, "unseen") for a, c in itertools.pairwise(tt) if c - a >= UNSEEN_S]
    return sorted(out, key=lambda c: (c.trajectory, c.t0, c.t1))


def _jumped(views: list[View], poses: list[F64], cut: Cut, jump: F64) -> list[F64]:
    """Poses with the trajectory's views after the cut moved by `jump`, spread linearly over the stretch."""
    yaw, t = np.radians(_yaw(jump[:3, :3])), jump[:3, 3]
    out: list[F64] = []
    for v, p in zip(views, poses, strict=True):
        if v.trajectory != cut.trajectory or v.t <= cut.t0:
            out.append(p)
            continue
        f = 1.0 if v.t >= cut.t1 or cut.t1 <= cut.t0 else (v.t - cut.t0) / (cut.t1 - cut.t0)
        m = np.eye(4)
        c, s = np.cos(f * yaw), np.sin(f * yaw)
        m[:2, :2] = [[c, -s], [s, c]]
        m[:3, 3] = f * t
        out.append(m @ p)
    return out


def check_cut(views: list[View], matches: list[PairMatches], cut: Cut, poses: list[F64]) -> Cut:
    """The consensus tests of one cut at `poses` (see above), with its jump and gain; returns the cut."""
    first: dict[int, float] = {}
    for v in views:
        first[v.trajectory] = min(first.get(v.trajectory, v.t), v.t)
    loose: list[tuple[int, int, int, F64, float]] = []  # (pair, earlier, later, offset of the later, yaw)
    tight = 0
    for n, p in enumerate(matches):
        if not len(p.ia) or p.rel is None:
            continue
        a, c = views[p.i], views[p.j]
        if a.trajectory != cut.trajectory or c.trajectory != cut.trajectory:
            continue
        if a.t <= cut.t0 and c.t >= cut.t1:
            e, later = p.i, p.j
        elif c.t <= cut.t0 and a.t >= cut.t1:
            e, later = p.j, p.i
        else:
            continue
        oi, oj, _ = pair_offsets(views, p, poses)
        if _consistent(views, p, poses):
            tight += 1
            continue
        at = poses[p.j] @ p.rel if later == p.i else poses[p.i] @ np.linalg.inv(p.rel)  # where e puts later
        loose.append((n, e, later, oi if later == p.i else oj, _yaw(at[:3, :3] @ poses[later][:3, :3].T)))
    cut.loose, cut.tight = len(loose), tight
    if len(loose) < GROUP_MIN:
        return cut
    off = np.array([x[3] for x in loose])
    med = np.median(off, axis=0)
    dist = np.array([np.linalg.norm(poses[x[1]][:3, 3] - poses[x[2]][:3, 3]) for x in loose])
    good = [
        x
        for x, ok in zip(loose, np.linalg.norm(off - med, axis=1) <= TOL_M + TOL_REL * dist, strict=True)
        if ok
    ]
    if not good:
        return cut
    bin_of = lambda i: int((views[i].t - first[cut.trajectory]) // BIN_S)  # noqa: E731
    fwd = np.array([poses[x[2]][:3, 2] for x in good])
    cut.agree, cut.offset, cut.members = len(good), med, [x[0] for x in good]
    cut.bins_before = len({bin_of(x[1]) for x in good})
    cut.bins_after = len({bin_of(x[2]) for x in good})
    cut.sessions_after = len({views[x[2]].image.session for x in good})
    cut.spread_deg = float(np.degrees(np.arccos(np.clip((fwd @ fwd.T).min(), -1.0, 1.0))))
    # the jump: yaw from the pairs' rotations, then the translation that puts the later cameras where the
    # earlier side's pairs put them (about their centroid)
    cut.yaw_deg = float(np.median([x[4] for x in good]))
    a = np.radians(cut.yaw_deg)
    r = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    src = np.array([poses[x[2]][:3, 3] for x in good])
    pivot = src.mean(axis=0)
    t = np.median(src + np.array([x[3] for x in good]) - (pivot + (src - pivot) @ r.T), axis=0)
    cut.jump = np.eye(4)
    cut.jump[:3, :3] = r
    cut.jump[:3, 3] = pivot + t - r @ pivot
    cut.admitted = (
        cut.agree >= RUN_BINS * GROUP_MIN
        and cut.agree >= AGREE * cut.loose
        and float(np.linalg.norm(med)) > DRIFT_M
        and cut.agree >= LOOSE_OVER_TIGHT * cut.tight
        and min(cut.bins_before, cut.bins_after) >= RUN_BINS
        and (cut.sessions_after >= 2 or cut.spread_deg >= SPREAD_DEG)
    )
    if cut.admitted:
        moved = _jumped(views, poses, cut, cut.jump)
        ok = [n for n, p in enumerate(matches) if len(p.ia) and p.rel is not None]
        cut.gain = sum(_consistent(views, matches[n], moved) for n in ok) - sum(
            _consistent(views, matches[n], poses) for n in ok
        )
        cut.admitted = cut.gain >= RUN_BINS * GROUP_MIN
    return cut


def trajectory_cuts(
    views: list[View],
    matches: list[PairMatches],
    frame_gaps: list[tuple[int, float, float]] = [],  # noqa: B006
) -> tuple[list[Cut], list[F64]]:
    """Every candidate cut as last tested, and the poses with the admitted ones applied (greedily: the one
    that makes most pairs consistent first, then the rest re-tested)."""
    poses = [v.pose for v in views]
    cands = cut_candidates(views, matches, frame_gaps)
    done: list[Cut] = []
    while cands:
        for c in cands:
            c.admitted = False
            check_cut(views, matches, c, poses)
        best = max((c for c in cands if c.admitted), key=lambda c: c.gain, default=None)
        if best is None:
            break
        poses = _jumped(views, poses, best, best.jump)
        done.append(best)
        cands = [c for c in cands if c is not best]
    for c in cands:
        c.admitted = False
    return sorted([*done, *cands], key=lambda c: (c.trajectory, c.t0)), poses


def cut_spans(views: list[View], cuts: list[Cut]) -> list[Span]:
    """The admitted cuts as drift spans of their trajectory: ARKit's relative motion over the stretch is
    loosened by the jump (`stretch_odometry`), and the stretch's views lose their pull in the coarse solve."""
    out: list[Span] = []
    for c in cuts:
        if not c.admitted:
            continue
        inside = [v for v in views if v.trajectory == c.trajectory and c.t0 <= v.t <= c.t1]
        s = inside[0].image.session if inside else ""
        rot = float(np.degrees(np.arccos(np.clip((np.trace(c.jump[:3, :3]) - 1) / 2, -1.0, 1.0))))
        out.append(Span(s, c.t0, c.t1, c.t1, c.jump[:3, 3].copy(), rot, c.trajectory))
    return out


def rigid_fit(before: list[F64], after: list[F64]) -> F64:
    """One world transform (yaw and translation) that takes the `before` poses closest to `after`: the
    median turn about z, then the median of what is left of the camera positions' move."""
    yaw = np.radians(np.median([_yaw(a[:3, :3] @ b_[:3, :3].T) for b_, a in zip(before, after, strict=True)]))
    r = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    pb = np.array([m[:3, 3] for m in before])
    pa = np.array([m[:3, 3] for m in after])
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = np.median(pa - pb @ r.T, axis=0)
    return out


# --------------------------------------------------------------------------- tracks


@dataclass
class Tracks:
    view: np.ndarray  # (M,) observation -> view index
    kp: np.ndarray  # (M,) keypoint index in that view
    track: np.ndarray  # (M,) observation -> track index
    n: int  # number of tracks


def build_tracks(feats: list[Features], matches: Iterable[PairMatches]) -> Tracks:
    """Union-find over matched keypoints; tracks with two keypoints in one view are dropped."""
    off = np.concatenate([[0], np.cumsum([len(f.uv) for f in feats])]).astype(np.int64)
    parent = np.arange(int(off[-1]))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != root:
            parent[x], x = root, int(parent[x])
        return root

    used = np.zeros(len(parent), bool)
    for m in matches:
        for a, c in zip(off[m.i] + m.ia, off[m.j] + m.ib, strict=True):
            ra, rc = find(int(a)), find(int(c))
            used[a] = used[c] = True
            if ra != rc:
                parent[max(ra, rc)] = min(ra, rc)
    nodes = np.flatnonzero(used)
    roots = np.array([find(int(x)) for x in nodes], dtype=np.int64)
    view = np.searchsorted(off, nodes, side="right") - 1
    order = np.lexsort((view, roots))
    nodes, roots, view = nodes[order], roots[order], view[order]
    _, first, counts = np.unique(roots, return_index=True, return_counts=True)
    bad = np.zeros(len(nodes), bool)
    dup = np.flatnonzero((roots[1:] == roots[:-1]) & (view[1:] == view[:-1]))
    bad_roots = set(roots[dup].tolist())
    if bad_roots:
        bad = np.isin(roots, list(bad_roots))
    keep = ~bad & np.repeat(counts >= 2, counts)
    nodes, roots, view = nodes[keep], roots[keep], view[keep]
    _, track = np.unique(roots, return_inverse=True)
    del first
    return Tracks(
        view=view.astype(np.int64),
        kp=(nodes - off[view]).astype(np.int64),
        track=track.astype(np.int64),
        n=int(track.max() + 1) if len(track) else 0,
    )


# --------------------------------------------------------------------------- bundle adjustment


@dataclass
class Odometry:
    i: int
    j: int
    rel: F64  # pose of view j in view i's camera, from ARKit's raw poses
    rot_deg: float  # sigmas
    trans_m: float
    extra: F64 | None = (
        None  # covariance added to theirs (gtsam's order: rotation, translation; view i's axes)
    )

    def noise(self) -> gtsam.noiseModel.Base:
        if self.extra is None:
            return _sigmas(self.rot_deg, self.trans_m)
        cov = np.diag([np.radians(self.rot_deg) ** 2] * 3 + [self.trans_m**2] * 3) + self.extra
        return gtsam.noiseModel.Gaussian.Covariance(cov)


def odometry(
    views: list[View], raw_pose: Callable[[PosedImage], F64 | None], loose: Callable[[View], bool]
) -> list[Odometry]:
    """ARKit's relative motion: consecutive frame views of a trajectory (mm over seconds, ~1 % of the distance
    walked; `loose` views x10, e.g. while a relocalised world settles), and each still to its nearest frame
    view of the same session (looser: a still is a separate capture)."""
    out: list[Odometry] = []
    frames = [i for i, v in enumerate(views) if not v.still]
    for a, c in itertools.pairwise(frames):
        va, vc = views[a], views[c]
        if va.trajectory != vc.trajectory:
            continue
        pa, pc = raw_pose(va.image), raw_pose(vc.image)
        if pa is None or pc is None:
            continue
        rel = np.linalg.inv(pa) @ pc
        dt = vc.t - va.t
        f = 10.0 if loose(va) or loose(vc) else 1.0
        out.append(
            Odometry(
                a,
                c,
                rel,
                f * (0.05 + 0.02 * dt),
                f * (0.001 + 0.01 * float(np.linalg.norm(rel[:3, 3])) + 0.001 * dt),
            )
        )
    ft = np.array([views[i].t for i in frames])
    for s, v in enumerate(views):
        if not v.still or not len(ft):
            continue
        same = [n for n, i in enumerate(frames) if views[i].image.session == v.image.session]
        if not same:
            continue
        n = min(same, key=lambda n: abs(ft[n] - v.t))
        i = frames[n]
        pa, ps = raw_pose(views[i].image), raw_pose(v.image)
        if pa is None or ps is None or abs(views[i].t - v.t) > 2.0:
            continue
        rel = np.linalg.inv(pa) @ ps
        out.append(Odometry(i, s, rel, 0.3, 0.005 + 0.01 * float(np.linalg.norm(rel[:3, 3]))))
    return out


@dataclass
class Observations:
    """One row per (track, view) observation that goes into the adjustment."""

    view: np.ndarray  # (M,)
    track: np.ndarray  # (M,) landmark index
    uv: F64  # (M, 2) at WIDTH x HEIGHT
    depth: F64  # (M,) LiDAR depth, 0 = none
    conf: np.ndarray  # (M,)


def observations(feats: list[Features], uvs: list[F64], tracks: Tracks) -> Observations:
    uv = np.array([uvs[v][k] for v, k in zip(tracks.view, tracks.kp, strict=True)]).reshape(-1, 2)
    depth = np.array([feats[v].depth[k] for v, k in zip(tracks.view, tracks.kp, strict=True)], np.float64)
    conf = np.array([feats[v].conf[k] for v, k in zip(tracks.view, tracks.kp, strict=True)], np.uint8)
    return Observations(tracks.view.copy(), tracks.track.copy(), uv, depth, conf)


def _rays(views: list[View], obs: Observations) -> F64:
    """Camera-frame rays (z = 1) of the observations."""
    fx = np.array([views[v].k.fx for v in obs.view])
    fy = np.array([views[v].k.fy for v in obs.view])
    cx = np.array([views[v].k.cx for v in obs.view])
    cy = np.array([views[v].k.cy for v in obs.view])
    return np.column_stack([(obs.uv[:, 0] - cx) / fx, (obs.uv[:, 1] - cy) / fy, np.ones(len(obs.view))])


def lidar_points(views: list[View], poses: list[F64], obs: Observations) -> F64:
    """World point of each observation from its LiDAR depth (NaN without)."""
    rays = _rays(views, obs) * obs.depth[:, None]
    rot = np.array([poses[v][:3, :3] for v in obs.view])
    t = np.array([poses[v][:3, 3] for v in obs.view])
    pts = np.einsum("nij,nj->ni", rot, rays) + t
    pts[obs.depth <= 0] = np.nan
    return pts


def triangulate(
    views: list[View], poses: list[F64], obs: Observations, n_tracks: int
) -> tuple[F64, np.ndarray]:
    """Least-squares point closest to each track's rays (each ray weighted by 1 / depth^2, roughly pixel
    error), and whether it is usable (in front of every camera, rays not parallel)."""
    rays = _rays(views, obs)
    rot = np.array([poses[v][:3, :3] for v in obs.view])
    d = np.einsum("nij,nj->ni", rot, rays)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = np.array([poses[v][:3, 3] for v in obs.view])
    proj = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    # first pass unweighted, then weighted by the distance it found
    pts = np.zeros((n_tracks, 3))
    w = np.ones(len(obs.view))
    a = np.zeros((n_tracks, 3, 3))
    good = np.zeros(n_tracks, bool)
    for _ in range(2):
        a = np.zeros((n_tracks, 3, 3))
        rhs = np.zeros((n_tracks, 3))
        np.add.at(a, obs.track, w[:, None, None] * proj)
        np.add.at(rhs, obs.track, w[:, None] * np.einsum("nij,nj->ni", proj, c))
        det = np.linalg.det(a)
        good = np.abs(det) > 1e-9
        pts[good] = np.linalg.solve(a[good], rhs[good][..., None])[..., 0]
        z = np.einsum("ni,ni->n", pts[obs.track] - c, d)
        w = 1.0 / np.maximum(z, 0.1) ** 2
    # conditioning: smallest eigenvalue of the normal matrix vs the largest
    ev = np.linalg.eigvalsh(np.where(good[:, None, None], a, np.eye(3)[None]))
    ok = good & (ev[:, 0] > 1e-4 * ev[:, 2])
    z = np.einsum("ni,ni->n", pts[obs.track] - c, d)
    front = np.ones(n_tracks, bool)
    np.logical_and.at(front, obs.track, z > 0.1)
    return pts, ok & front


def reprojection(views: list[View], poses: list[F64], obs: Observations, pts: F64) -> tuple[F64, F64]:
    """(pixel error at WIDTH, depth) of each observation of `pts` seen from `poses`."""
    w2c = [np.linalg.inv(p) for p in poses]
    rot = np.array([w2c[v][:3, :3] for v in obs.view])
    t = np.array([w2c[v][:3, 3] for v in obs.view])
    q = np.einsum("nij,nj->ni", rot, pts[obs.track]) + t
    z = q[:, 2]
    fx = np.array([views[v].k.fx for v in obs.view])
    fy = np.array([views[v].k.fy for v in obs.view])
    cx = np.array([views[v].k.cx for v in obs.view])
    cy = np.array([views[v].k.cy for v in obs.view])
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * q[:, 0] / z + cx
        v = fy * q[:, 1] / z + cy
    err = np.hypot(u - obs.uv[:, 0], v - obs.uv[:, 1])
    err[~(z > 0.05)] = np.inf
    return err, z


def init_landmarks(
    views: list[View], obs: Observations, n_tracks: int, max_spread: float = 0.10
) -> tuple[F64, np.ndarray, np.ndarray]:
    """Landmark positions from the observations' LiDAR points (median), else triangulated; plus which
    tracks to keep (LiDAR points within `max_spread` m of their median: a larger spread is a moved object,
    a reflection or a wrong merge) and which observations (LiDAR point > 2 x max_spread off: dropped)."""
    poses = [v.pose for v in views]
    lp = lidar_points(views, poses, obs)
    has = ~np.isnan(lp[:, 0])
    pts, tri_ok = triangulate(views, poses, obs, n_tracks)
    med = np.full((n_tracks, 3), np.nan)
    order = np.argsort(obs.track[has], kind="stable")
    tr = obs.track[has][order]
    p = lp[has][order]
    starts = np.flatnonzero(np.r_[True, tr[1:] != tr[:-1]])
    ends = np.r_[starts[1:], len(tr)]
    spread = np.full(n_tracks, np.nan)
    for s, e in zip(starts, ends, strict=True):
        m = np.median(p[s:e], axis=0)
        med[tr[s]] = m
        spread[tr[s]] = float(np.median(np.linalg.norm(p[s:e] - m, axis=1)))
    lidar = ~np.isnan(med[:, 0])
    out = np.where(lidar[:, None], med, pts)
    keep = np.where(lidar, spread <= max_spread, tri_ok)
    obs_ok = np.ones(len(obs.view), bool)
    far = has & (np.linalg.norm(lp - out[obs.track], axis=1) > 2 * max_spread)
    obs_ok[far] = False
    return out, keep, obs_ok


def useful_tracks(
    views: list[View], obs: Observations, n_tracks: int, cross_dt: float = CROSS_DT
) -> np.ndarray:
    """Tracks with >= 3 observations or spanning >= cross_dt / two trajectories (a cross-pass constraint)."""
    count = np.bincount(obs.track, minlength=n_tracks)
    t = np.array([views[v].t + 1e5 * views[v].trajectory for v in obs.view])
    lo = np.full(n_tracks, np.inf)
    hi = np.full(n_tracks, -np.inf)
    np.minimum.at(lo, obs.track, t)
    np.maximum.at(hi, obs.track, t)
    return (count >= 3) | (hi - lo >= cross_dt)


# --------------------------------------------------------------------------- floor heights


@dataclass
class Floor:
    """Floor observations: view `view[n]` saw floor cell `keys[cell[n]]` at the point `cam[n]` (its camera
    frame; the cell's centre at the floor height the view's LiDAR measured)."""

    view: np.ndarray  # (F,) view index
    cell: np.ndarray  # (F,) index into keys
    cam: F64  # (F, 3)
    keys: list[tuple[int, int]]
    level: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # (cells,) floor level, -1: none

    def __len__(self) -> int:
        return len(self.view)


def floor_views(views: list[View], min_move: float = FLOOR_MOVE, min_turn: float = FLOOR_TURN) -> list[int]:
    """Frame views thinned by motion per session (the pose graph's keyframe spacing): whose floor to measure."""
    out: list[int] = []
    last: dict[str, F64] = {}
    for i, v in enumerate(views):
        if v.still or not v.image.depth:
            continue
        lp = last.get(v.image.session)
        if lp is not None:
            move = float(np.linalg.norm(v.pose[:3, 3] - lp[:3, 3]))
            if move < min_move and _angle_deg(lp[:3, :3], v.pose[:3, :3]) < min_turn:
                continue
        last[v.image.session] = v.pose
        out.append(i)
    return out


def floor_observations(
    views: list[View],
    cells: dict[int, dict[tuple[int, int], float]],
    min_dt: float = 15.0,
    per_session: int = FLOOR_PER_SESSION,
    cell_size: float = 0.5,
) -> Floor:
    """Floor observations from per-view floor cells (`align.floors` at the views' poses: cell -> world z).
    Only cells seen at least `min_dt` apart or from two trajectories; per cell and session at most
    `per_session` views, spread evenly in time."""
    by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, cs in cells.items():
        for c in cs:
            by_cell[c].append(i)
    view: list[int] = []
    cell: list[int] = []
    cam: list[F64] = []
    keys: list[tuple[int, int]] = []
    for c, ids in sorted(by_cell.items()):
        t = np.array([views[i].t + 1e5 * views[i].trajectory for i in ids])
        if t.max() - t.min() < min_dt:
            continue
        by_s: dict[str, list[int]] = defaultdict(list)
        for i in sorted(ids, key=lambda i: views[i].t):
            by_s[views[i].image.session].append(i)
        keep: list[int] = []
        for ss in by_s.values():
            pick = np.linspace(0, len(ss) - 1, min(per_session, len(ss))).round().astype(int)
            keep += [ss[int(k)] for k in np.unique(pick)]
        for i in keep:
            p = np.array([(c[0] + 0.5) * cell_size, (c[1] + 0.5) * cell_size, cells[i][c]])
            pose = views[i].pose
            view.append(i)
            cell.append(len(keys))
            cam.append(pose[:3, :3].T @ (p - pose[:3, 3]))
        keys.append(c)
    out = Floor(
        np.array(view, np.int64), np.array(cell, np.int64), np.array(cam, np.float64).reshape(-1, 3), keys
    )
    out.level = floor_levels(out, floor_heights([v.pose for v in views], out))
    return out


def floor_levels(floor: Floor, heights: F64, step: float = 0.015, min_cells: int = 4) -> np.ndarray:
    """Per cell, its floor level: cells next to each other (8-neighbours) whose heights differ by less than
    `step` are one level (a room's floor, or rooms on one floor without a threshold step); -1 for levels
    of fewer than `min_cells` cells."""
    h = _cell_medians(heights, floor) if len(floor) else np.zeros(0)
    idx = {c: i for i, c in enumerate(floor.keys)}
    parent = list(range(len(floor.keys)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (cx, cy), i in idx.items():
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            j = idx.get((cx + dx, cy + dy))
            if j is not None and abs(h[i] - h[j]) < step:
                parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(floor.keys))], np.int64)
    out = np.full(len(floor.keys), -1, np.int64)
    for k, r in enumerate(r for r in np.unique(roots) if (roots == r).sum() >= min_cells):
        out[roots == r] = k
    return out


def floor_heights(poses: list[F64], floor: Floor) -> F64:
    """World z of each floor observation at `poses`."""
    if not len(floor):
        return np.zeros(0)
    rot = np.array([poses[v][2, :3] for v in floor.view])
    t = np.array([poses[v][2, 3] for v in floor.view])
    return np.einsum("ni,ni->n", rot, floor.cam) + t


def _cell_medians(h: F64, floor: Floor) -> F64:
    out = np.full(len(floor.keys), np.nan)
    order = np.argsort(floor.cell, kind="stable")
    c, hs = floor.cell[order], h[order]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(c)], strict=True):
        out[c[s]] = float(np.median(hs[s:e]))
    return out


def main_floor_lift(before: F64, after: F64, floor: Floor, level_m: float = 0.03) -> float:
    """Vertical shift that puts the main floor level (the most observed one, before the adjustment) back at
    its height before: median over its cells of the cell height before - after."""
    if not len(floor):
        return 0.0
    hb, ha = _cell_medians(before, floor), _cell_medians(after, floor)
    n = np.bincount(floor.cell, minlength=len(floor.keys))
    hist, edges = np.histogram(hb, bins=np.arange(hb.min(), hb.max() + 0.02, 0.01), weights=n)
    main = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2)
    on = np.abs(hb - main) < level_m
    return float(np.median((hb - ha)[on]))


def floor_report(views: list[View], before: list[F64], after: list[F64], floor: Floor) -> dict[str, object]:
    """How consistent the floor heights are, before and after: per observation, the distance from its cell's
    median height (mm); per pair of sessions, the median over their shared cells of the difference of their
    own cell heights (mm; a session's floor sitting higher or lower than another's, the drift these factors
    remove). Plus the camera height change fitted as a plane over the scan (mm per m along x and y)."""
    sessions = sorted({views[v].image.session for v in floor.view})
    sess = np.array([sessions.index(views[v].image.session) for v in floor.view], np.int64)

    def one(poses: list[F64]) -> tuple[dict[str, float] | None, dict[str, tuple[float, int]]]:
        h = floor_heights(poses, floor)
        if not len(h):
            return None, {}
        res = np.abs(h - _cell_medians(h, floor)[floor.cell])
        own: dict[tuple[int, int], list[float]] = defaultdict(list)
        for s, c, z in zip(sess, floor.cell, h, strict=True):
            own[(int(s), int(c))].append(float(z))
        med = {k: float(np.median(v)) for k, v in own.items()}
        pairs: dict[str, tuple[float, int]] = {}
        for a, b2 in itertools.combinations(range(len(sessions)), 2):
            d = [
                med[(a, c)] - med[(b2, c)] for c in range(len(floor.keys)) if (a, c) in med and (b2, c) in med
            ]
            if len(d) >= 3:
                pairs[f"{sessions[a]}/{sessions[b2]}"] = (float(np.median(d)), len(d))
        return {
            "median": round(1000 * float(np.median(res)), 1),
            "p90": round(1000 * float(np.percentile(res, 90)), 1),
        }, pairs

    res_b, pairs_b = one(before)
    res_a, pairs_a = one(after)
    offsets = {
        k: {"cells": n, "before_mm": round(1000 * pairs_b[k][0], 1), "after_mm": round(1000 * d, 1)}
        for k, (d, n) in pairs_a.items()
    }
    cams = np.array([p[:3, 3] for p in before])
    dz = np.array([a[2, 3] - b2[2, 3] for a, b2 in zip(after, before, strict=True)])
    plane = np.linalg.lstsq(np.c_[cams[:, :2], np.ones(len(cams))], dz, rcond=None)[0]
    return {
        "observations": len(floor),
        "cells": len(floor.keys),
        "views": len(np.unique(floor.view)),
        "residual_mm_before": res_b,
        "residual_mm_after": res_a,
        "session_offset_mm": offsets,
        "session_offset_max_mm": {
            "before": round(max((abs(v["before_mm"]) for v in offsets.values()), default=0.0), 1),
            "after": round(max((abs(v["after_mm"]) for v in offsets.values()), default=0.0), 1),
        },
        "camera_dz_plane": {
            "x_mm_per_m": round(1000 * float(plane[0]), 2),
            "y_mm_per_m": round(1000 * float(plane[1]), 2),
            "offset_mm": round(1000 * float(plane[2]), 1),
        },
    }


def _pose3(m: F64) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(m[:3, :3]), gtsam.Point3(*m[:3, 3]))


def _sigmas(rot_deg: float, trans_m: float) -> gtsam.noiseModel.Diagonal:
    return gtsam.noiseModel.Diagonal.Sigmas(np.array([np.radians(rot_deg)] * 3 + [trans_m] * 3))


@dataclass
class Adjusted:
    poses: list[F64]
    points: F64
    iterations: int
    error_before: float
    error_after: float


def adjust(
    views: list[View],
    obs: Observations,
    points: F64,
    use: np.ndarray,
    odo: list[Odometry],
    prior: tuple[float, float] = (3.0, 0.3),
    range_sigma: tuple[float, float] = (0.02, 0.01),
    max_iterations: int = 40,
    start: list[F64] | None = None,
    floor: Floor | None = None,
    floor_sigma: float = FLOOR_SIGMA,
    level_sigma: float = LEVEL_SIGMA,
    gravity_deg: float = GRAVITY_DEG,
) -> Adjusted:
    """Levenberg-Marquardt over all views and the landmarks of the `use`d observations (see the module
    docstring). `prior`: (deg, m) sigma of each view's pull to its start pose; `range_sigma`: (m, fraction
    of depth) of the LiDAR range prior (confidence-2 depth only). `start`: initial poses (default the views'
    own, which the priors always pull towards). `floor`: floor observations, a height landmark per cell
    (`floor_sigma` m in world z, Huber), the cells of one floor level held level (`level_sigma`, 0: not);
    `gravity_deg`: sigma of each view's pull to its start pose's gravity direction."""

    def x(i: int) -> int:
        return gtsam.symbol("x", i)

    def lm(j: int) -> int:
        return gtsam.symbol("l", j)

    g = gtsam.NonlinearFactorGraph()
    init = gtsam.Values()
    for i, v in enumerate(views):
        init.insert(x(i), _pose3(start[i] if start is not None else v.pose))
        g.add(gtsam.PriorFactorPose3(x(i), _pose3(v.pose), _sigmas(*prior)))
    down = gtsam.Unit3(np.array([0.0, 0.0, -1.0]))
    gravity = gtsam.noiseModel.Isotropic.Sigma(2, np.radians(gravity_deg))
    for i, v in enumerate(views):
        g.add(
            gtsam.AttitudeFactorPose3(
                x(i), down, gravity, gtsam.Unit3(v.pose[:3, :3].T @ np.array([0.0, 0.0, -1.0]))
            )
        )
    for o in odo:
        g.add(gtsam.BetweenFactorPose3(x(o.i), x(o.j), _pose3(o.rel), o.noise()))
    tracks = np.unique(obs.track[use])
    for j in tracks:
        init.insert(lm(int(j)), gtsam.Point3(*points[j]))
    noise_of: dict[float, gtsam.noiseModel.Base] = {}

    def pix(sigma: float) -> gtsam.noiseModel.Base:
        key = round(sigma, 1)
        if key not in noise_of:
            noise_of[key] = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(HUBER), gtsam.noiseModel.Isotropic.Sigma(2, key)
            )
        return noise_of[key]

    cal = [gtsam.Cal3_S2(v.k.fx, v.k.fy, 0.0, v.k.cx, v.k.cy) for v in views]
    rays = _rays(views, obs)
    for n in np.flatnonzero(use):
        v, j = int(obs.view[n]), int(obs.track[n])
        g.add(
            gtsam.GenericProjectionFactorCal3_S2(
                obs.uv[n].astype(np.float64), pix(views[v].sigma), x(v), lm(j), cal[v], False, False
            )
        )
        if obs.depth[n] > 0 and obs.conf[n] >= 2:
            rng = float(obs.depth[n] * np.linalg.norm(rays[n]))
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(HUBER),
                gtsam.noiseModel.Isotropic.Sigma(1, range_sigma[0] + range_sigma[1] * float(obs.depth[n])),
            )
            g.add(gtsam.RangeFactor3D(x(v), lm(j), rng, noise))
    if floor is not None and len(floor):
        _floor_factors(g, init, floor, floor_sigma, level_sigma, [v.pose for v in views], x)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iterations)
    params.setRelativeErrorTol(1e-4)
    opt = gtsam.LevenbergMarquardtOptimizer(g, init, params)
    before = g.error(init)
    result = opt.optimize()
    pts = points.copy()
    for j in tracks:
        pts[j] = result.atPoint3(lm(int(j)))
    return Adjusted(
        [result.atPose3(x(i)).matrix() for i in range(len(views))],
        pts,
        opt.iterations(),
        before,
        g.error(result),
    )


def _floor_factors(
    g: gtsam.NonlinearFactorGraph,
    init: gtsam.Values,
    floor: Floor,
    sigma: float,
    level_sigma: float,
    poses: list[F64],
    x: Callable[[int], int],
) -> None:
    """Per floor observation: its point in the camera frame (a variable held by a tight prior), and
    pose * point = the cell's landmark, the error in world coordinates: only z counts (x, y FLOOR_FREE).
    Per floor level: a shared height that each of its cells' landmarks keeps to (z only)."""

    def fp(n: int) -> int:
        return gtsam.symbol("p", n)

    def fl(c: int) -> int:
        return gtsam.symbol("f", c)

    held = gtsam.noiseModel.Isotropic.Sigma(3, FLOOR_HELD)
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(HUBER),
        gtsam.noiseModel.Diagonal.Sigmas(np.array([FLOOR_FREE, FLOOR_FREE, sigma])),
    )
    world = np.array(
        [poses[v][:3, :3] @ p + poses[v][:3, 3] for v, p in zip(floor.view, floor.cam, strict=True)]
    ).reshape(-1, 3)
    cz = np.array([np.median(world[floor.cell == c], axis=0) for c in range(len(floor.keys))]).reshape(-1, 3)
    for c in range(len(floor.keys)):
        init.insert(fl(c), gtsam.Point3(*cz[c]))
    for n, (v, c) in enumerate(zip(floor.view, floor.cell, strict=True)):
        init.insert(fp(n), gtsam.Point3(*floor.cam[n]))
        g.add(gtsam.PriorFactorPoint3(fp(n), gtsam.Point3(*floor.cam[n]), held))
        g.add(gtsam.ReferenceFrameFactorPoint3Pose3(fp(n), x(int(v)), fl(int(c)), noise))
    if level_sigma > 0:
        lnoise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(HUBER),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([FLOOR_FREE, FLOOR_FREE, level_sigma])),
        )
        for k in np.unique(floor.level[floor.level >= 0]):
            on = np.flatnonzero(floor.level == k)
            zc = np.median(cz[on], axis=0)
            init.insert(gtsam.symbol("z", int(k)), gtsam.Point3(*zc))
            for c in on:
                rel = gtsam.Point3(zc[0] - cz[c][0], zc[1] - cz[c][1], 0.0)  # only z is compared
                g.add(gtsam.BetweenFactorPoint3(fl(int(c)), gtsam.symbol("z", int(k)), rel, lnoise))


# --------------------------------------------------------------------------- diagnostics

BANDS = ((0.0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 99.0))


def residual_stats(
    views: list[View], poses: list[F64], obs: Observations, keep: np.ndarray, cross: np.ndarray
) -> dict[str, object]:
    """Reprojection error (px at 960) of the tracks re-triangulated from `poses`: median and p90 per
    distance band, for observations of cross-pass tracks and of single-pass ones, and per session."""
    n_tracks = int(obs.track.max()) + 1 if len(obs.track) else 0
    sub = Observations(obs.view[keep], obs.track[keep], obs.uv[keep], obs.depth[keep], obs.conf[keep])
    pts, ok = triangulate(views, poses, sub, n_tracks)
    err, z = reprojection(views, poses, sub, pts)
    good = ok[sub.track] & np.isfinite(err)
    xc = cross[sub.track]

    def stat(m: np.ndarray) -> dict[str, float | int] | None:
        e = err[m & good]
        if len(e) < 20:
            return None
        return {
            "n": len(e),
            "median": round(float(np.median(e)), 2),
            "p90": round(float(np.percentile(e, 90)), 2),
        }

    out: dict[str, object] = {"all": stat(np.ones(len(err), bool))}
    for name, m in (("cross_pass", xc), ("same_pass", ~xc)):
        out[name] = {f"{lo:g}-{hi:g}m": stat(m & (z >= lo) & (z < hi)) for lo, hi in BANDS} | {"all": stat(m)}
    sessions = sorted({v.image.session for v in views})
    out["sessions"] = {s: stat(np.array([views[v].image.session == s for v in sub.view])) for s in sessions}
    return out


def session_pairs(views: list[View], matches: list[PairMatches]) -> dict[str, dict[str, object]]:
    """Per pair of sessions ("a/b", a the earlier view; a session with itself: its other passes), over the
    cross pairs: how many were matched, accepted (PnP within the fine round's gate) and rejected (PnP
    solved but beyond it), their keypoint matches, and where the rejected pairs' PnP poses put the later
    camera (median offset, mm, and the share within TOL of it). `drift`: GROUP_MIN rejected pairs or more,
    AGREE of them on one offset larger than DRIFT_M: the poses are off by that, not the matches wrong."""
    rows: dict[str, list[PairMatches]] = defaultdict(list)
    for p in matches:
        if p.kind == "cross":
            rows[f"{views[p.i].image.session}/{views[p.j].image.session}"].append(p)
    out: dict[str, dict[str, object]] = {}
    for key, ps in sorted(rows.items()):
        acc = [p for p in ps if len(p.ia) and _tight(p)]
        rej = [p for p in ps if p.rel is not None and not _tight(p)]
        row: dict[str, object] = {
            "pairs": len(ps),
            "accepted": len(acc),
            "rejected": len(rej),
            "matches_accepted": int(sum(len(p.ia) for p in acc)),
            "matches_rejected": int(sum(p.candidates for p in rej)),
        }
        if rej:
            off = np.array([pair_offsets(views, p)[1] for p in rej])
            med = np.median(off, axis=0)
            dist = np.array([np.linalg.norm(views[p.i].pose[:3, 3] - views[p.j].pose[:3, 3]) for p in rej])
            agree = float(np.mean(np.linalg.norm(off - med, axis=1) <= TOL_M + TOL_REL * dist))
            row |= {
                "rejected_offset_mm": round(1000 * float(np.linalg.norm(med)), 1),
                "rejected_offset_xyz_mm": [round(1000 * float(x), 1) for x in med],
                "rejected_agree": round(agree, 2),
                "drift": len(rej) >= GROUP_MIN and agree >= AGREE and float(np.linalg.norm(med)) > DRIFT_M,
            }
        out[key] = row
    return out


def cross_tracks(views: list[View], obs: Observations, n_tracks: int) -> np.ndarray:
    t = np.array([views[v].t + 1e5 * views[v].trajectory for v in obs.view])
    lo = np.full(n_tracks, np.inf)
    hi = np.full(n_tracks, -np.inf)
    np.minimum.at(lo, obs.track, t)
    np.maximum.at(hi, obs.track, t)
    return hi - lo >= CROSS_DT


def correction(before: F64, after: F64) -> tuple[float, float]:
    """(translation m, rotation deg) between two camera poses."""
    return float(np.linalg.norm(after[:3, 3] - before[:3, 3])), _angle_deg(before[:3, :3], after[:3, :3])
