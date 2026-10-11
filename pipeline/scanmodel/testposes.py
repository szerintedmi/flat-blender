"""Re-solved test poses: each test frame's pose solved against the training frames only.

A test frame's PSNR measures the splat only if its pose agrees with the training poses the splat was
fitted in. Scan 5's lap drifted 48 cm in ARKit, and its extrap PSNR fell 7-9 dB for that alone. So
`scanmodel splat` re-solves every test frame (interp and extrap, stills too) from the training frames into
`test_poses.json` next to transforms.json (contract in trainer/README.md); `gs_trainer.evaluate` then scores
each set at the dataset poses and at the re-solved ones. `python -m scanmodel.testposes DATASET` (or
`scanmodel test-poses`) writes the file for a dataset exported before.

The re-solved pose is consistent with the training poses by construction, which is the point: it is
where the splat expects that photo, not where the camera "really" was. Interp frames are a sanity
channel: their references are the neighbouring frames of the same pass, so they should barely move.

The file is not part of the dataset's cloud hash (`gs_trainer.cloud.HASHED_FILES`): training never reads
it, so a new solver version re-scores old runs without re-uploading the dataset. It records the SHA-256
of the transforms.json and split.json it was solved from instead, and the eval refuses it on a mismatch.

Per test frame:

- References: training video frames with LiDAR depth (never a test frame) within `REF_DIST_M` and
  `REF_DEG` of the test frame's dataset pose, ranked by co-visibility (a grid of each frame's LiDAR points
  seen by the other camera, not hidden behind its LiDAR surface: a frame behind a wall shares no view),
  the best `MAX_REFS` with at least `MIN_OVERLAP`.
- Features: RootSIFT at 960 px (contrast threshold 0.02 for plain walls), matched without any pose
  guidance (the dataset pose may be far off): ratio test and mutual nearest neighbour per reference. The
  reference keypoint is lifted to 3D with its LiDAR depth (bilinear, confidence 2, not at a depth edge:
  the 4x4 depth pixels around it span less than 2 cm + 3 %) at its training pose; one 3D point per test
  keypoint (its best match) for the joint solve.
- Solve: PnP RANSAC (SQPnP) on those 2D-3D matches, then Levenberg-Marquardt on the inliers.
- Accepted only when (else "unresolved" with the reason, never a doubtful pose): enough inliers and a
  large enough share of them; inliers spread over the image (3 of 4 quadrants, or a large convex hull); a
  small median reprojection error; a small pose uncertainty from the PnP Jacobian (it ignores LiDAR error,
  so it reads ~mm on good frames; it catches ill-conditioned point sets such as one wall's features in a
  line); on video frames the frame's own LiDAR agreeing with the matched points' depth at the solved pose
  (an independent check); at least two references solved on their own whose inliers the joint pose
  explains (or one with many inliers on a video frame that passed the LiDAR check), and references that
  don't explain it holding less than half as many inliers (repeated doors, tiles and radiators give
  confident wrong poses; passes whose training poses disagree give two answers); a
  plausible offset from the dataset pose (ARKit drift is bounded); and, per session, the correction
  (re-solved vs dataset pose) not jumping against both neighbours in time (drift is slow). A pose that
  moved far from the one its references were picked at is solved again with references picked at the
  new pose; the two must agree.
- Kept at the dataset pose (`kept_dataset_pose`) when that pose reprojects the inliers about as well as
  the solve (`DATASET_EXPLAINS`): the matches give no evidence for a move then (close or planar points
  trade a degree of rotation for a cm of shift), and the frame should score as it is.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from .geom import GL_CV
from .models.common import F64

VERSION = 1
METHOD = "sift-pnp"
FILE = "test_poses.json"
FEATURE_SIDE = 960  # features at this longest side (1920 px images: half size)
N_FEATURES = 4000
CONTRAST = 0.02  # SIFT contrast threshold: OpenCV's 0.04 finds little on plain walls
RATIO = 0.75
REF_DIST_M, REF_DEG = 1.5, 45.0  # reference candidates around the pose the references are picked at
GRID = (32, 24)  # LiDAR depth samples per frame for the co-visibility of test frame and candidates
VIS_M, VIS_REL = 0.5, 0.1  # a point further than this behind a frame's LiDAR surface is hidden from it
MIN_OVERLAP = 0.2  # co-visibility of a reference at least
MAX_REFS = 5
MIN_DEPTH, MAX_DEPTH = 0.2, 6.0  # m: LiDAR readings used
EDGE_M, EDGE_REL = 0.02, 0.03  # a depth neighbourhood spanning more than 2 cm + 3 % is a depth edge
PNP_PX = 3.0  # RANSAC inlier threshold at FEATURE_SIDE (LiDAR noise on the 3D points is a few px)
RANSAC_ITERS = 2000
REF_RANSAC_ITERS = 500  # each reference on its own
MIN_MATCHES = 40
MIN_INLIERS = 40
MIN_INLIER_SHARE = 0.25
MIN_QUADRANTS, QUADRANT_MIN = 3, 5  # inliers in at least 3 image quadrants (5 each)...
MIN_HULL = 0.20  # ...or a convex hull of at least this share of the image
MAX_REPROJ_PX = 2.0  # median, at FEATURE_SIDE (1/240 s frames of scan 4: 1.3 typical)
MAX_SIGMA_M, MAX_SIGMA_DEG = 0.02, 0.3  # 1-sigma camera position / rotation from the PnP Jacobian
MAX_DEPTH_RESID_M = 0.05  # median |depth at the solved pose - the frame's own LiDAR| over inliers
MIN_DEPTH_CHECK = 15  # inliers with own LiDAR needed for that check
REF_MIN_INLIERS = 25  # a reference solves on its own with at least this many inliers
REF_AGREE = 0.5  # a reference agrees when the joint pose explains this share of its own PnP inliers...
REF_DISAGREE = 0.2  # ...and disagrees, with MIN_INLIERS of its own, below this share
REF_OUTVOTED = 0.5  # rejected when disagreeing references have this share of the agreeing ones' inliers
MIN_REFS_AGREE = 2  # references confirming the pose, or one with SINGLE_REF_INLIERS and the own LiDAR check
SINGLE_REF_INLIERS = 120
MAX_SHIFT_M, MAX_ROT_DEG = 1.0, 5.0  # a larger offset from the dataset pose is a mismatch, not drift
DATASET_EXPLAINS = 1.25  # the dataset pose is kept when its median reprojection is within this x the solve's
RESELECT_M, RESELECT_DEG = 0.10, 2.0  # moved this far from where the references were picked: pick again
AGREE_M, AGREE_DEG = 0.03, 0.5  # ...and the two solves must agree this well
TEMPORAL_M, TEMPORAL_M_PER_S = 0.05, 0.02  # correction change allowed between neighbours in time
TEMPORAL_DEG, TEMPORAL_DEG_PER_S = 1.0, 0.2
TEMPORAL_GAP_S = 4.0  # neighbours further apart in time aren't compared
MOVED_M, MOVED_DEG = 0.10, 2.0  # the summary's "moved" (gs_trainer.evaluate's pose_moved uses the same)


@dataclass
class Cam:
    """A dataset frame: intrinsics at its image size, camera-to-world in OpenCV axes."""

    stem: str
    image: Path
    K: F64
    w: int
    h: int
    c2w: F64
    still: bool
    session: str
    t: float
    depth: Path | None
    confidence: Path | None


@dataclass
class Feats:
    uv: (
        np.ndarray
    )  # (N, 2) float64 at the feature size, pixel-corner convention (pixel i's centre at i + 0.5)
    desc: np.ndarray  # (N, 128) float32 RootSIFT
    K: F64  # intrinsics at the feature size
    size: tuple[int, int]  # (w, h) at the feature size
    xyz: F64 | None = None  # (N, 3) world points of keypoints with LiDAR depth (references), NaN without


def cam_of(entry: Mapping[str, Any], data: Path) -> Cam:
    """A transforms.json / split.json frame entry -> Cam."""
    stem = Path(str(entry["file_path"])).stem
    k = np.array([[entry["fl_x"], 0.0, entry["cx"]], [0.0, entry["fl_y"], entry["cy"]], [0.0, 0.0, 1.0]])
    depth, conf = entry.get("depth_file_path"), entry.get("confidence_file_path")
    return Cam(
        stem=stem,
        image=data / str(entry["file_path"]),
        K=k,
        w=int(entry["w"]),
        h=int(entry["h"]),
        c2w=np.asarray(entry["transform_matrix"], dtype=np.float64) @ GL_CV,
        still=entry.get("camera") == "still",
        session=str(entry.get("session") or stem.rsplit("_", 1)[0]),
        t=float(entry.get("t") or 0.0),
        depth=data / str(depth) if depth else None,
        confidence=data / str(conf) if conf else None,
    )


# --- geometry ----------------------------------------------------------------------------------------


def rot_deg(a: F64, b: F64) -> float:
    """Angle between two rotations (3x3), degrees."""
    c = (np.trace(a @ b.T) - 1) / 2
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


def pose_diff(a: F64, b: F64) -> tuple[float, float]:
    """(camera centre distance m, rotation deg) between two camera-to-world matrices."""
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), rot_deg(a[:3, :3], b[:3, :3])


def project(K: F64, w2c: F64, xyz: F64) -> tuple[F64, F64]:
    """(uv (N, 2), z (N,)) of world points in a camera (pixel-corner convention, as K)."""
    pc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    safe = np.where(np.abs(z) > 1e-9, z, 1e-9)
    uv = pc[:, :2] / safe[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def sample_depth(depth: F64, conf: np.ndarray | None, uv: F64, w: int, h: int) -> F64:
    """LiDAR depth (m) at image points `uv` of a w x h image (the depth map covers the same view):
    bilinear, 0 where a reading is missing, below confidence 2 or at a depth edge (the 4x4 around)."""
    dh, dw = depth.shape
    x = uv[:, 0] * dw / w - 0.5
    y = uv[:, 1] * dh / h - 0.5
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    inside = (x0 >= 1) & (y0 >= 1) & (x0 < dw - 2) & (y0 < dh - 2)
    out = np.zeros(len(uv))
    if not inside.any():
        return out
    xi, yi = x0[inside], y0[inside]
    fx, fy = x[inside] - xi, y[inside] - yi
    win = np.stack([depth[yi + a, xi + c] for a in (-1, 0, 1, 2) for c in (-1, 0, 1, 2)], axis=1)
    z = (
        depth[yi, xi] * (1 - fx) * (1 - fy)
        + depth[yi, xi + 1] * fx * (1 - fy)
        + depth[yi + 1, xi] * (1 - fx) * fy
        + depth[yi + 1, xi + 1] * fx * fy
    )
    ok = (win.min(axis=1) > MIN_DEPTH) & (np.ptp(win, axis=1) < EDGE_M + EDGE_REL * z) & (z < MAX_DEPTH)
    if conf is not None:
        quad = (conf[yi, xi], conf[yi, xi + 1], conf[yi + 1, xi], conf[yi + 1, xi + 1])
        ok &= np.minimum.reduce(quad) >= 2
    out[inside] = np.where(ok, z, 0.0)
    return out


def read_depth(cam: Cam) -> tuple[F64, np.ndarray | None] | None:
    """(depth m, confidence) of a video frame, None for stills."""
    if cam.depth is None:
        return None
    d = cv2.imread(str(cam.depth), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(cam.depth)
    c = cv2.imread(str(cam.confidence), cv2.IMREAD_UNCHANGED) if cam.confidence else None
    return d.astype(np.float64) / 1000.0, c


def lift(cam: Cam, uv: F64, z: F64) -> F64:
    """World points of image points `uv` (at cam's size) at camera depth `z`."""
    k = cam.K
    pc = np.column_stack([(uv[:, 0] - k[0, 2]) / k[0, 0] * z, (uv[:, 1] - k[1, 2]) / k[1, 1] * z, z])
    return pc @ cam.c2w[:3, :3].T + cam.c2w[:3, 3]


def depth_grid(cam: Cam) -> F64 | None:
    """The frame's LiDAR depth sampled on a GRID (gh, gw), 0 = no reading; None for stills."""
    got = read_depth(cam)
    if got is None:
        return None
    depth, conf = got
    gw, gh = GRID
    dh, dw = depth.shape
    di = np.clip(((np.arange(gh) + 0.5) * dh / gh).astype(int), 0, dh - 1)
    dj = np.clip(((np.arange(gw) + 0.5) * dw / gw).astype(int), 0, dw - 1)
    z = depth[np.ix_(di, dj)]
    ok = (z > MIN_DEPTH) & (z < MAX_DEPTH)
    if conf is not None:
        ok &= conf[np.ix_(di, dj)] >= 1
    return np.where(ok, z, 0.0)


def grid_world(cam: Cam, z: F64, pose: F64) -> F64:
    """World points (M, 3) of a depth grid seen from `pose` (the readings only)."""
    gh, gw = z.shape
    u = (np.arange(gw) + 0.5) * cam.w / gw
    v = (np.arange(gh) + 0.5) * cam.h / gh
    uv = np.stack(np.meshgrid(u, v), axis=-1).reshape(-1, 2)
    zz = z.ravel()
    ok = zz > 0
    k = cam.K
    pc = np.column_stack(
        [(uv[ok, 0] - k[0, 2]) / k[0, 0] * zz[ok], (uv[ok, 1] - k[1, 2]) / k[1, 1] * zz[ok], zz[ok]]
    )
    return pc @ pose[:3, :3].T + pose[:3, 3]


def sees(cam: Cam, pose: F64, z: F64 | None, pts: F64) -> np.ndarray:
    """Which world points `pts` the camera at `pose` sees: in front, inside the image, and not behind
    its own LiDAR surface (`z`, a depth grid; by `VIS_M` + `VIS_REL` of the depth, which absorbs the pose
    error the test frames may have). (M,) bool."""
    uv, d = project(cam.K, np.linalg.inv(pose), pts)
    ok = (d > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.w) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.h)
    if z is not None and ok.any():
        gh, gw = z.shape
        gi = np.clip((uv[ok, 1] * gh / cam.h).astype(int), 0, gh - 1)
        gj = np.clip((uv[ok, 0] * gw / cam.w).astype(int), 0, gw - 1)
        surf = z[gi, gj]
        ok[ok] = (surf == 0) | (d[ok] <= surf + VIS_M + VIS_REL * surf)
    return ok


def covisibility(test: Cam, pose: F64, test_z: F64 | None, ref: Cam, ref_z: F64 | None) -> float:
    """How much a reference and the test frame (at `pose`) see of each other: the share of the
    reference's LiDAR points the test camera sees and, for a test frame with LiDAR, the share of its own
    points the reference sees; the smaller of the two."""
    if ref_z is None:
        return 0.0
    ref_pts = grid_world(ref, ref_z, ref.c2w)
    if not len(ref_pts):
        return 0.0
    score = float(sees(test, pose, test_z, ref_pts).mean())
    if test_z is not None:
        test_pts = grid_world(test, test_z, pose)
        if len(test_pts):
            score = min(score, float(sees(ref, ref.c2w, ref_z, test_pts).mean()))
    return score


def pick_refs(
    test: Cam,
    pose: F64,
    test_z: F64 | None,
    train: Sequence[Cam],
    grids: Sequence[F64 | None],
    max_refs: int = MAX_REFS,
) -> list[int]:
    """Indices into `train` of the references for `test` seen from `pose` (see the module docstring)."""
    if not train:
        return []
    centres = np.array([c.c2w[:3, 3] for c in train])
    fwd = np.array([c.c2w[:3, 2] for c in train])
    dist = np.linalg.norm(centres - pose[:3, 3], axis=1)
    ang = np.degrees(np.arccos(np.clip(fwd @ pose[:3, 2], -1.0, 1.0)))
    cand = np.flatnonzero((dist < REF_DIST_M) & (ang < REF_DEG))
    scored = [
        (covisibility(test, pose, test_z, train[j], grids[j]), -float(dist[j]), j) for j in cand.tolist()
    ]
    return [j for o, _, j in sorted(scored, reverse=True)[:max_refs] if o >= MIN_OVERLAP]


# --- features ----------------------------------------------------------------------------------------


def _rootsift(desc: np.ndarray) -> np.ndarray:
    d = desc.astype(np.float32)
    d /= np.maximum(d.sum(axis=1, keepdims=True), 1e-6)
    return np.sqrt(d)


def features(cam: Cam, with_xyz: bool = False) -> Feats:
    """RootSIFT of the frame at FEATURE_SIDE; with `with_xyz`, the keypoints' world points (LiDAR)."""
    img = cv2.imread(str(cam.image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(cam.image)
    s = min(1.0, FEATURE_SIDE / max(cam.w, cam.h))
    fw, fh = round(cam.w * s), round(cam.h * s)
    gray = (
        img
        if (img.shape[1], img.shape[0]) == (fw, fh)
        else cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)
    )
    sift = cv2.SIFT.create(nfeatures=N_FEATURES, contrastThreshold=CONTRAST)
    kps, desc = cast(tuple[Sequence[cv2.KeyPoint], np.ndarray | None], sift.detectAndCompute(gray, None))
    K = np.diag([fw / cam.w, fh / cam.h, 1.0]) @ cam.K
    if desc is None or not len(kps):
        return Feats(np.zeros((0, 2)), np.zeros((0, 128), np.float32), K, (fw, fh), np.zeros((0, 3)))
    uv = np.array([k.pt for k in kps], np.float64) + 0.5  # pixel-corner convention, like the intrinsics
    f = Feats(uv, _rootsift(desc), K, (fw, fh))
    if with_xyz:
        f.xyz = np.full((len(uv), 3), np.nan)
        got = read_depth(cam)
        if got is not None:
            uv_full = uv * [cam.w / fw, cam.h / fh]
            z = sample_depth(got[0], got[1], uv_full, cam.w, cam.h)
            ok = z > 0
            f.xyz[ok] = lift(cam, uv_full[ok], z[ok])
    return f


def mutual_matches(
    a: np.ndarray, b: np.ndarray, ratio: float = RATIO
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(index in a, index in b, descriptor distance) of ratio-test, mutual nearest-neighbour matches."""
    if len(a) < 2 or len(b) < 2:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(a, b, k=2)
    best_a = np.full(len(b), -1)
    for m in bf.match(b, a):
        best_a[m.queryIdx] = m.trainIdx
    ia: list[int] = []
    ib: list[int] = []
    dist: list[float] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance and best_a[m.trainIdx] == m.queryIdx:
            ia.append(m.queryIdx)
            ib.append(m.trainIdx)
            dist.append(m.distance)
    return np.array(ia, int), np.array(ib, int), np.array(dist)


# --- solve -------------------------------------------------------------------------------------------


@dataclass
class Solve:
    c2w: F64 | None  # OpenCV axes; None when rejected
    reason: str = ""  # why it was rejected ("" when accepted)
    matches: int = 0
    inliers: int = 0
    reproj_px: float | None = None  # median, at the feature size
    sigma_m: float | None = None
    sigma_deg: float | None = None
    quadrants: int | None = None
    hull: float | None = None
    depth_resid_m: float | None = None
    reproj_dataset_px: float | None = None  # the inliers' median reprojection error at the dataset pose
    refs_agree: list[str] = field(default_factory=list[str])
    refs_disagree: list[str] = field(default_factory=list[str])
    ref_checks: list[list[Any]] = field(
        default_factory=list[list[Any]]
    )  # [stem, own inliers, share explained]
    rejected: F64 | None = None  # the pose a rejected solve found


def _c2w(rvec: np.ndarray, tvec: np.ndarray) -> F64:
    w2c = np.eye(4)
    w2c[:3, :3] = cv2.Rodrigues(rvec)[0]
    w2c[:3, 3] = np.ravel(tvec)
    return np.linalg.inv(w2c)


def pose_sigma(xyz: F64, uv: F64, K: F64, rvec: np.ndarray, tvec: np.ndarray) -> tuple[float, float]:
    """1-sigma (camera position m, rotation deg) of a PnP solution: residual variance x (J^T J)^-1 in
    (rvec, tvec), the position propagated through C = -R^T t. Large on ill-conditioned point sets."""
    proj, jac = cv2.projectPoints(xyz, rvec, tvec, K, None)
    r = (proj.reshape(-1, 2) - uv).ravel()
    J = np.asarray(jac)[:, :6]
    try:
        cov = float(r @ r) / max(len(r) - 6, 1) * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return math.inf, math.inf

    def centre(p: F64) -> F64:
        return -cv2.Rodrigues(p[:3])[0].T @ p[3:]

    p0 = np.concatenate([np.ravel(rvec), np.ravel(tvec)]).astype(np.float64)
    jc = np.zeros((3, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = 1e-6
        jc[:, i] = (centre(p0 + d) - centre(p0 - d)) / 2e-6
    var_c = float(np.trace(jc @ cov @ jc.T))
    return math.sqrt(max(var_c, 0.0)), math.degrees(math.sqrt(max(float(np.trace(cov[:3, :3])), 0.0)))


def pnp(
    xyz: F64, uv: F64, K: F64, iterations: int = RANSAC_ITERS
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(rvec, tvec, inlier indices) of PnP RANSAC + LM on the inliers, inliers counted again at the
    refined pose (in front of the camera, within PNP_PX), refined once more; None without a pose."""
    if len(xyz) < 6:
        return None
    try:
        found = cv2.solvePnPRansac(
            xyz, uv, K, None, iterationsCount=iterations, reprojectionError=PNP_PX, confidence=0.9999,
            flags=cv2.SOLVEPNP_SQPNP,
        )  # fmt: skip
    except cv2.error:
        return None
    ok, rvec, tvec, inl = cast(tuple[bool, np.ndarray, np.ndarray, np.ndarray | None], found)
    if not ok or inl is None or len(inl) < 6:
        return None
    idx = inl.ravel()
    for _ in range(2):
        rvec, tvec = cv2.solvePnPRefineLM(xyz[idx], uv[idx], K, None, rvec, tvec)
        proj, z = project(K, np.linalg.inv(_c2w(rvec, tvec)), xyz)
        idx = np.flatnonzero((np.linalg.norm(proj - uv, axis=1) < PNP_PX) & (z > 0.05))
        if len(idx) < 6:
            return None
    return rvec, tvec, idx


def _quadrants(uv: F64, w: int, h: int) -> int:
    q = (uv[:, 0] >= w / 2).astype(int) + 2 * (uv[:, 1] >= h / 2).astype(int)
    return int((np.bincount(q, minlength=4) >= QUADRANT_MIN).sum())


def solve(
    test: Cam,
    qf: Feats,
    refs: Sequence[tuple[str, Feats]],
    own_depth: tuple[F64, np.ndarray | None] | None,
) -> Solve:
    """The test frame's pose from its matches to `refs` ((stem, features with world points))."""
    best: dict[int, tuple[float, F64]] = {}  # test keypoint -> (distance, world point)
    per_ref: list[tuple[str, np.ndarray, F64]] = []  # (stem, test keypoints, world points)
    for stem, rf in refs:
        assert rf.xyz is not None
        has = np.flatnonzero(np.isfinite(rf.xyz[:, 0]))
        ia, ib, dist = mutual_matches(qf.desc, rf.desc[has])
        pts = rf.xyz[has[ib]]
        per_ref.append((stem, ia, pts))
        for a, p, d in zip(ia.tolist(), pts, dist.tolist(), strict=True):
            if a not in best or d < best[a][0]:
                best[a] = (d, p)
    n = len(best)
    if n < MIN_MATCHES:
        return Solve(None, f"{n} matches < {MIN_MATCHES}", n)
    keys = sorted(best)
    uv = qf.uv[keys]
    xyz = np.array([best[k][1] for k in keys])
    K = qf.K
    got = pnp(xyz, uv, K)
    if got is None:
        return Solve(None, "PnP found no pose", n)
    rvec, tvec, idx = got
    c2w = _c2w(rvec, tvec)
    res = Solve(c2w, "", n, len(idx))
    proj, z = project(K, np.linalg.inv(c2w), xyz)
    res.reproj_px = float(np.median(np.linalg.norm(proj[idx] - uv[idx], axis=1)))
    fw, fh = qf.size
    res.quadrants = _quadrants(uv[idx], fw, fh)
    res.hull = float(cv2.contourArea(cv2.convexHull(uv[idx].astype(np.float32)))) / (fw * fh)
    res.sigma_m, res.sigma_deg = pose_sigma(xyz[idx], uv[idx], K, rvec, tvec)
    pd, _ = project(K, np.linalg.inv(test.c2w), xyz[idx])
    res.reproj_dataset_px = float(np.median(np.linalg.norm(pd - uv[idx], axis=1)))
    if own_depth is not None:
        own = sample_depth(own_depth[0], own_depth[1], uv[idx] * [test.w / fw, test.h / fh], test.w, test.h)
        has = own > 0
        if has.sum() >= MIN_DEPTH_CHECK:
            res.depth_resid_m = float(np.median(np.abs(z[idx][has] - own[has])))
    # Each reference on its own: its PnP inliers must be explained by the joint pose (compared in the
    # image, not as poses: one frame's points often fix a pose poorly, e.g. a degree of rotation traded
    # for a few cm of shift, while both explain the matches).
    agree_n = disagree_n = 0
    w2c = np.linalg.inv(c2w)
    for stem, ia, pts in per_ref:
        own_fit = pnp(pts, qf.uv[ia], K, iterations=REF_RANSAC_ITERS) if len(ia) >= REF_MIN_INLIERS else None
        if own_fit is None or len(own_fit[2]) < REF_MIN_INLIERS:
            continue
        own = own_fit[2]
        pj, zj = project(K, w2c, pts[own])
        share = float(np.mean((np.linalg.norm(pj - qf.uv[ia][own], axis=1) < PNP_PX) & (zj > 0.05)))
        res.ref_checks.append([stem, len(own), round(share, 3)])
        if share >= REF_AGREE:
            res.refs_agree.append(stem)
            agree_n += len(own)
        elif share < REF_DISAGREE and len(own) >= MIN_INLIERS:
            res.refs_disagree.append(stem)
            disagree_n += len(own)
    # One reference alone confirms a pose when it has many inliers and the frame's own LiDAR agrees.
    confirmed = len(res.refs_agree) >= MIN_REFS_AGREE or (
        len(res.refs_agree) == 1 and agree_n >= SINGLE_REF_INLIERS and res.depth_resid_m is not None
    )
    shift, rot = pose_diff(c2w, test.c2w)
    if res.inliers < MIN_INLIERS:
        res.reason = f"{res.inliers} inliers < {MIN_INLIERS}"
    elif res.inliers < MIN_INLIER_SHARE * n:
        res.reason = f"{res.inliers} inliers of {n} matches < {MIN_INLIER_SHARE:.0%}"
    elif res.quadrants < MIN_QUADRANTS and res.hull < MIN_HULL:
        res.reason = f"inliers in {res.quadrants} quadrants, hull {res.hull:.0%} of the image"
    elif res.reproj_px > MAX_REPROJ_PX:
        res.reason = f"median reprojection {res.reproj_px:.2f} px > {MAX_REPROJ_PX:g}"
    elif res.sigma_m > MAX_SIGMA_M or res.sigma_deg > MAX_SIGMA_DEG:
        res.reason = f"uncertain: sigma {100 * res.sigma_m:.1f} cm / {res.sigma_deg:.2f} deg"
    elif res.depth_resid_m is not None and res.depth_resid_m > MAX_DEPTH_RESID_M:
        res.reason = f"own LiDAR disagrees by {100 * res.depth_resid_m:.1f} cm"
    elif disagree_n > REF_OUTVOTED * agree_n:
        res.reason = (
            f"references disagree: {len(res.refs_disagree)} ({disagree_n} inliers) against"
            f" {len(res.refs_agree)} agreeing ({agree_n})"
        )
    elif not confirmed:
        res.reason = f"{len(res.refs_agree)} references confirm the pose on their own < {MIN_REFS_AGREE}"
    elif shift > MAX_SHIFT_M or rot > MAX_ROT_DEG:
        res.reason = f"implausible: {100 * shift:.0f} cm / {rot:.1f} deg from the dataset pose"
    if res.reason:
        res.c2w, res.rejected = None, c2w
    return res


# --- dataset -----------------------------------------------------------------------------------------


class _FeatureCache:
    """Reference features by training index, computed once (thread-safe: a lock per frame)."""

    def __init__(self, train: Sequence[Cam]) -> None:
        self.train = train
        self.feats: dict[int, Feats] = {}
        self.locks: dict[int, threading.Lock] = {}
        self.lock = threading.Lock()

    def get(self, j: int) -> tuple[str, Feats]:
        with self.lock:
            lk = self.locks.setdefault(j, threading.Lock())
        with lk:
            if j not in self.feats:
                self.feats[j] = features(self.train[j], with_xyz=True)
            return self.train[j].stem, self.feats[j]


def _r(x: float | None, nd: int, scale: float = 1.0) -> float | None:
    return None if x is None or not math.isfinite(x) else round(x * scale, nd)


def resolve_frame(
    test: Cam, train: Sequence[Cam], grids: Sequence[F64 | None], cache: _FeatureCache
) -> dict[str, Any]:
    """The `frames` entry of one test frame (see trainer/README.md); `pose` is kept for the temporal
    check and dropped before writing."""
    qf = features(test)
    own = None if test.still else read_depth(test)
    test_z = None if test.still else depth_grid(test)
    pose = test.c2w
    first: Solve | None = None
    refs: list[int] = []
    s = Solve(None, "no training frame sees this view")
    for _ in range(2):
        refs = pick_refs(test, pose, test_z, train, grids)
        if not refs:
            s = Solve(None, "no training frame sees this view")
            break
        s = solve(test, qf, [cache.get(j) for j in refs], own)
        if s.c2w is None:
            break
        if first is not None:
            assert first.c2w is not None
            dm, dd = pose_diff(s.c2w, first.c2w)
            if dm > AGREE_M or dd > AGREE_DEG:
                s.reason = f"references picked at the solved pose disagree: {100 * dm:.1f} cm / {dd:.2f} deg"
                s.c2w, s.rejected = None, s.c2w
            break
        moved_m, moved_deg = pose_diff(s.c2w, pose)
        if moved_m <= RESELECT_M and moved_deg <= RESELECT_DEG:
            break
        first, pose = s, s.c2w
    px = test.w / qf.size[0]  # feature-size pixels -> image pixels
    row: dict[str, Any] = {
        "status": "ok" if s.c2w is not None else "unresolved",
        "refs": [train[j].stem for j in refs],
        "matches": s.matches,
        "inliers": s.inliers,
        "reproj_px": _r(s.reproj_px, 3, px),
        "reproj_dataset_px": _r(s.reproj_dataset_px, 2, px),
        "sigma_cm": _r(s.sigma_m, 3, 100),
        "sigma_deg": _r(s.sigma_deg, 4),
        "quadrants": s.quadrants,
        "hull": _r(s.hull, 3),
        "depth_resid_cm": _r(s.depth_resid_m, 2, 100),
        "refs_agree": len(s.refs_agree),
        "refs_disagree": len(s.refs_disagree),
        "ref_checks": s.ref_checks,
    }
    if s.c2w is None:
        row["reason"] = s.reason
        if s.rejected is not None:
            row["rejected_offset_cm"] = [
                round(100 * float(v), 2) for v in s.rejected[:3, 3] - test.c2w[:3, 3]
            ]
        return row
    # The matches don't reject the dataset pose (it reprojects them about as well): no evidence for a move
    # within what one frame's matches can tell apart (close or planar points trade rotation for shift).
    pose = s.c2w
    assert s.reproj_px is not None and s.reproj_dataset_px is not None
    if s.reproj_dataset_px <= DATASET_EXPLAINS * s.reproj_px:
        shift, rot = pose_diff(s.c2w, test.c2w)
        row |= {
            "kept_dataset_pose": True,
            "solved_shift_cm": round(100 * shift, 2),
            "solved_rot_deg": round(rot, 3),
        }
        pose = test.c2w
    shift, rot = pose_diff(pose, test.c2w)
    gl = pose @ GL_CV
    gl[3] = [0.0, 0.0, 0.0, 1.0]
    row |= {
        "transform_matrix": [[round(float(v), 7) for v in r] for r in gl],
        "shift_cm": round(100 * shift, 2),
        "rot_deg": round(rot, 3),
        "offset_cm": [round(100 * float(v), 2) for v in pose[:3, 3] - test.c2w[:3, 3]],
        "pose": pose,
    }
    return row


def temporal_outliers(cams: Sequence[Cam], rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    """Indices of re-solved frames whose correction (re-solved vs dataset pose) jumps against both
    neighbours in time within their session while those two agree, with the reason."""
    out: dict[int, str] = {}
    by_session: dict[str, list[int]] = {}
    for i, (c, r) in enumerate(zip(cams, rows, strict=True)):
        if r["status"] == "ok":
            by_session.setdefault(c.session, []).append(i)

    def jump(i: int, j: int) -> tuple[bool, float, float]:
        dt = abs(cams[i].t - cams[j].t)
        dm = float(np.linalg.norm(np.subtract(rows[i]["offset_cm"], rows[j]["offset_cm"]))) / 100
        ci = rows[i]["pose"][:3, :3] @ cams[i].c2w[:3, :3].T
        cj = rows[j]["pose"][:3, :3] @ cams[j].c2w[:3, :3].T
        dd = rot_deg(ci, cj)
        return dm > TEMPORAL_M + TEMPORAL_M_PER_S * dt or dd > TEMPORAL_DEG + TEMPORAL_DEG_PER_S * dt, dm, dd

    for idx in by_session.values():
        idx = sorted(idx, key=lambda i: cams[i].t)
        for a, i, b in zip(idx, idx[1:], idx[2:], strict=False):
            if cams[b].t - cams[a].t > 2 * TEMPORAL_GAP_S:
                continue
            ja, dm, dd = jump(i, a)
            jb, _, _ = jump(i, b)
            if ja and jb and not jump(a, b)[0]:
                out[i] = f"temporal: correction jumps {100 * dm:.0f} cm / {dd:.2f} deg against its neighbours"
    return out


def params() -> dict[str, Any]:
    return {
        "feature_side": FEATURE_SIDE,
        "n_features": N_FEATURES,
        "ratio": RATIO,
        "ref_dist_m": REF_DIST_M,
        "ref_deg": REF_DEG,
        "min_overlap": MIN_OVERLAP,
        "max_refs": MAX_REFS,
        "pnp_px": PNP_PX,
        "min_inliers": MIN_INLIERS,
        "min_inlier_share": MIN_INLIER_SHARE,
        "min_quadrants": MIN_QUADRANTS,
        "min_hull": MIN_HULL,
        "max_reproj_px": MAX_REPROJ_PX,
        "max_sigma_cm": 100 * MAX_SIGMA_M,
        "max_sigma_deg": MAX_SIGMA_DEG,
        "max_depth_resid_cm": 100 * MAX_DEPTH_RESID_M,
        "ref_agree": REF_AGREE,
        "ref_disagree": REF_DISAGREE,
        "ref_outvoted": REF_OUTVOTED,
        "min_refs_agree": MIN_REFS_AGREE,
        "single_ref_inliers": SINGLE_REF_INLIERS,
        "max_shift_cm": 100 * MAX_SHIFT_M,
        "max_rot_deg": MAX_ROT_DEG,
        "dataset_explains": DATASET_EXPLAINS,
        "moved_cm": 100 * MOVED_M,
        "moved_deg": MOVED_DEG,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(
    data: Path,
    train_frames: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    workers: int | None = None,
) -> dict[str, Any]:
    """The `test_poses.json` content (without `source`) for a dataset whose images and depth are in
    `data`: `train_frames` are transforms.json's frames, `split` split.json's content."""
    started = time.monotonic()
    interp = set(split.get("interp", []))

    def stem(e: Mapping[str, Any]) -> str:
        return Path(str(e["file_path"])).stem

    train = [
        cam_of(e, data)
        for e in train_frames
        if stem(e) not in interp and e.get("camera") != "still" and e.get("depth_file_path")
    ]
    tests = [("interp", cam_of(e, data)) for e in train_frames if stem(e) in interp]
    tests += [("extrap", cam_of(e, data)) for e in split.get("extrap_frames", [])]
    cams = [c for _, c in tests]
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        grids = list(pool.map(depth_grid, train))
        cache = _FeatureCache(train)
        # The references at the dataset poses first, all in parallel (test frames next to each other
        # share references: computed lazily, the workers would queue on the same few).
        first = {
            j for c in cams for j in pick_refs(c, c.c2w, None if c.still else depth_grid(c), train, grids)
        }
        list(pool.map(cache.get, sorted(first)))
        rows = list(pool.map(lambda c: resolve_frame(c, train, grids, cache), cams))
    for i, reason in temporal_outliers(cams, rows).items():
        r = rows[i]
        for k in ("transform_matrix", "shift_cm", "rot_deg", "offset_cm"):
            r.pop(k)
        r |= {"status": "unresolved", "reason": reason}
    for r in rows:
        r.pop("pose", None)
    summary: dict[str, Any] = {}
    for name in ("interp", "extrap"):
        rs = [row for (s, _), row in zip(tests, rows, strict=True) if s == name]
        ok = [r for r in rs if r["status"] == "ok"]
        moved = [r for r in ok if r["shift_cm"] > 100 * MOVED_M or r["rot_deg"] > MOVED_DEG]
        summary[name] = {
            "n": len(rs),
            "resolved": len(ok),
            "unresolved": len(rs) - len(ok),
            "moved": len(moved),
            "shift_cm_median": round(float(np.median([r["shift_cm"] for r in ok])), 2) if ok else None,
            "shift_cm_max": round(max(r["shift_cm"] for r in ok), 2) if ok else None,
        }
    wall = time.monotonic() - started
    line = "; ".join(
        f"{k} {v['resolved']}/{v['n']} re-solved, {v['moved']} moved > {100 * MOVED_M:g} cm or {MOVED_DEG:g}°"
        for k, v in summary.items()
    )
    print(f"  test poses: {line} ({len(cache.feats)} reference frames, {wall:.0f} s)", flush=True)
    return {
        "version": VERSION,
        "method": METHOD,
        "params": params(),
        "summary": summary,
        "wall_s": round(wall, 1),
        "frames": {c.stem: row for c, row in zip(cams, rows, strict=True)},
    }


def write(data: Path, out: Path | None = None, workers: int | None = None) -> dict[str, Any]:
    """Re-solve a written dataset's test poses into `out` (default: `data`/test_poses.json)."""
    tf_path = data / "transforms.json"
    tf = json.loads(tf_path.read_text())
    split_path = data / tf.get("split_file_path", "split.json")
    split = json.loads(split_path.read_text()) if split_path.is_file() else {}
    result = resolve(data, tf["frames"], split, workers)
    source = {
        "transforms_sha256": sha256(tf_path),
        "split_sha256": sha256(split_path) if split_path.is_file() else None,
    }
    result = {"version": result.pop("version"), "method": result.pop("method"), "source": source, **result}
    target = out or data / FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scanmodel.testposes", description=write.__doc__)
    p.add_argument("data", type=Path, help="splat dataset (transforms.json, split.json, images/, depth/)")
    p.add_argument("--out", type=Path, default=None, help=f"file to write (default: the dataset's {FILE})")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args(sys.argv[1:] if argv is None else argv)
    write(a.data, a.out, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
