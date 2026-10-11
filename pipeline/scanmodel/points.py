"""Point clouds from depth maps and Open3D glue."""

from pathlib import Path

import numpy as np
import open3d as o3d

from .geom import apply, backproject
from .io import read_confidence, read_depth
from .models.common import F64, mat
from .models.derived import PosedImage

WALLS_DIR = "walls"  # derived/poses/walls/<session>@<s>.npz: wall_points() per window, scene frame
WINDOW_S = 30.0  # both faces of a wall within this long: ARKit drifts mm, not cm (windows overlap by half)


def sample_points(
    session_dir: Path,
    images: list[PosedImage],
    n_frames: int = 40,
    stride: int = 4,
    min_confidence: int = 1,
    max_depth: float = 4.0,
) -> F64:
    """World points from a spread of depth frames (cheap stand-in for a full fusion)."""
    with_depth = [im for im in images if im.depth]
    if not with_depth:
        return np.zeros((0, 3))
    pick = np.linspace(0, len(with_depth) - 1, min(n_frames, len(with_depth))).round().astype(int)
    out: list[F64] = []
    for i in pick:
        im = with_depth[int(i)]
        assert im.depth is not None
        d = read_depth(session_dir / im.depth)
        if im.confidence and min_confidence > 0:
            d[read_confidence(session_dir / im.confidence) < min_confidence] = 0.0
        d[d > max_depth] = 0.0
        out.append(apply(mat(im.camera_to_world), backproject(d, im.intrinsics, stride)))
    return np.concatenate(out)


def thin_frames(
    images: list[PosedImage], min_move: float = 0.05, min_turn_deg: float = 5.0
) -> list[PosedImage]:
    """Drop frames that barely moved or turned since the last kept one."""
    kept: list[PosedImage] = []
    last: F64 | None = None
    for im in images:
        m = mat(im.camera_to_world)
        if last is not None:
            move = float(np.linalg.norm(m[:3, 3] - last[:3, 3]))
            cos = (np.trace(last[:3, :3].T @ m[:3, :3]) - 1) / 2
            turn = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            if move < min_move and turn < min_turn_deg:
                continue
        kept.append(im)
        last = m
    return kept


def wall_points(
    session_dir: Path,
    images: list[PosedImage],
    min_confidence: int = 2,
    max_depth: float = 3.5,
    voxel: float = 0.015,
) -> tuple[F64, F64]:
    """Points on near-vertical surfaces from one session, normals facing the camera that saw them.

    One session's own depth only: within a session ARKit drifts little, so two faces of a wall that it
    saw keep their true distance. (Faces from two sessions also carry the drift between them.)
    """
    pts: list[F64] = []
    nrm: list[F64] = []
    for im in thin_frames([im for im in images if im.depth]):
        assert im.depth is not None
        d = read_depth(session_dir / im.depth)
        if im.confidence and min_confidence > 0:
            d[read_confidence(session_dir / im.confidence) < min_confidence] = 0.0
        d[d > max_depth] = 0.0
        h, w = d.shape
        k = im.intrinsics.scaled(w, h)
        v, u = np.mgrid[0:h, 0:w]
        p = np.stack([(u + 0.5 - k.cx) / k.fx * d, (v + 0.5 - k.cy) / k.fy * d, d], axis=-1)
        dx = p[1:-1, 2:] - p[1:-1, :-2]
        dy = p[2:, 1:-1] - p[:-2, 1:-1]
        c = p[1:-1, 1:-1]
        z = d[1:-1, 1:-1]
        ok = (z > 0.2) & (d[1:-1, 2:] > 0) & (d[1:-1, :-2] > 0) & (d[2:, 1:-1] > 0) & (d[:-2, 1:-1] > 0)
        ok &= (np.linalg.norm(dx, axis=-1) < 0.15 * z) & (
            np.linalg.norm(dy, axis=-1) < 0.15 * z
        )  # no depth edges
        n = np.cross(dx[ok], dy[ok])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        c = c[ok]
        n[np.einsum("ij,ij->i", n, c) > 0] *= -1  # towards the camera at the origin
        m = mat(im.camera_to_world)
        cw, nw = apply(m, c), n @ m[:3, :3].T
        keep = np.abs(nw[:, 2]) < 0.3
        pts.append(cw[keep][::2])
        nrm.append(nw[keep][::2])
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    pc = to_o3d(np.concatenate(pts), np.concatenate(nrm)).voxel_down_sample(voxel)
    n = np.asarray(pc.normals)
    return np.asarray(pc.points), n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def write_wall_points(path: Path, pts: F64, normals: F64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, points=pts.astype(np.float32), normals=normals.astype(np.float32))


def read_wall_points(path: Path) -> tuple[F64, F64]:
    with np.load(path) as z:
        return z["points"].astype(np.float64), z["normals"].astype(np.float64)


def to_o3d(pts: F64, normals: F64 | None = None) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=np.float64)))
    if normals is not None:
        pc.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals, dtype=np.float64))
    return pc


def with_normals(pts: F64, voxel: float = 0.02) -> tuple[F64, F64]:
    pc = to_o3d(pts).voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return np.asarray(pc.points), np.asarray(pc.normals)


def read_points(path: Path) -> tuple[F64, F64]:
    pc = o3d.io.read_point_cloud(path)
    if not pc.has_normals():
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    return np.asarray(pc.points), np.asarray(pc.normals)


def floor_height(pts: F64, normals: F64, min_share: float = 0.2) -> float | None:
    """Lowest strong cluster of upward-facing points (1 cm bins)."""
    up = pts[np.abs(normals[:, 2]) > 0.9, 2]
    if up.size < 100:
        return None
    lo, hi = float(up.min()), float(up.max())
    hist, edges = np.histogram(up, bins=max(1, int((hi - lo) / 0.01)), range=(lo, hi + 1e-6))
    strong = np.flatnonzero(hist >= min_share * hist.max())
    i = int(strong[0])
    sel = up[(up >= edges[i] - 0.01) & (up <= edges[i + 1] + 0.01)]
    return float(np.median(sel))


def icp_check(src: F64, dst: F64, init: F64, max_dist: float = 0.05) -> tuple[float, F64, float]:
    """Point-to-plane ICP of src onto dst. Returns (inlier rmse, correction, overlap fitness)."""
    s = to_o3d(src).voxel_down_sample(0.02)
    t = to_o3d(dst).voxel_down_sample(0.02)
    t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    reg = o3d.pipelines.registration.registration_icp(
        s, t, max_dist, init, o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return float(reg.inlier_rmse), np.asarray(reg.transformation) @ np.linalg.inv(init), float(reg.fitness)


def global_register(src: F64, dst: F64, voxel: float = 0.05) -> F64:
    """FPFH + RANSAC, refined by ICP: aligns a session that was captured without the shared world map."""
    reg = o3d.pipelines.registration

    def prep(p: F64) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        pc = to_o3d(p).voxel_down_sample(voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
        search = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100)
        f = reg.compute_fpfh_feature(pc, search, list(range(len(pc.points))))
        return pc, f

    s, fs = prep(src)
    t, ft = prep(dst)
    coarse = reg.registration_ransac_based_on_feature_matching(
        s,
        t,
        fs,
        ft,
        True,
        voxel * 1.5,
        reg.TransformationEstimationPointToPoint(False),
        3,
        [
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
        ],
        reg.RANSACConvergenceCriteria(200000, 0.999),
    )
    _, correction, _ = icp_check(src, dst, np.asarray(coarse.transformation))
    return correction @ np.asarray(coarse.transformation)
