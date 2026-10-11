"""How well a fused mesh agrees with each frame's own LiDAR depth at that frame's pose.

The splat trainer's depth and normal priors are the mesh rendered at the dataset's camera poses
(`gs_trainer.priors`), and its depth loss uses each frame's LiDAR. Both only agree when the mesh was fused
from the same poses: a mesh fused from the pose graph's poses, seen from bundle-adjusted (`scanmodel refine`)
poses, sits a few centimetres off the frame's own LiDAR wherever the two disagree.

Per sampled depth frame (confidence >= `min_confidence`, LiDAR within `max_depth`), the mesh is raycast at
the LiDAR's 256x192 grid with the frame's pose: z-depth residual (mesh - LiDAR) and the angle between the
mesh normal and the LiDAR's own normal (from its back-projected neighbours). Pooled per pixel, by distance
band, and per frame (the median of each frame's |residual|).
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c

from .config import from_repo, rel
from .io import read_confidence, read_depth, read_jsonl
from .models.common import mat
from .models.derived import PosedImage, Poses

BANDS = ((0.0, 1.5), (1.5, 2.5), (2.5, 4.0))
NORMAL_MAX_STEP = 0.05  # m: LiDAR neighbours further apart than this (an edge) give no normal


def scene_of(mesh_path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.t.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.t.geometry.TriangleMesh(mesh.vertex.positions.to(o3c.float32), mesh.triangle.indices)
    )
    return scene


def lidar_normals(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, 3) camera-frame normals from back-projected LiDAR (central differences), facing the camera,
    and where they are defined (all four neighbours valid, steps under NORMAL_MAX_STEP)."""
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = (
        valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
        & (np.linalg.norm(dx, axis=-1) < 2 * NORMAL_MAX_STEP)
        & (np.linalg.norm(dy, axis=-1) < 2 * NORMAL_MAX_STEP)
        & (norm[..., 0] > 1e-9)
    )  # fmt: skip
    n = n / np.maximum(norm, 1e-12)
    n = np.where((np.sum(n * points[1:-1, 1:-1], axis=-1) > 0)[..., None], -n, n)  # face the camera
    out = np.zeros_like(points)
    out[1:-1, 1:-1] = n
    defined = np.zeros(valid.shape, bool)
    defined[1:-1, 1:-1] = ok
    return out, defined


def frame_residuals(
    scene: o3d.t.geometry.RaycastingScene,
    im: PosedImage,
    depth: np.ndarray,
    conf: np.ndarray | None,
    min_confidence: int = 2,
    max_depth: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(LiDAR depth, mesh - LiDAR z residual, normal angle in degrees or NaN) at the frame's valid pixels."""
    h, w = depth.shape
    k = im.intrinsics.scaled(w, h)
    v, u = np.mgrid[0:h, 0:w] + 0.5  # pixel centres at +0.5 (the pipeline's convention)
    rays_c = np.stack([(u - k.cx) / k.fx, (v - k.cy) / k.fy, np.ones_like(u)], -1)  # z = 1: t_hit = z-depth
    c2w = mat(im.camera_to_world)
    rays_w = rays_c @ c2w[:3, :3].T
    origin = np.broadcast_to(c2w[:3, 3], rays_w.shape)
    hit = scene.cast_rays(
        o3c.Tensor.from_numpy(np.ascontiguousarray(np.concatenate([origin, rays_w], -1), np.float32))
    )
    t = hit["t_hit"].numpy()
    nm_w = hit["primitive_normals"].numpy()
    valid = (depth > 0.1) & (depth < max_depth) & np.isfinite(t)
    if conf is not None:
        valid &= conf >= min_confidence
    points = rays_c * depth[..., None]
    ln, defined = lidar_normals(points, (depth > 0.1) & (depth < max_depth))
    nm_c = nm_w @ c2w[:3, :3]  # world -> camera
    nm_c = np.where((np.sum(nm_c * rays_c, axis=-1) > 0)[..., None], -nm_c, nm_c)
    cos = np.clip(np.sum(nm_c * ln, axis=-1), -1.0, 1.0)
    angle = np.where(defined, np.degrees(np.arccos(cos)), np.nan)
    return depth[valid], (t - depth)[valid], angle[valid]


def _stats(d: np.ndarray, r: np.ndarray, a: np.ndarray) -> dict[str, float | int]:
    ar = np.abs(r)
    aa = a[np.isfinite(a)]
    return {
        "pixels": len(r),
        "abs_median_mm": round(float(np.median(ar)) * 1000, 2),
        "abs_p90_mm": round(float(np.percentile(ar, 90)) * 1000, 2),
        "bias_median_mm": round(float(np.median(r)) * 1000, 2),
        "absrel_median": round(float(np.median(ar / d)), 5),
        "within_1cm": round(float(np.mean(ar < 0.01)), 4),
        "within_2cm": round(float(np.mean(ar < 0.02)), 4),
        "normal_median_deg": round(float(np.median(aa)), 2) if len(aa) else float("nan"),
        "normal_p90_deg": round(float(np.percentile(aa, 90)), 2) if len(aa) else float("nan"),
        "normal_within_10deg": round(float(np.mean(aa < 10)), 4) if len(aa) else float("nan"),
    }


def mesh_vs_lidar(
    mesh: Path,
    poses_dir: Path,
    ids: Iterable[str] | None = None,
    max_frames: int = 400,
    min_confidence: int = 2,
    max_depth: float = 4.0,
    clip_m: float = 0.25,
) -> dict[str, object]:
    """Agreement of `mesh` with the LiDAR of `poses_dir`'s depth frames at their poses there.

    `ids`: the frames to check (e.g. a dataset's video frames), else every depth frame; evenly thinned to
    `max_frames`. Residuals beyond `clip_m` (the mesh behind or in front of something else: occlusion,
    holes) are counted in `outside_clip` and left out of the statistics."""
    poses = Poses.model_validate_json((poses_dir / "poses.json").read_text())
    raw = {s.session: from_repo(s.raw_dir) for s in poses.sessions}
    frames = [f for f in read_jsonl(poses_dir / "frames.jsonl", PosedImage) if f.depth]
    if ids is not None:
        want = set(ids)
        frames = [f for f in frames if f.id in want]
    if len(frames) > max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    scene = scene_of(mesh)
    ds: list[np.ndarray] = []
    rs: list[np.ndarray] = []
    angs: list[np.ndarray] = []
    per_frame: list[float] = []
    outside = total = 0
    for f in frames:
        assert f.depth is not None
        depth = read_depth(raw[f.session] / f.depth)
        conf = read_confidence(raw[f.session] / f.confidence) if f.confidence else None
        d, r, a = frame_residuals(scene, f, depth, conf, min_confidence, max_depth)
        keep = np.abs(r) < clip_m
        outside += int(np.sum(~keep))
        total += len(r)
        if keep.sum() < 100:
            continue
        ds.append(d[keep])
        rs.append(r[keep])
        angs.append(a[keep])
        per_frame.append(float(np.median(np.abs(r[keep]))))
    if not rs:
        raise RuntimeError(f"no frame of {poses_dir} has LiDAR the mesh agrees with")
    d, r, a = np.concatenate(ds), np.concatenate(rs), np.concatenate(angs)
    pf = np.array(per_frame) * 1000
    return {
        "mesh": rel(mesh),
        "poses": rel(poses_dir),
        "frames": len(per_frame),
        "outside_clip": round(outside / max(total, 1), 4),
        "all": _stats(d, r, a),
        "bands": {
            f"{lo:g}-{hi:g}m": _stats(d[m], r[m], a[m])
            for lo, hi in BANDS
            if (m := (d >= lo) & (d < hi)).sum() > 0
        },
        "per_frame_abs_median_mm": {
            "median": round(float(np.median(pf)), 2),
            "p90": round(float(np.percentile(pf, 90)), 2),
            "max": round(float(pf.max()), 2),
        },
    }
