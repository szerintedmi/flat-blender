"""Stage 3, fuse: TSDF fusion of LiDAR depth -> dense.ply (mesh) + points.ply (geometric ground truth).

Depth pixels below the confidence threshold or beyond `max_depth` are dropped. Frames are thinned
so consecutive integrated frames differ by a minimum motion, which keeps runtime proportional to
coverage rather than to session length. Surfaces need `min_weight` observations to be kept.
"""

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

from ..config import from_repo
from ..io import decode_frames, decoders, read_confidence, read_depth, read_jsonl
from ..models import bundle as b
from ..models.common import mat
from ..models.derived import PosedImage, Poses
from ..points import thin_frames

TRUNC_VOXELS = 4.0  # TSDF truncation band, in voxels (Open3D's trunc_voxel_multiplier)


def _tensor(a: np.ndarray) -> o3c.Tensor:
    return o3c.Tensor.from_numpy(np.ascontiguousarray(a))


def _frame_idx(f: PosedImage) -> int:
    return int(f.file.split("#")[1])


def _colours(
    raw: Path, frames: list[PosedImage], pool: ThreadPoolExecutor
) -> Future[dict[int, np.ndarray]] | None:
    """The frames' video colours at their depth maps' size (RGB uint8), decoded in `pool`; None without
    video or depth."""
    video = raw / b.VIDEO_FILE
    first = next((f for f in frames if f.depth), None)
    if not video.is_file() or first is None or first.depth is None:
        return None
    h, w = read_depth(raw / first.depth).shape

    def small(bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    return pool.submit(decode_frames, video, [_frame_idx(f) for f in frames], small, True)


def run(
    poses_dir: Path,
    out: Path,
    voxel: float = 0.01,
    min_confidence: int = 1,
    max_depth: float = 4.0,
    min_move: float = 0.05,
    min_turn_deg: float = 5.0,
    color: bool = True,
    min_weight: float = 2.0,
    block_count: int = 60_000,
) -> dict[str, object]:
    poses = Poses.model_validate_json((poses_dir / "poses.json").read_text())
    raw = {s.session: from_repo(s.raw_dir) for s in poses.sessions}
    frames = [
        f
        for f in read_jsonl(poses_dir / "frames.jsonl", PosedImage)
        if f.depth and f.tracking in (None, b.TrackingState.NORMAL)
    ]
    by_session: dict[str, list[PosedImage]] = defaultdict(list)
    for f in frames:
        by_session[f.session].append(f)

    # Tensor VoxelBlockGrid (Open3D's current TSDF API; the legacy ScalableTSDFVolume extracts
    # nothing in 0.20 on macOS arm64). Blocks of 8^3 voxels keep memory near the surfaces.
    names: tuple[str, ...] = ("tsdf", "weight", "color") if color else ("tsdf", "weight")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=names,
        attr_dtypes=tuple(o3c.float32 for _ in names),
        attr_channels=[o3c.SizeVector([3 if n == "color" else 1]) for n in names],
        voxel_size=voxel,
        block_resolution=8,
        block_count=block_count,
        device=o3c.Device("CPU:0"),
    )
    integrated = 0
    thinned = {sid: thin_frames(sframes, min_move, min_turn_deg) for sid, sframes in by_session.items()}
    # Video decoding is most of the time: every session's in parallel, integrated in order as before.
    pool = ThreadPoolExecutor(max_workers=decoders(len(thinned)))
    try:
        pending = {sid: _colours(raw[sid], fs, pool) if color else None for sid, fs in thinned.items()}
        for sid, sframes in thinned.items():
            job = pending.pop(sid)
            colours = job.result() if job is not None else {}
            for f in sframes:
                assert f.depth is not None
                d = read_depth(raw[sid] / f.depth)
                if f.confidence and min_confidence > 0:
                    d[read_confidence(raw[sid] / f.confidence) < min_confidence] = 0.0
                if np.count_nonzero((d > 0) & (d < max_depth)) < 100:  # ~nothing usable: Open3D aborts on it
                    continue
                h, w = d.shape
                depth = o3d.t.geometry.Image(_tensor(d.astype(np.float32)))
                k = f.intrinsics.scaled(w, h)
                # Our intrinsics put pixel centres at +0.5; Open3D puts them at integers.
                kt = _tensor(np.array([[k.fx, 0, k.cx - 0.5], [0, k.fy, k.cy - 0.5], [0, 0, 1]]))
                ext = _tensor(np.linalg.inv(mat(f.camera_to_world)))
                blocks = vbg.compute_unique_block_coordinates(depth, kt, ext, 1.0, max_depth, TRUNC_VOXELS)
                if color:
                    small = colours.get(_frame_idx(f))
                    if small is not None and small.shape[:2] != (h, w):  # a depth map of another size
                        small = cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA)
                    rgb = (
                        small.astype(np.float32) / 255.0
                        if small is not None
                        else np.full((h, w, 3), 0.5, np.float32)
                    )
                    img = o3d.t.geometry.Image(_tensor(rgb))
                    vbg.integrate(blocks, depth, img, kt, kt, ext, 1.0, max_depth, TRUNC_VOXELS)
                else:
                    vbg.integrate(blocks, depth, kt, ext, 1.0, max_depth, TRUNC_VOXELS)
                integrated += 1
    finally:
        pool.shutdown(cancel_futures=True)
    if integrated == 0:
        raise RuntimeError("no depth frames to fuse")

    mesh = vbg.extract_triangle_mesh(weight_threshold=min_weight).to_legacy()
    cloud = vbg.extract_point_cloud(weight_threshold=min_weight).to_legacy()
    if len(cloud.points) == 0:
        raise RuntimeError(f"fusion produced no surface from {integrated} frames (poses or depth wrong?)")
    mesh.compute_vertex_normals()
    if not cloud.has_normals():
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(out / "dense.ply", mesh)
    o3d.io.write_point_cloud(out / "points.ply", cloud)
    return {
        "frames_integrated": integrated,
        "frames_available": len(frames),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "points": len(cloud.points),
    }
