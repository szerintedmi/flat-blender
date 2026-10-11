"""LiDAR depth frames prepared for the free-space votes in `freespace.py`, saved as one .npz.

Every `every`-th depth frame with normal tracking: confident pixels only (confidence 2), up to 4.5 m, and
the minimum depth over a 5x5 window (holes ignored), so a thin object counts as surface where the coarse
sensor only half saw it. Intrinsics are scaled to the depth map; poses are world (scene frame) -> camera.
"""

from pathlib import Path

import cv2
import numpy as np

from .config import from_repo
from .io import read_confidence, read_depth, read_jsonl
from .models import bundle as b
from .models.common import F64, mat
from .models.derived import PosedImage, Poses


def depth_views(
    frames: list[PosedImage],
    raw: dict[str, Path],
    min_confidence: int = 2,
    window: int = 5,
    max_depth: float = 4.5,
) -> tuple[np.ndarray, F64, F64]:
    """(depth (n, h, w) float32, k (n, 4), w2c (n, 4, 4)) as `freespace.votes` takes them."""
    kernel = np.ones((window, window), np.uint8)
    depth: list[np.ndarray] = []
    ks: list[list[float]] = []
    w2cs: list[F64] = []
    for im in frames:
        assert im.depth is not None
        d = read_depth(raw[im.session] / im.depth).astype(np.float32)
        if im.confidence:
            d[read_confidence(raw[im.session] / im.confidence) < min_confidence] = 0.0
        d[d > max_depth] = 0.0
        # Minimum over the window, ignoring holes: a hole must not make its neighbours "free".
        d_min = -cv2.dilate(np.where(d > 0, -d, -np.inf).astype(np.float32), kernel)
        d_min[~(d_min <= max_depth)] = 0.0  # holes, and the border OpenCV pads with -FLT_MAX
        h, w = d.shape
        k = im.intrinsics
        sx, sy = w / k.width, h / k.height
        depth.append(d_min)
        ks.append([k.fx * sx, k.fy * sy, k.cx * sx, k.cy * sy])
        w2cs.append(np.linalg.inv(mat(im.camera_to_world)))
    return np.stack(depth), np.array(ks), np.stack(w2cs)


def write_views(poses_dir: Path, out: Path, every: int = 6) -> int:
    """Depth views of every `every`-th usable depth frame -> `out` (.npz). Returns the number of views."""
    poses = Poses.model_validate_json((poses_dir / "poses.json").read_text())
    raw = {s.session: from_repo(s.raw_dir) for s in poses.sessions}
    frames = [
        im
        for im in read_jsonl(poses_dir / "frames.jsonl", PosedImage)
        if im.depth and im.tracking in (None, b.TrackingState.NORMAL)
    ][::every]
    depth, k, w2c = depth_views(frames, raw)
    np.savez(out, depth=depth, k=k, w2c=w2c)
    return len(frames)
