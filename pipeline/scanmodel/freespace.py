"""Splats in LiDAR free space: the votes behind pruning during splat training.

numpy only and no package imports, so the msplat training script (`msplat_train.py`, run by msplat's own
Python) can load it too.

Trainers without depth supervision put a surface's colour on semi-transparent splats in front of it where
a few photos disagree (exposure, reflections, a pose a few mm off): from the training views that looks
right, from anywhere else it is fog. The LiDAR saw that space as empty. For a sample of depth frames, each
splat centre in the frame's view is compared with the measured depth there:

- **free**: in front of the measured surface by more than `margin` (sensor noise plus the splat's own
  size), so the sensor looked through it;
- **seen**: on the measured surface (within `margin`);
- behind the surface, outside the frame or over a hole: no evidence.

A splat with at least `min_free` free votes and more free than seen votes is dropped. The depth maps are
the *minimum* over a small window (`depthviews.py`), so thin things the 256x192 sensor only half sees
(plant leaves, chair legs, cables) count as surface, not as free space.

Pruning a finished splat leaves holes: the wall's colour was on the removed splats and nothing behind them
learned it (third scan: held-out PSNR -0.55 dB). Pruning *during* training lets the remaining splats on
the real surface take over (+0.05 dB, and visibly less fog away from the training views).
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def load_views(path: Path) -> tuple[NDArray[np.float32], F64, F64]:
    """(depth (n, h, w) metres with 0 = no data, k (n, 4) fx fy cx cy in depth pixels, w2c (n, 4, 4))."""
    z = np.load(path)
    return z["depth"], z["k"], z["w2c"]


def votes(
    centres: F64, radius: F64, depth: NDArray[np.float32], k: F64, w2c: F64
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """(free, seen) vote counts per splat centre (scene frame) over the depth views."""
    free = np.zeros(len(centres), np.int32)
    seen = np.zeros(len(centres), np.int32)
    _, h, w = depth.shape
    for d, (fx, fy, cx, cy), m in zip(depth, k, w2c, strict=True):
        x = centres @ m[:3, :3].T + m[:3, 3]
        z = x[:, 2]
        front = np.flatnonzero(z > 0.1)
        u = np.floor(fx * x[front, 0] / z[front] + cx).astype(np.int64)
        v = np.floor(fy * x[front, 1] / z[front] + cy).astype(np.int64)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = front[ok]
        obs = d[v[ok], u[ok]].astype(np.float64)
        valid = obs > 0
        idx, obs = idx[valid], obs[valid]
        margin = 0.04 + 0.015 * obs + radius[idx]
        zi = z[idx]
        free[idx[zi < obs - margin]] += 1
        seen[idx[np.abs(zi - obs) <= margin]] += 1
    return free, seen


def drop_mask(free: NDArray[np.int32], seen: NDArray[np.int32], min_free: int = 3) -> NDArray[np.bool_]:
    return (free >= min_free) & (free > seen)
