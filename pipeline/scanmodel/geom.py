"""Axis conventions and small geometry helpers.

Pipeline convention: world z up (metres); cameras in OpenCV axes (+x right, +y down, +z forward).
ARKit: world y up; cameras in OpenGL axes (+x right, +y up, camera looks down -z).
"""

import numpy as np

from .models.bundle import Intrinsics
from .models.common import F64

# ARKit world (x, y up, z) -> pipeline world (x, -z, y): gravity-aligned with z up, right-handed.
ARKIT_TO_ZUP: F64 = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)
# OpenGL camera axes <-> OpenCV camera axes (its own inverse).
GL_CV: F64 = np.diag([1.0, -1.0, -1.0, 1.0])


def arkit_to_pipeline(c2w_arkit: F64) -> F64:
    """ARKit camera_to_world (GL camera, y-up world) -> pipeline (CV camera, z-up world)."""
    return ARKIT_TO_ZUP @ c2w_arkit @ GL_CV


def rot_z(deg: float) -> F64:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


def translation(t: tuple[float, float, float] | F64) -> F64:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def apply(m: F64, pts: F64) -> F64:
    """Transform (N, 3) points by a 4x4 matrix."""
    return pts @ m[:3, :3].T + m[:3, 3]


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> F64:
    q = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def k_matrix(k: Intrinsics) -> F64:
    return np.array([[k.fx, 0.0, k.cx], [0.0, k.fy, k.cy], [0.0, 0.0, 1.0]])


def backproject(depth_m: F64, k: Intrinsics, stride: int = 1) -> F64:
    """Depth map (H, W) in metres -> (N, 3) points in OpenCV camera axes. Zero depth is dropped."""
    h, w = depth_m.shape
    ks = k.scaled(w, h)
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[::stride, ::stride]
    ok = z > 0
    x = (u[ok] + 0.5 - ks.cx) / ks.fx * z[ok]
    y = (v[ok] + 0.5 - ks.cy) / ks.fy * z[ok]
    return np.stack([x, y, z[ok]], axis=1)


def project(pts_cam: F64, k: Intrinsics, w: int, h: int) -> tuple[F64, F64]:
    """(N, 3) OpenCV camera points -> pixel (u, v) at size (w, h) and depth z."""
    ks = k.scaled(w, h)
    z = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = pts_cam[:, 0] / z * ks.fx + ks.cx - 0.5
        v = pts_cam[:, 1] / z * ks.fy + ks.cy - 0.5
    return np.stack([u, v], axis=1), z


def is_rigid(m: F64, tol: float = 1e-3) -> bool:
    r = m[:3, :3]
    return (
        bool(np.all(np.isfinite(m)))
        and np.allclose(r.T @ r, np.eye(3), atol=tol)
        and abs(np.linalg.det(r) - 1) < tol
        and np.allclose(m[3], [0, 0, 0, 1], atol=tol)
    )


def angle_mod90(deg: float | F64) -> F64:
    """Wrap angles to (-45, 45]."""
    return -((-np.asarray(deg, dtype=np.float64) + 45.0) % 90.0 - 45.0)


def dominant_axis_deg(angles_deg: F64, weights: F64) -> float:
    """Dominant direction of a set of lines, modulo 90 degrees (circular mean of 4*theta)."""
    a = np.radians(angles_deg) * 4
    s, c = float(np.sum(weights * np.sin(a))), float(np.sum(weights * np.cos(a)))
    return float(np.degrees(np.arctan2(s, c)) / 4)
