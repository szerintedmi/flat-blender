"""Focal length of the high-res stills, measured against the stream frames.

ARKit's `captureHighResolutionFrame` reports the stream camera's intrinsics scaled to the still's size, but
the 12 MP still covers a narrower field of view: on the M2 iPad Pro its true focal length is 4.8 % longer
(third scan: every still, spread 0.1 %; with the corrected focal the still matches its nearest stream
frame to 0.5 px median, so there is no extra distortion or principal-point offset to model). Uncorrected,
a still is off by up to ~45 px at its edges (1920 px wide), which blurs splats and would misplace textures.

Measured per session, not assumed: SIFT matches between a few stills and their nearest stream frames; the
rays of the frame's matches, turned into the still's camera, fix the one unknown, a scale on fx and fy
about the principal point. The poses are ARKit's own (still and nearest frame are captured together, so
any translation between them is negligible at room distances).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from .io import VideoReader
from .models import bundle as b
from .models.common import F64, Mat4, mat

GL_TO_CV = np.diag([1.0, -1.0, -1.0])
MATCH_SIDE = 1920  # stills are compared at the stream's width
MIN_INLIERS = 40


@dataclass
class StillScale:
    scale: float  # true focal / recorded focal (1.0 when not measured)
    stills: int  # stills that gave a fit
    residual_px: float | None  # median reprojection error after the correction, at MATCH_SIDE


def _k(i: b.Intrinsics, s: float = 1.0) -> F64:
    return np.array([[i.fx * s, 0.0, i.cx * s], [0.0, i.fy * s, i.cy * s], [0.0, 0.0, 1.0]])


def _rot_cv(c2w_arkit: Mat4) -> F64:
    return mat(c2w_arkit)[:3, :3] @ GL_TO_CV


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _fit_one(
    still_img: np.ndarray, frame_img: np.ndarray, still: b.Still, frame: b.Frame, sift: cv2.SIFT
) -> tuple[float, F64, F64, F64] | None:
    """(scale, still pixels, normalised still rays from the frame, focal-scaled K) for one pair."""
    assert still.intrinsics is not None and still.camera_to_world is not None
    s = MATCH_SIDE / still_img.shape[1]
    still_img = cv2.resize(
        still_img, (MATCH_SIDE, round(still_img.shape[0] * s)), interpolation=cv2.INTER_AREA
    )
    k1, d1 = cast(tuple[object, object], sift.detectAndCompute(_gray(still_img), None))
    k2, d2 = cast(tuple[object, object], sift.detectAndCompute(_gray(frame_img), None))
    if not isinstance(d1, np.ndarray) or not isinstance(d2, np.ndarray):  # blank image
        return None
    kp1, kp2 = cast(Sequence[cv2.KeyPoint], k1), cast(Sequence[cv2.KeyPoint], k2)
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    p1 = np.array([kp1[g.queryIdx].pt for g in good], np.float64)
    p2 = np.array([kp2[g.trainIdx].pt for g in good], np.float64)
    _, inl = cast(tuple[object, object], cv2.findHomography(p1, p2, cv2.RANSAC, 3.0))
    if not isinstance(inl, np.ndarray) or inl.sum() < MIN_INLIERS:
        return None
    p1, p2 = p1[inl.ravel() > 0], p2[inl.ravel() > 0]
    ks, kv = _k(still.intrinsics, s), _k(frame.intrinsics)
    r = _rot_cv(still.camera_to_world).T @ _rot_cv(frame.camera_to_world)  # frame camera -> still camera
    rays = (r @ np.linalg.solve(kv, np.c_[p2, np.ones(len(p2))].T)).T
    xn = rays[:, :2] / rays[:, 2:3]
    f = np.array([ks[0, 0], ks[1, 1]])
    pred, got = xn * f, p1 - ks[:2, 2]
    return float((pred * got).sum() / (pred * pred).sum()), got, xn, ks


def still_focal_scale(
    bundle_dir: Path, frames: list[b.Frame], stills: list[b.Still], max_stills: int = 8
) -> StillScale:
    """Median focal scale of up to `max_stills` stills (the earliest, so the video is decoded only that far)."""
    video = bundle_dir / b.VIDEO_FILE
    by_idx = {f.idx: f for f in frames}
    todo = sorted(
        (s for s in stills if s.intrinsics and s.camera_to_world and s.nearest_frame in by_idx),
        key=lambda s: s.nearest_frame or 0,
    )
    if not todo or not video.is_file():
        return StillScale(1.0, 0, None)
    sift = cv2.SIFT.create(nfeatures=4000)
    fits: list[tuple[float, F64, F64, F64]] = []
    reader = VideoReader(video)
    try:
        for still in todo:
            if len(fits) >= max_stills:
                break
            assert still.nearest_frame is not None
            frame_img = reader.read(still.nearest_frame)
            still_img = cv2.imread(str(bundle_dir / still.file))
            if frame_img is None:
                break
            if still_img is None:
                continue
            fit = _fit_one(still_img, frame_img, still, by_idx[still.nearest_frame], sift)
            if fit is not None:
                fits.append(fit)
    finally:
        reader.close()
    if len(fits) < 3:
        return StillScale(1.0, len(fits), None)
    scale = float(np.median([f[0] for f in fits]))
    err = np.concatenate(
        [np.linalg.norm(xn * np.array([k[0, 0], k[1, 1]]) * scale - got, axis=1) for _, got, xn, k in fits]
    )
    return StillScale(scale, len(fits), float(np.median(err)))


def corrected(i: b.Intrinsics, scale: float) -> b.Intrinsics:
    """Intrinsics with fx, fy scaled about the principal point."""
    return i.model_copy(update={"fx": i.fx * scale, "fy": i.fy * scale})
