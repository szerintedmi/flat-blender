"""Still focal calibration (scanmodel.stillcal)."""

import cv2
import numpy as np
import pytest

from scanmodel import stillcal
from scanmodel.models import bundle as b

KV = b.Intrinsics(fx=1500.0, fy=1500.0, cx=958.0, cy=721.0, width=1920, height=1440)
IDENTITY = np.eye(4).tolist()


def _texture(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Random blobs and lines: plenty of SIFT corners at every scale."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8), (3, 3), 0)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(300):
        p, q = rng.integers(0, [w, h], 2), rng.integers(0, [w, h], 2)
        cv2.line(
            img,
            tuple(int(v) for v in p),
            tuple(int(v) for v in q),
            tuple(int(v) for v in rng.integers(0, 255, 3)),
            3,
        )
    return img


@pytest.mark.parametrize("true_scale", [1.0, 1.048])
def test_still_focal_scale_is_recovered(true_scale: float) -> None:
    """A still whose real focal is `true_scale` times the recorded one (the stream's, scaled to 12 MP)."""
    frame_img = _texture(1920, 1440)
    ks = KV.scaled(4032, 3024)  # what ARKit records for the still
    # Still pixel q sees the ray (q - c) / (f * true_scale); the frame shows that ray at c_v + f_v * ray.
    qx, qy = np.meshgrid(np.arange(4032, dtype=np.float32), np.arange(3024, dtype=np.float32))
    map_x = (KV.cx + KV.fx * (qx - ks.cx) / (ks.fx * true_scale)).astype(np.float32)
    map_y = (KV.cy + KV.fy * (qy - ks.cy) / (ks.fy * true_scale)).astype(np.float32)
    still_img = cv2.remap(frame_img, map_x, map_y, cv2.INTER_CUBIC)
    still = b.Still(
        id=1, file="stills/000001.jpg", source=b.StillSource.AUTO, intrinsics=ks, camera_to_world=IDENTITY
    )
    frame = b.Frame(idx=0, t=0.0, camera_to_world=IDENTITY, intrinsics=KV, tracking=b.TrackingState.NORMAL)
    fit = stillcal._fit_one(still_img, frame_img, still, frame, cv2.SIFT.create(nfeatures=4000))  # pyright: ignore[reportPrivateUsage]
    assert fit is not None
    assert fit[0] == pytest.approx(true_scale, abs=0.002)
    fixed = stillcal.corrected(ks, fit[0])
    assert (fixed.cx, fixed.cy, fixed.width) == (ks.cx, ks.cy, ks.width)
    assert fixed.fx == pytest.approx(ks.fx * true_scale, rel=0.002)
