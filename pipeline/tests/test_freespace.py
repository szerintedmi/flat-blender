"""Pruning splats in LiDAR free space: depth views, votes, and msplat checkpoint surgery."""

import json
import struct
from pathlib import Path

import cv2
import numpy as np

from scanmodel import freespace, msplat_train
from scanmodel.depthviews import depth_views
from scanmodel.models import bundle as b
from scanmodel.models.derived import PosedImage

from .synth import look_at

KV = b.Intrinsics(fx=1500.0, fy=1500.0, cx=958.0, cy=721.0, width=1920, height=1440)


def test_free_space_votes(tmp_path: Path) -> None:
    """A camera facing a wall 3 m away: a splat at 1.5 m is in seen-through space, one at 3 m is on the
    wall, one at 4 m is hidden behind it; one outside the view and one in front of an edge hole get no
    vote."""
    raw = tmp_path / "s"
    (raw / "depth").mkdir(parents=True)
    depth = np.full((192, 256), 3000, np.uint16)
    depth[:, :20] = 0  # a hole at the image edge must not read as far away
    cv2.imwrite(str(raw / "depth" / "0.png"), depth)
    c2w = look_at(np.array([0.0, 0.0, 1.5]), np.array([0.0, 5.0, 1.5]))
    im = PosedImage.model_validate(
        {
            "id": "s/0",
            "session": "s",
            "kind": "frame",
            "file": "video.mov#0",
            "camera_to_world": c2w.tolist(),
            "intrinsics": KV.model_dump(),
            "depth": "depth/0.png",
        }
    )
    centres = np.array(
        [[0.0, 1.5, 1.5], [0.2, 3.0, 1.4], [0.0, 4.0, 1.5], [0.0, -2.0, 1.5], [-0.95, 1.5, 1.5]]
    )
    free, seen = freespace.votes(centres, np.full(5, 0.01), *depth_views([im] * 3, {"s": raw}))
    assert free.tolist() == [3, 0, 0, 0, 0]
    assert seen.tolist() == [0, 3, 0, 0, 0]
    assert freespace.drop_mask(free, seen).tolist() == [True, False, False, False, False]


def _checkpoint(path: Path, n: int) -> list[np.ndarray]:
    """An msplat 1.1.4 checkpoint: header, then (param, adam m, adam v) for six per-splat tensors."""
    rng = np.random.default_rng(1)
    shapes = [(n, 3), (n, 3), (n, 4), (n, 3), (n, 15, 3), (n, 1)] * 3
    tensors = [rng.standard_normal(s).astype("<f4") for s in shapes]
    header = b"MSPL" + struct.pack("<IIII", 1, 600, n, 3) + struct.pack("<I8f", 600, *range(8))
    msplat_train.write_checkpoint(path, header, tensors, n)
    return tensors


def test_checkpoint_rows_are_dropped_everywhere(tmp_path: Path) -> None:
    path = tmp_path / "ck.bin"
    tensors = _checkpoint(path, 5)
    header, n, back = msplat_train.read_checkpoint(path)
    assert (
        n == 5 and len(back) == 18 and all(np.array_equal(a, b) for a, b in zip(tensors, back, strict=True))
    )
    keep = np.array([True, False, True, True, False])
    msplat_train.write_checkpoint(path, header, [t[keep] for t in back], 3)
    _, n2, pruned = msplat_train.read_checkpoint(path)
    assert n2 == 3 and struct.unpack_from("<I", path.read_bytes(), 8)[0] == 600  # step kept
    assert all(p.shape[0] == 3 and np.array_equal(p, t[keep]) for p, t in zip(pruned, tensors, strict=True))


def test_internal_frame_inverts_msplat_scaling(tmp_path: Path) -> None:
    """msplat holds cameras scaled and shifted; the eval split (every 8th by path) is not among them."""
    pos = np.random.default_rng(2).uniform(-3, 9, (20, 3))
    frames = [
        {"file_path": f"images/{i:03d}.jpg", "transform_matrix": np.c_[np.eye(4, 3), [*p, 1]].tolist()}
        for i, p in enumerate(pos)
    ]
    (tmp_path / "transforms.json").write_text(
        json.dumps({"frames": frames[::-1]})
    )  # file order doesn't matter
    train = np.array([p for i, p in enumerate(pos) if i % 8 != 0])
    scale, offset = msplat_train.internal_frame(tmp_path, 0.17 * train + [0.1, -0.9, 0.0], 8, True)
    assert abs(scale - 0.17) < 1e-9 and np.allclose(offset, [0.1, -0.9, 0.0])
