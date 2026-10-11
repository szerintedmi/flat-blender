"""LED flicker banding: synthetic rolling-shutter bands with a known, spatially varying modulation."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from scanmodel import deband as db
from scanmodel.models.bundle import Intrinsics
from scanmodel.stages import splat

from .test_splat import _scan  # pyright: ignore[reportPrivateUsage]

W, H = 480, 360
K = Intrinsics(fx=400.0, fy=400.0, cx=240.0, cy=180.0, width=W, height=H)
FPS = 59.965  # the iPad's stream rate: 240.35° of 100 Hz flicker per frame
LIGHT = 100.0


def _rot(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    a, b = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx
    return m


def _scene(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A flicker-free 'world' image (seen by the reference camera, larger than a frame) and its flicker
    depth: a flat shaded wall on the left (lamp A, 25 %), texture on the right (lamp B, 6 %), a blown lamp."""
    rng = np.random.default_rng(seed)
    h, w = 2 * H, 2 * W
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    wall = 70 + 50 * x / w + 25 * y / h
    tex = cv2.GaussianBlur(rng.uniform(30, 160, (h, w)).astype(np.float32), (0, 0), 1.2)
    tex[(x.astype(int) // 23 + y.astype(int) // 31) % 2 == 0] *= 0.6  # hard edges
    grey = np.where(x < 0.55 * w, wall, tex)
    img = np.stack([grey * 0.85, grey, grey * 1.05], -1)
    cv2.circle(img, (int(0.3 * w), int(0.4 * h)), 25, (255, 255, 255), -1)
    depth = np.where(x < 0.55 * w, 0.25, 0.06).astype(np.float32)
    depth = cv2.GaussianBlur(depth, (0, 0), 15)
    return np.clip(img, 0, 255).astype(np.float32), depth


REF = db.View(np.zeros((1, 1, 3), np.uint8), 0.0, np.eye(4), K.scaled(2 * W, 2 * H))
REF.k = Intrinsics(fx=400.0, fy=400.0, cx=W, cy=H, width=2 * W, height=2 * H)


def _frame(
    world: np.ndarray, depth: np.ndarray, i: int, yaw: float = 0.0, pitch: float = 0.0, phase0: float = 0.4
):
    """Frame i (truth, banded) of a camera turning `yaw`/`pitch` degrees per frame: each row gets the flicker
    of its own readout time, as gain on (V + TRANSFER_OFFSET), with a 2nd harmonic."""
    t = i / FPS
    v = db.View(np.zeros((H, W, 3), np.uint8), t, _rot(yaw * i, pitch * i), K)
    hom = db.rotation_homography(v, REF)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    truth = cv2.warpPerspective(world, hom, (W, H), flags=flags, borderMode=cv2.BORDER_REFLECT)
    m = cv2.warpPerspective(depth, hom, (W, H), flags=flags, borderMode=cv2.BORDER_REFLECT)
    rows = np.arange(H, dtype=np.float32)[:, None]
    psi = phase0 + 2 * np.pi * LIGHT * t + 2 * np.pi * rows / db.period_rows(H, LIGHT)
    g = m * np.cos(psi) + 0.15 * m * np.cos(2 * psi + 1.0)
    a = db.TRANSFER_OFFSET
    banded = (truth / 255 + a) * np.exp(g)[..., None] - a
    clipped = truth >= 254.5  # a blown highlight stays blown
    out = np.where(clipped, 255, np.clip(banded * 255 + 0.5, 0, 255))
    v.img = out.astype(np.uint8)
    return v, truth.astype(np.uint8), m


def _band_amp(img: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Amplitude of the row-wise log ratio to the truth over `mask` (the band left)."""
    r = np.log(img.astype(np.float32).mean(2) + 1) - np.log(truth.astype(np.float32).mean(2) + 1)
    prof = np.array([np.median(r[y][mask[y]]) for y in range(H) if mask[y].any()])
    return float(np.percentile(prof, 95) - np.percentile(prof, 5)) / 2


@pytest.mark.parametrize(("yaw", "pitch"), [(0.0, 0.0), (0.4, 0.3)])
def test_deband_removes_spatially_varying_bands(yaw: float, pitch: float) -> None:
    world, depth = _scene()
    views = [_frame(world, depth, i, yaw, pitch) for i in (9, 10, 11)]
    cur, truth, _ = views[1]
    fixed, field = db.deband(cur, [views[0][0], views[2][0]], LIGHT)
    left = np.zeros((H, W), bool)
    left[:, 20 : int(0.4 * W)] = True  # the flat wall, strongly banded
    right = np.zeros((H, W), bool)
    right[:, int(0.7 * W) : W - 20] = True  # texture, weakly banded
    before, after = _band_amp(cur.img, truth, left), _band_amp(fixed, truth, left)
    assert before > 0.1 and after < before / 10
    assert _band_amp(fixed, truth, right) < 0.01
    # no new artefacts in the texture: closer to the truth than before, within a couple of DN
    err_before = np.abs(cur.img.astype(int) - truth)[right].mean()
    err_after = np.abs(fixed.astype(int) - truth)[right].mean()
    assert err_after < err_before and err_after < 2.5
    assert 0.1 < field.amplitude < 0.4


def test_deband_leaves_blown_highlights_and_unbanded_frames() -> None:
    world, depth = _scene()
    views = [_frame(world, depth, i) for i in (3, 4, 5)]
    cur, truth, _ = views[1]
    fixed, _ = db.deband(cur, [views[0][0], views[2][0]], LIGHT)
    assert (fixed[truth.min(2) == 255] == 255).all()
    flat = [_frame(world, depth * 0, i) for i in (3, 4, 5)]
    out, field = db.deband(flat[1][0], [flat[0][0], flat[2][0]], LIGHT)
    assert np.abs(out.astype(int) - flat[1][0].img).max() <= 2 and field.amplitude < 0.005


def test_rgb_get_their_own_gain() -> None:
    """A lamp whose flicker is weaker in red (a slow red phosphor): each channel corrected by its own field."""
    world, depth = _scene()
    views = []
    for i in (6, 7, 8):
        v, truth, m = _frame(world, depth, i)
        v.img[..., 2] = _frame(world, depth * 0.6, i)[0].img[..., 2]  # red: 60 % of the flicker
        views.append((v, truth, m))
    fixed, field = db.deband(views[1][0], [views[0][0], views[2][0]], LIGHT)
    amp = np.abs(field.w).reshape(-1, 3)
    assert np.median(amp[:, 2]) / np.median(amp[:, 1]) == pytest.approx(0.6, abs=0.06)
    left = np.zeros((H, W), bool)
    left[:, 20 : int(0.4 * W)] = True
    for c in range(3):
        r = np.log(fixed[..., c].astype(np.float32) + 1) - np.log(views[1][1][..., c].astype(np.float32) + 1)
        prof = np.nanmedian(np.where(left, r, np.nan), axis=1)
        assert np.nanpercentile(prof, 95) - np.nanpercentile(prof, 5) < 0.02


def test_still_gets_the_nearest_frames_gain() -> None:
    """A still at the frame's pose, twice the pixels and a 5 % longer focal: same gain at the same scene point."""
    world, depth = _scene()
    views = [_frame(world, depth, i) for i in (9, 10, 11)]
    frame = views[1][0]
    field = db.flicker_field(frame, [views[0][0], views[2][0]], LIGHT)
    k = Intrinsics(fx=840.0, fy=840.0, cx=480.0, cy=360.0, width=2 * W, height=2 * H)
    still = db.View(np.zeros((2 * H, 2 * W, 3), np.uint8), frame.t, frame.c2w, k)
    g_still = db.still_log_gain(field, frame, still, (2 * W, 2 * H))
    g_frame = db.log_gain(field, H, W)
    for sx, sy in ((480, 360), (200, 150), (700, 600)):
        fx, fy = (sx - 480) * 400 / 840 + 240, (sy - 360) * 400 / 840 + 180  # the same ray in the frame
        assert g_still[sy, sx, 1] == pytest.approx(
            db.STILL_SCALE * g_frame[round(fy), round(fx), 1], abs=0.01
        )


def test_neighbours_prefer_adjacent_and_skip_gaps() -> None:
    times = {i: i / FPS for i in range(10)}
    assert db.neighbours(5, times, LIGHT) == (4, 6)
    del times[4]
    assert db.neighbours(5, times, LIGHT) == (6, 7)  # +1 and +2: 240° and 120°
    assert db.neighbours(0, {0: 0.0, 1: 0.5, 2: 1.0}, LIGHT) is None  # every 0.5 s: always the same phase
    assert db.neighbours(42, times, LIGHT) is None


def _video(path: Path, banded: bool, n: int = 40) -> None:
    world, depth = _scene(1)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 60.0, (W, H))
    for i in range(n):
        v, truth, _ = _frame(world, depth if banded else depth * 0, i, yaw=0.3)
        out.write(v.img if banded else truth)
    out.release()


def test_detect_finds_banded_videos_and_caches(tmp_path: Path) -> None:
    _video(tmp_path / "banded.mov", True)
    _video(tmp_path / "clean.mov", False)
    found = db.detect(tmp_path / "banded.mov", LIGHT, samples=4)
    assert found["banded"] and float(found["amplitude"]) > 0.05  # type: ignore[arg-type]
    clean = db.detect(tmp_path / "clean.mov", LIGHT, samples=4)
    assert not clean["banded"] and float(clean["amplitude"]) < db.AUTO_THRESHOLD  # type: ignore[arg-type]
    cache = tmp_path / "deband" / "x.json"
    first = db.detect_cached(tmp_path / "banded.mov", cache, LIGHT)
    assert cache.is_file() and db.detect_cached(tmp_path / "banded.mov", cache, LIGHT) == first


def test_export_records_deband(tmp_path: Path) -> None:
    """The synthetic scan's video is unbanded (auto: left alone) and 0.5 s apart (on: no usable neighbours)."""
    poses, fuse = _scan(tmp_path / "scan")
    splat.export(poses, fuse, tmp_path / "a", select="time", deband="auto")
    sel = json.loads((tmp_path / "a" / "selection.json").read_text())["deband"]
    assert sel["mode"] == "auto" and set(sel["sessions"]) == {"s1-hallway", "s2-lap"}
    assert not any(s["applied"] for s in sel["sessions"].values())
    assert (tmp_path / "deband" / "s1-hallway.json").is_file()
    summary = splat.export(poses, fuse, tmp_path / "b", select="time", deband="on")
    sel = json.loads((tmp_path / "b" / "selection.json").read_text())["deband"]
    assert sel["sessions"]["s1-hallway"]["applied"] and sel["sessions"]["s1-hallway"]["skipped"] == 20
    assert summary["debanded"] == 0
    off = splat.export(poses, fuse, tmp_path / "c", select="time", deband="off")
    assert json.loads((tmp_path / "c" / "selection.json").read_text())["deband"]["sessions"] == {}
    assert off["images"] == summary["images"]
    with pytest.raises(ValueError, match="deband"):
        splat.export(poses, fuse, tmp_path / "x", deband="maybe")
