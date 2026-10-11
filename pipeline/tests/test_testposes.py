"""Re-solved test poses (`scanmodel.testposes`) on a synthetic textured room with known cameras and LiDAR."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from scanmodel import testposes as tp
from scanmodel.geom import GL_CV
from scanmodel.models.common import F64

from .synth import look_at

ROOM = (np.array([0.0, 0.0, 0.0]), np.array([4.0, 3.0, 2.5]))
W, H, F = 640, 480, 520.0
K = np.array([[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]])
TEX_PER_M = 300  # texture pixels per metre (a pixel of a 640 px view 2 m away covers ~4 mm)
FLOOR = (2, 0)  # (axis, side) of the floor: left blank, so a view of it alone can't be re-solved


def _texture(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Blobs at several scales (what SIFT finds), uint8 grey."""
    t = np.zeros((h, w), np.float32)
    for cells, amp in ((6, 0.6), (20, 0.8), (60, 1.0), (180, 0.5)):
        g = rng.random((max(2, round(cells * h / w)), cells)).astype(np.float32)
        t += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    t = (t - t.min()) / (t.max() - t.min())
    return (40 + 180 * t).astype(np.uint8)


def _textures(seed: int = 0) -> dict[tuple[int, int], np.ndarray]:
    """A texture per wall (axis, side), covering its two other axes' extent."""
    rng = np.random.default_rng(seed)
    lo, hi = ROOM
    out: dict[tuple[int, int], np.ndarray] = {}
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        for side in (0, 1):
            w, h = round((hi[a] - lo[a]) * TEX_PER_M), round((hi[b] - lo[b]) * TEX_PER_M)
            out[(axis, side)] = (
                np.full((h, w), 128, np.uint8) if (axis, side) == FLOOR else _texture(rng, w, h)
            )
    return out


def _cast(
    c2w: F64, k: F64, w: int, h: int, textures: dict[tuple[int, int], np.ndarray]
) -> tuple[np.ndarray, F64]:
    """(grey image, camera z-depth) of the room seen by an OpenCV camera, pixel centres at i + 0.5."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1) @ c2w[:3, :3].T
    o = c2w[:3, 3]
    lo, hi = ROOM
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lo, t_hi = (lo - o) / rays, (hi - o) / rays
    t_exit = np.where(rays > 0, t_hi, np.where(rays < 0, t_lo, np.inf))
    axis = np.argmin(t_exit, axis=-1)
    t = np.take_along_axis(t_exit, axis[..., None], -1)[..., 0]
    hit = o + rays * t[..., None]
    img = np.zeros((h, w), np.float32)
    for (ax, side), tex in textures.items():
        m = (axis == ax) & ((rays[..., ax] > 0) == (side == 1))
        if not m.any():
            continue
        a, b = [i for i in range(3) if i != ax]
        mx = ((hit[..., a] - lo[a]) * TEX_PER_M - 0.5).astype(np.float32)
        my = ((hit[..., b] - lo[b]) * TEX_PER_M - 0.5).astype(np.float32)
        s = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        img[m] = s[m]
    return img.clip(0, 255).astype(np.uint8), t  # rays have camera z = 1, so t is the z-depth


def _entry(stem: str, c2w: F64, session: str, t: float) -> dict[str, Any]:
    return {
        "file_path": f"images/{stem}.jpg",
        "transform_matrix": (c2w @ GL_CV).tolist(),
        "fl_x": F,
        "fl_y": F,
        "cx": W / 2,
        "cy": H / 2,
        "w": W,
        "h": H,
        "camera": "video",
        "session": session,
        "t": t,
        "depth_file_path": f"depth/{stem}.png",
        "confidence_file_path": f"confidence/{stem}.png",
    }


def _write_frame(data: Path, stem: str, c2w: F64, textures: dict[tuple[int, int], np.ndarray]) -> None:
    img, _ = _cast(c2w, K, W, H, textures)
    cv2.imwrite(str(data / f"images/{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    kd = np.diag([256 / W, 192 / H, 1.0]) @ K
    _, depth = _cast(c2w, kd, 256, 192, textures)
    cv2.imwrite(str(data / f"depth/{stem}.png"), np.round(depth * 1000).astype(np.uint16))
    cv2.imwrite(str(data / f"confidence/{stem}.png"), np.full((192, 256), 2, np.uint8))


def _yaw_pitch(eye: F64, yaw: float, pitch: float = -5.0) -> F64:
    d = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), math.tan(math.radians(pitch))])
    return look_at(eye, eye + d)


def _perturb(c2w: F64, shift: tuple[float, float, float], yaw_deg: float) -> F64:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = c2w.copy()
    out[:3, :3] = r @ c2w[:3, :3]
    out[:3, 3] += shift
    return out


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, F64]]:
    """A dataset v2 (images, depth, confidence, transforms.json, split.json) of the room: a training pass
    turning in place at two spots, an interp frame among them, and a lap whose dataset poses are off
    (`true` holds the poses its images were rendered from)."""
    data = tmp_path_factory.mktemp("testposes") / "dataset"
    for d in ("images", "depth", "confidence"):
        (data / d).mkdir(parents=True)
    textures = _textures()
    frames: list[dict[str, Any]] = []
    true: dict[str, F64] = {}
    i = 0
    for eye in (np.array([1.6, 1.4, 1.3]), np.array([2.4, 1.6, 1.4])):
        for yaw in range(-180, 180, 15):
            stem = f"room_{i:06d}"
            c2w = _yaw_pitch(eye, yaw)
            _write_frame(data, stem, c2w, textures)
            frames.append(_entry(stem, c2w, "room", i / 10))
            i += 1
    interp = [frames[5]["file_path"][7:-4], frames[29]["file_path"][7:-4]]
    lap: list[dict[str, Any]] = []
    lap_poses = [
        ("lap_000000", _yaw_pitch(np.array([2.0, 1.5, 1.35]), 22.0), (0.30, 0.10, -0.05), 1.5),
        ("lap_000012", _yaw_pitch(np.array([2.1, 1.45, 1.3]), 97.0), (0.0, 0.45, 0.0), -1.0),
        ("lap_000024", _yaw_pitch(np.array([1.9, 1.5, 1.3]), -150.0), (0.0, 0.0, 0.0), 0.0),
        ("lap_000036", look_at(np.array([2.0, 1.5, 0.8]), np.array([2.05, 1.5, 0.0])), (0.0, 0.0, 0.0), 0.0),
    ]
    for j, (stem, c2w, shift, yaw) in enumerate(lap_poses):
        _write_frame(data, stem, c2w, textures)
        true[stem] = c2w
        lap.append(_entry(stem, _perturb(c2w, shift, yaw), "lap", 100 + j))
    for e in frames:
        true[Path(e["file_path"]).stem] = np.asarray(e["transform_matrix"]) @ GL_CV
    (data / "transforms.json").write_text(
        json.dumps({"version": 2, "split_file_path": "split.json", "frames": frames})
    )
    split = {
        "interp": interp,
        "extrap": [e["file_path"][7:-4] for e in lap],
        "extrap_session": "lap",
        "extrap_frames": lap,
    }
    (data / "split.json").write_text(json.dumps(split))
    return data, true


@pytest.fixture(scope="module")
def solved(dataset: tuple[Path, dict[str, F64]]) -> dict[str, Any]:
    return tp.write(dataset[0], workers=4)


def _c2w(row: dict[str, Any]) -> F64:
    return np.asarray(row["transform_matrix"]) @ GL_CV


def test_recovers_perturbed_lap_poses_within_mm(
    dataset: tuple[Path, dict[str, F64]], solved: dict[str, Any]
) -> None:
    _, true = dataset
    for stem in ("lap_000000", "lap_000012"):
        row = solved["frames"][stem]
        assert row["status"] == "ok", row
        dm, dd = tp.pose_diff(_c2w(row), true[stem])
        assert dm < 0.002 and dd < 0.05, (stem, dm, dd)  # 0.4-0.6 mm, 0.02 deg here
        assert row["shift_cm"] > 25 and not row.get("kept_dataset_pose")
        assert row["inliers"] >= tp.MIN_INLIERS and row["refs_agree"] >= tp.MIN_REFS_AGREE
        assert row["reproj_dataset_px"] > 10 * row["reproj_px"]
    # Exact dataset poses barely move: the interp frames and an unperturbed lap frame.
    for stem in [*json.loads((dataset[0] / "split.json").read_text())["interp"], "lap_000024"]:
        row = solved["frames"][stem]
        assert row["status"] == "ok", (stem, row)
        assert row["shift_cm"] < 0.4 and row["rot_deg"] < 0.05, (stem, row)


def test_blank_view_is_unresolved_not_guessed(solved: dict[str, Any]) -> None:
    row = solved["frames"]["lap_000036"]  # looking straight down at the blank floor
    assert row["status"] == "unresolved" and row["reason"]
    assert "transform_matrix" not in row and "shift_cm" not in row


def test_references_are_training_frames_only(
    dataset: tuple[Path, dict[str, F64]], solved: dict[str, Any]
) -> None:
    split = json.loads((dataset[0] / "split.json").read_text())
    tests = set(split["interp"]) | set(split["extrap"])
    assert set(solved["frames"]) == tests
    used = {r for row in solved["frames"].values() for r in row["refs"]}
    assert used and not used & tests


def test_file_contract(dataset: tuple[Path, dict[str, F64]], solved: dict[str, Any]) -> None:
    """test_poses.json as trainer/README.md ("Dataset v2") describes it and gs_trainer.evaluate reads it."""
    data = dataset[0]
    on_disk = json.loads((data / tp.FILE).read_text())
    assert on_disk == json.loads(json.dumps(solved))
    assert on_disk["version"] == 1 and on_disk["method"] == "sift-pnp"
    sha = {k: hashlib.sha256((data / f).read_bytes()).hexdigest() for k, f in
           (("transforms_sha256", "transforms.json"), ("split_sha256", "split.json"))}  # fmt: skip
    assert on_disk["source"] == sha
    assert isinstance(on_disk["params"], dict) and on_disk["wall_s"] >= 0
    for name, n in (("interp", 2), ("extrap", 4)):
        s = on_disk["summary"][name]
        assert s["n"] == n == s["resolved"] + s["unresolved"]
        assert {"moved", "shift_cm_median", "shift_cm_max"} <= set(s)
    assert on_disk["summary"]["extrap"]["moved"] == 2
    common = {"status", "refs", "matches", "inliers", "reproj_px", "reproj_dataset_px", "sigma_cm", "sigma_deg",
              "quadrants", "hull", "depth_resid_cm", "refs_agree", "refs_disagree", "ref_checks"}  # fmt: skip
    for row in on_disk["frames"].values():
        assert common <= set(row), set(row)
        if row["status"] == "ok":
            m = np.asarray(row["transform_matrix"])
            assert m.shape == (4, 4) and np.allclose(m[3], [0, 0, 0, 1])
            assert np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-5)
            assert len(row["offset_cm"]) == 3 and row["shift_cm"] >= 0 and row["rot_deg"] >= 0
            assert "reason" not in row
        else:
            assert row["status"] == "unresolved" and isinstance(row["reason"], str)
            assert not {"transform_matrix", "shift_cm", "rot_deg", "offset_cm"} & set(row)
    # Dataset v2 itself is untouched (its cloud hash stays the same).
    assert "test_poses" not in json.loads((data / "split.json").read_text())


def test_write_to_another_file_leaves_the_dataset(
    dataset: tuple[Path, dict[str, F64]], tmp_path: Path
) -> None:
    data = dataset[0]
    before = (data / tp.FILE).read_bytes() if (data / tp.FILE).exists() else None
    out = tmp_path / "scratch" / "poses.json"
    assert tp.main([str(data), "--out", str(out), "--workers", "2"]) == 0
    split = json.loads((data / "split.json").read_text())
    assert set(json.loads(out.read_text())["frames"]) == set(split["interp"]) | set(split["extrap"])
    assert ((data / tp.FILE).read_bytes() if (data / tp.FILE).exists() else None) == before


def test_sample_depth_is_bilinear_and_drops_edges() -> None:
    depth = np.tile(np.linspace(1.0, 2.0, 64), (48, 1))  # a ramp along x
    depth[:, 40:] += 1.0  # a depth edge at column 40
    conf = np.full(depth.shape, 2, np.uint8)
    conf[30:, :10] = 1
    # image 640 x 480 over the 64 x 48 depth: depth pixel j's centre is at image x = (j + 0.5) * 10
    uv = np.array([[205.0, 100.0], [210.0, 100.0], [400.0, 100.0], [50.0, 400.0]])
    z = tp.sample_depth(depth, conf, uv, 640, 480)
    step = 1.0 / 63
    assert z[0] == pytest.approx(1.0 + 20 * step, abs=1e-9)
    assert z[1] == pytest.approx(1.0 + 20.5 * step, abs=1e-9)
    assert z[2] == 0.0  # at the edge
    assert z[3] == 0.0  # low confidence


def test_temporal_outlier_is_dropped() -> None:
    eye = np.eye(4)
    cams = [tp.Cam(f"s_{i}", Path("x"), K, W, H, eye, False, "s", float(i), None, None) for i in range(5)]
    rows: list[dict[str, Any]] = []
    for i in range(5):
        off = np.array([0.0, 0.45, 0.0]) if i != 2 else np.array([0.0, 0.10, 0.0])
        pose = eye.copy()
        pose[:3, 3] = off
        rows.append({"status": "ok", "offset_cm": (100 * off).tolist(), "pose": pose})
    out = tp.temporal_outliers(cams, rows)
    assert list(out) == [2] and out[2].startswith("temporal")
    rows[2]["status"] = "unresolved"
    assert tp.temporal_outliers(cams, rows) == {}


def test_pose_sigma_grows_on_degenerate_points() -> None:
    rng = np.random.default_rng(1)
    rvec, tvec = np.zeros(3), np.zeros(3)
    spread = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(-0.8, 0.8, 200), rng.uniform(2, 4, 200)])
    clump = np.column_stack([rng.uniform(-0.05, 0.05, 200), rng.uniform(-0.05, 0.05, 200), np.full(200, 3.0)])
    sig = []
    for pts in (spread, clump):
        uv = cv2.projectPoints(pts, rvec, tvec, K, None)[0].reshape(-1, 2) + rng.normal(0, 0.5, (200, 2))
        sig.append(tp.pose_sigma(pts, uv, K, rvec, tvec))
    assert sig[0][0] < 0.005 and sig[1][0] > 5 * sig[0][0]
