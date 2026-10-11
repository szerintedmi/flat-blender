"""Fusion of LiDAR and (fake) DA3 depth on the synthetic room, whose true depth is known in closed form."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from gs_trainer import calibrate, da3, fusion, priors, synth
from gs_trainer.data import Scene, load_depth, scaled_k

LIDAR_SIGMA = {2: 0.015, 1: 0.040}  # injected LiDAR noise per confidence (m)
DA3_SCALE = 1.1  # the fake DA3's scale error in posed windows (10 %)
DA3_REL = 0.01  # and its per-pixel relative noise
OUTLIER_STEM = "synth-ring_000045"  # the fake DA3 is 0.5 m off in a patch of this frame
PROCESS_RES = 56


def true_depth(c2w: np.ndarray, k: np.ndarray, w: int, h: int) -> np.ndarray:
    """z-depth (h, w) of the synthetic box room through pixel centres."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    cam = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    t, _, _ = synth.ray_box(c2w[:3, 3], cam @ c2w[:3, :3].T)
    return t.reshape(h, w)


class FakeDA3:
    """Stands in for `DepthAnything3`: true depth at the processing size, x a scale error, + noise.

    Checks that the poses and intrinsics it is given are the frames' own (OpenCV world-to-camera, K at the
    image's size). Unposed windows get an arbitrary scale (0.5), as the real model's would be.
    """

    def __init__(self, scene: Scene, seed: int = 0) -> None:
        self.by_path = {str(f.image_path): f for f in scene.frames}
        self.rng = np.random.default_rng(seed)
        self.calls: list[list[str]] = []

    def inference(
        self,
        image: list[str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        process_res: int = 504,
    ) -> SimpleNamespace:
        frames = [self.by_path[p] for p in image]
        self.calls.append([f.stem for f in frames])
        scale = DA3_SCALE if extrinsics is not None else 0.5
        depths = []
        for i, f in enumerate(frames):
            if extrinsics is not None and intrinsics is not None:
                np.testing.assert_allclose(extrinsics[i], np.linalg.inv(f.c2w), atol=1e-5)
                np.testing.assert_allclose(intrinsics[i], f.K, rtol=1e-6)
            s = process_res / max(f.w, f.h)
            w, h = round(f.w * s), round(f.h * s)
            z = true_depth(f.c2w, scaled_k(f.K, f.w, f.h, w, h), w, h) * scale
            z *= 1 + DA3_REL * self.rng.normal(size=z.shape)
            if f.stem == OUTLIER_STEM:
                z[10:30, 10:40] += 0.5
            depths.append(z)
        d = np.stack(depths).astype(np.float32)
        return SimpleNamespace(depth=d, conf=np.full_like(d, 2.0))


def degrade_lidar(data: Path, seed: int = 1) -> None:
    """Noise on every LiDAR reading (LIDAR_SIGMA by confidence) and a hole in the left fifth of each map."""
    rng = np.random.default_rng(seed)
    for p in sorted((data / "depth").glob("*.png")):
        mm = np.asarray(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), np.float64)
        conf = cv2.imread(str(data / "confidence" / p.name), cv2.IMREAD_UNCHANGED)
        sigma = np.where(conf == 2, LIDAR_SIGMA[2], np.where(conf == 1, LIDAR_SIGMA[1], 0.0)) * 1000
        noisy = np.where(mm > 0, np.clip(mm + rng.normal(size=mm.shape) * sigma, 1, 65535), 0)
        noisy[:, : mm.shape[1] // 5] = 0
        cv2.imwrite(str(p), noisy.round().astype(np.uint16))


def use_fake(monkeypatch: pytest.MonkeyPatch, data: Path) -> FakeDA3:
    fake = FakeDA3(Scene.load(data))
    monkeypatch.setattr(da3, "load_model", lambda name=None, device=None: fake)
    return fake


@pytest.fixture(scope="module")
def fused_data(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, dict[str, np.ndarray]], dict]:
    """A degraded synthetic dataset with priors, sigma table, fake DA3 and every fusion mode written.

    Returns (data, the npz contents before fusion, the fusion report).
    """
    data = synth.write_dataset(tmp_path_factory.mktemp("fusion") / "data", n_frames=24, size=(128, 96))
    degrade_lidar(data)
    priors.prepare(data, side=128)
    calibrate.calibrate(data)
    mp = pytest.MonkeyPatch()
    try:
        use_fake(mp, data)  # pyright: ignore[reportArgumentType]
        da3.run(data, window=6, overlap=3, process_res=PROCESS_RES)
    finally:
        mp.undo()
    before = {}
    for p in sorted((data / "prior").glob("*.npz")):
        with np.load(p) as z:
            before[p.stem] = {k: z[k] for k in z.files}
    report = fusion.fuse(data, "all")
    return data, before, report


def test_windows_cover_every_item_centrally() -> None:
    assert da3.windows(5, 16, 8) == [(0, 5)]
    assert da3.windows(0, 16, 8) == []
    wins = da3.windows(37, 8, 4)
    assert wins[0] == (0, 8) and wins[-1] == (29, 37)
    assert all(b - a == 8 for a, b in wins)
    owner = da3.central_window(37, wins)
    for j, k in enumerate(owner):
        a, b = wins[k]
        assert a <= j < b
        if 2 <= j < 35:  # away from the sequence's ends: inside the window's middle half
            assert a + 2 <= j < b - 2


def test_run_predicts_every_frame_once_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = synth.write_dataset(tmp_path / "data")
    fake = use_fake(monkeypatch, data)
    meta = da3.run(data, window=6, overlap=3, process_res=PROCESS_RES)
    scene = Scene.load(data)
    assert meta["computed"] == len(scene.frames) == 20 and meta["shape"] == [42, 56]
    for f in scene.frames:
        got = da3.load(data, f.stem)
        assert got is not None and got[0].shape == (42, 56) and got[1].shape == (42, 56)
        z = np.load(da3.da3_path(data, f.stem))
        assert z["depth"].dtype == np.float16 and z["conf"].dtype == np.float16
    # Video frames and stills (another size) never share a window; sessions neither.
    by_stem = {f.stem: f for f in scene.frames}
    for call in fake.calls:
        assert len({(by_stem[s].w, by_stem[s].session) for s in call}) == 1
    assert da3.run(data, window=6, overlap=3, process_res=PROCESS_RES)["computed"] == 0
    assert da3.run(data, window=4, overlap=2, process_res=PROCESS_RES)["computed"] == 20


def test_fuse_frame_inverse_variance_and_gating() -> None:
    d_s = np.array([2.0, 2.0, 2.0, 0.0, 2.0, 2.0])
    conf = np.array([2, 2, 1, 0, 2, 0], np.uint8)
    s_s = np.full(6, 0.01)
    d_m = np.array([2.02, 3.0, 3.0, 4.0, 0.0, 2.5])
    s_m = np.full(6, 0.01)
    depth, sigma, c = fusion.fuse_frame(d_s, conf, s_s, d_m, s_m, "fused", min_conf=1, gate=3.0)
    np.testing.assert_allclose(depth, [2.01, 2.0, 0.0, 4.0, 2.0, 2.5], rtol=1e-6)
    np.testing.assert_allclose(sigma, [0.01 / np.sqrt(2), 0.01, 0.0, 0.01, 0.01, 0.01], rtol=1e-6)
    assert c == {"sensor": 1, "da3": 2, "both": 1, "gated": 1, "masked": 1}

    # A 3 cm disagreement is within 3 sigma of the difference (4.2 cm), not of sigma_f (2.1 cm).
    one = (np.array([2.0]), np.array([2], np.uint8), np.array([0.01]), np.array([2.03]), np.array([0.01]))
    assert fusion.fuse_frame(*one)[2]["both"] == 1
    assert fusion.fuse_frame(*one, gate_sigma="fused")[2]["gated"] == 1

    depth, sigma, _ = fusion.fuse_frame(d_s, conf, s_s, d_m, s_m, "da3_holes_only")
    np.testing.assert_allclose(depth, [2.0, 2.0, 2.0, 4.0, 2.0, 2.5])
    depth, sigma, _ = fusion.fuse_frame(d_s, conf, s_s, d_m, s_m, "lidar_only")
    np.testing.assert_allclose(depth, [2.0, 2.0, 2.0, 0.0, 2.0, 0.0])
    np.testing.assert_allclose(sigma, [0.01, 0.01, 0.01, 0.0, 0.01, 0.0])
    depth, _, _ = fusion.fuse_frame(None, None, None, d_m, s_m, "fused")  # a still
    np.testing.assert_allclose(depth, [2.02, 3.0, 3.0, 4.0, 0.0, 2.5])
    with pytest.raises(ValueError):
        fusion.fuse_frame(d_s, conf, s_s, d_m, s_m, "both")


def test_robust_rel_recovers_noise() -> None:
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.5, 4.0, 20000)
    pred = ref * (1 + 0.03 * rng.normal(size=ref.size))
    pred[:500] *= 2  # outliers barely move it
    assert fusion.robust_rel(pred, ref) == pytest.approx(0.03, rel=0.05)
    assert fusion.robust_rel(ref * 1.1, ref) == pytest.approx(0.1, rel=1e-6)  # a scale error is error


def frames_and_truth(data: Path):
    scene = Scene.load(data)
    for f in scene.frames:
        assert f.prior_path is not None
        z = np.load(f.prior_path)
        h, w = z["depth"].shape
        yield f, z, true_depth(f.c2w, scaled_k(f.K, f.w, f.h, w, h), w, h)


def test_keys_dtypes_and_old_keys_intact(fused_data) -> None:
    data, before, report = fused_data
    for f, z, _ in frames_and_truth(data):
        old = before[f.stem]
        assert set(z.files) == {"depth", "normal", *(k for m in fusion.MODES for k in fusion.KEYS[m])}
        np.testing.assert_array_equal(z["depth"], old["depth"])
        np.testing.assert_array_equal(z["normal"], old["normal"])
        for m in fusion.MODES:
            kd, ks = fusion.KEYS[m]
            assert z[kd].dtype == np.float16 and z[ks].dtype == np.float16
            assert z[kd].shape == z["depth"].shape == z[ks].shape
            assert ((z[kd] > 0) == (z[ks] > 0)).all()  # 0 in both = no target
    assert report["fused"]["keys"] == ["fused", "sigma"]
    assert json.loads((data / "prior" / "fusion.json").read_text())["lidar_only"]["pixels"]["da3"] == 0


def test_fused_fills_holes_and_beats_both_sources(fused_data) -> None:
    data, _, report = fused_data
    hole_err, hole_raw, hole_cover = [], [], []
    both_f, both_s, both_m = [], [], []
    for f, z, truth in frames_and_truth(data):
        h, w = truth.shape
        fused = z["fused"].astype(np.float32)
        pred = da3.load(data, f.stem)
        assert pred is not None
        raw = cv2.resize(pred[0], (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
        lidar = load_depth(f, w, h)
        if lidar is None:  # a still: DA3 alone, scaled to the mesh
            ok = fused > 0
            assert ok.mean() > 0.95
            assert np.median(np.abs(fused[ok] - truth[ok]) / truth[ok]) < 0.02
            assert not z["fused_lidar_only"].any()
            continue
        d_s, conf = (x.numpy() for x in lidar)
        hole = d_s == 0
        hole_cover.append((fused[hole] > 0).mean())
        ok = hole & (fused > 0)
        hole_err.append(np.abs(fused[ok] - truth[ok]) / truth[ok])
        hole_raw.append(np.abs(raw[ok] - truth[ok]) / truth[ok])
        both = (d_s > 0) & (conf == 2) & (fused > 0) & (f.stem != OUTLIER_STEM)
        both_f.append(fused[both] - truth[both])
        both_s.append(d_s[both] - truth[both])
        both_m.append(raw[both] / DA3_SCALE - truth[both])  # DA3 at the right scale, its noise only
    assert np.mean(hole_cover) > 0.97
    err, raw_err = np.concatenate(hole_err), np.concatenate(hole_raw)
    assert np.median(err) < 0.015 < 0.05 < np.median(raw_err)  # scale-aligned DA3 in the holes
    rmse = {
        k: float(np.sqrt(np.mean(np.concatenate(v) ** 2)))
        for k, v in (("fused", both_f), ("lidar", both_s), ("da3", both_m))
    }
    assert rmse["fused"] < 0.9 * min(rmse["lidar"], rmse["da3"]), rmse
    assert report["fused"]["da3"]["scale"]["median"] == pytest.approx(1 / DA3_SCALE, rel=0.02)
    assert report["fused"]["da3"]["frames_dropped"] == 0


def test_gating_catches_the_injected_outlier(fused_data) -> None:
    data, _, report = fused_data
    scene = Scene.load(data)
    f = next(f for f in scene.frames if f.stem == OUTLIER_STEM)
    assert f.prior_path is not None
    z = np.load(f.prior_path)
    h, w = z["depth"].shape
    lidar = load_depth(f, w, h)
    assert lidar is not None
    d_s, conf = (x.numpy() for x in lidar)
    # The fake's patch [10:30, 10:40] at 56x42, in the prior's pixels (with a margin for resampling).
    s = w / PROCESS_RES
    patch = np.zeros((h, w), bool)
    patch[round(12 * s) : round(28 * s), round(12 * s) : round(38 * s)] = True
    fused = z["fused"].astype(np.float32)
    sure = patch & (d_s > 0) & (conf == 2)
    unsure = patch & (d_s > 0) & (conf == 1)
    assert sure.sum() > 50 and unsure.sum() > 0
    np.testing.assert_allclose(fused[sure], d_s[sure], rtol=1e-3)  # the sensor wins
    assert (fused[unsure] == 0).all()  # low confidence and a disagreement: no target
    assert report["fused"]["pixels"]["gated"] >= sure.sum()
    assert report["fused"]["pixels"]["masked"] >= unsure.sum()


def test_lidar_only_and_holes_only_modes(fused_data) -> None:
    data, _, _ = fused_data
    table = json.loads((data / "prior" / "sigma.json").read_text())
    for f, z, _ in frames_and_truth(data):
        h, w = z["depth"].shape
        lidar = load_depth(f, w, h)
        if lidar is None:
            continue
        d_s, conf = (x.numpy() for x in lidar)
        valid = (d_s > 0) & (conf >= 1)
        lo = z["fused_lidar_only"].astype(np.float32)
        np.testing.assert_allclose(lo[valid], d_s[valid], rtol=1e-3)
        assert not lo[~valid].any()
        edge = calibrate.edge_mask(z["depth"].astype(np.float32), radius=max(1, round(w / 256)))
        expect = calibrate.lidar_sigma(table, conf, d_s, edge)
        np.testing.assert_allclose(z["sigma_lidar_only"][valid].astype(np.float32), expect[valid], rtol=1e-3)
        ho = z["fused_da3_holes_only"].astype(np.float32)
        np.testing.assert_allclose(ho[valid], d_s[valid], rtol=1e-3)
        assert (ho[~valid] > 0).mean() > 0.95


def test_load_fused(fused_data) -> None:
    data, _, _ = fused_data
    f = next(f for f in Scene.load(data).frames if f.depth_path is not None)
    got = fusion.load_fused(f, 64, 48)
    assert got is not None
    depth, sigma = got
    assert depth.shape == sigma.shape == (48, 64) and depth.dtype == torch.float32
    assert fusion.load_fused(f, 64, 48, mode="lidar_only") is not None


def test_fuse_needs_priors_and_da3(tmp_path: Path) -> None:
    data = synth.write_dataset(tmp_path / "data")
    with pytest.raises(FileNotFoundError, match="priors"):
        fusion.fuse(data)
    priors.prepare(data)
    with pytest.raises(FileNotFoundError, match="da3"):
        fusion.fuse(data, "fused")
    report = fusion.fuse(data, "lidar_only")  # no DA3 needed; calibrates on the way
    assert (data / "prior" / "sigma.json").is_file() and report["lidar_only"]["da3"] is None
    assert fusion.main(["--data", str(data), "--mode", "lidar_only"]) == 0
