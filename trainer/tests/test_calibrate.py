import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gs_trainer import calibrate, synth
from gs_trainer.data import Scene

SIGMA_2 = 0.015  # injected LiDAR noise at confidence 2 (m)
SIGMA_1 = 0.040  # and at confidence 1


def add_lidar_noise(data: Path, seed: int = 1) -> None:
    """Gaussian noise on every LiDAR reading: SIGMA_2 at confidence 2, SIGMA_1 at confidence 1."""
    rng = np.random.default_rng(seed)
    for p in sorted((data / "depth").glob("*.png")):
        mm = np.asarray(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), np.float64)
        conf = cv2.imread(str(data / "confidence" / p.name), cv2.IMREAD_UNCHANGED)
        sigma = np.where(conf == 2, SIGMA_2, np.where(conf == 1, SIGMA_1, 0.0)) * 1000
        noisy = np.where(mm > 0, mm + rng.normal(size=mm.shape) * sigma, 0)
        cv2.imwrite(str(p), np.where(mm > 0, np.clip(noisy, 1, 65535), 0).round().astype(np.uint16))


@pytest.fixture(scope="module")
def noisy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data = synth.write_dataset(tmp_path_factory.mktemp("cal") / "data", n_frames=24, size=(128, 96))
    add_lidar_noise(data)
    return data


def test_recovers_injected_noise(noisy: Path) -> None:
    table = calibrate.calibrate(noisy)  # no priors: mesh depth raycast on the fly
    assert json.loads((noisy / "prior" / "sigma.json").read_text()) == table
    n2 = np.asarray(table["n"]["2"])
    s2 = np.asarray(table["sigma"]["2"])
    populated = n2 >= calibrate.MIN_PIXELS
    assert populated.sum() >= 3
    np.testing.assert_allclose(s2[populated], SIGMA_2, rtol=0.15)
    assert table["flat"]["1"]["sigma"] == pytest.approx(SIGMA_1, rel=0.2)
    assert abs(table["flat"]["2"]["bias"]) < 0.003
    # The box room's corners are continuous in depth: hardly any edges, and no edge penalty without one.
    assert table["edge_factor"]["2"] >= 1.0


def test_priors_and_raycast_agree(noisy: Path, tmp_path: Path) -> None:
    from gs_trainer import priors

    scene = Scene.load(noisy)
    frames = calibrate.sample_frames(scene, 4)
    sizes = [(64, 48)] * len(frames)
    cast = calibrate.mesh_depths(scene, frames, sizes)
    copy = synth.write_dataset(tmp_path / "data", n_frames=24, size=(128, 96))
    priors.prepare(copy, side=128)
    scene2 = Scene.load(copy)
    from_prior = calibrate.mesh_depths(scene2, calibrate.sample_frames(scene2, 4), sizes)
    for a, b in zip(cast, from_prior, strict=True):
        assert a.shape == (48, 64)
        np.testing.assert_allclose(a, b, rtol=0.02)


def test_sample_frames_spreads_over_video() -> None:
    frames = [SimpleNamespace(depth_path=None if i % 5 == 0 else Path(f"{i}.png")) for i in range(50)]
    got = calibrate.sample_frames(SimpleNamespace(frames=frames), 10)  # pyright: ignore[reportArgumentType]
    assert len(got) == 10 and all(f.depth_path is not None for f in got)
    assert got[0] is frames[1] and got[-1] is frames[49]


def test_edge_mask_finds_steps_not_holes_or_slopes() -> None:
    d = np.full((20, 20), 2.0, np.float32)
    d[:, 10:] = 3.0  # a 1 m step
    d += np.linspace(0, 0.02, 20)[None]  # a gentle slope
    d[5, 2] = 0.0  # a hole
    e = calibrate.edge_mask(d, radius=1)
    assert e[:, 8:12].all() and not e[:, :8].any() and not e[:, 12:].any()
    assert not e[5, 2]


def test_lidar_sigma_lookup() -> None:
    table = {
        "d_edges": [0.0, 1.0, 2.0, 5.0],
        "sigma": {"1": [0.02, 0.03, 0.05], "2": [0.01, 0.012, 0.02]},
        "edge_factor": {"1": 2.0, "2": 3.0},
    }
    conf = np.array([2, 2, 2, 1, 0, 2], np.uint8)
    depth = np.array([0.5, 1.5, 9.0, 0.5, 0.5, 0.5])
    edge = np.array([False, False, False, False, False, True])
    s = calibrate.lidar_sigma(table, conf, depth, edge)
    np.testing.assert_allclose(s, [0.01, 0.012, 0.02, 0.02, 0.02, 0.03], rtol=1e-6)
    # Confidence 0 isn't in the table: it gets the worst level's row (here confidence 1).


def test_main_prints_table(noisy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert calibrate.main(["--data", str(noisy), "--frames", "6"]) == 0
    out = capsys.readouterr().out
    assert "sigma/p90 mm" in out and "sigma.json" in out
