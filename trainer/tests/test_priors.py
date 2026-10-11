import json
from pathlib import Path

import numpy as np
import pytest

from gs_trainer import synth
from gs_trainer.data import Scene, load_depth, load_prior
from gs_trainer.priors import main, prepare


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return synth.write_dataset(tmp_path / "data")


def test_prepare_writes_every_frame(data: Path) -> None:
    meta = prepare(data)
    assert meta["frames"] == meta["computed"] == 20 and meta["skipped"] == 0
    assert json.loads((data / "prior" / "meta.json").read_text())["side"] == 640
    scene = Scene.load(data)
    assert all(f.prior_path is not None for f in scene.frames)
    for f in scene.frames:
        assert f.prior_path is not None
        z = np.load(f.prior_path)
        assert z["depth"].dtype == np.float16 and z["depth"].shape == (f.h, f.w)  # never enlarged
        assert z["normal"].dtype == np.int8 and z["normal"].shape == (f.h, f.w, 3)


def test_prior_depth_matches_geometry_and_lidar(data: Path) -> None:
    prepare(data)
    scene = Scene.load(data)
    texture = synth.Texture(np.random.default_rng(0))
    for i in scene.train + scene.interp + scene.extrap:
        f = scene.frames[i]
        got = load_prior(f, f.w, f.h)
        assert got is not None
        depth, normal = (x.numpy() for x in got)
        # Exactly the analytic z-depth of the box (up to float16).
        _, z = synth.render(f.c2w, f.K, f.w, f.h, texture)
        assert (depth > 0).all()
        np.testing.assert_allclose(depth, z, rtol=2e-3)
        # Box normals: unit, axis-aligned, facing the camera (inwards).
        np.testing.assert_allclose(np.linalg.norm(normal, axis=-1), 1.0, atol=1e-3)
        assert (np.abs(normal).max(-1) > 0.999).all()
        u, v = np.meshgrid(np.arange(f.w) + 0.5, np.arange(f.h) + 0.5)
        dirs = np.stack([(u - f.K[0, 2]) / f.K[0, 0], (v - f.K[1, 2]) / f.K[1, 1], np.ones_like(u)], -1)
        assert ((dirs @ f.c2w[:3, :3].T * normal).sum(-1) < 0).all()

        lidar = load_depth(f, f.w, f.h)
        if lidar is None:
            assert f.camera_idx == 1
            continue
        d, conf = (x.numpy() for x in lidar)
        ok = (d > 0) & (conf == 2)
        rel = np.abs(depth[ok] - d[ok]) / d[ok]
        # LiDAR is half resolution and in mm: its pixel centres sit up to 1 image pixel off.
        assert np.median(rel) < 0.01 and np.percentile(rel, 95) < 0.04


def test_prepare_resumes_and_rebuilds_on_new_side(data: Path) -> None:
    prepare(data)
    again = prepare(data)
    assert again["computed"] == 0 and again["skipped"] == 20
    first = sorted((data / "prior").glob("*.npz"))[0]
    first.unlink()
    assert prepare(data)["computed"] == 1
    assert main(["--data", str(data), "--side", "32"]) == 0
    meta = json.loads((data / "prior" / "meta.json").read_text())
    assert meta["side"] == 32 and meta["computed"] == 20
    assert np.load(first)["depth"].shape[1] <= 32
