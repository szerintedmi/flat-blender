import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from gs_trainer.data import GL_CV, Images, Scene, load_depth, load_prior, scaled_size, viewmat


@pytest.fixture(scope="module")
def scene(synth_dataset: Path) -> Scene:
    return Scene.load(synth_dataset)


def project(scene: Scene, i: int, k: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Init points seen by frame i: (pixel (n, 2) float, camera z (n,), point index (n,))."""
    vm = viewmat(scene.frames[i].c2w).double().numpy()
    cam = scene.points.astype(np.float64) @ vm[:3, :3].T + vm[:3, 3]
    uv = cam @ k.T
    uv = uv[:, :2] / uv[:, 2:]
    margin = 2  # away from the border, where nearest/area resampling has no neighbours on one side
    ok = (cam[:, 2] > 0.1) & (uv[:, 0] > margin) & (uv[:, 0] < w - margin)
    ok &= (uv[:, 1] > margin) & (uv[:, 1] < h - margin)
    return uv[ok], cam[ok, 2], np.flatnonzero(ok)


def test_scene_split_and_frames(scene: Scene, synth_dataset: Path) -> None:
    split = json.loads((synth_dataset / "split.json").read_text())
    assert len(scene.frames) == 16 + 4
    assert scene.interp == [0, 8]
    assert [scene.frames[i].stem for i in scene.interp] == split["interp"]
    assert scene.extrap == [16, 17, 18, 19]
    assert [scene.frames[i].stem for i in scene.extrap] == split["extrap"]
    assert sorted(scene.train + scene.interp) == list(range(16))
    assert all(scene.frames[i].session == "synth-lap" for i in scene.extrap)

    stills = [f for f in scene.frames if f.camera_idx == 1]
    assert len(stills) == 2
    assert all(f.depth_path is None and f.confidence_path is None for f in stills)
    assert all((f.w, f.h) == (96, 72) for f in stills)
    video = [f for f in scene.frames if f.camera_idx == 0]
    assert all(f.depth_path is not None and f.depth_path.is_file() for f in video)
    assert all(f.prior_path is None for f in scene.frames)
    assert all(f.image_path.is_file() for f in scene.frames)

    assert scene.points.shape == (3000, 3) and scene.points.dtype == np.float32
    assert scene.colors.shape == (3000, 3) and scene.colors.dtype == np.uint8
    centres = np.array([f.c2w[:3, 3] for f in scene.frames[:16]])
    expected = 1.1 * np.linalg.norm(centres - centres.mean(0), axis=1).max()
    assert scene.scene_scale == pytest.approx(expected)


def test_opencv_axes(scene: Scene, synth_dataset: Path) -> None:
    tf = json.loads((synth_dataset / "transforms.json").read_text())
    f = scene.frames[3]
    np.testing.assert_allclose(f.c2w, np.array(tf["frames"][3]["transform_matrix"]) @ GL_CV)
    # Cameras on the ring look outwards (+z away from the room's centre) and stand upright (+y down).
    centre = f.c2w[:3, 3]
    assert np.dot(f.c2w[:2, 2], centre[:2]) > 0
    assert f.c2w[2, 1] < -0.9


def test_dataset_depth_units_and_stationary_cameras(tmp_path: Path) -> None:
    from gs_trainer import synth

    data = synth.write_dataset(tmp_path / "dataset")
    path = data / "transforms.json"
    tf = json.loads(path.read_text())
    tf["depth_unit_scale_factor"] = 0.01
    for entry in tf["frames"]:
        entry["transform_matrix"][0][3] = 0
        entry["transform_matrix"][1][3] = 0
        entry["transform_matrix"][2][3] = 0
    path.write_text(json.dumps(tf))
    scene = Scene.load(data)
    expected = 1.1 * np.linalg.norm(scene.points - scene.points.mean(0), axis=1).max()
    assert scene.scene_scale == pytest.approx(expected)
    assert all(f.depth_scale == 0.01 for f in scene.frames)  # extrap frames inherit it too
    frame = next(f for f in scene.frames if f.depth_path is not None)
    raw = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    assert raw is not None
    loaded = load_depth(frame, raw.shape[1], raw.shape[0])
    assert loaded is not None
    depth, _ = loaded
    np.testing.assert_allclose(depth.numpy(), raw.astype(np.float32) * 0.01)


def test_viewmat(scene: Scene) -> None:
    c2w = scene.frames[5].c2w
    np.testing.assert_allclose(viewmat(c2w).double().numpy() @ c2w, np.eye(4), atol=1e-6)
    t = torch.tensor(c2w, dtype=torch.float32, requires_grad=True)
    vm = viewmat(t)
    assert vm.shape == (4, 4) and vm.dtype == torch.float32
    vm.sum().backward()
    assert t.grad is not None


def test_points_project_onto_their_colour_and_depth(scene: Scene) -> None:
    """A known 3D point (an init point on a wall) lands where the image shows its colour and the LiDAR
    its depth: checks K, the GL->CV conversion, viewmat, the image resize and the depth upsampling."""
    for i in (1, 3, 15, 17):  # video, video, still, extrap
        f = scene.frames[i]
        for max_side in (1000, 40):
            images = Images(scene, [i], max_side)
            img, k = images.get(i)
            w, h = images.size(i)
            assert img.shape == (h, w, 3)
            uv, z, idx = project(scene, i, k.double().numpy(), w, h)
            assert len(idx) > 50
            px = np.floor(uv).astype(int)
            seen = img.numpy()[px[:, 1], px[:, 0]] * 255
            err = np.abs(seen - scene.colors[idx].astype(float)).mean(1)
            assert np.median(err) < 8, (i, max_side, np.median(err))

            got = load_depth(f, w, h)
            if f.camera_idx == 1:
                assert got is None
                continue
            assert got is not None
            depth, conf = got
            d = depth.numpy()[px[:, 1], px[:, 0]]
            ok = (d > 0) & (conf.numpy()[px[:, 1], px[:, 0]] == 2)
            rel = np.abs(d[ok] - z[ok]) / z[ok]
            # The point sits anywhere inside a coarse depth pixel, sampled at its centre: ~1 % on slanted walls.
            assert np.median(rel) < 0.02 and np.percentile(rel, 95) < 0.05, (i, max_side)


def test_depth_shapes_and_values(scene: Scene) -> None:
    f = scene.frames[1]
    got = load_depth(f, 64, 48)
    assert got is not None
    depth, conf = got
    assert depth.shape == (48, 64) and depth.dtype == torch.float32
    assert conf.shape == (48, 64) and conf.dtype == torch.uint8
    assert set(conf.unique().tolist()) <= {0, 1, 2}
    assert (conf == 2).float().mean() > 0.8
    # Holes stay holes, readings are metres inside a 4 x 3 x 2.5 m room.
    valid = depth[depth > 0]
    assert valid.min() > 0.1 and valid.max() < 5.0
    # Nearest upsampling of the native 32 x 24 map: every 2 x 2 block is one reading.
    assert torch.equal(depth[0::2, 0::2], depth[1::2, 1::2])
    small = load_depth(f, 32, 24)
    assert small is not None and torch.equal(small[0], depth[0::2, 0::2])


def test_images_cache_on_and_off_agree(scene: Scene) -> None:
    on = Images(scene, scene.train, 40, cache=True)
    off = Images(scene, scene.train, 40, cache=False)
    auto = Images(scene, scene.train, 40)
    assert on.is_cached and not off.is_cached and auto.is_cached  # a few KB fits anywhere
    for i in scene.train:
        a, ka = on.get(i)
        b, kb = off.get(i)
        assert torch.equal(a, b) and torch.equal(ka, kb)
        assert a.dtype == torch.float32 and a.min() >= 0.0 and a.max() <= 1.0
        assert on.size(i) == (40, 30)


def test_images_scale_k(scene: Scene) -> None:
    i = scene.frames.index(next(f for f in scene.frames if f.camera_idx == 1))
    f = scene.frames[i]
    images = Images(scene, [i], 64)
    _, k = images.get(i)
    assert images.size(i) == scaled_size(f.w, f.h, 64) == (64, 48)
    np.testing.assert_allclose(k.numpy(), np.diag([64 / 96, 48 / 72, 1.0]) @ f.K, rtol=1e-6)
    _, k_full = Images(scene, [i], 4000).get(i)  # never enlarged
    np.testing.assert_allclose(k_full.numpy(), f.K, rtol=1e-6)


def test_no_prior_without_prepare(scene: Scene) -> None:
    assert load_prior(scene.frames[0], 64, 48) is None
