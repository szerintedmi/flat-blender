"""Fused mesh vs per-frame LiDAR (`scanmodel.meshcheck`), and the pose source the stages after `poses` use: refined
poses by default, `--poses posegraph` as before."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest

from scanmodel import meshcheck
from scanmodel.models import bundle as b
from scanmodel.models.bundle import Intrinsics
from scanmodel.models.derived import PosedImage, Poses, SessionPoses

from .synth import look_at

K = Intrinsics(fx=200.0, fy=200.0, cx=128.0, cy=96.0, width=256, height=192)


def _room(path: Path) -> Path:
    """A 4 x 5 x 2.5 m box, seen from inside."""
    box = o3d.geometry.TriangleMesh.create_box(4.0, 5.0, 2.5)
    box.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, box)
    return path


def _lidar(mesh: Path, c2w: np.ndarray) -> np.ndarray:
    """The z-depth LiDAR would measure at `c2w` (256 x 192)."""
    im = PosedImage(
        id="x/0", session="x", kind="frame", file="video.mov#0", camera_to_world=c2w.tolist(), intrinsics=K
    )
    zero = np.ones((K.height, K.width))
    d, r, _ = meshcheck.frame_residuals(meshcheck.scene_of(mesh), im, zero, None, max_depth=99.0)
    return (d + r).reshape(K.height, K.width)


def _frames(root: Path, mesh: Path, n: int = 6) -> tuple[Path, list[np.ndarray]]:
    """A poses folder of `n` depth frames facing the room's +y wall, LiDAR from the true mesh."""
    raw = root / "raw" / "s1"
    for sub in (b.DEPTH_DIR, b.CONFIDENCE_DIR):
        (raw / sub).mkdir(parents=True)
    poses = root / "poses"
    poses.mkdir()
    sessions = [
        SessionPoses.model_validate(
            {"session": "s1", "raw_dir": str(raw), "session_to_scene": np.eye(4).tolist()}
        )
    ]
    (poses / "poses.json").write_text(Poses(yaw_deg=0.0, floor_z=0.0, sessions=sessions).model_dump_json())
    truth, lines = [], []
    for i in range(n):
        c2w = look_at(np.array([1.0 + 0.3 * i, 2.0, 1.4]), np.array([1.2 + 0.3 * i, 5.0, 1.2]))
        truth.append(c2w)
        cv2.imwrite(
            str(raw / b.DEPTH_DIR / f"{i:06d}.png"), (_lidar(mesh, c2w) * 1000).round().astype(np.uint16)
        )
        cv2.imwrite(str(raw / b.CONFIDENCE_DIR / f"{i:06d}.png"), np.full((K.height, K.width), 2, np.uint8))
        im = PosedImage(
            id=f"s1/{i}", session="s1", kind="frame", file=f"video.mov#{i}", camera_to_world=c2w.tolist(),
            intrinsics=K.scaled(1920, 1440), tracking=b.TrackingState.NORMAL,
            depth=f"{b.DEPTH_DIR}/{i:06d}.png", confidence=f"{b.CONFIDENCE_DIR}/{i:06d}.png",
        )  # fmt: skip
        lines.append(im.model_dump_json(exclude_none=True) + "\n")
    (poses / "frames.jsonl").write_text("".join(lines))
    (poses / "stills.jsonl").write_text("")
    return poses, truth


def _moved(poses: Path, out: Path, shift: np.ndarray) -> Path:
    """A copy of a poses folder with every camera moved by `shift` (metres, world)."""
    shutil.copytree(poses, out)
    lines = []
    for line in (poses / "frames.jsonl").read_text().splitlines():
        im = PosedImage.model_validate_json(line)
        c2w = np.array(im.camera_to_world)
        c2w[:3, 3] += shift
        lines.append(im.model_copy(update={"camera_to_world": c2w.tolist()}).model_dump_json() + "\n")
    (out / "frames.jsonl").write_text("".join(lines))
    return out


def test_mesh_agrees_with_lidar_at_its_own_poses(tmp_path: Path) -> None:
    mesh = _room(tmp_path / "room.ply")
    poses, _ = _frames(tmp_path, mesh)
    r = meshcheck.mesh_vs_lidar(mesh, poses)
    a = r["all"]
    assert isinstance(a, dict) and r["frames"] == 6
    assert a["abs_median_mm"] < 1.0 and a["within_1cm"] > 0.99  # mm quantisation of the PNG only
    assert a["normal_median_deg"] < 2.0  # flat walls: LiDAR and mesh normals agree

    # Cameras 3 cm closer to the wall they face (+y): the mesh is 3 cm nearer than their LiDAR says.
    moved = _moved(poses, tmp_path / "moved", np.array([0.0, 0.03, 0.0]))
    m = meshcheck.mesh_vs_lidar(mesh, moved)["all"]
    assert isinstance(m, dict)
    assert m["bias_median_mm"] == pytest.approx(-30.0, abs=3.0) and m["within_1cm"] < 0.5
    assert m["normal_median_deg"] < 2.0  # a translation leaves normals alone


def test_lidar_normals_of_a_tilted_plane() -> None:
    v, u = np.mgrid[0:20, 0:30] + 0.5
    rays = np.stack([(u - 15) / 200, (v - 10) / 200, np.ones_like(u)], -1)  # ~1 cm per pixel at 2 m
    n = np.array([0.0, -np.sin(0.3), -np.cos(0.3)])  # facing the camera, tilted 0.3 rad
    depth = 2.0 / -(rays @ n)  # the plane n.x = -2
    normals, ok = meshcheck.lidar_normals(rays * depth[..., None], np.ones(depth.shape, bool))
    assert ok[1:-1, 1:-1].all() and not ok[0].any()
    np.testing.assert_allclose(normals[ok], np.broadcast_to(n, normals[ok].shape), atol=1e-6)


def test_refined_poses_are_the_default_and_posegraph_keeps_the_old_keys(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every stage after `poses` reads derived/refine/ by default; `--poses posegraph` gives the stage keys
    (inputs, params, code) of the pipeline before refine was the default. Fuse, splat and meshcheck stamp the
    pose source, and splat refuses fused points from other poses than its own."""
    from typer.testing import CliRunner

    from scanmodel import cli
    from scanmodel.cache import Stage, run_stage
    from scanmodel.config import load_config
    from scanmodel.points import WALLS_DIR
    from scanmodel.stages import fuse, structure

    from .test_splat import _scan  # pyright: ignore[reportPrivateUsage]

    tmp_path = repo
    (tmp_path / "settings.toml").write_text('project = "p"\n')
    data = tmp_path / "data" / "p"
    derived = data / "derived"
    _scan(derived)  # derived/poses, derived/fuse (a box), derived/raw
    box = Path(shutil.move(derived / "fuse", tmp_path / "box"))
    (derived / "poses" / "manifest.json").write_text(json.dumps({"params": {"pose_graph": True}}))
    refine = _moved(derived / "poses", derived / "refine", np.array([0.0, 0.02, 0.0]))
    (refine / "refine.json").write_text("{}")
    (refine / "manifest.json").unlink()
    (data / "scene").mkdir()
    cfg = load_config()
    assert cfg.poses == "refine"
    assert cli.pose_source(cfg.stage_dir("poses")) == "arkit-posegraph"
    assert cli.pose_source(cfg.stage_dir("refine")) == "refine"

    # The opt-out's keys are the old default stages' (as `scanmodel all` built them before refine was the default).
    p, f = derived / "poses", derived / "fuse"
    legacy = {
        "fuse": Stage(
            "fuse",
            inputs=lambda: [p / "poses.json", p / "frames.jsonl"],
            run=dict,
            root=tmp_path,
            params={"voxel": 0.01, "min_confidence": 1, "color": True},
            code=["stages/fuse.py", "points.py"],
        ),
        "structure": Stage(
            "structure",
            inputs=lambda: [
                p / "poses.json",
                f / "points.ply",
                p / WALLS_DIR,
                data / "scene/overrides.json",
            ],
            run=dict,
            root=tmp_path,
            params=structure.Params().__dict__,
            code=["stages/structure.py", "points.py", "roomplan.py", "rooms.py", "boxes.py", "ownership.py"],
        ),
        "build": Stage(
            "build",
            inputs=lambda: [
                data / "scene/model.json",
                p / "stills.jsonl",
                f / "dense.ply",
                tmp_path / "blender",
            ],
            run=dict,
            root=tmp_path,
            code=["stages/scanmesh.py", "boxes.py"],
        ),
    }
    stages = {
        "posegraph": [cli._fuse_stage(cfg, "posegraph"), cli._structure_stage(cfg, "posegraph"), cli._build_stage(cfg, "posegraph")],  # pyright: ignore[reportPrivateUsage]
        "refine": [cli._fuse_stage(cfg, "refine"), cli._structure_stage(cfg, "refine"), cli._build_stage(cfg, "refine")],  # pyright: ignore[reportPrivateUsage]
    }  # fmt: skip
    for st in stages["posegraph"]:
        assert st.key() == legacy[st.name].key(), st.name
    for st in stages["refine"]:
        assert st.params["poses"] == "refine" and st.key()["inputs"] != legacy[st.name].key()["inputs"], (
            st.name
        )

    calls: list[tuple[Path, Path]] = []

    def fake_fuse(poses_dir: Path, out: Path, **_: object) -> dict[str, object]:
        calls.append((poses_dir, out))
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(box / "dense.ply", out / "dense.ply")
        shutil.copy(box / "points.ply", out / "points.ply")
        return {}

    monkeypatch.setattr(fuse, "run", fake_fuse)
    runner = CliRunner()
    r = runner.invoke(cli.app, ["splat", "--select", "time"])
    assert r.exit_code != 0 and "scanmodel fuse" in " ".join(
        r.output.replace("│", " ").split()
    )  # nothing fused yet

    # Default: fused from derived/refine/ into derived/fuse/, the dataset on the refined poses.
    r = runner.invoke(cli.app, ["fuse"])
    assert r.exit_code == 0, r.output
    assert calls[-1] == (refine, f)
    m = json.loads((f / "manifest.json").read_text())
    assert m["params"]["poses"] == "refine" and m["summary"]["pose_source"] == "refine"
    assert cli.fused_from(cfg) == ("refine", "refine")
    r = runner.invoke(cli.app, ["splat", "--select", "time"])
    assert r.exit_code == 0, r.output
    t = json.loads((data / "splat/datasets/default/transforms.json").read_text())
    assert (t["pose_source"], t["mesh_pose_source"]) == ("refine", "refine")
    e = json.loads((data / "splat/datasets/default/manifest.json").read_text())
    assert e["params"]["poses"] == "refine" and "fuse" not in e["params"]
    v = next(x for x in t["frames"] if x["camera"] == "video")  # the frames' poses are the refined ones
    refined = {
        f"{im.session}_{int(im.id.split('/')[1]):06d}": np.array(im.camera_to_world)
        for line in (refine / "frames.jsonl").read_text().splitlines()
        for im in [PosedImage.model_validate_json(line)]
    }
    np.testing.assert_allclose(
        np.array(v["transform_matrix"])[:3, 3], refined[Path(v["file_path"]).stem][:3, 3]
    )

    # Mixing poses and fused points is refused.
    r = runner.invoke(cli.app, ["splat", "--poses", "posegraph", "--dataset", "d2", "--select", "time"])
    assert r.exit_code != 0 and "scanmodel fuse --poses posegraph" in " ".join(
        r.output.replace("│", " ").split()
    )
    assert runner.invoke(cli.app, ["fuse", "--poses", "bundle"]).exit_code != 0

    # Opt-out (here through settings.toml): the pose graph's poses, the old params, the old stamps.
    (tmp_path / "settings.toml").write_text('project = "p"\nposes = "posegraph"\n')
    r = runner.invoke(cli.app, ["fuse"])
    assert r.exit_code == 0, r.output
    assert calls[-1] == (p, f)
    m = json.loads((f / "manifest.json").read_text())
    assert m["params"] == {"voxel": 0.01, "min_confidence": 1, "color": True}
    assert m["summary"]["pose_source"] == "arkit-posegraph"
    assert m["inputs"] == legacy["fuse"].key()["inputs"]
    assert not run_stage(cli._fuse_stage(load_config(), "posegraph"), f)  # pyright: ignore[reportPrivateUsage]
    r = runner.invoke(cli.app, ["splat", "--dataset", "d2", "--select", "time"])
    assert r.exit_code == 0, r.output
    t = json.loads((data / "splat/datasets/d2/transforms.json").read_text())
    assert (t["pose_source"], t["mesh_pose_source"]) == ("arkit-posegraph", "arkit-posegraph")
    e = json.loads((data / "splat/datasets/d2/manifest.json").read_text())
    assert "poses" not in e["params"]  # the old default dataset's params
