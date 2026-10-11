"""End to end on a synthetic two-room home: ingest -> poses -> fuse -> structure -> eval."""

import json
from pathlib import Path

import numpy as np
import pytest

from scanmodel.cache import Stage, run_stage
from scanmodel.config import load_config
from scanmodel.io import find_bundles
from scanmodel.models.scene import Scene
from scanmodel.stages import eval as eval_stage
from scanmodel.stages import fuse, ingest, poses, structure

from . import synth


@pytest.fixture(scope="module")
def synth_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("repo")
    synth.write_two_rooms(root / "data" / "synth" / "raw")
    return root


@pytest.fixture(scope="module")
def built(synth_repo: Path) -> Path:
    """The project folder (data/synth) after the stages ran."""
    cfg = load_config(synth_repo, project="synth")
    ingest.run(find_bundles(cfg.raw), cfg.stage_dir("ingest"))
    poses.run(cfg.stage_dir("ingest"), cfg.scene_dir / "overrides.json", cfg.stage_dir("poses"))
    fuse.run(cfg.stage_dir("poses"), cfg.stage_dir("fuse"), color=False)
    structure.run(
        cfg.stage_dir("poses"),
        cfg.stage_dir("fuse"),
        cfg.scene_dir / "overrides.json",
        cfg.scene_dir / "model.json",
    )
    eval_stage.run(
        cfg.scene_dir / "model.json", cfg.stage_dir("fuse"), cfg.stage_dir("eval"), cfg.stage_dir("poses")
    )
    return cfg.data


def _scene(data: Path) -> Scene:
    return Scene.model_validate_json((data / "scene" / "model.json").read_text())


def test_ingest_reports_are_clean(built: Path) -> None:
    for rep in (built / "derived/ingest").glob("*/report.json"):
        r = json.loads(rep.read_text())
        assert r["ok"], r
        assert r["depth_consistency_m"] < 0.01, r


def test_scene_frame_recovered(built: Path) -> None:
    p = json.loads((built / "derived/poses/poses.json").read_text())
    # The hall session's first camera started at true (1.5, 0.5, 1.2) -> scene (0, 0.5, 1.2).
    first = json.loads((built / "derived/poses/frames.jsonl").read_text().splitlines()[0])
    np.testing.assert_allclose(np.array(first["camera_to_world"])[:3, 3], [0.0, 0.5, 1.2], atol=0.03)
    assert p["origin_door"] is not None


def test_walls_match_ground_truth(built: Path) -> None:
    scene = _scene(built)
    assert len(scene.rooms) == 2
    # Expected inner faces in the scene frame: (axis, coordinate) per face.
    hx0, hx1, hy0, hy1 = synth.HALL
    lx0, lx1, _, _ = synth.LIVING
    o = synth.DOOR_X
    truth = {("x", hx0 - o), ("x", hx1 - o), ("y", hy0), ("y", hy1), ("x", lx0 - o), ("x", lx1 - o)}
    faces: list[tuple[str, float]] = []
    for w in scene.walls:
        n = w.normal()
        axis = "x" if abs(n[0]) > 0.99 else "y"
        assert abs(n[0]) > 0.999 or abs(n[1]) > 0.999, f"{w.id} not axis-aligned: {n}"
        c = (np.array(w.start) + np.array(w.end)) / 2
        for side in (1, -1):
            faces.append((axis, float((c + side * n * w.thickness / 2)[0 if axis == "x" else 1])))
    for axis, coord in truth:
        err = min(abs(c - coord) for a, c in faces if a == axis)
        assert err < 0.01, f"face {axis}={coord:.3f} off by {err * 100:.1f} cm"
    shared = [w for w in scene.walls if w.thickness_measured]
    assert len(shared) == 1 and abs(shared[0].thickness - synth.SHARED_T) < 0.01, shared


def test_openings_and_objects(built: Path) -> None:
    scene = _scene(built)
    kinds = sorted(o.kind.value for o in scene.openings)
    # The synthetic doors are empty holes (no leaf in the scan), so both are doorways; the shared one is
    # merged across both sessions.
    assert kinds == ["opening", "opening", "window"], kinds
    front = min(
        scene.openings,
        key=lambda o: np.linalg.norm(
            np.array(scene.wall(o.wall).start) + o.offset * scene.wall(o.wall).direction()
        ),
    )
    assert front.kind.value == "opening"
    assert [o.category for o in scene.objects] == ["storage"]


def test_eval_passes(built: Path) -> None:
    m = json.loads((built / "derived/eval/metrics.json").read_text())
    s = m["summary"]
    assert s["pass"], s
    assert s["p95_m"] < 0.02
    assert s["openings_suspect"] == 0, m["openings"]
    assert (built / "derived/eval/report.html").is_file()
    # floors: every session sees each room's (level) floor at the same height
    f = m["floors"]
    seen = {room: r["sessions"] for room, r in f["rooms"].items()}
    assert any(len(v) >= 2 for v in seen.values()), seen
    assert s["spread_max_m"] is not None and s["spread_max_m"] < 0.01, seen
    assert s["level_offset_max_m"] is None or s["level_offset_max_m"] < 0.01, f["level_offset_m"]
    assert "Floors" in (built / "derived/eval/report.html").read_text()
    # doubled surfaces: the sessions' own walls agree, no split
    assert m["doubled"]["pairs_compared"] > 0 and not s["split_warning"], m["doubled"]
    assert "Doubled surfaces: none" in (built / "derived/eval/report.html").read_text()


def test_overrides_apply_last(built: Path) -> None:
    scene = _scene(built)
    wall, obj = scene.walls[0].id, scene.objects[0].id
    ov = {
        "version": 1,
        "patch": {wall: {"thickness": 0.3}},
        "remove": [obj],
        "room_names": {scene.rooms[0].id: "Hall"},
    }
    path = built / "scene" / "overrides.json"
    path.write_text(json.dumps(ov))
    try:
        structure.run(built / "derived/poses", built / "derived/fuse", path, built / "scene/model.json")
        s2 = _scene(built)
        assert s2.wall(wall).thickness == 0.3
        assert not s2.objects
        assert s2.rooms[0].name == "Hall"
    finally:
        path.unlink()


def test_one_session_is_split_into_rooms(tmp_path: Path) -> None:
    synth.write_one_session(tmp_path / "data" / "synth" / "raw")
    cfg = load_config(tmp_path, project="synth")
    ingest.run(find_bundles(cfg.raw), cfg.stage_dir("ingest"))
    poses.run(cfg.stage_dir("ingest"), cfg.scene_dir / "overrides.json", cfg.stage_dir("poses"))
    fuse.run(cfg.stage_dir("poses"), cfg.stage_dir("fuse"), color=False)
    structure.run(
        cfg.stage_dir("poses"),
        cfg.stage_dir("fuse"),
        cfg.scene_dir / "overrides.json",
        cfg.scene_dir / "model.json",
    )
    rooms = {r.id: r for r in _scene(cfg.data).rooms}
    assert sorted(rooms) == ["room-s01-home-bedroom", "room-s01-home-living-room"], rooms
    assert rooms["room-s01-home-bedroom"].name == "bedroom"
    assert rooms["room-s01-home-living-room"].name == "living room"
    for rid, (x0, x1, y0, y1) in (
        ("room-s01-home-bedroom", synth.HALL),
        ("room-s01-home-living-room", synth.LIVING),
    ):
        poly = np.array(rooms[rid].polygon)
        area = 0.5 * abs(
            np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))
        )
        assert abs(area - (x1 - x0) * (y1 - y0)) < 0.3, (rid, area)


def test_stage_cache_skips_unchanged(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    src.write_text("a")
    calls: list[int] = []
    st = Stage("t", inputs=lambda: [src], run=lambda: calls.append(1) or {}, root=tmp_path)
    assert run_stage(st, tmp_path / "out")
    assert not run_stage(st, tmp_path / "out")
    src.write_text("b")
    assert run_stage(st, tmp_path / "out")
    assert len(calls) == 2
