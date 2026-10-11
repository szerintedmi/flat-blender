"""Benchmark metrics (scanmodel.bench) on synthetic clouds with known errors, and checks on a toy scene."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapely import Polygon

from scanmodel import bench
from scanmodel.boxes import Box
from scanmodel.geom import apply, rot_z
from scanmodel.models.common import F64
from scanmodel.models.scene import Room, Scene, Source, SourceKind

RNG = np.random.default_rng(0)


def _plane(axis: int, at: float, lo: tuple[float, float], hi: tuple[float, float], sign: float, n: int):
    other = [k for k in range(3) if k != axis]
    p = np.zeros((n, 3))
    p[:, axis] = at
    p[:, other[0]] = RNG.uniform(lo[0], hi[0], n)
    p[:, other[1]] = RNG.uniform(lo[1], hi[1], n)
    nrm = np.zeros((n, 3))
    nrm[:, axis] = sign
    return p, nrm


def _room(n: int = 20000) -> tuple[F64, F64]:
    """An L-ish room 5 x 4 x 2.5 m with a cabinet: asymmetric, so registration has one answer."""
    parts = [
        _plane(2, 0.0, (0, 0), (5, 4), 1, n),  # floor
        _plane(2, 2.5, (0, 0), (5, 4), -1, n),  # ceiling
        _plane(0, 0.0, (0, 0), (4, 2.5), 1, n // 2),
        _plane(0, 5.0, (0, 0), (4, 2.5), -1, n // 2),
        _plane(1, 0.0, (0, 0), (5, 2.5), 1, n // 2),
        _plane(1, 4.0, (0, 0), (5, 2.5), -1, n // 2),
        _plane(1, 0.6, (3.5, 0), (4.5, 0.9), 1, n // 8),  # cabinet front
        _plane(0, 3.5, (0, 0), (0.6, 0.9), -1, n // 10),  # cabinet side
        _plane(2, 0.9, (3.5, 0), (4.5, 0.6), 1, n // 8),  # cabinet top
        _plane(0, 1.2, (1.0, 0.0), (2.0, 1.8), 1, n // 10),  # a pillar face
    ]
    return np.vstack([p for p, _ in parts]), np.vstack([q for _, q in parts])


def test_cloud_metrics() -> None:
    ref, _ = _room()
    ours = ref[::2] + np.array([0.0, 0.0, 0.01])  # 1 cm high everywhere, half the points
    m = bench.cloud_metrics(ours, ref)
    assert abs(m["accuracy_median_m"] - 0.01) < 0.003
    assert m["completeness_5cm"] > 0.95 and m["outliers"] == 0.0


def test_register_recovers_a_rigid_transform() -> None:
    ref, _ = _room()
    true = rot_z(37.0)
    true[:3, 3] = [0.4, -1.2, 0.05]
    m, fitness, _ = bench.register(ref, apply(true, ref))
    assert fitness > 0.9
    assert np.linalg.norm(m[:3, 3] - true[:3, 3]) < 0.01
    assert np.degrees(np.arccos(np.clip((np.trace(m[:3, :3].T @ true[:3, :3]) - 1) / 2, -1, 1))) < 0.3


def test_window_correction_and_drift() -> None:
    ref, nrm = _room()
    shifted = ref[::4] + np.array([0.02, 0.0, -0.03])  # a window whose poses are 2 cm / -3 cm off
    found = bench.window_correction(shifted, ref, nrm)
    assert found is not None
    np.testing.assert_allclose(found[0][:3, 3], [-0.02, 0.0, 0.03], atol=0.004)
    wall_only = ref[np.abs(ref[:, 0]) < 1e-9][::2]  # one wall: slides along it, not pinned
    assert bench.window_correction(wall_only, ref, nrm) is None

    a, c = np.eye(4), np.eye(4)
    c[:3, 3] = [0.03, 0.04, -0.02]
    d = bench.drift_summary([a, c])
    assert abs(d["max_xy_m"] - 0.05) < 1e-9 and abs(d["spread_z_m"] - 0.02) < 1e-9


def test_heights_agree_on_the_same_room() -> None:
    pts, nrm = _room()
    poly = bench.walk_polygon(np.array([[1.5, 1.5], [3.5, 1.5], [3.5, 3.0], [1.5, 3.0]]))
    cabinet = Polygon([(3.5, 0), (4.5, 0), (4.5, 0.6), (3.5, 0.6)])
    h = bench.heights(poly, (pts, nrm), (np.add(pts, [0, 0, 0.02]), nrm), [cabinet])
    assert h["ours"]["floor_z"] is not None and abs(h["ours"]["floor_z"]) < 0.01
    assert abs(h["floor_err_m"] + 0.02) < 0.005 and abs(h["ceiling_err_m"] + 0.02) < 0.005
    assert h["slope_agrees"]


def test_box_metrics_fit_beats_the_noise() -> None:
    pts, nrm = _room()
    # The cabinet (x 3.5..4.5, y 0..0.6, z 0..0.9) seen from the front, its free side and the top.
    truth = Box.of((4.0, 0.3, 0.45), (1.0, 0.6, 0.9), 0.0)
    m = bench.box_metrics([truth], pts, nrm)
    assert m["boxes"] == 1
    assert m["side_mean_after_m"] < m["side_mean_before_m"]
    assert m["top_mean_after_m"] < 0.01


def _scene(floors: dict[str, float]) -> Scene:
    sq = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 4.0)]
    rooms = [
        Room(
            id=f"room-x-{n}",
            name=n,
            polygon=sq,
            floor_z=z,
            ceiling_z=2.5,
            source=Source(kind=SourceKind.PLANE_FIT),
        )
        for n, z in floors.items()
    ]
    rooms.append(rooms[-1].model_copy(update={"id": "room-x-hall-2", "name": "hall"}))
    return Scene(rooms=rooms)


def test_checks_pass_fail_and_ambiguous() -> None:
    scene = _scene({"conservatory": -0.08, "living": 0.02, "hall": 0.0})
    checks = [
        bench.Check(
            id="c",
            kind="floor_delta",
            room="conservatory",
            relative_to="living",
            expected_m=-0.10,
            tolerance_m=0.04,
        ),
        bench.Check(id="w", kind="room_width", room="living", expected_m=3.2, tolerance_m=0.1),
        bench.Check(id="l", kind="room_length", room="living", expected_m=4.0, tolerance_m=0.05),
        bench.Check(id="h", kind="ceiling_height", room="hall", expected_m=2.5, tolerance_m=0.05),
    ]
    got = {r.id: r for r in bench.run_checks(scene, checks)}
    assert got["c"].ok and got["c"].measured_m is not None and abs(got["c"].measured_m + 0.10) < 1e-9
    assert not got["w"].ok and got["l"].ok
    assert not got["h"].ok and got["h"].measured_m is None and "2 matches" in got["h"].note


def test_project_checks_end_to_end(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """data/<project>/checks.json against the project's scene/model.json, no real data; results repo-relative."""
    tmp_path = repo
    office = tmp_path / "data" / "office"
    (office / "scene").mkdir(parents=True)
    scene = _scene({"conservatory": -0.08, "living": 0.02, "hall": 0.0})
    (office / "scene" / "model.json").write_text(scene.model_dump_json())
    check = {"id": "c", "kind": "floor_delta", "room": "conservatory", "relative_to": "living"}
    facts: dict[str, Any] = {
        "name": "office",
        "checks": [{**check, "expected_m": -0.10, "tolerance_m": 0.04}],
    }
    (office / "checks.json").write_text(json.dumps(facts))
    (tmp_path / "data" / "empty").mkdir()  # checks, but no scene yet
    (tmp_path / "data" / "empty" / "checks.json").write_text(json.dumps({"name": "empty", "checks": []}))

    assert bench.main(["--no-arkitscenes", "--project", "office"]) == 0
    (result,) = (tmp_path / "data" / "derived" / "bench").glob("20*.json")
    report = json.loads(result.read_text())
    assert report["checks"][0]["scene"] == "data/office/scene/model.json"
    assert report["checks"][0]["checks"][0]["ok"] is True
    assert str(tmp_path) not in result.read_text()

    assert bench.main(["--no-arkitscenes", "--project", "empty"]) == 1
    assert "just all --project empty" in capsys.readouterr().err

    assert bench.main(["--no-arkitscenes"]) == 0  # every project with checks; the one without a scene skipped
    assert "skipped empty" in capsys.readouterr().err

    facts["checks"][0]["expected_m"] = 0.10
    (office / "checks.json").write_text(json.dumps(facts))
    assert bench.main(["--no-arkitscenes", "--project", "office"]) == 1  # a failing check
