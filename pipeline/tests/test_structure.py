"""Structure: which side of a wall a face is on and what it is fitted to; eval: seen faces and gross errors."""

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from scanmodel.models.scene import Scene, Source, SourceKind, Wall
from scanmodel.stages import eval as eval_stage
from scanmodel.stages.structure import Face, Params, orient_face, refine_face, walls_from_faces


def _plane_x(
    rng: np.random.Generator, x: float, facing: float, n: int = 4000, y: tuple[float, float] = (0.0, 3.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Points on the plane x = `x`, y in `y`, 0 < z < 2.4, normals along +x * facing."""
    pts = np.column_stack(
        [np.full(n, x) + rng.normal(0, 0.002, n), rng.uniform(*y, n), rng.uniform(0, 2.4, n)]
    )
    return pts, np.tile([facing, 0.0, 0.0], (n, 1))


def _face() -> Face:
    """Along y at x = 0, room on +x."""
    f = Face("w", "s1", np.array([0.0, 3.0]), np.array([0.0, 0.0]), 0.0, 2.4, ids=["w"], sessions=["s1"])
    assert np.allclose(f.interior, [1, 0])
    return f


def test_refine_fits_the_rooms_surface_not_the_far_side_of_a_thin_wall() -> None:
    rng = np.random.default_rng(0)
    # A 6 cm partition: this room's surface at x = +0.02, the next room's at x = -0.04 (facing away,
    # better scanned). Both are within the 8 cm band of RoomPlan's face.
    mine, mn = _plane_x(rng, 0.02, 1.0, 1500)
    theirs, tn = _plane_x(rng, -0.04, -1.0, 6000)
    f = _face()
    refine_face(f, np.vstack([mine, theirs]), np.vstack([mn, tn]), Params())
    assert f.fit is not None and abs(f.p0[0] - 0.02) < 0.003, f.p0
    # Only the next room's surface in reach: not refined, RoomPlan's face stays.
    g = _face()
    refine_face(g, theirs, tn, Params())
    assert g.fit is None and g.p0[0] == 0.0


def test_a_faces_own_depth_points_overrule_far_cameras() -> None:
    rng = np.random.default_rng(1)
    p = Params()
    # The session saw this surface from -x (normals towards its cameras): the room is on -x.
    away, an = _plane_x(rng, 0.01, -1.0, 2000)
    f = _face()
    orient_face(f, away, an, p)
    assert f.flipped is not None and np.allclose(f.interior, [-1, 0])
    # Both sides seen, not 2:1 against the cameras: the cameras decide.
    toward, tn = _plane_x(rng, 0.0, 1.0, 1200)
    g = _face()
    orient_face(g, np.vstack([away, toward]), np.vstack([an, tn]), p)
    assert g.flipped is None and np.allclose(g.interior, [1, 0])
    # Too few points to overrule anything; and points beyond the band don't count.
    h = _face()
    orient_face(h, away[:150], an[:150], p)
    far, fn = _plane_x(rng, -0.4, -1.0, 2000)
    orient_face(h, far, fn, p)
    assert h.flipped is None


def test_walls_record_whether_the_back_face_was_seen() -> None:
    p = Params()
    alone = _face()
    (wall,), _, _ = walls_from_faces([alone], p)
    assert wall.faces_observed == (True, False) and not wall.thickness_measured
    # A face from another session, opposite and 12 cm behind: seen, though nobody measured the thickness.
    front = _face()
    back = Face(
        "v", "s2", np.array([-0.12, 0.0]), np.array([-0.12, 3.0]), 0.0, 2.4, ids=["v"], sessions=["s2"]
    )
    (wall,), _, _ = walls_from_faces([front, back], p)
    assert wall.faces_observed == (True, True) and not wall.thickness_measured


def test_nearest_surface_skips_patchy_furniture() -> None:
    rng = np.random.default_rng(2)
    s = rng.uniform(0, 3, 6000)
    wall = np.full(6000, 0.20) + rng.normal(0, 0.003, 6000)
    found = eval_stage.nearest_surface(wall, s)
    assert found is not None and abs(found - 0.20) < 0.005
    # A cabinet nearer the face, along a fifth of it, is not the wall.
    cab_s = rng.uniform(0, 0.6, 3000)
    cab = np.full(3000, 0.05) + rng.normal(0, 0.003, 3000)
    found = eval_stage.nearest_surface(np.concatenate([wall, cab]), np.concatenate([s, cab_s]))
    assert found is not None and abs(found - 0.20) < 0.005
    assert eval_stage.nearest_surface(wall[:50], s[:50]) is None


def test_eval_headline_is_over_seen_faces(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    # Wall along x (normal +y), 10 cm thick: its left face at y = +0.05 sits on the scanned surface; its
    # right face (y = -0.05) nobody saw, and the points there are 6 cm behind it (TSDF spill).
    n = 20000
    x, z = rng.uniform(0, 3, n), rng.uniform(0, 2.4, n)
    left = np.column_stack([x, np.full(n, 0.05) + rng.normal(0, 0.002, n), z])
    right = np.column_stack([x, np.full(n, -0.11) + rng.normal(0, 0.002, n), z])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([left, right])))
    pc.normals = o3d.utility.Vector3dVector(
        np.vstack([np.tile([0, 1, 0], (n, 1)), np.tile([0, -1, 0], (n, 1))])
    )
    (tmp_path / "fuse").mkdir()
    o3d.io.write_point_cloud(tmp_path / "fuse" / "points.ply", pc)
    wall = Wall(
        id="wall-a",
        start=(0.0, 0.0),
        end=(3.0, 0.0),
        thickness=0.1,
        faces_observed=(True, False),
        height=2.4,
        source=Source(kind=SourceKind.PLANE_FIT),
    )
    (tmp_path / "model.json").write_text(Scene(walls=[wall]).model_dump_json())
    s = eval_stage.run(tmp_path / "model.json", tmp_path / "fuse", tmp_path / "eval")
    assert s["observed_faces"] == s["observed_faces_total"] == 1 and s["unobserved_faces"] == 1
    assert isinstance(s["observed_p95_m"], float) and s["observed_p95_m"] < 0.01 and s["pass"], s
    assert isinstance(s["unobserved_mean_m"], float) and abs(s["unobserved_mean_m"] - 0.06) < 0.005, s
    assert isinstance(s["p95_m"], float) and s["p95_m"] > 0.05  # all faces, as before
    assert s["faces_gross"] == 0
    history = json.loads((tmp_path / "eval" / "history.jsonl").read_text().splitlines()[-1])
    assert history["observed_p95_m"] == s["observed_p95_m"]
    assert "Unseen faces" in (tmp_path / "eval" / "report.html").read_text()
    # Heatmap: face points (not the margins) coloured by its error (green on the seen face, red behind the unseen one).
    colours = np.asarray(o3d.io.read_point_cloud(tmp_path / "eval" / "heatmap.ply").colors)
    assert (colours[:n, 1] > 0.9).mean() > 0.8 and (colours[n:, 0] > 0.9).mean() > 0.8


def test_split_session_elements_follow_the_part_on_the_points() -> None:
    """A session refine cut in two: RoomPlan's west wall (and its door) were mapped before the jump, its
    east wall after it, in the jumped frame. Each lands where the fused points are: the east wall with the
    other part's move, the west wall and its door with the session's."""
    from scanmodel.roomplan import CapturedRoom, RoomObject, Surface
    from scanmodel.stages.structure import place_parts

    def surface(id_: str, kind: str, x: float, parent: str | None = None) -> Surface:
        m = np.eye(4)  # local x along the wall (ARKit -z), y up, z the normal (ARKit +x)
        m[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        m[:3, 3] = [x, 1.2, -2.0]
        return Surface(id_, kind, m, np.array([4.0 if kind == "wall" else 0.9, 2.4, 0.0]), parent)

    room = CapturedRoom(
        [surface("west", "wall", 0.0), surface("door", "door", 0.0, "west"), surface("east", "wall", 3.0)],
        [RoomObject("sofa", "sofa", np.eye(4), np.array([1.0, 0.8, 0.9]))],
    )
    s2f = np.eye(4)
    s2f[:3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]  # ARKit y up -> z up
    jump = np.eye(4)
    jump[:3, 3] = [0.3, 0.1, 0.0]
    other = jump @ s2f
    rng = np.random.default_rng(0)
    y, z = rng.uniform(0.0, 4.0, 3000), rng.uniform(0.0, 2.4, 3000)
    pts = np.concatenate(
        [np.column_stack([np.zeros(3000), y, z]), np.column_stack([np.full(3000, 3.3), y + 0.1, z])]
    )
    out, n = place_parts(room, s2f, [other], pts)
    placed = {s.id: (s2f @ s.transform)[:3, 3] for s in out.surfaces}
    assert n == 1
    assert np.allclose(placed["east"], [3.3, 2.1, 1.2]) and np.allclose(placed["west"], [0.0, 2.0, 1.2])
    assert np.allclose(placed["door"], placed["west"])
