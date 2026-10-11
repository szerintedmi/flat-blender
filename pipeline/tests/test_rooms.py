"""Room outline, floor, ceiling and wall heights (scanmodel.rooms) on synthetic points."""

from collections.abc import Callable

import numpy as np
from shapely import Polygon

from scanmodel.rooms import RoomParams, fit_heights, snap_outline, wall_heights

RNG = np.random.default_rng(0)
T = 0.10
# Room 4 x 3 m; walls' inner faces at x = 0, x = 4, y = 0, y = 3.
WALLS = [
    (np.array([-T / 2, -T / 2]), np.array([4 + T / 2, -T / 2]), T),
    (np.array([4 + T / 2, -T / 2]), np.array([4 + T / 2, 3 + T / 2]), T),
    (np.array([4 + T / 2, 3 + T / 2]), np.array([-T / 2, 3 + T / 2]), T),
    (np.array([-T / 2, 3 + T / 2]), np.array([-T / 2, -T / 2]), T),
]


def _horizontal(
    n: int,
    x: tuple[float, float],
    y: tuple[float, float],
    z: Callable[[np.ndarray, np.ndarray], np.ndarray],
    up: bool,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.column_stack([RNG.uniform(*x, n), RNG.uniform(*y, n)])
    zz = z(xy[:, 0], xy[:, 1]) + RNG.normal(0, 0.003, n)
    nrm = np.tile([0.0, 0.0, 1.0 if up else -1.0], (n, 1))
    return np.column_stack([xy, zz]), nrm


def test_outline_snaps_onto_the_wall_faces() -> None:
    # The session's own faces put the outline 6 cm inside two walls and 25 cm inside a third.
    outline = [(0.06, 0.0), (4.0, 0.0), (4.0, 2.94), (0.25, 2.94)]
    snapped = Polygon(snap_outline(outline, WALLS, RoomParams()))
    assert snapped.symmetric_difference(Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])).area < 0.01


def test_raised_floor_and_sloped_ceiling() -> None:
    floor, fn = _horizontal(6000, (0, 4), (0, 3), lambda x, y: np.full_like(x, 0.17), True)
    table, tn = _horizontal(3000, (1, 2), (1, 2), lambda x, y: np.full_like(x, 0.75), True)
    ceiling, cn = _horizontal(6000, (0, 4), (0, 3), lambda x, y: 2.2 + 0.2 * x, False)
    cn[:, 0] = 0.2  # tilted normals, still pointing down
    cn /= np.linalg.norm(cn, axis=1, keepdims=True)
    pts, nrm = np.vstack([floor, table, ceiling]), np.vstack([fn, tn, cn])
    z_floor, z_ceiling, plane = fit_heights([(0, 0), (4, 0), (4, 3), (0, 3)], pts, nrm, RoomParams())
    assert z_floor is not None and abs(z_floor - 0.17) < 0.01, z_floor
    assert (
        plane is not None
        and abs(plane[0] - 2.2) < 0.02
        and abs(plane[1] - 0.2) < 0.01
        and abs(plane[2]) < 0.01
    )
    assert z_ceiling is not None and abs(z_ceiling - 2.6) < 0.02  # at the centre, x = 2

    # The same room with a flat ceiling: no plane.
    flat, fln = _horizontal(6000, (0, 4), (0, 3), lambda x, y: np.full_like(x, 2.45), False)
    _, z_flat, plane = fit_heights(
        [(0, 0), (4, 0), (4, 3), (0, 3)], np.vstack([floor, flat]), np.vstack([fn, fln]), RoomParams()
    )
    assert plane is None and z_flat is not None and abs(z_flat - 2.45) < 0.01


def test_wall_runs_from_the_lower_floor_to_the_higher_ceiling() -> None:
    # Wall along y = 3 between a room below (y < 3: floor 0, flat ceiling 2.4) and a lean-to above
    # (y > 3: floor 0.17, ceiling 2.2 + 0.2 x).
    below = (Polygon([(0, 0), (4, 0), (4, 2.95), (0, 2.95)]), 0.0, 2.4, None)
    above = (Polygon([(0, 3.05), (4, 3.05), (4, 6), (0, 6)]), 0.17, 2.6, (2.2, 0.2, 0.0))
    found = wall_heights(np.array([0.0, 3.0]), np.array([4.0, 3.0]), T, [below, above])
    assert found is not None
    base, top0, top1 = found
    assert base == 0.0 and abs(top0 - 2.4) < 1e-9 and abs(top1 - 3.0) < 1e-9


def _level(n: int, x: tuple[float, float], y: tuple[float, float], z: float) -> np.ndarray:
    return np.column_stack([RNG.uniform(*x, n), RNG.uniform(*y, n), z + RNG.normal(0, 0.003, n)])


def test_floor_is_found_relative_to_the_room_not_the_origin() -> None:
    from scanmodel.rooms import floor_level

    def level(up: np.ndarray) -> float:
        z = floor_level(up, 0.8)
        assert z is not None
        return z

    # A split level 1.2 m up: still found, nothing assumes the scene's z = 0.
    assert abs(level(_level(3000, (0, 4), (0, 3), 1.20)) - 1.20) < 0.005
    # A bed (2 x 1.6 m at 0.5 m) RoomPlan missed, in a 3 x 3 m room: the floor is the lower level.
    assert (
        abs(level(np.vstack([_level(3000, (0, 3), (0, 3), 0.0), _level(6000, (0, 2), (0, 1.6), 0.5)])))
        < 0.005
    )


def test_a_room_is_where_its_cameras_were_not_the_garden_seen_through_the_glass() -> None:
    from scanmodel.roomplan import CapturedRoom
    from scanmodel.stages.structure import Face, Params, session_rooms

    def face(a: tuple[float, float], c: tuple[float, float]) -> Face:
        return Face("w", "s", np.array(a, float), np.array(c, float), 0.0, 2.4)

    # A 4 x 3 m conservatory (y 0..3) and, behind its glass wall at y = 0, a garden fenced off 4 m out.
    room = [face((0, 0), (4, 0)), face((4, 0), (4, 3)), face((4, 3), (0, 3)), face((0, 3), (0, 0))]
    fence = [face((0, -4), (0, 0)), face((4, 0), (4, -4)), face((4, -4), (0, -4))]
    cams = np.column_stack([RNG.uniform(0.8, 3.2, 200), RNG.uniform(0.8, 2.2, 200)])  # all indoors
    rooms, _ = session_rooms(
        "s", room + fence, CapturedRoom([], []), np.eye(4), "conservatory", Params(), cams
    )
    assert len(rooms) == 1
    assert abs(Polygon(rooms[0].polygon).area - 12.0) < 0.3, Polygon(rooms[0].polygon).bounds
