"""Object and stair boxes against synthetic points and walls (scanmodel.boxes, structure's use of it)."""

import numpy as np

from scanmodel.boxes import Box, BoxParams, fit_neighbours, fit_to_points, fit_to_walls
from scanmodel.models.scene import Object, Source, SourceKind, Stair
from scanmodel.stages.structure import merge_stairs

RNG = np.random.default_rng(0)
WALL = (np.array([-1.0, -0.05]), np.array([4.0, -0.05]), 0.10)  # face at y = 0, room at y > 0


def _rect(
    axis: int, at: float, lo: tuple[float, float], hi: tuple[float, float], normal: float, n: int = 3000
):
    """Points on the plane coordinate[axis] = at, the other two in [lo, hi], normals along +-axis."""
    other = [k for k in range(3) if k != axis]
    p = np.zeros((n, 3))
    p[:, axis] = at + RNG.normal(0, 0.002, n)
    p[:, other[0]] = RNG.uniform(lo[0], hi[0], n)
    p[:, other[1]] = RNG.uniform(lo[1], hi[1], n)
    nrm = np.zeros((n, 3))
    nrm[:, axis] = normal
    return p, nrm


def _cloud(*parts: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([q[0] for q in parts]), np.vstack([q[1] for q in parts])


def _extent(box: Box) -> tuple[float, float, float, float, float]:
    (cx, cy, cz), (sx, sy, sz) = box.recentred()
    return cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz + sz / 2


def test_cabinet_fits_its_faces_and_stands_against_the_wall() -> None:
    # True cabinet: x 1.0..1.6, y 0..0.6 (back on the wall), top 0.9. RoomPlan is a few cm off everywhere.
    pts, nrm = _cloud(
        _rect(1, 0.6, (1.0, 0.0), (1.6, 0.9), 1.0),  # front
        _rect(0, 1.0, (0.0, 0.0), (0.6, 0.9), -1.0),  # left side
        _rect(0, 1.6, (0.0, 0.0), (0.6, 0.9), 1.0),  # right side
        _rect(2, 0.9, (1.0, 0.0), (1.6, 0.6), 1.0),  # top
    )
    box = Box.of((1.295, 0.315, 0.43), (0.51, 0.53, 0.86), 0.0)  # x 1.04..1.55, y 0.05..0.58
    p = BoxParams()
    fit_to_points(box, pts, nrm, p)
    fit_to_walls(box, [WALL], p.wall_gap, p.wall_overlap)
    x0, x1, y0, y1, top = _extent(box)
    assert abs(x0 - 1.0) < 0.01 and abs(x1 - 1.6) < 0.01 and abs(y1 - 0.6) < 0.01, (x0, x1, y1)
    assert abs(y0) < 0.005, y0  # the unseen back now touches the wall
    assert abs(top - 0.9) < 0.01, top


def test_worktop_runs_on_to_the_tall_unit_beside_it() -> None:
    # Counter x 0..0.6 (worktop at 0.9), tall unit x 0.6..1.2 up to 2.2; RoomPlan leaves a 13 cm gap.
    pts, nrm = _cloud(
        _rect(1, 0.6, (0.0, 0.0), (0.6, 0.9), 1.0),  # counter front
        _rect(2, 0.9, (0.0, 0.0), (0.6, 0.6), 1.0),  # worktop
        _rect(1, 0.6, (0.6, 0.0), (1.2, 2.2), 1.0),  # tall unit front
        _rect(0, 0.6, (0.0, 0.9), (0.6, 2.2), -1.0),  # tall unit side, seen above the worktop
        _rect(0, 1.2, (0.0, 0.0), (0.6, 2.2), 1.0),  # tall unit far side
        _rect(2, 2.2, (0.6, 0.0), (1.2, 0.6), 1.0),  # tall unit top
    )
    counter = Box.of((0.235, 0.3, 0.45), (0.47, 0.6, 0.9), 0.0)  # x 0..0.47
    tall = Box.of((0.925, 0.3, 1.1), (0.55, 0.6, 2.2), 0.0)  # x 0.65..1.2
    p = BoxParams()
    for box in (counter, tall):
        fit_to_points(box, pts, nrm, p)
        fit_to_walls(box, [WALL], p.wall_gap, p.wall_overlap)
    fit_neighbours([counter, tall], p)
    assert abs(_extent(counter)[1] - 0.6) < 0.01, _extent(counter)
    assert abs(_extent(tall)[0] - 0.6) < 0.01, _extent(tall)
    assert abs(_extent(tall)[1] - 1.2) < 0.01, _extent(tall)


def test_sparse_furniture_keeps_roomplan_box() -> None:
    # A chair: a few leg points, nothing to fit to. The box must not wander.
    pts, nrm = _rect(2, 0.45, (2.0, 2.0), (2.4, 2.4), 1.0, n=40)
    box = Box.of((2.2, 2.2, 0.45), (0.45, 0.5, 0.9), 15.0)
    before = box.recentred()
    fit_to_points(box, pts, nrm, BoxParams())
    after = box.recentred()
    assert np.allclose(before[0][:2], after[0][:2]) and np.allclose(before[1][:2], after[1][:2])


def _stair(sid: str, x: float, treads: int) -> Stair:
    return Stair(
        id=f"stair-{sid}",
        center=(x, 1.0, 1.2),
        size=(0.9, 1.8, 2.4),
        tread_z=[0.2 * i for i in range(treads)],
        source=Source(kind=SourceKind.ROOMPLAN, sessions=[sid], roomplan_ids=[sid]),
    )


def test_one_flight_seen_twice_is_one_stair_and_stays_out_of_the_wall() -> None:
    # Two sessions saw the same flight 6 cm apart; it reaches 10 cm past a wall face at x = 0.
    wall = (np.array([-0.05, -1.0]), np.array([-0.05, 4.0]), 0.10)  # face at x = 0, room at x > 0
    merged = merge_stairs([_stair("a", 0.35, 8), _stair("b", 0.41, 11)], [wall])
    assert len(merged) == 1
    st = merged[0]
    assert st.source.sessions == ["a", "b"] and st.source.kind == SourceKind.MERGED
    assert len(st.tread_z) == 11  # the better-fitted copy is kept
    assert abs((st.center[0] - st.size[0] / 2) - 0.0) < 0.005, st  # ends at the wall face
    # Two flights side by side are not merged.
    assert len(merge_stairs([_stair("a", 0.5, 8), _stair("b", 2.0, 8)], [])) == 2


def test_worktop_running_on_into_the_next_unit_moves_nothing() -> None:
    # Three 0.6 m units under one worktop (x 0..1.8); RoomPlan's middle box is 8 cm short on its +x side.
    pts, nrm = _cloud(
        _rect(1, 0.6, (0.0, 0.0), (1.8, 0.9), 1.0),  # the fronts, flush
        _rect(2, 0.9, (0.0, 0.0), (1.8, 0.6), 1.0),  # one worktop over all three
    )
    middle = Box.of((0.86, 0.3, 0.45), (0.52, 0.6, 0.9), 0.0)  # x 0.6..1.12 (true: 0.6..1.2)
    fit_to_points(middle, pts, nrm, BoxParams())
    x0, x1, *_ = _extent(middle)
    # The worktop doesn't end here, so it can't place the side: it stays (no 15 cm grab).
    assert abs(x1 - 1.12) < 0.005 and abs(x0 - 0.6) < 0.005, (x0, x1)


def _obj(sid: str, x: float, width: float = 0.4) -> Object:
    return Object(
        id=f"object-{sid}-{x}",
        category="storage",
        center=(x, 0.3, 0.45),
        size=(width, 0.6, 0.9),
        source=Source(kind=SourceKind.ROOMPLAN, sessions=[sid], roomplan_ids=[f"{sid}{x}"]),
    )


def test_a_run_of_narrow_units_stays_a_run_but_one_unit_seen_twice_is_one() -> None:
    from scanmodel.stages.structure import same_object

    # One session: two 40 cm units side by side (centres 40 cm apart) are two objects.
    assert not same_object(_obj("a", 0.2), _obj("a", 0.6))
    # Two sessions saw the same unit 5 cm apart: one object. Side by side, still two.
    assert same_object(_obj("a", 0.2), _obj("b", 0.25))
    assert not same_object(_obj("a", 0.2), _obj("b", 0.6))


def test_small_rooms_are_rooms_slivers_are_not() -> None:
    from shapely import Polygon

    from scanmodel.stages.structure import Params, is_room

    p = Params()
    assert is_room(Polygon([(0, 0), (0.8, 0), (0.8, 1.2), (0, 1.2)]), p)  # a WC, 0.96 m2
    assert is_room(Polygon([(0, 0), (0.6, 0), (0.6, 0.9), (0, 0.9)]), p)  # a walk-in cupboard
    assert not is_room(Polygon([(0, 0), (0.3, 0), (0.3, 3.0), (0, 3.0)]), p)  # between a wall's two faces


def test_a_door_needs_a_leaf_in_the_scan() -> None:
    from scanmodel.models.scene import Opening, OpeningKind, Wall
    from scanmodel.stages.structure import door_leaf

    # Wall along x (y = 0, 10 cm thick), a 0.8 x 2.0 m door centred at x = 1.0.
    wall = Wall(
        id="w",
        start=(0.0, 0.0),
        end=(3.0, 0.0),
        thickness=0.1,
        height=2.4,
        source=Source(kind=SourceKind.ROOMPLAN),
    )
    door = Opening(
        id="door-1",
        kind=OpeningKind.DOOR,
        wall="w",
        offset=1.0,
        width=0.8,
        bottom=0.0,
        height=2.0,
        source=Source(kind=SourceKind.ROOMPLAN),
    )
    jambs = _cloud(
        _rect(0, 0.6, (-0.05, 0.0), (0.05, 2.0), 1.0, 300),
        _rect(0, 1.4, (-0.05, 0.0), (0.05, 2.0), -1.0, 300),
    )
    assert door_leaf(door, wall, *jambs) == "none"  # a doorway: frame only
    closed = _cloud(jambs, _rect(1, 0.02, (0.62, 0.0), (1.38, 2.0), 1.0))
    assert door_leaf(door, wall, *closed) == "closed"
    swung = _cloud(jambs, _rect(0, 0.62, (0.06, 0.0), (0.82, 1.95), 1.0))  # hinged at x = 0.6, open into +y
    assert door_leaf(door, wall, *swung) == "open"
