"""Room ownership: RoomPlan output a session saw from the next room's doorway (or through glass) is the next
session's room, not its own (scan 5: garden's RoomPlan had the living room's TV and walls)."""

from itertools import pairwise

import numpy as np
from shapely import Polygon

from scanmodel.models.bundle import Intrinsics
from scanmodel.models.common import to_mat4
from scanmodel.models.derived import PosedImage
from scanmodel.ownership import Votes, trim_tails
from scanmodel.stages import structure

K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def _frame(session: str, i: int, t: float, x: float, y: float, yaw: float = 0.0) -> PosedImage:
    m = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    m[:2, :2] = [[c, -s], [s, c]]
    m[:3, 3] = (x, y, 1.4)
    return PosedImage(
        id=f"{session}/{i}", session=session, kind="frame", file=f"video.mov#{i}", t=t,
        camera_to_world=to_mat4(m), intrinsics=K,
    )  # fmt: skip


def _walk(session: str, t0: float, pts: list[tuple[float, float]], per_leg: int = 40) -> list[PosedImage]:
    """Frames walking through `pts` in order, a frame per 0.1 s, turning as they go."""
    out: list[PosedImage] = []
    for (ax, ay), (bx, by) in pairwise(pts):
        for k in range(per_leg):
            u = k / per_leg
            n = len(out)
            out.append(_frame(session, n, t0 + 0.1 * n, ax + u * (bx - ax), ay + u * (by - ay), 0.3 * n))
    return out


def _flat() -> list[PosedImage]:
    """Room A (x 0-4) then room B (x 4-8) through a doorway at x = 4, y = 2. A ends standing in B's doorway
    (30 s typing, turning a little); B starts there and walks its room; then a lap through both."""
    a = _walk("a", 0.0, [(1, 1), (3, 3), (1, 3), (3, 1), (1, 1), (3, 2), (4.3, 2)])
    end = a[-1].t or 0.0
    a += [_frame("a", len(a) + i, end + 0.1 * (i + 1), 4.3, 2.0, 0.01 * i) for i in range(300)]
    b = _walk("b", end + 40.0, [(4.3, 2), (6, 1), (7, 3), (5, 3), (7, 1), (6, 2)])
    lap = _walk("lap", end + 200.0, [(6, 2), (2, 2), (6, 2)])
    return [*a, *b, *lap]


def test_tail_in_the_next_doorway_is_trimmed() -> None:
    frames = _flat()
    kept = trim_tails(frames, {"a", "b"})
    a_kept = [f for f in kept if f.session == "a"]
    assert len(a_kept) < sum(f.session == "a" for f in frames) - 300  # the dwell and the walk into it
    assert max(f.camera_to_world[0][3] for f in a_kept) < 3.5
    assert [f for f in kept if f.session == "b"] == [f for f in frames if f.session == "b"]
    # Before a session without RoomPlan (a closing lap) a session keeps its end: the lap starts in that room.
    assert len(trim_tails(frames, {"a"})) == len(frames)
    assert len([f for f in trim_tails(frames, {"a", "b"}) if f.session == "lap"]) == 80


def test_votes_give_the_next_room_to_its_session() -> None:
    votes = Votes(_flat(), {"a", "b"})  # the lap doesn't vote
    assert "lap" not in set(votes.sid.tolist())
    # An object A's RoomPlan saw in B, metres in: B's.
    v = votes.foreign("a", np.array([6.5, 2.0]))
    assert v is not None and v.owner == "b" and v.own < 0.15
    assert votes.foreign("b", np.array([6.5, 2.0])) is None
    assert votes.foreign("a", np.array([2.0, 2.0])) is None
    assert votes.owner(np.array([2.0, 2.0])) == "a"
    # A face of A's at x = 3.9 facing into A (normal -x): only cameras in front of it (in A) vote: A's.
    mid, normal = np.array([3.9, 1.0]), np.array([-1.0, 0.0])
    assert votes.foreign("a", mid + 0.5 * normal, (mid, normal)) is None
    # The same wall seen from A's doorway as B's face (normal +x, into B): B's room.
    v = votes.foreign("a", mid - 0.5 * normal, (mid, -normal))
    assert v is not None and v.owner == "b"
    # One session alone never loses anything.
    alone = Votes(_flat(), {"a"})
    assert alone.foreign("a", np.array([6.5, 2.0])) is None


def test_session_rooms_leave_out_the_next_room() -> None:
    """A's faces enclose its room and, seen from the doorway, B's: B's area is not a room of A's."""
    from scanmodel.roomplan import CapturedRoom

    frames = _flat()
    votes = Votes(frames, {"a", "b"})

    def face(i: int, p0: tuple[float, float], p1: tuple[float, float]) -> structure.Face:
        return structure.Face(f"w{i}", "a", np.array(p0, float), np.array(p1, float), 0.0, 2.5)

    # Counter-clockwise loops (room on the left): A's room and B's, sharing x = 4.
    loops = [[(0, 0), (4, 0), (4, 4), (0, 4)], [(4, 0), (8, 0), (8, 4), (4, 4)]]
    faces = [face(10 * k + i, lp[i], lp[(i + 1) % 4]) for k, lp in enumerate(loops) for i in range(4)]
    room = CapturedRoom(surfaces=[], objects=[])
    cams = np.array([f.camera_to_world for f in trim_tails(frames, {"a", "b"}) if f.session == "a"])[:, :2, 3]
    rooms, flags = structure.session_rooms("a", faces, room, np.eye(4), "a", structure.Params(), cams, votes)
    assert len(rooms) == 1, [r.polygon for r in rooms]
    assert Polygon(rooms[0].polygon).centroid.x < 4
    assert any("left out" in f.message and "b's room" in f.message for f in flags)
