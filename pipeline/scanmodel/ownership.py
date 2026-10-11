"""Which session's room a place is: votes by the sessions' cameras, for RoomPlan output seen from elsewhere.

RoomPlan keeps adding what it sees: a session that ends in the next room's doorway (Next room: the user
stands there typing the name, and the next session starts on that spot) or looks through a window or
doorway reports the next room's walls and furniture too (scan 5: garden's RoomPlan has the living room's
TV and a face on living-room furniture seen through the glass; living's has the hallway's walls, which
made a second living "room" over the hallway). Where a session's cameras were tells its own room:

- a session's trailing poses within `TAIL_M` of the next session's first pose are its doorway dwell,
  not its room (`trim_tails`);
- cameras are thinned by motion (standing still counts once), and only sessions with RoomPlan vote;
- a place is another session's when, among the voting cameras within `RADIUS_M`, that session holds at
  least `OTHER_MIN` of them (and `MIN_OTHER` in number) and this one under `OWN_MAX`. For a wall face
  the place is `FACE_PUSH_M` in front of it, and only cameras in front of it count: a glass wall seen
  from the garden keeps the garden's vote, whatever the living room's cameras did behind it.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .models.common import F64, mat
from .models.derived import PosedImage
from .points import thin_frames

TAIL_M = 1.0  # a session's last poses this close to the next session's first: the doorway dwell
THIN_M, THIN_DEG = 0.15, 10.0  # voting cameras: one per this much movement
RADIUS_M = 2.5  # cameras this close (horizontally) to a place vote on it
MIN_OTHER = 5  # another session needs at least this many voting cameras to take a place
OTHER_MIN = 0.6  # ...and this share of the votes,
OWN_MAX = 0.15  # while the session itself has less than this share
FACE_PUSH_M = 0.5  # a face's place: this far in front of its midpoint
FACE_SIDE_M = 0.3  # cameras at least this far in front of the face vote on it
AREA_RATIO = 2.0  # an enclosed area is another session's room if it holds this many times our cameras


def _xy(im: PosedImage) -> F64:
    return mat(im.camera_to_world)[:2, 3]


def trim_tails(
    frames: list[PosedImage], rooms: set[str] | None = None, tail_m: float = TAIL_M
) -> list[PosedImage]:
    """`frames` without each session's trailing doorway dwell: its last poses (a contiguous run up to the
    end) within `tail_m` of the next session's first pose (sessions in time order), when that next session
    is another room's (in `rooms`, the sessions with RoomPlan; default: any). A closing lap starts in the room
    the last one ended in, so that one keeps its end, as do sessions whose next one starts elsewhere."""
    by: dict[str, list[PosedImage]] = {}
    for f in frames:
        if f.t is not None:
            by.setdefault(f.session, []).append(f)
    for fs in by.values():
        fs.sort(key=lambda f: f.t or 0.0)
    order = sorted(by, key=lambda s: by[s][0].t or 0.0)
    cut: dict[str, float] = {}
    for a, b in pairwise(order):
        if rooms is not None and b not in rooms:
            continue
        start = _xy(by[b][0])
        fs = by[a]
        n = len(fs)
        while n > 0 and np.linalg.norm(_xy(fs[n - 1]) - start) < tail_m:
            n -= 1
        if 0 < n < len(fs):
            cut[a] = fs[n].t or 0.0
    return [f for f in frames if f.t is None or f.session not in cut or f.t < cut[f.session]]


@dataclass
class Vote:
    owner: str  # the session the place belongs to
    own: float  # the asking session's share of the votes
    other: float  # the owner's share


class Votes:
    """The voting cameras: thinned positions of the sessions with RoomPlan, their doorway dwell trimmed."""

    def __init__(self, frames: list[PosedImage], sessions: set[str]) -> None:
        trimmed = trim_tails(frames, sessions)
        xy: list[F64] = []
        sid: list[str] = []
        for s in sorted(sessions):
            own = sorted((f for f in trimmed if f.session == s and f.t is not None), key=lambda f: f.t or 0.0)
            for f in thin_frames(own, THIN_M, THIN_DEG):
                xy.append(_xy(f))
                sid.append(s)
        self.xy = np.array(xy).reshape(-1, 2)
        self.sid = np.array(sid)
        self.sessions = sessions

    def counts(self, place: F64, front: tuple[F64, F64] | None = None) -> Counter[str]:
        """Voting cameras per session within RADIUS_M of `place`; with `front` = (a point on a face, its
        normal), only those at least FACE_SIDE_M in front of the face."""
        near = np.linalg.norm(self.xy - place, axis=1) < RADIUS_M
        if front is not None:
            near &= (self.xy - front[0]) @ front[1] > FACE_SIDE_M
        return Counter(self.sid[near].tolist())

    def foreign(self, session: str, place: F64, front: tuple[F64, F64] | None = None) -> Vote | None:
        """The vote that makes `place` another session's room, else None (also with one session only)."""
        c = self.counts(place, front)
        total = sum(c.values())
        if not total or len(self.sessions) < 2:
            return None
        owner, n = max(((s, k) for s, k in c.items() if s != session), key=lambda x: x[1], default=("", 0))
        own = c.get(session, 0) / total
        if n >= MIN_OTHER and n / total >= OTHER_MIN and own < OWN_MAX:
            return Vote(owner, own, n / total)
        return None

    def owner(self, place: F64) -> str | None:
        """The session with the most voting cameras around `place` (None if none is near)."""
        c = self.counts(place)
        return c.most_common(1)[0][0] if c else None

    def inside(self, contains: np.ndarray) -> Counter[str]:
        """Voting cameras per session where `contains` (a mask over the cameras) is true."""
        return Counter(self.sid[contains].tolist())
