"""Oriented boxes (RoomPlan objects and stairs) refined against the fused points and the walls.

RoomPlan's boxes are loose by several cm. Rules, none tied to a particular building:
1. Points: each vertical side moves to the dominant outward-facing surface within `max_shift` of it,
   if that surface covers at least half the side. A side nobody saw (against a wall, against a
   neighbour) can still move to where the box's top surface ends, if it ends within reach (a worktop
   that runs on into the next unit says nothing). The top moves to the dominant upward surface near it.
   A side without enough support stays where RoomPlan put it.
2. Walls: a side within `wall_gap` in front of a parallel wall face, or up to `wall_overlap` behind it,
   ends at the face. Furniture stands against walls; nothing stands inside them.
3. Neighbours: facing sides of two parallel boxes less than `neighbour_gap` apart (or overlapping by up
   to `max_shift`) meet: at the side that has point support, else halfway.
"""

from dataclasses import dataclass, field

import numpy as np

from .models.common import F64

SIDES = ("+x", "-x", "+y", "-y")


@dataclass
class Box:
    centre: F64  # xy
    yaw_deg: float
    half: dict[str, float]  # side -> distance from the centre along its outward normal
    z0: float
    z1: float
    supported: set[str] = field(default_factory=set[str])  # sides placed from points

    @classmethod
    def of(cls, center: tuple[float, float, float], size: tuple[float, float, float], yaw_deg: float) -> Box:
        sx, sy, sz = size
        return cls(
            np.array(center[:2], float),
            yaw_deg,
            {"+x": sx / 2, "-x": sx / 2, "+y": sy / 2, "-y": sy / 2},
            center[2] - sz / 2,
            center[2] + sz / 2,
        )

    @property
    def axes(self) -> tuple[F64, F64]:
        a = np.radians(self.yaw_deg)
        return np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])

    def normal(self, side: str) -> F64:
        x, y = self.axes
        return {"+x": x, "-x": -x, "+y": y, "-y": -y}[side]

    def span(self, side: str) -> tuple[float, float, F64]:
        """The side's extent along its own direction (lo, hi) and that direction."""
        x, y = self.axes
        if side in ("+x", "-x"):
            return -self.half["-y"], self.half["+y"], y
        return -self.half["-x"], self.half["+x"], x

    def plane(self, side: str) -> float:
        """Offset of the side's plane along its normal, from the world origin."""
        return float(self.centre @ self.normal(side) + self.half[side])

    def move(self, side: str, offset: float) -> None:
        """Put the side's plane at `offset` (world, along its normal); the opposite side stays."""
        self.half[side] += offset - self.plane(side)

    def recentred(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """(center, size) for the scene model."""
        x, y = self.axes
        cx = (self.half["+x"] - self.half["-x"]) / 2
        cy = (self.half["+y"] - self.half["-y"]) / 2
        c = self.centre + cx * x + cy * y
        size = (self.half["+x"] + self.half["-x"], self.half["+y"] + self.half["-y"], self.z1 - self.z0)
        return (float(c[0]), float(c[1]), (self.z0 + self.z1) / 2), (
            float(size[0]),
            float(size[1]),
            float(size[2]),
        )


@dataclass
class BoxParams:
    max_shift: float = 0.15  # RoomPlan's boxes are off by up to about this much
    min_points: int = 150
    wall_gap: float = 0.08
    wall_overlap: float = 0.10
    neighbour_gap: float = 0.05


def peak(t: F64, bin_m: float = 0.005) -> tuple[float, float] | None:
    """Dominant value of t (smoothed 5 mm histogram, mean within 1.5 cm) and its share of t."""
    if len(t) == 0:
        return None
    h, e = np.histogram(t, bins=np.arange(t.min() - bin_m, t.max() + 2 * bin_m, bin_m))
    hs = np.convolve(h, [1, 2, 1], "same")
    c = (e[hs.argmax()] + e[hs.argmax() + 1]) / 2
    near = np.abs(t - c) < 0.015
    return float(t[near].mean()), float(near.mean())


def fit_to_points(box: Box, pts: F64, nrm: F64, p: BoxParams) -> None:
    """Rule 1. Mutates `box`."""
    rel = pts[:, :2] - box.centre
    x, y = box.axes
    lx, ly = rel @ x, rel @ y
    nx, ny = nrm[:, :2] @ x, nrm[:, :2] @ y
    reach = max(box.half.values()) + p.max_shift + 0.05  # + the look beyond a worktop's edge
    near = (
        (np.abs(lx) < reach)
        & (np.abs(ly) < reach)
        & (pts[:, 2] > box.z0 - 0.05)
        & (pts[:, 2] < box.z1 + p.max_shift)
    )
    # Top: the dominant upward surface near RoomPlan's top, over the box's middle.
    inner = near & (lx > -box.half["-x"] + 0.05) & (lx < box.half["+x"] - 0.05)
    inner &= (ly > -box.half["-y"] + 0.05) & (ly < box.half["+y"] - 0.05)
    up = inner & (nrm[:, 2] > 0.9) & (np.abs(pts[:, 2] - box.z1) < p.max_shift)
    top = peak(pts[up, 2]) if up.sum() >= p.min_points else None
    if top is not None and top[1] > 0.3:
        box.z1 = top[0]
    on_top = near & (nrm[:, 2] > 0.9) & (np.abs(pts[:, 2] - box.z1) < 0.02)
    for side in SIDES:
        along_side = {"+x": lx, "-x": -lx, "+y": ly, "-y": -ly}[side]
        across = {"+x": ly, "-x": ly, "+y": lx, "-y": lx}[side]
        facing = {"+x": nx, "-x": -nx, "+y": ny, "-y": -ny}[side]
        lo, hi, _ = box.span(side)
        window = (
            (np.abs(along_side - box.half[side]) < p.max_shift) & (across > lo + 0.05) & (across < hi - 0.05)
        )
        wall = near & window & (facing > 0.8) & (pts[:, 2] > box.z0 + 0.05) & (pts[:, 2] < box.z1 - 0.05)
        found = peak(along_side[wall]) if wall.sum() >= p.min_points else None
        if found is not None and found[1] > 0.3:
            on_face = wall & (np.abs(along_side - found[0]) < 0.015)
            cover = np.percentile(across[on_face], 95) - np.percentile(across[on_face], 5)
            if cover >= 0.5 * (hi - lo):  # the side's own surface, not an appliance edge or a handle
                box.half[side] = found[0]
                box.supported.add(side)
                continue
        # Hidden side: where the top surface ends (a worktop running on past RoomPlan's box). If it
        # runs on to the end of the window (into a neighbour), its edge says nothing: keep the side.
        reach = box.half[side] + p.max_shift
        edge = on_top & (np.abs(along_side - box.half[side]) < p.max_shift + 0.05) & (across > lo + 0.05)
        edge &= across < hi - 0.05
        if edge.sum() >= p.min_points:
            end = float(np.percentile(along_side[edge], 98))
            if end <= reach:  # it ends within reach (looking 5 cm beyond to tell)
                box.half[side] = end


def _faces(walls: list[tuple[F64, F64, float]]) -> list[tuple[F64, F64, F64]]:
    """Wall faces as (start, end, normal into the space in front of it), from (start, end, thickness)."""
    out: list[tuple[F64, F64, F64]] = []
    for a, c, t in walls:
        d = (c - a) / np.linalg.norm(c - a)
        n = np.array([-d[1], d[0]])
        out += [(a + n * t / 2, c + n * t / 2, n), (a - n * t / 2, c - n * t / 2, -n)]
    return out


def fit_to_walls(box: Box, walls: list[tuple[F64, F64, float]], gap: float, overlap: float) -> None:
    """Rule 2. `walls`: (centreline start, end, thickness). Mutates `box`."""
    faces = _faces(walls)
    for side in SIDES:
        u = box.normal(side)
        lo, hi, along = box.span(side)
        best: float | None = None
        for a, c, fn in faces:
            if fn @ u > -0.995:  # the face must look back at this side
                continue
            # overlap along the side
            sa, sc = sorted([(a - box.centre) @ along, (c - box.centre) @ along])
            if min(hi, sc) - max(lo, sa) < 0.5 * (hi - lo):
                continue
            move = float(a @ u) - box.plane(side)  # > 0: the face is in front of the side
            if -overlap < move < gap and (best is None or abs(move) < abs(best)):
                best = move
        if best is not None and (side not in box.supported or abs(best) < 0.02):
            box.half[side] += best


def fit_neighbours(boxes: list[Box], p: BoxParams) -> None:
    """Rule 3. Mutates the boxes."""
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, c = boxes[i], boxes[j]
            if min(a.z1, c.z1) - max(a.z0, c.z0) < 0.1:
                continue
            for sa in SIDES:
                for sc in SIDES:
                    ua, uc = a.normal(sa), c.normal(sc)
                    if ua @ uc > -0.995:
                        continue
                    gap = -(a.plane(sa) + c.plane(sc))  # along ua: from a's side to c's side
                    if not -p.max_shift < gap < p.neighbour_gap:
                        continue
                    lo_a, hi_a, along = a.span(sa)
                    lo_c, hi_c, _ = c.span(sc)
                    shift = float((c.centre - a.centre) @ along)
                    sl, sh = sorted(
                        [lo_c * np.sign(along @ c.span(sc)[2]), hi_c * np.sign(along @ c.span(sc)[2])]
                    )
                    if min(hi_a, sh + shift) - max(lo_a, sl + shift) < 0.5 * min(hi_a - lo_a, hi_c - lo_c):
                        continue
                    if sa in a.supported and sc not in c.supported:
                        target = a.plane(sa)
                    elif sc in c.supported and sa not in a.supported:
                        target = -c.plane(sc)
                    else:
                        target = a.plane(sa) + gap / 2
                    a.move(sa, target)
                    c.move(sc, -target)
