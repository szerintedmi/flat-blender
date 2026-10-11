"""Rooms fitted to the walls and the points: outline, floor height, ceiling (flat or sloped), wall heights.

Rules, none tied to a particular building:
- Outline: a room's floor outline comes from its own session's faces, while a shared wall is built from
  one session's face plus a thickness. Each outline edge moves onto the nearest parallel wall face that
  looks into the room (within `snap_m`), so floors and ceilings meet the walls.
- Floor: the lowest strong cluster of upward surfaces up to `floor_rise` above the room's lowest one
  (1st percentile), stair and furniture footprints left out; not tied to the scene frame's z = 0 (a split
  level, a step into a conservatory). Tried "the level with the largest area" instead (better for a
  shower tray): worse on the ARKitScenes bench, where drift leaves the fused floor in thin layers.
- Ceiling: a plane through the downward-facing points high in the room (RANSAC, fixed seed). Flat
  unless it tilts more than `slope_min_deg` (a lean-to roof, an attic).
- Walls: from the lowest floor beside them up to the highest ceiling above them, at each end, so a wall
  under a sloped ceiling gets a sloped top.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely import LineString, Polygon, contains_xy
from shapely.geometry.polygon import orient

from .models.common import F64
from .points import floor_height


@dataclass
class RoomParams:
    snap_m: float = 0.40  # an outline edge this close to a parallel wall face moves onto it
    floor_rise: float = 0.8  # the floor is within this of the room's lowest upward surface
    ceiling_above_floor: float = 1.6  # ceiling points are at least this high
    min_room_height: float = 2.0  # a "ceiling" lower than this over the room's centre is something else
    slope_min_deg: float = 2.0
    slope_max_deg: float = 45.0
    min_points: int = 300


def fit_plane(
    xyz: F64, tol: float = 0.02, iterations: int = 300
) -> tuple[F64, F64, NDArray[np.bool_]] | None:
    """(unit normal, point on the plane, inlier mask) of the plane with most points within `tol`."""
    if len(xyz) < 3:
        return None
    rng = np.random.default_rng(0)
    best = np.zeros(len(xyz), bool)
    for _ in range(iterations):
        a, b, c = xyz[rng.choice(len(xyz), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((xyz - a) @ n) < tol
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 3:
        return None
    c = xyz[best].mean(axis=0)
    _, _, vt = np.linalg.svd(xyz[best] - c, full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    return n, c, np.abs((xyz - c) @ n) < tol


def floor_level(up: F64, rise: float) -> float | None:
    """Floor height from upward points `up`: the lowest strong cluster within `rise` of the lowest
    (1st percentile) of them."""
    z = up[:, 2]
    lo = float(np.percentile(z, 1))
    band = (z >= lo - 0.01) & (z <= lo + rise)
    return (
        floor_height(up[band], np.tile([0.0, 0.0, 1.0], (int(band.sum()), 1))) if band.sum() >= 50 else None
    )


def ceiling_height(ceiling_z: float, plane: tuple[float, float, float] | None, x: float, y: float) -> float:
    return ceiling_z if plane is None else plane[0] + plane[1] * x + plane[2] * y


def fit_heights(
    polygon: list[tuple[float, float]],
    pts: F64,
    nrm: F64,
    p: RoomParams,
    exclude: list[Polygon] | None = None,
) -> tuple[float | None, float | None, tuple[float, float, float] | None]:
    """(floor_z, ceiling_z, ceiling plane (a, b, c): z = a + b x + c y, or None if flat) from the points."""
    inner = Polygon(polygon).buffer(-0.20)
    if inner.is_empty:
        return None, None, None
    x0, y0, x1, y1 = inner.bounds
    box = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
    idx = np.flatnonzero(box)
    idx = idx[contains_xy(inner, pts[idx, 0], pts[idx, 1])]
    q, n = pts[idx], nrm[idx]
    low = n[:, 2] > 0.9
    high = n[:, 2] < -0.5
    for e in exclude or []:  # stair treads and furniture tops aren't floor; a stair's underside isn't ceiling
        grown = e.buffer(0.05)
        low[low] &= ~contains_xy(grown, q[low, 0], q[low, 1])
        high[high] &= ~contains_xy(grown, q[high, 0], q[high, 1])
    floor = floor_level(q[low], p.floor_rise) if low.sum() >= p.min_points else None
    base = floor if floor is not None else 0.0
    high &= q[:, 2] > base + p.ceiling_above_floor
    if high.sum() < p.min_points:
        return floor, None, None
    found = fit_plane(q[high])
    if found is None or found[2].mean() < 0.3:
        return floor, None, None
    normal, c, inl = found
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
    centre = np.array(inner.centroid.coords[0])
    if tilt < p.slope_min_deg or tilt > p.slope_max_deg:
        flat = float(np.median(q[high][inl, 2]))
        return (floor, flat, None) if flat - base >= p.min_room_height else (floor, None, None)
    a = float(c[2] + (normal[0] * c[0] + normal[1] * c[1]) / normal[2])
    plane = (a, float(-normal[0] / normal[2]), float(-normal[1] / normal[2]))
    middle = ceiling_height(0.0, plane, *centre)
    return (floor, middle, plane) if middle - base >= p.min_room_height else (floor, None, None)


def wall_faces(walls: list[tuple[F64, F64, float]]) -> list[tuple[F64, F64, F64]]:
    """(start, end, normal into the space in front) of both faces of each (start, end, thickness)."""
    out: list[tuple[F64, F64, F64]] = []
    for a, c, t in walls:
        d = (c - a) / np.linalg.norm(c - a)
        n = np.array([-d[1], d[0]])
        out += [(a + n * t / 2, c + n * t / 2, n), (a - n * t / 2, c - n * t / 2, -n)]
    return out


def snap_outline(
    polygon: list[tuple[float, float]], walls: list[tuple[F64, F64, float]], p: RoomParams
) -> list[tuple[float, float]]:
    """The outline with each edge on the nearest parallel wall face looking into the room."""
    poly = orient(Polygon(polygon), 1.0)  # counter-clockwise: the room is left of each edge
    pts = np.array(poly.exterior.coords[:-1])
    faces = wall_faces(walls)
    lines: list[tuple[F64, F64]] = []  # (point, direction)
    for i in range(len(pts)):
        a, c = pts[i], pts[(i + 1) % len(pts)]
        length = float(np.linalg.norm(c - a))
        e = (c - a) / max(length, 1e-9)
        inward = np.array([-e[1], e[0]])
        mid = (a + c) / 2
        best: tuple[float, F64, F64] | None = None  # (distance, face start, face direction)
        for f0, f1, fn in faces:
            if fn @ inward < 0.995:
                continue
            dist = float((mid - f0) @ fn)
            s0, s1 = sorted([(f0 - a) @ e, (f1 - a) @ e])
            overlaps = min(length, s1) - max(0.0, s0) > 0.5 * length
            if abs(dist) < p.snap_m and overlaps and (best is None or abs(dist) < abs(best[0])):
                best = (dist, f0, (f1 - f0) / np.linalg.norm(f1 - f0))
        # snapped: the edge takes the face's line (direction too), else it keeps its own
        lines.append((a, e) if best is None else (best[1], best[2] if best[2] @ e > 0 else -best[2]))
    out: list[tuple[float, float]] = []
    for i in range(len(lines)):
        (p0, d0), (p1, d1) = lines[i - 1], lines[i]
        den = d0[0] * d1[1] - d0[1] * d1[0]
        if abs(den) < 0.1:  # (nearly) parallel neighbours: keep the vertex, on the current line
            v = p1 + ((pts[i] - p1) @ d1) * d1
        else:
            t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / den
            v = p0 + t * d0
        out.append((float(v[0]), float(v[1])))
    snapped = Polygon(out)
    if not snapped.is_valid or abs(snapped.area - poly.area) > 0.3 * poly.area:
        return polygon
    return out


def wall_heights(
    start: F64,
    end: F64,
    thickness: float,
    rooms: list[tuple[Polygon, float, float, tuple[float, float, float] | None]],
) -> tuple[float, float, float] | None:
    """(base_z, top at start, top at end) from the rooms beside the wall: (outline, floor, ceiling, plane)."""
    line = LineString([start, end])
    band = line.buffer(thickness / 2 + 0.15)
    beside = [r for r in rooms if r[0].exterior.intersection(band).length > 0.3 * line.length]
    if not beside:
        return None
    base = min(r[1] for r in beside)
    tops = [max(ceiling_height(r[2], r[3], *pt) for r in beside) for pt in (start, end)]
    return base, tops[0], tops[1]
