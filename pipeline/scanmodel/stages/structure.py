"""Stage 4, structure: RoomPlan + plane fitting -> data/<project>/scene/model.json.

1. RoomPlan walls become faces (the visible side of a wall) in the scene frame, oriented so the room
   is on the left (rooms come out counter-clockwise): the side its session's cameras were on, unless the
   session's own depth points on the face clearly face the other way.
   A face in another session's room (flat.ownership: RoomPlan kept adding what it saw from the next
   room's doorway or through glass) is left out, with a flag; so is an enclosed area that another
   session's cameras fill. A session's cameras exclude its doorway dwell at the end.
2. Each face is refined against a RANSAC plane in points.ply (points facing its room only: the far
   side of a thin wall is within reach), then snapped to a right angle when
   within `snap_deg` and, for a fitted face, when snapping moves its ends by at most `snap_tol_m`
   (so a long wall keeps a measured skew); then neighbouring faces of a room are trimmed/extended
   to meet at corners.
3. Faces are merged across sessions: the same face seen twice is averaged; two opposite faces a
   wall-thickness apart become one wall. Thickness is measured only where both faces were seen within
   one 30 s window of one session (its own depth points, in strips along the wall, openings left out):
   faces seen minutes apart are also apart by the drift in between. Otherwise the default thickness is
   added on the outside of the face. A wall records whether its back face was seen at all (eval
   scores a face nobody saw separately).
4. Openings attach to walls; objects and stairs become boxes (an object two sessions saw: the copy of
   the session whose room it is in), each object in the room it stands in; rooms come from the face loops. A
   session that walked through several rooms is split into the areas its faces enclose, named
   from RoomPlan's sections.
5. Sanity flags (fit quality), then overrides.json is applied last.
"""

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from shapely import LineString, MultiPoint, Point, Polygon, contains_xy
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union

from ..boxes import Box, BoxParams, fit_neighbours, fit_to_points, fit_to_walls
from ..config import from_repo
from ..geom import angle_mod90
from ..io import read_jsonl
from ..models import bundle as b
from ..models.common import F64, mat
from ..models.derived import PosedImage, Poses
from ..models.overrides import Overrides
from ..models.scene import (
    Fit,
    Flag,
    Object,
    Opening,
    OpeningKind,
    Representation,
    Room,
    Scene,
    Source,
    SourceKind,
    Stair,
    Wall,
)
from ..ownership import AREA_RATIO, FACE_PUSH_M, MIN_OTHER, Votes, trim_tails
from ..points import WALLS_DIR, read_points, read_wall_points
from ..roomplan import CapturedRoom, load_captured_room, short_id
from ..rooms import RoomParams, fit_heights, snap_outline, wall_heights

PARAMETRIC = {
    "storage",
    "table",
    "refrigerator",
    "stove",
    "oven",
    "dishwasher",
    "sink",
    "washerDryer",
    "bathtub",
}
COLLECTIONS = ("levels", "rooms", "walls", "openings", "objects", "stairs")


@dataclass
class Params:
    band_m: float = 0.08  # points this close to a RoomPlan face are candidates for its plane
    min_inliers: int = 200
    snap_deg: float = 2.0
    # A fitted face is snapped only if that moves its ends at most this far: half of eval's 2 cm p95
    # budget, leaving the rest for LiDAR noise and fit error. 1 cm allows ~1.1 deg on a 1 m face, ~0.3 deg on 4 m.
    snap_tol_m: float = 0.01
    corner_m: float = 0.30  # face ends this close are joined at the lines' intersection
    merge_offset_m: float = 0.03
    max_thickness_m: float = 0.90  # old masonry; above `thick_flag_m` a measured thickness is flagged
    thick_flag_m: float = 0.50
    min_thickness_m: float = 0.04
    default_thickness_m: float = 0.10
    max_rotation_deg: float = (
        10.0  # a fitted plane turned further than this from RoomPlan's face is another surface
    )
    strip_m: float = 0.50  # thickness is measured per strip of wall this long (half-overlapping)
    face_search_m: float = 0.15  # how far a session's own face may sit from the merged face (drift)
    thickness_agree_m: float = 0.02  # windows measuring one wall must agree this well, else the default
    min_room_width_m: float = 0.5  # narrower enclosed areas (slivers between a wall's faces) aren't rooms
    room_camera_share: float = 0.15  # an enclosed area is a session's room if this share of cameras was in it
    room_reach_m: float = 1.5  # with no such area, the room stays within this of the camera path
    close_gap_m: float = 1.0  # when splitting rooms, a face end is extended this far to meet another face
    # The cameras' side of a face is its room unless the session's own depth points on it (within
    # `orient_band_m`, at least `orient_min_points`) face the other way `orient_ratio` times as often:
    # cameras behind a short face, in the next room or across a doorway, still fall within its extent.
    orient_band_m: float = 0.15
    orient_min_points: int = 200
    orient_ratio: float = 2.0


@dataclass
class Face:
    id: str  # RoomPlan wall identifier
    session: str
    p0: F64
    p1: F64
    base_z: float
    top_z: float
    weight: float = 1.0
    fit: Fit | None = None
    corner: list[bool] = field(default_factory=lambda: [False, False])
    ids: list[str] = field(default_factory=list[str])
    sessions: list[str] = field(default_factory=list[str])
    note: str | None = None  # why refinement was skipped, for a flag
    flipped: str | None = None  # why the cameras' side was overruled, for a flag

    @property
    def dir(self) -> F64:
        d = self.p1 - self.p0
        return d / np.linalg.norm(d)

    @property
    def interior(self) -> F64:
        """Left normal: points into the room."""
        dx, dy = self.dir
        return np.array([-dy, dx])

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


def _xy(v: F64) -> F64:
    return np.asarray(v[:2], dtype=np.float64)


def _unit(v: F64) -> F64:
    return v / (np.linalg.norm(v) + 1e-12)


def _intersect(p: F64, d: F64, q: F64, e: F64) -> F64 | None:
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 0.5:  # need a clear angle between the lines
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p + t * d


# --------------------------------------------------------------------------- faces


def _faces_from_room(
    room: CapturedRoom, sid: str, s2f: F64, cams: F64, own: tuple[F64, F64] | None, p: Params
) -> list[Face]:
    """The session's RoomPlan walls as faces, room on the left; `own`: its depth points (see orient_face)."""
    faces: list[Face] = []
    for w in room.of("wall"):
        m = s2f @ w.transform
        c, x = m[:3, 3], _unit(_xy(m[:3, 0]))
        width, height = float(w.dimensions[0]), float(w.dimensions[1])
        p0, p1 = _xy(c) - x * width / 2, _xy(c) + x * width / 2
        f = Face(
            w.id, sid, p0, p1, float(c[2] - height / 2), float(c[2] + height / 2), ids=[w.id], sessions=[sid]
        )
        # Interior side: where this session's cameras were when in front of the face.
        rel = cams - p0
        along = rel @ f.dir
        near = (along > 0) & (along < f.length)
        side = (
            np.sign(rel[near] @ f.interior) if near.any() else np.sign((cams.mean(axis=0) - p0) @ f.interior)
        )
        if np.sum(side) < 0:
            f.p0, f.p1 = p1, p0
        if own is not None:
            orient_face(f, *own, p)
        faces.append(f)
    return faces


def orient_face(f: Face, pts: F64, nrm: F64, p: Params) -> None:
    """Flip `f` when its session's own depth points on it (normals towards the cameras that saw them)
    mostly face away from its room side: the camera vote counts every camera within the face's extent,
    however far behind it or out of sight."""
    rel = pts[:, :2] - f.p0
    d, s = rel @ f.interior, rel @ f.dir
    m = (
        (np.abs(d) < p.orient_band_m)
        & (s > 0.1)
        & (s < f.length - 0.1)
        & (pts[:, 2] > f.base_z + 0.15)
        & (pts[:, 2] < f.top_z - 0.15)
    )
    facing = nrm[m, :2] @ f.interior
    fwd, back = int((facing > 0.85).sum()), int((facing < -0.85).sum())
    if fwd + back >= p.orient_min_points and back > p.orient_ratio * fwd:
        f.p0, f.p1 = f.p1, f.p0
        f.flipped = f"room side from the session's own depth points ({back} face it, {fwd} the cameras' side)"


def fit_line(xy: F64, tol: float = 0.01, iterations: int = 300) -> F64:
    """Inliers of the 2D line with the most points within `tol` (RANSAC with a fixed seed: repeatable,
    unlike Open3D's parallel segment_plane), refined once by least squares."""
    if len(xy) < 2:
        return xy
    rng = np.random.default_rng(0)
    best = np.zeros(len(xy), bool)
    for _ in range(iterations):
        i, j = rng.choice(len(xy), 2, replace=False)
        d = xy[j] - xy[i]
        norm = float(np.linalg.norm(d))
        if norm < 0.05:
            continue
        inl = np.abs((xy - xy[i]) @ np.array([-d[1], d[0]]) / norm) < tol
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 2:
        return xy[:0]
    c = xy[best].mean(axis=0)
    _, _, vt = np.linalg.svd(xy[best] - c, full_matrices=False)
    return xy[np.abs((xy - c) @ np.array([-vt[0][1], vt[0][0]])) < tol]


def refine_face(f: Face, pts: F64, nrm: F64, p: Params) -> None:
    """Fit the face's plane in the fused points; keep RoomPlan's extent along the wall.

    Only points facing the face's room count: a partition thinner than the band has its other face,
    the next room's, within reach too, and fitting that would put this face on the wrong side."""
    rel = pts[:, :2] - f.p0
    d = rel @ f.interior
    s = rel @ f.dir
    m = (
        (np.abs(d) < p.band_m)
        & (s > 0.15)
        & (s < f.length - 0.15)
        & (pts[:, 2] > f.base_z + 0.15)
        & (pts[:, 2] < f.top_z - 0.15)
        & (nrm[:, :2] @ f.interior > 0.85)
    )
    if m.sum() < p.min_inliers:
        return
    xy = fit_line(pts[m, :2])
    if len(xy) < p.min_inliers:
        return
    centre = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - centre, full_matrices=False)
    direction = vt[0] if vt[0] @ f.dir > 0 else -vt[0]
    normal = np.array([-direction[1], direction[0]])
    resid = (xy - centre) @ normal
    old_mid = (f.p0 + f.p1) / 2
    shift = float(abs((old_mid - centre) @ normal))
    rot = float(np.degrees(np.arccos(np.clip(direction @ f.dir, -1, 1))))
    if rot > p.max_rotation_deg:
        f.note = f"not refined: the best plane near RoomPlan's face is turned {rot:.0f} deg (another surface)"
        return
    f.p0 = centre + ((f.p0 - centre) @ direction) * direction
    f.p1 = centre + ((f.p1 - centre) @ direction) * direction
    f.weight = float(len(xy))
    f.fit = Fit(inliers=len(xy), rms_m=float(np.sqrt(np.mean(resid**2))), shift_m=shift, rotation_deg=rot)


def snap_face(f: Face, p: Params) -> None:
    angle = float(np.degrees(np.arctan2(f.dir[1], f.dir[0])))
    delta = float(angle_mod90(angle))
    if abs(delta) >= p.snap_deg or delta == 0:
        return
    if f.fit is not None and f.length / 2 * np.sin(np.radians(abs(delta))) > p.snap_tol_m:
        return
    mid = (f.p0 + f.p1) / 2
    r = np.radians(-delta)
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    f.p0, f.p1 = mid + rot @ (f.p0 - mid), mid + rot @ (f.p1 - mid)


def _join_corners(faces: list[Face], p: Params) -> None:
    """Room loop: the end of one face meets the start of the next at the lines' intersection."""
    for a in faces:
        best: tuple[float, Face] | None = None
        for c in faces:
            if c is a:
                continue
            dist = float(np.linalg.norm(a.p1 - c.p0))
            if dist < p.corner_m and (best is None or dist < best[0]):
                best = (dist, c)
        if best is None:
            continue
        c = best[1]
        x = _intersect(a.p0, a.dir, c.p0, c.dir)
        if x is not None:
            a.p1, c.p0 = x.copy(), x.copy()
            a.corner[1] = c.corner[0] = True


# --------------------------------------------------------------------------- merging


def _overlap(a: Face, b2: Face) -> tuple[float, float]:
    """Overlap length and gap of b2 projected onto a's line."""
    s = sorted([(b2.p0 - a.p0) @ a.dir, (b2.p1 - a.p0) @ a.dir])
    lo, hi = max(0.0, s[0]), min(a.length, s[1])
    return float(hi - lo), float(max(0.0, s[0] - a.length, -s[1]))


def _merge_same(faces: list[Face], p: Params) -> list[Face]:
    """Same face seen in two sessions (or split by RoomPlan): one face, inlier-weighted."""
    out: list[Face] = []
    for f in sorted(faces, key=lambda f: -f.length):
        for g in out:
            if (
                f.interior @ g.interior > 0.99
                and abs((f.p0 - g.p0) @ g.interior) < p.merge_offset_m
                and abs((f.p1 - g.p0) @ g.interior) < p.merge_offset_m
                and _overlap(g, f)[1] < 0.10
            ):
                wf, wg = f.weight, g.weight
                offset = ((f.p0 + f.p1) / 2 - g.p0) @ g.interior * wf / (wf + wg)
                s = [0.0, g.length, (f.p0 - g.p0) @ g.dir, (f.p1 - g.p0) @ g.dir]
                base = g.p0 + offset * g.interior
                g.p0, g.p1 = base + min(s) * g.dir, base + max(s) * g.dir
                g.base_z, g.top_z = min(g.base_z, f.base_z), max(g.top_z, f.top_z)
                g.weight += wf
                g.ids += f.ids
                g.sessions += f.sessions
                g.corner = [g.corner[0] or f.corner[0], g.corner[1] or f.corner[1]]
                break
        else:
            out.append(f)
    return out


# --------------------------------------------------------------------------- thickness


@dataclass
class SessionPoints:
    """Each session's own wall points (normals towards its cameras) and the openings to leave out."""

    points: dict[str, tuple[F64, F64]]
    holes: list[tuple[F64, F64]] = field(default_factory=list[tuple[F64, F64]])  # opening ends, xy
    furniture: list[Polygon] = field(default_factory=list[Polygon])  # footprints: fronts aren't wall faces


def _peak(t: F64) -> float | None:
    """Offset of the dominant surface among `t` (5 mm bins, refined by the mean within 1.2 cm)."""
    if len(t) < 60:
        return None
    h, e = np.histogram(t, bins=np.arange(t.min() - 0.005, t.max() + 0.01, 0.005))
    hs = np.convolve(h, [1, 2, 1], "same")
    c = (e[hs.argmax()] + e[hs.argmax() + 1]) / 2
    near = np.abs(t - c) < 0.012
    if near.sum() < 40 or near.mean() < 0.3:
        return None
    return float(t[near].mean())


def measure_thickness(
    a: Face, lo: float, hi: float, sp: SessionPoints, p: Params
) -> tuple[float, dict[str, float]] | None:
    """Wall thickness behind face `a` (from `lo` to `hi` along it) from sessions that saw both faces.

    Per session and per strip: its own face near `a`, and the opposite-facing surface behind it.
    A session needs two strips; the result is the median over sessions (and their values).
    """
    out: dict[str, float] = {}
    for sid, (pts, nrm) in sp.points.items():
        rel = pts[:, :2] - a.p0
        s, t = rel @ a.dir, rel @ a.interior
        z0 = a.base_z + 1.2  # above furniture
        m = (
            (s > lo + 0.05)
            & (s < hi - 0.05)
            & (t > -p.face_search_m - p.max_thickness_m)
            & (t < p.face_search_m)
            & (pts[:, 2] > z0)
            & (pts[:, 2] < min(z0 + 1.0, a.top_z - 0.1))
        )
        for e0, e1 in sp.holes:  # openings: glass, door leaves and reveals aren't the wall's faces
            if abs((e0 - a.p0) @ a.interior) < 0.4 and abs((e1 - a.p0) @ a.interior) < 0.4:
                h0, h1 = sorted([(e0 - a.p0) @ a.dir, (e1 - a.p0) @ a.dir])
                m &= (s < h0 - 0.1) | (s > h1 + 0.1)
        if m.sum() < 100:
            continue
        nd = nrm[:, :2] @ a.interior
        front, back = m & (nd > 0.9) & (np.abs(t) < p.face_search_m), m & (nd < -0.9)
        for blocked in sp.furniture:  # a cabinet front before a hidden far face is not that face
            back[back] &= ~contains_xy(blocked, pts[back, 0], pts[back, 1])
        gaps: list[float] = []
        for s0 in np.arange(lo + 0.05, hi - 0.05 - p.strip_m / 2, p.strip_m / 2):
            strip = (s >= s0) & (s < s0 + p.strip_m)
            tf = _peak(t[front & strip])
            if tf is None:
                continue
            behind = back & strip & (t < tf - p.min_thickness_m) & (t > tf - p.max_thickness_m)
            tb = _peak(t[behind])
            if tb is not None:
                gaps.append(tf - tb)
        if len(gaps) >= 2:
            out[sid] = float(np.median(gaps))
    if not out:
        return None
    return float(np.median(list(out.values()))), out


def walls_from_faces(
    faces: list[Face], p: Params, sp: SessionPoints | None = None
) -> tuple[list[Wall], dict[str, str], list[Flag]]:
    walls: list[Wall] = []
    face_to_wall: dict[str, str] = {}
    flags: list[Flag] = []
    used: set[int] = set()
    order = sorted(range(len(faces)), key=lambda i: -faces[i].length)
    for i in order:
        if i in used:
            continue
        a = faces[i]
        partner: tuple[float, int] | None = None
        for j in order:
            if j == i or j in used:
                continue
            c = faces[j]
            sep = float(((c.p0 + c.p1) / 2 - a.p0) @ a.interior)  # c lies behind a: sep < 0
            ov, _ = _overlap(a, c)
            if (
                a.interior @ c.interior < -0.99
                and p.min_thickness_m < -sep < p.max_thickness_m
                and ov > 0.3 * min(a.length, c.length)
                and (partner is None or ov > partner[0])
            ):
                partner = (ov, j)
        used.add(i)
        ids = list(a.ids)
        sessions = list(a.sessions)
        fit = a.fit
        c = None
        if partner is not None:
            c = faces[partner[1]]
            used.add(partner[1])
        gap = float(-(((c.p0 + c.p1) / 2 - a.p0) @ a.interior)) if c is not None else None
        if c is not None:
            s = [0.0, a.length, float((c.p0 - a.p0) @ a.dir), float((c.p1 - a.p0) @ a.dir)]
            span = (min(s), max(s))
        else:
            span = (0.0, a.length)
        found = measure_thickness(a, *span, sp, p) if sp is not None else None
        disagree = found is not None and np.ptp(list(found[1].values())) > p.thickness_agree_m
        measured = found is not None and not disagree
        thickness = found[0] if measured and found is not None else p.default_thickness_m
        centre_off = -thickness / 2
        if c is not None:
            lo, hi = span
            base_z, top_z = min(a.base_z, c.base_z), max(a.top_z, c.top_z)
            ids += c.ids
            sessions += c.sessions
            kind = SourceKind.MERGED
        else:
            # Extend into corners so single-sided walls close the outside corner.
            lo = -thickness / 2 if a.corner[0] else 0.0
            hi = a.length + (thickness / 2 if a.corner[1] else 0.0)
            base_z, top_z = a.base_z, a.top_z
            kind = SourceKind.PLANE_FIT if fit else SourceKind.ROOMPLAN
            if len(a.ids) > 1:
                kind = SourceKind.MERGED
        wid = short_id("wall", sorted(ids)[0])
        if disagree and found is not None:
            per = ", ".join(f"{v * 100:.1f}" for v in found[1].values())
            flags.append(
                Flag(
                    element=wid,
                    severity="info",
                    message=f"windows measure {per} cm thick: disagree, default used",
                )
            )
        if measured and thickness > p.thick_flag_m:
            flags.append(
                Flag(
                    element=wid,
                    severity="info",
                    message=f"measured {thickness * 100:.0f} cm thick: old masonry?",
                )
            )
        if gap is not None and (not measured or abs(gap - thickness) > 0.02):
            what = f"measured {thickness * 100:.1f} cm" if measured else "not measured, default used"
            flags.append(
                Flag(
                    element=wid,
                    severity="info",
                    message=f"faces from two sessions are {gap * 100:.1f} cm apart ({what}): drift between sessions?",
                )
            )
        base = a.p0 + centre_off * a.interior
        if abs(base_z) < 0.05:
            base_z = 0.0
        walls.append(
            Wall(
                id=wid,
                start=(float(base[0] + lo * a.dir[0]), float(base[1] + lo * a.dir[1])),
                end=(float(base[0] + hi * a.dir[0]), float(base[1] + hi * a.dir[1])),
                thickness=thickness,
                thickness_measured=measured,
                faces_observed=(True, c is not None or measured),  # side -1 is behind `a`
                base_z=base_z,
                height=top_z - base_z,
                rooms=sorted({f"room-{s}" for s in sessions}),
                source=Source(kind=kind, sessions=sorted(set(sessions)), roomplan_ids=sorted(ids), fit=fit),
            )
        )
        for fid in ids:
            face_to_wall[fid] = wid
    return walls, face_to_wall, flags


# --------------------------------------------------------------------------- openings, rooms, objects


def _openings(
    rooms: dict[str, CapturedRoom], s2f: dict[str, F64], walls: list[Wall], face_to_wall: dict[str, str]
) -> list[Opening]:
    by_id = {w.id: w for w in walls}
    out: list[Opening] = []
    for sid, room in rooms.items():
        for o in [s for s in room.surfaces if s.kind in ("door", "window", "opening")]:
            m = s2f[sid] @ o.transform
            c = m[:3, 3]
            wall = by_id.get(face_to_wall.get(o.parent or "", ""))
            if wall is None:  # nearest parallel wall
                x = _unit(_xy(m[:3, 0]))
                best = min(
                    walls,
                    key=lambda w: (
                        abs((_xy(c) - np.array(w.start)) @ w.normal()) + 10 * (1 - abs(x @ w.direction()))
                    ),
                    default=None,
                )
                if (
                    best is None
                    or abs((_xy(c) - np.array(best.start)) @ best.normal()) > best.thickness / 2 + 0.15
                ):
                    continue
                wall = best
            width, height = float(o.dimensions[0]), float(o.dimensions[1])
            bottom = max(0.0, float(c[2] - height / 2 - wall.base_z))
            if o.kind == "door" and bottom < 0.05:
                bottom = 0.0
            new = Opening(
                id=short_id(o.kind, o.id),
                kind=OpeningKind(o.kind),
                wall=wall.id,
                offset=float((_xy(c) - np.array(wall.start)) @ wall.direction()),
                width=width,
                bottom=bottom,
                height=height,
                is_open=o.is_open,
                source=Source(kind=SourceKind.ROOMPLAN, sessions=[sid], roomplan_ids=[o.id]),
            )
            dup = next(
                (
                    e
                    for e in out
                    if e.wall == new.wall
                    and e.kind == new.kind
                    and abs(e.offset - new.offset) < new.width / 2
                ),
                None,
            )
            if dup is None:
                out.append(new)
            else:  # the same door scanned from both rooms
                dup.offset, dup.width = (dup.offset + new.offset) / 2, (dup.width + new.width) / 2
                dup.height, dup.bottom = (dup.height + new.height) / 2, (dup.bottom + new.bottom) / 2
                dup.source.kind = SourceKind.MERGED
                dup.source.sessions = sorted({*dup.source.sessions, sid})
                dup.source.roomplan_ids = sorted({*dup.source.roomplan_ids, o.id})
    return out


def door_leaf(o: Opening, w: Wall, pts: F64, nrm: F64, cell: float = 0.10) -> str:
    """What the scan shows in a door opening: "closed" (surface filling it, in the wall's plane), "open"
    (a panel at one jamb, swung into either room, facing along the wall), or "none" (a doorway).
    Coverage of 10 cm cells, so a frame or a handle doesn't pass for a leaf."""
    a, d, n = np.array(w.start), w.direction(), w.normal()
    rel = pts[:, :2] - a
    s, t = rel @ d, rel @ n
    z = pts[:, 2] - w.base_z - o.bottom
    lo, hi = o.offset - o.width / 2, o.offset + o.width / 2
    zs = (z > 0.2) & (z < o.height - 0.2)
    rows = max(1, int((o.height - 0.4) / cell))

    def cover(u: F64, v: F64, umax: float) -> float:
        cols = max(1, int(umax / cell))
        cells = {(int(a_ / cell), int(b_ / cell)) for a_, b_ in zip(u, v, strict=True) if 0 <= a_ < umax}
        return len(cells) / (cols * rows)

    # Closed: surface across the middle half of the opening (frames, architraves and RoomPlan's loose
    # door edges are at its sides; a leaf fills the middle).
    mid0, mid1 = o.offset - o.width / 4, o.offset + o.width / 4
    plane = zs & (s > mid0) & (s < mid1) & (np.abs(t) < w.thickness / 2 + 0.06)
    plane &= np.abs(nrm[:, :2] @ n) > 0.8
    if cover(s[plane] - mid0, z[plane] - 0.2, mid1 - mid0) > 0.4:
        return "closed"
    for hinge in (lo, hi):
        for side in (1, -1):
            out = side * t - w.thickness / 2  # distance out from this face into the room
            leaf = zs & (np.abs(s - hinge) < 0.12) & (out > 0.02) & (np.abs(nrm[:, :2] @ d) > 0.8)
            if cover(out[leaf], z[leaf] - 0.2, o.width - 0.05) > 0.3:
                return "open"
    return "none"


def _check_door(o: Opening, w: Wall, pts: F64, nrm: F64) -> Opening:
    """A RoomPlan "door" with no leaf in the scan is a doorway (RoomPlan calls a framed gap a door)."""
    leaf = door_leaf(o, w, pts, nrm)
    if leaf == "none":
        return o.model_copy(update={"kind": OpeningKind.OPENING, "is_open": None})
    return o.model_copy(update={"is_open": leaf == "open"})


def _room_polygon(faces: list[Face]) -> list[tuple[float, float]]:
    """Chain faces end-to-start into a loop; fall back to the convex hull of their ends."""
    if not faces:
        return []
    loop = [faces[0]]
    while len(loop) <= len(faces):
        nxt = min(faces, key=lambda f: float(np.linalg.norm(f.p0 - loop[-1].p1)))
        if np.linalg.norm(nxt.p0 - loop[-1].p1) > 1e-6:
            break
        if nxt is loop[0]:
            return [(float(f.p0[0]), float(f.p0[1])) for f in loop]
        loop.append(nxt)
    return _hull([(float(q[0]), float(q[1])) for f in faces for q in (f.p0, f.p1)])


@dataclass
class Area:
    polygon: Polygon
    labels: list[str]  # RoomPlan section labels whose centre lies inside, in room.json order
    ceiling_z: float


def _areas(faces: list[Face], sections: list[tuple[str, F64]], p: Params) -> list[Area]:
    """Areas enclosed by one session's faces, ordered by centroid (x, then y).

    Each face end is extended to the nearest face it points at within `close_gap_m` (a wall RoomPlan
    stopped short of its neighbour, a doorway with no wall above it), else by `corner_m`; the lines are
    then polygonized, and slivers (narrower than `min_room_width_m`) are dropped; small rooms (a WC, a
    cupboard) are kept.
    """
    segs = [LineString([f.p0 - f.dir * p.corner_m, f.p1 + f.dir * p.corner_m]) for f in faces]

    def reach(i: int, end: F64, out: F64) -> F64:
        ray = LineString([end, end + out * p.close_gap_m])
        hits = [ray.intersection(g) for j, g in enumerate(segs) if j != i]
        dist = [Point(end).distance(h) for h in hits if not h.is_empty]
        return end + out * max(p.corner_m, min(dist) + 0.01 if dist else 0.0)

    lines = [LineString([reach(i, f.p0, -f.dir), reach(i, f.p1, f.dir)]) for i, f in enumerate(faces)]
    out: list[Area] = []
    for poly in polygonize(unary_union(lines)):
        if not is_room(poly, p):
            continue
        poly = orient(cast(Polygon, Polygon(poly.exterior).simplify(0.01)), 1.0)
        near = [f.top_z for f in faces if poly.exterior.distance(Point((f.p0 + f.p1) / 2)) < p.corner_m]
        out.append(
            Area(
                poly,
                [label for label, c in sections if poly.contains(Point(c))],
                float(np.median(near or [f.top_z for f in faces])),
            )
        )
    return sorted(out, key=lambda a: (round(a.polygon.centroid.x, 1), round(a.polygon.centroid.y, 1)))


def is_room(poly: Polygon, p: Params) -> bool:
    """Not a sliver: the shorter side of its smallest enclosing rectangle is at least min_room_width_m."""
    if poly.area < 0.05:  # degenerate leftovers of polygonize
        return False
    with np.errstate(divide="ignore", invalid="ignore"):  # shapely divides by 0 on axis-aligned shapes
        rect = np.array(cast(Polygon, poly.minimum_rotated_rectangle).exterior.coords[:4])
    sides = np.linalg.norm(np.diff(np.vstack([rect, rect[:1]]), axis=0), axis=1)
    return float(sides.min()) >= p.min_room_width_m


def _section_name(label: str) -> str | None:
    """livingRoom -> "living room"; RoomPlan's "unidentified" -> None."""
    return None if label == "unidentified" else re.sub(r"(?<!^)(?=[A-Z])", " ", label).lower()


def session_rooms(
    sid: str,
    faces: list[Face],
    room: CapturedRoom,
    s2f: F64,
    session_name: str | None,
    p: Params,
    cams: F64 | None = None,
    votes: Votes | None = None,
) -> tuple[list[Room], list[Flag]]:
    """One room per session, or one per enclosed area when the session covered several rooms.

    A session's rooms are the enclosed areas its cameras were in (at least `room_camera_share` of them):
    RoomPlan also returns walls seen through windows and doorways (a garden fence, the next room), and
    those must not make the room bigger. With no such area, the outline through all faces is clipped to
    within `room_reach_m` of the camera path.

    Split room ids come from the section name (room-<session>-living-room), else the area's index,
    so overrides.json `room_names` keeps pointing at the same room across re-runs.

    An area that another session's cameras (`votes`) fill `AREA_RATIO` times as much as this one's is that
    session's room (seen from a doorway), not this one's.
    """
    if len(faces) < 3:
        return [], []
    source = Source(kind=SourceKind.PLANE_FIT, sessions=[sid])
    sections = [(sec.label, (s2f @ np.append(sec.center, 1.0))[:2]) for sec in room.sections]
    areas = _areas(faces, sections, p)
    foreign: list[Flag] = []
    if votes is not None and len(votes.xy):
        mine: list[Area] = []
        for a in areas:
            c = votes.inside(contains_xy(a.polygon, votes.xy[:, 0], votes.xy[:, 1]))
            other, n = max(((s, k) for s, k in c.items() if s != sid), key=lambda x: x[1], default=("", 0))
            if n >= MIN_OTHER and n >= AREA_RATIO * c.get(sid, 0):
                foreign.append(Flag(severity="info", message=f"session {sid}: an enclosed area of {a.polygon.area:.1f} m²"
                    f" left out: {other}'s room ({n} of its cameras in it, {c.get(sid, 0)} of this session's)"))  # fmt: skip
            else:
                mine.append(a)
        areas = mine
    if cams is not None and len(cams):
        areas = [
            a for a in areas if contains_xy(a.polygon, cams[:, 0], cams[:, 1]).mean() >= p.room_camera_share
        ]
    if len(areas) < 2:
        if areas:
            ceiling = areas[0].ceiling_z
            polygon = [(float(x), float(y)) for x, y in list(areas[0].polygon.exterior.coords)[:-1]]
        else:
            ceiling = float(np.median([f.top_z for f in faces]))
            polygon = _room_polygon(faces)
            if cams is not None and len(cams) >= 3:
                reach = MultiPoint([tuple(c) for c in cams]).convex_hull.buffer(p.room_reach_m)
                near = Polygon(polygon).intersection(reach)
                parts = [
                    g for g in getattr(near, "geoms", [near]) if isinstance(g, Polygon) and not g.is_empty
                ]
                if parts:
                    biggest = orient(max(parts, key=lambda g: g.area), 1.0)
                    polygon = [(float(x), float(y)) for x, y in list(biggest.exterior.coords)[:-1]]
        return [
            Room(
                id=f"room-{sid}",
                name=session_name,
                session=sid,
                polygon=polygon,
                ceiling_z=ceiling,
                source=source,
            )
        ], foreign
    names = [[n for n in dict.fromkeys(map(_section_name, a.labels)) if n] for a in areas]
    largest = max(range(len(areas)), key=lambda i: areas[i].polygon.area)
    rooms: list[Room] = []
    used: set[str] = set()
    for k, (a, ns) in enumerate(zip(areas, names, strict=True), 1):
        slug = "-".join(ns[0].split()) if ns else str(k)
        if slug in used:
            slug = str(k)
        used.add(slug)
        name = (
            " / ".join(ns) or (session_name if not any(names) and k - 1 == largest else None) or f"room {k}"
        )
        polygon = [(float(x), float(y)) for x, y in list(a.polygon.exterior.coords)[:-1]]
        rooms.append(
            Room(
                id=f"room-{sid}-{slug}",
                name=name,
                session=sid,
                polygon=polygon,
                ceiling_z=a.ceiling_z,
                source=source,
            )
        )
    flags = [
        *foreign,
        Flag(severity="info", message=f"session {sid} split into {len(rooms)} rooms by its walls"),
    ]
    enclosed = {label for a in areas for label in a.labels}
    flags += [
        Flag(severity="info", message=f"session {sid}: RoomPlan section '{label}' is not enclosed by walls")
        for label, _ in sections
        if label not in enclosed
    ]
    return rooms, flags


def _hull(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Counter-clockwise convex hull (monotone chain)."""
    pts = sorted(set(pts))

    def cross(o: tuple[float, float], a: tuple[float, float], c: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (c[1] - o[1]) - (a[1] - o[1]) * (c[0] - o[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def _objects(
    rooms: dict[str, CapturedRoom], s2f: dict[str, F64], pts: F64, nrm: F64, votes: Votes | None = None
) -> tuple[list[Object], list[Stair]]:
    """RoomPlan's objects and stairs. An object two sessions saw is kept once: the copy of the session whose
    room it stands in (`votes`), else the larger."""
    objects: list[Object] = []
    stairs: list[Stair] = []
    for sid, room in rooms.items():
        for o in room.objects:
            m = s2f[sid] @ o.transform
            c = m[:3, 3]
            x = _unit(_xy(m[:3, 0]))
            yaw = float(np.degrees(np.arctan2(x[1], x[0])))
            size = (float(o.dimensions[0]), float(o.dimensions[2]), float(o.dimensions[1]))
            centre = (float(c[0]), float(c[1]), float(c[2]))
            src = Source(kind=SourceKind.ROOMPLAN, sessions=[sid], roomplan_ids=[o.id])
            if o.category == "stairs":
                stairs.append(
                    Stair(
                        id=short_id("stair", o.id),
                        center=centre,
                        size=size,
                        yaw_deg=yaw,
                        tread_z=_treads(centre, size, yaw, pts, nrm),
                        source=src,
                    )
                )
                continue
            new = Object(
                id=short_id("object", o.id),
                category=o.category,
                center=centre,
                size=size,
                yaw_deg=yaw,
                representation=Representation.PARAMETRIC if o.category in PARAMETRIC else Representation.BOX,
                room=f"room-{sid}",
                source=src,
            )
            dup = next((e for e in objects if same_object(e, new)), None)
            if dup is None:
                objects.append(new)
                continue
            owner = votes.owner(np.array(centre[:2])) if votes is not None else None
            mine, theirs = owner in new.source.sessions, owner in dup.source.sessions
            if mine != theirs:
                if mine:
                    objects[objects.index(dup)] = new
            elif np.prod(new.size) > np.prod(dup.size):
                objects[objects.index(dup)] = new
    return objects, stairs


def _yaw(m: F64) -> float:
    x = _unit(_xy(m[:3, 0]))
    return float(np.degrees(np.arctan2(x[1], x[0])))


def _footprint(
    center: tuple[float, float, float], size: tuple[float, float, float], yaw_deg: float
) -> Polygon:
    box = Box.of(center, size, yaw_deg)
    x, y = box.axes
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    return Polygon([box.centre + sx * box.half["+x"] * x + sy * box.half["+y"] * y for sx, sy in corners])


def _on_floor(o: Object, rooms: list[Room], snap: float = 0.08) -> Object:
    """An object whose bottom is within `snap` of its room's floor stands on it."""
    room = next((r for r in rooms if Polygon(r.polygon).contains(Point(o.center[0], o.center[1]))), None)
    if room is None:
        return o
    bottom, top = o.center[2] - o.size[2] / 2, o.center[2] + o.size[2] / 2
    if abs(bottom - room.floor_z) >= snap or top - room.floor_z < 0.05:
        return o
    return o.model_copy(
        update={
            "center": (o.center[0], o.center[1], (room.floor_z + top) / 2),
            "size": (o.size[0], o.size[1], top - room.floor_z),
        }
    )


def _in_room(o: Object, rooms: list[Room]) -> Object:
    """The room an object stands in (the nearest if none holds its centre), whichever session saw it."""
    if not rooms:
        return o
    c = Point(o.center[0], o.center[1])
    room = min(rooms, key=lambda r: Polygon(r.polygon).distance(c))
    return o if room.id == o.room else o.model_copy(update={"room": room.id})


def _fit_rooms(
    rooms: list[Room], walls: list[Wall], pts: F64, nrm: F64, exclude: list[Polygon]
) -> tuple[list[Room], list[Wall]]:
    """Outlines onto the wall faces, floor and ceiling from the points, walls from floor to ceiling."""
    rp = RoomParams()
    lines = [(np.array(w.start), np.array(w.end), w.thickness) for w in walls]
    fitted: list[Room] = []
    for r in rooms:
        polygon = snap_outline(list(r.polygon), lines, rp)
        floor, ceiling, plane = fit_heights(polygon, pts, nrm, rp, exclude)
        fitted.append(
            r.model_copy(
                update={
                    "polygon": polygon,
                    "floor_z": r.floor_z if floor is None else floor,
                    "ceiling_z": r.ceiling_z if ceiling is None else ceiling,
                    "ceiling_plane": plane,
                }
            )
        )
    beside = [(Polygon(r.polygon), r.floor_z, r.ceiling_z, r.ceiling_plane) for r in fitted]
    out: list[Wall] = []
    for w in walls:
        found = wall_heights(np.array(w.start), np.array(w.end), w.thickness, beside)
        if found is None or min(found[1:]) - found[0] < 0.5:
            out.append(w)
            continue
        base, top0, top1 = found
        out.append(
            w.model_copy(
                update={
                    "base_z": base,
                    "height": top0 - base,
                    "height_end": top1 - base if abs(top1 - top0) > 0.02 else None,
                }
            )
        )
    return fitted, out


def _refine_objects(
    objects: list[Object], walls: list[tuple[F64, F64, float]], pts: F64, nrm: F64
) -> list[Object]:
    """RoomPlan's loose boxes fitted to the points, the walls and each other (see scanmodel.boxes)."""
    bp = BoxParams()
    found = [Box.of(o.center, o.size, o.yaw_deg) for o in objects]
    for box in found:
        fit_to_points(box, pts, nrm, bp)
        fit_to_walls(box, walls, bp.wall_gap, bp.wall_overlap)
    fit_neighbours(found, bp)
    out: list[Object] = []
    for o, box in zip(objects, found, strict=True):
        center, size = box.recentred()
        out.append(o.model_copy(update={"center": center, "size": size}) if min(size) > 0.02 else o)
    return out


def merge_stairs(stairs: list[Stair], walls: list[tuple[F64, F64, float]]) -> list[Stair]:
    """The same flight seen by several sessions becomes one (the one with most fitted treads); then a
    flight reaching a little way into a wall ends at its face."""

    def footprint(st: Stair) -> Polygon:
        return _footprint(st.center, st.size, st.yaw_deg)

    out: list[Stair] = []
    for st in sorted(stairs, key=lambda st: (-len(st.tread_z), -st.size[0] * st.size[1])):
        fp = footprint(st)
        dup = next(
            (
                i
                for i, kept in enumerate(out)
                if abs(kept.center[2] - st.center[2]) < 0.4
                and fp.intersection(footprint(kept)).area > 0.5 * min(fp.area, footprint(kept).area)
            ),
            None,
        )
        if dup is None:
            out.append(st)
            continue
        kept = out[dup]
        out[dup] = kept.model_copy(
            update={
                "source": Source(
                    kind=SourceKind.MERGED,
                    sessions=sorted({*kept.source.sessions, *st.source.sessions}),
                    roomplan_ids=sorted({*kept.source.roomplan_ids, *st.source.roomplan_ids}),
                )
            }
        )
    clipped: list[Stair] = []
    for st in out:
        box = Box.of(st.center, st.size, st.yaw_deg)
        fit_to_walls(box, walls, gap=0.0, overlap=0.30)
        center, size = box.recentred()
        clipped.append(st.model_copy(update={"center": center, "size": size}))
    return clipped


def same_object(a: Object, c: Object) -> bool:
    """The same piece seen by two sessions: same category, footprints overlapping by more than half
    (the smaller one), similar height. Never within one session: RoomPlan reports a run of identical
    units as separate objects."""
    if a.category != c.category or set(a.source.sessions) & set(c.source.sessions):
        return False
    fa, fc = _footprint(a.center, a.size, a.yaw_deg), _footprint(c.center, c.size, c.yaw_deg)
    return fa.intersection(fc).area > 0.5 * min(fa.area, fc.area) and abs(a.center[2] - c.center[2]) < 0.3


def _treads(
    centre: tuple[float, float, float], size: tuple[float, float, float], yaw: float, pts: F64, nrm: F64
) -> list[float]:
    r = np.radians(-yaw)
    rel = pts - np.array(centre)
    lx = rel[:, 0] * np.cos(r) - rel[:, 1] * np.sin(r)
    ly = rel[:, 0] * np.sin(r) + rel[:, 1] * np.cos(r)
    inside = (
        (np.abs(lx) < size[0] / 2) & (np.abs(ly) < size[1] / 2) & (np.abs(rel[:, 2]) < size[2] / 2 + 0.05)
    )
    z = pts[inside & (nrm[:, 2] > 0.9), 2]
    if z.size < 50:
        return []
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.01))
    peaks: list[float] = []
    for i in np.argsort(-hist):
        if hist[i] < max(20, 0.15 * hist.max()):
            break
        zc = float((edges[i] + edges[i + 1]) / 2)
        if all(abs(zc - q) > 0.10 for q in peaks):
            peaks.append(zc)
    return sorted(round(q, 3) for q in peaks)


# --------------------------------------------------------------------------- checks + overrides


def merge_patch(target: Any, patch: Any) -> Any:
    """RFC 7396 JSON merge patch."""
    if not isinstance(patch, dict):
        return patch
    out: dict[str, Any] = dict(cast(dict[str, Any], target)) if isinstance(target, dict) else {}
    for k, v in cast(dict[str, Any], patch).items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = merge_patch(out.get(k), v)
    return out


def apply_overrides(scene: Scene, ov: Overrides) -> Scene:
    data = scene.model_dump(mode="json")
    flags: list[dict[str, Any]] = data["flags"]
    removed = set(ov.remove)
    for coll in COLLECTIONS:
        data[coll] = [e for e in data[coll] if e["id"] not in removed]
        data[coll] += [
            merge_patch(
                e.model_dump(mode="json"),
                {"source": {"kind": "override"}} if "source" in type(e).model_fields else {},
            )
            for e in getattr(ov.add, coll)
        ]
    index = {e["id"]: (coll, i) for coll in COLLECTIONS for i, e in enumerate(data[coll])}
    for eid, patch in ov.patch.items():
        if eid not in index:
            flags.append(
                {"element": eid, "severity": "warn", "message": "overrides.json patches an unknown id"}
            )
            continue
        coll, i = index[eid]
        data[coll][i] = merge_patch(data[coll][i], patch)
    for room in data["rooms"]:
        if room["id"] in ov.room_names:
            room["name"] = ov.room_names[room["id"]]
    walls = {w["id"] for w in data["walls"]}
    for o in [o for o in data["openings"] if o["wall"] not in walls]:
        data["openings"].remove(o)
        flags.append(
            {"element": o["id"], "severity": "warn", "message": f"opening dropped: wall {o['wall']} gone"}
        )
    for key in ("views", "paths"):
        mine = {v.id: v.model_dump(mode="json") for v in getattr(ov, key)}
        data[key] = [mine.pop(v["id"], v) for v in data[key]] + list(mine.values())
    return Scene.model_validate(data)


def _round(v: Any, nd: int = 4) -> Any:
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, list):
        return [_round(x, nd) for x in cast(list[Any], v)]
    if isinstance(v, dict):
        return {k: _round(x, nd) for k, x in cast(dict[str, Any], v).items()}
    return v


def write_scene(scene: Scene, path: Path) -> None:
    """Stable, diff-friendly output: no timestamp, 0.1 mm rounding, None fields omitted."""
    data = _round(json.loads(scene.model_dump_json(exclude_none=True)))
    path.write_text(json.dumps(data, indent=2) + "\n")


# --------------------------------------------------------------------------- split sessions

PART_NEAR_M = 0.04  # a sample of a RoomPlan element on a fused point: this close
PART_MARGIN = 0.15  # another part's move wins by this much more of the samples on fused points


def _samples(m: F64, dims: F64, box: bool) -> F64:
    """Points over a RoomPlan surface (its local x-y plane) or box (its six faces), through `m`, scene frame."""
    w, h, d = (float(x) for x in dims)
    u = np.linspace(-0.45, 0.45, 9)
    gu, gv = np.meshgrid(u, u, indexing="ij")
    gu, gv = gu.ravel(), gv.ravel()
    if not box:
        local = np.column_stack([gu * w, gv * h, np.zeros_like(gu)])
    else:
        faces = []
        for axis, size in enumerate((w, h, d)):
            for sign in (-0.5, 0.5):
                q = np.zeros((len(gu), 3))
                o = [a for a in range(3) if a != axis]
                q[:, o[0]] = gu * (w, h, d)[o[0]]
                q[:, o[1]] = gv * (w, h, d)[o[1]]
                q[:, axis] = sign * size
                faces.append(q)
        local = np.concatenate(faces)
    return local @ m[:3, :3].T + m[:3, 3]


def place_parts(room: CapturedRoom, s2f: F64, parts: list[F64], pts: F64) -> tuple[CapturedRoom, int]:
    """A session refine cut in two (a trajectory jump inside it): RoomPlan kept mapping what it saw after the
    jump in the jumped frame, so each element is placed with whichever move (`s2f` or one of `parts`, ARKit
    world -> scene) puts more of it on the fused points (`pts`); the element's transform is rewritten so that
    `s2f` places it there. Doors, windows and openings follow their wall. Returns the room and how many
    elements moved to another part."""
    import open3d.core as o3c

    nns = o3c.nns.NearestNeighborSearch(o3c.Tensor.from_numpy(np.ascontiguousarray(pts, np.float32)))
    nns.hybrid_index(PART_NEAR_M)
    cands = [s2f, *parts]

    def support(m: F64, dims: F64, box: bool) -> float:
        q = _samples(m, dims, box)
        _, _, counts = nns.hybrid_search(
            o3c.Tensor.from_numpy(np.ascontiguousarray(q, np.float32)), PART_NEAR_M, 1
        )
        return float(np.mean(counts.numpy() > 0))

    def best(transform: F64, dims: F64, box: bool) -> int:
        scores = [support(c @ transform, dims, box) for c in cands]
        k = int(np.argmax(scores))
        return k if k and scores[k] >= scores[0] + PART_MARGIN else 0

    choice: dict[str, int] = {}
    for srf in room.surfaces:
        if srf.kind == "wall":
            choice[srf.id] = best(srf.transform, srf.dimensions, False)
    for srf in room.surfaces:
        if srf.kind != "wall":
            choice[srf.id] = (
                choice[srf.parent] if srf.parent in choice else best(srf.transform, srf.dimensions, False)
            )
    for o in room.objects:
        choice[o.id] = best(o.transform, o.dimensions, True)

    def moved(t: F64, k: int) -> F64:
        return t if k == 0 else np.linalg.inv(s2f) @ cands[k] @ t

    out = CapturedRoom(
        [replace(srf, transform=moved(srf.transform, choice[srf.id])) for srf in room.surfaces],
        [replace(o, transform=moved(o.transform, choice[o.id])) for o in room.objects],
        room.sections,
    )
    return out, sum(k > 0 for k in choice.values())


# --------------------------------------------------------------------------- stage


def run(
    poses_dir: Path,
    fuse_dir: Path,
    overrides_path: Path,
    out_path: Path,
    params: Params | None = None,
) -> dict[str, object]:
    p = params or Params()
    poses = Poses.model_validate_json((poses_dir / "poses.json").read_text())
    ov = (
        Overrides.model_validate_json(overrides_path.read_text()) if overrides_path.is_file() else Overrides()
    )
    pts, nrm = read_points(fuse_dir / "points.ply")
    vertical = np.abs(nrm[:, 2]) < 0.3
    vpts, vnrm = pts[vertical], nrm[vertical]
    frames = read_jsonl(poses_dir / "frames.jsonl", PosedImage)

    sp = SessionPoints({w.stem: read_wall_points(w) for w in sorted((poses_dir / WALLS_DIR).glob("*.npz"))})
    own: dict[str, tuple[F64, F64]] = {}  # per session, all its windows (`<session>@<s>`)
    for key, (wp, wn) in sp.points.items():
        sid = key.split("@")[0]
        prev = own.get(sid)
        own[sid] = (wp, wn) if prev is None else (np.vstack([prev[0], wp]), np.vstack([prev[1], wn]))

    rooms: dict[str, CapturedRoom] = {}
    s2f: dict[str, F64] = {}
    names: dict[str, str | None] = {}
    faces_by_session: dict[str, list[Face]] = {}
    room_els: list[Room] = []
    flags: list[Flag] = []
    for sess in poses.sessions:
        raw = from_repo(sess.raw_dir)
        room = load_captured_room(raw / b.ROOM_JSON_FILE)
        if room is None:
            continue
        sid = sess.session
        if sess.session_to_scene_parts:
            room, chosen = place_parts(
                room, mat(sess.session_to_scene), [mat(m) for m in sess.session_to_scene_parts], vpts
            )
            if chosen:
                flags.append(
                    Flag(
                        element=None,
                        severity="info",
                        message=f"session {sid} was cut in two by refine: {chosen} of its RoomPlan elements"
                        " placed with its other part's move",
                    )
                )
        rooms[sid], s2f[sid] = room, mat(sess.session_to_scene)
        names[sid] = b.Session.model_validate_json((raw / b.SESSION_FILE).read_text()).room_name
    # Where each session's cameras were, without its doorway dwell at the end: what is its own room.
    votes = Votes(frames, set(rooms))
    in_room = trim_tails(frames, set(rooms))
    for sid, room in rooms.items():
        cams = np.array([mat(f.camera_to_world)[:2, 3] for f in in_room if f.session == sid]).reshape(-1, 2)
        faces = []
        for f in _faces_from_room(room, sid, s2f[sid], cams, own.get(sid), p):
            mid = (f.p0 + f.p1) / 2
            vote = votes.foreign(sid, mid + FACE_PUSH_M * f.interior, (mid, f.interior))
            if vote is None:
                faces.append(f)
            else:
                flags.append(Flag(severity="info", message=f"session {sid}: RoomPlan wall {short_id('wall', f.id)} left"
                    f" out: in {vote.owner}'s room ({vote.own:.0%} of the cameras in front of it, {vote.other:.0%} that"
                    " session's)"))  # fmt: skip
        for f in faces:
            refine_face(f, vpts, vnrm, p)
            snap_face(f, p)
        _join_corners(faces, p)
        faces_by_session[sid] = faces
        found_rooms, session_flags = session_rooms(sid, faces, room, s2f[sid], names[sid], p, cams, votes)
        room_els += found_rooms
        flags += session_flags
    if not rooms:
        raise RuntimeError("no session has a room.json (RoomPlan); nothing to build walls from")

    for sid, room in rooms.items():
        for o in room.surfaces:
            if o.kind in ("door", "window", "opening"):
                m = s2f[sid] @ o.transform
                half = _unit(_xy(m[:3, 0])) * float(o.dimensions[0]) / 2
                sp.holes.append((_xy(m[:3, 3]) - half, _xy(m[:3, 3]) + half))
        for o in room.objects:  # RoomPlan's own boxes, grown a little: they are loose by several cm
            m = s2f[sid] @ o.transform
            sp.furniture.append(
                _footprint(
                    tuple(m[:3, 3]), (float(o.dimensions[0]), float(o.dimensions[2]), 1.0), _yaw(m)
                ).buffer(0.05)
            )
    if sp.furniture:
        merged_feet = cast(Polygon, unary_union(sp.furniture))
        sp.furniture.clear()
        sp.furniture.append(merged_feet)
    merged = _merge_same([f for fs in faces_by_session.values() for f in fs], p)
    walls, face_to_wall, wall_flags = walls_from_faces(merged, p, sp)
    flags += wall_flags
    objects, stairs = _objects(rooms, s2f, pts, nrm, votes)
    wall_lines = [(np.array(w.start), np.array(w.end), w.thickness) for w in walls]
    objects = _refine_objects(objects, wall_lines, pts, nrm)
    stairs = merge_stairs(stairs, wall_lines)
    # Stairs and furniture tops aren't floor: leave their footprints out of the floor heights.
    footprints = [_footprint(e.center, e.size, e.yaw_deg) for e in [*objects, *stairs]]
    room_els, walls = _fit_rooms(room_els, walls, pts, nrm, footprints)
    objects = [_in_room(_on_floor(o, room_els), room_els) for o in objects]
    for f in merged:
        if f.note:
            flags.append(Flag(element=face_to_wall.get(f.id), severity="info", message=f.note))
    session_faces = [f for fs in faces_by_session.values() for f in fs]
    for f in session_faces:
        if f.flipped:
            flags.append(Flag(element=face_to_wall.get(f.id), severity="info", message=f.flipped))
    openings = _openings(rooms, s2f, walls, face_to_wall)
    by_id = {w.id: w for w in walls}
    openings = [
        _check_door(o, by_id[o.wall], pts, nrm) if o.kind == OpeningKind.DOOR else o for o in openings
    ]

    noted = {face_to_wall.get(f.id) for f in merged if f.note}
    for w in walls:
        if w.source.fit is None:
            if w.id not in noted:
                flags.append(
                    Flag(
                        element=w.id, severity="info", message="not refined: too few fused points on the face"
                    )
                )
        elif w.source.fit.shift_m > 0.05 or w.source.fit.rotation_deg > 3:
            flags.append(
                Flag(
                    element=w.id,
                    severity="info",
                    message=f"refinement moved RoomPlan's face {w.source.fit.shift_m * 100:.1f} cm / "
                    f"{w.source.fit.rotation_deg:.1f} deg",
                )
            )
    scene = Scene(
        sessions=sorted(rooms),
        rooms=sorted(room_els, key=lambda r: r.id),
        walls=sorted(walls, key=lambda w: w.id),
        openings=sorted(openings, key=lambda o: o.id),
        objects=sorted(objects, key=lambda o: o.id),
        stairs=sorted(stairs, key=lambda s: s.id),
    )
    scene.flags = flags
    scene = apply_overrides(scene, ov)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_scene(scene, out_path)
    for f in scene.flags:
        if f.severity != "info":
            print(f"  {f.severity}: {f.element or ''} {f.message}")
    return {
        "rooms": len(scene.rooms),
        "walls": len(scene.walls),
        "walls_refined": sum(w.source.fit is not None for w in scene.walls),
        "walls_thickness_measured": sum(w.thickness_measured for w in scene.walls),
        "faces": len(session_faces),
        "faces_unrefined": sum(f.fit is None for f in session_faces),
        "faces_flipped": sum(f.flipped is not None for f in session_faces),
        "openings": len(scene.openings),
        "objects": len(scene.objects),
        "stairs": len(scene.stairs),
        "flags": len(scene.flags),
    }
