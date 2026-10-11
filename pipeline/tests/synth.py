"""Synthetic captures with known ground truth: two rooms, ray-cast LiDAR depth, RoomPlan-like JSON.

Ground-truth frame ("true"): z up, the front door's threshold centre at (DOOR_X, 0). The bundles are
written in an arbitrary ARKit world (y up, rotated and shifted), exactly as the app would write them,
so the pipeline has to recover the scene frame on its own.

Rooms (inner faces): hall x in [0, 3], y in [0, 4]; living x in [3.12, 7], y in [0, 4]; height 2.4.
The shared wall is 0.12 m thick with a door in it; the front door is in the hall's south wall.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from scanmodel.geom import ARKIT_TO_ZUP, GL_CV
from scanmodel.models import bundle as b
from scanmodel.models.common import F64, to_mat4

HEIGHT = 2.4
SHARED_T = 0.12
DOOR_X = 1.5  # front door centre
HALL = (0.0, 3.0, 0.0, 4.0)
LIVING = (3.0 + SHARED_T, 7.0, 0.0, 4.0)
DEPTH_W, DEPTH_H = 256, 192
VIDEO_K = b.Intrinsics(fx=1450.0, fy=1450.0, cx=960.0, cy=720.0, width=1920, height=1440)

# ARKit world = some rotation about its y axis + an offset (device started 1.3 m above the floor).
_c, _s = np.cos(np.radians(23.0)), np.sin(np.radians(23.0))
ARKIT_WORLD: F64 = np.array([[_c, 0, _s, 0.7], [0, 1, 0, -1.3], [-_s, 0, _c, 2.1], [0, 0, 0, 1.0]])
TRUE_TO_ARKIT: F64 = ARKIT_WORLD @ np.linalg.inv(ARKIT_TO_ZUP)


@dataclass
class Quad:
    """Planar rectangle o + s*a + t*b, s, t in [0, 1], minus holes given in metres along a and b."""

    o: F64
    a: F64
    b: F64
    holes: list[tuple[float, float, float, float]] = field(
        default_factory=list[tuple[float, float, float, float]]
    )


def _box_faces(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> list[Quad]:
    p = np.array
    return [
        Quad(p([x0, y0, z0]), p([x1 - x0, 0, 0]), p([0, 0, z1 - z0])),
        Quad(p([x0, y1, z0]), p([x1 - x0, 0, 0]), p([0, 0, z1 - z0])),
        Quad(p([x0, y0, z0]), p([0, y1 - y0, 0]), p([0, 0, z1 - z0])),
        Quad(p([x1, y0, z0]), p([0, y1 - y0, 0]), p([0, 0, z1 - z0])),
        Quad(p([x0, y0, z1]), p([x1 - x0, 0, 0]), p([0, y1 - y0, 0])),
        Quad(p([x0, y0, z0]), p([x1 - x0, 0, 0]), p([0, y1 - y0, 0])),
    ]


# Openings as (wall x or y, along-range, z-range)
FRONT_DOOR = (DOOR_X - 0.45, DOOR_X + 0.45, 0.0, 2.0)  # hall south wall (y = 0), along x
INNER_DOOR = (1.5, 2.4, 0.0, 2.0)  # shared wall, along y
WINDOW = (4.5, 5.7, 0.9, 2.0)  # living north wall (y = 4), along x
CABINET = (5.5, 6.9, 0.1, 0.7, 0.0, 0.9)


def world_quads() -> list[Quad]:
    q: list[Quad] = []
    for room in (HALL, LIVING):
        x0, x1, y0, y1 = room
        faces = _box_faces(x0, x1, y0, y1, 0.0, HEIGHT)
        south, north, west, east = faces[:4]
        if room is HALL:
            south.holes.append((FRONT_DOOR[0] - x0, FRONT_DOOR[1] - x0, *FRONT_DOOR[2:]))
            east.holes.append(INNER_DOOR)
        else:
            west.holes.append(INNER_DOOR)
            north.holes.append((WINDOW[0] - x0, WINDOW[1] - x0, *WINDOW[2:]))
        q += faces
    q += _box_faces(*CABINET)[:5]  # no bottom face
    return q


def raycast(c2w_cv: F64, quads: list[Quad], k: b.Intrinsics, w: int, h: int) -> F64:
    ks = k.scaled(w, h)
    v, u = np.mgrid[0:h, 0:w]
    d_cam = np.stack([(u + 0.5 - ks.cx) / ks.fx, (v + 0.5 - ks.cy) / ks.fy, np.ones_like(u, dtype=float)], -1)
    d = d_cam.reshape(-1, 3) @ c2w_cv[:3, :3].T  # z_cam = 1, so the ray parameter equals depth
    origin = c2w_cv[:3, 3]
    best = np.full(len(d), np.inf)
    for q in quads:
        n = np.cross(q.a, q.b)
        den = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((q.o - origin) @ n) / den
        p = origin + t[:, None] * d - q.o
        la, lb = np.linalg.norm(q.a), np.linalg.norm(q.b)
        sa, sb = p @ q.a / la, p @ q.b / lb  # metres along a and b
        hit = (t > 0.05) & (sa >= 0) & (sa <= la) & (sb >= 0) & (sb <= lb)
        for a0, a1, b0, b1 in q.holes:
            hit &= ~((sa > a0) & (sa < a1) & (sb > b0) & (sb < b1))
        best = np.where(hit & (t < best), t, best)
    depth = np.where(np.isfinite(best) & (best < 5.0), best, 0.0)
    return depth.reshape(h, w)


def look_at(eye: F64, target: F64) -> F64:
    """OpenCV camera (+z forward, +y down) in the true z-up frame."""
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    dn = np.cross(f, r)
    m = np.eye(4)
    m[:3, :3] = np.stack([r, dn, f], axis=1)
    m[:3, 3] = eye
    return m


def trajectory(room: tuple[float, float, float, float], n: int, start: F64 | None = None) -> list[F64]:
    """Two slow loops at chest height, looking outwards, as in the capture protocol."""
    x0, x1, y0, y1 = room
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2 - 0.8, (y1 - y0) / 2 - 0.8
    out: list[F64] = []
    for i in range(n):
        a = 2 * np.pi * 2 * i / n - np.pi / 2
        eye = np.array([cx + rx * np.cos(a), cy + ry * np.sin(a), 1.0 + 0.3 * np.sin(5 * a)])
        if start is not None and i < n // 10:
            eye = start + (eye - start) * (i / (n // 10))
        look = a + np.radians(60) * np.sin(3 * a)
        target = eye + np.array([np.cos(look), np.sin(look), -0.15 + 0.25 * np.cos(2 * a)])
        out.append(look_at(eye, target))
    return out


def doorway_start() -> list[F64]:
    """Next room: in the hall, look at the shared wall beside the inner door, then walk through it."""
    out: list[F64] = []
    for y in (*np.linspace(0.3, 1.2, 6), *np.linspace(2.7, 3.7, 6)):
        out.append(look_at(np.array([2.0, y, 1.4]), np.array([HALL[1], y, 1.7])))
    door_y = (INNER_DOOR[0] + INNER_DOOR[1]) / 2
    for x in np.linspace(2.2, 3.9, 6):
        out.append(look_at(np.array([x, door_y, 1.3]), np.array([x + 1.0, door_y, 1.3])))
    return out


# --------------------------------------------------------------------------- RoomPlan-like JSON


def _simd(m: F64) -> list[float]:
    return [float(v) for v in m.T.reshape(-1)]  # column by column, like simd_float4x4 Codable


def _surface(
    centre: F64, along: F64, w: float, h: float, kind: str, parent: str | None = None, **cat: object
) -> dict[str, object]:
    ex = along / np.linalg.norm(along)
    ey = np.array([0, 0, 1.0])
    m = np.eye(4)
    m[:3, :3] = np.stack([ex, ey, np.cross(ex, ey)], axis=1)
    m[:3, 3] = centre
    s: dict[str, object] = {
        "identifier": str(uuid.uuid5(uuid.NAMESPACE_OID, f"{kind}{centre.round(3)}")).upper(),
        "category": {kind: dict(cat)},
        "confidence": {"high": {}},
        "dimensions": [w, h, 0.0],
        "transform": _simd(TRUE_TO_ARKIT @ m),
        "completedEdges": [],
        "curve": None,
        "polygonCorners": [],
        "story": 0,
    }
    if parent:
        s["parentIdentifier"] = parent
    return s


def roomplan_json(
    room: tuple[float, float, float, float], rng: np.random.Generator, err: float = 0.03
) -> dict[str, object]:
    """Walls of one room with RoomPlan-sized errors (a few cm, ~1 degree)."""
    x0, x1, y0, y1 = room
    walls: list[dict[str, object]] = []
    sides = {
        "south": (np.array([(x0 + x1) / 2, y0]), np.array([1.0, 0]), x1 - x0),
        "east": (np.array([x1, (y0 + y1) / 2]), np.array([0, 1.0]), y1 - y0),
        "north": (np.array([(x0 + x1) / 2, y1]), np.array([-1.0, 0]), x1 - x0),
        "west": (np.array([x0, (y0 + y1) / 2]), np.array([0, -1.0]), y1 - y0),
    }
    ids: dict[str, str] = {}
    for name, (c, along, length) in sides.items():
        normal = np.array([-along[1], along[0]])
        c = c + normal * rng.uniform(-err, err)
        ang = np.radians(rng.uniform(-1.0, 1.0))
        along = np.array(
            [along[0] * np.cos(ang) - along[1] * np.sin(ang), along[0] * np.sin(ang) + along[1] * np.cos(ang)]
        )
        s = _surface(
            np.array([c[0], c[1], HEIGHT / 2]),
            np.append(along, 0),
            length + rng.uniform(-0.03, 0.03),
            HEIGHT,
            "wall",
        )
        ids[name] = str(s["identifier"])
        walls.append(s)
    doors, windows = [], []
    if room is HALL:
        doors.append(
            _surface(
                np.array([DOOR_X, y0, 1.0]),
                np.array([1.0, 0, 0]),
                0.9,
                2.0,
                "door",
                ids["south"],
                isOpen=True,
            )
        )
        doors.append(
            _surface(
                np.array([x1, 1.95, 1.0]), np.array([0, 1.0, 0]), 0.9, 2.0, "door", ids["east"], isOpen=True
            )
        )
    else:
        doors.append(
            _surface(
                np.array([x0, 1.95, 1.0]), np.array([0, -1.0, 0]), 0.9, 2.0, "door", ids["west"], isOpen=True
            )
        )
        windows.append(
            _surface(np.array([5.1, y1, 1.45]), np.array([-1.0, 0, 0]), 1.2, 1.1, "window", ids["north"])
        )
    objects = []
    if room is LIVING:
        cx0, cx1, cy0, cy1, _, cz1 = CABINET
        o = _surface(
            np.array([(cx0 + cx1) / 2, (cy0 + cy1) / 2, cz1 / 2]),
            np.array([1.0, 0, 0]),
            cx1 - cx0,
            cz1,
            "storage",
        )
        o["dimensions"] = [cx1 - cx0, cz1, cy1 - cy0]
        objects.append(o)
    return {
        "identifier": str(uuid.uuid4()).upper(),
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "openings": [],
        "objects": objects,
        "floors": [],
        "sections": [],
        "version": 2,
    }


# --------------------------------------------------------------------------- bundles


def write_session(
    raw: Path,
    session_id: str,
    room: tuple[float, float, float, float] | list[tuple[float, float, float, float]],
    seed: int,
    world_map_in: str | None,
    n_frames: int = 160,
    start: F64 | None = None,
    started_at: datetime | None = None,
    room_name: str | None = None,
    sections: list[tuple[str, tuple[float, float, float, float]]] | None = None,
    prelude: list[F64] | None = None,
) -> Path:
    """One session; a list of rooms is walked through in one recording (n_frames per room).

    `sections` are RoomPlan section labels with the room they belong to (centre at mid-height).
    `prelude` are camera poses (OpenCV, true frame) recorded before the loops.
    """
    rng = np.random.default_rng(seed)
    rooms = room if isinstance(room, list) else [room]
    d = raw / session_id
    (d / b.DEPTH_DIR).mkdir(parents=True)
    (d / b.CONFIDENCE_DIR).mkdir()
    (d / b.STILLS_DIR).mkdir()
    quads = world_quads()
    frames: list[b.Frame] = []
    poses = [
        *(prelude or []),
        *(m for i, r in enumerate(rooms) for m in trajectory(r, n_frames, start if i == 0 else None)),
    ]
    for i, c2w_cv in enumerate(poses):
        depth = raycast(c2w_cv, quads, VIDEO_K, DEPTH_W, DEPTH_H)
        depth = np.where(depth > 0, depth + rng.normal(0, 0.002 + 0.002 * depth, depth.shape), 0)
        conf = np.where(depth > 0, 2, 0).astype(np.uint8)
        cv2.imwrite(str(d / b.DEPTH_DIR / f"{i:06d}.png"), np.round(depth * 1000).astype(np.uint16))
        cv2.imwrite(str(d / b.CONFIDENCE_DIR / f"{i:06d}.png"), conf)
        frames.append(
            b.Frame(
                idx=i,
                t=100.0 + i / 10,
                camera_to_world=to_mat4(TRUE_TO_ARKIT @ c2w_cv @ GL_CV),
                intrinsics=VIDEO_K,
                tracking=b.TrackingState.NORMAL,
                depth=f"{b.DEPTH_DIR}/{i:06d}.png",
                confidence=f"{b.CONFIDENCE_DIR}/{i:06d}.png",
            )
        )
    stills: list[b.Still] = []
    for j, i in enumerate((10, n_frames // 2)):
        k = VIDEO_K.scaled(64, 48)
        cv2.imwrite(str(d / b.STILLS_DIR / f"{j:06d}.jpg"), np.full((48, 64, 3), 128, np.uint8))
        stills.append(
            b.Still(
                id=j,
                file=f"{b.STILLS_DIR}/{j:06d}.jpg",
                t=frames[i].t,
                source=b.StillSource.TAP,
                intrinsics=k,
                camera_to_world=frames[i].camera_to_world,
                tracking=b.TrackingState.NORMAL,
                nearest_frame=i,
            )
        )
    session = b.Session(
        project_id="synthetic",
        session_id=session_id,
        type=b.SessionType.SCAN,
        device=b.Device(model="synthetic", os_version="26.0", app_version="0"),
        video=b.VideoFormat(width=1920, height=1440, fps=10.0),
        depth=b.DepthFormat(),
        world_map_in=world_map_in,
        world_map_out=session_id,
        started_at=started_at or datetime(2026, 10, 6, 12, 0, tzinfo=UTC),
        room_name=room_name,
    )
    (d / b.SESSION_FILE).write_text(session.model_dump_json(indent=2, exclude_none=True))
    (d / b.FRAMES_FILE).write_text("".join(f.model_dump_json(exclude_none=True) + "\n" for f in frames))
    (d / b.STILLS_FILE).write_text("".join(s.model_dump_json(exclude_none=True) + "\n" for s in stills))
    captured = roomplan_json(rooms[0], rng)
    for r in rooms[1:]:
        more = roomplan_json(r, rng)
        for key in ("walls", "doors", "windows", "openings", "objects"):
            cast(list[object], captured[key]).extend(cast(list[object], more[key]))
    captured["sections"] = [
        {
            "label": label,
            "center": [float(v) for v in (TRUE_TO_ARKIT @ [(x0 + x1) / 2, (y0 + y1) / 2, 1.2, 1])[:3]],
            "story": 0,
        }
        for label, (x0, x1, y0, y1) in sections or []
    ]
    (d / b.ROOM_JSON_FILE).write_text(json.dumps(captured))
    (d / b.MESH_FILE).write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    return d


def write_two_rooms(raw: Path) -> list[Path]:
    """Hall (first, starts by the front door) and living room (loads the hall's world map)."""
    t0 = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
    hall = write_session(
        raw, "s01-hall", HALL, 1, None, start=np.array([DOOR_X, 0.5, 1.2]), started_at=t0, room_name="hall"
    )
    living = write_session(
        raw,
        "s02-living",
        LIVING,
        2,
        "s01-hall",
        started_at=t0 + timedelta(minutes=12),
        room_name="living",
        prelude=doorway_start(),
    )
    return [hall, living]


def write_one_session(raw: Path) -> Path:
    """Both rooms walked through in one recording, with RoomPlan sections naming them."""
    return write_session(
        raw,
        "s01-home",
        [HALL, LIVING],
        3,
        None,
        n_frames=120,
        start=np.array([DOOR_X, 0.5, 1.2]),
        room_name="home",
        sections=[("bedroom", HALL), ("livingRoom", LIVING)],
    )


def true_to_scene(p: F64) -> F64:
    """Expected scene-frame coordinates of a true-frame point (origin at the front door)."""
    return np.asarray(p, dtype=float) - np.array([DOOR_X, 0.0, 0.0])[: len(p)]
