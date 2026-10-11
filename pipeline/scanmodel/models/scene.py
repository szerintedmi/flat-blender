"""`scene/model.json`: the structure layer, the single source of truth for the model.

Scene frame: metres, +z up (against gravity), main floor at z = 0. The origin is the selected
RoomPlan door's threshold (by default nearest the first camera), or the first camera's XY without a
door. The x/y axes follow the walls, not geographic east/north.
Yaw is snapped so the dominant wall directions run along x and y. Every z in this file is absolute
(scene frame), including on upper levels; `Level.elevation` is informational.

Element ids are stable across re-runs (derived from RoomPlan identifiers), so `overrides.json`
keeps applying to the same element after a re-scan of other rooms.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import Field

from .common import F64, Mat4, Strict, Vec2, Vec3

FORMAT = "scan-model"
VERSION = 1


class SourceKind(StrEnum):
    ROOMPLAN = "roomplan"  # taken as-is from RoomPlan
    PLANE_FIT = "plane_fit"  # RoomPlan, refined against the fused point cloud
    MERGED = "merged"  # several segments (e.g. both faces, or two sessions) combined
    OVERRIDE = "override"  # added or changed by overrides.json
    POC = "poc"  # estimate from the proof of concept (only used for checks)


class Fit(Strict):
    inliers: int
    rms_m: float
    shift_m: float = 0.0  # how far refinement moved the RoomPlan face
    rotation_deg: float = 0.0  # how far refinement rotated it


class Source(Strict):
    kind: SourceKind
    sessions: list[str] = []
    roomplan_ids: list[str] = []
    fit: Fit | None = None


class Level(Strict):
    id: str
    name: str
    elevation: float = 0.0


class Room(Strict):
    id: str
    name: str | None = None
    level: str = "ground"
    session: str | None = None
    polygon: list[Vec2] = Field(min_length=3)  # floor outline, counter-clockwise, inner faces
    floor_z: float = 0.0
    ceiling_z: float  # at the outline's centre if the ceiling slopes
    ceiling_plane: tuple[float, float, float] | None = None  # sloped ceiling: z = a + b*x + c*y
    source: Source


class Wall(Strict):
    """A straight wall as a centreline with a thickness. Faces are offset +-thickness/2 to each side.

    Walls meeting at a corner end at the intersection of their centrelines; a builder closes the
    outside corner by extending one of them by half the other's thickness.
    """

    id: str
    level: str = "ground"
    start: Vec2
    end: Vec2
    thickness: float = Field(gt=0)
    thickness_measured: bool = False  # false: a default (one face seen, or two never in one window)
    # (left, right): did a session see that face? An unseen one is placed a thickness behind the other.
    faces_observed: tuple[bool, bool] = (True, True)
    base_z: float = 0.0
    height: float = Field(gt=0)  # at the start
    height_end: float | None = None  # at the end, if the top slopes (under a sloped ceiling)
    rooms: list[str] = []
    source: Source

    def direction(self) -> F64:
        d = np.subtract(self.end, self.start)
        return d / np.linalg.norm(d)

    def normal(self) -> F64:
        """Left-hand normal of start -> end."""
        dx, dy = self.direction()
        return np.array([-dy, dx])

    def length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.end, self.start)))


class OpeningKind(StrEnum):
    DOOR = "door"
    WINDOW = "window"
    OPENING = "opening"  # doorway without a door, arch, pass-through


class Opening(Strict):
    """A hole in a wall. Positions are along the wall's centreline from its start."""

    id: str
    kind: OpeningKind
    wall: str
    offset: float  # wall start -> opening centre, metres along the wall
    width: float = Field(gt=0)
    bottom: float = Field(ge=0)  # sill height above the wall base (0 for doors)
    height: float = Field(gt=0)
    is_open: bool | None = None  # RoomPlan door state when scanned
    source: Source


class Representation(StrEnum):
    PARAMETRIC = "parametric"  # generated geometry (cabinets, doors, tables)
    SPLAT = "splat"  # Gaussian splat cut-out (clutter)
    TSDF_CROP = "tsdf_crop"  # crop of the fused mesh (odd shapes)
    BOX = "box"  # placeholder until one of the above exists


class Object(Strict):
    """Furniture or fitting as an oriented box: `size` is along its local x/y/z, then rotated by `yaw_deg`."""

    id: str
    category: str  # RoomPlan category: storage, refrigerator, stove, bed, sink, table, sofa, ...
    center: Vec3
    size: Vec3
    yaw_deg: float = 0.0
    representation: Representation = Representation.BOX
    room: str | None = None
    source: Source


class Stair(Strict):
    id: str
    center: Vec3
    size: Vec3
    yaw_deg: float = 0.0  # local +y is the climbing direction
    tread_z: list[float] = []  # absolute heights of fitted treads, bottom to top
    source: Source


class View(Strict):
    """A named camera, shared by Blender renders and the web viewer."""

    id: str
    name: str | None = None
    camera_to_world: Mat4 | None = None  # Blender/OpenGL camera axes (looks down -z)
    position: Vec3 | None = None  # alternatively position + look_at
    look_at: Vec3 | None = None
    fov_deg: float = 70.0  # across the longer image side
    still: str | None = None  # PosedImage id "<session>/s<still id>" if made from a still


class CameraPath(Strict):
    id: str
    name: str | None = None
    points: list[Vec2] = Field(min_length=2)  # floor-plan waypoints
    eye_height: float = 1.6
    speed: float = 1.0  # m/s


class Flag(Strict):
    """Something a human should look at (e.g. a dimension far from the proof-of-concept estimate)."""

    element: str | None = None
    severity: Literal["info", "warn", "error"] = "warn"
    message: str


class Scene(Strict):
    format: Literal["scan-model"] = FORMAT
    version: Literal[1] = VERSION
    generated_at: datetime | None = None
    sessions: list[str] = []
    levels: list[Level] = [Level(id="ground", name="Ground floor")]
    rooms: list[Room] = []
    walls: list[Wall] = []
    openings: list[Opening] = []
    objects: list[Object] = []
    stairs: list[Stair] = []
    views: list[View] = []
    paths: list[CameraPath] = []
    flags: list[Flag] = []

    def wall(self, wall_id: str) -> Wall:
        return next(w for w in self.walls if w.id == wall_id)
