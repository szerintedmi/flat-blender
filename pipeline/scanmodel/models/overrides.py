"""`scene/overrides.json`: manual fixes, applied last by `scanmodel structure`.

Generated values never overwrite these: the structure stage rebuilds `model.json` from the scans
and then re-applies this file. Patches are JSON merge patches (RFC 7396) keyed by element id.

    {
      "version": 1,
      "frame": {"origin_door": "door-1a2b3c4d"},
      "patch": {"wall-9f8e7d6c": {"thickness": 0.25}},
      "remove": ["object-0badf00d"],
      "add": {"objects": [{"id": "object-cat-tree", ...}]},
      "views": [{"id": "kitchen", "position": [0.2, 1.0, 1.6], "look_at": [0.5, 3.5, 1.2]}],
      "drop_frames": {"20261009-125646-living": [[9463, 9530]]}
    }

`drop_frames` is read by `scanmodel ingest` (not structure): per session, inclusive frame idx ranges left out of
frames.jsonl, with the stills shot meanwhile, as ingest does by itself for an excursion
(scanmodel.checks.excursions) ARKit's tracking state didn't flag.
"""

from typing import Any, Literal

from .common import Strict
from .scene import CameraPath, Level, Object, Opening, Room, Stair, View, Wall


class FrameChoice(Strict):
    """How `scanmodel poses` fixes the scene frame. Defaults: see scanmodel.stages.poses."""

    origin_door: str | None = None  # RoomPlan door identifier (or "door-<first 8 hex>") of the front door
    yaw_offset_deg: float = 0.0  # extra rotation after the snap, multiples of 90 to fix north
    floor_z: float | None = None  # override the detected floor height (pipeline axes, before origin shift)


class Additions(Strict):
    levels: list[Level] = []
    rooms: list[Room] = []
    walls: list[Wall] = []
    openings: list[Opening] = []
    objects: list[Object] = []
    stairs: list[Stair] = []


class Overrides(Strict):
    version: Literal[1] = 1
    frame: FrameChoice = FrameChoice()
    room_names: dict[str, str] = {}  # room id -> name
    patch: dict[str, dict[str, Any]] = {}
    remove: list[str] = []
    add: Additions = Additions()
    views: list[View] = []
    paths: list[CameraPath] = []
    drop_frames: dict[str, list[tuple[int, int]]] = {}  # session -> [[first idx, last idx], ...]
