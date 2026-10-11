"""Read RoomPlan's `CapturedRoom` JSON (Swift Codable output) into plain surfaces and objects.

Parsing is deliberately tolerant: Swift encodes enums as `{"wall": {}}`, simd matrices as column
arrays (flat or nested), and fields vary between iOS releases. Phase 0 checks this against a real
capture; anything unexpected raises with the key path.

RoomPlan surface local axes: origin at the centre, +x along the width, +y up, +z the face normal.
`dimensions` = (width, height, depth); walls and openings have depth 0.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from .models.common import F64

SURFACE_KEYS = {
    "walls": "wall",
    "doors": "door",
    "windows": "window",
    "openings": "opening",
    "floors": "floor",
}


@dataclass
class Surface:
    id: str
    kind: str  # wall | door | window | opening | floor
    transform: F64  # 4x4 in ARKit world (y up)
    dimensions: F64  # (width, height, depth)
    parent: str | None = None
    is_open: bool | None = None
    polygon: F64 | None = None  # (N, 3) corners in the local frame, if RoomPlan gave them
    confidence: str | None = None


@dataclass
class RoomObject:
    id: str
    category: str
    transform: F64
    dimensions: F64
    parent: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class Section:
    """RoomPlan's guess at one room inside a capture: a label (livingRoom, kitchen, ...) and a centre."""

    label: str
    center: F64  # (3,) in ARKit world (y up)


@dataclass
class CapturedRoom:
    surfaces: list[Surface]
    objects: list[RoomObject]
    sections: list[Section] = field(default_factory=list[Section])

    def of(self, kind: str) -> list[Surface]:
        return [s for s in self.surfaces if s.kind == kind]


def _enum(v: Any) -> tuple[str, dict[str, Any]]:
    """Swift enum with associated values: {"door": {"isOpen": true}} -> ("door", {...}); "high" -> ("high", {})."""
    if isinstance(v, str):
        return v, {}
    if isinstance(v, dict) and len(v) == 1:
        k, payload = next(iter(cast(dict[str, Any], v).items()))
        inner = cast(dict[str, Any], payload) if isinstance(payload, dict) else {}
        # Swift's synthesized Codable wraps unlabeled associated values as {"_0": ...}
        if "_0" in inner and isinstance(inner["_0"], dict):
            inner = cast(dict[str, Any], inner["_0"])
        return k, inner
    raise ValueError(f"not an enum value: {v!r}")


def matrix_from_simd(v: Any) -> F64:
    """simd_float4x4 Codable: 16 floats column by column, flat or as 4 columns of 4."""
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size != 16:
        raise ValueError(f"expected 16 floats for a 4x4 transform, got {a.size}")
    return a.reshape(4, 4).T  # columns -> rows


def _uuid(v: Any) -> str:
    return str(v).lower()


def parse_captured_room(data: dict[str, Any]) -> CapturedRoom:
    surfaces: list[Surface] = []
    for key, kind in SURFACE_KEYS.items():
        for i, s in enumerate(cast(list[dict[str, Any]], data.get(key, []))):
            try:
                _, cat = _enum(s["category"]) if "category" in s else (kind, {})
                poly = s.get("polygonCorners")
                surfaces.append(
                    Surface(
                        id=_uuid(s["identifier"]),
                        kind=kind,
                        transform=matrix_from_simd(s["transform"]),
                        dimensions=np.asarray(s["dimensions"], dtype=np.float64).reshape(3),
                        parent=_uuid(s["parentIdentifier"]) if s.get("parentIdentifier") else None,
                        is_open=cat.get("isOpen"),
                        polygon=np.asarray(poly, dtype=np.float64).reshape(-1, 3) if poly else None,
                        confidence=_enum(s["confidence"])[0] if "confidence" in s else None,
                    )
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"room.json {key}[{i}]: {e}") from e
    objects: list[RoomObject] = []
    for i, o in enumerate(cast(list[dict[str, Any]], data.get("objects", []))):
        try:
            category, _ = _enum(o["category"])
            objects.append(
                RoomObject(
                    id=_uuid(o["identifier"]),
                    category=category,
                    transform=matrix_from_simd(o["transform"]),
                    dimensions=np.asarray(o["dimensions"], dtype=np.float64).reshape(3),
                    parent=_uuid(o["parentIdentifier"]) if o.get("parentIdentifier") else None,
                    attributes=cast(dict[str, Any], o.get("attributes") or {}),
                )
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"room.json objects[{i}]: {e}") from e
    sections: list[Section] = []
    for i, sec in enumerate(cast(list[dict[str, Any]], data.get("sections") or [])):
        try:
            sections.append(
                Section(
                    label=_enum(sec["label"])[0],
                    center=np.asarray(sec["center"], dtype=np.float64).reshape(3),
                )
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"room.json sections[{i}]: {e}") from e
    return CapturedRoom(surfaces=surfaces, objects=objects, sections=sections)


def load_captured_room(path: Path) -> CapturedRoom | None:
    return parse_captured_room(json.loads(path.read_text())) if path.is_file() else None


def short_id(prefix: str, uuid: str) -> str:
    return f"{prefix}-{uuid.replace('-', '')[:8]}"
