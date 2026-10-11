"""Generate JSON Schemas from the Pydantic models (`scanmodel schema`)."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models.bundle import AnchorEvent, Calibration, Frame, Session, Still
from .models.common import Strict
from .models.overrides import Overrides
from .models.scene import Scene


class BundleV1(Strict):
    """Capture bundle v1. Each property is one file (the .jsonl files: the schema of one line)."""

    session_json: Session
    frames_jsonl_line: Frame
    stills_jsonl_line: Still
    anchors_jsonl_line: AnchorEvent
    calibration_json: Calibration


def _schema(model: type[BaseModel], title: str) -> dict[str, Any]:
    s = model.model_json_schema(mode="serialization")
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **s, "title": title}


def write_schemas(repo: Path) -> list[Path]:
    out = {
        repo / "pipeline/scanmodel/schemas/bundle.v1.json": _schema(BundleV1, "flat-bundle v1"),
        repo / "scene/schema.json": _schema(Scene, "scan-model v1"),
        repo / "scene/overrides.schema.json": _schema(Overrides, "scan-model overrides v1"),
    }
    for path, schema in out.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(schema, indent=2) + "\n")
    return list(out)
