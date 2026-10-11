"""Paths, settings and tool locations, never hard-coded.

Settings, highest first: command-line options, the repo's `settings.toml`, the defaults here.
`settings.toml` holds `project`, `pose_graph`, `poses` and an optional `[tools]` table (blender, brush,
msplat). The repo root is the one above the working directory (else the one this code is in); tests and
callers that work on another root pass it explicitly.

Everything the pipeline writes refers to files by repo-relative posix paths (`rel`); files outside the
repo stay absolute. Readers take both (`from_repo`).
"""

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SETTINGS_FILE = "settings.toml"
POSE_SOURCES = ("refine", "posegraph")  # settings.toml `poses` / --poses: what stages after `poses` read
TOOLS = {
    "blender": "/Applications/Blender.app/Contents/MacOS/Blender",
    "brush": "/Applications/brush_app",  # the release binary; Homebrew's `brush` is a shell
    "msplat": "msplat-train",  # uv tool install --python 3.13 'msplat[cli]'
}
SETTINGS = {"project": str, "pose_graph": bool, "poses": str}
CODE_ROOT = Path(__file__).resolve().parents[2]  # the repo this code is in


class ConfigError(ValueError):
    """Settings that can't be used: no project, an unknown key, a bad value."""


def find_root(start: Path) -> Path:
    """The repo at or above `start`: the folder with pipeline/pyproject.toml and the tracked scene/ schemas."""
    for p in [start, *start.parents]:
        if (p / "pipeline" / "pyproject.toml").is_file() and (p / "scene").is_dir():
            return p
    raise FileNotFoundError(f"no repo (pipeline/ + scene/) at or above {start}")


def default_root() -> Path:
    """The repo above the working directory, else the one this code is in."""
    try:
        return find_root(Path.cwd())
    except FileNotFoundError:
        return CODE_ROOT


def rel(path: Path | str, root: Path | None = None) -> str:
    """`path` as stored in written files: posix and relative to the repo root when inside it, else absolute.

    Relative input is returned as is (already repo-relative)."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    root = root or default_root()
    for base, target in (
        (Path(os.path.abspath(root)), Path(os.path.abspath(p))),
        (root.resolve(), p.resolve()),
    ):
        try:
            return target.relative_to(base).as_posix() or "."
        except ValueError:
            continue
    return p.as_posix()


def from_repo(value: Path | str, root: Path | None = None) -> Path:
    """Inverse of `rel`: an absolute path (old files, or outside the repo) as is, a relative one under root."""
    p = Path(value)
    return p if p.is_absolute() else (root or default_root()) / p


def read_settings(path: Path) -> dict[str, Any]:
    """settings.toml, checked: known keys of the right type, `[tools]` paths as strings. {} if missing."""
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text())
    tools = data.pop("tools", {})
    if unknown := set(data) - set(SETTINGS):
        raise ConfigError(
            f"{path}: unknown setting(s) {', '.join(sorted(unknown))} (known: {', '.join(SETTINGS)})"
        )
    if not isinstance(tools, dict) or (unknown := set(tools) - set(TOOLS)):  # pyright: ignore[reportUnknownArgumentType]
        raise ConfigError(f"{path}: [tools] takes {', '.join(TOOLS)}")
    for key, value in [*data.items(), *tools.items()]:  # pyright: ignore[reportUnknownMemberType]
        kind = SETTINGS.get(key, str)
        if not isinstance(value, kind):
            raise ConfigError(f"{path}: {key} must be a {kind.__name__}")
    return {**data, **({"tools": tools} if tools else {})}


@dataclass(frozen=True)
class Config:
    root: Path
    project: str
    project_source: str  # --project or settings.toml
    pose_graph: bool  # poses corrects ARKit's drift with a keyframe pose graph (scanmodel.align)
    poses: str  # refine (`scanmodel refine`'s bundle-adjusted poses, the default) or posegraph (`scanmodel poses` alone)
    blender: Path
    brush: str
    msplat: str  # msplat-train (uv tool install --python 3.13 'msplat[cli]')

    @property
    def data(self) -> Path:
        return self.root / "data" / self.project

    @property
    def raw(self) -> Path:  # what ingest reads
        return self.data / "raw"

    @property
    def derived(self) -> Path:
        return self.data / "derived"

    @property
    def scene_dir(self) -> Path:  # overrides, views and the generated model
        return self.data / "scene"

    def stage_dir(self, stage: str) -> Path:
        return self.data / stage if stage in ("splat", "build", "export") else self.derived / stage

    @property
    def views_file(self) -> Path:
        return self.scene_dir / "views.json"

    def dataset_dir(self, name: str | None = None) -> Path:
        return self.stage_dir("splat") / "datasets" / (name or "default")

    @property
    def runs_dir(self) -> Path:
        return self.stage_dir("splat") / "runs"

    def rel(self, path: Path | str) -> str:
        return rel(path, self.root)

    def from_repo(self, value: Path | str) -> Path:
        return from_repo(value, self.root)


def load_config(root: Path | None = None, project: str | None = None) -> Config:
    """The settings for `project` (default: settings.toml's) in `root` (default: `default_root()`)."""
    root = root or default_root()
    s = read_settings(root / SETTINGS_FILE)
    project = (project or "").strip() or None  # an empty --project falls through to settings.toml
    source = "--project" if project else SETTINGS_FILE
    project = project or str(s.get("project", "")).strip() or None
    if project is None:
        raise ConfigError(f"no project: pass --project NAME or set `project` in {SETTINGS_FILE}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", project):
        raise ConfigError("project must be a plain folder name")
    poses = str(s.get("poses", POSE_SOURCES[0])).strip().lower()
    tools = {**TOOLS, **s.get("tools", {})}
    blender = Path(os.path.expanduser(tools["blender"]))
    return Config(
        root=root,
        project=project,
        project_source=source,
        pose_graph=bool(s.get("pose_graph", True)),
        poses=poses,
        blender=blender if blender.is_absolute() else (root / blender).resolve(),
        brush=tools["brush"],
        msplat=tools["msplat"],
    )
