"""Stage cache: each stage writes `manifest.json` in its output folder (`data/<project>/derived/<stage>/`, ...)
with a hash of its inputs.

A stage is skipped when its input files, parameters and code are unchanged. Small files are
hashed by content; large ones (videos, 1-3 GB bundles) by path + size + mtime, which is what
makes re-running cheap. Paths are hashed repo-relative (scanmodel.config.rel), so a moved checkout keeps its
caches.
"""

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from .config import CODE_ROOT, rel

CONTENT_HASH_LIMIT = 8 * 1024 * 1024
PACKAGE = Path(__file__).parent
console = Console()


def _files(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.is_file() and not f.name.startswith(".")))
        elif p.is_file():
            out.append(p)
    return out


def fingerprint(paths: Iterable[Path], root: Path | None = None) -> str:
    """Hash of the files (directories: every file in them) by repo-relative path + content (or size + mtime).

    `root`: the repo the paths are relative to (Config.root; default: the repo this code is in)."""
    root = root or CODE_ROOT
    h = hashlib.sha256()
    for f in _files(paths):
        st = f.stat()
        h.update(rel(f, root).encode())
        if st.st_size <= CONTENT_HASH_LIMIT:
            h.update(hashlib.sha256(f.read_bytes()).digest())
        else:
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


@dataclass
class Stage:
    name: str
    inputs: Callable[[], list[Path]]
    run: Callable[[], dict[str, Any]]  # returns a small summary for the manifest
    root: Path  # the repo (Config.root): input paths are hashed relative to it
    params: dict[str, Any] = field(default_factory=dict[str, Any])
    code: list[str] = field(default_factory=list[str])  # module files (relative to the package) it depends on

    def key(self) -> dict[str, Any]:
        # Source files only: a package folder also holds __pycache__/*.pyc, which change on every recompile.
        code = [
            f
            for f in _files(PACKAGE / c for c in [*self.code, "models", "geom.py", "io.py"])
            if f.suffix == ".py"
        ]
        return {
            "inputs": fingerprint(self.inputs(), self.root),
            "params": json.loads(json.dumps(self.params, default=str)),
            "code": fingerprint(code, CODE_ROOT),
        }


def run_stage(stage: Stage, out_dir: Path, force: bool = False) -> bool:
    """Run the stage unless its manifest matches. Returns True if it ran."""
    manifest = out_dir / "manifest.json"
    key = stage.key()
    if not force and manifest.is_file():
        old = json.loads(manifest.read_text())
        if {k: old.get(k) for k in key} == key:
            console.print(f"[dim]{stage.name}: up to date[/dim]")
            return False
    console.print(f"[bold]{stage.name}[/bold]: running")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.unlink(missing_ok=True)  # a failed run must not look up to date
    started = datetime.now(UTC)
    summary = stage.run()
    finished = datetime.now(UTC)
    record = {
        **key,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "summary": summary,
    }
    manifest.write_text(json.dumps(record, indent=2, default=str) + "\n")
    console.print(f"[green]{stage.name}[/green]: done in {(finished - started).total_seconds():.1f}s")
    for k, v in summary.items():
        console.print(f"  {k}: {_brief(v)}", markup=False, highlight=False, soft_wrap=True)
    return True


def _brief(v: Any) -> str:
    """One-line rendering of a summary value: floats to 3 significant figures, dicts as k=v."""
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, dict):
        return " ".join(f"{k}={_brief(x)}" for k, x in v.items())  # type: ignore[union-attr]
    if isinstance(v, list | tuple):
        return ", ".join(_brief(x) for x in v) or "-"  # type: ignore[union-attr]
    return str(v)
