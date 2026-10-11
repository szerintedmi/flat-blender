"""The Mac side of the cloud runs: datasets, runs, progress, cost and downloads (`gs-trainer cloud ...`).

Everything a run needs lives in two Modal volumes. `flat-gs-data` holds each dataset once under its
content hash (`datasets/<hash>/`), so a re-run of an unchanged dataset uploads nothing. `flat-gs-runs`
holds one folder per run (`<run id>/`) with the trainer's outputs and a `run.json` that the remote
function keeps up to date (status, attempts, provenance). The local commands only read that state; the
remote side is in `modal_app.py`.

Each run is its own detached Modal app named `flat-gs <run id>`. That makes `modal app list` show which
runs are alive, `modal app stop` stop exactly one run, and a closed laptop not kill a run. The other apps
are `flat-gs prepare|depth-prior|pack-prior|eval <hash>`, `flat-gs-build` (`cloud build`) and `flat-gs-gpu-tests`
(`cloud gpu-test`); `ps` lists every live app whose name starts with `flat-gs` and the month's actual
bill from `modal billing report`, grouped by these names. The pure logic
(hashing, ids, parsing, status, cost, manifest) is at the top and unit-tested offline; the functions that
talk to Modal come after it and are only exercised by real runs.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, tzinfo
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from gs_trainer.datapack import BULK_TAR, PRIOR_TAR, TarStats, pack

if TYPE_CHECKING:
    import modal

# The Modal names keep the trainer's old name `flat-gs` on purpose: the volumes hold the uploaded datasets
# and every run, the app names group the month's bill (`billing_group`), and the hash salt below keys every
# uploaded dataset. Renaming any of them would orphan that data or split the bill.
APP_NAME = "flat-gs"
BUILD_APP = f"{APP_NAME}-build"  # `cloud build`: an app only to build the image in
TEST_APP = f"{APP_NAME}-gpu-tests"  # `cloud gpu-test`
DATA_VOLUME = "flat-gs-data"
RUNS_VOLUME = "flat-gs-runs"
DATA_MOUNT = "/data"
RUNS_MOUNT = "/runs"
DEFAULT_GPU = "L40S"
# Tried in order when no type is asked for: L40S capacity on Modal ran out for over half an hour (2026-10-08).
# Only GPUs our kernels are compiled for (TORCH_CUDA_ARCH_LIST 8.9;9.0): L40S/L4 (8.9), H100 (9.0); no A100.
GPU_FALLBACK = ["L40S", "L4", "H100"]
DEFAULT_STEPS = 30_000  # gs_trainer.train's default, for the progress bar when a run doesn't say
DA3_MODEL = "depth-anything/DA3-BASE"  # gs_trainer.da3.DEFAULT_MODEL (Apache-2.0; baked into the image)
# gs_trainer.fusion.MODES, copied so this module imports without torch (a test keeps them equal). Each is a
# `--depth-target` of gs_trainer.train besides the raw `lidar`.
FUSION_MODES = ("fused", "lidar_only", "da3_holes_only")

# Modal's prices per hour (docs/plans/2026-10-08-gsplat-cloud.md, "Services"), plus a flat estimate for
# the CPU cores and RAM a GPU function also bills. Good enough for "what has this cost so far".
GPU_PRICES = {"L40S": 1.95, "L4": 0.80, "A100-80GB": 2.50, "H100": 3.95, "A10": 1.10}
GPU_ALIASES = {"A10G": "A10", "H100!": "H100", "A100-80G": "A100-80GB"}
CPU_RAM_PER_HOUR = 0.45

# What identifies a dataset: these files by content, the image/depth/confidence folders by name and size
# (1,600 JPEGs by content would take longer than the upload check is worth).
HASHED_FILES = ("transforms.json", "split.json", "points.ply", "mesh.ply", "views.npz")
LISTED_DIRS = ("images", "depth", "confidence")
# Re-solved test poses (`scanmodel splat`): eval-only, so neither hashed nor pushed with the dataset; `cloud eval`
# uploads them beside the PLY (`eval_test_poses`).
TEST_POSES = "test_poses.json"
# Written last (after the files and `bulk.tar`), so a half-finished upload is never taken for a dataset.
# JSON; its `bulk_tar` entry says the tar is there (a dataset pushed before the tars lacks it).
COMPLETE = ".complete"
BULK_KEY = "bulk_tar"

TRAINER_DIR = Path(__file__).resolve().parents[1]
REPO = TRAINER_DIR.parent
STEP = re.compile(r"step=\s*(\d+)")  # same as pipeline/scanmodel/progress.py

console = Console()


# --- pure logic ---------------------------------------------------------------------------------------


def dataset_files(data: Path) -> list[tuple[Path, str]]:
    """The files of a dataset that go to the cloud, as (local path, path relative to the dataset).

    `prior/` is left out on purpose: the priors are made on the cloud (`prepare`, `depth_prior`), from
    the mesh, the LiDAR and the images.
    """
    files = [(data / name, name) for name in HASHED_FILES if (data / name).is_file()]
    for sub in LISTED_DIRS:
        d = data / sub
        if d.is_dir():
            files += [
                (f, f.relative_to(data).as_posix())
                for f in sorted(d.rglob("*"))
                if f.is_file() and not any(p.startswith(".") for p in f.relative_to(data).parts)
            ]
    return files


def dataset_hash(data: Path) -> str:
    """Content hash of a dataset: the small files' bytes and the sorted (path, size) of the folders.

    The same dataset gives the same hash on any machine, whatever the file times, so a dataset is
    uploaded once and every run on it records which data it saw.
    """
    if not (data / "transforms.json").is_file():
        raise FileNotFoundError(
            f"{data / 'transforms.json'} missing: is {data} a splat dataset (`scanmodel splat`)?"
        )
    h = hashlib.sha256(b"flat-gs dataset v2\n")
    for name in HASHED_FILES:
        f = data / name
        digest = _sha256_file(f) if f.is_file() else "missing"
        h.update(f"{name} {digest}\n".encode())
    for path, rel in dataset_files(data):
        if rel.split("/", 1)[0] in LISTED_DIRS:
            h.update(f"{rel} {path.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def upload_plan(marker: Mapping[str, Any] | None) -> str:
    """What `push` uploads, from the dataset's `.complete` marker on the volume: `all` (not there yet:
    the files and `bulk.tar`), `bulk` (pushed before the tars existed: only `bulk.tar`) or `none`."""
    if marker is None:
        return "all"
    return "none" if marker.get(BULK_KEY) else "bulk"


def complete_marker(
    old: Mapping[str, Any] | None, h: str, files: Sequence[tuple[Path, str]], data: Path, bulk: TarStats,
    now: datetime,
) -> dict[str, Any]:  # fmt: skip
    """The `.complete` marker after an upload: the files' count and size (kept from `old` when only the
    tar went up) and the tar's."""
    marker: dict[str, Any] = dict(old or {})
    if files or old is None:
        size = sum(p.stat().st_size for p, _ in files)
        marker.update(hash=h, files=len(files), bytes=size, at=now.isoformat())
    marker["from"] = store_path(data)
    marker[BULK_KEY] = {"files": bulk.files, "bytes": bulk.bytes, "at": now.isoformat()}
    return marker


def repo_rel(path: Path | str, root: Path) -> str:
    """`scanmodel.config.rel`, copied (the trainer doesn't import the pipeline): an absolute `path` posix and
    relative to `root` when inside it, else as it is; a relative one as it is."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    for base, target in (
        (Path(os.path.abspath(root)), Path(os.path.abspath(p))),
        (root.resolve(), p.resolve()),
    ):
        try:
            return target.relative_to(base).as_posix() or "."
        except ValueError:
            continue
    return p.as_posix()


def pipeline_fingerprint(paths: Iterable[Path], root: Path | None = None) -> str:
    """`scanmodel.cache.fingerprint`, copied so the run's manifest says whether it saw the current dataset.

    The viewer marks a splat "older dataset" when this differs from the pipeline's own fingerprint of
    `transforms.json` + `points.ply`. It hashes each path relative to `root` (default `REPO`) too, as
    the pipeline does (`data/<project>/splat/datasets/default/...`).
    """
    root = root if root is not None else REPO
    limit = 8 * 1024 * 1024  # scanmodel.cache.CONTENT_HASH_LIMIT
    h = hashlib.sha256()
    for f in paths:
        if not f.is_file():
            continue
        st = f.stat()
        h.update(repo_rel(f, root).encode())
        if st.st_size <= limit:
            h.update(hashlib.sha256(f.read_bytes()).digest())
        else:
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def slug(name: str) -> str:
    """A run name safe for paths and app names: lower case, letters, digits and dashes."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "run"


def make_run_id(name: str, now: datetime) -> str:
    """`gs-<name>-<YYYYmmdd-HHMMSS>`: sorts by time, and says what the run was for."""
    return f"gs-{slug(name)}-{now:%Y%m%d-%H%M%S}"


def app_description(run_id: str) -> str:
    """The Modal app name of a run, which is how `ps` and `stop` find it in `modal app list`."""
    return f"{APP_NAME} {run_id}"


def run_of_app(description: str) -> str | None:
    """The run id in an app's description, or None for our other apps (prepare, eval)."""
    parts = description.split()
    if len(parts) == 2 and parts[0] == APP_NAME and parts[1].startswith("gs-"):
        return parts[1]
    return None


def step_of(line: str) -> int | None:
    """The step in a trainer line (`step=  1200 loss=...`), if it is one."""
    m = STEP.search(line)
    return int(m[1]) if m else None


def train_args(steps: int | None, max_side: int | None, extra: Sequence[str]) -> list[str]:
    """`gs_trainer.train` options from the cloud command's own shortcuts plus the pass-through ones."""
    args: list[str] = []
    if steps is not None:
        args += ["--steps", str(steps)]
    if max_side is not None:
        args += ["--max-side", str(max_side)]
    return [*args, *extra]


def remote_train_argv(dataset_hash: str, run_id: str, args: Sequence[str]) -> list[str]:
    """The full `gs_trainer.train` command line on the cloud. `--resume auto` makes a retried attempt (after
    a preemption) continue from the last checkpoint instead of starting over."""
    return [
        "--data", f"{DATA_MOUNT}/datasets/{dataset_hash}",
        "--out", f"{RUNS_MOUNT}/{run_id}",
        "--resume", "auto",
        *args,
    ]  # fmt: skip


def eval_views_file(argv: Sequence[str]) -> tuple[list[str], Path | None]:
    """Remove the explicit local named-view file for upload. Views are scene-specific and opt-in."""
    out: list[str] = []
    views = None
    it = iter(argv)
    for arg in it:
        if arg == "--views":
            value = next(it, None)
            if value is None:
                raise ValueError("--views needs a file")
            views = Path(value)
        elif arg.startswith("--views="):
            views = Path(arg.split("=", 1)[1])
        else:
            out.append(arg)
    return out, None if "--no-views" in out else views


def eval_test_poses(argv: Sequence[str], default: Path | None) -> tuple[list[str], Path | None]:
    """(`gs_trainer.evaluate` options without a local `--test-poses FILE`, the local test poses file to upload
    for the eval: that FILE, else `default` (the dataset's `test_poses.json`) if it exists; None with
    `--no-test-poses`). The file isn't part of the dataset on the volume (nor of its hash: training never
    reads it), so a newer solve re-scores old runs without re-uploading the dataset."""
    out: list[str] = []
    file = default if default is not None and default.is_file() else None
    it = iter(argv)
    for a in it:
        if a == "--test-poses":
            file = Path(next(it, ""))
        elif a.startswith("--test-poses="):
            file = Path(a.split("=", 1)[1])
        else:
            out.append(a)
    if "--no-test-poses" in out:
        return out, None
    return out, file


def depth_target_of(args: Sequence[str]) -> str:
    """The `--depth-target` among `gs_trainer.train` options (the last one wins, as in argparse); `lidar`
    when there is none."""
    target = "lidar"
    for i, a in enumerate(args):
        if a == "--depth-target" and i + 1 < len(args):
            target = args[i + 1]
        elif a.startswith("--depth-target="):
            target = a.split("=", 1)[1]
    return target


def fused_modes_of(report: Mapping[str, Any] | None) -> list[str]:
    """The fusion modes a dataset's `prior/fusion.json` says were written."""
    return [m for m in FUSION_MODES if report and m in report]


def needs_depth_prior(
    report: Mapping[str, Any] | None, da3_meta: Mapping[str, Any] | None, model: str
) -> bool:
    """Whether `depth_prior` has to run: a fusion mode is missing, or DA3 ran with another model."""
    if len(fused_modes_of(report)) < len(FUSION_MODES):
        return True
    return (da3_meta or {}).get("model") != model


def module_cmd(module: str, argv: Sequence[str], python: str = sys.executable) -> list[str]:
    """Run `module.main(argv)` in a fresh interpreter, whether or not the module has a `__main__` guard.

    A subprocess keeps a CUDA crash or a hard exit in the trainer from taking the status bookkeeping
    down with it."""
    code = f"import sys; from {module} import main; sys.exit(main(sys.argv[1:]))"
    return [python, "-c", code, *argv]


def args_from_mapping(options: Mapping[str, Any]) -> list[str]:
    """`{"depth_lambda": 0.5, "pose_opt": True, "prune": False}` ->
    `["--depth-lambda", "0.5", "--pose-opt", "--no-prune"]` (gs_trainer.train's `--x/--no-x` booleans)."""
    out: list[str] = []
    for key, value in options.items():
        name = key.replace("_", "-")
        if isinstance(value, bool):
            out.append(f"--{name}" if value else f"--no-{name}")
        else:
            out += [f"--{name}", str(value)]
    return out


@dataclass
class SweepRun:
    name: str
    args: list[str]
    steps: int | None = None
    max_side: int | None = None
    gpu: str | None = None


def parse_sweep_config(path: Path) -> SweepRun:
    """One run of a sweep from a TOML or JSON file: `name`, `args` (a list of options, or a table of
    option = value), and optionally `steps`, `max_side`, `gpu`."""
    text = path.read_text()
    cfg: dict[str, Any] = json.loads(text) if path.suffix == ".json" else tomllib.loads(text)
    if "name" not in cfg:
        raise ValueError(f"{path}: needs a `name`")
    raw = cfg.get("args", [])
    if isinstance(raw, Mapping):
        args = args_from_mapping(raw)  # type: ignore[arg-type]
    elif isinstance(raw, list):
        args = [str(a) for a in raw]  # type: ignore[union-attr]
    else:
        raise ValueError(f"{path}: `args` must be a list or a table")
    return SweepRun(str(cfg["name"]), args, cfg.get("steps"), cfg.get("max_side"), cfg.get("gpu"))


def gpu_price(gpu: str | None) -> float:
    """Price per hour of a Modal GPU spec (`L40S`, `H100:2`), CPU and RAM included; 0 if unknown.
    `None` is a CPU-only function."""
    if gpu is None:
        return CPU_RAM_PER_HOUR
    kind, _, count = gpu.upper().partition(":")
    kind = GPU_ALIASES.get(kind, kind)
    if kind not in GPU_PRICES:
        return 0.0
    return GPU_PRICES[kind] * (int(count) if count else 1) + CPU_RAM_PER_HOUR


def parse_time(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def billed_seconds(run: Mapping[str, Any], now: datetime) -> float:
    """Seconds of container time across a run's attempts; an attempt still going counts up to `now`.

    A preempted attempt has no end; its time is counted up to the next attempt's start."""
    attempts: list[dict[str, Any]] = list(run.get("attempts", []))
    total = 0.0
    for i, a in enumerate(attempts):
        start = parse_time(a.get("started_at"))
        if start is None:
            continue
        end = parse_time(a.get("finished_at"))
        if end is None:
            nxt = attempts[i + 1].get("started_at") if i + 1 < len(attempts) else None
            end = parse_time(nxt) or (now if run.get("status") in ("running", None) else start)
        total += max(0.0, (end - start).total_seconds())
    return total


def cost_usd(run: Mapping[str, Any], now: datetime) -> float:
    return billed_seconds(run, now) / 3600 * gpu_price(run.get("gpu", DEFAULT_GPU))


def elapsed_s(run: Mapping[str, Any], now: datetime) -> float | None:
    """Wall time since the run was submitted (includes waiting for a GPU and image build)."""
    start = parse_time(run.get("submitted_at") or run.get("started_at"))
    if start is None:
        return None
    end = parse_time(run.get("finished_at")) or now
    return max(0.0, (end - start).total_seconds())


def run_status(run: Mapping[str, Any], live_app_ids: set[str]) -> str:
    """The status `ps` shows: done / failed as the run recorded it; a run that says it is running is
    `running` (or `resumed` after a preemption) only while its app is alive, else `stopped` (stopped by
    hand, timed out, or the container died before it could record anything)."""
    status = str(run.get("status", "unknown"))
    if status in ("running", "starting"):
        if run.get("app_id") in live_app_ids:
            return "resumed" if len(run.get("attempts", [])) > 1 else status
        return "stopped"
    return status


def is_our_app(description: str) -> bool:
    """`flat-gs`, `flat-gs <anything>` (runs, prepare, eval) or `flat-gs-<anything>` (the build and test
    apps, and ad-hoc scripts named that way); not `flat-gsx`."""
    return description == APP_NAME or description.startswith((APP_NAME + " ", APP_NAME + "-"))


def live_apps(app_list: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Our apps that are still alive, from `modal app list --json` (keys: app_id, description, state,
    tasks, created_at, stopped_at)."""
    dead = ("stopped", "disabled")
    out: list[dict[str, Any]] = []
    for a in app_list:
        if not is_our_app(str(a.get("description") or "")):
            continue
        if str(a.get("state", "")).startswith(dead):
            continue
        out.append(dict(a))
    return out


# What `ps` groups the bill by, in this order (`billing_group`).
BILLING_GROUPS = (
    "training runs",
    "evals",
    "prepare / depth prior",
    "image builds",
    "tests",
    "other flat-gs",
    "not flat-gs",
)


def billing_group(description: str, object_id: str = "") -> str:
    """Which `BILLING_GROUPS` entry a `modal billing report` row belongs to, by its app's name."""
    if run_of_app(description) is not None:
        return "training runs"
    if description.startswith(f"{APP_NAME} eval"):
        return "evals"
    if description.startswith((f"{APP_NAME} prepare", f"{APP_NAME} depth-prior", f"{APP_NAME} pack-prior")):
        return "prepare / depth prior"
    if object_id.startswith("im-"):  # in case Modal ever bills a build to the image itself
        return "image builds"
    if not is_our_app(description):
        return "not flat-gs"
    if "build" in description:
        return "image builds"
    if "test" in description:
        return "tests"
    return "other flat-gs"


@dataclass
class Bill:
    """The actual spend from `modal billing report --resolution h --json`, in dollars."""

    month: Decimal = Decimal(0)
    today: Decimal = Decimal(0)
    by_group: dict[str, tuple[Decimal, Decimal]] = field(default_factory=dict)  # group -> (month, today)
    through: datetime | None = None  # end of the last hour in the report (UTC)
    rows: int = 0


def summarize_bill(rows: Sequence[Mapping[str, Any]], today: date, tz: tzinfo | None = None) -> Bill:
    """Sum hourly report rows (keys `object_id`, `description`, `interval_start`, `cost`; the start is
    naive UTC, the cost a decimal string) into the month's and `today`'s totals, per `billing_group`.
    "Today" is the calendar day in `tz` (None: this machine's time zone)."""
    bill = Bill()
    groups: dict[str, list[Decimal]] = {}
    for r in rows:
        try:
            cost = Decimal(str(r.get("cost") or 0))
            start = datetime.fromisoformat(str(r["interval_start"]))
        except (KeyError, ValueError, InvalidOperation):
            continue
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        is_today = start.astimezone(tz).date() == today
        g = groups.setdefault(billing_group(str(r.get("description") or ""), str(r.get("object_id") or "")),
                              [Decimal(0), Decimal(0)])  # fmt: skip
        g[0] += cost
        bill.month += cost
        if is_today:
            g[1] += cost
            bill.today += cost
        end = start + timedelta(hours=1)
        bill.through = max(bill.through, end) if bill.through else end
        bill.rows += 1
    bill.by_group = {k: (groups[k][0], groups[k][1]) for k in BILLING_GROUPS if k in groups}
    return bill


def start_attempt(
    run: Mapping[str, Any], meta: Mapping[str, Any] | None, now: datetime, env: Mapping[str, str]
) -> dict[str, Any]:
    """`run.json` at the start of an attempt (the first, or a retry after a preemption)."""
    out: dict[str, Any] = {**(meta or {}), **run}
    out.setdefault("started_at", now.isoformat())
    out["status"] = "running"
    out.pop("finished_at", None)
    out["image_id"] = env.get("MODAL_IMAGE_ID", out.get("image_id"))
    out["region"] = env.get("MODAL_REGION", out.get("region"))
    out["attempts"] = [
        *out.get("attempts", []),
        {"started_at": now.isoformat(), "task_id": env.get("MODAL_TASK_ID")},
    ]
    return out


def finish_attempt(
    run: Mapping[str, Any], returncode: int, ply_exists: bool, now: datetime
) -> dict[str, Any]:
    """`run.json` at the end: done only if the trainer exited cleanly and wrote the PLY."""
    out = dict(run)
    out["status"] = "done" if returncode == 0 and ply_exists else "failed"
    out["returncode"] = returncode
    out["finished_at"] = now.isoformat()
    attempts = [dict(a) for a in out.get("attempts", [])]
    if attempts:
        attempts[-1]["finished_at"] = now.isoformat()
        attempts[-1]["returncode"] = returncode
    out["attempts"] = attempts
    return out


def run_manifest(
    run: Mapping[str, Any], metrics: Mapping[str, Any], run_dir: Path, data: Path | None
) -> dict[str, Any]:
    """A `manifest.json` in the pipeline's format (`scanmodel.cache.run_stage`), so `scanmodel export` lists the
    downloaded run as a viewer layer (it needs `params.trainer`) and can tell whether it was trained on
    the current dataset (`inputs`)."""
    git = run.get("git", {})
    ply = run_dir / "splat.ply"
    scalars = {k: v for k, v in metrics.items() if isinstance(v, int | float | str | bool)}
    summary: dict[str, Any] = {
        "trainer": "gsplat",
        "run_id": run.get("run_id"),
        "iterations": run.get("steps"),
        **scalars,
        "gpu": run.get("gpu"),
        "cost_usd": round(cost_usd(run, parse_time(run.get("finished_at")) or datetime.now(UTC)), 2),
    }
    if ply.is_file():
        summary["ply_mb"] = round(ply.stat().st_size / 1e6, 1)
    inputs = pipeline_fingerprint([data / "transforms.json", data / "points.ply"]) if data else None
    code = hashlib.sha256(f"{git.get('commit')}:{git.get('diff_sha256')}".encode()).hexdigest()
    return {
        "inputs": inputs,
        "params": {
            "trainer": "gsplat",
            "iterations": run.get("steps"),
            "max_side": run.get("max_side"),
            "gpu": run.get("gpu"),
            "extra": run.get("args", []),
            "dataset": run.get("dataset"),
        },
        "code": code,
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "summary": summary,
    }


def git_provenance(repo: Path = REPO) -> dict[str, Any]:
    """The commit and a hash of the uncommitted changes, so a viewer layer can be traced to its code.
    Untracked files (not ignored) count too: `git diff HEAD` alone misses a new module."""

    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        diff = git("diff", "HEAD")
        untracked = sorted(git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"))
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "diff_sha256": None, "dirty": None}
    h = hashlib.sha256(diff)
    for name in filter(None, untracked):
        path = repo / name.decode()
        if path.is_file():
            h.update(name + b"\0" + path.read_bytes())
    dirty = bool(diff) or any(untracked)
    return {
        "commit": commit,
        "diff_sha256": h.hexdigest() if dirty else hashlib.sha256(b"").hexdigest(),
        "dirty": dirty,
    }


def gsplat_commit() -> str | None:
    """The gsplat git commit installed (from its `direct_url.json`), for the run's provenance."""
    from importlib import metadata

    try:
        info = json.loads(metadata.distribution("gsplat").read_text("direct_url.json") or "{}")
    except metadata.PackageNotFoundError:
        return None
    return info.get("vcs_info", {}).get("commit_id")


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


# --- Modal ------------------------------------------------------------------------------------------


def data_volume() -> modal.Volume:
    import modal

    return modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)


def runs_volume() -> modal.Volume:
    import modal

    return modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)


def _exists(vol: modal.Volume, path: str) -> bool:
    import modal

    try:
        return bool(vol.listdir(path))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return False


def _read_json(vol: modal.Volume, path: str) -> dict[str, Any] | None:
    import modal

    try:
        return json.loads(b"".join(vol.read_file(path)))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None


class CloudError(RuntimeError):
    """A problem the user can act on; the CLI prints it without a traceback."""


def _modal_cli(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """The `modal` CLI of this environment, for what the SDK has no public call for (app list/stop/logs)."""
    r = subprocess.run([sys.executable, "-m", "modal", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        # The CLI draws its errors in a box; keep only the words.
        words = " ".join(ln.strip(" │╭╮╰╯─").strip() for ln in r.stderr.splitlines()).split()
        message = " ".join(w for w in words if w != "Error") or f"exit {r.returncode}"
        hint = " Run `uv run --project trainer modal setup` once." if "token" in message.lower() else ""
        raise CloudError(f"`modal {' '.join(args)}` failed: {message}{hint}")
    return r


def list_live_apps() -> list[dict[str, Any]]:
    out = _modal_cli("app", "list", "--json").stdout
    return live_apps(json.loads(out or "[]"))


def report_running() -> None:
    """The safety net after every command that starts or finishes something: what is still billing."""
    try:
        apps = list_live_apps()
    except (CloudError, json.JSONDecodeError) as e:
        console.print(f"[yellow]could not list running Modal apps ({e}); check `just gs ps`[/yellow]")
        return
    if not apps:
        console.print("[dim]nothing running on Modal[/dim]")
        return
    console.print(f"[yellow]still running on Modal ({len(apps)}):[/yellow]")
    for a in apps:
        console.print(
            f"  {a.get('app_id')}  {a.get('description')}  {a.get('state')}  tasks {a.get('tasks', '-')}"
            f"  since {a.get('created_at')}"
        )
    console.print("  stop with `just gs stop <run>` or `just gs stop --all`")


def setup() -> None:
    """Check the token (a missing one raises `AuthError`, which the CLI turns into a hint) and create
    both volumes."""
    for vol, name in ((data_volume(), DATA_VOLUME), (runs_volume(), RUNS_VOLUME)):
        vol.hydrate()
        console.print(f"volume [bold]{name}[/bold]: ready")
    console.print(
        "Set a monthly budget once: modal.com -> Settings -> Usage (the plan suggests $50).\n"
        "Runs use Modal's default (US) region; the volumes are private to the workspace."
    )


def push(data: Path, da3: bool = False, da3_model: str = DA3_MODEL) -> str:
    """Upload a dataset under its content hash unless it is already there, then make what is missing on
    the cloud: the mesh priors and the LiDAR sigma table (`prepare`), with `da3` the DA3 depth and every
    fusion mode (`depth_prior`), and `prior.tar` (`pack_prior`). Returns the hash.

    Besides the files, the many small ones (`images/`, `depth/`, `confidence/`) go up once more as one
    uncompressed `bulk.tar` (`gs_trainer.datapack`), which the containers read in one go; a dataset pushed
    before the tars gets just the tar."""
    data = data.resolve()
    h = dataset_hash(data)
    vol = data_volume()
    prefix = f"datasets/{h}"
    old = _read_json(vol, f"{prefix}/{COMPLETE}")
    plan = upload_plan(old)
    if plan == "none":
        console.print(f"dataset {h}: already on the cloud")
    else:
        files = dataset_files(data) if plan == "all" else []
        with tempfile.TemporaryDirectory(prefix="gs-trainer-") as tmp:
            tar = Path(tmp) / BULK_TAR
            console.print(f"dataset {h}: packing images/, depth/, confidence/ into {BULK_TAR}")
            bulk = pack(data, BULK_TAR, tar)  # streamed to disk, file by file
            what = f"{len(files)} files and {BULK_TAR}" if files else f"only {BULK_TAR} (the files are there)"
            size = sum(p.stat().st_size for p, _ in files) + tar.stat().st_size
            console.print(f"dataset {h}: uploading {what}, {size / 1e9:.2f} GB")
            with vol.batch_upload(force=True) as batch:
                for path, rel in files:
                    batch.put_file(path, f"/{prefix}/{rel}")
                batch.put_file(tar, f"/{prefix}/{BULK_TAR}")
        marker = complete_marker(old, h, files, data, bulk, _now())
        with vol.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(json.dumps(marker).encode()), f"/{prefix}/{COMPLETE}")
        console.print(f"dataset {h}: uploaded")
    if not (_exists(vol, f"{prefix}/prior/meta.json") and _exists(vol, f"{prefix}/prior/sigma.json")):
        prepare_remote(h)  # writes prior.tar too
    if da3:
        report = _read_json(vol, f"{prefix}/prior/fusion.json")
        meta = _read_json(vol, f"{prefix}/prior/da3/meta.json")
        if needs_depth_prior(report, meta, da3_model):
            depth_prior_remote(h, da3_model)  # rewrites prior.tar
        else:
            console.print(f"dataset {h}: DA3 ({da3_model}) and fused depth already on the cloud")
    if not _exists(vol, f"{prefix}/{PRIOR_TAR}"):
        pack_prior_remote(h)
    return h


def prepare_remote(h: str, args: Sequence[str] = ()) -> None:
    """Mesh depth and normals for every view, then the LiDAR sigma table, on a many-core CPU container,
    cached in the dataset."""
    import modal

    from gs_trainer import modal_app

    console.print(f"dataset {h}: making priors on the cloud (prepare: priors, calibrate)")
    with modal.enable_output(), modal_app.app.run(name=f"{APP_NAME} prepare {h}"):
        rc = modal_app.prepare.remote(h, list(args))
    if rc != 0:
        raise CloudError(f"prepare failed on dataset {h} (exit {rc}); see the output above")


def pack_prior_remote(h: str) -> None:
    """`prior.tar` of a dataset whose priors were made before the tars (`modal_app.pack_prior`, CPU)."""
    import modal

    from gs_trainer import modal_app

    console.print(f"dataset {h}: packing prior/ into {PRIOR_TAR} on the cloud (pack_prior)")
    with modal.enable_output(), modal_app.app.run(name=f"{APP_NAME} pack-prior {h}"):
        rc = modal_app.pack_prior.remote(h)
    if rc != 0:
        raise CloudError(f"pack_prior failed on dataset {h} (exit {rc}); see the output above")


def depth_prior_remote(h: str, model: str = DA3_MODEL) -> None:
    """DA3 depth for every view on a GPU, then every fusion mode, cached in the dataset."""
    import modal

    from gs_trainer import modal_app

    console.print(f"dataset {h}: DA3 ({model}) and depth fusion on the cloud (depth_prior)")
    with modal.enable_output(), modal_app.app.run(name=f"{APP_NAME} depth-prior {h}"):
        rc = modal_app.depth_prior.remote(h, model)
    if rc != 0:
        raise CloudError(f"depth_prior failed on dataset {h} (exit {rc}); see the output above")


def check_depth_targets(h: str, arg_lists: Iterable[Sequence[str]]) -> None:
    """Refuse runs whose `--depth-target` is a fusion mode the dataset doesn't have on the cloud, before
    a GPU starts (the trainer would refuse too, but only after the container came up)."""
    wanted = sorted({depth_target_of(a) for a in arg_lists} - {"lidar"})
    if not wanted:
        return
    unknown = [t for t in wanted if t not in FUSION_MODES]
    if unknown:
        raise CloudError(f"unknown --depth-target {', '.join(unknown)}: lidar or {', '.join(FUSION_MODES)}")
    have = fused_modes_of(_read_json(data_volume(), f"datasets/{h}/prior/fusion.json"))
    missing = [t for t in wanted if t not in have]
    if missing:
        raise CloudError(
            f"--depth-target {', '.join(missing)} needs the fused depth priors, and dataset {h} has none"
            " on the cloud yet: run `just gs push --da3` first (DA3 and the fusion, on a GPU)"
        )


def _now() -> datetime:
    return datetime.now(UTC)


def start_train(
    data: Path, h: str, name: str, steps: int | None, max_side: int | None, gpu: str, extra: Sequence[str],
    *, detach: bool, project: str | None = None, out: Path | None = None,
) -> tuple[str, dict[str, Any] | None]:  # fmt: skip
    """Start one run as its own detached app. Attached (`detach=False`) it then waits for the result,
    with the remote output (the `step=` lines) printed as it comes; Ctrl-C stops the run. Returns the
    run id and the remote function's result (None when detached or when the connection was lost)."""
    import modal

    from gs_trainer import modal_app

    run_id = make_run_id(name, datetime.now())
    args = train_args(steps, max_side, extra)
    meta: dict[str, Any] = {
        "run_id": run_id,
        "name": name,
        "dataset": h,
        "data_dir": store_path(data),  # repo-relative (`store_path`); older runs have absolute paths
        "project": project,
        "local_out": store_path(out),
        "gpu": gpu,
        "steps": steps,
        "max_side": max_side,
        "args": args,
        "git": git_provenance(),
        "submitted_at": _now().isoformat(),
    }
    result: dict[str, Any] | None = None
    # detach=True even when attached: a lost connection (laptop lid) then leaves the run going instead
    # of killing it; Ctrl-C stops it explicitly below.
    with modal.enable_output(), modal_app.app.run(name=app_description(run_id), detach=True):
        fn = modal_app.train if gpu == DEFAULT_GPU else modal_app.train.with_options(gpu=gpu)
        meta["app_id"] = modal_app.app.app_id
        call = fn.spawn(h, run_id, args, meta)
        console.print(f"run [bold]{run_id}[/bold] started (app {meta['app_id']}, call {call.object_id})")
        if detach:
            console.print(
                f"follow: just gs logs {run_id}   stop: just gs stop {run_id}   fetch: just gs pull {run_id}"
            )
            return run_id, None
        try:
            result = call.get()
        except KeyboardInterrupt:
            console.print(f"[yellow]stopping {run_id}[/yellow]")
            stop_app(str(meta["app_id"]))
            raise
        except (ConnectionError, OSError, modal.exception.ConnectionError) as e:
            console.print(
                f"[yellow]lost the connection ({e}); the run continues: just gs logs {run_id}[/yellow]"
            )
        except modal.exception.FunctionTimeoutError as e:
            result = {"run_id": run_id, "status": "timeout", "error": str(e)}
        except modal.exception.Error as e:  # the function itself raised, after its retries
            result = {"run_id": run_id, "status": "failed", "error": str(e)}
    return run_id, result


def train(
    data: Path, out: Path | None, name: str, steps: int | None, gpu: str, max_side: int | None,
    detach: bool, extra: Sequence[str], *, project: str | None = None,
) -> int:  # fmt: skip
    """`gs-trainer cloud train` (the contract in trainer/README.md). Exit code 0 only if splat.ply arrived."""
    import modal

    try:
        h = push(data)
        check_depth_targets(h, [extra])
        options: dict[str, Any] = {"detach": detach}
        if project is not None:
            options["project"] = project
        if out is not None:
            options["out"] = out
        run_id, result = start_train(data, h, name, steps, max_side, gpu, extra, **options)
        if detach:
            return 0
        if result is None:
            return 1
        dest = out or default_run_dir(run_id, project=project, data=data)
        if result.get("status") == "done":
            pull(run_id, dest, data)
            if (dest / "splat.ply").is_file():
                return 0
        console.print(f"[red]run {run_id}: {result.get('status')}[/red] {result.get('error', '')}")
        try:  # whatever the run left (train.log, run.json), to see why
            pull(run_id, dest, data)
            console.print(f"see {dest / 'train.log'}")
        except (CloudError, modal.exception.Error) as e:
            console.print(f"nothing to download from {run_id} ({e})")
        return 1
    finally:
        report_running()


def sweep(data: Path, configs: Sequence[Path], gpu: str, *, project: str | None = None) -> list[str]:
    """One detached run per config, in parallel, all on the same uploaded dataset."""
    runs = [parse_sweep_config(c) for c in configs]  # fail on a bad file before starting anything
    names = [slug(r.name) for r in runs]
    if len(set(names)) < len(names):
        raise CloudError(f"sweep configs need distinct names: {names}")
    h = push(data)
    check_depth_targets(h, [r.args for r in runs])
    ids: list[str] = []
    try:
        for r in runs:
            options: dict[str, Any] = {"project": project} if project is not None else {}
            run_id, _ = start_train(
                data, h, r.name, r.steps, r.max_side, r.gpu or gpu, r.args, detach=True, **options
            )
            ids.append(run_id)
    finally:
        report_running()
    return ids


SETTINGS = "settings.toml"


def _settings_project() -> str | None:
    """The top-level `project` of the repo's tracked `settings.toml` (the repo's default scan), or None.
    Read here with tomllib rather than through the pipeline package, which the trainer doesn't import."""
    path = REPO / SETTINGS
    if not path.is_file():
        return None
    try:
        value = tomllib.loads(path.read_text()).get("project")
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise CloudError(f"can't read {path}: {e}") from e
    if value is None:
        return None
    if not isinstance(value, str):
        raise CloudError(f"{path}: `project` must be a string, not {value!r}")
    return value.strip() or None


def selected_project(project: str | None = None) -> str | None:
    """The project (`data/<name>/`) the commands work on: `--project`, else `project` in the repo's
    `settings.toml`; None when neither names one (`require_project` turns that into an error). An empty
    value counts as not set."""
    name = (project or "").strip() or _settings_project()
    if name is None:
        return None
    if name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise CloudError(f"project {name!r} must be a simple directory name (data/<name>/)")
    return name


def require_project(project: str | None = None) -> str:
    """`selected_project`, or a `CloudError` saying how to pick one."""
    name = selected_project(project)
    if name is None:
        raise CloudError(f"no project: pass --project NAME or set `project` in {SETTINGS} at the repo root")
    return name


def project_dataset(project: str | None = None) -> Path:
    """The selected project's default dataset, `data/<project>/splat/datasets/default`."""
    return REPO / "data" / require_project(project) / "splat" / "datasets" / "default"


def default_run_dir(run_id: str, *, project: str | None = None, data: Path | None = None) -> Path:
    """`data/<project>/splat/runs/<run id>`: the selected project's, else the one `data` (a dataset in
    `data/<p>/splat/datasets/`) belongs to; an error when neither says."""
    name = selected_project(project)
    if name:
        return REPO / "data" / name / "splat" / "runs" / run_id
    if data is not None:
        resolved = data.resolve()
        if resolved.parent.name == "datasets" and resolved.parent.parent.name == "splat":
            return resolved.parent.parent / "runs" / run_id
    return REPO / "data" / require_project() / "splat" / "runs" / run_id


def local_repo() -> Path | None:
    """The repo checkout this code runs from, or None inside a Modal container: there `gs_trainer` sits in
    /root with no repo around it, so `REPO` would be `/` and every path would look repo-relative."""
    return REPO if TRAINER_DIR.name == "trainer" and (TRAINER_DIR / "pyproject.toml").is_file() else None


def store_path(path: Path | str | None) -> str | None:
    """A local path as the trainer saves it (run metadata, manifests, eval.json, config.json): relative to
    the repo, in posix form, when it is inside the repo, so the record works from any checkout; else
    absolute. Unchanged inside a Modal container, whose paths aren't local ones. `stored_local_path`
    reads it back."""
    if path is None:
        return None
    repo = local_repo()
    if repo is None:
        return str(path)
    return repo_rel(os.path.abspath(path), repo)  # a relative CLI path is relative to the cwd


def stored_local_path(value: str | None, *, dataset: bool) -> Path | None:
    """A path from saved run metadata as a local path: a repo-relative one (`store_path`, newer runs)
    against this checkout, an absolute one (older runs) as it is. Missing pre-migration paths are mapped
    to the project layout; explicit CLI paths are left alone."""
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    if path.exists():
        return path
    try:
        parts = path.resolve().relative_to((REPO / "data" / "derived").resolve()).parts
    except ValueError:
        return path
    if len(parts) == 2 and parts[0] == "splat":
        project, leaf = "third", parts[1]
    elif len(parts) == 4 and parts[:3] in {("fourth", "derived", "splat"), ("fifth", "derived", "splat")}:
        project, leaf = parts[0], parts[3]
    else:
        return path
    root = REPO / "data" / project / "splat"
    if dataset:
        if leaf == "dataset":
            return root / "datasets" / "default"
        if leaf.startswith("dataset-") and leaf.removeprefix("dataset-"):
            return root / "datasets" / leaf.removeprefix("dataset-")
        return path
    if leaf == "dataset" or leaf.startswith("dataset-"):
        return path
    return root / "runs" / leaf


def list_runs() -> list[dict[str, Any]]:
    """Every run on the runs volume with its `run.json` and `progress.json` (`_progress`), oldest first."""
    from concurrent.futures import ThreadPoolExecutor

    from modal.types import FileEntryType

    vol = runs_volume()
    dirs = [
        e.path.strip("/")
        for e in vol.listdir("/")
        if e.type == FileEntryType.DIRECTORY and not e.path.strip("/").startswith("_")
    ]

    def load(d: str) -> dict[str, Any] | None:
        run = _read_json(vol, f"{d}/run.json")
        if run is None:
            return None
        run.setdefault("run_id", d)
        run["_progress"] = _read_json(vol, f"{d}/progress.json") or {}
        # When the run last wrote anything: a run stopped by hand (or killed) never records its end, and
        # counting it up to now inflated the estimate (the first night: $69 shown, ~$12 real).
        # Only runs that never recorded an end need it (a listing per run hits Modal's rate limit).
        if run.get("status") in ("running", "starting") and not run.get("finished_at"):
            try:
                mtimes = [e.mtime for e in vol.listdir(d) if getattr(e, "mtime", None)]
            except Exception:  # rate limited or gone: fall back to counting up to now
                mtimes = []
            if mtimes:
                run["_last_seen"] = datetime.fromtimestamp(max(mtimes), UTC).isoformat()
        return run

    with ThreadPoolExecutor(8) as pool:
        runs = [r for r in pool.map(load, sorted(dirs)) if r]
    return runs


def fetch_bill(today: date) -> Bill:
    """This month's actual spend, hour by hour, from `modal billing report` (the CLI: it parses
    "this month" and prints plain rows)."""
    # Hourly reports can't span more than 7 days: the month's earlier days come daily, today's completed
    # hours hourly (both UTC days).
    # The daily report does include the day in progress (so far), so its row for today (UTC) is dropped.
    utc_today = datetime.now(UTC).date().isoformat()
    rows: list[Any] = []
    for args in (("--for", "this month", "--resolution", "d"), ("--for", "today", "--resolution", "h")):
        out = _modal_cli("billing", "report", *args, "--json").stdout
        part = json.loads(out or "[]")
        if not isinstance(part, list):
            raise CloudError(f"unexpected billing report: {str(part)[:80]}")
        if "d" in args:
            part = [r for r in part if not str(r.get("interval_start", "")).startswith(utc_today)]
        rows += part
    return summarize_bill(rows, today)


def show_bill() -> None:
    """The bill table of `ps`; one warning line instead when the report can't be had."""
    try:
        bill = fetch_bill(datetime.now().astimezone().date())
    except (CloudError, OSError, ValueError, TypeError) as e:  # JSONDecodeError is a ValueError
        console.print(f"[yellow]no actual bill ({e}); see modal.com -> Settings -> Usage[/yellow]")
        return
    table = Table("spent on Modal (actual)", "today", "this month", title="Modal bill")
    for group, (month, today) in bill.by_group.items():
        table.add_row(group, f"${today:.2f}", f"${month:.2f}")
    table.add_row("[bold]total[/bold]", f"[bold]${bill.today:.2f}[/bold]", f"[bold]${bill.month:.2f}[/bold]")
    console.print(table)
    through = f" up to {bill.through.astimezone():%H:%M} local time" if bill.through else ""
    console.print(
        f"[dim]full hours only{through}: the current hour (a running job's latest minutes) isn't in it"
        " yet. Month in UTC, today in local time.[/dim]"
    )


def show_apps(apps: Sequence[Mapping[str, Any]]) -> None:
    """Every live app of ours (runs, prepare, eval, builds, tests, ad-hoc `flat-gs*` names)."""
    if not apps:
        console.print("[dim]nothing running on Modal[/dim]")
        return
    table = Table("app", "name", "state", "tasks", "since", title="live flat-gs apps on Modal")
    for a in apps:
        table.add_row(
            str(a.get("app_id")), str(a.get("description")), str(a.get("state")), str(a.get("tasks", "-")),
            str(a.get("created_at", "-")),
        )  # fmt: skip
    console.print(table)
    console.print("  stop with `just gs stop <run>`, `just gs stop --all` or `modal app stop <app>`")


def ps() -> None:
    """Every run's status, step, elapsed time, GPU and estimated cost, the actual bill, and every live
    app of ours."""
    apps = list_live_apps()
    live = {str(a["app_id"]) for a in apps}
    now = _now()
    table = Table("run", "status", "step", "splats", "eta", "elapsed", "gpu", "cost", title="gs-trainer runs")
    total = 0.0
    for run in list_runs():
        prog = run.get("_progress", {})
        step, steps = prog.get("step"), prog.get("total") or run.get("steps")
        status = run_status(run, live)
        # A stopped run never wrote its end: its time ends where its files stop changing.
        run_now = (parse_time(run.get("_last_seen")) or now) if status == "stopped" else now
        cost = cost_usd(run, run_now)
        total += cost
        splats = prog.get("splats")
        table.add_row(
            str(run["run_id"]),
            status,
            f"{step}/{steps}" if step is not None else "-",
            f"{splats:,}" if isinstance(splats, int) else "-",
            fmt_duration(prog.get("eta")) if status in ("running", "resumed") else "-",
            fmt_duration(elapsed_s(run, run_now)),
            str(run.get("gpu", "-")),
            f"${cost:.2f}",
        )
    console.print(table)
    console.print(
        f"runs' estimate: ${total:.2f} (training containers only, from their run time and list prices;"
        " the actual bill, builds and evals included, is below)"
    )
    show_bill()
    show_apps(apps)


def _run_json(run_id: str) -> dict[str, Any]:
    run = _read_json(runs_volume(), f"{run_id}/run.json")
    if run is None:
        raise CloudError(f"no run {run_id} on the runs volume (`just gs ps` lists them)")
    return run


def follow_lines(lines: Iterable[str], total: int, label: str) -> int | None:
    """Show trainer output with a progress bar: `step=` lines move the bar, other lines are printed.
    Returns the last step seen."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    last: int | None = None
    with Progress(
        TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as bar:  # fmt: skip
        task = bar.add_task(label, total=total)
        for line in lines:
            step = step_of(line)
            if step is not None:
                last = step
                bar.update(task, completed=step)
            else:
                bar.console.print(line.rstrip("\n"), markup=False, highlight=False)
    return last


def logs(run_id: str) -> None:
    """Follow a run's output (`modal app logs -f`) with a progress bar. Ctrl-C stops following only."""
    run = _run_json(run_id)
    app_id = str(run.get("app_id"))
    prog = _read_json(runs_volume(), f"{run_id}/progress.json") or {}
    total = int(prog.get("total") or run.get("steps") or DEFAULT_STEPS)
    alive = app_id in {str(a["app_id"]) for a in list_live_apps()}
    mode = ["-f"] if alive else ["--tail", "300"]
    p = subprocess.Popen(
        [sys.executable, "-m", "modal", "app", "logs", app_id, *mode],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )  # fmt: skip
    assert p.stdout is not None
    try:
        follow_lines(p.stdout, total, run_id)
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
        console.print(f"stopped following; {run_id} keeps running (`just gs stop {run_id}` stops it)")
    run = _run_json(run_id)
    console.print(f"{run_id}: {run_status(run, {str(a['app_id']) for a in list_live_apps()})}")


def _download_dir(vol: modal.Volume, remote: str, dest: Path, skip: Sequence[str] = ()) -> list[Path]:
    from modal.types import FileEntryType

    remote = remote.strip("/")
    got: list[Path] = []
    for e in vol.listdir(remote, recursive=True):
        rel = e.path.strip("/").removeprefix(remote).lstrip("/")
        if e.type != FileEntryType.FILE or not rel or rel.split("/", 1)[0] in skip:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        with tmp.open("wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        tmp.replace(target)
        got.append(target)
    return got


def pull(
    run_id: str, out: Path | None = None, data: Path | None = None, *, project: str | None = None
) -> Path:
    """Download a run (everything but its checkpoints) and write the pipeline's manifest.json there."""
    run = _run_json(run_id)
    data_dir = data or stored_local_path(run.get("data_dir"), dataset=True)
    saved_out = stored_local_path(run.get("local_out"), dataset=False)
    selected = selected_project(project)
    dest = (
        out
        or (saved_out if selected is None else None)
        or default_run_dir(run_id, project=selected or run.get("project"), data=data_dir)
    )
    dest.mkdir(parents=True, exist_ok=True)
    files = _download_dir(runs_volume(), run_id, dest, skip=("ckpts",))
    console.print(f"{run_id}: {len(files)} files -> {dest}")
    metrics_file = dest / "metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.is_file() else {}
    if (dest / "splat.ply").is_file():
        manifest = run_manifest(run, metrics, dest, data_dir.resolve() if data_dir else None)
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        console.print(f"{run_id}: manifest.json written; `just export` shows it in the viewer")
    else:
        console.print(f"[yellow]{run_id}: no splat.ply (status {run.get('status')}); no manifest[/yellow]")
    return dest


def stop_app(app_id: str) -> None:
    r = _modal_cli("app", "stop", "-y", app_id, check=False)
    msg = (r.stdout + r.stderr).strip()
    console.print(f"stopped {app_id}" + (f" ({msg})" if msg else ""))


def stop(run_id: str | None, all_: bool) -> None:
    """Stop one run's app, or every flat-gs app. Billing ends within seconds."""
    try:
        if all_:
            for a in list_live_apps():
                stop_app(str(a["app_id"]))
        elif run_id:
            stop_app(str(_run_json(run_id)["app_id"]))
        else:
            raise ValueError("give a run id or --all")
    finally:
        report_running()


def evaluate(
    data: Path | None,
    ply: Path | None,
    run_id: str | None,
    out: Path | None,
    args: Sequence[str],
    *,
    project: str | None = None,
) -> Path:
    """Score a splat on the shared split on a cloud GPU: a run's PLY, or any local PLY (uploaded once,
    by content hash). Downloads the metrics and renders into `out`. A run is scored on the dataset it
    trained on unless `data` says otherwise (another scan's split scores garbage: PSNR ~7). The
    re-solved test poses (`--test-poses FILE`, else the local dataset's test_poses.json: `data`, or the
    run's `data_dir`) are uploaded beside the PLY; the eval refuses them if they were solved from another
    transforms.json / split.json."""
    import modal

    from gs_trainer import modal_app

    try:
        run_meta = _run_json(run_id) if run_id else {}
        trained_on = str(run_meta["dataset"]) if run_id else None
        if data is None:
            if trained_on is None:
                raise ValueError("give --data with --ply")
            h = trained_on
        else:
            h = push(data)
            if trained_on and h != trained_on:
                console.print(
                    f"[yellow]{run_id} trained on dataset {trained_on}; scoring it on {h} ({data})[/yellow]"
                )
        stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
        if run_id:
            remote_ply = f"{RUNS_MOUNT}/{run_id}/splat.ply"
            remote_out = f"{run_id}/eval-{stamp}"
            run_data = stored_local_path(run_meta.get("data_dir"), dataset=True)
            selected = selected_project(project)
            saved_out = (
                stored_local_path(run_meta.get("local_out"), dataset=False) if selected is None else None
            )
            run_dir = saved_out or default_run_dir(
                run_id, project=selected or run_meta.get("project"), data=run_data
            )
            dest = out or run_dir / f"eval-{stamp}"
        elif ply:
            digest = _sha256_file(ply)[:16]
            vol = runs_volume()
            if not _exists(vol, f"_uploads/{digest}.ply"):
                console.print(f"uploading {ply} ({ply.stat().st_size / 1e6:.0f} MB)")
                with vol.batch_upload(force=True) as batch:
                    batch.put_file(ply, f"/_uploads/{digest}.ply")
            remote_ply = f"{RUNS_MOUNT}/_uploads/{digest}.ply"
            remote_out = f"_evals/{slug(ply.parent.name + '-' + ply.stem)}-{stamp}"
            selected = selected_project(project)
            run_dir = (
                default_run_dir(slug(ply.parent.name + "-" + ply.stem), project=selected)
                if selected
                else ply.parent
            )
            dest = out or run_dir / f"eval-{stamp}"
        else:
            raise ValueError("give --ply or --run")
        local_data = data or stored_local_path(run_meta.get("data_dir"), dataset=True)
        args, views = eval_views_file(args)
        if views is not None:
            if not views.is_file():
                raise ValueError(f"no named views file {views}")
            name = f"_uploads/{_sha256_file(views)[:16]}.views.json"
            vol = runs_volume()
            if not _exists(vol, name):
                with vol.batch_upload(force=True) as batch:
                    batch.put_file(views, f"/{name}")
            args = [*args, "--views", f"{RUNS_MOUNT}/{name}"]
        args, poses = eval_test_poses(args, local_data / TEST_POSES if local_data else None)
        if poses is not None:
            if not poses.is_file():
                raise ValueError(f"no test poses file {poses}")
            name = f"_uploads/{_sha256_file(poses)[:16]}.test_poses.json"
            vol = runs_volume()
            if not _exists(vol, name):
                with vol.batch_upload(force=True) as batch:
                    batch.put_file(poses, f"/{name}")
            args = [*args, "--test-poses", f"{RUNS_MOUNT}/{name}"]
            console.print(f"test poses: {poses}")
        else:
            console.print(
                "test poses: none (`scanmodel test-poses` writes them): scored at the dataset poses only"
            )
        with modal.enable_output(), modal_app.app.run(name=f"{APP_NAME} eval {h}"):
            rc = modal_app.evaluate.remote(h, remote_ply, f"{RUNS_MOUNT}/{remote_out}", list(args))
        files = _download_dir(runs_volume(), remote_out, dest)
        console.print(f"eval: {len(files)} files -> {dest}" + ("" if rc == 0 else f" [red](exit {rc})[/red]"))
        return dest
    finally:
        report_running()


def gc(keep: int, everything: bool, yes: bool) -> None:
    """Delete old runs' checkpoints (keeping their PLY and metrics), or with `everything` both volumes:
    every run and every dataset (the scan photos)."""
    import modal

    live = {str(a["app_id"]) for a in list_live_apps()}
    if everything:
        if live:
            raise CloudError("something is still running; `just gs stop --all` first")
        if (
            not yes
            and input(f"delete the volumes {DATA_VOLUME} and {RUNS_VOLUME} for good? [y/N] ").lower() != "y"
        ):
            return
        for name in (DATA_VOLUME, RUNS_VOLUME):
            modal.Volume.objects.delete(name, allow_missing=True)
            console.print(f"deleted volume {name}")
        return
    vol = runs_volume()
    runs = sorted(list_runs(), key=lambda r: str(r.get("submitted_at") or r.get("started_at") or r["run_id"]))
    old = runs[: max(0, len(runs) - keep)]
    for run in old:
        if run_status(run, live) in ("running", "resumed"):
            continue
        if _exists(vol, f"{run['run_id']}/ckpts"):
            vol.remove_file(f"{run['run_id']}/ckpts", recursive=True)
            console.print(f"{run['run_id']}: checkpoints deleted")
    console.print(f"kept the checkpoints of the newest {min(keep, len(runs))} runs")


def build() -> str:
    """Build (or reuse, if cached) the image now, with the build log shown, in an app of its own
    (`flat-gs-build`). Returns the image id."""
    import modal

    from gs_trainer import modal_app

    build_app = modal.App.lookup(BUILD_APP, create_if_missing=True)
    console.print(f"building the image in app {BUILD_APP} ({build_app.app_id}); a cached one returns at once")
    try:
        with modal.enable_output():
            modal_app.image.build(build_app)
    except KeyboardInterrupt:
        console.print(
            f"[yellow]interrupted. A build already started may continue (and bill) on Modal's side. Check\n"
            f"  just gs ps   (or `uv run --project trainer modal app list`, or modal.com/apps)\n"
            f"and stop the app there if it is still live: modal app stop {build_app.app_id}[/yellow]"
        )
        raise SystemExit(130) from None
    image_id = str(modal_app.image.object_id)
    console.print(f"image [bold]{image_id}[/bold]")
    # The build app is a deployed app with no functions: stop it so `ps` shows it only while building.
    _modal_cli("app", "stop", "-y", str(build_app.app_id), check=False)
    return image_id


def gpu_test(args: Sequence[str]) -> int:
    """The trainer's tests on a cloud GPU (`modal_app.gpu_test`), as the app `flat-gs-gpu-tests`; returns
    pytest's exit code. Ctrl-C stops the app."""
    import modal

    from gs_trainer import modal_app

    try:
        with modal.enable_output(), modal_app.app.run(name=TEST_APP):
            return int(modal_app.gpu_test.remote(list(args)))
    finally:
        report_running()
