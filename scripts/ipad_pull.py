#!/usr/bin/env python3
"""Pull FlatBlender scan projects and the app's crash reports off a USB-connected iPad into data/.

Each app project (Documents/Projects/<project>/ in the app container) becomes
data/<project>/raw/<session>/ plus data/<project>/raw/project.json, the layout `scanmodel ingest` reads.
Every installed FlatBlender build is searched (dev.flatblender.FlatBlender and suffixed ids such as .pp).

Status of a session, from a quick top-level listing on the iPad (no per-frame files, so a few seconds):
  new         not in data/
  downloaded  pulled, and the iPad's top-level files still match what was pulled
  changed     pulled, but the iPad's top-level files differ now (pull re-copies it)
  incomplete  an interrupted copy, or (for sessions pulled before .pull.json existed) files missing
Each pulled session holds a .pull.json: when it was pulled and every file with its size as on the iPad.

A pull lists the chosen sessions in full, copies each into data/<project>/.ipad-partial/<session>/
(outside raw/, so ingest never sees half a session), checks every file's size, then moves it into raw/.
Sessions pulled before .pull.json existed are checked file by file and get one written.

The app writes no log files (it logs through os.Logger), but iOS keeps its crash reports. They go to
data/<project>/logs/ipad/ when they happened during or within an hour after that project's sessions,
otherwise to data/ipad-logs/.

    scripts/ipad_pull.py --list           # what is on the iPad and what is already local
    scripts/ipad_pull.py                  # pull everything new or changed, and the crash reports
    scripts/ipad_pull.py salut figaro     # only these projects (and their crash reports)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path

BUNDLE_PREFIX = "dev.flatblender.FlatBlender"
PROJECTS = "Documents/Projects"
REPO = Path(__file__).resolve().parent.parent
RECORD = ".pull.json"  # in each pulled session: what was on the iPad when it was pulled
PARTIAL = ".ipad-partial"  # data/<project>/.ipad-partial/<session>/: copies in progress
CRASH_DIR = Path("logs/ipad")  # under data/<project>/
UNMATCHED_CRASHES = "ipad-logs"  # under data/
CRASH_WINDOW = timedelta(hours=1)  # a crash this long after a project's last session still belongs to it
CRASH_DOMAIN = ["--domain-type", "systemCrashLogs"]
WORKERS = 8  # parallel devicectl listings


class PullError(Exception):
    """A failure that stops the whole run (no device, devicectl missing, ...)."""


@dataclass
class Entry:
    path: str  # relative to the listed folder
    size: int
    is_dir: bool
    modified: datetime | None


@dataclass
class Session:
    project: str
    name: str
    bundle: str
    modified: datetime | None = None  # the session folder's modification time, about when recording ended
    top: dict[str, int] = field(default_factory=dict)  # top-level files -> bytes (the quick listing)
    files: dict[str, int] | None = None  # every file -> bytes, listed only when needed

    @property
    def size(self) -> int:
        """Every file's bytes once listed in full, else only the top-level files' (an underestimate)."""
        return sum((self.files if self.files is not None else self.top).values())

    @property
    def source(self) -> str:
        return f"{PROJECTS}/{self.project}/{self.name}"

    @property
    def started(self) -> datetime | None:
        """From the folder name (YYYYMMDD-HHMMSS[-room], the iPad's local time, taken to be this Mac's)."""
        try:
            return datetime.strptime(self.name[:15], "%Y%m%d-%H%M%S").astimezone()
        except ValueError:
            return None


@dataclass
class Project:
    name: str
    bundle: str
    sessions: dict[str, Session] = field(default_factory=dict)
    has_manifest: bool = False  # project.json present

    def window(self) -> tuple[datetime, datetime] | None:
        """From the first session's start to CRASH_WINDOW after the last one ended."""
        starts = [t for s in self.sessions.values() if (t := s.started)]
        ends = [t for s in self.sessions.values() for t in (s.modified, s.started) if t]
        return (min(starts), max(ends) + CRASH_WINDOW) if starts else None


@dataclass
class Crash:
    path: str  # in the crash log domain
    size: int
    modified: datetime | None
    home: str | None = None  # project it belongs to

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def target(self, data: Path) -> Path:
        folder = data / self.home / CRASH_DIR if self.home else data / UNMATCHED_CRASHES
        return folder / self.name


# --- devicectl ------------------------------------------------------------------------------------


def devicectl(*args: str, timeout: float | None = 120) -> dict:
    """Run `xcrun devicectl <args> --json-output` and return the parsed result; raise on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        cmd = ["xcrun", "devicectl", *args, "--json-output", str(out)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as e:
            raise PullError("xcrun not found: install Xcode (or the command line tools)") from e
        except subprocess.TimeoutExpired as e:
            raise PullError(f"devicectl timed out after {timeout:.0f}s: {' '.join(args[:3])}") from e
        data = json.loads(out.read_text()) if out.exists() and out.stat().st_size else {}
        if proc.returncode != 0:
            lines = (proc.stderr or proc.stdout).strip().splitlines()
            msg = _error_text(data) or (lines[-1] if lines else f"exit code {proc.returncode}")
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=msg)
        return data.get("result", {})


def _error_text(data: dict) -> str:
    err = data.get("error") or {}
    info = err.get("userInfo", {})
    for key in ("NSLocalizedDescription", "NSLocalizedFailureReason"):
        if isinstance(info.get(key), dict) and "string" in info[key]:
            return info[key]["string"]
        if isinstance(info.get(key), str):
            return info[key]
    return ""


def container(bundle: str) -> list[str]:
    return ["--domain-type", "appDataContainer", "--domain-identifier", bundle]


def find_device(wanted: str | None) -> tuple[str, str]:
    """(udid, name) of the connected, paired physical iPad/iPhone, or the one matching `wanted`."""
    devices = devicectl("list", "devices").get("devices", [])
    physical = [d for d in devices if d.get("hardwareProperties", {}).get("reality") == "physical"]
    candidates = []
    for d in physical:
        udid = d.get("hardwareProperties", {}).get("udid", "")
        name = d.get("deviceProperties", {}).get("name", "?")
        ident = d.get("identifier", "")
        if wanted and wanted not in (udid, name, ident):
            continue
        conn = d.get("connectionProperties", {})
        candidates.append((udid, name, conn.get("pairingState"), conn.get("tunnelState")))
    if not candidates:
        what = f"matching {wanted!r}" if wanted else "connected"
        known = ", ".join(d.get("deviceProperties", {}).get("name", "?") for d in physical) or "none"
        raise PullError(f"no physical device {what} (known devices: {known}); plug the iPad in and unlock it")
    usable = [c for c in candidates if c[2] == "paired" and c[3] != "unavailable"]
    if not usable:
        udid, name, pairing, tunnel = candidates[0]
        raise PullError(f"{name} is not reachable (pairing: {pairing}, tunnel: {tunnel}); unlock it and trust this Mac")
    if len(usable) > 1:
        names = ", ".join(f"{n} ({u})" for u, n, *_ in usable)
        raise PullError(f"several devices connected: {names}; choose one with --device")
    return usable[0][0], usable[0][1]


def flatblender_bundles(udid: str) -> list[str]:
    apps = devicectl("device", "info", "apps", "--device", udid).get("apps", [])
    ids = sorted({a["bundleIdentifier"] for a in apps if a.get("bundleIdentifier", "").startswith(BUNDLE_PREFIX)})
    if not ids:
        raise PullError(f"no FlatBlender app ({BUNDLE_PREFIX}*) installed on the device")
    return ids


def list_dir(udid: str, domain: list[str], sub: str = "", recurse: bool = False) -> list[Entry]:
    """Entries under `sub` (paths relative to it), hidden ones left out. One level unless `recurse`."""
    args = ["device", "info", "files", "--device", udid, *domain]
    if sub:
        args += ["--subdirectory", sub]
    if not recurse:
        args.append("--no-recurse")
    result = devicectl(*args, timeout=300)
    prefix = f"{sub.rstrip('/')}/" if sub else ""
    entries = []
    for f in result.get("files", []):
        rel = (f.get("relativePath") or f.get("name", "")).removeprefix(prefix)
        if not rel or rel == sub.rstrip("/") or any(part.startswith(".") for part in rel.split("/")):
            continue
        meta = f.get("metadata", {})
        stamp = meta.get("lastModDate")
        entries.append(Entry(
            rel,
            int(meta.get("size", 0)),
            bool(f.get("resources", {}).get("isDirectory", False)),
            datetime.fromisoformat(stamp.replace("Z", "+00:00")) if stamp else None,
        ))
    return entries


def copy_from(udid: str, domain: list[str], source: str, dest: Path, timeout: float) -> None:
    """Copy a file or folder off the device to `dest` (the folder itself, not its parent)."""
    devicectl(
        "device", "copy", "from", "--device", udid, *domain,
        "--source", source, "--destination", str(dest), timeout=timeout,
    )


def list_in_full(udid: str, session: Session) -> None:
    entries = list_dir(udid, container(session.bundle), session.source, recurse=True)
    session.files = {e.path: e.size for e in entries if not e.is_dir}


def parallel(fn: Callable, items: list) -> list:
    """fn over items on WORKERS threads, in order; the first exception is raised."""
    if not items:
        return []
    with ThreadPoolExecutor(min(WORKERS, len(items))) as pool:
        return list(pool.map(fn, items))


# --- gathering ------------------------------------------------------------------------------------


def gather(udid: str, bundles: list[str], only: list[str]) -> tuple[dict[str, Project], dict[str, Project]]:
    """(every project, the selected ones). Sessions of every project are named and dated; only the
    selected ones get their top-level files listed. The same project in two containers is an error."""
    everything: dict[str, Project] = {}
    for bundle in bundles:
        try:
            entries = list_dir(udid, container(bundle), PROJECTS)
        except subprocess.CalledProcessError as e:
            print(f"warning: cannot list {bundle}: {e.output}", file=sys.stderr)
            continue
        for e in entries:
            if not e.is_dir:  # exported zips next to the projects
                continue
            if e.path in everything:
                raise PullError(f"project {e.path!r} exists in both {everything[e.path].bundle} and {bundle}; pull with --bundle")
            everything[e.path] = Project(e.path, bundle)

    unknown = [p for p in only if p not in everything]
    if unknown:
        raise PullError(f"not on the device: {', '.join(unknown)} (found: {', '.join(sorted(everything)) or 'none'})")

    def sessions_of(project: Project) -> None:
        for e in list_dir(udid, container(project.bundle), f"{PROJECTS}/{project.name}"):
            if e.is_dir:
                project.sessions[e.path] = Session(project.name, e.path, project.bundle, e.modified)
            elif e.path == "project.json":
                project.has_manifest = True

    def top_of(session: Session) -> None:
        for e in list_dir(udid, container(session.bundle), session.source):
            if not e.is_dir:
                session.top[e.path] = e.size

    parallel(sessions_of, list(everything.values()))
    selected = {p: everything[p] for p in only} if only else everything
    parallel(top_of, [s for p in selected.values() for s in p.sessions.values()])
    return everything, selected


def gather_crashes(udid: str, everything: dict[str, Project]) -> list[Crash]:
    """The app's crash reports, each assigned to the project whose sessions it happened during/after."""
    try:
        entries = list_dir(udid, CRASH_DOMAIN, recurse=True)
    except subprocess.CalledProcessError as e:
        print(f"warning: cannot list crash reports: {e.output}", file=sys.stderr)
        return []
    crashes = [
        Crash(e.path, e.size, e.modified)
        for e in entries
        if not e.is_dir and e.path.rsplit("/", 1)[-1].startswith("FlatBlender") and e.path.endswith(".ips")
    ]
    windows = [(name, w) for name, p in everything.items() if (w := p.window())]
    for crash in crashes:
        if crash.modified:
            hits = [(start, name) for name, (start, end) in windows if start <= crash.modified <= end]
            crash.home = max(hits)[1] if hits else None  # the latest project that started before it
    return sorted(crashes, key=lambda c: c.modified or datetime.min.replace(tzinfo=timezone.utc))


# --- local side -----------------------------------------------------------------------------------


def local_files(folder: Path) -> dict[str, int]:
    if not folder.is_dir():
        return {}
    return {
        str(p.relative_to(folder)): p.stat().st_size
        for p in folder.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    }


def tree_bytes(folder: Path) -> int:
    """Bytes under `folder` so far (it grows while devicectl copies into it)."""
    total = 0
    for root, _, files in os.walk(folder):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:  # renamed or removed between listing and stat
                pass
    return total


def diff(expected: dict[str, int], actual: dict[str, int]) -> str:
    """'' when `actual` has every expected file with the same size, else a short description."""
    missing = [p for p in expected if p not in actual]
    wrong = [p for p in expected if p in actual and actual[p] != expected[p]]
    problems = []
    if missing:
        problems.append(f"{len(missing)} missing (e.g. {missing[0]})")
    if wrong:
        problems.append(f"{len(wrong)} with a different size (e.g. {wrong[0]})")
    return ", ".join(problems)


def read_record(folder: Path) -> dict | None:
    try:
        return json.loads((folder / RECORD).read_text())
    except (OSError, ValueError):
        return None


def write_record(folder: Path, session: Session, device: str) -> None:
    record = {
        "pulled": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "bundle": session.bundle,
        "source": session.source,
        "bytes": session.size,
        "top": session.top,
        "files": session.files,
    }
    (folder / RECORD).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def raw_dir(data: Path, project: str) -> Path:
    return data / project / "raw"


def partial_dir(data: Path, session: Session) -> Path:
    return data / session.project / PARTIAL / session.name


def status(session: Session, data: Path) -> tuple[str, bool]:
    """(new | downloaded | changed | incomplete, whether the local copy has a .pull.json)."""
    final = raw_dir(data, session.project) / session.name
    if not final.is_dir():
        return ("incomplete" if partial_dir(data, session).exists() else "new"), False
    record = read_record(final)
    if record is not None:
        return ("downloaded" if record.get("top") == session.top else "changed"), True
    # pulled before .pull.json existed: compare the top-level files on disk
    local = {p.name: p.stat().st_size for p in final.iterdir() if p.is_file() and not p.name.startswith(".")}
    if any(name not in local for name in session.top):
        return "incomplete", False
    return ("downloaded" if all(local[n] == size for n, size in session.top.items()) else "changed"), False


def crash_status(crash: Crash, data: Path) -> str:
    target = crash.target(data)
    return "downloaded" if target.is_file() and target.stat().st_size == crash.size else "new"


# --- pulling --------------------------------------------------------------------------------------


def pull_session(
    udid: str, session: Session, data: Path, timeout: float, device: str, tick: Callable[[int], None]
) -> None:
    """Copy one (fully listed) session into .ipad-partial/, verify it, record it, move it into raw/."""
    assert session.files is not None
    raw = raw_dir(data, session.project)
    final = raw / session.name
    partial = partial_dir(data, session)
    if partial.exists():
        shutil.rmtree(partial)
    partial.parent.mkdir(parents=True, exist_ok=True)
    errors: list[BaseException] = []

    def copy() -> None:
        try:
            copy_from(udid, container(session.bundle), session.source, partial, timeout)
        except BaseException as e:  # handed back to the main thread below
            errors.append(e)

    worker = threading.Thread(target=copy, daemon=True)  # daemon: Ctrl-C does not wait for it
    worker.start()
    while worker.is_alive():
        tick(tree_bytes(partial))
        worker.join(0.5)
    if errors:
        raise errors[0]
    tick(tree_bytes(partial))
    if problem := diff(session.files, local_files(partial)):
        raise RuntimeError(f"copy incomplete: {problem} (kept in {partial})")
    write_record(partial, session, device)
    raw.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
    try:
        partial.parent.rmdir()  # only when empty
    except OSError:
        pass


def pull_crashes(udid: str, crashes: list[Crash], data: Path) -> list[str]:
    """Copy the new crash reports; returns the ones that failed."""
    failed = []
    for crash in crashes:
        target = crash.target(data)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".partial")
            copy_from(udid, CRASH_DOMAIN, crash.path, tmp, 120)
            tmp.replace(target)
            print(f"crash report {crash.name} -> {target.relative_to(data)}")
        except (subprocess.CalledProcessError, PullError, OSError) as e:
            failed.append(f"crash report {crash.name}")
            print(f"crash report {crash.name}  FAILED: {getattr(e, 'output', None) or e}", file=sys.stderr)
    return failed


class ProjectBar:
    """One progress line per project: bytes copied over the project's sessions, rate and ETA.

    Redrawn in place on a terminal; elsewhere (a pipe, a log) only a line per finished session is printed.
    """

    def __init__(self, name: str, sessions: list[Session]) -> None:
        self.name = name
        self.total = sum(s.size for s in sessions) or 1
        self.count = len(sessions)
        self.done = 0  # bytes of the sessions already finished (or given up on)
        self.failed = 0
        self.start = time.monotonic()
        self.live = sys.stdout.isatty()

    def draw(self, index: int, session: str, current: int) -> None:
        if not self.live:
            return
        copied = self.done + current
        frac = min(copied / self.total, 1.0)
        elapsed = time.monotonic() - self.start
        rate = copied / elapsed if elapsed > 1 else 0
        eta = f"ETA {_clock((self.total - copied) / rate)}" if rate and frac < 1 else ""
        sizes = f"{human(copied)}/{human(self.total)}"
        right = f" {frac:4.0%}  {sizes:>19s}  {human(rate) + '/s':>10s}  {eta:<11s} [{index}/{self.count}] {session}"
        width = shutil.get_terminal_size((100, 20)).columns
        bar_w = max(10, min(40, width - len(self.name) - len(right) - 6))
        filled = int(bar_w * frac)
        line = f"{self.name}  |{'█' * filled}{'░' * (bar_w - filled)}|{right}"
        sys.stdout.write("\r\033[K" + line[: width - 1])
        sys.stdout.flush()

    def session_done(self, session: Session, seconds: float) -> None:
        self.done += session.size
        if not self.live:
            print(f"{self.name}/{session.name}  {human(session.size)}  ok ({seconds:.0f}s)", flush=True)

    def session_failed(self, session: Session, message: str) -> None:
        self.failed += 1
        self.done += session.size  # keep the bar honest about what is left
        if self.live:
            sys.stdout.write("\r\033[K")
        print(f"{self.name}/{session.name}  FAILED: {message}", file=sys.stderr, flush=True)

    def finish(self) -> None:
        elapsed = time.monotonic() - self.start
        state = f"{self.failed} of {self.count} failed" if self.failed else f"{self.count} session(s) ok"
        line = f"{self.name}  {human(self.total)} in {_clock(elapsed)} — {state}"
        if self.live:
            sys.stdout.write("\r\033[K")
        print(line, flush=True)


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


# --- main -----------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("projects", nargs="*", help="projects to pull (default: all)")
    parser.add_argument("--list", action="store_true", help="show what is on the device and stop")
    parser.add_argument("--dry-run", "-n", action="store_true", help="show what would be copied")
    parser.add_argument("--force", action="store_true", help="re-copy sessions that are already downloaded")
    parser.add_argument("--no-crashes", action="store_true", help="leave the crash reports")
    parser.add_argument("--device", help="device name or UDID (default: the one connected)")
    parser.add_argument("--bundle", action="append", help=f"app bundle id(s) to read (default: every {BUNDLE_PREFIX}*)")
    parser.add_argument("--data", type=Path, default=REPO / "data", help="data folder (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds allowed per session copy")
    args = parser.parse_args()
    data: Path = args.data

    try:
        udid, device = find_device(args.device)
        bundles = args.bundle or flatblender_bundles(udid)
        print(f"device   {device} ({udid})")
        print(f"apps     {', '.join(bundles)}")
        print(f"data     {data}")
        everything, projects = gather(udid, bundles, args.projects)
        crashes = [] if args.no_crashes else gather_crashes(udid, everything)
    except PullError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"error: devicectl failed: {e.output}", file=sys.stderr)
        return 2
    if args.projects:  # crash reports of the chosen projects only
        crashes = [c for c in crashes if c.home in projects]

    todo: list[Session] = []
    unrecorded: list[Session] = []  # downloaded before .pull.json existed: verified and recorded on a pull
    approximate = False  # some size counts top-level files only
    print()
    for project in sorted(projects.values(), key=lambda p: p.name):
        print(f"{project.name}  ({project.bundle.removeprefix(BUNDLE_PREFIX) or 'current app'})")
        for session in sorted(project.sessions.values(), key=lambda s: s.name):
            state, recorded = status(session, data)
            if state != "downloaded" or args.force:
                todo.append(session)
            elif not recorded:
                unrecorded.append(session)
            record = read_record(raw_dir(data, project.name) / session.name) if recorded else None
            exact = record is not None and state == "downloaded"
            approximate |= not exact
            size = human(record["bytes"]) if exact else f"≥{human(session.size)}"
            print(f"  {session.name:36s} {size:>10s}  {state}")
    if crashes:
        print("crash reports")
        for crash in crashes:
            when = crash.modified.astimezone().strftime("%Y-%m-%d %H:%M") if crash.modified else "?"
            where = crash.target(data).parent.relative_to(data)
            print(f"  {crash.name:46s} {when}  -> {str(where) + '/':22s} {crash_status(crash, data)}")
    new_crashes = [c for c in crashes if crash_status(c, data) == "new"]

    size = f" (≥{human(sum(s.size for s in todo))})" if todo else ""
    print(f"\n{len(todo)} session(s) to copy{size}, {len(new_crashes)} crash report(s)")
    if unrecorded:
        print(f"{len(unrecorded)} session(s) pulled before .pull.json existed: a pull checks them file by file")
    if approximate:
        print("≥: top-level files only; the depth, confidence and stills folders add more")
    if args.list or args.dry_run or not (todo or unrecorded or new_crashes):
        return 0

    failed = pull_crashes(udid, new_crashes, data)

    print(f"\nlisting every file of {len(todo) + len(unrecorded)} session(s) ...", flush=True)
    try:
        parallel(lambda s: list_in_full(udid, s), todo + unrecorded)
    except (subprocess.CalledProcessError, PullError) as e:
        print(f"error: listing failed: {getattr(e, 'output', None) or e}", file=sys.stderr)
        return 2

    for session in unrecorded:  # check earlier pulls; record the good ones, re-copy the rest
        final = raw_dir(data, session.project) / session.name
        if problem := diff(session.files or {}, local_files(final)):
            print(f"{session.project}/{session.name}: {problem}; copying it again")
            todo.append(session)
        else:
            write_record(final, session, device)
    if unrecorded:
        ok = len(unrecorded) - sum(s in todo for s in unrecorded)
        print(f"checked {len(unrecorded)} earlier pull(s): {ok} complete (.pull.json written)")
    todo.sort(key=lambda s: (s.project, s.name))

    for project in {s.project for s in todo}:  # small and cheap: refresh project.json
        if not projects[project].has_manifest:
            continue
        raw = raw_dir(data, project)
        raw.mkdir(parents=True, exist_ok=True)
        try:
            tmp = raw / "project.json.partial"
            copy_from(udid, container(projects[project].bundle), f"{PROJECTS}/{project}/project.json", tmp, 120)
            tmp.replace(raw / "project.json")
        except (subprocess.CalledProcessError, PullError, OSError) as e:
            failed.append(f"{project}/project.json")
            print(f"{project}/project.json  FAILED: {getattr(e, 'output', None) or e}", file=sys.stderr)

    done_bytes = 0
    start = time.monotonic()
    if todo:
        print(f"copying {len(todo)} session(s), {human(sum(s.size for s in todo))}\n")
    for name, group in groupby(todo, key=lambda s: s.project):
        sessions = list(group)
        bar = ProjectBar(name, sessions)
        for i, session in enumerate(sessions, 1):
            t0 = time.monotonic()
            try:
                pull_session(
                    udid, session, data, args.timeout, device,
                    lambda n, i=i, session=session: bar.draw(i, session.name, n),
                )
            except KeyboardInterrupt:
                print("\ninterrupted; re-run to resume (finished sessions are kept)", file=sys.stderr)
                return 130
            except (subprocess.CalledProcessError, PullError, RuntimeError, OSError) as e:
                failed.append(f"{name}/{session.name}")
                bar.session_failed(session, getattr(e, "output", None) or str(e))
                continue
            done_bytes += session.size
            bar.session_done(session, time.monotonic() - t0)
        bar.finish()

    if todo:
        print(f"\ncopied {human(done_bytes)} in {_clock(time.monotonic() - start)}")
    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)} — re-run to retry", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
