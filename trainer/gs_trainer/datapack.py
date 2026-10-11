"""Tars of a dataset's many small files, so the cloud reads one big file instead of thousands.

The Modal volume is slow with many small files: copying `depth/`, `confidence/`, `prior/` to the
container's disk and reading 1,864 JPEGs from the volume took ~35 min of a 59 min run. So each dataset on
the volume also has

- `bulk.tar`: `images/`, `depth/`, `confidence/`, built on the Mac and uploaded by `gs-trainer cloud push`;
- `prior.tar`: `prior/` without `prior/da3/` (DA3's raw outputs, only fusion's input), written on the
  cloud when `prepare` or `depth_prior` finish (or by `pack_prior`, which `push` runs when it is missing).

Both are uncompressed (JPEG, PNG and npz are compressed already), with paths relative to the dataset.
In the container `modal_app._local_copy` copies each tar to local disk in one sequential read and unpacks
it; when a tar is missing it falls back to copying (or linking) file by file (`copy_plan`).

Standard library only: imported by `gs_trainer.cloud` on the Mac and by `gs_trainer.modal_app` in the container.
"""

from __future__ import annotations

import os
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

BULK_TAR = "bulk.tar"
PRIOR_TAR = "prior.tar"
# Which top-level folders go into which tar.
TAR_DIRS: dict[str, tuple[str, ...]] = {
    BULK_TAR: ("images", "depth", "confidence"),
    PRIOR_TAR: ("prior",),
}
# Left out of a tar (paths relative to the dataset).
TAR_EXCLUDE: dict[str, tuple[str, ...]] = {PRIOR_TAR: ("prior/da3",)}
# Without their tar, these folders are copied file by file (read every step: from the volume that took
# ~300 ms a step); everything else is linked (images are read once, then cached).
COPIED = ("depth", "confidence", "prior")
COPY_IGNORE = ("da3",)  # left out of the per-file copy of `prior/`, as of its tar


@dataclass(frozen=True)
class TarStats:
    files: int
    bytes: int  # the files' bytes (the tar is a little larger: headers, padding)


@dataclass(frozen=True)
class CopyPlan:
    """How the container makes its local copy of a dataset."""

    unpack: list[str]  # tars to copy to local disk and unpack
    copy: list[str]  # folders without a tar, copied file by file
    link: list[str]  # entries linked to the volume


def _skipped(rel: str) -> bool:
    """Dot files (`.DS_Store`, `.complete`) and the writers' half-written temp files (`x.tmp.npz`)."""
    parts = rel.split("/")
    return any(p.startswith(".") for p in parts) or ".tmp." in parts[-1] or parts[-1].endswith(".tmp")


def tar_members(data: Path, tar_name: str) -> list[tuple[Path, str]]:
    """The files of `data` that go into `tar_name`, as (path, path relative to the dataset), sorted."""
    exclude = TAR_EXCLUDE.get(tar_name, ())
    out: list[tuple[Path, str]] = []
    for sub in TAR_DIRS[tar_name]:
        d = data / sub
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            rel = f.relative_to(data).as_posix()
            if any(rel == e or rel.startswith(e + "/") for e in exclude) or _skipped(rel):
                continue
            if f.is_file():
                out.append((f, rel))
    return out


def _clean(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def write_tar(members: Iterable[tuple[Path, str]], dst: Path) -> TarStats:
    """An uncompressed tar of `members` at `dst`, streamed to disk file by file. Written to `<dst>.tmp`
    and renamed, so a reader never finds half a tar."""
    tmp = dst.with_name(dst.name + ".tmp")
    files = size = 0
    try:
        with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tar:
            for path, rel in members:
                tar.add(path, arcname=rel, recursive=False, filter=_clean)
                files += 1
                size += path.stat().st_size
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)
    return TarStats(files, size)


def pack(data: Path, tar_name: str, dst: Path | None = None) -> TarStats:
    """Write `tar_name` (`BULK_TAR` or `PRIOR_TAR`) of the dataset in `data`, by default into `data`."""
    return write_tar(tar_members(data, tar_name), dst or data / tar_name)


def extract(tar_path: Path, dst: Path) -> int:
    """Unpack a tar into `dst` (refusing paths outside it); returns the number of entries."""
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        tar.extractall(dst, members=members, filter="data")
    return len(members)


def copy_plan(entries: Iterable[str], prior: bool = True) -> CopyPlan:
    """Which tars to unpack, folders to copy and entries to link, from the names at the top of a dataset
    on the volume. A folder in a tar that is there comes from the tar; without the tar, `COPIED` folders
    are copied file by file and the rest linked, as before the tars. `prior=False` leaves `prior/` out
    altogether (the eval doesn't read it)."""
    present = set(entries)
    unpack: list[str] = []
    covered: set[str] = set()
    for tar, dirs in TAR_DIRS.items():
        if tar in present and (prior or tar != PRIOR_TAR):
            unpack.append(tar)
            covered.update(dirs)
    if not prior:
        covered.add("prior")
    rest = sorted(e for e in present - covered - set(TAR_DIRS) if not _skipped(e))
    copy = [e for e in rest if e in COPIED]
    link = [e for e in rest if e not in COPIED]
    return CopyPlan(unpack, copy, link)
