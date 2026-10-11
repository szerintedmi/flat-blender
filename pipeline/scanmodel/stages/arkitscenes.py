"""ARKitScenes (github.com/apple/ARKitScenes) as benchmark ground truth: download, laser clouds, importer.

ARKitScenes: 2020 iPad Pro captures of single rooms with ARKit's own poses, LiDAR depth + confidence,
Faro laser scans of the same room ("visit") and hand-annotated 3D boxes. Licence: Apple's LICENSE in
that repo (non-commercial use, plus a commercial grant below 700 M monthly users; no redistribution of
the data here: it stays in data/external/, which is gitignored).

What we use from the `raw` split, per video (one room, one ARKit recording):
- `lowres_wide.traj`: ARKit's on-device camera poses at 10 Hz, one line per pose:
  t, axis-angle (3), translation (3) of world_to_camera. World: gravity-aligned, z up (the 3dod frame,
  also that of the boxes); camera: OpenCV axes.
- `lowres_depth/<video>_<t>.png` (uint16 mm, 256x192, 60 Hz), `confidence/` (0..2).
- `vga_wide/<video>_<t>.png` (640x480, 30 Hz) + `vga_wide_intrinsics/<video>_<t>.pincam`
  (w h fx fy cx cy): becomes our video.mov (SIFT for the pose graph, TSDF colour).
- `<video>_3dod_annotation.json`: oriented boxes (`obbAligned`, metres, same frame as the poses).
- Per visit: laser scans `<scan>.ply` (~1.9 GB each, 43 M points, scanner frame) and `<scan>_pose.txt`
  (scanner -> venue). Half of each scan is read (range requests) and voxel-thinned; never stored whole.

Command line (data/ is gitignored; `--dest` defaults to data/external/arkitscenes):
  uv run python -m scanmodel.stages.arkitscenes download 42447329 43828439 ...
  uv run python -m scanmodel.stages.arkitscenes import 42447329
"""

import argparse
import csv
import io
import json
import os
import shutil
import struct
import sys
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from http.client import HTTPResponse
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import open3d as o3d

from ..checks import depth_consistency
from ..config import default_root
from ..geom import ARKIT_TO_ZUP, GL_CV
from ..io import Bundle, write_json, write_jsonl
from ..models import bundle as b
from ..models.common import F64, mat, to_mat4

URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
SMALL = ("lowres_wide.traj", "{video}_3dod_annotation.json")
LASER_VOXEL = 0.01  # laser clouds are thinned to this (the TSDF voxel is 1 cm too)
LASER_RANGE = 12.0  # laser points further than this from the scanner are dropped (outdoors, far rooms)
MATCH_S = 0.005  # a depth / colour frame belongs to a pose this close in time

# Subset used by `scanmodel bench` (chosen 2026-10-07 from 3dod videos with laser scans; see
# docs/research/2026-10-07-threshold-audit.md): (video, visit, split, why).
SUBSET: tuple[tuple[str, str, str, str], ...] = (
    ("42447329", "422377", "Training", "ordinary room, longest sequence (234 s, 49 m walked)"),
    ("43828439", "437524", "Training", "kitchen, 10 cabinets; two ceiling levels (2.05 / 2.77 m)"),
    ("47204322", "471098", "Training", "dining room, 7 chairs; floor peaks 6 cm apart (step?)"),
    ("42899822", "435724", "Training", "bedroom, fireplace, 3.3 m ceiling, lower area in view"),
)


# --------------------------------------------------------------------------- formats


def read_traj(path: Path) -> list[tuple[float, F64]]:
    """(t, camera_to_world) per line: z-up world, OpenCV camera axes."""
    out: list[tuple[float, F64]] = []
    for line in path.read_text().splitlines():
        v = [float(x) for x in line.split()]
        if len(v) != 7:
            continue
        r, _ = cv2.Rodrigues(np.array(v[1:4]))
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = r, v[4:7]
        out.append((v[0], np.linalg.inv(w2c)))
    return out


def write_traj(path: Path, poses: list[tuple[float, F64]]) -> None:
    """Inverse of read_traj (used by the tests' synthetic captures)."""
    lines: list[str] = []
    for t, c2w in poses:
        w2c = np.linalg.inv(c2w)
        aa, _ = cv2.Rodrigues(w2c[:3, :3])
        lines.append(" ".join(f"{x:.8f}" for x in (t, *aa.ravel(), *w2c[:3, 3])))
    path.write_text("\n".join(lines) + "\n")


def read_pincam(path: Path) -> b.Intrinsics:
    w, h, fx, fy, cx, cy = np.loadtxt(path)
    return b.Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=int(w), height=int(h))


def _stamp(name: str) -> float:
    """`<video>_<t>.png` -> t."""
    return float(Path(name).stem.split("_")[-1])


@dataclass
class GtBox:
    label: str
    center: tuple[float, float, float]  # 3dod frame (z up, metres)
    size: tuple[float, float, float]  # along the box's own x, y and z (z vertical)
    yaw_deg: float


def read_boxes(path: Path) -> list[GtBox]:
    """`obbAligned` boxes. Corners are R^T @ (+-l/2, +-w/2, +-h/2) + centre, so R's rows are the box axes;
    boxes are upright (third row ~ z), only yaw is free."""
    out: list[GtBox] = []
    for d in json.loads(path.read_text()).get("data", []):
        obb = d["segments"]["obbAligned"]
        r = np.array(obb["normalizedAxes"], dtype=float).reshape(3, 3)
        x = r[0]
        if abs(r[2, 2]) < 0.99:  # not upright: not comparable with RoomPlan-style boxes
            continue
        c, s = obb["centroid"], obb["axesLengths"]
        out.append(
            GtBox(
                str(d["label"]),
                (float(c[0]), float(c[1]), float(c[2])),
                (float(s[0]), float(s[1]), float(s[2])),
                float(np.degrees(np.arctan2(x[1], x[0]))),
            )
        )
    return out


# --------------------------------------------------------------------------- download

# Downloads use HTTP range requests, so only what the benchmark reads crosses the network: the 10 Hz
# frames that have a pose (not the 30-60 Hz zips, up to 3 GB each) and every other 1 MB block of a
# laser scan (the scans are stored row by row, ~2 rows per MB: the gaps are < 1 cm at 3 m).
BLOCK = 1 << 20
THREADS = 8


def _get(url: str, first: int | None = None, last: int | None = None) -> HTTPResponse:
    req = urllib.request.Request(url)
    if first is not None:
        req.add_header("Range", f"bytes={first}-{'' if last is None else last}")
    return cast(HTTPResponse, urllib.request.urlopen(req, timeout=120))


def _read(url: str, first: int, last: int) -> bytes:
    """Bytes first..last (fewer at the end of the file), retried on network errors."""
    for attempt in range(4):
        try:
            with _get(url, first, last) as r:
                data = r.read()
                total = str(r.headers.get("Content-Range", "")).rpartition("/")[2]
            want = min(last, int(total) - 1) - first + 1 if total.isdigit() else last - first + 1
            if len(data) == want:
                return data
        except OSError:
            if attempt == 3:
                raise
    raise OSError(f"short read {url} [{first}, {last}]")


def _size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def _fetch(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with _get(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, BLOCK)
    tmp.rename(dst)


class RemoteFile(io.RawIOBase):
    """A read-only, seekable view of a URL (range requests), enough for zipfile's central directory."""

    def __init__(self, url: str) -> None:
        self.url, self.size, self.pos = url, _size(url), 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b: Any) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = _read(self.url, self.pos, self.pos + n - 1)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def zip_members(url: str) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(RemoteFile(url)) as zf:
        return zf.infolist()


def _member(url: str, zi: zipfile.ZipInfo) -> bytes:
    """One member's bytes, from its local header on (one range request, two if the extra field is long)."""
    head = 30 + len(zi.filename.encode()) + 256
    raw = _read(url, zi.header_offset, zi.header_offset + head + zi.compress_size - 1)
    n, m = struct.unpack("<HH", raw[26:30])
    start = 30 + n + m
    if start + zi.compress_size > len(raw):
        raw = _read(url, zi.header_offset, zi.header_offset + start + zi.compress_size - 1)
    data = raw[start : start + zi.compress_size]
    if zi.compress_type == zipfile.ZIP_STORED:
        return data
    if zi.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(data, -15)
    raise ValueError(f"{url}: {zi.filename} uses compression {zi.compress_type}")


def fetch_members(url: str, members: list[zipfile.ZipInfo], dst_dir: Path) -> None:
    """Write the given zip members under dst_dir (paths as in the zip, checked to stay inside)."""
    todo: list[tuple[zipfile.ZipInfo, Path]] = []
    for zi in members:
        rel = Path(zi.filename)
        if rel.is_absolute() or ".." in rel.parts:  # untrusted archive
            raise ValueError(f"unsafe path in {url}: {zi.filename}")
        if not (dst_dir / rel).exists():
            todo.append((zi, dst_dir / rel))

    def one(item: tuple[zipfile.ZipInfo, Path]) -> None:
        zi, out = item
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        tmp.write_bytes(_member(url, zi))
        tmp.rename(out)

    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(one, todo))


def wanted(names: list[str], times: list[float], tol: float) -> list[str]:
    """For each time, the name whose `<video>_<t>.<ext>` stamp is nearest (within tol)."""
    stamped = sorted((_stamp(n), n) for n in names if not n.endswith("/"))
    if not stamped:
        return []
    ts = np.array([t for t, _ in stamped])
    out: set[str] = set()
    for t in times:
        i = int(np.argmin(np.abs(ts - t)))
        if abs(ts[i] - t) <= tol:
            out.add(stamped[i][1])
    return sorted(out)


def ply_header(data: bytes) -> tuple[int, np.dtype, int]:
    """(offset of the vertex data, vertex dtype, vertex count) of a binary little-endian PLY."""
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii", "replace").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("only binary little-endian PLY is supported")
    types = {"double": "<f8", "float": "<f4", "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4"}
    fields: list[tuple[str, str]] = []
    count, in_vertex = 0, False
    for h in header:
        p = h.split()
        if p[:2] == ["element", "vertex"]:
            count, in_vertex = int(p[2]), True
        elif p[:1] == ["element"]:
            in_vertex = False
        elif p[:1] == ["property"] and in_vertex and p[1] != "list":
            fields.append((p[2], types[p[1]]))
    return end, np.dtype(fields), count


def _voxel(xyz: F64, voxel: float) -> F64:
    return np.asarray(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)).voxel_down_sample(voxel).points
    )


def thin_records(v: np.ndarray, origin: F64, max_range: float = LASER_RANGE) -> F64:
    """Finite points within `max_range` of the scanner at `origin`."""
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz - origin, axis=1) < max_range)]


def thin_scan(url: str, origin: F64, stride: int = 2, voxel: float = LASER_VOXEL) -> F64:
    """A remote laser scan: every `stride`-th block of BLOCK bytes, voxel-thinned, coordinates as stored
    (ARKitScenes stores them in venue coordinates already; `origin` is the scanner's position there)."""
    start, dt, count = ply_header(_read(url, 0, 65535))
    per = BLOCK // dt.itemsize
    blocks = list(range(0, -(-count // per), stride))

    def one(i: int) -> F64:
        n = min(per, count - i * per)
        a = start + i * per * dt.itemsize
        data = _read(url, a, a + n * dt.itemsize - 1)
        return _voxel(thin_records(np.frombuffer(data, dt), origin), voxel)

    kept: list[F64] = []
    with ThreadPoolExecutor(THREADS) as ex:
        for k in range(0, len(blocks), 64):
            part = [p for p in ex.map(one, blocks[k : k + 64]) if len(p)]
            if part:
                kept.append(_voxel(np.concatenate([*kept[-1:], *part]), voxel))
                kept = kept[-1:]
    return kept[-1] if kept else np.zeros((0, 3))


def read_scan_pose(text: str) -> F64:
    """`<scan>_pose.txt`: 4 lines of 4 numbers, the scanner->venue matrix stored transposed
    (rotation in lines 0-2, translation in line 3)."""
    m = np.array([[float(x) for x in line.replace(",", " ").split()] for line in text.strip().splitlines()])
    return m.T


def download(video: str, visit: str, split: str, dest: Path, laser: bool = True) -> Path:
    """Fetch what the benchmark needs for one video (and its visit's laser scans). Returns the video dir."""
    vdir = dest / "raw" / video
    vdir.mkdir(parents=True, exist_ok=True)
    base = f"{URL}/raw/{split}/{video}"
    for name in SMALL:
        _fetch(f"{base}/{name.format(video=video)}", vdir / name.format(video=video))
    times = [t for t, _ in read_traj(vdir / "lowres_wide.traj")]
    for z, tol in (("lowres_depth", MATCH_S), ("confidence", MATCH_S), ("vga_wide", 0.02)):
        members = zip_members(f"{base}/{z}.zip")
        keep = set(wanted([m.filename for m in members], times, tol))
        print(f"  {video}: {z}: {len(keep)} of {len(members)} files", flush=True)
        fetch_members(f"{base}/{z}.zip", [m for m in members if m.filename in keep], vdir)
    intr = f"{base}/vga_wide_intrinsics.zip"
    names = {Path(n).stem for n in os.listdir(vdir / "vga_wide")}
    fetch_members(intr, [m for m in zip_members(intr) if Path(m.filename).stem in names], vdir)
    if laser:
        for scan in laser_scan_ids(visit, dest):
            out = dest / "laser" / visit / f"{scan}.npz"
            if out.exists():
                continue
            url = f"{URL}/raw/laser_scanner_point_clouds/{visit}/{scan}"
            with _get(f"{url}_pose.txt") as r:
                pose_text = r.read().decode()
            print(
                f"  laser scan {visit}/{scan}: half the blocks, thinned to {LASER_VOXEL * 100:.0f} cm",
                flush=True,
            )
            pose = read_scan_pose(pose_text)
            pts = thin_scan(f"{url}.ply", pose[:3, 3])
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, points=pts, pose=pose)
    return vdir


def laser_scan_ids(visit: str, dest: Path) -> list[str]:
    mapping = dest / "meta" / "laser_scanner_point_clouds_mapping.csv"
    _fetch(f"{URL}/raw/laser_scanner_point_clouds/laser_scanner_point_clouds_mapping.csv", mapping)
    with mapping.open() as f:
        return sorted(r["laser_scanner_point_clouds_id"] for r in csv.DictReader(f) if r["visit_id"] == visit)


def _scan_points(path: Path) -> tuple[F64, F64]:
    """(points as stored, scanner->venue matrix)."""
    with np.load(path) as z:
        return z["points"].astype(np.float64), z["pose"].astype(np.float64)


def laser_cloud(visit_dir: Path) -> tuple[F64, F64]:
    """All scans of a visit in the venue frame, with normals facing their scanner.

    The released scans are already in venue coordinates (the pose's translation sits among the points).
    That is measured per scan rather than assumed: points far from the pose's translation but around the
    origin are in the scanner frame and get the pose applied, both readings of its rotation (R, R^T) tried,
    keeping the one under which the scans agree best."""
    scans = [_scan_points(p) for p in sorted(visit_dir.glob("*.npz"))]
    if not scans:
        raise FileNotFoundError(f"no laser scans in {visit_dir}")

    def place(transposed: bool) -> list[tuple[F64, F64]]:
        out: list[tuple[F64, F64]] = []
        for pts, m in scans:
            t = m[:3, 3]
            if np.median(np.linalg.norm(pts - t, axis=1)) < np.median(np.linalg.norm(pts, axis=1)):
                out.append((pts, t))  # venue frame already
                continue
            r = m[:3, :3].T if transposed else m[:3, :3]
            out.append((pts @ r.T + t, t))
        return out

    def agreement(placed: list[tuple[F64, F64]]) -> float:
        a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(placed[0][0])).voxel_down_sample(0.05)
        rest = np.concatenate([p for p, _ in placed[1:]])
        c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rest)).voxel_down_sample(0.05)
        d = np.asarray(a.compute_point_cloud_distance(c))
        return float(np.mean(d < 0.03))  # share of scan 0 that another scan confirms

    placed = max([place(False), place(True)], key=agreement) if len(scans) > 1 else place(False)
    pts_all: list[F64] = []
    nrm_all: list[F64] = []
    for pts, origin in placed:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        n = np.asarray(pc.normals)
        n[np.einsum("ij,ij->i", n, origin - pts) < 0] *= -1
        pts_all.append(pts)
        nrm_all.append(n)
    return np.concatenate(pts_all), np.concatenate(nrm_all)


# --------------------------------------------------------------------------- import


def _by_stamp(folder: Path) -> tuple[np.ndarray, list[Path]]:
    files = sorted(folder.glob("*.png")) if folder.is_dir() else []
    return np.array([_stamp(f.name) for f in files]), files


def _nearest(stamps: np.ndarray, t: float) -> int | None:
    if stamps.size == 0:
        return None
    i = int(np.argmin(np.abs(stamps - t)))
    return i if abs(stamps[i] - t) < MATCH_S else None


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def import_video(src: Path, raw_root: Path, project_id: str = "arkitscenes") -> Path:
    """One ARKitScenes video -> bundle v1 at raw_root/<video>/.

    One frame per pose in lowres_wide.traj (10 Hz) that has a depth map within MATCH_S; the colour frame
    is the nearest vga_wide image (written in order into video.mov). Poses are converted to ARKit's
    convention (y-up world, OpenGL camera), so `scanmodel ingest` turns them back into the 3dod frame exactly.
    """
    video = src.name
    dst = raw_root / video
    if (dst / b.SESSION_FILE).exists():
        return dst
    traj = read_traj(src / "lowres_wide.traj")
    d_t, d_files = _by_stamp(src / "lowres_depth")
    c_t, c_files = _by_stamp(src / "confidence")
    v_t, v_files = _by_stamp(src / "vga_wide")
    if not traj or d_t.size == 0 or v_t.size == 0:
        raise FileNotFoundError(f"{src}: needs lowres_wide.traj, lowres_depth/ and vga_wide/")
    zup_to_arkit = np.linalg.inv(ARKIT_TO_ZUP)
    frames: list[b.Frame] = []
    images: list[Path] = []
    for t, c2w in traj:
        i = _nearest(d_t, t)
        if i is None:
            continue
        j = _nearest(c_t, t)
        k = int(np.argmin(np.abs(v_t - t)))
        img = v_files[k]
        pincam = src / "vga_wide_intrinsics" / (img.stem + ".pincam")
        if abs(v_t[k] - t) > 0.02 or not pincam.is_file():  # colour from another moment misleads SIFT
            continue
        depth = f"{b.DEPTH_DIR}/{d_files[i].name}"
        _link(d_files[i], dst / depth)
        conf = None
        if j is not None:
            conf = f"{b.CONFIDENCE_DIR}/{c_files[j].name}"
            _link(c_files[j], dst / conf)
        frames.append(
            b.Frame(
                idx=len(frames),
                t=t,
                camera_to_world=to_mat4(zup_to_arkit @ c2w @ GL_CV),
                intrinsics=read_pincam(pincam),
                tracking=b.TrackingState.NORMAL,
                depth=depth,
                confidence=conf,
            )
        )
        images.append(img)
    if not frames:
        raise RuntimeError(f"{src}: no pose has a depth map and a colour frame")
    w, h = frames[0].intrinsics.width, frames[0].intrinsics.height
    fps = round(1 / float(np.median(np.diff([f.t for f in frames]))), 3)
    _write_video(dst / b.VIDEO_FILE, images, w, h, fps)
    dh, dw = cast(np.ndarray, cv2.imread(str(d_files[0]), cv2.IMREAD_UNCHANGED)).shape[:2]
    session = b.Session(
        project_id=project_id,
        session_id=video,
        type=b.SessionType.SCAN,
        device=b.Device(
            model="iPad Pro 2020", os_version="unknown", app_name="ARKitScenes", app_version="v1"
        ),
        video=b.VideoFormat(width=w, height=h, fps=fps, codec="h264"),
        depth=b.DepthFormat(width=dw, height=dh),
        started_at=datetime.fromtimestamp(0, UTC),
        room_name=f"arkitscenes {video}",
        importer="arkitscenes",
        notes=f"imported from {src}",
    )
    bundle = Bundle(dir=dst, session=session, frames=frames, stills=[])
    write_json(dst / b.SESSION_FILE, session)
    write_jsonl(dst / b.FRAMES_FILE, frames)
    (dst / b.STILLS_FILE).touch()
    # A wrong axis convention shows as depth that disagrees between nearby frames (cm, not mm).
    check = depth_consistency(bundle, lambda f: ARKIT_TO_ZUP @ mat(f.camera_to_world) @ GL_CV)
    session.notes += f"; depth consistency {check if check is None else round(check, 4)} m"
    write_json(dst / b.SESSION_FILE, session)
    return dst


def _write_video(path: Path, images: list[Path], w: int, h: int, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writers = (cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*c), fps, (w, h)) for c in ("avc1", "mp4v"))
    out = next((v for v in writers if v.isOpened()), None)
    if out is None:
        raise RuntimeError("no usable video codec in OpenCV")
    grey = np.full((h, w, 3), 128, np.uint8)
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            img = grey
        elif img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        out.write(img)
    out.release()


# --------------------------------------------------------------------------- cli


def default_dest(root: Path | None = None) -> Path:
    """data/external/arkitscenes of the repo (default: `default_root()`); shared by all projects."""
    return (root or default_root()) / "data" / "external" / "arkitscenes"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scanmodel.stages.arkitscenes", description=(__doc__ or "").split("\n")[0]
    )
    ap.add_argument("action", choices=["download", "import"])
    ap.add_argument("videos", nargs="*", help="video ids (default: the benchmark subset)")
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("--no-laser", action="store_true")
    a = ap.parse_args(argv)
    dest = a.dest or default_dest()
    known = {v: (visit, split) for v, visit, split, _ in SUBSET}
    for v in a.videos or list(known):
        if a.action == "download":
            if v not in known:
                print(f"{v}: not in SUBSET; add (video, visit, split) there first", file=sys.stderr)
                return 2
            download(v, *known[v], dest, laser=not a.no_laser)
        else:
            print(import_video(dest / "raw" / v, dest / "bundles"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
