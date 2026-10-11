"""One-way importer: a stock Stray Scanner capture -> bundle v1 (`scanmodel ingest --from stray <dir>`).

Stray layout: rgb.mp4, depth/NNNNNN.png (uint16 mm, 256x192), confidence/NNNNNN.png,
odometry.csv (timestamp, frame, x, y, z, qx, qy, qz, qw), camera_matrix.csv (3x3, video resolution).
Stray has no stills, tracking state or exposure per frame.

Stray's docs don't pin the camera-axis convention of the odometry quaternion, so we measure it:
both candidates are scored by reprojecting depth between nearby frames, and the one whose depths
agree is kept (and recorded in session.json `notes`).
"""

import csv
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from ..checks import depth_consistency
from ..geom import ARKIT_TO_ZUP, GL_CV, quat_to_matrix
from ..io import Bundle, write_json, write_jsonl
from ..models import bundle as b
from ..models.common import F64, mat, to_mat4


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _read_odometry(path: Path) -> list[tuple[float, int, F64]]:
    rows: list[tuple[float, int, F64]] = []
    with path.open() as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        col = {name: i for i, name in enumerate(header)}
        for r in reader:
            if not r:
                continue
            v = [float(x) for x in r]
            m = np.eye(4)
            m[:3, :3] = quat_to_matrix(v[col["qx"]], v[col["qy"]], v[col["qz"]], v[col["qw"]])
            m[:3, 3] = [v[col["x"]], v[col["y"]], v[col["z"]]]
            rows.append((v[col["timestamp"]], int(v[col["frame"]]), m))
    return rows


def import_stray(src: Path, raw_root: Path, project_id: str, session_id: str | None = None) -> Path:
    if not (src / "odometry.csv").is_file():
        raise FileNotFoundError(f"{src} is not a Stray capture (no odometry.csv)")
    session_id = session_id or f"stray-{src.name}"
    dst = raw_root / session_id
    if (dst / b.SESSION_FILE).exists():
        raise FileExistsError(f"{dst} already exists")

    k = np.loadtxt(src / "camera_matrix.csv", delimiter=",").reshape(3, 3)
    video = src / "rgb.mp4"
    vw, vh, fps, fourcc = 0, 0, 0.0, ""
    if video.is_file():
        cap = cv2.VideoCapture(str(video))
        vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fourcc = (
            (int(cap.get(cv2.CAP_PROP_FOURCC)) & 0xFFFFFFFF).to_bytes(4, "little").decode(errors="replace")
        )
        cap.release()
    if vw == 0:  # no decodable video: fall back to the resolution Stray records at
        vw, vh = 1920, 1440
    fps = fps or 60.0
    intr = b.Intrinsics(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2], width=vw, height=vh)

    frames: list[b.Frame] = []
    for t, idx, m in _read_odometry(src / "odometry.csv"):
        depth = src / "depth" / f"{idx:06d}.png"
        conf = src / "confidence" / f"{idx:06d}.png"
        if depth.is_file():
            _link_or_copy(depth, dst / b.DEPTH_DIR / depth.name)
        if conf.is_file():
            _link_or_copy(conf, dst / b.CONFIDENCE_DIR / conf.name)
        frames.append(
            b.Frame(
                idx=idx,
                t=t,
                camera_to_world=to_mat4(m),
                intrinsics=intr,
                tracking=b.TrackingState.NORMAL,
                depth=f"{b.DEPTH_DIR}/{depth.name}" if depth.is_file() else None,
                confidence=f"{b.CONFIDENCE_DIR}/{conf.name}" if conf.is_file() else None,
            )
        )
    if video.is_file():
        _link_or_copy(video, dst / b.VIDEO_FILE)

    first_depth = next((f.depth for f in frames if f.depth), None)
    img = cv2.imread(str(dst / first_depth), cv2.IMREAD_UNCHANGED) if first_depth else None
    dh, dw = img.shape[:2] if img is not None else (192, 256)
    session = b.Session(
        project_id=project_id,
        session_id=session_id,
        type=b.SessionType.SCAN,
        device=b.Device(
            model="unknown", os_version="unknown", app_name="Stray Scanner", app_version="unknown"
        ),
        video=b.VideoFormat(
            width=vw, height=vh, fps=fps, codec="hevc" if fourcc.lower() in {"hvc1", "hev1"} else "h264"
        ),
        depth=b.DepthFormat(width=dw, height=dh),
        started_at=datetime.fromtimestamp(src.stat().st_mtime, UTC),
        importer="stray",
        notes=f"imported from {src}",
    )
    bundle = Bundle(dir=dst, session=session, frames=frames, stills=[])

    # Decide the camera convention by measurement. Our format stores ARKit (OpenGL) camera axes.
    gl = depth_consistency(bundle, lambda f: ARKIT_TO_ZUP @ mat(f.camera_to_world) @ GL_CV)
    cv = depth_consistency(bundle, lambda f: ARKIT_TO_ZUP @ mat(f.camera_to_world))
    if gl is not None and cv is not None and cv < gl:
        frames = [
            f.model_copy(update={"camera_to_world": to_mat4(mat(f.camera_to_world) @ GL_CV)}) for f in frames
        ]
        session.notes += (
            f"; odometry used OpenCV camera axes (consistency cv={cv:.4f} m vs gl={gl:.4f} m), converted"
        )
    elif gl is not None:
        session.notes += f"; odometry uses ARKit camera axes (consistency gl={gl:.4f} m vs cv={cv} m)"

    write_json(dst / b.SESSION_FILE, session)
    write_jsonl(dst / b.FRAMES_FILE, frames)
    (dst / b.STILLS_FILE).touch()
    return dst
