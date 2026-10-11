"""Reading and writing bundles, JSONL files and depth maps."""

import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from .models import bundle as b
from .models.common import F64


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    if not path.is_file():
        return []
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def iter_jsonl_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open() as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                yield n, line


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for item in items:
            f.write(item.model_dump_json(exclude_none=True) + "\n")
            n += 1
    return n


def write_json(path: Path, item: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(item.model_dump_json(indent=2, exclude_none=True) + "\n")


def read_depth(path: Path) -> F64:
    """16-bit PNG in millimetres -> float metres (0 = no data)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype != np.uint16:
        raise ValueError(f"{path}: depth must be 16-bit, got {img.dtype}")
    return img.astype(np.float64) / 1000.0


def write_depth(path: Path, depth_m: F64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), mm)


def read_confidence(path: Path) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.uint8)


@dataclass
class Bundle:
    dir: Path
    session: b.Session
    frames: list[b.Frame]
    stills: list[b.Still]

    @property
    def id(self) -> str:
        return self.session.session_id

    def path(self, rel: str) -> Path:
        return self.dir / rel

    def depth_m(self, frame: b.Frame, min_confidence: int = 0) -> F64 | None:
        if frame.depth is None:
            return None
        d = read_depth(self.path(frame.depth))
        if min_confidence > 0 and frame.confidence is not None:
            d[read_confidence(self.path(frame.confidence)) < min_confidence] = 0.0
        return d


def load_bundle(d: Path) -> Bundle:
    session = b.Session.model_validate_json((d / b.SESSION_FILE).read_text())
    return Bundle(
        dir=d,
        session=session,
        frames=read_jsonl(d / b.FRAMES_FILE, b.Frame),
        stills=read_jsonl(d / b.STILLS_FILE, b.Still),
    )


def find_bundles(raw: Path, skip: Iterable[str] = ()) -> list[Path]:
    """Session bundles in `raw`/<session>/ (data/<project>/raw/) or one folder deeper (the app's project folder).

    `skip` leaves out sessions by folder name.
    """
    found = [*raw.glob(f"*/{b.SESSION_FILE}"), *raw.glob(f"*/*/{b.SESSION_FILE}")]
    skipped = set(skip)
    return sorted(p.parent for p in found if p.parent.name not in skipped)


class VideoReader:
    """Sequential reader that returns requested frame indices (seeking HEVC is slow and inexact)."""

    def __init__(self, path: Path, hardware: bool = False) -> None:
        """`hardware`: decode with AVFoundation (macOS, the media engine: ~5x faster, next to no CPU) when it
        opens the video with FFmpeg's frame count; its colours differ by ~1/255 (another YUV conversion), so
        only where that doesn't matter (fused colours), never for features."""
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        if hardware and hasattr(cv2, "CAP_AVFOUNDATION"):
            hw = cv2.VideoCapture(str(path), cv2.CAP_AVFOUNDATION)
            if hw.isOpened() and int(hw.get(cv2.CAP_PROP_FRAME_COUNT)) == self.frame_count():
                self.cap.release()
                self.cap = hw
            else:
                hw.release()
        self.next_idx = 0

    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def size(self) -> tuple[int, int]:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self, idx: int) -> np.ndarray | None:
        """BGR image of frame `idx`; indices must be requested in increasing order."""
        if idx < self.next_idx:
            raise ValueError("VideoReader is sequential")
        while self.next_idx < idx:
            if not self.cap.grab():
                return None
            self.next_idx += 1
        ok, img = self.cap.read()
        self.next_idx += 1
        return img if ok else None

    def close(self) -> None:
        self.cap.release()


def decode_frames[T](
    video: Path, indices: Iterable[int], fn: Callable[[np.ndarray], T], hardware: bool = False
) -> dict[int, T]:
    """`fn(BGR image)` of each frame index, in one sequential pass over the video (frames past its end are
    left out). Run it for several videos in threads: OpenCV decodes without the GIL. `hardware`: see
    VideoReader."""
    reader = VideoReader(video, hardware)
    out: dict[int, T] = {}
    try:
        for idx in sorted(set(indices)):
            img = reader.read(idx)
            if img is None:
                break
            out[idx] = fn(img)
    finally:
        reader.close()
    return out


def decoders(videos: int) -> int:
    """Videos to decode at once: FFmpeg's HEVC decoder already keeps ~3 cores busy on one."""
    return max(1, min(videos, (os.cpu_count() or 4) // 3, 4))
