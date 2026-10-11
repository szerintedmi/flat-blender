"""Capture bundle format v1: the contract between FlatBlender (capture/) and the pipeline.

The app writes values exactly as ARKit reports them; `scanmodel ingest` converts them to pipeline axes.

Conventions
- World frame: ARKit world, metres, +y up (against gravity), right-handed.
- `camera_to_world`: `ARCamera.transform` as a row-major 4x4 nested list (`m[row][col]`, translation in
  `m[0..2][3]`). Camera axes are ARKit's: +x right, +y up, +z towards the viewer (the camera looks
  down -z), in the sensor's native landscape orientation (home button / USB-C on the right).
- Intrinsics: pixels, for the image they are stored with. Continuous pixel coordinates: pixel (0, 0)
  spans [0, 1) x [0, 1), so its centre is (0.5, 0.5); scaling an image scales fx, fy, cx, cy. Stream frames: the video resolution.
  Stills: the still resolution. Depth maps share the stream frame's orientation and field of view,
  so their intrinsics are the frame's scaled to the depth size.
- Timestamps `t`: seconds, `ARFrame.timestamp` (host uptime clock, shared by frames and stills).
- Depth PNGs: 16-bit greyscale, millimetres, 0 = no data. Confidence PNGs: 8-bit, 0/1/2
  (`ARConfidenceLevel` low/medium/high).

Layout of `<session>/`: see the `*_FILE` / `*_DIR` constants below.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from .common import Mat4, Strict, Tolerant, Vec3

FORMAT = "flat-bundle"
VERSION = 1

SESSION_FILE = "session.json"
FRAMES_FILE = "frames.jsonl"
STILLS_FILE = "stills.jsonl"
VIDEO_FILE = "video.mov"
DEPTH_DIR = "depth"
CONFIDENCE_DIR = "confidence"
STILLS_DIR = "stills"
CALIBRATION_FILE = "calibration.json"
MESH_FILE = "mesh.ply"
ROOM_JSON_FILE = "room.json"
ROOM_USDZ_FILE = "room.usdz"
WORLDMAP_FILE = "worldmap.arworldmap"
ANCHORS_FILE = "anchors.jsonl"


class SessionType(StrEnum):
    SCAN = "scan"  # stream + auto stills + RoomPlan
    LAP = "lap"  # closing lap: stream + stills, no RoomPlan; continues the last room's ARSession
    TEXTURE = "texture"  # relocalise, then tap-shoot close-ups only
    ULTRAWIDE = "ultrawide"  # AVFoundation 10 MP, no ARKit, no poses


class TrackingState(StrEnum):
    NORMAL = "normal"
    LIMITED_INITIALIZING = "limited_initializing"
    LIMITED_EXCESSIVE_MOTION = "limited_excessive_motion"
    LIMITED_INSUFFICIENT_FEATURES = "limited_insufficient_features"
    LIMITED_RELOCALIZING = "limited_relocalizing"
    NOT_AVAILABLE = "not_available"


class StillSource(StrEnum):
    TAP = "tap"
    AUTO = "auto"
    TEXTURE = "texture"
    ULTRAWIDE = "ultrawide"
    VIDEO = "video"  # extracted from video.mov (used by importers that have no stills)


class Intrinsics(Strict):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def scaled(self, width: int, height: int) -> Intrinsics:
        sx, sy = width / self.width, height / self.height
        # Pixel-centre convention: scale about the image corner, which is what ARKit uses.
        return Intrinsics(
            fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy, width=width, height=height
        )


class Exposure(Strict):
    duration_s: float | None = None  # ARCamera.exposureDuration
    offset_ev: float | None = None  # ARCamera.exposureOffset
    iso: float | None = None  # AVCaptureDevice.iso at capture time


class WhiteBalanceGains(Strict):
    red: float
    green: float
    blue: float


class Device(Tolerant):
    model: str  # hw.machine, e.g. "iPad14,3"
    os_version: str
    app_name: str = "FlatBlender"
    app_version: str


class VideoFormat(Strict):
    width: int
    height: int
    fps: float
    codec: Literal["hevc", "h264"] = "hevc"


class DepthFormat(Strict):
    width: int = 256
    height: int = 192
    unit: Literal["mm"] = "mm"
    smoothed: bool = False  # true if from smoothedSceneDepth instead of sceneDepth


class LockedCamera(Strict):
    """Camera settings locked after metering, so colour and brightness match across frames."""

    locked: bool
    exposure_duration_s: float | None = None
    iso: float | None = None
    lens_position: float | None = None  # AVCaptureDevice.lensPosition (focus)
    white_balance: WhiteBalanceGains | None = None
    metered: Exposure | None = None  # what auto exposure chose before the lock (app 0.3+)
    # Stops darker than metered, when the ISO ceiling at the longest allowed shutter couldn't match it
    underexposure_ev: float | None = None
    # Shutter policy (Settings): sharp = 1/240 s cap; flicker_50hz / flicker_60hz = fixed 1/100 / 1/120 s
    shutter_mode: Literal["sharp", "flicker_50hz", "flicker_60hz"] | None = None
    # Mains flicker the metered shutter points to (within ~1 % of k/100 s: 50hz, k/120 s: 60hz), app 0.4+
    flicker_suspected: Literal["50hz", "60hz"] | None = None
    # The "LED flicker likely" prompt at Lock (sharp mode only), app 0.4+; None = not shown
    flicker_prompt: Literal["shown", "accepted", "declined"] | None = None


class Sweep(Strict):
    """A guided doorway sweep: both faces of the wall around a door within a few seconds."""

    t0: float  # ARFrame.timestamp
    t1: float


class Session(Tolerant):
    """`session.json`."""

    format: Literal["flat-bundle"] = FORMAT
    version: Literal[1] = VERSION
    project_id: str
    session_id: str
    type: SessionType
    device: Device
    video: VideoFormat | None = None  # None for ultrawide sessions
    depth: DepthFormat | None = None  # None if the session has no LiDAR depth
    camera: LockedCamera | None = None
    world_map_in: str | None = None  # session whose ARWorldMap was loaded, or whose ARSession this continued
    world_map_out: str | None = None  # id of the world map saved by this session (worldmap.arworldmap)
    # The session whose ARSession this one continued (Next room / closing lap, app 0.5+: recording resumes
    # on a Record tap, so after any pause). Omitted when not continued and by older apps.
    continued_from: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    room_name: str | None = None  # what the user typed, e.g. "hall"
    importer: str | None = None  # set when converted from another format, e.g. "stray"
    sweeps: list[Sweep] = []
    notes: str = ""


class Frame(Strict):
    """One line of `frames.jsonl`. `idx` is the frame's index in video.mov."""

    idx: int = Field(ge=0)
    t: float
    camera_to_world: Mat4
    intrinsics: Intrinsics
    tracking: TrackingState
    exposure: Exposure | None = None
    depth: str | None = None  # "depth/000123.png", relative to the session dir
    confidence: str | None = None  # "confidence/000123.png"


class StillMotion(Strict):
    """How the camera moved when a still was taken (app 0.3+): blur predicted from the motion."""

    angular_speed_deg_s: float
    speed_m_s: float
    blur_px: float  # predicted motion blur at the still's resolution
    # Written by app < 0.4 only: stream frame sharpness (long side 640 px) near the still, and its median
    # over the last ~1.5 s (~10 s before app 0.3.0 (7)). Kept optional so old bundles validate (extra="forbid").
    frame_sharpness: float | None = None
    median_sharpness: float | None = None


class Still(Strict):
    """One line of `stills.jsonl`."""

    id: int = Field(ge=0)
    file: str  # "stills/000001.jpg"
    t: float | None = None
    source: StillSource
    intrinsics: Intrinsics | None = None  # None for ultrawide stills (solved by `scanmodel register`)
    camera_to_world: Mat4 | None = None  # None for ultrawide stills
    tracking: TrackingState | None = None
    exposure: Exposure | None = None
    white_balance: WhiteBalanceGains | None = None
    sharpness: float | None = None  # written by app < 0.4: variance of Laplacian on a downscaled luma image
    nearest_frame: int | None = None  # idx of the closest stream frame in time
    # Unit gravity vector in camera axes. ARKit sessions: R^T (0, -1, 0) from the pose; ultrawide: CoreMotion.
    gravity: Vec3 | None = None
    motion: StillMotion | None = None
    flagged_blurry: bool | None = None  # written by app < 0.4: a tapped still taken while moving


class AnchorEvent(Strict):
    """One line of `anchors.jsonl`: an ARAnchor we added (at each still and every 1 s) and ARKit's later
    updates to it, so ARKit's own loop-closure corrections can be read back."""

    t: float  # latest ARFrame.timestamp when observed
    id: str  # ARAnchor identifier (UUID)
    name: str  # "<session_id>/still/<id>" or "<session_id>/tick/<n>"
    event: Literal["add", "update", "final"]
    transform: Mat4  # ARKit world, row-major like camera_to_world


class Calibration(Strict):
    """`calibration.json`, from AVCameraCalibrationData of the wide camera."""

    reference_width: int
    reference_height: int
    intrinsic_matrix: list[list[float]] = Field(
        min_length=3, max_length=3
    )  # row-major 3x3, at reference size
    lens_distortion_center: tuple[float, float]  # pixels, at reference size
    lens_distortion_lookup_table: list[float]
    inverse_lens_distortion_lookup_table: list[float]
    pixel_size_mm: float | None = None
