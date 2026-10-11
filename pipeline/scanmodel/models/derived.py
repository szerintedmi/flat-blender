"""Files the pipeline writes for other tools (Blender, the viewer, later stages)."""

from typing import Literal

from .bundle import Intrinsics, StillSource, TrackingState
from .common import Mat4, Strict


class PosedImage(Strict):
    """An image with a pose in the scene frame (z up).

    `camera_to_world` uses OpenCV camera axes (+x right, +y down, +z forward), the pipeline's internal
    convention. Blender and nerfstudio want OpenGL axes: multiply on the right by diag(1, -1, -1, 1).
    """

    id: str  # "<session>/<frame idx>" or "<session>/s<still id>"
    session: str
    kind: Literal["frame", "still"]
    file: str  # path relative to the session dir (video frames: "video.mov#<idx>")
    t: float | None = None
    camera_to_world: Mat4
    intrinsics: Intrinsics
    tracking: TrackingState | None = None
    source: StillSource | None = None
    depth: str | None = None  # relative to the session dir
    confidence: str | None = None


class DroppedFrames(Strict):
    """Frames ingest left out mid-session, and the stills shot meanwhile (`t0` <= t <= `t1`)."""

    reason: Literal["excursion", "override"]  # checks.excursions, or overrides.json `drop_frames`
    idx0: int  # first and last frame idx left out (inclusive)
    idx1: int
    t0: float
    t1: float
    frames: int
    stills: list[int] = []  # still ids
    step_m: float | None = None  # excursion: the jump out
    step_deg: float | None = None


class IngestReport(Strict):
    """`data/<project>/derived/ingest/<session>/report.json`."""

    session: str
    raw_dir: str
    ok: bool
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    tracking_normal_fraction: float | None = None
    depth_consistency_m: float | None = None  # median depth disagreement between nearby frames
    dropped: list[DroppedFrames] = []  # left out of frames.jsonl / stills.jsonl (after the unsettled start)


class SessionPoses(Strict):
    session: str
    raw_dir: str  # the bundle folder, repo-relative (absolute if outside the repo, or in older files)
    session_to_scene: Mat4  # ARKit world (y up) -> scene frame (z up)
    # a session refine's coarse round cut in two (`scanmodel.refine` trajectory cuts): session_to_scene of its other
    # parts; structure places each RoomPlan element with whichever puts it on the fused points
    session_to_scene_parts: list[Mat4] = []
    icp_check_rms_m: float | None = None  # mesh overlap check against the first session, if run


class Poses(Strict):
    """`data/<project>/derived/poses/poses.json`. Frames and stills are listed in frames.jsonl / stills.jsonl next to it."""

    version: Literal[1] = 1
    origin_door: str | None = None
    yaw_deg: float  # rotation applied about z after gravity alignment
    floor_z: float  # floor height that was moved to z = 0 (gravity-aligned axes)
    sessions: list[SessionPoses]
