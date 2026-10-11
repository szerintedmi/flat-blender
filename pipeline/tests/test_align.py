"""Pose graph (scanmodel.align) on a synthetic loop: drift that grows with time, exact loop closures."""

from datetime import UTC, datetime

import numpy as np

from scanmodel import align
from scanmodel.models import bundle as b
from scanmodel.models.common import to_mat4
from scanmodel.models.derived import PosedImage

from . import synth


def _drift(t: float) -> np.ndarray:
    """What ARKit adds by time t: 1 mm/s sideways, 0.5 mm/s up, 0.02 deg/s of yaw."""
    a = np.radians(0.02 * t)
    m = np.eye(4)
    m[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    m[:3, 3] = [0.001 * t, 0.001 * t, 0.0005 * t]
    return m


def _loop(n: int = 90) -> tuple[list[np.ndarray], list[float]]:
    """Two laps of a 4 x 3 m rectangle at 1.3 m, looking along the walk, a keyframe every 2 s."""
    truth, times = [], []
    for i in range(n):
        u = (i / (n / 2)) % 1.0
        a = 2 * np.pi * u
        eye = np.array([2.0 + 2.0 * np.cos(a), 1.5 + 1.5 * np.sin(a), 1.3])
        ahead = np.array([2.0 + 2.0 * np.cos(a + 0.3), 1.5 + 1.5 * np.sin(a + 0.3), 1.3])
        truth.append(synth.look_at(eye, ahead))
        times.append(2.0 * i)
    return truth, times


def test_pose_graph_removes_drift_that_grows_with_time() -> None:
    truth, times = _loop()
    k = b.Intrinsics(fx=1450.0, fy=1450.0, cx=960.0, cy=720.0, width=1920, height=1440)
    kfs = []
    for i, (m, t) in enumerate(zip(truth, times, strict=True)):
        drifted = _drift(t) @ m
        im = PosedImage(
            id=f"s/{i}",
            session="s",
            kind="frame",
            file=f"video.mov#{i}",
            t=t,
            camera_to_world=to_mat4(drifted),
            intrinsics=k,
        )
        kfs.append(align.Keyframe("s", 0, t, drifted, im))
    # Loop closures where the second lap passes the first (exact relative poses, as matches would give).
    links = [
        align.Closure(i, j, np.linalg.inv(truth[i]) @ truth[j], 50, ("s", "s"))
        for i in range(len(truth))
        for j in range(i + 1, len(truth))
        if times[j] - times[i] >= align.MIN_DT and np.linalg.norm(truth[i][:3, 3] - truth[j][:3, 3]) < 0.5
    ]
    assert links
    solved = align.solve(kfs, links)
    before = max(np.linalg.norm(k.pose[:3, 3] - m[:3, 3]) for k, m in zip(kfs, truth, strict=True))
    after = max(np.linalg.norm(s[:3, 3] - m[:3, 3]) for s, m in zip(solved, truth, strict=True))
    assert before > 0.15, before
    assert after < 0.02, after
    # Frames between keyframes get the interpolated correction.
    correct = align.corrector(kfs, solved)
    mid = correct("s", 0, 101.0, _drift(101.0) @ truth[50])
    assert np.linalg.norm(mid[:3, 3] - truth[50][:3, 3]) < 0.03


def test_trajectories_follow_next_room_but_not_relocalisation() -> None:
    dev = b.Device(model="t", os_version="27", app_version="0")
    t0 = datetime(2026, 10, 6, tzinfo=UTC)

    def session(sid: str, kind: b.SessionType, world_map_in: str | None) -> b.Session:
        return b.Session(
            project_id="p", session_id=sid, type=kind, device=dev, started_at=t0, world_map_in=world_map_in
        )

    traj = align.trajectories(
        [
            ("hall", session("hall", b.SessionType.SCAN, None), 0.0, 100.0),
            ("kitchen", session("kitchen", b.SessionType.SCAN, "hall"), 105.0, 200.0),  # Next room
            ("lap", session("lap", b.SessionType.LAP, "kitchen"), 204.0, 260.0),  # closing lap
            ("close-ups", session("close-ups", b.SessionType.TEXTURE, "lap"), 400.0, 450.0),  # relocalised
        ]
    )
    assert traj == {"hall": 0, "kitchen": 0, "lap": 0, "close-ups": 1}


def test_trajectories_follow_continued_from_after_any_pause() -> None:
    """App 0.5+: Next room / the lap record on a Record tap, so a continued session may start minutes later;
    `continued_from` says so. Without it, the 30 s rule of older bundles still applies."""
    dev = b.Device(model="t", os_version="27", app_version="0")
    t0 = datetime(2026, 10, 6, tzinfo=UTC)

    def session(sid: str, kind: b.SessionType, wm: str | None, cont: str | None) -> b.Session:
        return b.Session(
            project_id="p", session_id=sid, type=kind, device=dev, started_at=t0, world_map_in=wm,
            continued_from=cont,
        )  # fmt: skip

    S, L = b.SessionType.SCAN, b.SessionType.LAP
    traj = align.trajectories(
        [
            ("hall", session("hall", S, None, None), 0.0, 100.0),
            ("kitchen", session("kitchen", S, "hall", "hall"), 190.0, 300.0),  # 90 s typing, then Record
            ("lap", session("lap", L, "kitchen", "kitchen"), 420.0, 480.0),
            ("bath", session("bath", S, "lap", None), 600.0, 700.0),  # old rule: 120 s is too long
            ("attic", session("attic", S, None, "hall"), 705.0, 800.0),  # continued another, not the previous
        ]
    )
    assert traj == {"hall": 0, "kitchen": 0, "lap": 0, "bath": 1, "attic": 2}


def test_floors_undo_vertical_drift_without_any_image_matches() -> None:
    truth, times = _loop()
    k = b.Intrinsics(fx=1450.0, fy=1450.0, cx=960.0, cy=720.0, width=1920, height=1440)
    kfs, cells = [], []
    for i, (m, t) in enumerate(zip(truth, times, strict=True)):
        drifted = _drift(t) @ m
        im = PosedImage(
            id=f"s/{i}", session="s", kind="frame", file=f"video.mov#{i}", t=t,
            camera_to_world=to_mat4(drifted), intrinsics=k,
        )  # fmt: skip
        kfs.append(align.Keyframe("s", 0, t, drifted, im))
        # The floor (true z = 0) under and around the camera, seen through the drifted pose.
        cx, cy = np.floor(m[:2, 3] / align.FLOOR_CELL).astype(int)
        cells.append(
            {
                (int(cx + a), int(cy + c)): float(drifted[2, 3] - m[2, 3])
                for a in (-1, 0, 1)
                for c in (-1, 0, 1)
            }
        )
    floor = align.floor_links(kfs, cells)
    assert floor
    solved = align.solve(kfs, [], floor=floor)
    before = max(abs(kk.pose[2, 3] - m[2, 3]) for kk, m in zip(kfs, truth, strict=True))
    after = max(abs(s[2, 3] - m[2, 3]) for s, m in zip(solved, truth, strict=True))
    assert before > 0.08, before
    assert after < 0.015, after
