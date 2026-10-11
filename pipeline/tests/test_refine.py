"""Pose refinement (scanmodel.refine) on synthetic scenes: known poses, a second pass offset by a few cm."""

import itertools
from pathlib import Path

import numpy as np
import pytest

from scanmodel import refine as rf
from scanmodel.models import bundle as b
from scanmodel.models.common import to_mat4
from scanmodel.models.derived import PosedImage

from . import synth

K = b.Intrinsics(fx=1524.0, fy=1524.0, cx=957.0, cy=721.0, width=1920, height=1440)
KS = K.scaled(rf.WIDTH, rf.HEIGHT)


def _offset(dx: float, dy: float, dz: float, yaw_deg: float = 0.0) -> np.ndarray:
    a = np.radians(yaw_deg)
    m = np.eye(4)
    m[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    m[:3, 3] = [dx, dy, dz]
    return m


def _wall_points(rng: np.random.Generator, n: int = 600) -> np.ndarray:
    """Points on two walls of a corner (y = 3 and x = 2.5) and a sofa-like box in front, so depth varies."""
    a = np.column_stack([rng.uniform(-2.0, 2.5, n // 2), np.full(n // 2, 3.0), rng.uniform(0.2, 2.3, n // 2)])
    c = np.column_stack([np.full(n // 4, 2.5), rng.uniform(0.5, 3.0, n // 4), rng.uniform(0.2, 2.3, n // 4)])
    s = np.column_stack(
        [rng.uniform(-1.0, 1.0, n // 4), rng.uniform(1.8, 2.2, n // 4), rng.uniform(0.3, 0.8, n // 4)]
    )
    return np.concatenate([a, c, s])


def _pass(y: float, t0: float, n: int = 16) -> tuple[list[np.ndarray], list[float]]:
    """A slow walk along x at distance (3 - y) from the wall, looking at it and slightly to the right."""
    poses, times = [], []
    for i in range(n):
        x = -1.2 + 2.4 * i / (n - 1)
        eye = np.array([x, y, 1.3 + 0.05 * np.sin(i)])
        poses.append(synth.look_at(eye, np.array([x + 0.6, 3.0, 1.2])))
        times.append(t0 + 0.5 * i)
    return poses, times


def _view(pose: np.ndarray, t: float, traj: int, i: int, session: str = "s") -> rf.View:
    im = PosedImage(
        id=f"{session}/{i}", session=session, kind="frame", file=f"video.mov#{i}", t=t,
        camera_to_world=to_mat4(pose), intrinsics=K, depth=f"depth/{i:06d}.png",
    )  # fmt: skip
    return rf.View(im, traj, t, pose, KS)


def _observe(
    pose: np.ndarray, pts: np.ndarray, rng: np.random.Generator, px: float = 0.3, depth_noise: float = 0.005
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(point indices, uv at 960 px, LiDAR depth) of the points this camera sees."""
    w2c = np.linalg.inv(pose)
    q = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[:, :2] / z[:, None] * [KS.fx, KS.fy] + [KS.cx, KS.cy]
    ok = (z > 0.3) & (uv[:, 0] > 5) & (uv[:, 0] < rf.WIDTH - 5) & (uv[:, 1] > 5) & (uv[:, 1] < rf.HEIGHT - 5)
    idx = np.flatnonzero(ok)
    return idx, uv[idx] + rng.normal(0, px, (len(idx), 2)), z[idx] + rng.normal(0, depth_noise, len(idx))


def test_bundle_adjustment_brings_an_offset_pass_back() -> None:
    """Pass B (a separate trajectory, 30 s later) is 3 cm / 0.3 deg off. Tracks across both passes, LiDAR
    depth and each pass's own relative motion must make the two passes agree to a few mm."""
    rng = np.random.default_rng(0)
    pts = _wall_points(rng)
    ta, tt_a = _pass(0.6, 0.0)
    tb, tt_b = _pass(0.9, 30.0)
    truth = [*ta, *tb]
    err = _offset(0.03, -0.01, 0.015, 0.3)
    start = [*ta, *[err @ m for m in tb]]
    views = [
        _view(p, t, 0 if i < len(ta) else 1, i)
        for i, (p, t) in enumerate(zip(start, [*tt_a, *tt_b], strict=True))
    ]
    view, track, uv, depth = [], [], [], []
    for i, m in enumerate(truth):
        idx, u, z = _observe(m, pts, rng)
        view += [i] * len(idx)
        track += idx.tolist()
        uv.append(u)
        depth.append(z)
    obs = rf.Observations(
        np.array(view),
        np.array(track),
        np.concatenate(uv),
        np.concatenate(depth),
        np.full(len(view), 2, np.uint8),
    )
    n_tracks = len(pts)
    odo = [
        rf.Odometry(i, i + 1, np.linalg.inv(truth[i]) @ truth[i + 1], 0.06, 0.0015)
        for i in range(len(truth) - 1)
        if views[i].trajectory == views[i + 1].trajectory
    ]
    points, keep, obs_ok = rf.init_landmarks(views, obs, n_tracks, max_spread=0.10)
    use = obs_ok & keep[obs.track]
    cross = rf.cross_tracks(views, obs, n_tracks)
    assert cross.sum() > 100
    before = rf.residual_stats(views, start, obs, use, cross)
    out = rf.adjust(views, obs, points, use, odo)
    after = rf.residual_stats(views, out.poses, obs, use, cross)
    # B relative to A, as it truly is
    rel_err = [
        np.linalg.norm(
            (np.linalg.inv(out.poses[0]) @ out.poses[i])[:3, 3] - (np.linalg.inv(truth[0]) @ truth[i])[:3, 3]
        )
        for i in range(len(truth))
    ]
    assert max(rel_err) < 0.004, rel_err
    assert after["cross_pass"]["all"]["median"] < 0.6  # type: ignore[index]
    assert before["cross_pass"]["all"]["median"] > 3 * after["cross_pass"]["all"]["median"]  # type: ignore[index]
    # Gravity is kept and the global frame is not dragged: the mean shift is about half the offset.
    for p, t in zip(out.poses, truth, strict=True):
        assert abs(p[2, 2] - t[2, 2]) < 2e-3
    shift = np.mean([p[:3, 3] - s[:3, 3] for p, s in zip(out.poses, start, strict=True)], axis=0)
    assert np.linalg.norm(shift) < 0.025


def test_guided_matching_rejects_a_repeated_pattern() -> None:
    """Two views of the same wall, the second 3 cm off; every feature has a twin 0.4 m to the side with the
    same descriptor (a repeated pattern). Guided matching finds the true correspondences, not the twins."""
    rng = np.random.default_rng(1)
    base = np.column_stack([rng.uniform(-0.8, 0.8, 200), np.full(200, 3.0), rng.uniform(0.6, 2.0, 200)])
    twins = base + np.array([0.4, 0.0, 0.0])
    pts = np.concatenate([base, twins])
    desc_base = rng.integers(0, 120, (200, 128))
    desc = np.concatenate([desc_base, desc_base]).astype(np.uint8)  # twins look the same
    pa = synth.look_at(np.array([0.0, 1.2, 1.3]), np.array([0.1, 3.0, 1.3]))
    pb_true = synth.look_at(np.array([0.5, 1.0, 1.3]), np.array([0.3, 3.0, 1.3]))
    pb = _offset(0.03, 0.0, -0.01) @ pb_true
    va, vb = _view(pa, 0.0, 0, 0), _view(pb, 40.0, 0, 1)
    feats, uvs = [], []
    for m in (pa, pb_true):
        idx, uv, z = _observe(m, pts, rng, px=0.3)
        noisy = np.clip(desc[idx].astype(int) + rng.integers(-3, 4, (len(idx), 128)), 0, 255).astype(np.uint8)
        feats.append(
            (
                idx,
                rf.Features(
                    uv.astype(np.float32), noisy, z.astype(np.float32), np.full(len(idx), 2, np.uint8)
                ),
            )
        )
        uvs.append(uv)
    (ia_pts, fa), (ib_pts, fb) = feats
    m = rf.match_pair(va, vb, fa, fb, uvs[0], uvs[1], rf._rootsift(fa.desc), rf._rootsift(fb.desc))  # pyright: ignore[reportPrivateUsage]
    ia, ib = m.ia, m.ib
    assert m.candidates >= rf.MIN_INLIERS
    assert len(ia) > 50
    correct = ia_pts[ia] == ib_pts[ib]
    assert correct.mean() > 0.98, correct.mean()


def test_tracks_join_matches_and_drop_conflicts() -> None:
    def feats(n: int) -> rf.Features:
        return rf.Features(
            np.zeros((n, 2), np.float32),
            np.zeros((n, 128), np.uint8),
            np.zeros(n, np.float32),
            np.zeros(n, np.uint8),
        )

    fs = [feats(5), feats(5), feats(5)]
    m = [
        rf.PairMatches(0, 1, "seq", np.array([0, 1, 2]), np.array([0, 1, 3]), 3),
        rf.PairMatches(1, 2, "seq", np.array([0, 1, 3]), np.array([4, 1, 2]), 3),
        rf.PairMatches(0, 2, "cross", np.array([0, 2]), np.array([4, 0]), 2),  # 2 -> 0 in view 2 conflicts
    ]
    t = rf.build_tracks(fs, m)
    tracks = {}
    for v, k, tr in zip(t.view, t.kp, t.track, strict=True):
        tracks.setdefault(int(tr), []).append((int(v), int(k)))
    got = sorted(sorted(x) for x in tracks.values())
    # track of (0,0): (0,0),(1,0),(2,4); track of (0,1): (0,1),(1,1),(2,1);
    # (0,2),(1,3),(2,2) + (2,0) from the cross match -> two keypoints in view 2 -> dropped
    assert got == [[(0, 0), (1, 0), (2, 4)], [(0, 1), (1, 1), (2, 1)]]


def test_pairs_link_passes_and_neighbours() -> None:
    ta, tt_a = _pass(0.6, 0.0, n=8)
    tb, tt_b = _pass(0.9, 30.0, n=8)
    views = [_view(p, t, 0, i) for i, (p, t) in enumerate(zip([*ta, *tb], [*tt_a, *tt_b], strict=True))]
    rng = np.random.default_rng(2)
    pts = _wall_points(rng, 2000)
    fs = []
    for v in views:
        idx, uv, z = _observe(v.pose, pts, rng)
        fs.append(
            rf.Features(
                uv.astype(np.float32),
                np.zeros((len(idx), 128), np.uint8),
                z.astype(np.float32),
                np.full(len(idx), 2, np.uint8),
            )
        )
    got = rf.pairs(views, rf.view_voxels(views, fs))
    seq = {(i, j) for i, j, k in got if k == "seq"}
    cross = [(i, j) for i, j, k in got if k == "cross"]
    assert (0, 1) in seq and (0, 3) in seq and (7, 8) not in seq  # 7 -> 8 is 23 s apart
    assert cross and all(i < 8 <= j for i, j in cross)
    assert {i for i, _ in cross} == set(range(8))  # every view of pass A is linked to pass B


def test_rolling_shutter_correction_moves_rows_by_their_time() -> None:
    v = _view(np.eye(4), 0.0, 0, 0)
    v.omega = np.array([0.0, np.radians(30.0), 0.0])  # turning about the camera's y (down) axis
    uv = np.array([[480.0, 0.0], [480.0, 360.0], [480.0, 720.0]])
    out = rf.rs_corrected(uv, v, 0.0147)
    shift = out[:, 0] - uv[:, 0]
    assert abs(shift[1]) < 1e-9
    assert np.isclose(shift[0], -shift[2], atol=1e-3)
    assert abs(abs(shift[0]) - KS.fx * np.radians(30.0) * 0.0147 / 2) < 0.05


def test_turning_frames_get_noisier_keypoints() -> None:
    still = rf.skew_sigma(KS.fx, np.zeros(3))
    turning = rf.skew_sigma(KS.fx, np.array([0.0, np.radians(60.0), 0.0]))
    corrected = rf.skew_sigma(KS.fx, np.array([0.0, np.radians(60.0), 0.0]), readout_s=rf.READOUT_S)
    assert still == rf.PIXEL_SIGMA
    assert 5.0 < turning < 7.0  # ~6 px of shear either side of the middle row at 60 deg/s
    assert still < corrected < turning


def test_views_thin_frames_and_keep_stills() -> None:
    frames = []
    for i in range(120):  # 2 s at 60 fps, walking 0.1 m/s; depth every 6th frame
        pose = synth.look_at(np.array([0.1 * i / 60, 0.0, 1.3]), np.array([0.1 * i / 60, 3.0, 1.3]))
        frames.append(
            PosedImage(
                id=f"s/{i}",
                session="s",
                kind="frame",
                file=f"video.mov#{i}",
                t=i / 60,
                camera_to_world=to_mat4(pose),
                intrinsics=K,
                depth=f"depth/{i:06d}.png" if i % 6 == 0 else None,
            )
        )
    still = frames[60].model_copy(update={"id": "s/s1", "kind": "still", "file": "stills/000001.jpg"})
    views = rf.select_views(frames, [still], {"s": 0})
    vt = [v.t for v in views if not v.still]
    assert all(views[i].t <= views[i + 1].t for i in range(len(views) - 1))
    assert sum(v.still for v in views) == 1
    # slow walking: one frame view per MAX_GAP_S, each with depth
    assert 4 <= len(vt) <= 6
    assert all(b - a <= rf.MAX_GAP_S + 0.11 for a, b in itertools.pairwise(vt))


def test_refine_stage_stops_without_video(tmp_path: Path) -> None:
    """The stage can't refine depth frames it has no images for: a RefineError naming the opt-out."""
    from scanmodel.stages import refine as stage

    from .test_meshcheck import _frames, _room  # pyright: ignore[reportPrivateUsage]

    poses, _ = _frames(tmp_path, _room(tmp_path / "room.ply"))  # depth frames, no video.mov
    (tmp_path / "ingest").mkdir()
    with pytest.raises(stage.RefineError, match=r"no video\.mov in session s1.*--poses posegraph"):
        stage.run(poses, tmp_path / "ingest", tmp_path / "refine")
    assert not (tmp_path / "refine").exists()


def _floor_cells(view: rf.View, truth: np.ndarray) -> dict[tuple[int, int], float]:
    """The floor (z = 0) cells around and ahead of the camera, as its LiDAR sees them at `view.pose`: world z
    of the true floor point brought into the camera frame by the true pose, then out by the view's pose."""
    out: dict[tuple[int, int], float] = {}
    for dx, dy in itertools.product((-1.0, -0.5, 0.0, 0.5, 1.0), (0.5, 1.0, 1.5)):
        p = np.array([truth[0, 3] + dx, truth[1, 3] + dy, 0.0])
        c = (int(np.floor(p[0] / 0.5)), int(np.floor(p[1] / 0.5)))
        p = np.array([(c[0] + 0.5) * 0.5, (c[1] + 0.5) * 0.5, 0.0])
        cam = truth[:3, :3].T @ (p - truth[:3, 3])
        out[c] = float((view.pose[:3, :3] @ cam + view.pose[:3, 3])[2])
    return out


def test_floor_heights_bring_a_drifted_pass_back() -> None:
    """Pass B is 3 cm too high and shares no tracks with pass A (ARKit's vertical drift, nothing visual to
    correct it): the adjustment alone keeps the offset; floor observations of the cells both passes saw
    bring B down to A, and the main floor back to z = 0."""
    rng = np.random.default_rng(3)
    ta, tt_a = _pass(0.6, 0.0)
    tb, tt_b = _pass(0.9, 30.0)
    truth = [*ta, *tb]
    start = [*ta, *[_offset(0.0, 0.0, 0.03) @ m for m in tb]]
    views = [
        _view(p, t, 0 if i < len(ta) else 1, i, "a" if i < len(ta) else "b")
        for i, (p, t) in enumerate(zip(start, [*tt_a, *tt_b], strict=True))
    ]
    pts_a, pts_b = _wall_points(rng), _wall_points(rng)  # each pass its own features: no cross-pass track
    view, track, uv, depth = [], [], [], []
    for i, m in enumerate(truth):
        idx, u, z = _observe(m, pts_a if i < len(ta) else pts_b, rng)
        view += [i] * len(idx)
        track += (idx + (0 if i < len(ta) else len(pts_a))).tolist()
        uv.append(u)
        depth.append(z)
    obs = rf.Observations(
        np.array(view),
        np.array(track),
        np.concatenate(uv),
        np.concatenate(depth),
        np.full(len(view), 2, np.uint8),
    )
    n_tracks = len(pts_a) + len(pts_b)
    odo = [
        rf.Odometry(i, i + 1, np.linalg.inv(truth[i]) @ truth[i + 1], 0.06, 0.0015)
        for i in range(len(truth) - 1)
        if views[i].trajectory == views[i + 1].trajectory
    ]
    points, keep, obs_ok = rf.init_landmarks(views, obs, n_tracks)
    use = obs_ok & keep[obs.track]
    floor = rf.floor_observations(
        views, {i: _floor_cells(v, m) for i, (v, m) in enumerate(zip(views, truth, strict=True))}
    )
    assert len(floor) and len(set(floor.view.tolist()) & set(range(len(ta), len(truth))))

    def b_above_a(poses: list[np.ndarray]) -> float:
        return float(np.mean([p[2, 3] - t[2, 3] for p, t in zip(poses[len(ta) :], tb, strict=True)])) - float(
            np.mean([p[2, 3] - t[2, 3] for p, t in zip(poses[: len(ta)], ta, strict=True)])
        )

    plain = rf.adjust(views, obs, points, use, odo)
    assert b_above_a(plain.poses) > 0.02  # nothing to correct it with
    out = rf.adjust(views, obs, points, use, odo, floor=floor)
    assert abs(b_above_a(out.poses)) < 0.004, b_above_a(out.poses)
    before, after = rf.floor_heights(start, floor), rf.floor_heights(out.poses, floor)
    lift = rf.main_floor_lift(before, after, floor)
    report = rf.floor_report(views, start, out.poses, floor)
    assert report["session_offset_max_mm"]["before"] > 25  # type: ignore[index]
    assert report["session_offset_max_mm"]["after"] < 4  # type: ignore[index]
    lifted = [p.copy() for p in out.poses]
    for p in lifted:
        p[2, 3] += lift
    # the main floor (pass A's and B's cells alike): back where it was before, on average
    assert abs(float(np.median(rf.floor_heights(lifted, floor) - before))) < 0.02


def _walk(y: float, t0: float) -> tuple[list[np.ndarray], list[float]]:
    """40 views along x at 1.3 m, one per second, looking at a wall 3 m ahead and slightly right."""
    eyes = [np.array([-2.0 + 4.0 * k / 39, y, 1.3]) for k in range(40)]
    return [synth.look_at(e, e + np.array([0.3, 3.0, -0.1])) for e in eyes], [t0 + k for k in range(40)]


def _pair(i: int, j: int, truth: list[np.ndarray], start: list[np.ndarray], wrong: np.ndarray | None = None):
    """A cross pair as the coarse round gives it: PnP finds the true relative pose (or, `wrong`, one with
    camera j moved by that world transform), compared with the current one (`start`)."""
    tj = truth[j] if wrong is None else wrong @ truth[j]
    rel = np.linalg.inv(tj) @ truth[i]  # camera i -> camera j
    cur = np.linalg.inv(start[j]) @ start[i]
    dev_m = float(np.linalg.norm(rel[:3, 3] - cur[:3, 3]))
    dev_deg = rf._angle_deg(rel[:3, :3], cur[:3, :3])  # pyright: ignore[reportPrivateUsage]
    return rf.PairMatches(i, j, "cross", np.arange(30), np.arange(30), 40, rel, dev_m, dev_deg)


def _drifted_lap() -> tuple[list[rf.View], list[np.ndarray], list[np.ndarray], list[rf.PairMatches]]:
    """Two reference passes (a, c) along a wall and a lap (b) past the same places, whose ARKit poses are
    right for 10 s, drift over the next 8 s (nothing matched there) to 0.5 m and 1 deg off, then stay off.
    Cross pairs: a with c, and each lap view (but in the drift) with the a and c views beside it. Returns
    the views (at the drifted poses), truth, the ARKit poses, the pairs."""
    ta, tt_a = _walk(0.0, 0.0)
    tc, tt_c = _walk(-0.3, 50.0)
    tb, tt_b = _walk(0.3, 100.0)

    def drift(k: int) -> np.ndarray:
        w = float(np.clip((k - 10) / 8, 0.0, 1.0))
        return _offset(0.0, 0.5 * w, 0.0, 1.0 * w)

    truth = [*ta, *tc, *tb]
    start = [*ta, *tc, *[drift(k) @ m for k, m in enumerate(tb)]]
    names = ["a"] * 40 + ["c"] * 40 + ["b"] * 40
    times = [*tt_a, *tt_c, *tt_b]
    views = [_view(p, t, n // 40, n, names[n]) for n, (p, t) in enumerate(zip(start, times, strict=True))]
    pairs = [_pair(k + s, 40 + k, truth, start) for k in range(40) for s in (-1, 0, 1) if 0 <= k + s < 40]
    for k in [*range(10), *range(18, 40)]:
        pairs += [_pair(r + s, 80 + k, truth, start) for r in (k, 40 + k) for s in (-1, 1) if 0 <= k + s < 40]
    return views, truth, start, pairs


def test_coarse_round_finds_drift_by_consensus_and_takes_it_out() -> None:
    views, truth, start, pairs = _drifted_lap()
    groups = rf.coarse_groups(views, pairs)
    admitted = [g for g in groups if g.admitted]
    assert admitted and {g.session for g in admitted} == {"b"}
    assert all(g.t0 >= 16.0 for g in admitted)  # the lap's drifted part only
    assert all(abs(g.offset[1] + 0.5) < 0.06 for g in admitted)  # where a and c put the lap's cameras
    # a and c see the lap's pairs as 0.5 m off too, but their tight pairs with each other outvote them
    assert not any(g.ok for g in groups if g.session != "b")
    raw = {v.image.id: p for v, p in zip(views, start, strict=True)}
    odo = rf.odometry(views, lambda im: raw[im.id], lambda v: False)
    spans = rf.drift_spans(views, pairs, groups)
    assert len(spans) == 1 and spans[0].session == "b" and spans[0].t0 == 109.0 and spans[0].t_end == 139.0
    stretched, n = rf.stretch_odometry(views, odo, spans)
    assert 6 <= n <= 14  # the lap's links from its last right view (9 s) to the first drifted bin's end
    poses, used = rf.coarse_solve(views, stretched, pairs, groups, spans)
    assert used > 0.9 * len(pairs)
    err = [float(np.linalg.norm(p[:3, 3] - t[:3, 3])) for p, t in zip(poses, truth, strict=True)]
    unseen = set(range(90, 98))  # while the lap drifted nothing was matched: the drift is spread evenly
    assert max(e for n, e in enumerate(err) if n not in unseen) < 0.01, err
    assert max(err) < 0.08, max(err)
    report = rf.session_pairs(views, pairs)
    assert report["a/b"]["drift"] and report["a/b"]["rejected_offset_mm"] > 400  # type: ignore[operator]
    assert report["a/c"]["rejected"] == 0


def test_coarse_round_ignores_a_local_consistent_alias() -> None:
    """No drift, but a repeated pattern in one place: the lap's views of it (one 4 s bin) all match 30 cm
    to the side, consistently. One bin is not drift: nothing is admitted."""
    views, truth, _, _ = _drifted_lap()
    for v, t in zip(views, truth, strict=True):
        v.pose = t
    shift = _offset(0.3, 0.0, 0.0)
    pairs = [_pair(k, 40 + k, truth, truth) for k in range(40)]
    pairs += [
        _pair(r + s, 80 + k, truth, truth, shift if 20 <= k < 24 else None)
        for k in range(40)
        for r in (k, 40 + k)
        for s in (-1, 1)
        if 0 <= k + s < 40
    ]
    groups = rf.coarse_groups(views, pairs)
    assert any(g.ok for g in groups)  # the alias passes the bin's own tests...
    assert not any(g.admitted for g in groups)  # ...but not the run


def _inherited_drift(
    gap: bool = False,
) -> tuple[
    list[rf.View], list[np.ndarray], list[np.ndarray], list[rf.PairMatches], list[tuple[int, float, float]]
]:
    """One ARKit trajectory, three sessions that continue each other (a, then m, then n), as scan 5's living
    session and the hallway / kitchen / lap after it. m drifts 0.5 m and 1 deg over 8 s in which nothing is
    matched (its views 10-17); n, continuing m's ARSession, inherits the drift. With `gap`, m jumps instead
    at a gap in its frames (view 14 missing, as a dropped excursion leaves it), and everything else is
    matched. Pairs: each later view with the earlier ones beside it (a with m, a and m with n). Returns the
    views (at the drifted poses), truth, the drifted poses, the pairs and the frame gaps."""
    ta, tt_a = _walk(0.0, 0.0)
    tm, tt_m = _walk(-0.3, 40.0)
    tn, tt_n = _walk(0.3, 80.0)

    def drift(k: int) -> np.ndarray:
        w = float(k > 14) if gap else float(np.clip((k - 10) / 8, 0.0, 1.0))
        return _offset(0.0, 0.5 * w, 0.0, 1.0 * w)

    truth = [*ta, *tm, *tn]
    start = [*ta, *[drift(k) @ p for k, p in enumerate(tm)], *[drift(40) @ p for p in tn]]
    names = ["a"] * 40 + ["m"] * 40 + ["n"] * 40
    times = [*tt_a, *tt_m, *tt_n]
    keep = [n for n in range(120) if not (gap and n == 54)]
    unseen = set() if gap else set(range(50, 58))
    views = [_view(start[n], times[n], 0, n, names[n]) for n in keep]
    at = {n: i for i, n in enumerate(keep)}
    truth, start = [truth[n] for n in keep], [start[n] for n in keep]
    pairs = []
    for k in range(40):
        for s in (-1, 1):
            if not 0 <= k + s < 40:
                continue
            for r, q in ((k + s, 40 + k), (k + s, 80 + k), (40 + k + s, 80 + k)):
                if r in at and q in at and r not in unseen and q not in unseen:
                    pairs.append(_pair(at[r], at[q], truth, start))
    gaps = [(0, times[53], times[55])] if gap else []
    return views, truth, start, pairs, gaps


@pytest.mark.parametrize("gap", [False, True])
def test_trajectory_cut_rejoins_a_block_that_inherited_drift(gap: bool) -> None:
    views, truth, _, pairs, gaps = _inherited_drift(gap)
    cuts, poses = rf.trajectory_cuts(views, pairs, gaps)
    admitted = [c for c in cuts if c.admitted]
    assert len(admitted) == 1, [(c.kind, c.t0, c.t1, c.agree, c.tight) for c in cuts]
    c = admitted[0]
    assert c.kind == ("gap" if gap else "unseen")
    assert (c.t0, c.t1) == ((53.0, 55.0) if gap else (49.0, 58.0))
    assert c.tight == 0 and c.sessions_after == 2
    assert abs(np.degrees(np.arctan2(c.jump[1, 0], c.jump[0, 0])) + 1.0) < 0.05
    err = [float(np.linalg.norm(p[:3, 3] - t[:3, 3])) for p, t in zip(poses, truth, strict=True)]
    inside = [e for v, e in zip(views, err, strict=True) if c.t0 < v.t < c.t1]
    assert max(e for v, e in zip(views, err, strict=True) if not c.t0 < v.t < c.t1) < 0.01, err
    assert not inside or max(inside) < 0.08  # nothing seen there: the jump is spread evenly over it
    # the stretch becomes a span of the trajectory: its ARKit links loosened, the stretch's views free
    spans = rf.cut_spans(views, cuts)
    assert len(spans) == 1 and spans[0].trajectory == 0
    raw = {v.image.id: v.pose for v in views}
    _, n = rf.stretch_odometry(views, rf.odometry(views, lambda im: raw[im.id], lambda v: False), spans)
    assert n == (1 if gap else 9)


def test_trajectory_cut_ignores_a_local_alias() -> None:
    """No drift, but n's views of a repeated pattern (one 4 s bin) all match 30 cm to the side, and an
    unseen stretch before them: not a cut (too few bins after it, and the tight pairs outvote it)."""
    views, truth, _, _, _ = _inherited_drift()
    for v, t in zip(views, truth, strict=True):
        v.pose = t
    shift = _offset(0.3, 0.0, 0.0)
    pairs = [
        _pair(r, q, truth, truth, shift if q >= 80 and 20 <= q - 80 < 24 else None)
        for k in range(40)
        for s in (-1, 1)
        if 0 <= k + s < 40
        for r, q in ((k + s, 40 + k), (k + s, 80 + k), (40 + k + s, 80 + k))
        if not (50 <= r < 58 or 50 <= q < 58)
    ]
    cuts, poses = rf.trajectory_cuts(views, pairs)
    assert cuts and not any(c.admitted for c in cuts)
    assert all(np.array_equal(p, v.pose) for p, v in zip(poses, views, strict=True))


def test_rigid_fit_recovers_a_block_move() -> None:
    rng = np.random.default_rng(1)
    move = _offset(0.2, -0.4, 0.01, 2.0)
    before = [synth.look_at(e, e + np.array([1.0, 0.5, 0.0])) for e in rng.uniform(-3, 3, (20, 3))]
    after = [move @ p for p in before]
    assert np.allclose(rf.rigid_fit(before, after), move, atol=1e-6)
