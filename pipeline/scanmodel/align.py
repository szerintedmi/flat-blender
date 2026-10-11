"""Drift correction: a keyframe pose graph over ARKit's trajectories (`scanmodel poses --pose-graph`).

ARKit drifts a few cm per minute and does not re-publish poses it already reported after a loop
closure, so places recorded minutes apart disagree (5-15 cm on the second scan). Rigid per-session
corrections can't model that: drift grows with time.

- Keyframes: frames with depth, thinned to 15 cm / 10 deg of motion.
- Odometry edges between consecutive keyframes of one continuous ARSession: ARKit's relative pose,
  accurate to mm over seconds. Sessions joined by Next room (or a closing lap) are one trajectory;
  a session that relocalised against a world map starts its own.
- Gravity: every keyframe keeps ARKit's roll and pitch.
- Loop closures: SIFT matches between keyframes at least `MIN_DT` apart that saw the same space, lifted
  to 3D with each frame's LiDAR depth, RANSAC-Kabsch. Gated against ARKit: repeated things (stair
  steps, identical cabinet doors) match consistently but wrongly, and real drift is small vertically.
- Anchors (anchors.jsonl): if ARKit moved our anchors after its own loop closures, each moved anchor
  says where the camera really was at that time.
- Cauchy kernel on closures; Levenberg-Marquardt. Corrections are interpolated in time to every frame
  and still.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations_with_replacement
from pathlib import Path
from typing import cast

import cv2
import gtsam
import numpy as np
from shapely import Polygon, contains_xy
from shapely.ops import unary_union

from .io import VideoReader, iter_jsonl_lines, read_confidence, read_depth
from .models import bundle as b
from .models.common import F64, mat
from .models.derived import PosedImage
from .points import thin_frames
from .roomplan import CapturedRoom

MIN_DT = 15.0  # a loop closure links moments at least this far apart
FLOOR_CELL = 0.5  # floor heights are compared per cell of this size
WIDTH, HEIGHT = 960, 720  # feature image size
# A closure may disagree with the floor-corrected trajectory by at most GATE_Z + RATE_Z * dt vertically
# (the floors already fixed the vertical drift) and GATE_XY + RATE_XY * dt horizontally, capped. Rates:
# the worst room of the ARKitScenes bench (drift between 20 s windows: up to ~4 cm z, ~12 cm xy).
GATE_Z, RATE_Z = 0.03, 0.0002
GATE_XY, RATE_XY, GATE_XY_MAX = 0.05, 0.002, 0.40


@dataclass
class Keyframe:
    session: str
    trajectory: int
    t: float
    pose: F64  # camera_to_world, scene frame, OpenCV axes
    image: PosedImage


@dataclass
class Closure:
    i: int
    j: int
    rel: F64  # pose of keyframe j in keyframe i's camera frame
    inliers: int
    pair: tuple[str, str]  # sessions, sorted


@dataclass
class Features:
    uv: np.ndarray
    desc: np.ndarray
    xyz: F64  # scene frame, from ARKit's pose and the frame's depth
    voxels: set[tuple[int, int, int]] = field(default_factory=set[tuple[int, int, int]])


RESUME_S = 30.0  # bundles without `continued_from`: Next room resumed recording within this


def trajectories(sessions: list[tuple[str, b.Session, float, float]]) -> dict[str, int]:
    """Session -> trajectory index. Input: (id, session.json, first t, last t), in time order.

    A session that continued the previous session's ARSession (Next room / closing lap) is the same
    trajectory: `continued_from` names the previous session (app 0.5+, any pause before recording resumed);
    in older bundles, a scan or lap session whose `world_map_in` is the previous one and that resumed
    within `RESUME_S`.
    """
    out: dict[str, int] = {}
    prev: tuple[str, float] | None = None
    k = -1
    for sid, s, t0, t1 in sessions:
        continued = (
            prev is not None
            and t0 - prev[1] >= 0
            and (
                s.continued_from == prev[0]
                if s.continued_from is not None
                else s.type in (b.SessionType.SCAN, b.SessionType.LAP)
                and s.world_map_in == prev[0]
                and t0 - prev[1] < RESUME_S
            )
        )
        if not continued:
            k += 1
        out[sid] = k
        prev = (sid, t1)
    return out


def keyframes(frames: list[PosedImage], trajectory: dict[str, int]) -> list[Keyframe]:
    by: dict[str, list[PosedImage]] = defaultdict(list)
    for f in frames:
        if f.depth and f.t is not None and f.tracking in (None, b.TrackingState.NORMAL):
            by[f.session].append(f)
    out = [
        Keyframe(sid, trajectory[sid], f.t or 0.0, mat(f.camera_to_world), f)
        for sid, fs in by.items()
        for f in thin_frames(sorted(fs, key=lambda f: f.t or 0.0), 0.15, 10.0)
    ]
    return sorted(out, key=lambda k: (k.trajectory, k.t))


def features(raw: Path, kfs: list[Keyframe]) -> list[Features | None]:
    """SIFT on the keyframes' video frames, keypoints with flat, valid depth lifted to 3D."""
    out: list[Features | None] = [None] * len(kfs)
    video = raw / b.VIDEO_FILE
    if not video.is_file():
        return out
    sift = cv2.SIFT.create(nfeatures=1500)
    reader = VideoReader(video)
    order = sorted(range(len(kfs)), key=lambda i: int(kfs[i].image.file.split("#")[1]))
    for i in order:
        im = kfs[i].image
        bgr = reader.read(int(im.file.split("#")[1]))
        if bgr is None or im.depth is None:
            continue
        gray = cv2.cvtColor(
            cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
        )
        kps, desc = cast(tuple[object, object], sift.detectAndCompute(gray, None))
        if not isinstance(desc, np.ndarray) or not isinstance(kps, tuple | list) or not kps:  # blank frame
            continue
        d = read_depth(raw / im.depth)
        if im.confidence:
            d[read_confidence(raw / im.confidence) < 2] = 0.0
        dh, dw = d.shape
        uv = np.array([k.pt for k in kps])
        du = np.clip(((uv[:, 0] + 0.5) * dw / WIDTH).astype(int), 1, dw - 2)
        dv = np.clip(((uv[:, 1] + 0.5) * dh / HEIGHT).astype(int), 1, dh - 2)
        win = np.stack([d[dv + a, du + c] for a in (-1, 0, 1) for c in (-1, 0, 1)], axis=1)
        z = np.median(win, axis=1)
        ok = (win.min(axis=1) > 0.2) & (np.ptp(win, axis=1) < 0.02) & (z < 3.5)  # no depth edges
        if ok.sum() < 20:
            continue
        k = im.intrinsics.scaled(WIDTH, HEIGHT)
        z, uv = z[ok], uv[ok]
        cam = np.column_stack([(uv[:, 0] + 0.5 - k.cx) / k.fx * z, (uv[:, 1] + 0.5 - k.cy) / k.fy * z, z])
        m = kfs[i].pose
        xyz = cam @ m[:3, :3].T + m[:3, 3]
        out[i] = Features(
            uv.astype(np.float32), desc[ok], xyz, set(map(tuple, np.floor(xyz / 0.25).astype(int)))
        )
    reader.close()
    return out


def footprints(rooms: list[tuple[CapturedRoom, F64]], margin: float = 0.15) -> list[Polygon]:
    """Stair and furniture footprints (RoomPlan objects, scene frame) grown by `margin`: their treads
    and tops are horizontal but not floor. `rooms`: (CapturedRoom, its session_to_scene)."""
    out: list[Polygon] = []
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    for room, s2f in rooms:
        for o in room.objects:
            m = s2f @ o.transform
            x = m[:2, 0] / (np.linalg.norm(m[:2, 0]) + 1e-12)
            y = np.array([-x[1], x[0]])
            hx, hy = float(o.dimensions[0]) / 2, float(o.dimensions[2]) / 2
            out.append(Polygon([m[:2, 3] + sx * hx * x + sy * hy * y for sx, sy in corners]).buffer(margin))
    return out


def floors(
    raw: Path, kfs: list[Keyframe], exclude: list[Polygon] | None = None
) -> list[dict[tuple[int, int], float]]:
    """Per keyframe: floor height in each FLOOR_CELL it saw. Floor = horizontal surfaces (normals from
    the depth image) at least 0.6 m below the camera, outside stair and furniture footprints; per cell
    the lowest layer, if it holds at least 30 % of the cell's points."""
    blocked = unary_union(exclude) if exclude else None
    out: list[dict[tuple[int, int], float]] = []
    for k in kfs:
        im = k.image
        cells: dict[tuple[int, int], float] = {}
        out.append(cells)
        if im.depth is None:
            continue
        d = read_depth(raw / im.depth)
        if im.confidence:
            d[read_confidence(raw / im.confidence) < 2] = 0.0
        dh, dw = d.shape
        kd = im.intrinsics.scaled(dw, dh)
        v, u = np.mgrid[0:dh, 0:dw]
        p = np.stack([(u + 0.5 - kd.cx) / kd.fx * d, (v + 0.5 - kd.cy) / kd.fy * d, d], axis=-1)
        dx, dy = p[1:-1, 2:] - p[1:-1, :-2], p[2:, 1:-1] - p[:-2, 1:-1]
        z = d[1:-1, 1:-1]
        ok = (
            (z > 0.2)
            & (z < 4.0)
            & (d[1:-1, 2:] > 0)
            & (d[1:-1, :-2] > 0)
            & (d[2:, 1:-1] > 0)
            & (d[:-2, 1:-1] > 0)
        )
        ok &= (np.linalg.norm(dx, axis=-1) < 0.1 * z) & (
            np.linalg.norm(dy, axis=-1) < 0.1 * z
        )  # no depth edges
        n = np.cross(dx[ok], dy[ok])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        rot, t = k.pose[:3, :3], k.pose[:3, 3]
        world = p[1:-1, 1:-1][ok] @ rot.T + t
        keep = (np.abs(n @ rot[2]) > 0.95) & (world[:, 2] < t[2] - 0.6)  # horizontal, well below the camera
        world = world[keep][::2]
        if blocked is not None and len(world):
            world = world[~contains_xy(blocked, world[:, 0], world[:, 1])]
        if len(world) < 50:
            continue
        keys = np.floor(world[:, :2] / FLOOR_CELL).astype(int)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        for c in range(len(uniq)):
            zc = world[inv.ravel() == c, 2]
            if len(zc) < 40:
                continue
            h, e = np.histogram(zc, bins=np.arange(zc.min(), zc.max() + 0.02, 0.01))
            lowest = int(np.flatnonzero(h >= 0.3 * h.max())[0])
            layer = (zc >= e[lowest] - 0.01) & (zc <= e[lowest + 1] + 0.01)
            if layer.mean() >= 0.3:
                cells[(int(uniq[c][0]), int(uniq[c][1]))] = float(np.median(zc[layer]))
    return out


def floor_links(
    kfs: list[Keyframe], cells: list[dict[tuple[int, int], float]], per_cell: int = 30
) -> list[tuple[int, int, float]]:
    """(i, j, dz): keyframes >= MIN_DT apart that saw the floor in the same cells must agree on its
    height, so camera i is dz above camera j. Floors are horizontal and don't move; repeated things
    (stair steps, identical doors) can't fool this, and a step between rooms stays a step."""
    seen: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, cs in enumerate(cells):
        for c in cs:
            seen[c].append(i)
    diffs: dict[tuple[int, int], list[float]] = defaultdict(list)
    for c, ids in seen.items():
        ids = ids[:: max(1, len(ids) // per_cell)]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                i, j = ids[x], ids[y]
                if abs(kfs[i].t - kfs[j].t) >= MIN_DT:
                    diffs[(i, j)].append(cells[i][c] - cells[j][c])
    out: list[tuple[int, int, float]] = []
    for (i, j), ds in diffs.items():
        if len(ds) >= 3:
            arr = np.array(ds)
            med = float(np.median(arr))
            if np.median(np.abs(arr - med)) < 0.02:  # one consistent floor, not a step edge
                out.append((i, j, float(kfs[i].pose[2, 3] - kfs[j].pose[2, 3]) - med))
    return out


def gate(kfs: list[Keyframe], poses: list[F64], links: list[Closure]) -> list[Closure]:
    """Closures that agree with `poses` (the floor-corrected trajectory) within the drift possible in
    the time between them (GATE_*, RATE_*). Repeated things (stair steps ~20 cm apart, identical
    cabinet doors) match consistently but wrongly."""
    out: list[Closure] = []
    for c in links:
        e = (poses[c.i] @ c.rel)[:3, 3] - poses[c.j][:3, 3]  # where the closure puts j, minus where it is
        dt = abs(kfs[c.i].t - kfs[c.j].t)
        xy = min(GATE_XY + RATE_XY * dt, GATE_XY_MAX)
        if abs(e[2]) <= GATE_Z + RATE_Z * dt and np.linalg.norm(e[:2]) <= xy:
            out.append(c)
    return out


def closure_report(
    kfs: list[Keyframe], poses: list[F64], links: list[Closure], kept: list[Closure], drift_m: float = 0.10
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Per session pair: candidate closures, how many the gate kept and rejected, and where the rejected ones
    put the later keyframe vs `poses` (median offset, mm, and the share within 10 cm of it); `drift`: five or
    more rejected closures, 60 % agreeing on an offset larger than `drift_m` (the poses are off, not the
    matches). Plus the sessions without a single closure to another session: the pose graph can't correct
    their drift at all (scan 5's lap: too few features in its fast frames)."""
    ok = {(c.i, c.j) for c in kept}
    by: dict[tuple[str, str], list[Closure]] = defaultdict(list)
    for c in links:
        by[c.pair].append(c)
    out: dict[str, dict[str, object]] = {}
    for pair, cs in sorted(by.items()):
        rej = [c for c in cs if (c.i, c.j) not in ok]
        row: dict[str, object] = {"candidates": len(cs), "accepted": len(cs) - len(rej), "rejected": len(rej)}
        if rej:
            e = np.array([(poses[c.i] @ c.rel)[:3, 3] - poses[c.j][:3, 3] for c in rej])
            med = np.median(e, axis=0)
            agree = float(np.mean(np.linalg.norm(e - med, axis=1) <= 0.10))
            row |= {
                "rejected_offset_mm": round(1000 * float(np.linalg.norm(med)), 1),
                "rejected_offset_xyz_mm": [round(1000 * float(x), 1) for x in med],
                "rejected_agree": round(agree, 2),
                "drift": len(rej) >= 5 and agree >= 0.6 and float(np.linalg.norm(med)) > drift_m,
            }
        out["/".join(pair)] = row
    linked = {s for c in kept if c.pair[0] != c.pair[1] for s in c.pair}
    alone = sorted({k.session for k in kfs} - linked)
    return out, alone


def _kabsch(a: F64, c: F64) -> tuple[F64, F64]:
    """R, t with R c + t ~ a."""
    ca, cc = a.mean(0), c.mean(0)
    u, _, vt = np.linalg.svd((c - cc).T @ (a - ca))
    r = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return r, ca - r @ cc


def closures(kfs: list[Keyframe], feats: list[Features | None], per_pair: int = 600) -> list[Closure]:
    """Loop closures between keyframes >= MIN_DT apart that saw the same 25 cm voxels."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    rng = np.random.default_rng(0)
    cv2.setRNGSeed(0)  # USAC's sampling: repeatable closures, so repeatable poses
    by: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(kfs):
        if feats[i] is not None:
            by[k.session].append(i)
    out: list[Closure] = []
    for sa, sb in combinations_with_replacement(sorted(by), 2):
        cand: list[tuple[float, int, int]] = []
        for i in by[sa]:
            for j in by[sb]:
                fi, fj = feats[i], feats[j]
                assert fi is not None and fj is not None
                if (sa == sb and j <= i) or abs(kfs[i].t - kfs[j].t) < MIN_DT:
                    continue
                if kfs[i].pose[:3, 2] @ kfs[j].pose[:3, 2] < 0:  # looking opposite ways
                    continue
                inter = len(fi.voxels & fj.voxels)
                if inter and inter / min(len(fi.voxels), len(fj.voxels)) > 0.25:
                    cand.append((-inter / min(len(fi.voxels), len(fj.voxels)), i, j))
        for _, i, j in sorted(cand)[:per_pair]:
            c = _closure(kfs, feats, i, j, bf, rng)
            if c is not None:
                out.append(c)
    return out


def _closure(
    kfs: list[Keyframe],
    feats: list[Features | None],
    i: int,
    j: int,
    bf: cv2.BFMatcher,
    rng: np.random.Generator,
) -> Closure | None:
    fi, fj = feats[i], feats[j]
    assert fi is not None and fj is not None
    knn = bf.knnMatch(fi.desc, fj.desc, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 20:
        return None
    ia = np.array([m.queryIdx for m in good])
    ib = np.array([m.trainIdx for m in good])
    try:
        found = cast(
            tuple[object, object], cv2.findFundamentalMat(fi.uv[ia], fj.uv[ib], cv2.USAC_MAGSAC, 1.5, 0.999)
        )
    except cv2.error:  # degenerate, e.g. all points on one plane
        return None
    f, inl = found  # None, None when no model was found
    if not isinstance(f, np.ndarray) or not isinstance(inl, np.ndarray) or f.size == 0:
        return None
    keep = inl.ravel().astype(bool)
    ti, tj = kfs[i].pose, kfs[j].pose
    la = (fi.xyz[ia[keep]] - ti[:3, 3]) @ ti[:3, :3]  # camera-local
    lb = (fj.xyz[ib[keep]] - tj[:3, 3]) @ tj[:3, :3]
    if len(la) < 20:
        return None
    best = np.zeros(len(la), bool)
    for _ in range(200):
        s = rng.choice(len(la), 3, replace=False)
        r, t = _kabsch(la[s], lb[s])
        inl3 = np.linalg.norm(lb @ r.T + t - la, axis=1) < 0.03
        if inl3.sum() > best.sum():
            best = inl3
    if best.sum() < 20:
        return None
    r, t = _kabsch(la[best], lb[best])
    if np.sqrt(np.mean(np.sum((lb[best] @ r.T + t - la[best]) ** 2, axis=1))) > 0.02:
        return None
    rel = np.eye(4)
    rel[:3, :3], rel[:3, 3] = r, t
    e = (ti @ rel)[:3, 3] - tj[:3, 3]
    if np.linalg.norm(e) > 0.6:  # far beyond any drift: a wrong match
        return None
    return Closure(
        i, j, rel, int(best.sum()), (min(kfs[i].session, kfs[j].session), max(kfs[i].session, kfs[j].session))
    )


def anchor_priors(kfs: list[Keyframe], moves: list[tuple[str, float, F64]]) -> list[tuple[int, F64]]:
    """(keyframe, corrected pose) from moved anchors: (session, t, world correction at t)."""
    out: list[tuple[int, F64]] = []
    for sid, t, delta in moves:
        cands = [i for i, k in enumerate(kfs) if k.session == sid]
        if cands:
            i = min(cands, key=lambda i: abs(kfs[i].t - t))
            if abs(kfs[i].t - t) < 1.0:
                out.append((i, delta @ kfs[i].pose))
    return out


def anchor_moves(
    logs: list[tuple[Path, F64]], dropped: dict[Path, list[tuple[float, float]]] | None = None
) -> tuple[list[tuple[str, float, F64]], float]:
    """Corrections ARKit applied to our anchors, over all sessions' anchors.jsonl (an anchor added in
    one session can be updated while a later one records): (session that added it, t added, world
    correction in the scene frame), and the largest move in metres (0: ARKit never moved them).

    `logs`: (session dir, that session's session_to_scene). `dropped`: session dir -> time windows ingest
    left out (IngestReport.dropped); events logged in them are ignored. An anchor placed during a pose
    excursion sits where the camera wrongly was, and ARKit later moves it back (scan 5: still 56, 64 cm):
    that is the excursion's undoing, not a correction for the frames around it."""
    first: dict[str, tuple[float, F64]] = {}
    last: dict[str, tuple[float, F64]] = {}
    names: dict[str, str] = {}
    for raw, s2f in logs:
        path = raw / b.ANCHORS_FILE
        if not path.is_file():
            continue
        to_scene = np.linalg.inv(s2f)
        skip = (dropped or {}).get(raw, [])
        for _, line in iter_jsonl_lines(path):
            ev = b.AnchorEvent.model_validate_json(line)
            if any(t0 <= ev.t <= t1 for t0, t1 in skip):
                continue
            m = s2f @ mat(ev.transform) @ to_scene
            if ev.event == "add":
                first[ev.id] = (ev.t, m)
                names[ev.id] = ev.name
            elif ev.id not in last or ev.t >= last[ev.id][0]:
                last[ev.id] = (ev.t, m)
    out: list[tuple[str, float, F64]] = []
    biggest = 0.0
    for aid, (t, m0) in first.items():
        if aid in last:
            m1 = last[aid][1]
            biggest = max(biggest, float(np.linalg.norm(m1[:3, 3] - m0[:3, 3])))
            out.append((names[aid].split("/")[0], t, m1 @ np.linalg.inv(m0)))
    return out, biggest


def solve(
    kfs: list[Keyframe],
    links: list[Closure],
    priors: list[tuple[int, F64]] = [],  # noqa: B006 (read only)
    drop: Callable[[Closure], bool] | None = None,
    floor: list[tuple[int, int, float]] = [],  # noqa: B006 (read only)
) -> list[F64]:
    """Corrected keyframe poses."""

    def pose3(m: F64) -> gtsam.Pose3:
        return gtsam.Pose3(gtsam.Rot3(m[:3, :3]), gtsam.Point3(*m[:3, 3]))

    def sigmas(rot_deg: float, trans_m: float) -> gtsam.noiseModel.Diagonal:
        return gtsam.noiseModel.Diagonal.Sigmas(np.array([np.radians(rot_deg)] * 3 + [trans_m] * 3))

    g = gtsam.NonlinearFactorGraph()
    init = gtsam.Values()
    for i, k in enumerate(kfs):
        init.insert(i, pose3(k.pose))
    starts: dict[int, int] = {}
    for i, k in enumerate(kfs):
        starts.setdefault(k.trajectory, i)
    for traj, i in starts.items():  # gauge: the first trajectory is the reference, others start near ARKit
        g.add(
            gtsam.PriorFactorPose3(
                i, pose3(kfs[i].pose), sigmas(1e-3, 1e-4) if traj == 0 else sigmas(5.0, 1.0)
            )
        )
    down = gtsam.Unit3(np.array([0.0, 0.0, -1.0]))
    gravity = gtsam.noiseModel.Isotropic.Sigma(2, np.radians(0.05))
    for i, k in enumerate(kfs):
        g.add(
            gtsam.AttitudeFactorPose3(
                i, down, gravity, gtsam.Unit3(k.pose[:3, :3].T @ np.array([0.0, 0.0, -1.0]))
            )
        )
    for i in range(len(kfs) - 1):
        a, c = kfs[i], kfs[i + 1]
        if a.trajectory != c.trajectory:
            continue
        rel = np.linalg.inv(a.pose) @ c.pose
        dt = c.t - a.t
        # ARKit's relative motion: mm over seconds, about 1 % of the distance travelled
        g.add(
            gtsam.BetweenFactorPose3(
                i,
                i + 1,
                pose3(rel),
                sigmas(0.05 + 0.02 * dt, 0.002 + 0.01 * float(np.linalg.norm(rel[:3, 3])) + 0.001 * dt),
            )
        )
    robust = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(1.0), sigmas(0.5, 0.015)
    )
    for c in links:
        if drop is None or not drop(c):
            g.add(gtsam.BetweenFactorPose3(c.i, c.j, pose3(c.rel), robust))
    for i, m in priors:
        g.add(gtsam.PriorFactorPose3(i, pose3(m), sigmas(0.2, 0.01)))
    floor_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(1.0), gtsam.noiseModel.Isotropic.Sigma(1, 0.015)
    )
    for i, j, dz in floor:
        g.add(gtsam.CustomFactor(floor_noise, [i, j], partial(_height_error, dz)))
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    result = gtsam.LevenbergMarquardtOptimizer(g, init, params).optimize()
    return [result.atPose3(i).matrix() for i in range(len(kfs))]


def _height_error(
    dz: float, this: gtsam.CustomFactor, values: gtsam.Values, jac: list[np.ndarray] | None
) -> np.ndarray:
    """Camera i is dz above camera j (world z). Pose3 moves its translation by R v, so dz/dv = R[2]."""
    a, c = values.atPose3(this.keys()[0]), values.atPose3(this.keys()[1])
    if jac is not None:
        jac[0] = np.concatenate([np.zeros(3), a.rotation().matrix()[2]])[None]
        jac[1] = -np.concatenate([np.zeros(3), c.rotation().matrix()[2]])[None]
    return np.array([a.translation()[2] - c.translation()[2] - dz])


def corrector(kfs: list[Keyframe], solved: list[F64]) -> Callable[[str, int, float, F64], F64]:
    """f(session, trajectory, t, pose) -> corrected pose, interpolating keyframe corrections in time."""
    by: dict[int, list[int]] = defaultdict(list)
    for i, k in enumerate(kfs):
        by[k.trajectory].append(i)
    deltas = [gtsam.Pose3(s @ np.linalg.inv(k.pose)) for s, k in zip(solved, kfs, strict=True)]
    times = {traj: np.array([kfs[i].t for i in ids]) for traj, ids in by.items()}

    def correct(session: str, trajectory: int, t: float, pose: F64) -> F64:
        ids = by.get(trajectory)
        if not ids:
            return pose
        ts = times[trajectory]
        a = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ids) - 1))
        c = min(a + 1, len(ids) - 1)
        w = 0.0 if c == a else float(np.clip((t - ts[a]) / (ts[c] - ts[a]), 0.0, 1.0))
        delta = deltas[ids[a]].slerp(w, deltas[ids[c]])
        return delta.matrix() @ pose

    return correct


def stability(
    kfs: list[Keyframe],
    links: list[Closure],
    priors: list[tuple[int, F64]],
    solved: list[F64],
    keep: Callable[[Keyframe], bool],
    floor: list[tuple[int, int, float]] = [],  # noqa: B006 (read only)
) -> dict[str, float]:
    """Leave one session pair's closures out: largest move of a kept keyframe's camera (metres)."""
    out: dict[str, float] = {}
    for pair in sorted({c.pair for c in links}):
        again = solve(kfs, links, priors, drop=lambda c, pair=pair: c.pair == pair, floor=floor)
        moves = [
            np.linalg.norm(a[:3, 3] - s[:3, 3]) for a, s, k in zip(again, solved, kfs, strict=True) if keep(k)
        ]
        out["/".join(pair)] = float(max(moves, default=0.0))
    return out
