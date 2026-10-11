"""Stage 2, poses: merge sessions into one frame and fix the scene frame.

- Sessions that loaded the project's world map share ARKit's frame already. A session without it
  is registered globally (FPFH + RANSAC + ICP) against the first session.
- Every later session gets an ICP check against the sessions before it: a correction above 3 cm
  means drift and is reported (not applied: the world map is the source of alignment).
- Scene frame: z = gravity (ARKit is gravity-aligned already), floor at z = 0, yaw snapped so the
  dominant wall directions are x/y, origin at the front door with the scan lying towards +y.
  Front door: overrides.json `frame.origin_door`, else the first session's door nearest to where
  that session started.
- Optional bundle adjustment (LightGlue + pycolmap) is phase 5 and not in this stage yet.

Output in data/<project>/derived/poses/: poses.json (Poses), frames.jsonl and stills.jsonl (PosedImage, scene frame),
walls/<session>@<s>.npz (wall points with camera-facing normals per 30 s window of one session, scene frame).
"""

import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import align
from ..config import from_repo, rel
from ..geom import ARKIT_TO_ZUP, angle_mod90, apply, dominant_axis_deg, rot_z, translation
from ..io import decoders, read_jsonl, write_json, write_jsonl
from ..models import bundle as b
from ..models.common import F64, mat, to_mat4
from ..models.derived import IngestReport, PosedImage, Poses, SessionPoses
from ..models.overrides import Overrides
from ..points import (
    WALLS_DIR,
    WINDOW_S,
    floor_height,
    global_register,
    icp_check,
    sample_points,
    wall_points,
    with_normals,
    write_wall_points,
)
from ..roomplan import CapturedRoom, load_captured_room, short_id


@dataclass
class SessionData:
    id: str
    raw: Path
    session: b.Session
    frames: list[PosedImage]
    stills: list[PosedImage]
    room: CapturedRoom | None
    to_zup: F64  # this session's z-up world -> the first session's z-up world
    points: F64
    dropped: list[tuple[float, float]]  # time windows ingest left out mid-session (IngestReport.dropped)


def _load(ingest_dir: Path) -> list[SessionData]:
    out: list[SessionData] = []
    for rep_path in sorted(ingest_dir.glob("*/report.json")):
        rep = IngestReport.model_validate_json(rep_path.read_text())
        if not rep.ok:
            continue
        raw = from_repo(rep.raw_dir)
        session = b.Session.model_validate_json((raw / b.SESSION_FILE).read_text())
        frames = read_jsonl(rep_path.parent / "frames.jsonl", PosedImage)
        out.append(
            SessionData(
                id=rep.session,
                raw=raw,
                session=session,
                frames=frames,
                stills=read_jsonl(rep_path.parent / "stills.jsonl", PosedImage),
                room=load_captured_room(raw / b.ROOM_JSON_FILE),
                to_zup=np.eye(4),
                points=sample_points(
                    raw, [f for f in frames if f.tracking in (None, b.TrackingState.NORMAL)]
                ),
                dropped=[(w.t0, w.t1) for w in rep.dropped],
            )
        )
    # Earliest first: later sessions relocalise against the first one's world map.
    out.sort(key=lambda s: s.session.started_at)
    return out


def _wall_angles(sessions: list[SessionData]) -> tuple[F64, F64]:
    angles: list[float] = []
    weights: list[float] = []
    for s in sessions:
        if s.room is None:
            continue
        for w in s.room.of("wall"):
            x = (s.to_zup @ ARKIT_TO_ZUP @ w.transform)[:3, 0]
            angles.append(float(np.degrees(np.arctan2(x[1], x[0]))))
            weights.append(float(w.dimensions[0]))
    return np.array(angles), np.array(weights)


def _normal_angles(pts: F64, normals: F64) -> tuple[F64, F64]:
    vertical = np.abs(normals[:, 2]) < 0.1
    n = normals[vertical]
    return np.degrees(np.arctan2(n[:, 1], n[:, 0])), np.ones(len(n))


@dataclass
class Door:
    id: str
    session: str
    centre: F64  # z-up world of the first session
    normal: F64  # horizontal unit vector


def _doors(sessions: list[SessionData]) -> list[Door]:
    out: list[Door] = []
    for s in sessions:
        if s.room is None:
            continue
        for d in s.room.of("door"):
            m = s.to_zup @ ARKIT_TO_ZUP @ d.transform
            n = m[:3, 2].copy()
            n[2] = 0
            out.append(Door(d.id, s.id, m[:3, 3], n / (np.linalg.norm(n) + 1e-12)))
    return out


def _pose_graph(
    sessions: list[SessionData],
    sess_out: list[SessionPoses],
    frames: list[PosedImage],
    stills: list[PosedImage],
    stability: bool = False,
) -> dict[str, object]:
    """Correct the drift in place (see scanmodel.align): frames, stills, and each session's RoomPlan frame."""
    span = {
        s.id: (min(f.t or 0.0 for f in s.frames), max(f.t or 0.0 for f in s.frames))
        for s in sessions
        if s.frames
    }
    ordered = sorted((s for s in sessions if s.id in span), key=lambda s: span[s.id][0])
    traj = align.trajectories([(s.id, s.session, *span[s.id]) for s in ordered])
    kfs = align.keyframes(frames, traj)
    feats: list[align.Features | None] = [None] * len(kfs)
    idx_of = {s.id: [i for i, k in enumerate(kfs) if k.session == s.id] for s in ordered}
    # Decoding the video is most of it: the sessions' in parallel (OpenCV decodes and runs SIFT without the GIL).
    with ThreadPoolExecutor(max_workers=decoders(len(ordered))) as pool:
        found = pool.map(lambda s: align.features(s.raw, [kfs[i] for i in idx_of[s.id]]), ordered)
        for s, fs in zip(ordered, found, strict=True):
            for i, f in zip(idx_of[s.id], fs, strict=True):
                feats[i] = f
    # Stairs and furniture (every session's RoomPlan objects) are horizontal but not floor.
    scene_of = {so.session: mat(so.session_to_scene) for so in sess_out}
    blocked = align.footprints([(s.room, scene_of[s.id]) for s in sessions if s.room is not None])
    cells: list[dict[tuple[int, int], float]] = [{} for _ in kfs]
    for s in ordered:
        idx = [i for i, k in enumerate(kfs) if k.session == s.id]
        for i, c in zip(idx, align.floors(s.raw, [kfs[i] for i in idx], blocked), strict=True):
            cells[i] = c
    floor = align.floor_links(kfs, cells)
    s2f = {so.session: mat(so.session_to_scene) for so in sess_out}
    moves, anchors_moved = align.anchor_moves(
        [(s.raw, s2f[s.id]) for s in sessions], {s.raw: s.dropped for s in sessions}
    )
    priors = align.anchor_priors(kfs, moves) if anchors_moved > 0.005 else []
    # Pass 1: ARKit's motion, gravity and the floors fix the vertical drift. Pass 2: image closures,
    # gated against pass 1 rather than raw ARKit (which is off vertically by up to ~10 cm).
    first = align.solve(kfs, [], priors, floor=floor)
    candidates = align.closures(kfs, feats)
    links = align.gate(kfs, first, candidates)
    by_pair, alone = align.closure_report(kfs, first, candidates, links)
    for name, row in by_pair.items():
        if row.get("drift"):
            print(
                f"  warn: {name}: {row['rejected']} rejected closures agree on an offset of"
                f" {row['rejected_offset_mm']} mm: drift the gate kept out?"
            )
    if alone and len({k.session for k in kfs}) > 1:
        print(
            f"  warn: no closure to another session for {', '.join(alone)}: its drift stays uncorrected here"
        )
    solved = align.solve(kfs, links, priors, floor=floor)
    unlifted = solved
    # The main floor level (the most floor seen) at z = 0.
    heights = [
        z + s_[2, 3] - k.pose[2, 3] for s_, k, cs in zip(solved, kfs, cells, strict=True) for z in cs.values()
    ]
    if heights:
        hist, edges = np.histogram(heights, bins=np.arange(min(heights), max(heights) + 0.02, 0.01))
        main = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2)
        lift = np.eye(4)
        lift[2, 3] = -main
        solved = [lift @ m for m in solved]
    correct = align.corrector(kfs, solved)
    for images in (frames, stills):
        for n, im in enumerate(images):
            if im.t is not None and im.session in traj:
                pose = correct(im.session, traj[im.session], im.t, mat(im.camera_to_world))
                images[n] = im.model_copy(update={"camera_to_world": to_mat4(pose)})
    for n, so in enumerate(sess_out):  # RoomPlan: one rigid correction, from mid-session
        if so.session in traj:
            mid = sum(span[so.session]) / 2
            delta = correct(so.session, traj[so.session], mid, np.eye(4))
            sess_out[n] = so.model_copy(update={"session_to_scene": to_mat4(delta @ s2f[so.session])})

    def error(poses: list[F64]) -> F64:
        return np.array(
            [np.linalg.norm((np.linalg.inv(poses[c.i]) @ poses[c.j])[:3, 3] - c.rel[:3, 3]) for c in links]
        )

    before, after = error([k.pose for k in kfs]), error(solved)
    rooms = {s.id for s in sessions if s.session.type in (b.SessionType.SCAN, b.SessionType.LAP)}
    # A diagnostic for tuning (one more solve per session pair, ~30 % of the stage): on request only.
    stable = (
        align.stability(kfs, links, priors, unlifted, keep=lambda k: k.session in rooms, floor=floor)
        if stability
        else None
    )
    moved = [float(np.linalg.norm(a[:3, 3] - k.pose[:3, 3])) for a, k in zip(unlifted, kfs, strict=True)]
    summary: dict[str, object] = {
        "keyframes": len(kfs),
        "closures": len(links),
        "floor_links": len(floor),
        "closure_error_m": {
            "before_median": float(np.median(before)) if links else None,
            "after_median": float(np.median(after)) if links else None,
            "after_p90": float(np.percentile(after, 90)) if links else None,
        },
        "max_correction_m": max(moved, default=0.0),
        "leave_one_pair_out_max_m": max(stable.values(), default=0.0) if stable is not None else None,
        "anchors_max_move_m": anchors_moved,
        "closures_by_session_pair": by_pair,
        "sessions_without_closures": alone,
    }
    lopo = f"leave-one-pair-out {max(stable.values(), default=0.0):.3f} m, " if stable is not None else ""
    print(
        f"  pose graph: {len(kfs)} keyframes, {len(links)} closures, corrections up to {summary['max_correction_m']:.3f} m, "
        f"{lopo}anchors moved {anchors_moved:.3f} m"
    )
    return summary


def run(
    ingest_dir: Path, overrides_path: Path, out: Path, pose_graph: bool = False, stability: bool = False
) -> dict[str, object]:
    """`stability`: also the pose graph's leave-one-session-pair-out check (`leave_one_pair_out_max_m`)."""
    sessions = _load(ingest_dir)
    if not sessions:
        raise RuntimeError("no ingested sessions")
    ov = (
        Overrides.model_validate_json(overrides_path.read_text()) if overrides_path.is_file() else Overrides()
    )
    log: dict[str, object] = {}

    # 1. Merge sessions.
    first = sessions[0]
    icp: dict[str, float] = {}
    for i, s in enumerate(sessions[1:], 1):
        before = np.concatenate([apply(p.to_zup, p.points) for p in sessions[:i]])
        shared = next(
            (
                p
                for p in sessions[:i]
                if s.session.world_map_in is not None
                and s.session.world_map_in in (p.id, p.session.world_map_out)
            ),
            None,
        )
        if shared is not None:
            s.to_zup = shared.to_zup.copy()
        if s.session.world_map_in is None and s.points.size and before.size:
            s.to_zup = global_register(s.points, before)
            log[f"{s.id}.registered"] = "global (no world map)"
        if s.points.size and before.size:
            pts = apply(s.to_zup, s.points)
            rmse, corr, fitness = icp_check(pts, before, np.eye(4))
            shift = float(np.linalg.norm(corr[:3, 3]))
            icp[s.id] = rmse
            log[f"{s.id}.icp"] = {"rmse_m": rmse, "correction_m": shift, "overlap": fitness}
            if shift > 0.03:
                print(
                    f"  warn: session {s.id} disagrees with earlier sessions by {shift * 100:.1f} cm (drift?)"
                )

    all_pts = np.concatenate([apply(s.to_zup, s.points) for s in sessions if s.points.size])
    pts, normals = with_normals(all_pts)

    # 2. Floor.
    floor = ov.frame.floor_z if ov.frame.floor_z is not None else floor_height(pts, normals)
    if floor is None:
        floor = float(np.percentile(all_pts[:, 2], 1))
        print("  warn: no clear floor plane; using the 1st percentile of point heights")

    # 3. Yaw: dominant wall direction -> axis-aligned.
    angles, weights = _wall_angles(sessions)
    if angles.size == 0:
        angles, weights = _normal_angles(pts, normals)
    yaw = -dominant_axis_deg(angles, weights) if angles.size else 0.0

    # 4. Origin at the front door, the scan lying towards +y.
    doors = _doors(sessions)
    door: Door | None = None
    if ov.frame.origin_door:
        want = ov.frame.origin_door.lower()
        door = next((d for d in doors if want in (d.id, short_id("door", d.id))), None)
        if door is None:
            raise RuntimeError(f"overrides frame.origin_door {want!r} not found among {len(doors)} doors")
    elif doors and first.frames:
        start = apply(first.to_zup, mat(first.frames[0].camera_to_world)[None, :3, 3])[0]
        own = [d for d in doors if d.session == first.id] or doors
        door = min(own, key=lambda d: float(np.linalg.norm((d.centre - start)[:2])))
    if door is not None:
        centroid = np.mean([mat(f.camera_to_world)[:3, 3] for f in first.frames], axis=0)
        inward = door.normal * np.sign(np.dot((centroid - door.centre)[:2], door.normal[:2]) or 1.0)
        a = float(np.degrees(np.arctan2(inward[1], inward[0]))) + yaw  # inward angle after the snap
        yaw += 90.0 - a + float(angle_mod90(a - 90.0))  # nearest multiple of 90 that points it to +y
        origin = door.centre
    else:
        origin = mat(first.frames[0].camera_to_world)[:3, 3] if first.frames else np.zeros(3)
        print("  warn: no RoomPlan door; origin at the first camera position")
    yaw += ov.frame.yaw_offset_deg
    r = rot_z(yaw)
    o = apply(r, origin[None])[0]
    world_to_scene = translation((-o[0], -o[1], -floor)) @ r

    # 5. Write.
    out_frames: list[PosedImage] = []
    out_stills: list[PosedImage] = []
    sess_out: list[SessionPoses] = []
    for s in sessions:
        m = world_to_scene @ s.to_zup
        for src, dst in ((s.frames, out_frames), (s.stills, out_stills)):
            dst.extend(
                im.model_copy(update={"camera_to_world": to_mat4(m @ mat(im.camera_to_world))}) for im in src
            )
        sess_out.append(
            SessionPoses(
                session=s.id,
                raw_dir=rel(s.raw),
                session_to_scene=to_mat4(m @ ARKIT_TO_ZUP),
                icp_check_rms_m=icp.get(s.id),
            )
        )
    if pose_graph:
        log["pose_graph"] = _pose_graph(sessions, sess_out, out_frames, out_stills, stability)

    # Wall points per short window of one session, for structure's wall thickness (see points.wall_points).
    walls_dir = out / WALLS_DIR
    shutil.rmtree(walls_dir, ignore_errors=True)
    for s in sessions:
        own = sorted(
            (
                (im.t, im)
                for im in out_frames
                if im.session == s.id and im.t is not None and im.tracking in (None, b.TrackingState.NORMAL)
            ),
            key=lambda ti: ti[0],
        )
        if not own:
            continue
        t0 = own[0][0]
        for start in np.arange(t0, max(t0, own[-1][0] - WINDOW_S / 2) + 1e-9, WINDOW_S / 2):
            part = [im for t, im in own if start <= t < start + WINDOW_S]
            if len(part) >= 10:
                write_wall_points(walls_dir / f"{s.id}@{start - t0:04.0f}.npz", *wall_points(s.raw, part))
    poses = Poses(origin_door=door.id if door else None, yaw_deg=yaw, floor_z=floor, sessions=sess_out)
    write_json(out / "poses.json", poses)
    write_jsonl(out / "frames.jsonl", out_frames)
    write_jsonl(out / "stills.jsonl", out_stills)
    return {
        "sessions": [s.id for s in sessions],
        "origin_door": poses.origin_door,
        "yaw_deg": yaw,
        "floor_z": floor,
        **log,
    }
