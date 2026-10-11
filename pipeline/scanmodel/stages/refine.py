"""Stage 2b, refine: bundle-adjust the poses on image features matched across passes (`scanmodel.refine`).

Reads derived/poses/ (poses.json, frames.jsonl, stills.jsonl) and ingest's raw ARKit poses (for the
relative motion), writes derived/refine/ in the same format. Every stage after it reads these poses
unless `--poses posegraph` (or settings.toml `poses = "posegraph"`) asks for the pose graph's alone:

  poses.json, frames.jsonl, stills.jsonl   every frame and still with its corrected pose; frames of a
                                           session's first `settle_drop_s` after an unsettled start are
                                           left out (ARKit's relocalised world still settling), and those
                                           inside an admitted trajectory cut's stretch (their poses are a
                                           guess); poses.json: `session_to_scene` composed with the move of
                                           a session the coarse round moved as a block
  walls/<session>@<s>.npz                  wall points per 30 s window at the refined poses (as poses/ has
                                           them at the pose graph's), for structure's wall thickness
  refine.json                              what was matched and how much it moved: views, pairs, tracks,
                                           reprojection error before/after per distance band (cross-pass
                                           and same-pass tracks) and session, corrections per session;
                                           `floor`: floor-cell residuals, the offset between sessions'
                                           floors over shared cells, the camera height change as a plane;
                                           `session_pairs`: accepted / rejected cross pairs per session
                                           pair and the rejected ones' median offset (`drift` > 10 cm);
                                           `coarse`: the coarse round (its pair table, the bins that agree
                                           on drift, the drift spans, whether applied, its corrections)
  refine.png                               correction per view over time, error before/after per band
  features/<session>.npz                   SIFT cache (kept between runs)
"""

import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw

from .. import align, progress
from .. import refine as rf
from ..cache import console
from ..config import from_repo
from ..io import read_jsonl, write_jsonl
from ..models import bundle as b
from ..models.common import F64, mat, to_mat4
from ..models.derived import PosedImage, Poses
from ..points import WALLS_DIR, WINDOW_S, wall_points, write_wall_points
from ..roomplan import load_captured_room

SETTLE_DROP_S = 3.0
MIN_TRACKS = 200  # fewer usable tracks than this and the adjustment has too little to go on

OPT_OUT = (
    "run with `--poses posegraph` (or poses = 'posegraph' in settings.toml) to use the pose graph's poses"
)


class RefineError(RuntimeError):
    """Refine can't run on this capture (no video, too few tracks): the message says why and how to opt out."""


def _raw_poses(ingest_dir: Path) -> dict[str, F64]:
    """id -> ARKit's own camera_to_world (session z-up world; only relative motion is used)."""
    out: dict[str, F64] = {}
    for rep in sorted(ingest_dir.glob("*/report.json")):
        for name in ("frames.jsonl", "stills.jsonl"):
            for im in read_jsonl(rep.parent / name, PosedImage):
                out[im.id] = mat(im.camera_to_world)
    return out


def _unsettled_start(frames: list[PosedImage]) -> dict[str, float]:
    """Session -> time of its first kept frame, for sessions whose first frames ingest left out
    (`scanmodel.checks.settled_from`: the first kept frame isn't the video's first)."""
    first: dict[str, PosedImage] = {}
    for f in frames:
        if f.t is not None and (f.session not in first or f.t < (first[f.session].t or 0.0)):
            first[f.session] = f
    return {s: f.t or 0.0 for s, f in first.items() if int(f.file.split("#")[1]) > 0}


def run(
    poses_dir: Path,
    ingest_dir: Path,
    out: Path,
    settle_drop_s: float = SETTLE_DROP_S,
    readout_s: float = 0.0,
    floor_sigma: float = rf.FLOOR_SIGMA,
) -> dict[str, object]:
    started = time.monotonic()
    poses = Poses.model_validate_json((poses_dir / "poses.json").read_text())
    raw = {s.session: from_repo(s.raw_dir) for s in poses.sessions}
    frames = read_jsonl(poses_dir / "frames.jsonl", PosedImage)
    stills = read_jsonl(poses_dir / "stills.jsonl", PosedImage)
    raw_pose = _raw_poses(ingest_dir)
    if no_video := sorted(
        s for s in {f.session for f in frames if f.depth} if not (raw[s] / b.VIDEO_FILE).is_file()
    ):
        raise RefineError(f"refine: no {b.VIDEO_FILE} in session {', '.join(no_video)}; {OPT_OUT}")

    # Sessions that started unsettled: their first seconds go (the world was still moving under them).
    unsettled = _unsettled_start(frames)
    cut = {s: t + settle_drop_s for s, t in unsettled.items()} if settle_drop_s > 0 else {}

    def kept(im: PosedImage) -> bool:
        return im.t is None or im.session not in cut or im.t >= cut[im.session]

    dropped = [im.id for im in [*frames, *stills] if not kept(im)]
    frames = [f for f in frames if kept(f)]
    stills = [s for s in stills if kept(s)]

    span = {
        s: (min(f.t or 0.0 for f in fs), max(f.t or 0.0 for f in fs)) for s, fs in _by_session(frames).items()
    }
    sessions = sorted((s for s in poses.sessions if s.session in span), key=lambda s: span[s.session][0])
    traj = align.trajectories(
        [
            (
                s.session,
                b.Session.model_validate_json((raw[s.session] / b.SESSION_FILE).read_text()),
                *span[s.session],
            )
            for s in sessions
        ]
    )
    views = rf.select_views(frames, stills, traj, readout_s=readout_s)
    console.print(
        f"  {len(views)} views ({sum(v.still for v in views)} stills) from {len(frames)} frames;"
        f" dropped {len(dropped)} settling frames/stills",
        highlight=False,
    )
    t0 = time.monotonic()
    with progress.bar("views") as bar:
        task = bar.add_task("features", total=len(views))
        feats = rf.features(views, raw, out / "features", lambda n: bar.advance(task, n))
    t_feat = time.monotonic() - t0
    uvs = [rf.rs_corrected(f.uv, v, readout_s) for f, v in zip(feats, views, strict=True)]
    loose = {s: t + 5.0 for s, t in unsettled.items()}

    def is_loose(v: rf.View) -> bool:
        return v.image.session in loose and v.t < loose[v.image.session]

    odo = rf.odometry(views, lambda im: raw_pose.get(im.id), is_loose)

    # Coarse round: drift beyond what the fine round's windows and gate allow (see scanmodel.refine).
    initial = [v.pose for v in views]  # the pose graph's: corrections are reported and applied against these
    t0 = time.monotonic()
    coarse_cands = rf.coarse_pairs(views, feats)
    with progress.bar("pairs") as bar:
        task = bar.add_task("coarse", total=len(coarse_cands))
        coarse_matches = rf.match_all(
            views, feats, uvs, coarse_cands, lambda n: bar.advance(task, n), rf.COARSE_WINDOW_M,
            (rf.LOOSE_M, rf.LOOSE_DEG),
        )  # fmt: skip
    first_t = {s: min(v.t for v in views if v.image.session == s) for s in {v.image.session for v in views}}
    # Trajectory cuts first: drift that later sessions of the same ARSession inherited (see scanmodel.refine).
    cuts, joined = rf.trajectory_cuts(views, coarse_matches, frame_gaps(frames, traj))
    cut_rows = [_cut_row(c, traj, first_t) for c in cuts]
    if any(c.admitted for c in cuts):
        for v, p in zip(views, joined, strict=True):
            v.pose = p
        coarse_cands = rf.coarse_pairs(views, feats)  # matched again at the joined poses
        with progress.bar("pairs") as bar:
            task = bar.add_task("coarse (joined)", total=len(coarse_cands))
            coarse_matches = rf.match_all(
                views, feats, uvs, coarse_cands, lambda n: bar.advance(task, n), rf.COARSE_WINDOW_M,
                (rf.LOOSE_M, rf.LOOSE_DEG),
            )  # fmt: skip
    groups = rf.coarse_groups(views, coarse_matches)
    coarse = coarse_report(views, coarse_cands, coarse_matches, groups)
    coarse["cuts"] = cut_rows
    coarse["applied"] = bool(coarse["applied"]) or any(c.admitted for c in cuts)
    if coarse["applied"]:
        # ARKit's relative motion is known wrong over each drift span and cut: loosened there for the bundle
        # adjustment too, or it would pull the drifted views back towards ARKit
        spans = [*rf.cut_spans(views, cuts), *rf.drift_spans(views, coarse_matches, groups)]
        odo, coarse["odometry_loosened"] = rf.stretch_odometry(views, odo, spans)
        coarse["spans"] = [
            {"session": sp.session, "trajectory": sp.trajectory,
             "t_s": [round(sp.t0 - first_t[sp.session], 1), round(sp.t1 - first_t[sp.session], 1),
                     round(sp.t_end - first_t[sp.session], 1)],
             "offset_mm": [round(1000 * float(x), 1) for x in sp.offset], "rot_deg": round(sp.rot_deg, 2)}
            for sp in spans
        ]  # fmt: skip
        start, coarse["pairs_solved"] = rf.coarse_solve(views, odo, coarse_matches, groups, spans)
        coarse["correction"] = _corrections(views, initial, start)
        for v, p in zip(views, start, strict=True):
            v.pose = p
    t_coarse = time.monotonic() - t0
    for line in coarse_lines(coarse):
        console.print(line, highlight=False)

    voxels = rf.view_voxels(views, feats)
    cands = rf.pairs(views, voxels)
    t0 = time.monotonic()
    with progress.bar("pairs") as bar:
        task = bar.add_task("matching", total=len(cands))
        matches = rf.match_all(views, feats, uvs, cands, lambda n: bar.advance(task, n))
    t_match = time.monotonic() - t0
    good = [m for m in matches if len(m.ia)]
    tracks = rf.build_tracks(feats, good)
    obs = rf.observations(feats, uvs, tracks)
    points, keep_track, obs_ok = rf.init_landmarks(views, obs, tracks.n)
    useful = rf.useful_tracks(views, obs, tracks.n)
    use = obs_ok & keep_track[obs.track] & useful[obs.track]
    # a track needs two observations left
    left = np.bincount(obs.track[use], minlength=tracks.n)
    use &= left[obs.track] >= 2
    if (n_tracks := len(np.unique(obs.track[use]))) < MIN_TRACKS:
        raise RefineError(
            f"refine: {n_tracks} usable tracks from {len(good)}/{len(cands)} matched pairs ({len(views)} views),"
            f" fewer than {MIN_TRACKS}; {OPT_OUT}"
        )
    cross = rf.cross_tracks(views, obs, tracks.n)
    t0 = time.monotonic()
    floor = floor_observations(poses, raw, views) if floor_sigma > 0 else rf.Floor(*_no_floor())
    t_floor = time.monotonic() - t0
    console.print(
        f"  {len(good)}/{len(cands)} pairs matched ({sum(m.kind == 'cross' for m in good)} cross-pass),"
        f" {len(np.unique(obs.track[use]))} tracks, {int(use.sum())} observations"
        f" ({int((use & cross[obs.track]).sum())} on cross-pass tracks), {len(odo)} odometry links,"
        f" {len(floor)} floor observations of {len(floor.keys)} cells",
        highlight=False,
    )
    before = rf.residual_stats(views, initial, obs, use, cross)
    t0 = time.monotonic()
    first = rf.adjust(views, obs, points, use, odo, floor=floor, floor_sigma=floor_sigma)
    err, _ = rf.reprojection(views, first.poses, obs, first.points)
    sigma = np.array([v.sigma for v in views])
    inl = use & (err < rf.OUTLIER_PX * sigma[obs.view] / rf.PIXEL_SIGMA)
    left = np.bincount(obs.track[inl], minlength=tracks.n)
    inl &= left[obs.track] >= 2
    second = rf.adjust(
        views, obs, first.points, inl, odo, start=first.poses, floor=floor, floor_sigma=floor_sigma
    )
    t_ba = time.monotonic() - t0
    # The main floor back at its height (the scene frame's z = 0): the adjustment's vertical gauge is weak.
    lift = rf.main_floor_lift(rf.floor_heights(initial, floor), rf.floor_heights(second.poses, floor), floor)
    if lift:
        for p in second.poses:
            p[2, 3] += lift
        second.points[:, 2] += lift
    after = rf.residual_stats(views, second.poses, obs, inl, cross)
    before_inl = rf.residual_stats(views, initial, obs, inl, cross)
    err2, _ = rf.reprojection(views, second.poses, obs, second.points)

    # Corrections to every frame and still: interpolated between the frame views in time.
    kfs = [
        align.Keyframe(v.image.session, v.trajectory, v.t, p, v.image)
        for v, p in zip(views, initial, strict=True)
        if not v.still
    ]
    solved = [p for p, v in zip(second.poses, views, strict=True) if not v.still]
    correct = align.corrector(kfs, solved)
    still_pose = {v.image.id: p for p, v in zip(second.poses, views, strict=True) if v.still}
    out_frames: list[PosedImage] = []
    out_stills: list[PosedImage] = []
    # Inside an admitted cut's stretch the poses are a guess (the jump spread evenly over what nothing
    # anchored): those frames and stills are left out, like an unsettled start's
    guessed = [(c.trajectory, c.t0, c.t1) for c in cuts if c.admitted]

    def in_cut(im: PosedImage) -> bool:
        return im.t is not None and any(traj.get(im.session) == k and t0 < im.t < t1 for k, t0, t1 in guessed)

    cut_dropped = [im.id for im in [*frames, *stills] if in_cut(im)]
    for src, dst in ((frames, out_frames), (stills, out_stills)):
        for im in src:
            if in_cut(im):
                continue
            pose = mat(im.camera_to_world)
            if im.id in still_pose:
                new = still_pose[im.id]
            elif im.t is not None and im.session in traj:
                new = correct(im.session, traj[im.session], im.t, pose)
            else:
                new = pose
            dst.append(im.model_copy(update={"camera_to_world": to_mat4(new)}))

    moved = [rf.correction(a, c) for a, c in zip(initial, second.poses, strict=True)]
    corrections = _corrections(views, initial, second.poses)
    pairs_by_session = rf.session_pairs(views, matches)
    drifted = [k for k, r in pairs_by_session.items() if r.get("drift")]
    for k in drifted:
        r = pairs_by_session[k]
        console.print(
            f"  [yellow]warning[/yellow]: {k}: {r['rejected']} rejected pairs agree on an offset of"
            f" {cast(float, r['rejected_offset_mm']) / 10:.0f} cm (share {r['rejected_agree']}): drift left"
            " in the poses?",
            highlight=False,
        )
    obs_per_view = np.bincount(obs.view[inl], minlength=len(views))
    summary: dict[str, object] = {
        "views": len(views),
        "stills": sum(v.still for v in views),
        "settle_dropped": len(dropped),
        "cut_dropped": len(cut_dropped),
        "unsettled_sessions": sorted(unsettled),
        "pairs": {
            "candidates": len(cands),
            "matched": len(good),
            "cross_matched": sum(m.kind == "cross" for m in good),
            "cross_candidates": sum(m.kind == "cross" for m in matches),
        },
        "tracks": len(np.unique(obs.track[inl])),
        "cross_pass_tracks": len(np.unique(obs.track[inl & cross[obs.track]])),
        "observations": int(inl.sum()),
        "outliers_dropped": int((use & ~inl).sum()),
        "views_without_observations": int((obs_per_view == 0).sum()),
        "odometry_links": len(odo),
        "iterations": [first.iterations, second.iterations],
        "ba_error": [round(first.error_before, 1), round(first.error_after, 1), round(second.error_after, 1)],
        "ba_reprojection_px": {
            "median": round(float(np.median(err2[inl])), 3),
            "p90": round(float(np.percentile(err2[inl], 90)), 3),
        },
        "residual_px_before": before_inl,
        "residual_px_after": after,
        "residual_px_before_all_matches": before,
        "correction": corrections,
        "coarse": coarse,
        "session_pairs": pairs_by_session,
        "drift_warnings": drifted,
        "readout_s": readout_s,
        "floor_sigma_m": floor_sigma,
        "floor_lift_mm": round(1000 * lift, 1),
        "floor": rf.floor_report(views, initial, second.poses, floor),
        "seconds": {
            "features": round(t_feat, 1),
            "coarse": round(t_coarse, 1),
            "floor": round(t_floor, 1),
            "matching": round(t_match, 1),
            "adjust": round(t_ba, 1),
            "total": round(time.monotonic() - started, 1),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    if coarse["applied"]:
        # The coarse round moved whole stretches: RoomPlan's rooms (`session_to_scene`) move with their session
        poses, summary["session_moved"] = moved_sessions(poses, views, initial, second.poses, cuts)
        (out / "poses.json").write_text(poses.model_dump_json(indent=1) + "\n")
    else:
        shutil.copyfile(poses_dir / "poses.json", out / "poses.json")
    write_jsonl(out / "frames.jsonl", out_frames)
    write_jsonl(out / "stills.jsonl", out_stills)
    write_walls(out / WALLS_DIR, raw, out_frames)
    per_view = [
        {"id": v.image.id, "t": round(v.t, 3), "trans_mm": round(1000 * m[0], 2), "rot_deg": round(m[1], 4),
         "observations": int(obs_per_view[n])}
        for n, (v, m) in enumerate(zip(views, moved, strict=True))
    ]  # fmt: skip
    (out / "refine.json").write_text(
        json.dumps(
            {**summary, "dropped": dropped, "cut_dropped": cut_dropped, "per_view": per_view}, indent=1
        )
        + "\n"
    )
    plot(out / "refine.png", views, moved, before_inl, after)
    a, c = before_inl["cross_pass"], after["cross_pass"]  # type: ignore[index]
    console.print(
        "  cross-pass reprojection px (median) by band, before -> after: "
        + ", ".join(
            f"{k} {a[k]['median']} -> {c[k]['median']}"  # type: ignore[index]
            for k in a  # type: ignore[union-attr]
            if a[k] and c[k]  # type: ignore[index]
        ),
        highlight=False,
    )
    fl = summary["floor"]
    console.print(
        f"  floor: {fl['observations']} observations, residual median {fl['residual_mm_before']} ->"  # type: ignore[index]
        f" {fl['residual_mm_after']} mm, session offsets max {fl['session_offset_max_mm']} mm,"  # type: ignore[index]
        f" camera dz plane {fl['camera_dz_plane']}, lift {summary['floor_lift_mm']} mm",  # type: ignore[index]
        highlight=False,
    )
    tables = (
        "residual_px_before",
        "residual_px_after",
        "residual_px_before_all_matches",
        "floor",
        "session_pairs",
    )

    def medians(r: dict[str, object], group: str) -> dict[str, object]:
        return {k: v["median"] for k, v in r[group].items() if v}  # type: ignore[union-attr, index]

    return {
        **{k: v for k, v in summary.items() if k not in tables},
        "cross_pass_px_before": medians(before_inl, "cross_pass"),
        "cross_pass_px_after": medians(after, "cross_pass"),
        "same_pass_px_before": medians(before_inl, "same_pass"),
        "same_pass_px_after": medians(after, "same_pass"),
        "floor_session_offset_max_mm": fl["session_offset_max_mm"],  # type: ignore[index]
        "floor_camera_dz_plane": fl["camera_dz_plane"],  # type: ignore[index]
        "coarse": {k: v for k, v in coarse.items() if k != "session_pairs"},
    }


def _corrections(views: list[rf.View], before: list[F64], after: list[F64]) -> dict[str, dict[str, float]]:
    """Per session: how far the views moved from `before` to `after` (mm, degrees)."""
    by_s: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for v, a, c in zip(views, before, after, strict=True):
        by_s[v.image.session].append(rf.correction(a, c))
    return {
        s: {
            "views": len(ms),
            "trans_mm_median": round(1000 * float(np.median([m[0] for m in ms])), 1),
            "trans_mm_p90": round(1000 * float(np.percentile([m[0] for m in ms], 90)), 1),
            "trans_mm_max": round(1000 * float(max(m[0] for m in ms)), 1),
            "rot_deg_median": round(float(np.median([m[1] for m in ms])), 3),
            "rot_deg_p90": round(float(np.percentile([m[1] for m in ms], 90)), 3),
        }
        for s, ms in sorted(by_s.items())
    }


def coarse_report(
    views: list[rf.View],
    cands: list[tuple[int, int, str]],
    matches: list[rf.PairMatches],
    groups: list[rf.Group],
) -> dict[str, object]:
    """refine.json's `coarse`: the coarse round's pairs, its per-session-pair table (`rf.session_pairs`:
    "rejected" there are the pairs beyond the fine round's gate, what the fine round would have thrown
    away), and every bin that passed the consensus tests (`drift`; `admitted`: in a run of RUN_BINS)."""
    return {
        "pairs": len(cands),
        "matched": sum(bool(len(m.ia)) for m in matches),
        "beyond_fine_gate": sum(
            bool(len(m.ia)) and (m.dev_m > rf.GATE_M or m.dev_deg > rf.GATE_DEG) for m in matches
        ),
        "applied": any(g.admitted for g in groups),
        "pairs_solved": 0,
        "drift": [
            {
                "session": g.session,
                "t_s": [round(g.t0, 1), round(g.t0 + rf.BIN_S, 1)],
                "loose": g.loose,
                "tight": g.tight,
                "agree": g.agree,
                "offset_mm": [round(1000 * float(x), 1) for x in g.offset],
                "other_sessions": g.sessions,
                "spread_deg": round(g.spread_deg, 1),
                "admitted": g.admitted,
            }
            for g in groups
            if g.ok or g.admitted
        ],
        "session_pairs": rf.session_pairs(views, matches),
    }


def coarse_lines(coarse: dict[str, object]) -> list[str]:
    """Console lines for the coarse round: drift found per session (time range, offset), whether applied.
    Bins that pass the tests but no run are counted only: mostly the other side of a drifted session's
    pairs (its partners' cameras look off by minus its offset), or a local alias."""
    drift = cast(list[dict[str, Any]], coarse["drift"])
    adm = [d for d in drift if d["admitted"]]
    out = [
        f"  coarse: {coarse['matched']}/{coarse['pairs']} pairs matched under the loose gate,"
        f" {coarse['beyond_fine_gate']} beyond the fine gate; {len(adm)} bins drifted"
        f" ({len(drift) - len(adm)} more agree on an offset, not admitted)"
    ]
    for c in cast(list[dict[str, Any]], coarse.get("cuts", [])):
        if c["admitted"]:
            out.append(
                f"  [yellow]warning[/yellow]: trajectory {c['trajectory']} jumped {np.linalg.norm(c['jump_mm']) / 10:.0f}"
                f" cm / {c['jump_yaw_deg']:+.1f} deg between {c['from']} and {c['to']} s ({c['agree']} crossing pairs"
                f" agree, {c['tight']} against): the views after it are moved back"
            )
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in adm:
        by[d["session"]].append(d)
    for s, ds in by.items():
        off = max(float(np.linalg.norm(d["offset_mm"])) for d in ds)
        out.append(
            f"  [yellow]warning[/yellow]: {s} drifted: {len(ds)} bins from t {ds[0]['t_s'][0]:.0f} to"
            f" {ds[-1]['t_s'][1]:.0f} s agree on offsets up to {off / 10:.0f} cm"
        )
    if coarse["applied"]:
        corr = cast(dict[str, dict[str, float]], coarse["correction"])
        moved = ", ".join(f"{s} {c['trans_mm_max'] / 10:.0f} cm" for s, c in corr.items())
        out.append(
            f"  coarse alignment applied ({coarse['pairs_solved']} pairs, {coarse['odometry_loosened']} ARKit"
            f" links loosened over {coarse['spans']}), views moved up to: {moved}"
        )
    return out


def _no_floor() -> tuple[np.ndarray, np.ndarray, F64, list[tuple[int, int]]]:
    return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3)), []


def floor_observations(poses: Poses, raw: dict[str, Path], views: list[rf.View]) -> rf.Floor:
    """The views' floor cells (`align.floors`, with every session's RoomPlan furniture and stairs left out,
    as the pose graph does) -> floor observations."""
    blocked = align.footprints(
        [
            (room, mat(s.session_to_scene))
            for s in poses.sessions
            if s.session in raw
            and (room := load_captured_room(raw[s.session] / b.ROOM_JSON_FILE)) is not None
        ]
    )
    idx = rf.floor_views(views)
    cells: dict[int, dict[tuple[int, int], float]] = {}
    by: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        by[views[i].image.session].append(i)
    for sid, ids in by.items():
        kfs = [
            align.Keyframe(sid, views[i].trajectory, views[i].t, views[i].pose, views[i].image) for i in ids
        ]
        for i, c in zip(ids, align.floors(raw[sid], kfs, blocked), strict=True):
            if c:
                cells[i] = c
    return rf.floor_observations(views, cells, min_dt=align.MIN_DT, cell_size=align.FLOOR_CELL)


def write_walls(walls_dir: Path, raw: dict[str, Path], frames: list[PosedImage]) -> None:
    """Wall points per `WINDOW_S` window (overlapping by half) of one session, at these frames' poses
    (`scanmodel.points.wall_points`): `<session>@<seconds from its first frame>.npz`, as `scanmodel poses` writes them."""
    shutil.rmtree(walls_dir, ignore_errors=True)
    for sid, fs in sorted(_by_session(frames).items()):
        own = sorted(
            ((f.t, f) for f in fs if f.t is not None and f.tracking in (None, b.TrackingState.NORMAL)),
            key=lambda tf: tf[0],
        )
        if not own:
            continue
        t0 = own[0][0]
        for start in np.arange(t0, max(t0, own[-1][0] - WINDOW_S / 2) + 1e-9, WINDOW_S / 2):
            part = [f for t, f in own if start <= t < start + WINDOW_S]
            if len(part) >= 10:
                write_wall_points(walls_dir / f"{sid}@{start - t0:04.0f}.npz", *wall_points(raw[sid], part))


def moved_sessions(
    poses: Poses,
    views: list[rf.View],
    before: list[F64],
    after: list[F64],
    cuts: list[rf.Cut] = [],  # noqa: B006
) -> tuple[Poses, dict[str, dict[str, object]]]:
    """`session_to_scene` of each session whose frame views moved as a block by more than `rf.DRIFT_M` (or 1
    deg), composed with that move (`rf.rigid_fit`, medians). A session an admitted cut splits is fitted per
    part: its largest part gives `session_to_scene`, the others that moved differently go to
    `session_to_scene_parts` (RoomPlan kept mapping what it saw after the jump in the jumped frame). Only after
    the coarse round moved something; a few cm of refinement leave RoomPlan where it was."""
    moved: dict[str, dict[str, object]] = {}
    sessions = []
    for sp in poses.sessions:
        idx = [n for n, v in enumerate(views) if v.image.session == sp.session and not v.still]
        if not idx:
            sessions.append(sp)
            continue
        bounds = sorted(c.t1 for c in cuts if c.admitted and c.trajectory == views[idx[0]].trajectory)
        parts = [p for p in np.split(idx, np.searchsorted([views[n].t for n in idx], bounds)) if len(p) >= 20]
        parts = sorted(parts or [np.array(idx)], key=len, reverse=True)
        fits = [rf.rigid_fit([before[n] for n in p], [after[n] for n in p]) for p in parts]

        def size(m: F64, part: np.ndarray) -> tuple[float, float]:
            centre = np.mean([before[n][:3, 3] for n in part], axis=0)
            return float(np.linalg.norm(m[:3, :3] @ centre + m[:3, 3] - centre)), float(
                np.degrees(np.arctan2(m[1, 0], m[0, 0]))
            )

        shift, yaw = size(fits[0], parts[0])
        main = fits[0] if shift > rf.DRIFT_M or abs(yaw) > 1.0 else np.eye(4)
        others = []
        for m, part in zip(fits[1:], parts[1:], strict=True):
            d_shift, d_yaw = size(m @ np.linalg.inv(main), part)
            if d_shift > rf.DRIFT_M or abs(d_yaw) > 1.0:
                others.append(to_mat4(m @ mat(sp.session_to_scene)))
        if main is not fits[0] and not others:
            sessions.append(sp)
            continue
        sp = sp.model_copy(
            update={
                "session_to_scene": to_mat4(main @ mat(sp.session_to_scene)),
                "session_to_scene_parts": others,
            }
        )
        moved[sp.session] = {
            "shift_mm": round(1000 * shift, 1),
            "yaw_deg": round(yaw, 2),
            "parts": 1 + len(others),
        }
        sessions.append(sp)
    return poses.model_copy(update={"sessions": sessions}), moved


def frame_gaps(frames: list[PosedImage], traj: dict[str, int]) -> list[tuple[int, float, float]]:
    """(trajectory, last frame time before, first after) for every gap of `rf.FRAME_GAP_S` or more in a
    session's frames with normal tracking: windows ingest dropped (excursions, `drop_frames`), tracking lost."""
    out: list[tuple[int, float, float]] = []
    for sid, fs in _by_session(frames).items():
        if sid not in traj:
            continue
        ts = np.sort([f.t for f in fs if f.t is not None and f.tracking in (None, b.TrackingState.NORMAL)])
        out += [
            (traj[sid], float(ts[i]), float(ts[i + 1])) for i in np.flatnonzero(np.diff(ts) >= rf.FRAME_GAP_S)
        ]
    return sorted(out)


def _cut_row(c: rf.Cut, traj: dict[str, int], first_t: dict[str, float]) -> dict[str, object]:
    """refine.json's `coarse.cuts` row: where (session and seconds from its start), the vote, the jump."""
    own = [s for s in first_t if traj.get(s) == c.trajectory] or list(first_t)

    def at(t: float) -> str:
        s = max((s for s in own if first_t[s] <= t + 1e-6), key=lambda s: first_t[s], default=None)
        s = s if s is not None else min(own, key=lambda s: first_t[s])
        return f"{s}@{t - first_t[s]:.1f}"

    a = float(np.degrees(np.arctan2(c.jump[1, 0], c.jump[0, 0])))
    return {
        "trajectory": c.trajectory, "kind": c.kind, "from": at(c.t0), "to": at(c.t1),
        "loose": c.loose, "tight": c.tight, "agree": c.agree, "bins": [c.bins_before, c.bins_after],
        "sessions_after": c.sessions_after, "spread_deg": round(c.spread_deg, 1),
        "offset_mm": [round(1000 * float(x), 1) for x in c.offset],
        "jump_mm": [round(1000 * float(x), 1) for x in c.jump[:3, 3]], "jump_yaw_deg": round(a, 2),
        "gain": c.gain, "admitted": c.admitted,
    }  # fmt: skip


def _by_session(frames: list[PosedImage]) -> dict[str, list[PosedImage]]:
    out: dict[str, list[PosedImage]] = defaultdict(list)
    for f in frames:
        out[f.session].append(f)
    return out


def plot(
    path: Path,
    views: list[rf.View],
    moved: list[tuple[float, float]],
    before: dict[str, object],
    after: dict[str, object],
) -> None:
    """Left: correction (mm) per view over time, a colour per session. Right: median reprojection error
    (px at 960) per distance band before and after, cross-pass and same-pass tracks."""
    w, h, pad = 1400, 520, 60
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    colours = ["#2e86c1", "#c0392b", "#27ae60", "#8e44ad", "#d35400", "#16a085"]
    sessions = sorted({v.image.session for v in views})
    # timeline: sessions laid end to end
    order = sorted(range(len(views)), key=lambda i: (views[i].trajectory, views[i].image.session, views[i].t))
    x0, x1, y0, y1 = pad, w // 2 - 20, pad, h - pad
    ymax = max(10.0, float(np.percentile([1000 * m[0] for m in moved], 99)) * 1.1) if moved else 10.0
    d.rectangle((x0, y0, x1, y1), outline="#999")
    d.text((x0, 15), "correction per view (mm), views in time order", fill="black")
    for k in range(0, int(ymax) + 1, max(1, int(ymax // 5))):
        y = y1 - (y1 - y0) * k / ymax
        d.line((x0 - 4, y, x0, y), fill="#999")
        d.text((x0 - 40, y - 6), f"{k}", fill="#555")
    n = max(1, len(order) - 1)
    for rank, i in enumerate(order):
        x = x0 + (x1 - x0) * rank / n
        y = y1 - (y1 - y0) * min(1000 * moved[i][0], ymax) / ymax
        c = colours[sessions.index(views[i].image.session) % len(colours)]
        r = 3 if views[i].still else 1
        d.ellipse((x - r, y - r, x + r, y + r), fill=c)
    for s, name in enumerate(sessions):
        d.text((x0 + 10, y0 + 8 + 14 * s), name, fill=colours[s % len(colours)])
    # residual bars
    x0, x1 = w // 2 + 40, w - 20
    d.rectangle((x0, y0, x1, y1), outline="#999")
    d.text((x0, 15), "reprojection error median (px at 960): before (light) / after (dark)", fill="black")
    bands = [f"{lo:g}-{hi:g}m" for lo, hi in rf.BANDS]
    groups = [("cross_pass", "#c0392b", "#f1948a"), ("same_pass", "#2e86c1", "#85c1e9")]
    vals: list[float] = []
    for g, _, _ in groups:
        for src in (before, after):
            for bnd in bands:
                s = src[g][bnd]  # type: ignore[index]
                if s:
                    vals.append(float(s["median"]))  # type: ignore[index]
    vmax = max([1.0, *vals]) * 1.1
    slot = (x1 - x0) / len(bands)
    bw = slot / 6
    for bi, bnd in enumerate(bands):
        cx = x0 + slot * bi + slot / 2
        d.text((cx - 25, y1 + 8), bnd, fill="#555")
        for gi, (g, dark, light) in enumerate(groups):
            for si, (src, col) in enumerate(((before, light), (after, dark))):
                s = src[g][bnd]  # type: ignore[index]
                if not s:
                    continue
                v = float(s["median"])  # type: ignore[index]
                bx = cx + (gi * 2 + si - 2) * bw
                d.rectangle((bx, y1 - (y1 - y0) * v / vmax, bx + bw - 2, y1), fill=col)
                d.text((bx, y1 - (y1 - y0) * v / vmax - 12), f"{v:.1f}", fill="#333")
    d.text((x0 + 10, y0 + 8), "red: cross-pass tracks, blue: same-pass", fill="#333")
    img.save(path)
