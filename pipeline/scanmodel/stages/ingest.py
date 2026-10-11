"""Stage 1, ingest: check each bundle in data/<project>/raw and convert its poses to pipeline axes.

Output per session in data/<project>/derived/ingest/<session>/:
  report.json   IngestReport (errors fail the stage; warnings are for a look)
  frames.jsonl  PosedImage per stream frame, z-up gravity-aligned world, OpenCV camera axes
  stills.jsonl  PosedImage per posed still

Frames from before tracking settled at a session's start (`checks.settled_from`) are left out, and so are
pose excursions (`checks.excursions`) and the frames of overrides.json `drop_frames`, with the stills shot
during them; report.json `dropped` lists what went.
"""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import ValidationError

from ..checks import EXCURSION_MAX_S, depth_consistency, excursions, pose_jumps, settled_from
from ..config import rel
from ..geom import arkit_to_pipeline, is_rigid
from ..io import Bundle, VideoReader, iter_jsonl_lines, write_json, write_jsonl
from ..models import bundle as b
from ..models.common import mat, to_mat4
from ..models.derived import DroppedFrames, IngestReport, PosedImage
from ..stillcal import corrected, still_focal_scale


class IngestError(RuntimeError):
    pass


def _parse_lines[M: b.Frame | b.Still | b.AnchorEvent](
    path: Path, model: type[M], errors: list[str]
) -> list[M]:
    out: list[M] = []
    if not path.is_file():
        return out
    for n, line in iter_jsonl_lines(path):
        try:
            out.append(model.model_validate_json(line))
        except ValidationError as e:
            errors.append(f"{path.name}:{n}: {e.errors()[0]['loc']} {e.errors()[0]['msg']}")
            if len(errors) > 20:
                break
    return out


def _dropped(
    frames: list[b.Frame],
    stills: list[b.Still],
    start: int,
    drop_frames: list[tuple[int, int]],
    warnings: list[str],
) -> list[DroppedFrames]:
    """Excursions after the settled start (`start`), then the hand-picked idx ranges."""
    out: list[DroppedFrames] = []

    def window(first: int, last: int, **kw: object) -> DroppedFrames:
        t0, t1 = frames[first].t, frames[last].t
        return DroppedFrames.model_validate(
            {
                "idx0": frames[first].idx,
                "idx1": frames[last].idx,
                "t0": t0,
                "t1": t1,
                "frames": last - first + 1,
                "stills": [s.id for s in stills if s.t is not None and t0 <= s.t <= t1],
                **kw,
            }
        )

    for e in excursions(frames[start:]):
        w = window(start + e.first, start + e.last, reason="excursion", step_m=e.out_m, step_deg=e.out_deg)
        if e.dropped:
            out.append(w)
        else:
            warnings.append(
                f"pose excursion at idx {w.idx0}-{w.idx1} ({w.t1 - w.t0:.1f} s, {e.out_m * 100:.0f} cm out and"
                f" back) longer than {EXCURSION_MAX_S:g} s: kept (overrides.json drop_frames leaves it out)"
            )
    idx = np.array([f.idx for f in frames])
    for i0, i1 in drop_frames:
        hit = np.flatnonzero((idx >= i0) & (idx <= i1))
        if hit.size:
            out.append(window(int(hit[0]), int(hit[-1]), reason="override"))
        else:
            warnings.append(f"overrides.json drop_frames [{i0}, {i1}]: no frame in that range")
    return out


def _kept[T: b.Frame | b.Still](images: list[T], dropped: list[DroppedFrames]) -> list[T]:
    """`images` minus the frames (by idx) and stills (by time) in a dropped window."""

    def out(im: b.Frame | b.Still) -> bool:
        if isinstance(im, b.Frame):
            return any(w.idx0 <= im.idx <= w.idx1 for w in dropped)
        return im.t is not None and any(w.t0 <= im.t <= w.t1 for w in dropped)

    return [im for im in images if not out(im)] if dropped else images


def check_bundle(
    d: Path, drop_frames: list[tuple[int, int]] | None = None
) -> tuple[IngestReport, Bundle | None]:
    """Check a bundle; `drop_frames`: idx ranges (inclusive) to leave out, from overrides.json."""
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    try:
        session = b.Session.model_validate_json((d / b.SESSION_FILE).read_text())
    except (OSError, ValidationError) as e:
        return IngestReport(session=d.name, raw_dir=rel(d), ok=False, errors=[f"session.json: {e}"]), None

    frames = _parse_lines(d / b.FRAMES_FILE, b.Frame, errors)
    stills = _parse_lines(d / b.STILLS_FILE, b.Still, errors)
    anchors = _parse_lines(d / b.ANCHORS_FILE, b.AnchorEvent, errors)
    counts |= {"frames": len(frames), "stills": len(stills), "anchor_events": len(anchors)}
    if session.sweeps:
        counts["sweeps"] = len(session.sweeps)
    is_ar = session.type != b.SessionType.ULTRAWIDE

    # Frames: ordering, timing, tracking, poses, files.
    if session.type in (b.SessionType.SCAN, b.SessionType.LAP) and not frames:
        errors.append("no frames")
    elif not frames and not stills:
        errors.append("no frames and no stills")
    idx = np.array([f.idx for f in frames])
    t = np.array([f.t for f in frames])
    dropped: list[DroppedFrames] = []
    if frames:
        if np.any(np.diff(idx) <= 0):
            errors.append("frame idx not strictly increasing")
        if np.any(np.diff(t) <= 0):
            errors.append("frame timestamps not strictly increasing")
        fps = session.video.fps if session.video else 60.0
        gaps = int(np.sum(np.diff(t) > 3.0 / fps))
        counts["timestamp_gaps"] = gaps
        if gaps:
            warnings.append(f"{gaps} gaps longer than 3 frame intervals")
        bad_pose = [f.idx for f in frames if not is_rigid(mat(f.camera_to_world))]
        if bad_pose:
            errors.append(f"{len(bad_pose)} frames with a non-rigid pose, first idx {bad_pose[0]}")
        settled = settled_from(frames, fps)
        counts["unsettled_frames"] = settled
        if settled:
            warnings.append(
                f"first {settled} frames ({t[settled] - t[0]:.1f} s) left out: tracking hadn't settled"
            )
        dropped = _dropped(frames, stills, settled, drop_frames or [], warnings)
        for reason in ("excursion", "override"):
            counts[f"{reason}_frames"] = sum(w.frames for w in dropped if w.reason == reason)
        for w in dropped:
            why = (
                f"pose excursion, {(w.step_m or 0.0) * 100:.0f} cm out and back"
                if w.reason == "excursion"
                else "overrides.json drop_frames"
            )
            with_stills = (
                f" and still{'s' * (len(w.stills) > 1)} {', '.join(map(str, w.stills))}" if w.stills else ""
            )
            warnings.append(f"frames {w.idx0}-{w.idx1} ({w.t1 - w.t0:.1f} s){with_stills} left out: {why}")
        # Jumps left in: one-way shifts (drift, for the pose graph) or relocalisations.
        jumps = [j for j in pose_jumps(frames) if not any(w.idx0 <= j <= w.idx1 for w in dropped)]
        counts["pose_jumps"] = len(jumps)
        if jumps:
            warnings.append(f"{len(jumps)} pose jumps (relocalisation?), first at idx {jumps[0]}")
        for f in frames[:1]:
            k = f.intrinsics
            if not (0 < k.cx < k.width and 0 < k.cy < k.height):
                errors.append(f"principal point outside the image: {k}")
    normal = sum(f.tracking == b.TrackingState.NORMAL for f in frames)
    tracking_fraction = normal / len(frames) if frames else None
    if tracking_fraction is not None and tracking_fraction < 0.9:
        warnings.append(f"tracking normal for only {tracking_fraction:.0%} of frames")
    lost_stills = sum(s.tracking not in (None, b.TrackingState.NORMAL) for s in stills)
    if lost_stills:
        warnings.append(f"{lost_stills} stills shot without normal tracking")
    # A session that loaded a world map but never tracked normally never relocalised: its poses are in
    # a frame of their own and must not be merged.
    if (
        session.world_map_in is not None
        and (frames or stills)
        and normal == 0
        and not any(s.tracking == b.TrackingState.NORMAL for s in stills)
    ):
        errors.append(
            f"never relocalised against {session.world_map_in}: poses aren't in the project's frame"
        )

    depth_frames = [f for f in frames if f.depth]
    counts["depth_frames"] = len(depth_frames)
    missing = [f.depth for f in depth_frames if f.depth and not (d / f.depth).is_file()]
    missing += [f.confidence for f in depth_frames if f.confidence and not (d / f.confidence).is_file()]
    if missing:
        errors.append(f"{len(missing)} depth/confidence files missing, e.g. {missing[0]}")
    if is_ar and session.depth is not None and not depth_frames:
        errors.append("session.json declares depth but no frame references a depth map")
    if depth_frames and not missing and depth_frames[0].depth:
        with Image.open(d / depth_frames[0].depth) as im:
            if session.depth and im.size != (session.depth.width, session.depth.height):
                errors.append(f"depth size {im.size} != session.json {session.depth}")
            if im.mode not in ("I;16", "I;16B", "I;16L", "I"):
                errors.append(f"depth PNG mode {im.mode}, expected 16-bit")

    if is_ar and not stills:
        warnings.append("no stills (textures and splat will use video frames only)")
    video = d / b.VIDEO_FILE
    if session.type == b.SessionType.SCAN and not video.is_file():
        warnings.append("no video.mov (splat stage will have stills only)")
    elif video.is_file() and frames:
        reader = VideoReader(video)
        n_video = reader.frame_count()
        reader.close()
        counts["video_frames"] = n_video
        if n_video and int(idx.max()) >= n_video:
            errors.append(f"frame idx {int(idx.max())} beyond video length {n_video}")

    # Stills.
    frame_idx = set(idx.tolist())
    for s in stills:
        p = d / s.file
        if not p.is_file():
            errors.append(f"still {s.id}: missing {s.file}")
            continue
        if s.intrinsics is not None:
            with Image.open(p) as im:
                if im.size != (s.intrinsics.width, s.intrinsics.height):
                    errors.append(f"still {s.id}: image {im.size} != intrinsics size")
        if is_ar and s.source != b.StillSource.ULTRAWIDE and s.camera_to_world is None:
            errors.append(f"still {s.id}: no pose")
        if s.camera_to_world is not None and not is_rigid(mat(s.camera_to_world)):
            errors.append(f"still {s.id}: non-rigid pose")
        if s.nearest_frame is not None and s.nearest_frame not in frame_idx:
            warnings.append(f"still {s.id}: nearest_frame {s.nearest_frame} not in frames.jsonl")

    if session.type == b.SessionType.SCAN:
        for name in (b.ROOM_JSON_FILE, b.MESH_FILE):
            if not (d / name).is_file():
                warnings.append(f"no {name}")

    bundle = Bundle(dir=d, session=session, frames=frames, stills=stills)
    consistency = None
    if not errors and depth_frames:
        consistency = depth_consistency(bundle, lambda f: arkit_to_pipeline(mat(f.camera_to_world)))
        if consistency is not None and consistency > 0.03:
            warnings.append(f"depth disagrees between nearby frames by {consistency * 100:.1f} cm (poses?)")

    report = IngestReport(
        session=session.session_id,
        raw_dir=rel(d),
        ok=not errors,
        errors=errors,
        warnings=warnings,
        counts=counts,
        tracking_normal_fraction=tracking_fraction,
        depth_consistency_m=consistency,
        dropped=dropped,
    )
    return report, (bundle if not errors else None)


def posed_images(
    bundle: Bundle, still_scale: float = 1.0, dropped: list[DroppedFrames] | None = None
) -> tuple[list[PosedImage], list[PosedImage]]:
    """Frames and stills in pipeline axes; stills' focal lengths times `still_scale` (see scanmodel.stillcal).
    Frames and stills from before tracking settled (`checks.settled_from`) are left out, and those in a
    `dropped` window (the report's: excursions, overrides.json `drop_frames`)."""
    sid = bundle.id
    start = settled_from(bundle.frames, bundle.session.video.fps if bundle.session.video else 60.0)
    t0 = bundle.frames[start].t if start else None
    frames = [
        PosedImage(
            id=f"{sid}/{f.idx}",
            session=sid,
            kind="frame",
            file=f"{b.VIDEO_FILE}#{f.idx}",
            t=f.t,
            camera_to_world=to_mat4(arkit_to_pipeline(mat(f.camera_to_world))),
            intrinsics=f.intrinsics,
            tracking=f.tracking,
            depth=f.depth,
            confidence=f.confidence,
        )
        for f in _kept(bundle.frames[start:], dropped or [])
    ]
    stills = [
        PosedImage(
            id=f"{sid}/s{s.id}",
            session=sid,
            kind="still",
            file=s.file,
            t=s.t,
            camera_to_world=to_mat4(arkit_to_pipeline(mat(s.camera_to_world))),
            intrinsics=corrected(s.intrinsics, still_scale),
            tracking=s.tracking,
            source=s.source,
        )
        for s in _kept(bundle.stills, dropped or [])
        if s.camera_to_world is not None
        and s.intrinsics is not None
        and (t0 is None or s.t is None or s.t >= t0)
    ]
    return frames, stills


def run(
    raw_dirs: list[Path], out: Path, drop_frames: dict[str, list[tuple[int, int]]] | None = None
) -> dict[str, object]:
    """`drop_frames`: overrides.json's, session folder -> frame idx ranges to leave out."""
    drop_frames = drop_frames or {}
    if not raw_dirs:
        raise IngestError("no bundles to ingest (expected data/<project>/raw/<session>/session.json)")
    summary: dict[str, object] = {}
    failed: list[str] = []
    for d in raw_dirs:
        report, bundle = check_bundle(d, drop_frames.get(d.name))
        sdir = out / report.session
        write_json(sdir / "report.json", report)
        entry: dict[str, object] = {
            "ok": report.ok,
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            **{k: report.counts.get(k, 0) for k in ("frames", "depth_frames", "stills")},
            **{k: n for k in ("excursion_frames", "override_frames") if (n := report.counts.get(k))},
        }
        if bundle is not None:
            if bundle.session.notes:
                print(f"  [{report.session}] notes {bundle.session.notes}")
            cal = still_focal_scale(bundle.dir, bundle.frames, bundle.stills)
            if cal.stills:
                res = f", residual {cal.residual_px:.1f} px" if cal.residual_px is not None else ""
                print(f"  [{report.session}] stills' focal x{cal.scale:.4f} from {cal.stills} stills{res}")
                entry["still_focal_scale"] = round(cal.scale, 5)
            frames, stills = posed_images(bundle, cal.scale, report.dropped)
            write_jsonl(sdir / "frames.jsonl", frames)
            write_jsonl(sdir / "stills.jsonl", stills)
        else:
            failed.append(report.session)
        summary[report.session] = entry
        for e in report.errors:
            print(f"  [{report.session}] ERROR {e}")
        for w in report.warnings:
            print(f"  [{report.session}] warn  {w}")
    if unknown := sorted(set(drop_frames) - {d.name for d in raw_dirs}):
        print(f"  overrides.json drop_frames: no session {', '.join(unknown)} ingested")
    # Later stages read every session in `out`: drop ones that were moved, deleted or skipped.
    for stale in out.glob("*/report.json"):
        if stale.parent.name not in summary:
            shutil.rmtree(stale.parent)
    if failed:
        raise IngestError(f"bundle check failed for: {', '.join(failed)} (see {out}/*/report.json)")
    return summary
