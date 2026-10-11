"""`scanmodel bench`: the pipeline's geometric parts on captures with ground truth, so a rule change can be
checked for regressions on more than the one scan it was tuned on.

Two kinds of ground truth, neither ever fed into the model:

1. ARKitScenes videos (scanmodel.stages.arkitscenes): iPad LiDAR + ARKit poses, a Faro laser scan of the same
   room and hand-annotated object boxes. Each video runs ingest -> poses -> fuse (pose graph off and
   on) in data/derived/bench/work/ (shared by all projects), then:
   - cloud: the laser cloud is aligned to our fused points by one rigid transform (FPFH + RANSAC, then
     ICP). Accuracy: distance from each fused point to the laser (median, p95; beyond 20 cm counts as
     an outlier, e.g. where the laser saw nothing). Completeness: share of laser points in our cloud's
     bounding box with a fused point within 2 / 5 cm.
   - drift: our depth over 20 s windows (with our poses), ICP'd onto the aligned laser. The correction
     each window needs is the pose error the global alignment couldn't absorb; reported as the largest
     correction and the largest difference between two windows, horizontal and vertical separately.
     Windows whose surfaces can't pin all three axes (e.g. one wall only) are left out.
   - heights: scanmodel.rooms.fit_heights over the area the camera walked (+0.5 m), on our points and on the
     laser's: floor and ceiling differences, and whether a sloped ceiling was called on either.
   - boxes: each annotated box, every side and the top moved by 5-10 cm (RoomPlan-like), refitted with
     scanmodel.boxes.fit_to_points (+ fit_neighbours); mean side error before and after.
2. Checks on our own scans: `data/<project>/checks.json` (CheckFile, facts filled in by hand: relative
   floor heights, room sizes, ...), evaluated on the project's
   scene/model.json or on a fresh run of its raw/ (`--rerun`, no overrides): pass / fail.

Output: a table on stdout, `data/derived/bench/<date>.json` (paths repo-relative), and a line in
`history.jsonl` beside it.

  just bench                                   # ARKitScenes subset + every data/*/checks.json
  just bench --project fifth --no-arkitscenes  # one project's checks only (fast, reads its scene/model.json)
  just bench --project second --rerun          # re-run our pipeline on its raw/ first
"""

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import open3d as o3d
from pydantic import BaseModel, ConfigDict
from shapely import MultiPoint, Polygon, contains_xy

from .boxes import SIDES, Box, BoxParams, fit_neighbours, fit_to_points
from .config import Config, default_root, from_repo, load_config, rel
from .geom import ARKIT_TO_ZUP, apply
from .io import find_bundles, read_jsonl
from .models.common import F64, mat
from .models.derived import PosedImage, Poses
from .models.scene import Room, Scene
from .points import read_points, sample_points, to_o3d
from .rooms import RoomParams, ceiling_height, fit_heights
from .stages import arkitscenes as ark

# --------------------------------------------------------------------------- metrics (pure)


def _distances(a: F64, b: F64) -> F64:
    """Distance from each point of a to its nearest point of b."""
    if len(a) == 0:
        return np.zeros(0)
    if len(b) == 0:
        return np.full(len(a), np.inf)
    return np.asarray(to_o3d(a).compute_point_cloud_distance(to_o3d(b)))


def cloud_metrics(ours: F64, ref: F64, trunc: float = 0.20, margin: float = 0.05) -> dict[str, float]:
    """Accuracy of `ours` against `ref` and completeness of `ours` over `ref` (same frame)."""
    d = _distances(ours, ref)
    inl = d[d < trunc]
    lo, hi = ours.min(axis=0) - margin, ours.max(axis=0) + margin
    seen = ref[np.all((ref > lo) & (ref < hi), axis=1)]
    c = _distances(seen, ours)
    return {
        "accuracy_median_m": float(np.median(inl)) if inl.size else float("nan"),
        "accuracy_p95_m": float(np.percentile(inl, 95)) if inl.size else float("nan"),
        "outliers": float(np.mean(d >= trunc)) if d.size else float("nan"),
        "completeness_2cm": float(np.mean(c < 0.02)) if c.size else float("nan"),
        "completeness_5cm": float(np.mean(c < 0.05)) if c.size else float("nan"),
    }


def _fpfh(pc: o3d.geometry.PointCloud, voxel: float) -> o3d.pipelines.registration.Feature:
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    return o3d.pipelines.registration.compute_fpfh_feature(
        pc, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100), list(range(len(pc.points)))
    )


RELIABLE_FITNESS = 0.6  # below this share of our points matching the laser, drift numbers are doubtful


def register(src: F64, dst: F64, voxel: float = 0.05, tries: int = 3) -> tuple[F64, float, float]:
    """Rigid transform taking `src` onto `dst` (FPFH + RANSAC, then point-to-plane ICP at 5 / 2 / 1 cm).

    Returns (transform, ICP fitness at 1 cm, inlier rmse). The best of `tries` seeded RANSAC runs."""
    reg = o3d.pipelines.registration
    s = to_o3d(src).voxel_down_sample(voxel)
    t = to_o3d(dst).voxel_down_sample(voxel)
    fs, ft = _fpfh(s, voxel), _fpfh(t, voxel)
    fine_s = to_o3d(src).voxel_down_sample(0.01)
    fine_t = to_o3d(dst).voxel_down_sample(0.01)
    fine_t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
    best: tuple[F64, float, float] | None = None
    for k in range(tries):
        o3d.utility.random.seed(k)
        coarse = reg.registration_ransac_based_on_feature_matching(
            s,
            t,
            fs,
            ft,
            True,
            voxel * 1.5,
            reg.TransformationEstimationPointToPoint(False),
            3,
            [
                reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                reg.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
            ],
            reg.RANSACConvergenceCriteria(200_000, 0.999),
        )
        m = np.asarray(coarse.transformation)
        for dist in (0.10, 0.04, 0.02):
            m = np.asarray(
                reg.registration_icp(
                    fine_s, fine_t, dist, m, reg.TransformationEstimationPointToPlane()
                ).transformation
            )
        ev = reg.evaluate_registration(fine_s, fine_t, 0.02, m)
        if best is None or ev.fitness > best[1]:
            best = (m, float(ev.fitness), float(ev.inlier_rmse))
    assert best is not None
    return best


def constrained(normals: F64, min_eig: float = 0.1) -> bool:
    """Do these surface normals pin a rigid translation in all three axes (no sliding direction)?"""
    if len(normals) < 100:
        return False
    return float(np.linalg.eigvalsh(normals.T @ normals / len(normals))[0]) > min_eig


def window_correction(
    pts: F64, ref: F64, ref_nrm: F64, max_dist: float = 0.05, min_fitness: float = 0.5
) -> tuple[F64, float] | None:
    """Point-to-plane ICP of one time window's points onto the reference: (correction, fitness)."""
    reg = o3d.pipelines.registration
    s = to_o3d(pts).voxel_down_sample(0.02)
    s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    if not constrained(np.asarray(s.normals)):
        return None
    t = to_o3d(ref, ref_nrm)
    res = reg.registration_icp(s, t, max_dist, np.eye(4), reg.TransformationEstimationPointToPlane())
    if res.fitness < min_fitness:
        return None
    return np.asarray(res.transformation), float(res.fitness)


def drift_summary(corrections: list[F64]) -> dict[str, float | int]:
    """Largest window correction and largest difference between two windows (horizontal, vertical)."""
    if not corrections:
        return {"windows": 0}
    t = np.array([c[:3, 3] for c in corrections])
    rot = [float(np.degrees(np.arccos(np.clip((np.trace(c[:3, :3]) - 1) / 2, -1, 1)))) for c in corrections]
    dxy = np.linalg.norm(t[:, None, :2] - t[None, :, :2], axis=2)
    dz = np.abs(t[:, None, 2] - t[None, :, 2])
    return {
        "windows": len(corrections),
        "max_xy_m": float(np.linalg.norm(t[:, :2], axis=1).max()),
        "max_z_m": float(np.abs(t[:, 2]).max()),
        "spread_xy_m": float(dxy.max()),
        "spread_z_m": float(dz.max()),
        "max_rot_deg": max(rot),
    }


def walk_polygon(cameras_xy: F64, buffer_m: float = 0.5) -> Polygon:
    """The area the camera walked, grown by `buffer_m`: where floor and ceiling belong to this room."""
    hull = MultiPoint([tuple(p) for p in cameras_xy]).convex_hull
    return Polygon(hull.buffer(buffer_m).exterior)


def heights(
    polygon: Polygon, ours: tuple[F64, F64], ref: tuple[F64, F64], exclude: list[Polygon]
) -> dict[str, Any]:
    """fit_heights on our points and on the reference's, over the same outline. Also the dominant floor
    (densest 1 cm layer of upward points), to see what fit_heights' lowest-strong-cluster rule adds."""
    p = RoomParams()
    poly = [(float(x), float(y)) for x, y in list(polygon.exterior.coords)[:-1]]
    centre = polygon.centroid
    out: dict[str, Any] = {}
    for name, (pts, nrm) in (("ours", ours), ("ref", ref)):
        f, c, plane = fit_heights(poly, pts, nrm, p, exclude)
        out[name] = {
            "floor_z": f,
            "ceiling_z": c,
            "sloped": plane is not None,
            "ceiling_at_centre": None if c is None else ceiling_height(c, plane, centre.x, centre.y),
            "floor_dominant_z": dominant_floor(polygon, pts, nrm, exclude, p),
        }
    o, r = out["ours"], out["ref"]
    out["floor_dominant_err_m"] = (
        None
        if o["floor_dominant_z"] is None or r["floor_dominant_z"] is None
        else o["floor_dominant_z"] - r["floor_dominant_z"]
    )
    out["floor_err_m"] = None if o["floor_z"] is None or r["floor_z"] is None else o["floor_z"] - r["floor_z"]
    out["ceiling_err_m"] = (
        None if o["ceiling_z"] is None or r["ceiling_z"] is None else o["ceiling_z"] - r["ceiling_z"]
    )
    out["slope_agrees"] = o["sloped"] == r["sloped"]
    return out


def dominant_floor(
    polygon: Polygon, pts: F64, nrm: F64, exclude: list[Polygon], p: RoomParams
) -> float | None:
    """Mean height of the densest 1 cm layer of upward points within `floor_rise` of the room's lowest
    upward surface (inset 20 cm)."""
    inner = polygon.buffer(-0.20)
    m = nrm[:, 2] > 0.9
    m[m] = contains_xy(inner, pts[m, 0], pts[m, 1])
    for e in exclude:
        m[m] &= ~contains_xy(e.buffer(0.05), pts[m, 0], pts[m, 1])
    z = pts[m, 2]
    if z.size:
        lo = float(np.percentile(z, 1))
        z = z[(z >= lo - 0.01) & (z <= lo + p.floor_rise)]
    if z.size < p.min_points:
        return None
    h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.01))
    c = (e[h.argmax()] + e[h.argmax() + 1]) / 2
    return float(z[np.abs(z - c) < 0.01].mean())


def perturb(box: Box, rng: np.random.Generator, lo: float = 0.05, hi: float = 0.10) -> Box:
    """A copy with every side and the top moved in or out by lo..hi (RoomPlan's boxes are loose by cm)."""
    half = {s: max(0.02, v + rng.choice([-1, 1]) * rng.uniform(lo, hi)) for s, v in box.half.items()}
    z1 = box.z1 + rng.choice([-1, 1]) * rng.uniform(lo, hi)
    return Box(box.centre.copy(), box.yaw_deg, half, box.z0, max(z1, box.z0 + 0.05))


def side_errors(box: Box, truth: Box) -> F64:
    """|plane offset| per side (same yaw) and of the top, metres: [+x, -x, +y, -y, top]."""
    return np.array([abs(box.plane(s) - truth.plane(s)) for s in SIDES] + [abs(box.z1 - truth.z1)])


OPEN = {"table", "chair", "stool"}  # legs and a top: no side surfaces to fit, so no side error to fix


def box_metrics(
    truths: list[Box], pts: F64, nrm: F64, labels: list[str] | None = None, seeds: int = 3
) -> dict[str, Any]:
    """RoomPlan-like noise on the true boxes, refit, error before and after (over `seeds` draws).

    Side errors are reported for solid objects (cabinets, appliances, beds, sofas...); `OPEN` ones
    (tables, chairs, stools) only count for the top."""
    if not truths:
        return {"boxes": 0}
    labels = labels or ["box"] * len(truths)
    solid = np.array([lb not in OPEN for lb in labels] * seeds)
    before: list[F64] = []
    after: list[F64] = []
    p = BoxParams()
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        noisy = [perturb(t, rng) for t in truths]
        before += [side_errors(n, t) for n, t in zip(noisy, truths, strict=True)]
        for n in noisy:
            fit_to_points(n, pts, nrm, p)
        fit_neighbours(noisy, p)
        after += [side_errors(n, t) for n, t in zip(noisy, truths, strict=True)]
    b, a = np.array(before), np.array(after)
    out: dict[str, Any] = {
        "boxes": len(truths),
        "solid": int(solid.sum() // seeds),
        "labels": sorted(set(labels)),
        "top_mean_before_m": float(b[:, 4].mean()),
        "top_mean_after_m": float(a[:, 4].mean()),
    }
    if solid.any():
        bs, as_ = b[solid, :4], a[solid, :4]
        out |= {
            "side_mean_before_m": float(bs.mean()),
            "side_mean_after_m": float(as_.mean()),
            "side_p90_after_m": float(np.percentile(as_, 90)),
            "sides_worse": float(np.mean(as_ > bs + 0.01)),
            "sides_within_2cm": float(np.mean(as_ < 0.02)),
        }
    return out


# --------------------------------------------------------------------------- checks (own scans)


class Check(BaseModel):
    """One fact about a real building, measured or eyeballed by a person. Never an input to the model."""

    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["floor_delta", "ceiling_height", "room_width", "room_length"]
    room: str  # room name or id (or a unique part of the id)
    relative_to: str | None = None  # floor_delta: the other room
    expected_m: float
    tolerance_m: float
    source: str = ""  # how it was measured: "tape", "laser meter", "eyeballed"


class CheckFile(BaseModel):
    """`data/<project>/checks.json`: facts only. The scene and raw captures are the project's own."""

    model_config = ConfigDict(extra="forbid")
    name: str
    skip: list[str] = []  # sessions --rerun leaves out
    raw: str | None = (
        None  # --rerun: another bundle folder (repo-relative or absolute) than the project's raw/
    )
    checks: list[Check]


def find_room(scene: Scene, key: str) -> Room:
    k = key.lower()
    hits = [r for r in scene.rooms if r.id.lower() == k or (r.name or "").lower() == k]
    hits = hits or [r for r in scene.rooms if k in r.id.lower()]
    if len(hits) != 1:
        raise LookupError(f"room {key!r}: {len(hits)} matches among {[r.id for r in scene.rooms]}")
    return hits[0]


def _sides(room: Room) -> tuple[float, float]:
    """(short, long) side of the room outline's smallest enclosing rectangle (any orientation)."""
    hull = np.array(cast(Polygon, Polygon(room.polygon).convex_hull).exterior.coords)
    best: tuple[float, float, float] | None = None  # (area, short, long)
    for a, c in itertools.pairwise(hull):
        e = (c - a) / (np.linalg.norm(c - a) + 1e-12)
        u, v = hull @ e, hull @ np.array([-e[1], e[0]])
        w, h = float(np.ptp(u)), float(np.ptp(v))
        if best is None or w * h < best[0]:
            best = (w * h, min(w, h), max(w, h))
    assert best is not None
    return best[1], best[2]


def measure(scene: Scene, c: Check) -> float:
    room = find_room(scene, c.room)
    match c.kind:
        case "floor_delta":
            if c.relative_to is None:
                raise ValueError(f"{c.id}: floor_delta needs relative_to")
            return room.floor_z - find_room(scene, c.relative_to).floor_z
        case "ceiling_height":
            return room.ceiling_z - room.floor_z
        case "room_width":
            return _sides(room)[0]
        case "room_length":
            return _sides(room)[1]


@dataclass
class CheckResult:
    id: str
    expected_m: float
    tolerance_m: float
    measured_m: float | None
    ok: bool
    note: str = ""


def run_checks(scene: Scene, checks: list[Check]) -> list[CheckResult]:
    out: list[CheckResult] = []
    for c in checks:
        try:
            v = measure(scene, c)
        except (LookupError, ValueError) as e:
            out.append(CheckResult(c.id, c.expected_m, c.tolerance_m, None, False, str(e)))
            continue
        out.append(CheckResult(c.id, c.expected_m, c.tolerance_m, v, abs(v - c.expected_m) <= c.tolerance_m))
    return out


# --------------------------------------------------------------------------- running the pipeline


def _empty_overrides(work: Path) -> Path:
    """Bench runs never use overrides.json (hand fixes would hide what the rules do)."""
    return work / "no-overrides.json"  # does not exist: stages fall back to Overrides()


def run_pipeline(bundles: list[Path], work: Path, pose_graph: bool, structure_too: bool) -> dict[str, Any]:
    """ingest -> poses -> fuse (-> structure) into `work`, outside data/derived's own stages.
    Each stage's summary is written to work/log.json and returned."""
    from .stages import fuse, ingest, poses, structure

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    log: dict[str, Any] = {"ingest": ingest.run(bundles, work / "ingest")}
    log["poses"] = poses.run(work / "ingest", _empty_overrides(work), work / "poses", pose_graph)
    log["fuse"] = fuse.run(work / "poses", work / "fuse")
    if structure_too:
        log["structure"] = structure.run(
            work / "poses", work / "fuse", _empty_overrides(work), work / "model.json"
        )
    (work / "log.json").write_text(json.dumps(log, indent=2, default=str) + "\n")
    return log


@dataclass
class VideoResult:
    video: str
    why: str
    pose_graph: bool
    frames: int = 0
    registration: dict[str, float] = field(default_factory=dict[str, float])
    cloud: dict[str, float] = field(default_factory=dict[str, float])
    drift: dict[str, Any] = field(default_factory=dict[str, Any])
    heights: dict[str, Any] = field(default_factory=dict[str, Any])
    boxes: dict[str, Any] = field(default_factory=dict[str, Any])
    pose_graph_log: dict[str, Any] = field(default_factory=dict[str, Any])
    error: str | None = None


def bench_video(
    video: str, why: str, dest: Path, work: Path, pose_graph: bool, reuse: bool, window_s: float = 20.0
) -> VideoResult:
    res = VideoResult(video, why, pose_graph)
    bundle = ark.import_video(dest / "raw" / video, dest / "bundles")
    visit = next(v for vid, v, _, _ in ark.SUBSET if vid == video)
    run_dir = work / f"{video}-{'pg' if pose_graph else 'arkit'}"
    if not (reuse and (run_dir / "fuse" / "points.ply").is_file()):
        run_pipeline([bundle], run_dir, pose_graph, structure_too=False)
    poses = Poses.model_validate_json((run_dir / "poses" / "poses.json").read_text())
    log = json.loads((run_dir / "log.json").read_text())
    res.pose_graph_log = log["poses"].get("pose_graph", {})
    frames = [f for f in read_jsonl(run_dir / "poses" / "frames.jsonl", PosedImage) if f.depth]
    res.frames = len(frames)
    pts, nrm = read_points(run_dir / "fuse" / "points.ply")

    # Laser -> scene frame, one rigid transform.
    laser, laser_n = ark.laser_cloud(dest / "laser" / visit)
    m, fitness, rmse = register(laser, pts)
    # Again on the laser near our cloud only (the scans reach into other rooms and outdoors).
    near = np.all(
        np.abs(apply(m, laser) - (pts.min(axis=0) + pts.max(axis=0)) / 2) < np.ptp(pts, axis=0) / 2 + 0.5,
        axis=1,
    )
    m2, fitness2, rmse2 = register(laser[near], pts)
    if fitness2 > fitness:
        m, fitness, rmse = m2, fitness2, rmse2
    laser, laser_n = apply(m, laser), laser_n @ m[:3, :3].T
    res.registration = {"fitness_2cm": fitness, "rmse_m": rmse, "reliable": fitness >= RELIABLE_FITNESS}
    res.cloud = cloud_metrics(pts, laser)

    # Drift: per window, our depth with our poses vs the aligned laser.
    raw = from_repo(poses.sessions[0].raw_dir)
    ts = np.array([f.t or 0.0 for f in frames])
    corrections: list[F64] = []
    detail: list[list[float]] = []  # [seconds from start, dx, dy, dz] per pinned window
    for t0 in np.arange(ts.min(), ts.max() - window_s / 2, window_s):
        part = [f for f, t in zip(frames, ts, strict=True) if t0 <= t < t0 + window_s]
        if len(part) < 10:
            continue
        found = window_correction(sample_points(raw, part, n_frames=20, stride=2), laser, laser_n)
        if found is not None:
            corrections.append(found[0])
            detail.append([float(t0 - ts.min()), *map(float, found[0][:3, 3])])
    res.drift = {**drift_summary(corrections), "duration_s": float(np.ptp(ts)), "per_window": detail}

    # Heights and boxes, in the scene frame.
    to_scene = mat(poses.sessions[0].session_to_scene) @ np.linalg.inv(ARKIT_TO_ZUP)
    yaw = float(np.degrees(np.arctan2(to_scene[1, 0], to_scene[0, 0])))
    truths: list[Box] = []
    labels: list[str] = []
    for g in ark.read_boxes(dest / "raw" / video / f"{video}_3dod_annotation.json"):
        c = apply(to_scene, np.array([g.center]))[0]
        truths.append(Box.of((float(c[0]), float(c[1]), float(c[2])), g.size, g.yaw_deg + yaw))
        labels.append(g.label)
    cams = np.array([mat(f.camera_to_world)[:2, 3] for f in frames])
    footprints = [_footprint(t) for t in truths]
    res.heights = heights(walk_polygon(cams), (pts, nrm), (laser, laser_n), footprints)
    res.boxes = box_metrics(truths, pts, nrm, labels)
    return res


def _footprint(box: Box) -> Polygon:
    x, y = box.axes
    corners = [
        box.centre + sx * box.half["+x"] * x + sy * box.half["+y"] * y
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return Polygon([tuple(c) for c in corners])


def missing_scene(cfg: Config) -> str | None:
    """Why the project's checks can't be read from its scene (None if they can)."""
    if (cfg.scene_dir / "model.json").is_file():
        return None
    p = cfg.project
    return (
        f"{cfg.rel(cfg.scene_dir / 'model.json')} missing: run `just all --project {p}` first, or"
        f" `just bench --project {p} --rerun` (a fresh run of its raw/, no overrides)"
    )


def bench_checks(cfg: Config, path: Path, rerun: bool, pose_graph: bool) -> dict[str, Any]:
    cf = CheckFile.model_validate_json(path.read_text())
    if rerun:
        work = cfg.derived / "bench" / "work" / f"{cf.name}-{'pg' if pose_graph else 'arkit'}"
        raw = cfg.from_repo(cf.raw) if cf.raw else cfg.raw
        run_pipeline(find_bundles(raw, cf.skip), work, pose_graph, structure_too=True)
        scene_path = work / "model.json"
    else:
        scene_path = cfg.scene_dir / "model.json"
    scene = Scene.model_validate_json(scene_path.read_text())
    results = run_checks(scene, cf.checks)
    return {
        "name": cf.name,
        "project": cfg.project,
        "scene": cfg.rel(scene_path),
        "rerun": rerun,
        "pose_graph": pose_graph if rerun else None,
        "rooms": {
            r.id: {"name": r.name, "floor_z": r.floor_z, "ceiling_z": r.ceiling_z} for r in scene.rooms
        },
        "checks": [asdict(r) for r in results],
    }


# --------------------------------------------------------------------------- report


def _cm(v: Any, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v * 100:+.1f}" if signed else f"{v * 100:.1f}"


def _pct(v: Any) -> str:
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v * 100:.0f}%"


def table(videos: list[VideoResult], checks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    if videos:
        lines.append("ARKitScenes (cm unless %): acc = fused->laser, compl = laser covered within 5 cm,")
        lines.append("drift = largest 20 s window correction / spread between windows,")
        lines.append(
            f"reg = our points within 2 cm of the aligned laser ('?': below {RELIABLE_FITNESS:.0%}, drift doubtful)"
        )
        lines.append(
            "floor / ceil = fit_heights minus the same on the laser, fl_dom = densest floor layer instead"
        )
        lines.append("box = mean side error of solid objects before (box0) / after the fit, top = top error")
        head = (
            f"{'video':<9} {'poses':<6} {'reg':>4} {'acc50':>6} {'acc95':>6} {'compl':>6} "
            f"{'dxy':>5} {'dz':>5} {'sp_xy':>5} {'sp_z':>5} {'floor':>6} {'fl_dom':>6} {'ceil':>6} {'slope':>5} "
            f"{'box0':>5} {'box':>5} {'top0':>5} {'top':>5} {'worse':>5}"
        )
        lines += [head, "-" * len(head)]
        for r in videos:
            if r.error:
                lines.append(f"{r.video:<9} {'pg' if r.pose_graph else 'arkit':<6} ERROR {r.error}")
                continue
            h, d, c, bx = r.heights, r.drift, r.cloud, r.boxes
            slope = "ok" if h.get("slope_agrees") else f"{int(h['ours']['sloped'])}/{int(h['ref']['sloped'])}"
            reg = _pct(r.registration.get("fitness_2cm")) + ("" if r.registration.get("reliable") else "?")
            lines.append(
                f"{r.video:<9} {'pg' if r.pose_graph else 'arkit':<6} {reg:>4} "
                f"{_cm(c.get('accuracy_median_m')):>6} {_cm(c.get('accuracy_p95_m')):>6} {_pct(c.get('completeness_5cm')):>6} "
                f"{_cm(d.get('max_xy_m')):>5} {_cm(d.get('max_z_m')):>5} {_cm(d.get('spread_xy_m')):>5} {_cm(d.get('spread_z_m')):>5} "
                f"{_cm(h.get('floor_err_m'), True):>6} {_cm(h.get('floor_dominant_err_m'), True):>6} "
                f"{_cm(h.get('ceiling_err_m'), True):>6} {slope:>5} "
                f"{_cm(bx.get('side_mean_before_m')):>5} {_cm(bx.get('side_mean_after_m')):>5} "
                f"{_cm(bx.get('top_mean_before_m')):>5} {_cm(bx.get('top_mean_after_m')):>5} {_pct(bx.get('sides_worse')):>5}"
            )
        lines.append("")
    for cf in checks:
        lines.append(f"checks: {cf['name']} ({'re-run' if cf['rerun'] else cf['scene']})")
        for c in cf["checks"]:
            got = "-" if c["measured_m"] is None else f"{c['measured_m'] * 100:+.1f}"
            lines.append(
                f"  {'PASS' if c['ok'] else 'FAIL'}  {c['id']:<28} measured {got:>7} cm, expected "
                f"{c['expected_m'] * 100:+.1f} +- {c['tolerance_m'] * 100:.1f} cm {c['note']}"
            )
        lines.append("")
    return "\n".join(lines)


def _git(root: Path) -> dict[str, Any]:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()

    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "pipeline")),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scanmodel bench", description=(__doc__ or "").split("\n")[0])
    ap.add_argument(
        "--videos",
        nargs="*",
        help="ARKitScenes video ids (default: the subset in scanmodel.stages.arkitscenes)",
    )
    ap.add_argument("--no-arkitscenes", action="store_true")
    ap.add_argument(
        "--checks", type=Path, help="with --project: this check file instead of data/<project>/checks.json"
    )
    ap.add_argument(
        "--rerun",
        action="store_true",
        help="re-run our pipeline on the project's raw/ for the checks instead of reading its scene/model.json",
    )
    ap.add_argument("--pose-graph", choices=["on", "off", "both"], default="both")
    ap.add_argument(
        "--reuse", action="store_true", help="reuse earlier ARKitScenes runs in data/derived/bench/work"
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--project", help="only this project's checks (default: every data/<project>/ with a checks.json)"
    )
    a = ap.parse_args(argv)
    if a.checks is not None and not a.project:
        ap.error("--checks needs --project")
    root = default_root()
    bench_dir = root / "data" / "derived" / "bench"
    graphs = {"on": [True], "off": [False], "both": [False, True]}[a.pose_graph]

    # The projects' checks: (config, check file). Fail before the slow part when a named one can't run.
    jobs: list[tuple[Config, Path]] = []
    if a.project:
        cfg = load_config(root, project=a.project)
        path = a.checks or cfg.data / "checks.json"
        if not path.is_file():
            print(f"{cfg.rel(path)} missing: no checks for project {a.project}", file=sys.stderr)
            return 1
        if not a.rerun and (why := missing_scene(cfg)):
            print(why, file=sys.stderr)
            return 1
        jobs.append((cfg, path))
    else:
        for path in sorted((root / "data").glob("*/checks.json")):
            pcfg = load_config(root, project=path.parent.name)
            if not a.rerun and (why := missing_scene(pcfg)):
                print(f"skipped {pcfg.project}: {why}", file=sys.stderr)
                continue
            jobs.append((pcfg, path))

    videos: list[VideoResult] = []
    if not a.no_arkitscenes:
        dest = ark.default_dest(root)
        work = bench_dir / "work"
        why = {v: w for v, _, _, w in ark.SUBSET}
        for v in a.videos or list(why):
            if not (dest / "raw" / v).is_dir():
                print(
                    f"{v}: not downloaded (uv run python -m scanmodel.stages.arkitscenes download {v})",
                    file=sys.stderr,
                )
                continue
            for g in graphs:
                print(f"bench {v} ({'pose graph' if g else 'ARKit poses'})", flush=True)
                try:
                    videos.append(bench_video(v, why.get(v, ""), dest, work, g, a.reuse))
                except Exception as e:  # one broken capture shouldn't hide the others
                    videos.append(VideoResult(v, why.get(v, ""), g, error=f"{type(e).__name__}: {e}"))
    checks = [bench_checks(c, p, a.rerun, g) for c, p in jobs for g in (graphs if a.rerun else [True])]

    print(table(videos, checks))
    now = datetime.now()
    out = a.out or bench_dir / f"{now:%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "time": now.isoformat(timespec="seconds"),
        "git": _git(root),
        "arkitscenes": [asdict(v) for v in videos],
        "checks": checks,
    }
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    with (out.parent / "history.jsonl").open("a") as f:
        f.write(json.dumps(report, default=float) + "\n")
    print(f"wrote {rel(out, root)}")
    failed = sum(not c["ok"] for cf in checks for c in cf["checks"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
