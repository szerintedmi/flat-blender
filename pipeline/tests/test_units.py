"""Unit tests: RoomPlan parsing, splat export, CLI wiring."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scanmodel.config import load_config
from scanmodel.io import find_bundles
from scanmodel.models import bundle as b
from scanmodel.models.bundle import Frame
from scanmodel.models.common import to_mat4
from scanmodel.models.scene import Fit
from scanmodel.roomplan import matrix_from_simd, parse_captured_room
from scanmodel.stages.structure import (
    Face,
    Params,
    SessionPoints,
    fit_line,
    measure_thickness,
    merge_patch,
    snap_face,
)

from . import synth


def test_simd_matrix_single_list_or_nested() -> None:
    m = np.arange(16, dtype=float).reshape(4, 4)
    cols = m.T.tolist()  # simd: 4 columns
    np.testing.assert_array_equal(matrix_from_simd(cols), m)
    np.testing.assert_array_equal(matrix_from_simd([v for col in cols for v in col]), m)


def test_captured_room_enum_shapes() -> None:
    wall = {
        "identifier": "A-B",
        "category": {"wall": {}},
        "dimensions": [3, 2.4, 0],
        "transform": np.eye(4).reshape(-1).tolist(),
        "confidence": {"high": {}},
    }
    door = {
        **wall,
        "identifier": "C",
        "category": {"door": {"_0": {"isOpen": False}}},
        "parentIdentifier": "A-B",
    }
    obj = {"identifier": "D", "category": "storage", "dimensions": [1, 1, 1], "transform": np.eye(4).tolist()}
    room = parse_captured_room({"walls": [wall], "doors": [door], "objects": [obj]})
    assert room.of("wall")[0].id == "a-b"
    assert room.of("door")[0].is_open is False and room.of("door")[0].parent == "a-b"
    assert room.objects[0].category == "storage"


def test_merge_patch() -> None:
    assert merge_patch({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"c": None, "e": 4}}) == {
        "a": 1,
        "b": {"d": 3, "e": 4},
    }


def _settings(root: Path, extra: str = "", tools: str = "") -> None:
    """settings.toml of a test repo: project p, no Blender."""
    (root / "settings.toml").write_text(
        f'project = "p"\n{extra}[tools]\nblender = "{root / "no-blender"}"\n{tools}'
    )


def _cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """`scanmodel ARGS` run in `root` (a repo skeleton with its settings.toml)."""
    return subprocess.run(
        [sys.executable, "-m", "scanmodel.cli", *args], cwd=root, text=True, capture_output=True, check=False
    )


def test_cli_all_then_cached_then_splat(repo: Path) -> None:
    tmp_path = repo
    _settings(tmp_path)
    cfg = load_config(tmp_path)
    synth.write_two_rooms(cfg.raw)
    # Refined poses are the default, and refine needs the video: it stops with how to opt out, no fallback.
    r = _cli(tmp_path, "all", "--no-build")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "no video.mov" in r.stdout and "--poses posegraph" in r.stdout, r.stdout
    assert "Traceback" not in r.stdout + r.stderr
    assert not (cfg.scene_dir / "model.json").exists()
    assert not (cfg.stage_dir("refine") / "manifest.json").exists()
    r = _cli(tmp_path, "all", "--no-build", "--poses", "posegraph")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (cfg.scene_dir / "model.json").is_file()
    _settings(tmp_path, 'poses = "posegraph"\n')
    r = _cli(tmp_path, "all", "--no-build")
    _settings(tmp_path)
    assert r.stdout.count("up to date") == 5, r.stdout
    r = _cli(tmp_path, "splat")  # the default asks for refined poses, which this capture can't have
    assert r.returncode != 0 and "run `scanmodel refine` first" in " ".join((r.stdout + r.stderr).split())
    r = _cli(tmp_path, "splat", "--poses", "posegraph")
    assert r.returncode == 0, r.stdout + r.stderr
    t = json.loads((cfg.dataset_dir() / "transforms.json").read_text())
    assert len(t["frames"]) == 4  # 2 stills per session; no video in the synthetic bundles
    assert np.allclose(np.array(t["frames"][0]["transform_matrix"])[3], [0, 0, 0, 1])
    assert [f["file_path"] for f in t["frames"]] == sorted(f["file_path"] for f in t["frames"])

    # --train with a stand-in for msplat's Python (the interpreter on msplat-train's first line), which the
    # pipeline runs with scanmodel/msplat_train.py: check the arguments, the summary, and the cache.
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json, sys\na = sys.argv[1:]\n"
        "open(a[a.index('--output') + 1], 'w').write('ply\\nelement vertex 3\\nend_header\\n')\n"
        "print(json.dumps(a)); print('PSNR:  21.5'); print('SSIM:  0.8')\n"
    )
    fake_python.chmod(0o755)
    fake = tmp_path / "fake-msplat"
    fake.write_text(f"#!{fake_python}\n")
    fake.chmod(0o755)
    _settings(tmp_path, 'poses = "posegraph"\n', f'msplat = "{fake}"\n')
    r = _cli(tmp_path, "splat", "--train", "-n", "50", "--trainer-arg=--sh-degree=1")
    assert r.returncode == 0, r.stdout + r.stderr
    out = cfg.runs_dir / "msplat"
    args = json.loads((out / "train.log").read_text().splitlines()[0])
    assert args[0].endswith("msplat_train.py") and args[args.index("--num-iters") + 1] == "50"
    assert Path(args[args.index("--views") + 1]).name == "views.npz" and args[-1] == "--sh-degree=1"
    summary = json.loads((out / "manifest.json").read_text())["summary"]
    assert summary["splats"] == 3 and summary["psnr"] == 21.5 and summary["ssim"] == 0.8
    r = _cli(tmp_path, "splat", "--train", "-n", "50", "--trainer-arg=--sh-degree=1")
    assert r.stdout.count("up to date") == 2, r.stdout
    assert "msplat  3 splats" in _cli(tmp_path, "status").stdout


def test_committed_schemas_are_current(tmp_path: Path) -> None:
    """scene/schema.json etc. are generated from the models; regenerate with `scanmodel schema`."""
    from scanmodel.schema import write_schemas

    repo = Path(__file__).resolve().parents[2]
    for generated in write_schemas(tmp_path):
        committed = repo / generated.relative_to(tmp_path)
        assert committed.read_text() == generated.read_text(), f"{committed} is stale: run `scanmodel schema`"


def test_snap_keeps_a_long_fitted_skew() -> None:
    def face(length: float, deg: float, fitted: bool) -> Face:
        d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        fit = Fit(inliers=1000, rms_m=0.003) if fitted else None
        return Face("w", "s", np.zeros(2), length * d, 0.0, 2.5, fit=fit)

    def angle(f: Face) -> float:
        return float(np.degrees(np.arctan2(f.dir[1], f.dir[0])))

    p = Params()
    long_fitted, short_fitted, long_roomplan = (
        face(4.8, 1.2, True),
        face(0.8, 1.2, True),
        face(4.8, 1.2, False),
    )
    for f in (long_fitted, short_fitted, long_roomplan):
        snap_face(f, p)
    assert angle(long_fitted) == pytest.approx(1.2)  # snapping would move its ends 5 cm
    assert angle(short_fitted) == pytest.approx(0.0, abs=1e-9)  # 0.8 cm: within snap_tol_m
    assert angle(long_roomplan) == pytest.approx(0.0, abs=1e-9)  # not fitted: RoomPlan's angle isn't measured


def test_find_bundles_reads_session_and_project_folders(tmp_path: Path) -> None:
    for d in (tmp_path / "s1", tmp_path / "proj" / "s2", tmp_path / "proj" / "s3"):
        d.mkdir(parents=True)
        (d / "session.json").write_text("{}")
    (tmp_path / "proj" / "project.json").write_text("{}")
    assert [d.name for d in find_bundles(tmp_path)] == ["s2", "s3", "s1"]
    assert [d.name for d in find_bundles(tmp_path, ["s2"])] == ["s3", "s1"]


def test_ingest_selects_a_folder_and_drops_stale_sessions(repo: Path) -> None:
    tmp_path = repo
    _settings(tmp_path)
    cfg = load_config(tmp_path)
    raw = cfg.raw
    synth.write_two_rooms(raw / "second")
    synth.write_one_session(raw / "first")
    ingested = cfg.stage_dir("ingest")

    def sessions() -> list[str]:
        return sorted(p.parent.name for p in ingested.glob("*/report.json"))

    r = _cli(tmp_path, "ingest")
    assert r.returncode == 0, r.stdout + r.stderr
    assert sessions() == ["s01-hall", "s01-home", "s02-living"]
    r = _cli(tmp_path, "ingest", "--raw", str(raw / "second"), "--skip", "s02-living")
    assert r.returncode == 0, r.stdout + r.stderr
    assert sessions() == ["s01-hall"]
    r = _cli(tmp_path, "ingest", "--raw", "data/p/raw/second", "--skip", "s01-hall")
    assert r.returncode == 0, r.stdout + r.stderr
    assert sessions() == ["s02-living"]


def _wall_face(
    rng: np.random.Generator, x: float, facing: float, n: int = 4000
) -> tuple[np.ndarray, np.ndarray]:
    """Points on the plane x = `x`, 0 < y < 3, 0 < z < 2.4, normals along +x * facing."""
    pts = np.column_stack(
        [np.full(n, x) + rng.normal(0, 0.002, n), rng.uniform(0, 3, n), rng.uniform(0, 2.4, n)]
    )
    return pts, np.tile([facing, 0.0, 0.0], (n, 1))


def test_thickness_comes_from_one_session_not_from_drift() -> None:
    rng = np.random.default_rng(0)
    # Face along y at x = 0, room on +x. The wall is 12 cm thick: its back face is at x = -0.12.
    a = Face("w", "s1", np.array([0.0, 3.0]), np.array([0.0, 0.0]), 0.0, 2.4)
    assert np.allclose(a.interior, [1, 0])
    front, fn = _wall_face(rng, 0.0, 1.0)
    back, bn = _wall_face(rng, -0.12, -1.0)
    # s1 saw both faces; s2 saw only the back face and drifted 6 cm (its copy sits at -0.18).
    drifted, dn = _wall_face(rng, -0.18, -1.0)
    both = SessionPoints({"s1": (np.vstack([front, back]), np.vstack([fn, bn])), "s2": (drifted, dn)})
    found = measure_thickness(a, 0.0, 3.0, both, Params())
    assert found is not None and abs(found[0] - 0.12) < 0.005 and set(found[1]) == {"s1"}, found
    # Nobody saw both faces: no measurement (structure then uses the default), however far apart.
    apart = SessionPoints({"s1": (front, fn), "s2": (drifted, dn)})
    assert measure_thickness(a, 0.0, 3.0, apart, Params()) is None
    # A door in the wall: its strip is left out, the rest still measures.
    holed = SessionPoints(both.points, holes=[(np.array([0.0, 1.0]), np.array([0.0, 1.9]))])
    found = measure_thickness(a, 0.0, 3.0, holed, Params())
    assert found is not None and abs(found[0] - 0.12) < 0.005


def test_fit_line_is_repeatable() -> None:
    rng = np.random.default_rng(1)
    xy = np.vstack(
        [
            np.column_stack([rng.uniform(0, 3, 3000), rng.normal(0, 0.003, 3000)]),
            rng.uniform(-1, 3, (2000, 2)),
        ]
    )
    first = fit_line(xy)
    assert all(np.array_equal(first, fit_line(xy)) for _ in range(3))
    assert np.abs(first[:, 1]).max() < 0.02 and len(first) > 2800


def test_cabinet_fronts_are_not_a_walls_far_face() -> None:
    from shapely import Polygon

    rng = np.random.default_rng(3)
    # Face along y at x = 0, room on +x; its far face is hidden behind cabinets 60 cm deep whose fronts,
    # facing away, sit at x = -0.67. One session saw both the face and the fronts.
    a = Face("w", "s1", np.array([0.0, 3.0]), np.array([0.0, 0.0]), 0.0, 2.4)
    front, fn = _wall_face(rng, 0.0, 1.0)
    fronts, cn = _wall_face(rng, -0.67, -1.0)
    seen = {"s1": (np.vstack([front, fronts]), np.vstack([fn, cn]))}
    found = measure_thickness(a, 0.0, 3.0, SessionPoints(seen), Params())
    assert found is not None and abs(found[0] - 0.67) < 0.01  # without furniture: fooled
    cabinets = Polygon([(-0.75, -0.1), (-0.05, -0.1), (-0.05, 3.1), (-0.75, 3.1)])
    assert measure_thickness(a, 0.0, 3.0, SessionPoints(seen, furniture=[cabinets]), Params()) is None


def _frame(i: int, t: float, x: float, tracking: str = "normal") -> Frame:

    m = np.eye(4)
    m[0, 3] = x
    k = b.Intrinsics(fx=1500, fy=1500, cx=960, cy=720, width=1920, height=1440)
    return b.Frame(idx=i, t=t, camera_to_world=to_mat4(m), intrinsics=k, tracking=b.TrackingState(tracking))


def test_settled_from_skips_a_startup_relocalisation() -> None:
    """Scan 4's start: 4 frames tracked "normal", an 11 s stall, re-initialising, 5 "normal" frames, then
    ARKit's 31 cm snap and a few cm of settling. Frames before the snap are in another world."""
    from scanmodel.checks import settled_from

    fps = 60.0
    xs = [0.0] * 4 + [0.18] * 117 + [0.26] * 5 + [0.57, 0.667, 0.697, 0.706] + [0.706] * 120
    ts = [i / fps for i in range(4)] + [11.1 + i / fps for i in range(len(xs) - 4)]
    states = ["normal"] * 4 + ["limited_initializing"] * 60 + ["limited_relocalizing"] * 57 + ["normal"] * 129
    frames = [_frame(i, t, x, s) for i, (t, x, s) in enumerate(zip(ts, xs, states, strict=True))]
    assert settled_from(frames, fps) == 127  # after the 31 cm snap and the 9.7 cm settling step

    steady = [_frame(i, i / fps, 0.001 * i) for i in range(300)]
    assert settled_from(steady, fps) == 0
    # A session that starts tracked: a jump at its first frame or mid-session is drift for the pose graph.
    later = [_frame(i, i / fps, 0.001 * i + (0.3 if i in (0, 150) else 0.0)) for i in range(300)]
    assert settled_from(later, fps) == 0
    assert settled_from(steady[:30], fps) == 0  # too short to judge
    fast = [_frame(i, i / 10, 0.25 * i) for i in range(50)]  # 10 fps import or synthetic: 2.5 m/s steps
    assert settled_from(fast, 10.0) == 0


def test_ingest_leaves_out_frames_and_stills_before_tracking_settled(tmp_path: Path) -> None:
    from scanmodel.stages.ingest import check_bundle, posed_images

    d = synth.write_one_session(tmp_path)
    frames = [b.Frame.model_validate_json(x) for x in (d / b.FRAMES_FILE).read_text().splitlines()]
    for f in frames[:6]:  # 10 fps: 3 frames tracked, 3 initialising, all 0.5 m off the settled world
        m = np.array(f.camera_to_world)
        m[0, 3] += 0.5
        f.camera_to_world = m.tolist()
        if f.idx >= 3:
            f.tracking = b.TrackingState.LIMITED_INITIALIZING
    (d / b.FRAMES_FILE).write_text("".join(f.model_dump_json(exclude_none=True) + "\n" for f in frames))
    stills = [b.Still.model_validate_json(x) for x in (d / b.STILLS_FILE).read_text().splitlines()]
    stills[0].t = frames[2].t
    (d / b.STILLS_FILE).write_text("".join(s.model_dump_json(exclude_none=True) + "\n" for s in stills))

    report, bundle = check_bundle(d)
    assert report.ok and bundle is not None
    assert report.counts["unsettled_frames"] == 6
    assert any("first 6 frames" in w for w in report.warnings), report.warnings
    posed, posed_stills = posed_images(bundle)
    assert posed[0].id == f"{bundle.id}/6" and len(posed) == len(frames) - 6
    assert [s.id for s in posed_stills] == [f"{bundle.id}/s1"]


def _walk(
    n: int = 600, fps: float = 60.0, off: dict[int, float] | None = None, yaw: dict[int, float] | None = None
) -> list[Frame]:
    """A 0.5 m/s walk along x, with `off` m added along y and `yaw` degrees about z at some frames."""
    out: list[Frame] = []
    k = b.Intrinsics(fx=1500, fy=1500, cx=960, cy=720, width=1920, height=1440)
    for i in range(n):
        a = np.radians((yaw or {}).get(i, 0.0))
        m = np.eye(4)
        m[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        m[:3, 3] = [0.5 * i / fps, (off or {}).get(i, 0.0), 1.4]
        out.append(
            Frame(idx=i, t=i / fps, camera_to_world=to_mat4(m), intrinsics=k, tracking=b.TrackingState.NORMAL)
        )
    return out


def test_excursions_drop_out_and_back_only() -> None:
    """Scan 5 (living): ARKit's pose 60 cm off for a second, tracking "normal", then back. A one-way
    jump (scan 5's garden: 11.6 cm, once) is drift and stays; so do jitter and sparse imports."""
    from scanmodel.checks import excursions

    away = range(300, 362)  # 62 frames: +0.6 m out at step 299 -> 300, back at 361 -> 362
    (e,) = excursions(_walk(off=dict.fromkeys(away, 0.6)))
    assert (e.first, e.last, e.kind, e.dropped) == (298, 363, "move", True)  # 2 frames either side
    assert abs(e.out_m - 0.6) < 0.01
    assert excursions(_walk(off=dict.fromkeys(range(300, 600), 0.12))) == []  # one-way
    rng = np.random.default_rng(0)
    assert excursions(_walk(off={i: float(rng.normal(0, 0.003)) for i in range(600)})) == []  # jitter
    # Rotation only: 20° away for a second (a brisk hand-held turn is ~2° a frame, and turns back slowly).
    (r,) = excursions(_walk(yaw=dict.fromkeys(away, 20.0)))
    assert (r.first, r.last, r.kind) == (298, 363, "turn") and abs(r.out_deg - 20.0) < 0.1
    smooth = {i: 40.0 * np.sin(np.pi * (i - 300) / 60) for i in range(300, 420)}  # 40° turn and back
    assert excursions(_walk(yaw=smooth)) == []
    # Longer than 3 s: reported, not dropped. Sparse frames (5 fps import): the rule doesn't apply.
    (long,) = excursions(_walk(off=dict.fromkeys(range(200, 420), 0.6)))
    assert not long.dropped
    assert excursions(_walk(n=100, fps=5.0, off=dict.fromkeys(range(40, 45), 0.6))) == []


def test_anchor_moves_skip_anchors_placed_in_a_dropped_window(tmp_path: Path) -> None:
    """Scan 5: still 56's anchor went in during the excursion and ARKit later moved it back 64 cm."""
    from scanmodel.align import anchor_moves

    def ev(t: float, aid: str, event: str, x: float) -> str:
        m = np.eye(4)
        m[0, 3] = x
        e = {"t": t, "id": aid, "name": f"s/{aid}", "event": event, "transform": to_mat4(m)}
        return b.AnchorEvent.model_validate(e).model_dump_json()

    lines = [
        ev(1.0, "a", "add", 0.0),
        ev(5.0, "b", "add", 0.6),
        ev(9.0, "a", "final", 0.01),
        ev(9.0, "b", "final", 0.0),
    ]
    (tmp_path / b.ANCHORS_FILE).write_text("\n".join(lines) + "\n")
    moves, biggest = anchor_moves([(tmp_path, np.eye(4))])
    assert len(moves) == 2 and abs(biggest - 0.6) < 1e-6
    moves, biggest = anchor_moves([(tmp_path, np.eye(4))], {tmp_path: [(4.5, 5.5)]})
    assert [t for _, t, _ in moves] == [1.0] and abs(biggest - 0.01) < 1e-6


def test_ingest_drops_excursions_and_override_ranges(tmp_path: Path) -> None:
    from scanmodel.stages.ingest import check_bundle, posed_images

    d = synth.write_one_session(tmp_path)  # 10 fps, stills at frames 10 and 60
    report, bundle = check_bundle(d, [(55, 64), (5000, 6000)])
    assert report.ok and bundle is not None
    n = len(bundle.frames)
    (w,) = report.dropped
    assert (w.reason, w.idx0, w.idx1, w.frames, w.stills) == ("override", 55, 64, 10, [1])
    assert report.counts["override_frames"] == 10 and report.counts["excursion_frames"] == 0
    assert any("[5000, 6000]: no frame" in x for x in report.warnings), report.warnings
    posed, posed_stills = posed_images(bundle, dropped=report.dropped)
    assert len(posed) == n - 10 and not any(55 <= int(p.file.split("#")[1]) <= 64 for p in posed)
    assert [s.id for s in posed_stills] == [f"{bundle.id}/s0"]

    # A 60 Hz walk with frames 30-41 0.6 m off (the synthetic one moves ~15 cm a frame): found by itself.
    frames = [b.Frame.model_validate_json(x) for x in (d / b.FRAMES_FILE).read_text().splitlines()]
    for f, w in zip(frames, _walk(len(frames), off=dict.fromkeys(range(30, 42), 0.6)), strict=True):
        f.t, f.camera_to_world = 100.0 + w.t, w.camera_to_world
    (d / b.FRAMES_FILE).write_text("".join(f.model_dump_json(exclude_none=True) + "\n" for f in frames))
    stills = [b.Still.model_validate_json(x) for x in (d / b.STILLS_FILE).read_text().splitlines()]
    for s in stills:
        s.t = frames[s.nearest_frame or 0].t
    stills[0].t = frames[35].t
    (d / b.STILLS_FILE).write_text("".join(s.model_dump_json(exclude_none=True) + "\n" for s in stills))
    report, bundle = check_bundle(d)
    assert report.ok and bundle is not None
    (w,) = report.dropped
    assert (w.reason, w.idx0, w.idx1, w.stills) == ("excursion", 28, 43, [0])
    assert report.counts["excursion_frames"] == 16 and report.counts["pose_jumps"] == 0
    posed, posed_stills = posed_images(bundle, dropped=report.dropped)
    assert len(posed) == n - 16 and [s.id for s in posed_stills] == [f"{bundle.id}/s1"]
