"""Splat stage: dataset conventions and the trainer wiring (no actual training)."""

import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest

from scanmodel.geom import GL_CV
from scanmodel.models import bundle as b
from scanmodel.models.bundle import Intrinsics
from scanmodel.models.derived import PosedImage, Poses, SessionPoses
from scanmodel.stages import splat

from .synth import look_at

K = Intrinsics(fx=1500.0, fy=1500.0, cx=960.0, cy=720.0, width=1920, height=1440)


def _posed(c2w: np.ndarray) -> PosedImage:
    return PosedImage.model_validate(
        {
            "id": "s/0",
            "session": "s",
            "kind": "frame",
            "file": "video.mov#0",
            "camera_to_world": c2w.tolist(),
            "intrinsics": K.model_dump(),
        }
    )


def test_transform_matrix_is_opengl_camera_to_world() -> None:
    """A point projected with the nerfstudio (OpenGL) pose lands where the pipeline's OpenCV pose puts it."""
    c2w_cv = look_at(np.array([1.0, 2.0, 1.5]), np.array([3.0, 4.0, 1.0]))
    c2w_cv[3, 3] = 0.9999999  # what the stills carry after the pose graph
    e = splat.frame_entry(_posed(c2w_cv), K, "images/a.jpg", None)
    tm = np.array(e["transform_matrix"])
    np.testing.assert_array_equal(tm[3], [0, 0, 0, 1])
    np.testing.assert_allclose(tm[:3] @ GL_CV, c2w_cv[:3], atol=1e-12)

    def project(p: list[float]) -> tuple[float, float]:
        x = np.linalg.inv(tm) @ [*p, 1.0]  # OpenGL camera: looks down -z, y up
        assert x[2] < 0
        return K.fx * x[0] / -x[2] + K.cx, -K.fy * x[1] / -x[2] + K.cy

    assert project([3.0, 4.0, 1.0]) == pytest.approx((K.cx, K.cy))  # the look-at target: image centre
    assert project([3.0, 4.0, 1.5])[1] < K.cy  # world up is image up


def test_points_ply_is_float_xyz_uchar_rgb(tmp_path: Path) -> None:
    xyz = np.array([[0.0, 1.0, 2.0], [-1.5, 0.25, 3.0]])
    rgb = np.array([[255, 0, 10], [1, 2, 3]], np.uint8)
    path = tmp_path / "p.ply"
    splat.write_points_ply(path, xyz, rgb)
    header = path.read_bytes().split(b"end_header\n")[0].decode()
    assert "property float x" in header and "property uchar red" in header and "double" not in header
    assert splat.ply_vertex_count(path) == 2
    pc = o3d.io.read_point_cloud(path)
    np.testing.assert_allclose(np.asarray(pc.points), xyz)
    np.testing.assert_allclose(np.asarray(pc.colors) * 255, rgb)


def _dataset(root: Path, n: int, w: int = 1920, h: int = 1440) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    frames = [{"file_path": f"images/{i}.jpg", "w": w, "h": h} for i in range(n)]
    (root / "transforms.json").write_text(json.dumps({"frames": frames}))
    return root


def test_msplat_downscale_fits_memory(tmp_path: Path) -> None:
    assert splat.msplat_downscale(_dataset(tmp_path / "big", 1482)) == 4.0  # the 2026-10-06 scan
    assert splat.msplat_downscale(_dataset(tmp_path / "small", 50)) == 1.0
    f = splat.msplat_downscale(_dataset(tmp_path / "mid", 400))
    assert 1.0 < f <= 2.25 and 400 * 1920 * 1440 / f**2 <= splat.MSPLAT_MAX_PIXELS


def test_commands(tmp_path: Path) -> None:
    ds, out = _dataset(tmp_path / "ds", 3), tmp_path / "out"
    exe = tmp_path / "msplat-train"
    exe.write_text("#!/opt/tools/msplat/bin/python\nimport sys\n")
    m = splat.command("msplat", str(exe), ds, out, 7000, 8, 4.0, ["--sh-degree", "2"], tmp_path / "v.npz")
    assert m[:4] == ["/opt/tools/msplat/bin/python", str(splat.MSPLAT_SCRIPT), "--input", str(ds)]
    assert m[m.index("--output") + 1] == str(out / "splat.ply")
    assert m[m.index("--num-iters") + 1] == "7000" and m[m.index("--views") + 1] == str(tmp_path / "v.npz")
    assert m[m.index("--downscale-factor") + 1] == "4" and m[m.index("--test-every") + 1] == "8"
    assert m[-2:] == ["--sh-degree", "2"]
    assert "--eval" not in splat.command("msplat", str(exe), ds, out, 10, 0, None, [])
    assert "--views" not in splat.command("msplat", str(exe), ds, out, 10, 0, None, [])
    assert "--downscale-factor" not in splat.command("msplat", str(exe), ds, out, 10, 0, 1.0, [])
    with pytest.raises(RuntimeError, match="not a Python console script"):
        splat.command("msplat", str(_dataset(tmp_path / "x", 1) / "transforms.json"), ds, out, 1, 0, None, [])

    br = splat.command("brush", "/Applications/brush_app", ds, out, 30000, 8, 2.0, [])
    assert br[:2] == ["/Applications/brush_app", str(ds)]
    assert br[br.index("--total-steps") + 1] == "30000"
    assert br[br.index("--export-path") + 1] == str(out) and br[br.index("--export-name") + 1] == "splat.ply"
    assert br[br.index("--export-every") + 1] == "30000"
    assert br[br.index("--max-resolution") + 1] == "960"
    assert br[br.index("--eval-split-every") + 1] == "8" and "--eval-save-to-disk" in br
    assert "--with-viewer" not in br
    with pytest.raises(ValueError, match="unknown trainer"):
        splat.command("nerfacto", "x", ds, out, 1, 0, None, [])

    gs = splat.command(
        "gsplat", "/usr/bin/uv", ds, out, 30000, 8, 2.0, ["--depth-lambda", "0.5"], name="depth"
    )
    assert gs[:7] == [
        "/usr/bin/uv",
        "run",
        "--project",
        str(splat.TRAINER_PROJECT),
        "gs-trainer",
        "cloud",
        "train",
    ]
    assert (splat.TRAINER_PROJECT / "pyproject.toml").is_file()
    assert gs[gs.index("--data") + 1] == str(ds) and gs[gs.index("--out") + 1] == str(out)
    assert gs[gs.index("--name") + 1] == "depth" and gs[gs.index("--steps") + 1] == "30000"
    assert gs[gs.index("--max-side") + 1] == "960"  # from the downscale, as Brush's --max-resolution
    assert gs[-3:] == ["--", "--depth-lambda", "0.5"]  # gs_trainer.train options after --
    plain = splat.command("gsplat", "uv", ds, out, 100, 8, None, [])
    project_cmd = splat.command("gsplat", "uv", ds, out, 100, 8, None, [], project="office")
    train_at = project_cmd.index("train")
    assert project_cmd[train_at + 1 : train_at + 3] == ["--project", "office"]
    assert plain.count("--project") == 1  # uv's Python project only
    assert "--max-side" not in plain and "--" not in plain and plain[plain.index("--name") + 1] == "gsplat"


def test_train_runs_trainer_and_summarises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds, out = _dataset(tmp_path / "ds", 1482), tmp_path / "out"
    (ds / "views.npz").write_bytes(b"npz")  # written by the export
    exe = tmp_path / "msplat-train"
    exe.write_text(f"#!{sys.executable}\n")
    exe.chmod(0o755)
    calls: list[tuple[list[str], Path]] = []

    def fake_run(cmd: list[str], cwd: Path, log: Path, total: int = 0, label: str = "") -> None:
        calls.append((cmd, cwd))
        assert cmd[cmd.index("--views") + 1] == str((ds / "views.npz").resolve())
        splat.write_points_ply(out / "splat.ply", np.zeros((7, 3)), np.zeros((7, 3), np.uint8))
        (out / "prune.json").write_text('{"views": 3, "removed": 42, "rounds": []}')
        log.write_text(
            "step=  7000  splats=7\n=== Evaluation (186 test views) ===\n  PSNR:  20.5369\n  SSIM:  0.7506\n"
        )

    monkeypatch.setattr(splat, "run_logged", fake_run)
    s = splat.train("msplat", str(exe), ds, out, 7000, prune=True)
    [(cmd, cwd)] = calls
    assert cmd[:2] == [sys.executable, str(splat.MSPLAT_SCRIPT)] and cwd == out
    assert cmd[cmd.index("--downscale-factor") + 1] == "4"  # picked from the pixel budget
    assert s["splats"] == 7 and s["psnr"] == 20.5369 and s["ssim"] == 0.7506 and s["downscale"] == 4.0
    assert s["pruned"] == 42 and (ds / "views.npz").is_file()  # the dataset's copy stays
    with pytest.raises(FileNotFoundError, match=r"\[tools\] msplat"):
        splat.train("msplat", str(tmp_path / "missing"), ds, out, 10)
    (ds / "views.npz").unlink()
    with pytest.raises(FileNotFoundError, match=r"views\.npz"):
        splat.train("msplat", str(exe), ds, out, 10, prune=True)


def test_train_gsplat_through_the_cloud_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds, out = _dataset(tmp_path / "ds", 10), tmp_path / "runs" / "depth"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text('{"psnr": 1.0}')  # from an earlier run: must not count
    calls: list[tuple[list[str], Path, Path]] = []

    def fake_run(cmd: list[str], cwd: Path, log: Path, total: int = 0, label: str = "") -> None:
        calls.append((cmd, cwd, log))
        assert not (out / "metrics.json").exists()
        splat.write_points_ply(out / "splat.ply", np.zeros((5, 3)), np.zeros((5, 3), np.uint8))
        (out / "metrics.json").write_text(
            '{"psnr": 27.25, "ssim": 0.88, "lpips": 0.21, "splats": 5, "wall_s": 900}'
        )
        (out / "run.json").write_text('{"run_id": "gs-depth-20261009-1412"}')
        (out / "train.log").write_text("step=   100 loss=0.1\n")  # the remote log, downloaded
        log.write_text("uploading dataset\n")

    monkeypatch.setattr(splat, "run_logged", fake_run)
    s = splat.train("gsplat", sys.executable, ds, out, 30000, name="depth", extra=["--depth-lambda", "0.5"])
    [(cmd, cwd, log)] = calls
    assert cmd[1:3] == ["run", "--project"] and cmd[cmd.index("--name") + 1] == "depth" and cwd == out
    assert log == out / "cloud.log" and (out / "train.log").read_text().startswith("step=")
    assert s["psnr"] == 27.25 and s["ssim"] == 0.88 and s["lpips"] == 0.21 and s["splats"] == 5
    assert s["gpu_wall_s"] == 900 and s["run"] == "gs-depth-20261009-1412" and s["iterations"] == 30000

    monkeypatch.setattr(splat, "run_logged", lambda *a, **k: None)  # the download failed
    with pytest.raises(FileNotFoundError, match=r"cloud\.log"):
        splat.train("gsplat", sys.executable, ds, out, 10)


def test_run_streams_and_logs(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    splat.run_logged([sys.executable, "-c", "print('step=   100'); print('PSNR: 1.5')"], tmp_path, log, 200)
    assert log.read_text() == "step=   100\nPSNR: 1.5\n"
    assert splat.parse_metrics(log.read_text()) == {"psnr": 1.5}


def test_eval_renders_scores_against_photos(tmp_path: Path) -> None:
    import cv2

    (tmp_path / "ds" / "images").mkdir(parents=True)
    (tmp_path / "eval").mkdir()
    rng = np.random.default_rng(0)
    photo = cv2.GaussianBlur(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8), (5, 5), 2)
    cv2.imwrite(str(tmp_path / "ds" / "images" / "a.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(str(tmp_path / "eval" / "a.png"), cv2.resize(photo, (64, 48), interpolation=cv2.INTER_AREA))
    good = splat.eval_renders(tmp_path / "ds", tmp_path / "eval")
    assert good["psnr"] > 35 and good["ssim"] > 0.97
    cv2.imwrite(str(tmp_path / "eval" / "a.png"), np.zeros((48, 64, 3), np.uint8))
    bad = splat.eval_renders(tmp_path / "ds", tmp_path / "eval")
    assert bad["psnr"] < 20 and bad["ssim"] < 0.5
    assert splat.eval_renders(tmp_path / "ds", tmp_path / "none") == {}


def test_progress_reads_msplat_steps_and_indicatif_bars() -> None:
    from scanmodel.progress import step_of

    assert step_of("step=  1200  splats= 929,224  ms=4.6", 7000) == 1200
    assert step_of("[00:01:02] ######## 1200/7000 training (12.3/s, 8m remaining)", 7000) == 1200
    assert step_of("Loaded 1296 train cameras, 186 test cameras", 7000) is None
    assert step_of("2026/10 something 3/4", 7000) is None


def test_progress_bar_from_a_redrawn_line(tmp_path: Path) -> None:
    from scanmodel import progress

    # An indicatif-style bar: redrawn in place with \r, then a final line.
    script = (
        "import sys, time\n"
        "for i in (10, 50, 100):\n"
        "    sys.stdout.write(f'\\r[00:00:01] ### {i}/100 (5/s)'); sys.stdout.flush(); time.sleep(0.05)\n"
        "print(); print('Done: wrote splat.ply')\n"
    )
    log = tmp_path / "train.log"
    assert progress.run([sys.executable, "-c", script], tmp_path, log, 100, "test") == 0
    assert log.read_text() == "Done: wrote splat.ply\n"


# --------------------------------------------------------------------------- dataset v2

DEPTH_SIZE = (256, 192)  # ARKit's depth map


def _session(raw: Path, sid: str, n: int, n_stills: int, seed: int) -> list[PosedImage]:
    """A session folder (video, raw depth + confidence, stills) and its posed frames and stills."""
    rng = np.random.default_rng(seed)
    d = raw / sid
    for sub in (b.DEPTH_DIR, b.CONFIDENCE_DIR, b.STILLS_DIR):
        (d / sub).mkdir(parents=True)
    k = Intrinsics(fx=60.0, fy=60.0, cx=32.0, cy=24.0, width=64, height=48)
    video = cv2.VideoWriter(str(d / b.VIDEO_FILE), cv2.VideoWriter.fourcc(*"mp4v"), 2.0, (64, 48))
    ims: list[PosedImage] = []
    for i in range(n):
        video.write(np.full((48, 64, 3), 10 * i % 255, np.uint8))
        depth = rng.integers(300, 4000, DEPTH_SIZE[::-1]).astype(np.uint16)
        depth[:20] = 0  # holes
        cv2.imwrite(str(d / b.DEPTH_DIR / f"{i:06d}.png"), depth)
        cv2.imwrite(
            str(d / b.CONFIDENCE_DIR / f"{i:06d}.png"), rng.integers(0, 3, DEPTH_SIZE[::-1], np.uint8)
        )
        c2w = look_at(np.array([0.1 * i, 0.0, 1.5]), np.array([0.1 * i, 3.0, 1.2]))
        ims.append(
            PosedImage.model_validate(
                {
                    "id": f"{sid}/{i}",
                    "session": sid,
                    "kind": "frame",
                    "file": f"{b.VIDEO_FILE}#{i}",
                    "t": 100.0 + 0.5 * i,
                    "camera_to_world": c2w.tolist(),
                    "intrinsics": k.model_dump(),
                    "tracking": "normal",
                    "depth": f"{b.DEPTH_DIR}/{i:06d}.png",
                    "confidence": f"{b.CONFIDENCE_DIR}/{i:06d}.png",
                }
            )
        )
    video.release()
    for j in range(n_stills):
        cv2.imwrite(str(d / b.STILLS_DIR / f"{j:06d}.jpg"), np.full((96, 128, 3), 128, np.uint8))
        ims.append(
            ims[j].model_copy(
                update={
                    "id": f"{sid}/s{j}",
                    "kind": "still",
                    "file": f"{b.STILLS_DIR}/{j:06d}.jpg",
                    "intrinsics": k.scaled(128, 96),
                    "depth": None,
                    "confidence": None,
                    "source": b.StillSource.TAP,
                }
            )
        )
    return ims


def _scan(root: Path) -> tuple[Path, Path]:
    """(poses dir, fuse dir) of a two-session scan: a hallway and the closing lap."""
    raw, poses, fuse = root / "raw", root / "poses", root / "fuse"
    poses.mkdir(parents=True)
    fuse.mkdir()
    ims = [*_session(raw, "s1-hallway", 20, 2, 1), *_session(raw, "s2-lap", 6, 1, 2)]
    sessions = [
        SessionPoses.model_validate(
            {"session": s, "raw_dir": str(raw / s), "session_to_scene": np.eye(4).tolist()}
        )
        for s in ("s1-hallway", "s2-lap")
    ]
    (poses / "poses.json").write_text(Poses(yaw_deg=0.0, floor_z=0.0, sessions=sessions).model_dump_json())
    for name, kind in (("frames.jsonl", "frame"), ("stills.jsonl", "still")):
        (poses / name).write_text("".join(im.model_dump_json() + "\n" for im in ims if im.kind == kind))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.default_rng(0).uniform(0, 3, (500, 3))))
    o3d.io.write_point_cloud(fuse / "points.ply", pc)
    box = o3d.geometry.TriangleMesh.create_box(3.0, 4.0, 2.5)
    box.compute_vertex_normals()
    box.vertex_colors = o3d.utility.Vector3dVector(np.full((8, 3), 0.5))
    o3d.io.write_triangle_mesh(fuse / "dense.ply", box)  # doubles, normals, colours: as fuse writes it
    return poses, fuse


@pytest.fixture(scope="module")
def scan(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return _scan(tmp_path_factory.mktemp("scan"))


@pytest.fixture(scope="module")
def dataset(scan: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("splat") / "dataset"
    summary = splat.export(*scan, out, select="time")  # every 0.5 s = every frame of the synthetic video
    assert summary["images"] == 29 and summary["extrap"] == 7 and summary["extrap_session"] == "s2-lap"
    assert summary["interp"] == 3 and summary["train"] == 19 and summary["with_depth"] == 26
    return out


def test_dataset_v2_frames(dataset: Path) -> None:
    t = json.loads((dataset / "transforms.json").read_text())
    assert t["version"] == 2 and t["camera_model"] == "OPENCV" and t["depth_unit_scale_factor"] == 0.001
    assert (t["ply_file_path"], t["mesh_file_path"], t["split_file_path"]) == (
        "points.ply",
        "mesh.ply",
        "split.json",
    )
    frames = t["frames"]
    assert [f["file_path"] for f in frames] == sorted(f["file_path"] for f in frames)
    assert {f["session"] for f in frames} == {"s1-hallway"}  # the lap is extrap only
    for f in frames:
        assert (dataset / f["file_path"]).is_file() and isinstance(f["t"], float)
        if f["camera"] == "still":
            assert "depth_file_path" not in f and "confidence_file_path" not in f
            assert (f["w"], f["h"]) == (128, 96) and f["file_path"].endswith(("_s0.jpg", "_s1.jpg"))
        else:
            assert f["camera"] == "video" and (f["w"], f["h"]) == (64, 48)
            assert f["depth_file_path"] == f"depth/{splat.stem(f)}.png"
            assert f["confidence_file_path"] == f"confidence/{splat.stem(f)}.png"
    assert sum(f["camera"] == "still" for f in frames) == 2


def test_dataset_v2_depth_is_native_and_unmasked(dataset: Path, scan: tuple[Path, Path]) -> None:
    raw = Path(json.loads((scan[0] / "poses.json").read_text())["sessions"][0]["raw_dir"])
    depth = np.asarray(cv2.imread(str(dataset / "depth" / "s1-hallway_000003.png"), cv2.IMREAD_UNCHANGED))
    conf = np.asarray(cv2.imread(str(dataset / "confidence" / "s1-hallway_000003.png"), cv2.IMREAD_UNCHANGED))
    assert depth.dtype == np.uint16 and depth.shape == DEPTH_SIZE[::-1]  # not upsampled to the image
    assert conf.dtype == np.uint8 and conf.shape == DEPTH_SIZE[::-1] and set(np.unique(conf)) <= {0, 1, 2}
    np.testing.assert_array_equal(depth, cv2.imread(str(raw / "depth" / "000003.png"), cv2.IMREAD_UNCHANGED))
    assert (depth[conf == 0] > 0).any()  # low confidence kept: the trainer picks the level
    np.testing.assert_array_equal(
        conf, cv2.imread(str(raw / "confidence" / "000003.png"), cv2.IMREAD_UNCHANGED)
    )


def test_dataset_v2_split(dataset: Path) -> None:
    frames = json.loads((dataset / "transforms.json").read_text())["frames"]
    s = json.loads((dataset / "split.json").read_text())
    assert s["interp"] == [splat.stem(f) for i, f in enumerate(frames) if i % 8 == 0]
    assert s["extrap_session"] == "s2-lap"
    assert s["extrap"] == [*(f"s2-lap_{i:06d}" for i in range(6)), "s2-lap_s0"]
    assert [splat.stem(f) for f in s["extrap_frames"]] == s["extrap"]
    for f in s["extrap_frames"]:
        assert f["session"] == "s2-lap" and (dataset / f["file_path"]).is_file()
        assert f["camera"] == "still" or (dataset / f["depth_file_path"]).is_file()
    assert not set(s["extrap"]) & {splat.stem(f) for f in frames}


def test_extrap_session_choice(scan: tuple[Path, Path], tmp_path: Path) -> None:
    assert splat.extrap_session_of(["a-kitchen", "b-lap"]) == "b-lap"
    assert splat.extrap_session_of(["a-kitchen"]) is None
    assert splat.extrap_session_of(["a-lap"]) is None
    with pytest.raises(ValueError, match="only session"):
        splat.extrap_session_of(["a-lap"], "a-lap")
    assert splat.extrap_session_of(["a-kitchen", "b-lap"], "none") is None
    assert splat.extrap_session_of(["a-kitchen", "b-lap"], "a-kitchen") == "a-kitchen"
    with pytest.raises(ValueError, match="no session"):
        splat.extrap_session_of(["a-kitchen"], "c-hall")
    s = splat.export(*scan, tmp_path / "none", extrap_session="none", select="time")
    assert s["extrap"] == 0 and s["interp"] == 4  # 29 frames: 0, 8, 16, 24
    split = json.loads((tmp_path / "none" / "split.json").read_text())
    assert split["extrap"] == [] and split["extrap_session"] is None and split["extrap_frames"] == []


def test_dataset_v2_mesh_and_views(dataset: Path) -> None:
    data = (dataset / "mesh.ply").read_bytes()
    header, body = data.split(b"end_header\n")
    assert header.decode().splitlines()[2:] == [
        "element vertex 8", "property float x", "property float y", "property float z",
        "element face 12", "property list uchar int vertex_indices",
    ]  # fmt: skip
    verts = np.frombuffer(body[: 8 * 12], "<f4").reshape(8, 3)
    faces = np.frombuffer(body[8 * 12 :], [("n", "u1"), ("v", "<i4", 3)])
    assert len(faces) == 12 and (faces["n"] == 3).all() and faces["v"].max() == 7
    np.testing.assert_allclose(verts.max(axis=0), [3.0, 4.0, 2.5])
    mesh = o3d.io.read_triangle_mesh(dataset / "mesh.ply")
    assert len(mesh.triangles) == 12 and not mesh.has_vertex_normals()

    from scanmodel.freespace import load_views

    depth, k, w2c = load_views(dataset / "views.npz")
    assert depth.shape == (5, *DEPTH_SIZE[::-1]) and k.shape == (5, 4) and w2c.shape == (5, 4, 4)  # every 6th


# --------------------------------------------------------------------------- input quality (P6 a1, a2)


def _texture(h: int, w: int, seed: int) -> np.ndarray:
    """A colourful mid-grey texture (values well exposed), smooth enough to survive video compression."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(60, 190, (h // 4, w // 4, 3)).astype(np.float32)
    return np.clip(cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)


def test_sharpness_is_the_4_neighbour_laplacian_variance() -> None:
    img = _texture(48, 64, 0)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c = grey[1:-1, 1:-1]
    lap = 4 * c - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:]  # Sharpness.swift
    assert splat.sharpness(img) == pytest.approx(lap.var(), rel=0.05)  # border handling differs
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    assert splat.sharpness(blurred) < 0.3 * splat.sharpness(img)
    big = cv2.resize(_texture(360, 480, 1), (1920, 1440), interpolation=cv2.INTER_NEAREST)
    assert splat.sharpness(big) == pytest.approx(
        splat.sharpness(_texture(360, 480, 1)), rel=0.05
    )  # at 480 px


def test_video_sharpness_is_cached(tmp_path: Path) -> None:
    path = tmp_path / "video.mov"
    video = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(12):
        img = _texture(48, 64, i)
        video.write(cv2.GaussianBlur(img, (0, 0), 2.5) if i % 3 == 0 else img)
    video.release()
    s, took = splat.video_sharpness(path, tmp_path / "cache" / "x.npy")
    assert s.shape == (12,) and np.isfinite(s).all() and took > 0
    assert all(s[i] < 0.6 * np.median(s) for i in (0, 3, 6, 9))  # the blurred frames
    again, took = splat.video_sharpness(path, tmp_path / "cache" / "x.npy")
    assert took == 0.0 and np.array_equal(again, s)
    _, took = splat.video_sharpness(path, tmp_path / "cache" / "x.npy", side=32)  # other settings: redone
    assert took > 0


def _walk(
    n: int, fps: float = 60.0, speed: float = 0.0, turn_dps: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Times and camera-to-world poses of a camera walking along x at `speed` m/s turning `turn_dps` °/s."""
    t = np.arange(n) / fps
    c2w = np.stack(
        [
            look_at(
                np.array([speed * ti, 0.0, 1.5]),
                np.array(
                    [
                        speed * ti + math.cos(math.radians(turn_dps * ti)),
                        math.sin(math.radians(turn_dps * ti)),
                        1.5,
                    ]
                ),
            )
            for ti in t
        ]
    )
    return t, c2w


def test_motion_slots_by_distance_turn_and_time() -> None:
    t, c2w = _walk(600, speed=0.61)  # 10 s, 6.1 m: a slot per 10 frames (10.2 cm)
    assert [len(s) for s in splat.motion_slots(t, c2w, 0.10, 5.0, 1.0)] == [10] * 60
    t, c2w = _walk(600, turn_dps=31.0)  # 310° on the spot: a slot per 10 frames (5.2°)
    assert [len(s) for s in splat.motion_slots(t, c2w, 0.10, 5.0, 1.0)] == [10] * 60
    t, c2w = _walk(600)  # standing still: a slot per max_gap_s
    slots = splat.motion_slots(t, c2w, 0.10, 5.0, 1.0)
    assert len(slots) == 10 and all(len(s) == 60 for s in slots)
    assert np.array_equal(np.concatenate(slots), np.arange(600))  # consecutive, nothing lost


def test_select_motion_picks_the_relatively_sharpest() -> None:
    t, c2w = _walk(600, speed=0.61)
    rng = np.random.default_rng(0)
    s = rng.uniform(80, 100, 600)
    s[300:] *= 2  # a more textured place (a 10x step would make its neighbours look blurry to the drop)
    sharp = np.arange(3, 600, 10)  # one clearly sharper frame per 10 cm slot
    s[sharp] *= 1.5
    kept, notes, n_slots = splat.select_motion(t, c2w, s, 0.10, 5.0, 1.0, 0.6)
    assert n_slots == 60 and not notes
    assert kept == sharp.tolist()
    rel = splat.relative_sharpness(t, s)
    assert rel[sharp].min() > 1.1 and np.median(rel) == pytest.approx(1.0, abs=0.05)


def test_blur_drop_against_all_neighbours_ignores_texture_changes() -> None:
    t, c2w = _walk(600, speed=0.61)
    rng = np.random.default_rng(1)
    s = rng.uniform(80, 100, 600)
    s *= 10
    s[240:312] /= 10  # bookshelves, a glance at a plain wall (4 s to 5.2 s), bookshelves again
    _, notes, _ = splat.select_motion(t, c2w, s, 0.10, 5.0, 1.0, 0.6)
    assert not notes  # neighbours: the plain wall is as sharp as the frames around it
    _, notes_picked, _ = splat.select_motion(t, c2w, s, 0.10, 5.0, 1.0, 0.6, blur_ref="picked")
    assert notes_picked and all(4.0 <= t[i] < 5.2 for i in notes_picked)  # plain-wall picks look blurry
    # A pick blurrier than the frames around it (which aren't candidates) goes, unless it's the only view.
    all_t = np.arange(3600) / 360  # the full-rate video around the candidates
    all_s = np.interp(all_t, t, s)
    s2 = s.copy()
    s2[100:110] = 30.0  # slot 10: every candidate blurry, the video around sharp
    kept2, notes2, _ = splat.select_motion(t, c2w, s2, 0.10, 5.0, 1.0, 0.6, "neighbours", all_t, all_s)
    [(i, note)] = notes2.items()
    assert 100 <= i < 110 and note.startswith("blurry") and i not in kept2
    with pytest.raises(ValueError, match="blur_ref"):
        splat.select_motion(t, c2w, s, 0.10, 5.0, 1.0, 0.6, blur_ref="room")


def test_drop_blurry_keeps_the_only_view() -> None:
    t = np.arange(200) / 10
    s = np.full(len(t), 100.0)
    picked = list(range(0, 100, 3))  # dense picks for 10 s ...
    picked += [115, 130, 145, 190]  # ... then 1.5 s apart
    s[[30, 31, 130]] = 20.0  # blurry: one among dense picks (30), one sparse (130); 31 is not picked
    kept, notes = splat.drop_blurry(picked, t, s, 0.6, 0.5)
    assert 30 not in kept and notes[30].startswith("blurry")  # 2.7 s -> 3.3 s: still seen
    assert 130 in kept and notes[130].startswith("kept")  # 11.5 s -> 14.5 s would leave 3 s > 2 x 0.5 s
    assert kept == [i for i in picked if i != 30]
    # Two blurry picks in a row: the first goes, the second then has to stay.
    s2 = np.full(len(t), 100.0)
    picked2 = [0, 10, 20, 30, 40, 50]  # 1 s apart
    s2[[20, 30]] = 10.0
    kept2, notes2 = splat.drop_blurry(picked2, t, s2, 0.6, 1.0)
    assert kept2 == [0, 10, 30, 40, 50] and notes2[30].startswith("kept")
    # The session's ends count as edges: a blurry first pick 3 s in is kept.
    kept3, _ = splat.drop_blurry([30, 40, 50, 60], t, np.where(np.arange(len(t)) == 30, 1.0, 100.0), 0.6, 1.0)
    assert kept3 == [30, 40, 50, 60]
    # With a reference per frame (the neighbours' median) the gap rule is the same.
    ref = np.full(len(t), 100.0)
    s4 = np.where(np.isin(np.arange(len(t)), [20, 30]), 10.0, 100.0)
    kept4, notes4 = splat.drop_blurry(picked2, t, s4, 0.6, 1.0, ref)
    assert kept4 == [0, 10, 30, 40, 50] and "all frames" in notes4[20] and notes4[30].startswith("kept")


def test_blurry_still_rule() -> None:
    assert splat.blurry_still(20, 100, 50, 0.6)  # blurrier than the room and the video around it
    assert not splat.blurry_still(55, 100, 50, 0.6)  # soft, but sharper than the video there
    assert not splat.blurry_still(70, 100, 90, 0.6)  # not blurry for the room
    assert splat.blurry_still(20, 100, math.nan, 0.6)  # no video: the room decides


def test_brightness_gain_recovers_a_known_gain() -> None:
    ref = _texture(360, 480, 3)
    ref[:40] = 255  # blown highlights and crushed shadows don't count
    ref[-40:] = 0
    truth = np.array([0.92, 0.95, 0.88])  # BGR: the still is brighter, blue most
    still = np.clip(ref.astype(np.float64) / truth, 0, 255).astype(np.uint8)
    still = cv2.resize(np.roll(still, 6, axis=1), (1920, 1440))  # a slightly different view, bigger
    gain = splat.brightness_gain(still, ref)
    np.testing.assert_allclose(gain, truth, atol=0.015)
    np.testing.assert_allclose(splat.brightness_gain(ref, ref), 1.0)
    np.testing.assert_allclose(splat.brightness_gain((ref * 0.5).astype(np.uint8), ref), 1.25)  # clipped
    np.testing.assert_allclose(splat.brightness_gain(np.zeros_like(ref), ref), 1.0)  # nothing usable
    out = splat.apply_gain(still, gain)
    assert out.dtype == np.uint8 and abs(float(np.median(out)) - float(np.median(ref))) < 4


def test_export_motion_drops_blurry_stills_and_matches_brightness(tmp_path: Path) -> None:
    poses, fuse = _scan(tmp_path / "scan")
    raw = Path(json.loads((poses / "poses.json").read_text())["sessions"][0]["raw_dir"])
    # Textured video; stills: s0 the frame-0 view blurred hard, s1 frame 1's view with a colour cast.
    video = cv2.VideoWriter(str(raw / b.VIDEO_FILE), cv2.VideoWriter.fourcc(*"mp4v"), 2.0, (64, 48))
    frames = [_texture(48, 64, i) for i in range(20)]
    for f in frames:
        video.write(f)
    video.release()
    cast = np.array([1.1, 1.0, 0.9])  # BGR
    s1 = np.clip(cv2.resize(frames[1], (128, 96)) * cast, 0, 255).astype(np.uint8)
    cv2.imwrite(str(raw / b.STILLS_DIR / "000001.jpg"), s1, [cv2.IMWRITE_JPEG_QUALITY, 100])
    s0 = cv2.GaussianBlur(cv2.resize(frames[0], (128, 96)), (0, 0), 4.0)
    cv2.imwrite(str(raw / b.STILLS_DIR / "000000.jpg"), s0)

    out = tmp_path / "dataset"
    summary = splat.export(poses, fuse, out, max_side=64)
    sel = json.loads((out / "selection.json").read_text())
    assert summary["select"] == "motion" and sel["select"] == "motion"
    assert set(sel["thresholds"]) >= {"min_move_m", "min_turn_deg", "max_gap_s", "rel_blur", "every_s"}
    assert "s1-hallway_s0" in sel["dropped"] and sel["dropped"]["s1-hallway_s0"].startswith("blurry still")
    assert not (out / "images" / "s1-hallway_s0.jpg").exists()
    hall = sel["sessions"]["s1-hallway"]
    assert hall["candidates"] == 20 and hall["time_every_s"] == 20 and hall["video"] == hall["slots"] == 10
    assert (tmp_path / "sharpness" / "s1-hallway.npy").is_file()  # next to the dataset

    t = json.loads((out / "transforms.json").read_text())["frames"]
    still = next(f for f in t if splat.stem(f) == "s1-hallway_s1")
    np.testing.assert_allclose(still["brightness_gain"], 1 / cast[::-1], atol=0.03)  # recorded as RGB
    img = np.asarray(cv2.imread(str(out / still["file_path"])))
    ref = frames[1]
    assert np.abs(np.median(img.reshape(-1, 3), axis=0) - np.median(ref.reshape(-1, 3), axis=0)).max() < 6
    assert all("brightness_gain" not in f for f in t if f["camera"] == "video")
    assert sel["brightness_gain"]["stills"] == 2  # s1 + the lap's still
    split = json.loads((out / "split.json").read_text())
    assert split["interp"] == [splat.stem(f) for f in t[:: splat.INTERP_EVERY]]
    assert split["extrap"] and all(e["session"] == "s2-lap" for e in split["extrap_frames"])
    with pytest.raises(ValueError, match="select"):
        splat.export(poses, fuse, tmp_path / "x", select="best")


# --------------------------------------------------------------------------- named views


def test_view_from_viewer_pose_url() -> None:
    c2w_cv = look_at(np.array([1.0, 2.0, 1.5]), np.array([3.0, 4.0, 1.0]))
    gl = c2w_cv @ GL_CV
    url = f"http://localhost:5173/?splat=msplat&pose={','.join(map(str, gl.ravel()))}&fov=55"
    v = splat.view_from_url("hall", url, "doorframe doubles")
    assert v["name"] == "hall" and v["note"] == "doorframe doubles" and (v["w"], v["h"]) == (960, 720)
    assert v["fov_y_deg"] == 55.0
    np.testing.assert_allclose(np.array(v["c2w"]), c2w_cv, atol=1e-6)  # stored with OpenCV axes
    again = splat.view_from_url("hall", splat.view_url(v))
    np.testing.assert_allclose(np.array(again["c2w"]), np.array(v["c2w"]), atol=1e-5)
    assert again["fov_y_deg"] == 55.0


def test_view_from_viewer_cam_look_url() -> None:
    v = splat.view_from_url("k", "?cam=1,2,1.5&look=1,5,1.5", size=(640, 480))
    c2w = np.array(v["c2w"])
    np.testing.assert_allclose(c2w[:3, 3], [1, 2, 1.5])
    np.testing.assert_allclose(c2w[:3, 2], [0, 1, 0], atol=1e-9)  # looks north
    np.testing.assert_allclose(c2w[:3, 1], [0, 0, -1], atol=1e-9)  # image down is world down
    assert v["fov_y_deg"] == splat.VIEWER_FOV and (v["w"], v["h"]) == (640, 480)
    for bad, msg in (
        ("http://localhost:5173/?splat=msplat", "needs \\?pose="),
        ("?pose=1,2,3", "16 comma-separated"),
        (f"?pose={','.join(['2'] * 16)}", "rigid"),
        ("?cam=0,0,1&look=1,0,1&fov=0", "fov"),
    ):
        with pytest.raises(ValueError, match=msg):
            splat.view_from_url("x", bad)


def _viewer_copy_url(eye: np.ndarray, yaw: float, pitch: float, fov: float) -> tuple[str, str]:
    """The two URLs the viewer's Copy view (C / shift+C) builds for a walking camera, step by step as
    main.ts `getPose`/`copyView` and params.ts `viewHref` do: three.js is y-up, scene (x, y, z) -> three
    (x, z, -y); the camera's rotation is Euler YXZ (yaw, then pitch); numbers rounded to 6 decimals."""
    scene_to_three = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # rotation -90 deg about x (Z_UP_TO_Y_UP)
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    rot = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    world = np.eye(4)  # camera.matrixWorld, three.js coordinates
    world[:3, :3], world[:3, 3] = rot, scene_to_three @ eye
    to_scene = np.eye(4)
    to_scene[:3, :3] = scene_to_three.T
    pose = to_scene @ world  # premultiply by the inverse of Z_UP_TO_Y_UP

    def nums(v: np.ndarray) -> str:
        return ",".join(f"{round(float(x), 6) + 0.0:g}" for x in v.ravel())

    base = "http://localhost:5173/?splat=msplat&model=0"
    look = eye + scene_to_three.T @ (rot @ [0.0, 0.0, -1.0])  # walk.target(): 1 m ahead
    return f"{base}&pose={nums(pose)}&fov={fov:g}", f"{base}&cam={nums(eye)}&look={nums(look)}&fov={fov:g}"


@pytest.mark.parametrize(("yaw", "pitch"), [(0.0, 0.0), (-np.pi / 2, -0.4), (2.5, 0.3)])
def test_view_from_viewer_copy_url(yaw: float, pitch: float) -> None:
    eye = np.array([1.25, -2.5, 1.6])
    exact, walkable = _viewer_copy_url(eye, yaw, pitch, 62.5)
    # yaw 0 faces north (+y), positive yaw turns left (counterclockwise seen from above), pitch looks up
    forward = np.array([-np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    for url in (exact, walkable):
        v = splat.view_from_url("spot", url)
        c2w = np.array(v["c2w"])  # OpenCV axes: +z forward, +y image down, +x image right
        np.testing.assert_allclose(c2w[:3, 3], eye, atol=1e-6)
        np.testing.assert_allclose(c2w[:3, 2], forward, atol=1e-5)
        assert abs(c2w[2, 0]) < 1e-5 and c2w[2, 1] < 0  # no roll: image right is level, image down is down
        np.testing.assert_allclose(c2w[:3, 0], np.cross(c2w[:3, 1], c2w[:3, 2]), atol=1e-5)
        assert v["fov_y_deg"] == 62.5
        again = splat.view_from_url("spot", splat.view_url(v))  # the stored view reopens the same camera
        np.testing.assert_allclose(np.array(again["c2w"]), c2w, atol=1e-5)


def test_named_views_file(tmp_path: Path) -> None:
    path = tmp_path / "splat_views.json"
    a = splat.view_from_url("a", "?cam=0,0,1&look=1,0,1")
    assert not splat.add_named_view(path, a)
    assert not splat.add_named_view(path, splat.view_from_url("b", "?cam=0,0,1&look=0,1,1"))
    assert splat.add_named_view(path, {**a, "note": "again"})  # same name: replaced in place
    views = splat.read_named_views(path)
    assert [v["name"] for v in views] == ["a", "b"] and views[0]["note"] == "again"
    assert "[0.0, 0.0, 1.0, 0.0]," in path.read_text()  # one matrix row per line
    assert splat.read_named_views(tmp_path / "missing.json") == []


def test_publish_lists_named_runs(tmp_path: Path) -> None:
    from scanmodel.cli import publish_splats

    splat_dir, viewer = tmp_path / "splat", tmp_path / "viewer"
    (splat_dir / "dataset").mkdir(parents=True)
    for run, trainer in (("msplat", "msplat"), ("gs-depth", "gsplat")):
        (splat_dir / run).mkdir()
        splat.write_points_ply(splat_dir / run / "splat.ply", np.zeros((3, 3)), np.zeros((3, 3), np.uint8))
        manifest = {"params": {"trainer": trainer}, "summary": {"splats": 3, "command": ["x"]}}
        (splat_dir / run / "manifest.json").write_text(json.dumps(manifest))
    entries = publish_splats(splat_dir, viewer, "msplat")
    assert [(e["id"], e["trainer"]) for e in entries] == [("gs-depth", "gsplat"), ("msplat", "msplat")]
    entries = publish_splats(splat_dir, viewer, "msplat", "data/scans/fifth/")
    assert entries[0]["url"] == "data/scans/fifth/splats/gs-depth.ply"


def test_export_scans_side_by_side(tmp_path: Path) -> None:
    """Each project exports its scan into viewer/public/data/scans/<project>/ and scans.json lists them;
    exporting another scan keeps the first one's folder and the default."""
    from scanmodel import cli
    from scanmodel.config import load_config

    assert cli.viewer_data_dir(tmp_path) == tmp_path / "viewer/public/data"
    data = cli.viewer_data_dir(tmp_path)
    roots = {}
    for name, day in (("third", "20261007"), ("fifth", "20261009")):
        cfg = load_config(tmp_path, project=name)
        cfg.stage_dir("poses").mkdir(parents=True)
        sessions = [{"session": f"{day}-1200{i}0-{room}", "raw_dir": f"data/{name}/raw/{day}-1200{i}0-{room}"}
                    for i, room in enumerate(("conservatory", "kitchen", "kitchen"))]  # fmt: skip
        (cfg.stage_dir("poses") / "poses.json").write_text(json.dumps({"sessions": sessions}))
        roots[name] = cfg
    assert cli.scan_label(roots["third"], "third") == "third · 2026-10-07 · conservatory, kitchen"

    views = roots["fifth"].views_file
    splat.add_named_view(views, splat.view_from_url("door", "?cam=1,2,1.5&look=1,3,1.5&fov=50"))
    assert cli.publish_views(views, tmp_path / "views.json") == 1
    v = json.loads((tmp_path / "views.json").read_text())["views"][0]
    assert (v["name"], v["fov"]) == ("door", 50.0)
    again = splat.view_from_url("door", f"?pose={','.join(map(str, v['pose']))}&fov=50")
    assert again["c2w"] == splat.read_named_views(views)[0]["c2w"]  # the camera `scanmodel view list` shows
    assert cli.publish_views(tmp_path / "none.json", tmp_path / "empty.json") == 0

    data.mkdir(parents=True)
    entry = {"id": "third", "base": "data/scans/third/"}
    assert cli.register_scan(data, entry, False)["default"] == "third"  # the first one
    index = cli.register_scan(data, {"id": "fifth", "base": "data/scans/fifth/"}, False)
    assert index["default"] == "third" and [s["id"] for s in index["scans"]] == ["fifth", "third"]
    index = cli.register_scan(data, {**entry, "splats": 3}, False)  # re-export: replaced in place
    assert len(index["scans"]) == 2 and index["scans"][1]["splats"] == 3
    assert cli.register_scan(data, {"id": "fifth", "base": "x"}, True)["default"] == "fifth"
    assert json.loads((data / cli.SCANS_INDEX).read_text())["default"] == "fifth"


def test_view_from_image_and_per_project_views(repo: Path) -> None:
    """`scanmodel view add NAME --image ID` takes a captured frame's pose and field of view, and writes the
    project's views file."""
    from typer.testing import CliRunner

    from scanmodel.cli import app

    c2w = look_at(np.array([0.5, -1.0, 1.5]), np.array([0.5, 2.0, 1.2]))
    k = Intrinsics(fx=1500.0, fy=1500.0, cx=960.0, cy=720.0, width=1920, height=1440)
    frame = PosedImage(
        id="s1/342",
        session="s1",
        kind="frame",
        file="video.mov#342",
        camera_to_world=c2w.tolist(),
        intrinsics=k,
    )
    still = frame.model_copy(update={"id": "s1/s7", "kind": "still", "file": "stills/000007.jpg"})
    tmp_path = repo
    (tmp_path / "settings.toml").write_text('project = "p"\n')
    derived = tmp_path / "data/p/derived"
    for stage in ("poses", "refine"):
        d = derived / stage
        d.mkdir(parents=True)
        (d / "frames.jsonl").write_text(frame.model_dump_json() + "\n")
        (d / "stills.jsonl").write_text(still.model_dump_json() + "\n")
    assert splat.find_image(derived / "poses", "s1_000342").id == "s1/342"
    assert splat.find_image(derived / "poses", "s1_s7").id == "s1/s7"
    with pytest.raises(KeyError):
        splat.find_image(derived / "poses", "s1/343")

    v = splat.view_from_image("door", frame, "the door")
    np.testing.assert_allclose(np.array(v["c2w"]), c2w, atol=1e-6)
    assert (v["w"], v["h"]) == (960, 720)
    assert v["fov_y_deg"] == pytest.approx(math.degrees(2 * math.atan(720 / 1500)), abs=1e-3)
    assert (
        splat.view_from_url("door", splat.view_url(v))["c2w"] == v["c2w"]
    )  # the viewer shows the same camera

    runner = CliRunner()
    r = runner.invoke(
        app, ["view", "add", "door", "--image", "s1_000342", "--poses", "refine", "--note", "n"]
    )
    assert r.exit_code == 0, r.output
    views = splat.read_named_views(tmp_path / "data/p/scene/views.json")
    assert [x["name"] for x in views] == ["door"] and views[0]["note"] == "n"
    assert runner.invoke(app, ["view", "add", "x"]).exit_code != 0  # neither a URL nor --image
    assert (
        runner.invoke(app, ["view", "add", "x", "?cam=0,0,1&look=1,0,1", "--image", "s1/342"]).exit_code != 0
    )


def test_publish_keeps_links_to_other_scans(tmp_path: Path) -> None:
    """A splat linked by hand from another scan's root (and its splats.json entry) survives an export; a link
    whose file is gone (a deleted run, a dangling link) does not."""
    from scanmodel.cli import publish_splats

    splat_dir, viewer, other = tmp_path / "splat", tmp_path / "viewer", tmp_path / "fourth" / "s4-run"
    (splat_dir / "dataset").mkdir(parents=True)
    for run in ("msplat", "old"):
        (splat_dir / run).mkdir()
        splat.write_points_ply(splat_dir / run / "splat.ply", np.zeros((3, 3)), np.zeros((3, 3), np.uint8))
        (splat_dir / run / "manifest.json").write_text(json.dumps({"params": {"trainer": "msplat"}}))
    other.mkdir(parents=True)
    splat.write_points_ply(other / "splat.ply", np.zeros((3, 3)), np.zeros((3, 3), np.uint8))
    publish_splats(splat_dir, viewer, "msplat")
    (viewer / "splats" / "s4-run.ply").symlink_to(other / "splat.ply")
    (viewer / "splats" / "gone.ply").symlink_to(tmp_path / "nothing.ply")
    index = json.loads((viewer / "splats.json").read_text())
    extra = {"id": "s4-run", "label": "scan 4", "url": "data/splats/s4-run.ply", "trainer": "gsplat"}
    index["splats"].append(extra)
    index["default"] = "s4-run"
    (viewer / "splats.json").write_text(json.dumps(index))

    shutil.rmtree(splat_dir / "old")
    entries = publish_splats(splat_dir, viewer, "msplat")
    assert [e["id"] for e in entries] == ["msplat", "s4-run"]
    assert sorted(p.name for p in (viewer / "splats").iterdir()) == ["msplat.ply", "s4-run.ply"]
    assert (viewer / "splats" / "s4-run.ply").resolve() == (other / "splat.ply").resolve()
    index = json.loads((viewer / "splats.json").read_text())
    assert index["default"] == "msplat" and index["splats"][1] == extra
    # an unknown --trainer keeps the previous default
    publish_splats(splat_dir, viewer, "brush")
    assert json.loads((viewer / "splats.json").read_text())["default"] == "msplat"
