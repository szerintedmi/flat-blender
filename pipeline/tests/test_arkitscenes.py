"""ARKitScenes importer and readers on a tiny synthetic capture in ARKitScenes' layout."""

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from scanmodel.geom import arkit_to_pipeline
from scanmodel.models.common import F64, mat
from scanmodel.stages import arkitscenes as ark
from scanmodel.stages.ingest import check_bundle

from . import synth

VIDEO = "40000001"
K = synth.VIDEO_K.scaled(640, 480)


def _write_capture(src: Path, n: int = 40) -> list[tuple[float, F64]]:
    """traj at 10 Hz; depth / confidence at 60 Hz stamps (only every 6th has a pose); vga at 30 Hz."""
    quads = synth.world_quads()
    poses = [(1000.0 + i / 10, m) for i, m in enumerate(synth.trajectory(synth.HALL, n))]
    src.mkdir(parents=True)
    ark.write_traj(src / "lowres_wide.traj", poses)
    for folder in ("lowres_depth", "confidence", "vga_wide", "vga_wide_intrinsics"):
        (src / folder).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for t, c2w in poses:
        for dt in (0.0, 1 / 60, 2 / 60):  # the in-between stamps have no pose
            name = f"{VIDEO}_{t + dt:.3f}"
            depth = synth.raycast(c2w, quads, synth.VIDEO_K, synth.DEPTH_W, synth.DEPTH_H)
            cv2.imwrite(str(src / "lowres_depth" / f"{name}.png"), np.round(depth * 1000).astype(np.uint16))
            cv2.imwrite(str(src / "confidence" / f"{name}.png"), np.where(depth > 0, 2, 0).astype(np.uint8))
        name = f"{VIDEO}_{t + 0.001:.3f}"  # colour 1 ms off the pose
        cv2.imwrite(str(src / "vga_wide" / f"{name}.png"), rng.integers(0, 255, (480, 640, 3), np.uint8))
        (src / "vga_wide_intrinsics" / f"{name}.pincam").write_text(f"640 480 {K.fx} {K.fy} {K.cx} {K.cy}")
    return poses


def test_traj_round_trip(tmp_path: Path) -> None:
    poses = [(5.0, m) for m in synth.trajectory(synth.HALL, 5)]
    ark.write_traj(tmp_path / "a.traj", poses)
    for (t, a), (u, c) in zip(poses, ark.read_traj(tmp_path / "a.traj"), strict=True):
        assert t == u
        np.testing.assert_allclose(a, c, atol=1e-6)


def test_import_keeps_the_3dod_frame(tmp_path: Path) -> None:
    src = tmp_path / "raw" / VIDEO
    poses = _write_capture(src)
    dst = ark.import_video(src, tmp_path / "bundles")
    report, bundle = check_bundle(dst)
    assert report.ok, report.errors
    assert bundle is not None and bundle.session.importer == "arkitscenes"
    assert len(bundle.frames) == len(poses)  # one per pose, not one per depth map
    assert report.depth_consistency_m is not None and report.depth_consistency_m < 0.005
    # What ingest makes of the bundle is the traj's own z-up frame and OpenCV camera, exactly.
    for f, (t, c2w) in zip(bundle.frames, poses, strict=True):
        assert f.t == t
        np.testing.assert_allclose(arkit_to_pipeline(mat(f.camera_to_world)), c2w, atol=1e-6)
    assert (dst / "video.mov").is_file()


def test_boxes_and_scan_pose(tmp_path: Path) -> None:
    a = np.radians(30)
    r = [np.cos(a), np.sin(a), 0, -np.sin(a), np.cos(a), 0, 0, 0, 1]  # rows: the box's axes
    box = {"centroid": [1, 2, 0.5], "axesLengths": [0.6, 0.4, 1.0], "normalizedAxes": r}
    tilted = {"centroid": [0, 0, 0], "axesLengths": [1, 1, 1], "normalizedAxes": [1, 0, 0, 0, 0, 1, 0, -1, 0]}
    data = {"data": [{"label": "cabinet", "segments": {"obbAligned": o}} for o in (box, tilted)]}
    (tmp_path / "a.json").write_text(json.dumps(data))
    (got,) = ark.read_boxes(tmp_path / "a.json")
    assert got.label == "cabinet" and got.size == (0.6, 0.4, 1.0) and abs(got.yaw_deg - 30) < 1e-9

    m = ark.read_scan_pose("0,1,0,0\n-1,0,0,0\n0,0,1,0\n5,6,7,1\n")
    np.testing.assert_allclose(m @ [1, 0, 0, 1], [5, 7, 7, 1])  # rotation +90 deg about z, then t


def test_wanted_picks_the_nearest_stamp() -> None:
    names = ["d/", "d/v_10.000.png", "d/v_10.017.png", "d/v_10.100.png", "d/v_10.300.png"]
    assert ark.wanted(names, [10.001, 10.1, 10.2], 0.005) == ["d/v_10.000.png", "d/v_10.100.png"]


def test_ply_records_and_zip_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("red", "u1"), ("quality", "<f8")])
    v = np.zeros(3, dt)
    v["x"], v["z"] = [1.0, 20.0, np.nan], [0.5, 0.0, 0.0]
    head = b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty double x\nproperty double y\n"
    head += b"property double z\nproperty uchar red\nproperty double quality\nelement face 0\n"
    head += b"property list uchar int vertex_indices\nend_header\n"
    start, got_dt, count = ark.ply_header(head + v.tobytes())
    assert (start, got_dt, count) == (len(head), dt, 3)
    np.testing.assert_allclose(ark.thin_records(v, np.zeros(3)), [[1.0, 0.0, 0.5]])  # far and NaN dropped

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lowres_depth/v_1.000.png", b"abc" * 100)
        zf.writestr("../evil.png", b"x")
    blob = buf.getvalue()
    # The server, as range requests see it.
    monkeypatch.setattr(ark, "_size", lambda url: len(blob))
    monkeypatch.setattr(ark, "_read", lambda url, first, last: blob[first : min(last, len(blob) - 1) + 1])
    members = ark.zip_members("https://example.invalid/a.zip")
    ark.fetch_members("https://example.invalid/a.zip", members[:1], tmp_path / "out")
    assert (tmp_path / "out" / "lowres_depth" / "v_1.000.png").read_bytes() == b"abc" * 100
    with pytest.raises(ValueError, match="unsafe"):
        ark.fetch_members("https://example.invalid/a.zip", members[1:], tmp_path / "out")


def test_laser_cloud_frames(tmp_path: Path) -> None:
    """Two scans of one room in venue coordinates far from the origin (as released), and the same scans
    stored in their scanner frames (second scanner turned 90 deg): both give one consistent cloud."""
    rng = np.random.default_rng(1)
    room = np.add(
        np.vstack(
            [
                rng.uniform([0, 0, 0], [4, 3, 0.001], (4000, 3)),
                rng.uniform([0, 0, 0], [0.001, 3, 2.5], (3000, 3)),
                rng.uniform([0, 0, 0], [4, 0.001, 2.5], (3000, 3)),
            ]
        ),
        [50.0, 20.0, 140.0],
    )
    for stored in ("venue", "scanner"):
        d = tmp_path / stored
        d.mkdir()
        for i, (deg, t) in enumerate(((0.0, [51.0, 21.0, 141.0]), (90.0, [53.0, 22.0, 141.2]))):
            c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]], t
            pts = room if stored == "venue" else (room - m[:3, 3]) @ m[:3, :3]
            np.savez(d / f"{i}.npz", points=pts, pose=m)
        pts, nrm = ark.laser_cloud(d)
        np.testing.assert_allclose(pts[: len(room)], room, atol=1e-6)
        np.testing.assert_allclose(pts[len(room) :], room, atol=1e-6)
        assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)
