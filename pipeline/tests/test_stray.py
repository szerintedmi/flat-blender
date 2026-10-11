"""Stray importer: converts a stock Stray capture and detects its camera-axis convention."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from scanmodel.geom import GL_CV
from scanmodel.io import load_bundle
from scanmodel.models.common import F64, mat
from scanmodel.stages.ingest import check_bundle
from scanmodel.stages.stray import import_stray

from . import synth


def _quat(r: F64) -> tuple[float, float, float, float]:
    """Rotation matrix -> (qx, qy, qz, qw)."""
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return float(x), float(y), float(z), float(w)


def _write_stray(src_bundle: Path, out: Path, opencv_axes: bool) -> None:
    bundle = load_bundle(src_bundle)
    out.mkdir()
    shutil.copytree(src_bundle / "depth", out / "depth")
    shutil.copytree(src_bundle / "confidence", out / "confidence")
    k = bundle.frames[0].intrinsics
    np.savetxt(out / "camera_matrix.csv", [[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]], delimiter=",")
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw"]
    for f in bundle.frames:
        m = mat(f.camera_to_world) @ (GL_CV if opencv_axes else np.eye(4))
        rows.append(", ".join(str(v) for v in (f.t, f.idx, *m[:3, 3], *_quat(m[:3, :3]))))
    (out / "odometry.csv").write_text("\n".join(rows) + "\n")


@pytest.mark.parametrize("opencv_axes", [False, True])
def test_stray_import_detects_convention(tmp_path: Path, opencv_axes: bool) -> None:
    ref = synth.write_session(tmp_path / "ref", "ref", synth.HALL, 1, None, n_frames=60)
    _write_stray(ref, tmp_path / "stray", opencv_axes)
    dst = import_stray(tmp_path / "stray", tmp_path / "raw", "office")
    report, bundle = check_bundle(dst)
    assert report.ok, report.errors
    assert bundle is not None and bundle.session.importer == "stray"
    assert report.depth_consistency_m is not None and report.depth_consistency_m < 0.01
    # Whatever Stray wrote, the bundle now holds ARKit (OpenGL-axis) poses equal to the originals.
    want = load_bundle(ref).frames
    for a, b in zip(bundle.frames, want, strict=True):
        np.testing.assert_allclose(mat(a.camera_to_world), mat(b.camera_to_world), atol=1e-6)
