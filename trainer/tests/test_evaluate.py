"""The shared evaluator on the synthetic dataset, with the CPU reference renderer."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from gsplat import export_splats

from gs_trainer import evaluate as ev
from gs_trainer import synth
from gs_trainer.data import Scene, viewmat

SH0 = 0.28209479177387814


@pytest.fixture(autouse=True)
def offline_lpips(monkeypatch: pytest.MonkeyPatch) -> None:
    """LPIPS with a random AlexNet trunk: no torchvision download in tests (the linear heads ship with lpips)."""
    import lpips

    models: dict[str, torch.nn.Module] = {}

    def model(device: torch.device) -> torch.nn.Module:
        if str(device) not in models:
            models[str(device)] = lpips.LPIPS(net="alex", pnet_rand=True, verbose=False).to(device).eval()
        return models[str(device)]

    monkeypatch.setattr(ev, "lpips_model", model)


def synth_splats(data: Path, colour_shift: float = 0.0) -> dict[str, torch.Tensor]:
    """Small isotropic Gaussians on the synthetic room's init points, coloured like them (SH degree 1)."""
    scene = Scene.load(data)
    n = len(scene.points)
    rgb = torch.from_numpy(scene.colors.astype(np.float32) / 255) + colour_shift
    return {
        "means": torch.from_numpy(scene.points.astype(np.float32)),
        "scales": torch.full((n, 3), math.log(0.1)),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        "opacities": torch.full((n,), 3.0),
        "sh0": ((rgb - 0.5) / SH0)[:, None, :],
        "shN": torch.zeros(n, 3, 3),
    }


def write_splat_ply(splats: dict[str, torch.Tensor], path: Path) -> Path:
    s = splats
    export_splats(s["means"], s["scales"], s["quats"], s["opacities"], s["sh0"], s["shN"], save_to=str(path))
    return path


def test_load_ply_matches_gsplat_layout(tmp_path: Path) -> None:
    g = torch.Generator().manual_seed(0)
    n = 50
    splats = {
        "means": torch.randn(n, 3, generator=g),
        "scales": torch.randn(n, 3, generator=g),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.randn(n, generator=g),
        "sh0": torch.randn(n, 1, 3, generator=g),
        "shN": torch.randn(n, 15, 3, generator=g),
    }
    loaded, degree = ev.load_ply(write_splat_ply(splats, tmp_path / "a.ply"))
    assert degree == 3
    for k, v in splats.items():
        torch.testing.assert_close(loaded[k], v)

    # Another trainer's layout: other property order, extra normals, a double, SH degree 0.
    props = [("nx", "<f4"), ("x", "<f8"), ("y", "<f4"), ("z", "<f4"), ("scale_0", "<f4"), ("scale_1", "<f4"),
             ("scale_2", "<f4"), ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
             ("opacity", "<f4"), ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4")]  # fmt: skip
    rec = np.zeros(3, dtype=props)
    rec["x"], rec["opacity"], rec["f_dc_2"] = [1, 2, 3], [0.5, 0.6, 0.7], [7, 8, 9]
    header = "ply\nformat binary_little_endian 1.0\ncomment brush\nelement vertex 3\n"
    header += "".join(f"property {'double' if t == '<f8' else 'float'} {k}\n" for k, t in props)
    (tmp_path / "b.ply").write_bytes((header + "end_header\n").encode() + rec.tobytes())
    loaded, degree = ev.load_ply(tmp_path / "b.ply")
    assert degree == 0 and loaded["shN"].shape == (3, 0, 3)
    assert loaded["means"][:, 0].tolist() == [1, 2, 3]
    assert loaded["sh0"][:, 0, 2].tolist() == [7, 8, 9]
    assert loaded["means"].dtype == torch.float32


def test_depth_errors_count_only_valid_pixels() -> None:
    gt = torch.full((4, 5), 2.0)
    depth = torch.full((4, 5), 2.04)  # 2 % off: inside δ<1.05
    conf = torch.full((4, 5), 2, dtype=torch.uint8)
    alpha = torch.ones(4, 5)
    depth[0, 0] = 2.5  # 25 % off, valid: counts
    gt[1, 0], depth[1, 0] = 0.05, 9.0  # LiDAR too close
    conf[2, 0], depth[2, 0] = 1, 9.0  # low confidence
    alpha[3, 0], depth[3, 0] = 0.3, 9.0  # transparent splat
    gt[3, 1], depth[3, 1] = 0.0, 9.0  # no reading
    rel, close, n = ev.depth_errors(depth, gt, conf, alpha)
    assert n == 20 - 4
    assert close == n - 1
    assert rel / n == pytest.approx((0.25 + 0.02 * (n - 1)) / n, rel=1e-5)


def test_shrink_keeps_the_projection() -> None:
    """The half-size pose fit sees the same scene: a point projects to the same place, scaled."""
    image = torch.rand(48, 64, 3)
    K = torch.tensor([[50.0, 0.0, 31.0], [0.0, 52.0, 25.0], [0.0, 0.0, 1.0]])
    small, k = ev.shrink(image, K, 0.5)
    assert small.shape == (24, 32, 3)
    torch.testing.assert_close(small.mean(), image.mean())
    p = torch.tensor([0.3, -0.2, 2.0])
    uv, uv_small = (K @ p)[:2] / p[2], (k @ p)[:2] / p[2]
    torch.testing.assert_close(uv_small, uv * 0.5)
    assert ev.shrink(image, K, 1.0)[0] is image
    assert (ev.POSE_ALIGN_STEPS, ev.POSE_ALIGN_SCALE) == (100, 0.5)


def test_align_pose_recovers_a_perturbed_camera(synth_dataset: Path) -> None:
    scene = Scene.load(synth_dataset)
    splats = ev.frozen(synth_splats(synth_dataset), "cpu")
    frame = scene.frames[scene.interp[0]]
    K = torch.from_numpy(frame.K).float()
    W, H = frame.w, frame.h
    w2c = viewmat(frame.c2w).float()
    with torch.no_grad():
        target, _, _ = ev.render_one(splats, w2c, K, W, H, 1, "torch", {})
    target = target.clamp(0, 1)
    wrong = ev.se3_exp(torch.tensor([0.01, -0.012, 0.008, 0.02, -0.015, 0.01])) @ w2c
    with torch.no_grad():
        before = ev.psnr(ev.render_one(splats, wrong, K, W, H, 1, "torch", {})[0], target)
    aligned, size = ev.align_pose(splats, target, wrong, K, 1, steps=25, backend="torch", lr=2e-3)
    with torch.no_grad():
        after = ev.psnr(ev.render_one(splats, aligned, K, W, H, 1, "torch", {})[0], target)
    assert after > before + 3, (before, after)
    assert size["rot_deg"] > 0 and size["shift_cm"] > 0


def test_evaluate_scores_and_writes_pages(tmp_path: Path) -> None:
    # Own copy of the synthetic dataset: one still joins interp (stills have no LiDAR), and the first
    # interp frame's pose is off by ~0.6 degrees and ~1.5 cm, which pose alignment should take back.
    data = synth.write_dataset(tmp_path / "data")
    tf = json.loads((data / "transforms.json").read_text())
    split = json.loads((data / "split.json").read_text())
    still = next(Path(f["file_path"]).stem for f in tf["frames"] if f["camera"] == "still")
    split["interp"].append(still)
    (data / "split.json").write_text(json.dumps(split))
    moved = split["interp"][0]
    for f in tf["frames"][len(tf["frames"]) // 2 :]:  # a second session: the per-session breakdown
        f["session"] = "20261007-183715-kitchen"
    for f in tf["frames"]:
        if Path(f["file_path"]).stem == moved:
            nudge = ev.se3_exp(torch.tensor([0.006, -0.008, 0.004, 0.01, -0.008, 0.006], dtype=torch.float64))
            f["transform_matrix"] = (np.asarray(f["transform_matrix"]) @ nudge.numpy()).tolist()
    (data / "transforms.json").write_text(json.dumps(tf))

    scene = Scene.load(data)
    c2w = scene.frames[scene.interp[1]].c2w
    views = tmp_path / "splat_views.json"
    views.write_text(json.dumps({"views": [
        {"name": "ring", "c2w": np.asarray(c2w).tolist(), "fov_y_deg": 55, "w": 80, "h": 60, "note": "a frame"},
    ]}))  # fmt: skip

    # The good splat gets pose alignment on interp only (the slow part on the CPU); the corrupted one
    # (colours shifted) is scored on both sets without it, through the CLI.
    good_ply = write_splat_ply(synth_splats(data), tmp_path / "good.ply")
    bad_ply = write_splat_ply(synth_splats(data, colour_shift=0.3), tmp_path / "bad.ply")
    good = ev.evaluate(data, good_ply, tmp_path / "good", sets=("interp",), pose_align_steps=20, views=views,
                       max_side=64, backend="torch", device="cpu")  # fmt: skip
    assert ev.main(["--data", str(data), "--ply", str(bad_ply), "--out", str(tmp_path / "bad"),
                    "--pose-align-steps", "0", "--max-side", "64", "--backend", "torch",
                    "--device", "cpu", "--views", str(views)]) == 0  # fmt: skip
    bad = json.loads((tmp_path / "bad" / "eval.json").read_text())
    assert ev.main(["--data", str(data), "--ply", str(bad_ply), "--out", str(tmp_path / "noviews"),
                    "--sets", "extrap", "--pose-align-steps", "0", "--max-side", "64", "--backend", "torch",
                    "--device", "cpu", "--views", str(views), "--no-views", "--samples", "3"]) == 0  # fmt: skip
    noviews = json.loads((tmp_path / "noviews" / "eval.json").read_text())
    assert noviews["views"] == [] and not (tmp_path / "noviews" / "views").exists()
    extrap = [scene.frames[i].stem for i in scene.extrap]
    samples = noviews["sets"]["extrap"]["samples"]
    assert [r["stem"] for r in samples] == [extrap[0], extrap[round((len(extrap) - 1) / 2)], extrap[-1]]
    assert all((tmp_path / "noviews" / r["file"]).is_file() for r in samples)
    assert good["sets"]["interp"]["samples"] == [] and not (tmp_path / "good" / "samples").exists()

    assert good["splats"] == len(scene.points) and good["sh_degree"] == 1
    assert (good["pose_align_steps"], good["pose_align_scale"]) == (20, 0.5)
    assert set(good["sets"]) == {"interp"} and set(bad["sets"]) == {"interp", "extrap"}
    for run, name in (("good", "interp"), ("bad", "interp"), ("bad", "extrap")):
        s = (good if run == "good" else bad)["sets"][name]
        assert s["n"] == len(getattr(scene, name)) > 0
        for k in ("psnr", "ssim", "lpips", "abs_rel", "delta105"):
            assert s[k] is not None and math.isfinite(s[k]), (run, name, k)
        assert (s["psnr_aligned"] is None) == (run == "bad")
        assert s["depth_px"] > 0 and 0 <= s["delta105"] <= 1
        assert s["depth_px"] == sum(r["depth_px"] for r in s["per_frame"])
        assert len(s["per_frame"]) == s["n"]
        assert len(s["worst"]) == min(ev.WORST, s["n"])
        for w in s["worst"]:
            assert (tmp_path / run / w["file"]).is_file()
        # Stills have no LiDAR: no depth metrics, none of their pixels in the pooled count.
        for r in s["per_frame"]:
            assert (r["abs_rel"] is None) == (r["camera"] == "still") == (r["depth_px"] == 0)
    interp = good["sets"]["interp"]["per_frame"]
    assert any(r["camera"] == "still" for r in interp)
    assert all(r["psnr_aligned"] >= r["psnr"] - 0.05 for r in interp)
    row = next(r for r in interp if r["stem"] == moved)
    assert row["psnr_aligned"] > row["psnr"] + 1, row
    assert row["pose_correction"]["rot_deg"] > 0.1

    assert good["sets"]["interp"]["psnr"] > bad["sets"]["interp"]["psnr"] + 2
    assert good["sets"]["interp"]["ssim"] > bad["sets"]["interp"]["ssim"]

    # Per session: every frame carries its session; the sessions' aggregates add up to the set's.
    for run, ev_ in (("good", good), ("bad", bad)):
        for name, s in ev_["sets"].items():
            sessions = s["sessions"]
            assert set(sessions) == {r["session"] for r in s["per_frame"]}
            assert sum(g["n"] for g in sessions.values()) == s["n"]
            mean = sum(g["n"] * g["psnr"] for g in sessions.values()) / s["n"]
            assert mean == pytest.approx(s["psnr"], abs=1e-3)
            assert sum(g["depth_px"] for g in sessions.values()) == s["depth_px"]
            pooled = sum(g["abs_rel"] * g["depth_px"] for g in sessions.values() if g["abs_rel"] is not None)
            assert pooled / s["depth_px"] == pytest.approx(s["abs_rel"], abs=1e-4), (run, name)
    assert set(good["sets"]["interp"]["sessions"]) == {synth.SESSION, "20261007-183715-kitchen"}
    assert set(bad["sets"]["extrap"]["sessions"]) == {synth.LAP_SESSION}
    text = (tmp_path / "good" / "eval.json").read_text()
    assert json.loads(text) == good
    assert len(text.splitlines()) < 80 + len(good["sets"]["interp"]["per_frame"])  # a row per frame

    assert (tmp_path / "good" / "views" / "ring.png").is_file()
    assert good["views"][0]["w"] == 64 and good["views"][0]["h"] == 48  # capped at max_side
    page = (tmp_path / "good" / "index.html").read_text()
    assert "views/ring.png" in page and "worst/interp/" in page and "prefers-color-scheme" in page
    assert "interp by session" in page and "20261007-183715-kitchen" in page

    out = ev.compare([tmp_path / "good", tmp_path / "bad"], tmp_path / "cmp")
    assert (
        ev.main(["compare", str(tmp_path / "good"), str(tmp_path / "bad"), "--out", str(tmp_path / "cmp")])
        == 0
    )
    page = out.read_text()
    assert ">good<" in page and ">bad<" in page
    assert page.count("ring.png'") >= 2 and "class='best'" in page
    assert "../good/views/ring.png" in page
    assert "interp by room" in page and ">kitchen<" in page


def _row(stem: str, psnr: float, abs_rel: float | None, px: int, **kw: object) -> dict:
    return {"stem": stem, "camera": "video", "psnr": psnr, "ssim": 0.8, "lpips": 0.3, "abs_rel": abs_rel,
            "delta105": 0.9 if abs_rel is not None else None, "depth_px": px, "psnr_aligned": None, **kw}  # fmt: skip


def test_compare_by_room_on_eval_json_without_sessions(tmp_path: Path) -> None:
    # eval.json as written before the per-session breakdown: no `session` in the rows, no `sessions`. The
    # session comes from the stem; rooms are matched across scans by name; the garden only one scan has is
    # left out of `rooms in every run`.
    def write(name: str, frames: list[dict]) -> Path:
        d = tmp_path / name
        d.mkdir()
        s = {"n": len(frames), "psnr": float(np.mean([f["psnr"] for f in frames])), "per_frame": frames}
        (d / "eval.json").write_text(json.dumps({"ply": f"{name}.ply", "splats": 10, "sh_degree": 3,
                                                 "sets": {"interp": s}, "views": []}))  # fmt: skip
        return d

    s3 = write("s3", [_row("20261007-182728-conservatory_000000", 20.0, 0.02, 100),
                      _row("20261007-182728-conservatory_000008", 22.0, 0.04, 300),
                      _row("20261007-183715-kitchen_000000", 25.0, None, 0)])  # fmt: skip
    s5 = write("s5", [_row("20261009-124955-conservatory_000000", 23.0, 0.01, 100),
                      _row("20261009-125352-garden_000000", 15.0, 0.2, 100),
                      _row("20261009-130247-kitchen_000000", 24.0, 0.03, 100, pose="resolved", psnr_resolved=24.5),
                      _row("20261009-130247-kitchen_000008", 26.0, 0.03, 100, pose="unresolved")])  # fmt: skip

    old = json.loads((s3 / "eval.json").read_text())["sets"]["interp"]
    sessions = ev.set_sessions(old)
    assert list(sessions) == ["20261007-182728-conservatory", "20261007-183715-kitchen"]
    cons = sessions["20261007-182728-conservatory"]
    assert (cons["n"], cons["psnr"], cons["psnr_min"]) == (2, 21.0, 20.0)
    assert cons["worst"] == old["per_frame"][0]["stem"]
    assert cons["abs_rel"] == pytest.approx((0.02 * 100 + 0.04 * 300) / 400)
    assert sessions["20261007-183715-kitchen"]["abs_rel"] is None and cons["n_resolved"] is None
    s5_interp = json.loads((s5 / "eval.json").read_text())["sets"]["interp"]
    kitchen5 = ev.set_sessions(s5_interp)["20261009-130247-kitchen"]
    assert kitchen5["n_resolved"] == 1
    assert (kitchen5["psnr_resolved"], kitchen5["psnr_same_frames"]) == (24.5, 24.0)
    assert ev.session_label("20261009-125352-garden") == "garden"
    assert ev.session_label("synth-ring") == "synth-ring"

    page = ev.compare([s3, s5], tmp_path / "cmp").read_text()
    assert "interp by room" in page
    common = page.split("rooms in every run (conservatory, kitchen)")[1].split("</tr>")[0]
    # scan 3: (20 + 22 + 25) / 3; scan 5 without its garden: (23 + 24 + 26) / 3
    assert f"{67 / 3:.2f}" in common and f"{73 / 3:.2f}" in common
    garden = page.split(">garden<")[1].split("</tr>")[0]
    assert "&ndash;" in garden and "15.00" in garden


def _write_test_poses(data: Path, frames: dict[str, dict], source: dict | None = None) -> Path:
    """A `test_poses.json` as `scanmodel splat` writes it (trainer/README.md, "Dataset v2"), solved from
    `data`'s transforms.json and split.json unless `source` says otherwise."""
    sha = {k: hashlib.sha256((data / f).read_bytes()).hexdigest()
           for k, f in (("transforms_sha256", "transforms.json"), ("split_sha256", "split.json"))}  # fmt: skip
    path = data / ev.TEST_POSES
    path.write_text(json.dumps({"version": 1, "method": "sift-pnp", "source": source or sha, "params": {},
                                "summary": {}, "wall_s": 1.0, "frames": frames}))  # fmt: skip
    return path


def _eval(data: Path, ply: Path, out: Path, **kw: object) -> dict:
    return ev.evaluate(data, ply, out, sets=("extrap",), pose_align_steps=0, max_side=64, backend="torch",
                       device="cpu", **kw)  # type: ignore[arg-type]  # fmt: skip


def test_evaluate_scores_at_re_solved_poses(tmp_path: Path) -> None:
    # The first lap frame's dataset pose is 8.6 deg and 5 cm off; its re-solved pose is the true one. The second
    # failed to re-solve. The splat itself is right, so only the dataset-pose score of the first suffers.
    data = synth.write_dataset(tmp_path / "data")
    split = json.loads((data / "split.json").read_text())
    lap = split["extrap_frames"]
    true = [np.asarray(f["transform_matrix"]) for f in lap]
    turn = ev.se3_exp(torch.tensor([0.0, 0.15, 0.0, 0.05, 0.0, 0.0], dtype=torch.float64)).numpy()
    lap[0]["transform_matrix"] = (true[0] @ turn).tolist()
    (data / "split.json").write_text(json.dumps(split))
    stems = [Path(f["file_path"]).stem for f in lap]
    ok = {
        "status": "ok",
        "refs": [],
        "matches": 200,
        "inliers": 150,
        "reproj_px": 1.0,
        "reproj_dataset_px": 30.0,
    }
    frames: dict[str, dict] = {
        s: {**ok, "transform_matrix": t.tolist()} for s, t in zip(stems, true, strict=True)
    }
    frames[stems[1]] = {
        "status": "unresolved",
        "reason": "12 matches < 40",
        "refs": [],
        "matches": 12,
        "inliers": 0,
    }
    _write_test_poses(data, frames)
    ply = write_splat_ply(synth_splats(data), tmp_path / "splat.ply")

    r = _eval(data, ply, tmp_path / "eval")
    assert r["test_poses"]["status"] == "ok" and r["test_poses"]["method"] == "sift-pnp"
    s = r["sets"]["extrap"]
    assert (s["n"], s["n_resolved"], s["n_unresolved"]) == (len(stems), len(stems) - 1, 1)
    rows = {f["stem"]: f for f in s["per_frame"]}
    assert rows[stems[1]]["pose"] == "unresolved" and "psnr_resolved" not in rows[stems[1]]
    moved = rows[stems[0]]
    assert moved["pose"] == "resolved" and moved["pose_rot_deg"] == pytest.approx(8.59, abs=0.01)
    assert moved["psnr_resolved"] > moved["psnr"] + 3, moved
    for st in stems[2:]:  # the others re-solved to their own dataset pose: the same score
        assert rows[st]["psnr_resolved"] == pytest.approx(rows[st]["psnr"], abs=1e-3)
        assert rows[st]["pose_shift_cm"] == pytest.approx(0, abs=1e-3)
    assert [m["stem"] for m in s["pose_moved"]] == [stems[0]]
    assert s["pose_moved"][0]["reproj_dataset_px"] == 30.0
    res = [rows[st] for st in stems if st != stems[1]]
    assert s["psnr_resolved"] == pytest.approx(np.mean([f["psnr_resolved"] for f in res]), abs=1e-3)
    assert s["psnr_same_frames"] == pytest.approx(np.mean([f["psnr"] for f in res]), abs=1e-3)
    assert s["psnr_unresolved"] == pytest.approx(rows[stems[1]]["psnr"], abs=1e-3)
    assert s["psnr_resolved"] > s["psnr_same_frames"]
    for k in ("ssim_resolved", "lpips_resolved", "abs_rel_resolved", "delta105_resolved"):
        assert s[k] is not None and math.isfinite(s[k]), k
    page = (tmp_path / "eval" / "index.html").read_text()
    assert "PSNR re-solved" in page and "frames whose pose moved" in page

    # Solved from another split.json (the dataset changed since): refused, scored at the dataset poses.
    _write_test_poses(data, frames, source={"transforms_sha256": "x", "split_sha256": "y"})
    r2 = _eval(data, ply, tmp_path / "eval2")
    s2 = r2["sets"]["extrap"]
    assert r2["test_poses"]["status"] == "mismatch"
    assert s2["n_resolved"] is None and s2["psnr_resolved"] is None and s2["pose_moved"] == []
    assert s2["psnr"] == pytest.approx(s["psnr"], abs=1e-6)
    # None at all, or switched off: as before.
    (data / ev.TEST_POSES).unlink()
    assert ev.main(["--data", str(data), "--ply", str(ply), "--out", str(tmp_path / "eval3"), "--sets", "extrap",
                    "--pose-align-steps", "0", "--max-side", "64", "--backend", "torch", "--device", "cpu",
                    "--no-views"]) == 0  # fmt: skip
    r3 = json.loads((tmp_path / "eval3" / "eval.json").read_text())
    assert r3["test_poses"]["status"] == "missing" and r3["sets"]["extrap"]["n_resolved"] is None
    elsewhere = _write_test_poses(data, frames).rename(tmp_path / "elsewhere.json")
    assert _eval(data, ply, tmp_path / "eval4", test_poses=elsewhere)["sets"]["extrap"]["n_resolved"] == 3
    r5 = _eval(data, ply, tmp_path / "eval5", test_poses=elsewhere, use_test_poses=False)
    assert r5["test_poses"]["status"] == "off" and r5["sets"]["extrap"]["n_resolved"] is None
