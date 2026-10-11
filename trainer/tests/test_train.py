"""The training loop on the CPU: the generated CLI, and a short run on the synthetic dataset with the torch
renderer (loss goes down, densification and pruning run, a preempted run resumes exactly where its
checkpoint left off, the PLY round-trips)."""

import json
import math
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from gs_trainer import train
from gs_trainer.evaluate import load_ply


@pytest.fixture
def one_thread() -> Iterator[None]:
    """The dense renderer works on small tensors, where intra-op threads only add contention (badly so
    when other test processes share the machine)."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_cli_is_generated_from_config(tmp_path: Path) -> None:
    cfg = train.parse_config(
        ["--data", "d", "--out", str(tmp_path), *words("--steps 10 --no-prune --pose-opt --color affine")]
    )
    assert cfg.data == Path("d") and cfg.out == tmp_path
    assert (cfg.steps, cfg.prune, cfg.pose_opt, cfg.color) == (10, False, True, "affine")
    assert (cfg.max_side, cfg.strategy, cfg.ssim_lambda, cfg.depth_loss) == (960, "default", 0.2, False)
    assert cfg.views is None
    with pytest.raises(SystemExit):
        train.parse_config(["--data", "d", "--out", "o", "--color", "sepia"])
    help_text = train.build_parser().format_help()
    for flag in ("--depth-lambda", "--no-antialiased", "--resume", "--bilagrid-w", "--mcmc-scale-reg"):
        assert flag in help_text


def test_config_json_paths_are_repo_relative(tmp_path: Path) -> None:
    from gs_trainer import cloud

    data = cloud.REPO / "data" / "fifth" / "splat" / "datasets" / "default"
    saved = train.config_dict(train.parse_config(["--data", str(data), "--out", str(tmp_path / "out")]))
    assert saved["data"] == "data/fifth/splat/datasets/default"
    assert saved["out"] == (tmp_path / "out").as_posix()  # outside the repo: absolute


def words(text: str) -> list[str]:
    return text.split()


def step_losses(log: str) -> dict[int, str]:
    return {int(m[1]): m[2] for m in re.finditer(r"step=\s*(\d+) loss=(\S+)", log)}


def test_smoke_train_prune_resume_export(
    synth_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, one_thread: None
) -> None:
    out = tmp_path / "run"
    args = [
        "--data", str(synth_dataset), "--out", str(out), "--backend", "torch", "--device", "cpu",
        "--steps", "160", "--max-side", "32", "--max-init-points", "400", "--sh-degree", "1",
        "--sh-degree-interval", "40", "--refine-start", "20", "--refine-every", "20",
        "--prune-every", "80", "--prune-until", "160", "--log-every", "40", "--ckpt-every", "80",
        "--depth-loss", "--pose-opt", "--color", "affine", "--flatten-lambda", "0.01",
        "--scale-ratio-lambda", "0.01",
    ]  # fmt: skip

    # A run that dies after its last step (as if preempted): the checkpoint at step 80 survives.
    def preempted(*_: object) -> int:
        raise RuntimeError("preempted")

    monkeypatch.setattr(train, "export_ply", preempted)
    with pytest.raises(RuntimeError, match="preempted"):
        train.main(args)
    monkeypatch.undo()
    first = (out / "train.log").read_text()
    assert json.loads((out / "progress.json").read_text())["status"] == "failed"
    assert torch.load(out / "ckpts" / "latest.pt", weights_only=False)["step"] == 80

    losses = step_losses(first)
    assert sorted(losses) == [40, 80, 120, 160]
    assert float(losses[160]) < 0.8 * float(losses[40])  # it learns
    assert "depth=" in first
    assert "Pruned at step 80" in first
    splat_counts = {int(c.replace(",", "")) for c in re.findall(r"splats=([\d,]+)", first)}
    assert len(splat_counts) > 1  # densification (and pruning) changed the count

    # --resume auto picks the checkpoint up at step 80 and reproduces the lost steps exactly.
    metrics = train.train(train.parse_config([*args, "--resume", "auto"]))
    second = (out / "train.log").read_text()[len(first) :]
    assert "resumed from" in second and "at step 80" in second
    again = step_losses(second)
    assert sorted(again) == [120, 160]
    assert again[120] == losses[120] and again[160] == losses[160]

    progress = json.loads((out / "progress.json").read_text())
    assert (progress["status"], progress["step"], progress["total"]) == ("done", 160, 160)
    assert metrics["steps"] == 160 and metrics["wall_s"] > 0
    assert metrics["pruned"] == sum(r["removed"] for r in metrics["prune_rounds"])
    assert [r["step"] for r in metrics["prune_rounds"]] == [80]  # the first run's round came back
    assert metrics["psnr"] > 10 and 0 < metrics["ssim"] <= 1
    assert metrics["pose_t_max_m"] >= 0
    assert json.loads((out / "metrics.json").read_text())["splats"] == metrics["splats"]
    assert json.loads((out / "config.json").read_text())["steps"] == 160

    splats, degree = load_ply(out / "splat.ply")
    assert degree == 1
    assert len(splats["means"]) == metrics["splats"] == progress["splats"]
    assert splats["shN"].shape == (metrics["splats"], 3, 3)


def test_normals_bilagrid_absgrad_antialiased(synth_dataset: Path, tmp_path: Path, one_thread: None) -> None:
    """The optional paths run together: mesh-prior normal loss (extra render channels), bilateral grid,
    AbsGS gradients, antialiasing, no pruning views."""
    import shutil

    import numpy as np

    data = tmp_path / "data"
    shutil.copytree(synth_dataset, data, ignore=shutil.ignore_patterns("views.npz"))
    (data / "prior").mkdir()
    for img in (data / "images").glob("*.jpg"):
        normal = np.zeros((24, 32, 3), np.int8)
        normal[..., 2] = 127
        np.savez(data / "prior" / f"{img.stem}.npz", depth=np.ones((24, 32), np.float16), normal=normal)
    out = tmp_path / "run"
    metrics = train.train(
        train.parse_config(
            [
                "--data",
                str(data),
                "--out",
                str(out),
                *words(
                    "--backend torch --device cpu --steps 30 --max-side 32 --max-init-points 200 --sh-degree 0"
                    " --log-every 10 --normal-loss --normal-start 0 --color bilagrid --bilagrid-x 4"
                    " --bilagrid-y 4 --bilagrid-w 2 --absgrad --antialiased --refine-start 5 --refine-every 10"
                    " --no-eval"
                ),
            ]
        )
    )
    log = (out / "train.log").read_text()
    assert "normal=" in log and "color=" in log and "free-space pruning off" in log
    assert metrics["pruned"] == 0 and "psnr" not in metrics
    assert (out / "splat.ply").is_file()


@pytest.fixture(scope="module")
def fused_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic dataset with the whole P4 preparation: mesh priors, LiDAR sigma table, (fake) DA3 and
    every fusion mode, as `gs-trainer cloud push --da3` leaves it on the volume."""
    from gs_trainer import calibrate, da3, fusion, priors, synth
    from tests.test_fusion import PROCESS_RES, use_fake

    data = synth.write_dataset(tmp_path_factory.mktemp("fused") / "data")
    priors.prepare(data)
    calibrate.calibrate(data)
    mp = pytest.MonkeyPatch()
    try:
        use_fake(mp, data)  # pyright: ignore[reportArgumentType]
        da3.run(data, window=6, overlap=3, process_res=PROCESS_RES)
    finally:
        mp.undo()
    fusion.fuse(data, "all")
    return data


def depth_config(data: Path, out: Path, target: str, weighting: str = "edge_log") -> train.Config:
    return train.parse_config(
        ["--data", str(data), "--out", str(out), "--depth-loss", "--depth-target", target,
         "--depth-weighting", weighting]
    )  # fmt: skip


def test_depth_targets(fused_dataset: Path, tmp_path: Path) -> None:
    """What each target supervises: the LiDAR only video frames, with its confidence schedule; the fusion
    modes every frame they have values for (stills too, except lidar_only), with 1/sigma weights of mean 1."""
    from gs_trainer.data import Scene

    scene = Scene.load(fused_dataset)
    video = next(f for f in scene.frames if f.camera_idx == 0)
    still = next(f for f in scene.frames if f.camera_idx == 1)
    cpu = torch.device("cpu")
    W, H = 32, 24

    lidar = depth_config(fused_dataset, tmp_path, "lidar")
    assert train.depth_supervision(lidar, still, W, H, cpu, 0) is None
    early = train.depth_supervision(lidar, video, W, H, cpu, 0)
    late = train.depth_supervision(lidar, video, W, H, cpu, lidar.depth_conf_until)
    assert early is not None and late is not None
    assert early[2] is None and late[1].sum() <= early[1].sum()  # confidence 2 only after the switch

    for target, still_supervised in (("fused", True), ("lidar_only", False), ("da3_holes_only", True)):
        cfg = depth_config(fused_dataset, tmp_path, target, "sigma")
        for frame, expect in ((video, True), (still, still_supervised)):
            got = train.depth_supervision(cfg, frame, W, H, cpu, 0)
            assert got is not None
            d, valid, w = got
            assert d.shape == valid.shape == (H, W) and w is not None
            assert bool(valid.any()) == expect, (target, frame.stem)
            assert bool((d[valid] > 0).all()) and bool((w[~valid] == 0).all())
            if expect:
                assert float(w[valid].mean()) == pytest.approx(1.0, rel=1e-4)
        edge = train.depth_supervision(depth_config(fused_dataset, tmp_path, target), video, W, H, cpu, 0)
        assert edge is not None and edge[2] is None  # edge_log: no weights


def test_depth_config_is_checked(synth_dataset: Path, fused_dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="needs a sigma"):
        train.check_depth_config(depth_config(fused_dataset, tmp_path, "lidar", "sigma"))
    with pytest.raises(SystemExit, match="no fused depth priors"):
        train.check_depth_config(depth_config(synth_dataset, tmp_path, "fused"))
    assert train.fused_modes(synth_dataset) == []
    assert train.fused_modes(fused_dataset) == ["fused", "lidar_only", "da3_holes_only"]
    train.check_depth_config(depth_config(fused_dataset, tmp_path, "da3_holes_only", "sigma"))
    with pytest.raises(SystemExit):
        train.parse_config(["--data", "d", "--out", "o", "--depth-target", "mesh"])


def test_train_on_fused_depth_with_sigma_weights(
    fused_dataset: Path, tmp_path: Path, one_thread: None
) -> None:
    out = tmp_path / "run"
    args = [
        "--data", str(fused_dataset), "--out", str(out),
        *words("--backend torch --device cpu --steps 30 --max-side 32 --max-init-points 200 --sh-degree 0"
               " --log-every 10 --no-eval --no-prune --depth-loss --depth-target fused"
               " --depth-weighting sigma"),
    ]  # fmt: skip
    train.train(train.parse_config(args))
    log = (out / "train.log").read_text()
    assert "1/sigma-weighted L1 against fused depth" in log
    depth_terms = [float(x) for x in re.findall(r"depth=(\S+)", log)]
    assert len(depth_terms) == 3 and all(math.isfinite(x) and x > 0 for x in depth_terms)
    config = json.loads((out / "config.json").read_text())
    assert (config["depth_target"], config["depth_weighting"]) == ("fused", "sigma")


def test_surface_losses_and_pruning_after_refine_stop(
    synth_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, one_thread: None
) -> None:
    """P6 a3: depth variance and opacity entropy next to depth, normals (both share the extra render
    channels) and antialiasing; free-space pruning keeps running after the default strategy stops refining
    (ops.remove on parameters, Adam moments and the strategy's frozen state)."""
    import shutil

    import numpy as np

    from gs_trainer import prune

    data = tmp_path / "data"
    shutil.copytree(synth_dataset, data)
    (data / "prior").mkdir()
    for img in (data / "images").glob("*.jpg"):
        normal = np.zeros((24, 32, 3), np.int8)
        normal[..., 2] = 127
        np.savez(data / "prior" / f"{img.stem}.npz", depth=np.ones((24, 32), np.float16), normal=normal)

    real_drop = prune.free_space_drop

    def drop_more(splats: torch.nn.ParameterDict, *args: object) -> torch.Tensor:
        """The real votes plus every 10th splat, so each round removes something."""
        drop = real_drop(splats, *args)  # pyright: ignore[reportArgumentType]
        drop[::10] = True
        return drop

    monkeypatch.setattr(prune, "free_space_drop", drop_more)
    out = tmp_path / "run"
    args = [
        "--data", str(data), "--out", str(out),
        *words("--backend torch --device cpu --steps 130 --max-side 32 --max-init-points 300 --sh-degree 0"
               " --log-every 20 --no-eval --refine-start 10 --refine-every 10 --refine-stop 40"
               " --prune-every 40 --prune-until 125 --depth-loss --normal-loss --normal-start 0"
               " --flatten-lambda 0.01 --scale-ratio-lambda 0.01 --antialiased"
               " --depth-var-lambda 1.0 --depth-var-start 20 --opacity-entropy-lambda 0.01"
               " --opacity-entropy-start 40"),
    ]  # fmt: skip
    metrics = train.train(train.parse_config(args))
    log = (out / "train.log").read_text()
    assert "depth variance loss: lambda 1.0 from step 20" in log
    assert "opacity entropy loss: lambda 0.01 from step 40" in log
    lines = {int(m[1]): m[0] for m in re.finditer(r"step=\s*(\d+) .*", log)}
    assert "dvar=" not in lines[20] and "dvar=" in lines[40]  # the window of steps 20-39 has it
    assert "ent=" not in lines[40] and "ent=" in lines[60]
    for line in lines.values():
        for value in re.findall(r"(?:loss|dvar|ent|normal|depth)=(\S+)", line):
            assert math.isfinite(float(value))
    assert [r["step"] for r in metrics["prune_rounds"]] == [40, 80, 120]
    late = [r for r in metrics["prune_rounds"] if r["step"] > 40]
    assert all(r["removed"] > 0 for r in late)
    # After the refine stop only pruning changes the count: each round removes exactly what it reports.
    assert late[1]["splats"] == late[0]["splats"] - late[0]["removed"]
    assert metrics["splats"] == late[1]["splats"] - late[1]["removed"]
    assert late[0]["splats"] == metrics["prune_rounds"][0]["splats"] - metrics["prune_rounds"][0]["removed"]
