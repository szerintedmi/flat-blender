"""The cloud plumbing's pure logic, offline: nothing here talks to Modal."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gs_trainer import cloud

T0 = datetime(2026, 10, 9, 14, 12, 0, tzinfo=UTC)


def make_dataset(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "transforms.json").write_text(json.dumps({"version": 2, "frames": []}))
    (root / "split.json").write_text("{}")
    (root / "points.ply").write_bytes(b"ply points")
    (root / "mesh.ply").write_bytes(b"ply mesh")
    for sub, name, size in (("images", "a.jpg", 10), ("images", "b.jpg", 20), ("depth", "a.png", 5)):
        (root / sub).mkdir(exist_ok=True)
        (root / sub / name).write_bytes(b"x" * size)
    (root / "images" / ".DS_Store").write_bytes(b"junk")
    (root / "prior").mkdir()
    (root / "prior" / "a.npz").write_bytes(b"made on the cloud")
    return root


def test_dataset_files_skip_dotfiles_and_priors(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "d")
    rels = [rel for _, rel in cloud.dataset_files(data)]
    assert rels == [
        "transforms.json", "split.json", "points.ply", "mesh.ply", "images/a.jpg", "images/b.jpg", "depth/a.png"
    ]  # fmt: skip


def test_dataset_hash_is_content_not_place_or_time(tmp_path: Path) -> None:
    a = make_dataset(tmp_path / "a")
    b = tmp_path / "b"
    shutil.copytree(a, b)
    os.utime(b / "images" / "a.jpg", (0, 0))
    (b / "images" / ".DS_Store").write_bytes(b"other junk")
    (b / "prior" / "a.npz").write_bytes(b"different priors")
    h = cloud.dataset_hash(a)
    assert len(h) == 16
    assert cloud.dataset_hash(b) == h

    (b / "transforms.json").write_text(json.dumps({"version": 2, "frames": [1]}))
    assert cloud.dataset_hash(b) != h
    shutil.copy(a / "transforms.json", b / "transforms.json")
    (b / "images" / "b.jpg").write_bytes(b"x" * 21)  # images count by size
    assert cloud.dataset_hash(b) != h
    (b / "images" / "b.jpg").write_bytes(b"x" * 20)
    (b / "views.npz").write_bytes(b"views")  # a file that was missing
    assert cloud.dataset_hash(b) != h


def test_dataset_hash_needs_a_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cloud.dataset_hash(tmp_path)


def test_pipeline_fingerprint_matches_scanmodel_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest's `inputs` must be what `scanmodel export` computes, or every run shows as stale."""
    monkeypatch.syspath_prepend(str(cloud.TRAINER_DIR.parent / "pipeline"))  # the real repo's pipeline
    ref = importlib.import_module("scanmodel.cache")
    data = make_dataset(tmp_path / "repo" / "data" / "p" / "splat" / "datasets" / "default")
    (data / "points.ply").write_bytes(b"p" * (9 * 1024 * 1024))  # above the content-hash limit
    paths = [data / "transforms.json", data / "points.ply"]
    for root in (tmp_path / "repo", tmp_path / "elsewhere"):  # inside the repo (relative), outside
        assert cloud.pipeline_fingerprint(paths, root) == ref.fingerprint(paths, root)
    monkeypatch.setattr(cloud, "REPO", tmp_path / "repo")
    assert cloud.pipeline_fingerprint(paths) == ref.fingerprint(paths, tmp_path / "repo")


def test_run_ids_and_app_names() -> None:
    run_id = cloud.make_run_id("Depth λ 0.5!", datetime(2026, 10, 9, 14, 12, 5))
    assert run_id == "gs-depth-0-5-20261009-141205"
    assert cloud.make_run_id("", T0).startswith("gs-run-")
    assert cloud.app_description(run_id) == f"flat-gs {run_id}"
    assert cloud.run_of_app(cloud.app_description(run_id)) == run_id
    assert cloud.run_of_app("flat-gs prepare abc") is None
    assert cloud.run_of_app("something-else gs-x") is None


def test_step_of() -> None:
    assert cloud.step_of("step=  1200 loss=0.0412 splats=1,204,331 ms=38.2") == 1200
    assert cloud.step_of("2026-10-09 step=5 loss=1") == 5
    assert cloud.step_of("interp: PSNR 23.1") is None


def test_train_arguments() -> None:
    args = cloud.train_args(7000, 960, ["--depth-lambda", "0.5"])
    assert args == ["--steps", "7000", "--max-side", "960", "--depth-lambda", "0.5"]
    assert cloud.train_args(None, None, []) == []
    assert cloud.remote_train_argv("abc", "gs-x-1", args) == [
        "--data", "/data/datasets/abc", "--out", "/runs/gs-x-1", "--resume", "auto", *args
    ]  # fmt: skip


def test_eval_views_file() -> None:
    assert cloud.eval_views_file(["--sets", "interp"]) == (["--sets", "interp"], None)
    assert cloud.eval_views_file(["--views", "/x.json"]) == ([], Path("/x.json"))
    assert cloud.eval_views_file(["--views=/x.json"]) == ([], Path("/x.json"))
    assert cloud.eval_views_file(["--views", "/x.json", "--no-views"]) == (["--no-views"], None)
    with pytest.raises(ValueError, match="needs a file"):
        cloud.eval_views_file(["--views"])


def test_eval_test_poses_are_uploaded_not_hashed(tmp_path: Path) -> None:
    default = tmp_path / cloud.TEST_POSES
    assert cloud.eval_test_poses(["--sets", "extrap"], default) == (["--sets", "extrap"], None)  # no file
    default.write_text("{}")
    assert cloud.eval_test_poses(["--sets", "extrap"], default) == (["--sets", "extrap"], default)
    other = tmp_path / "other.json"
    assert cloud.eval_test_poses(["--test-poses", str(other), "-x"], default) == (["-x"], other)
    assert cloud.eval_test_poses([f"--test-poses={other}"], None) == ([], other)
    assert cloud.eval_test_poses(["--no-test-poses"], default) == (["--no-test-poses"], None)
    # Eval-only: neither part of the dataset's hash nor pushed with it.
    data = tmp_path / "data"
    data.mkdir()
    (data / "transforms.json").write_text("{}")
    h = cloud.dataset_hash(data)
    (data / cloud.TEST_POSES).write_text('{"frames": {}}')
    assert cloud.dataset_hash(data) == h
    assert all(rel != cloud.TEST_POSES for _, rel in cloud.dataset_files(data))


def test_module_cmd_runs_main_without_a_main_guard(tmp_path: Path) -> None:
    (tmp_path / "fake_trainer.py").write_text(
        "import sys\ndef main(argv):\n    print('args', argv)\n    return 3\n"
    )
    cmd = cloud.module_cmd("fake_trainer", ["--data", "/x"])
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(tmp_path)})
    assert r.returncode == 3
    assert r.stdout.strip() == "args ['--data', '/x']"


def test_sweep_configs(tmp_path: Path) -> None:
    toml = tmp_path / "a.toml"
    toml.write_text(
        'name = "depth"\nsteps = 7000\n[args]\ndepth_lambda = 0.5\npose_opt = true\nprune = false\n'
    )
    r = cloud.parse_sweep_config(toml)
    assert (r.name, r.steps, r.gpu) == ("depth", 7000, None)
    assert r.args == ["--depth-lambda", "0.5", "--pose-opt", "--no-prune"]

    js = tmp_path / "b.json"
    js.write_text(json.dumps({"name": "h100", "gpu": "H100", "args": ["--max-side", 1920]}))
    r = cloud.parse_sweep_config(js)
    assert (r.name, r.gpu, r.args) == ("h100", "H100", ["--max-side", "1920"])

    bad = tmp_path / "c.toml"
    bad.write_text("args = []\n")
    with pytest.raises(ValueError):
        cloud.parse_sweep_config(bad)


def test_gpu_price() -> None:
    assert cloud.gpu_price("L40S") == pytest.approx(1.95 + 0.45)
    assert cloud.gpu_price("h100:2") == pytest.approx(2 * 3.95 + 0.45)
    assert cloud.gpu_price("A10G") == pytest.approx(1.10 + 0.45)
    assert cloud.gpu_price(None) == pytest.approx(0.45)
    assert cloud.gpu_price("B200") == 0.0


def iso(dt: datetime) -> str:
    return dt.isoformat()


def test_cost_counts_attempts_including_a_preempted_one() -> None:
    run = {
        "gpu": "L40S",
        "status": "running",
        "submitted_at": iso(T0 - timedelta(minutes=5)),
        "attempts": [
            {"started_at": iso(T0)},  # preempted: no end, counts until the retry starts
            {"started_at": iso(T0 + timedelta(minutes=30))},
        ],
    }
    now = T0 + timedelta(minutes=60)
    assert cloud.billed_seconds(run, now) == pytest.approx(3600)
    assert cloud.cost_usd(run, now) == pytest.approx(2.40)
    assert cloud.elapsed_s(run, now) == pytest.approx(65 * 60)

    done = cloud.finish_attempt(run, 0, True, T0 + timedelta(minutes=45))
    assert cloud.billed_seconds(done, now) == pytest.approx(45 * 60)
    assert cloud.elapsed_s(done, now) == pytest.approx(50 * 60)
    assert cloud.elapsed_s({}, now) is None


def test_run_status() -> None:
    running = {"status": "running", "app_id": "ap-1", "attempts": [{}]}
    assert cloud.run_status(running, {"ap-1"}) == "running"
    assert cloud.run_status({**running, "attempts": [{}, {}]}, {"ap-1"}) == "resumed"
    assert cloud.run_status(running, set()) == "stopped"  # app gone without recording an end
    assert cloud.run_status({"status": "done", "app_id": "ap-1"}, {"ap-1"}) == "done"
    assert cloud.run_status({"status": "failed"}, set()) == "failed"


def test_live_apps_from_modal_app_list() -> None:
    listing = [
        {"app_id": "ap-1", "description": "flat-gs gs-a-1", "state": "ephemeral (detached)", "tasks": "1"},
        {"app_id": "ap-2", "description": "flat-gs gs-b-1", "state": "stopped", "tasks": "0"},
        {"app_id": "ap-3", "description": "flat-gs prepare abc", "state": "ephemeral", "tasks": "1"},
        {"app_id": "ap-4", "description": "other-app", "state": "deployed", "tasks": "0"},
        {"app_id": "ap-5", "description": "flat-gsx", "state": "ephemeral", "tasks": "1"},
        {"app_id": "ap-6", "description": "flat-gs-build", "state": "deployed", "tasks": "1"},
        {"app_id": "ap-7", "description": "flat-gs-gpu-tests", "state": "ephemeral", "tasks": "1"},
        {"app_id": "ap-8", "description": "flat-gs-adhoc", "state": "ephemeral (detached)", "tasks": "0"},
        {"app_id": "ap-9", "description": "flat-gs-build", "state": "stopped", "tasks": "0"},
        {"app_id": "ap-10", "description": "flat-gs", "state": "stopping", "tasks": "1"},
    ]
    live = [a["app_id"] for a in cloud.live_apps(listing)]
    assert live == ["ap-1", "ap-3", "ap-6", "ap-7", "ap-8", "ap-10"]


def test_billing_groups() -> None:
    g = cloud.billing_group
    assert g("flat-gs gs-final-default-geo-20261008-021500") == "training runs"
    assert g("flat-gs eval 3f2a") == "evals"
    assert g("flat-gs prepare 3f2a") == "prepare / depth prior"
    assert g("flat-gs depth-prior 3f2a") == "prepare / depth prior"
    assert g("flat-gs-build") == "image builds"
    assert g("flat-gs-image-build-test") == "image builds"  # an ad-hoc build script
    assert g("flat-gs-gpu-tests") == "tests"
    assert g("flat-gs") == "other flat-gs"
    assert g("flat-gs-scratch") == "other flat-gs"
    assert g("anything", "im-123") == "image builds"
    assert g("my-other-project") == "not flat-gs"
    assert g("flat-gsx-build") == "not flat-gs"


# What `modal billing report --for "this month" --resolution h --json` prints (modal 1.6.1): naive UTC
# hours, costs as decimal strings.
BILLING_JSON = """[
  {"object_id": "ap-1", "description": "flat-gs-build", "environment": "main",
   "interval_start": "2026-10-07T22:00:00", "cost": "4.10"},
  {"object_id": "ap-2", "description": "flat-gs gs-r1-base-20261007-230000", "environment": "main",
   "interval_start": "2026-10-07T23:00:00", "cost": "2.40"},
  {"object_id": "ap-2", "description": "flat-gs gs-r1-base-20261007-230000", "environment": "main",
   "interval_start": "2026-10-08T00:00:00", "cost": "2.40"},
  {"object_id": "ap-3", "description": "flat-gs eval 3f2a", "environment": "main",
   "interval_start": "2026-10-08T01:00:00", "cost": "0.55"},
  {"object_id": "ap-4", "description": "flat-gs-gpu-tests", "environment": "main",
   "interval_start": "2026-10-08T02:00:00", "cost": "0.30"},
  {"object_id": "ap-5", "description": "someone-else", "environment": "main",
   "interval_start": "2026-10-01T10:00:00", "cost": "1.00"},
  {"object_id": "ap-6", "description": "flat-gs prepare 3f2a", "environment": "main",
   "interval_start": "not a time", "cost": "9.99"}
]"""


def test_summarize_bill_groups_and_sums() -> None:
    from decimal import Decimal

    rows = json.loads(BILLING_JSON)
    bill = cloud.summarize_bill(rows, datetime(2026, 10, 8).date(), tz=UTC)
    assert bill.rows == 6  # the malformed row is skipped
    assert bill.month == Decimal("10.75")
    assert bill.today == Decimal("3.25")  # 00:00 UTC onwards
    assert list(bill.by_group) == ["training runs", "evals", "image builds", "tests", "not flat-gs"]
    assert bill.by_group["training runs"] == (Decimal("4.80"), Decimal("2.40"))
    assert bill.by_group["image builds"] == (Decimal("4.10"), Decimal(0))
    assert bill.through == datetime(2026, 10, 8, 3, tzinfo=UTC)

    # "today" is the local calendar day: at UTC+2 the 22:00 and 23:00 UTC hours are already the 8th
    from datetime import timezone

    plus2 = cloud.summarize_bill(rows, datetime(2026, 10, 8).date(), tz=timezone(timedelta(hours=2)))
    assert plus2.today == Decimal("9.75")
    assert cloud.summarize_bill([], datetime(2026, 10, 8).date()).through is None


def test_show_bill_survives_a_failing_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        raise cloud.CloudError("`modal billing report` failed: no permission")

    monkeypatch.setattr(cloud, "_modal_cli", broken)
    with cloud.console.capture() as cap:
        cloud.show_bill()
    assert "no actual bill" in cap.get()

    def garbage(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(args), 0, stdout="not json", stderr="")

    monkeypatch.setattr(cloud, "_modal_cli", garbage)
    with cloud.console.capture() as cap:
        cloud.show_bill()
    assert "no actual bill" in cap.get()


def test_show_bill_prints_totals(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, ...]] = []

    def canned(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        seen.append(args)  # the month's days: nothing; today's hours: the canned rows
        stdout = BILLING_JSON if "today" in args else "[]"
        return subprocess.CompletedProcess(list(args), 0, stdout=stdout, stderr="")

    monkeypatch.setattr(cloud, "_modal_cli", canned)
    with cloud.console.capture() as cap:
        cloud.show_bill()
    out = cap.get()
    # hourly reports can't span more than 7 days: the month comes daily, today hourly
    assert seen == [
        ("billing", "report", "--for", "this month", "--resolution", "d", "--json"),
        ("billing", "report", "--for", "today", "--resolution", "h", "--json"),
    ]
    assert "$10.75" in out and "image builds" in out and "current hour" in out


def test_show_apps_lists_tasks() -> None:
    apps = [{"app_id": "ap-6", "description": "flat-gs-build", "state": "deployed", "tasks": "1"}]
    with cloud.console.capture() as cap:
        cloud.show_apps(apps)
    assert "flat-gs-build" in cap.get()
    with cloud.console.capture() as cap:
        cloud.show_apps([])
    assert "nothing running" in cap.get()


def test_run_json_through_attempts() -> None:
    meta = {"run_id": "gs-a-1", "app_id": "ap-1", "gpu": "L40S", "args": ["--steps", "10"]}
    env = {"MODAL_IMAGE_ID": "im-1", "MODAL_TASK_ID": "ta-1", "MODAL_REGION": "us-east"}
    first = cloud.start_attempt({}, meta, T0, env)
    assert first["status"] == "running"
    assert first["started_at"] == iso(T0)
    assert first["image_id"] == "im-1"
    assert first["attempts"] == [{"started_at": iso(T0), "task_id": "ta-1"}]

    # a retry after a preemption reads the old run.json: same start, one more attempt
    later = T0 + timedelta(minutes=20)
    second = cloud.start_attempt(first, meta, later, {**env, "MODAL_TASK_ID": "ta-2"})
    assert second["started_at"] == iso(T0)
    assert [a["task_id"] for a in second["attempts"]] == ["ta-1", "ta-2"]

    failed = cloud.finish_attempt(second, 0, False, later)  # exit 0 but no PLY
    assert failed["status"] == "failed"
    done = cloud.finish_attempt(second, 0, True, later)
    assert done["status"] == "done"
    assert done["attempts"][-1]["returncode"] == 0
    assert done["finished_at"] == iso(later)


def test_manifest_is_what_scanmodel_export_reads(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "dataset")
    run_dir = tmp_path / "gs-a-1"
    run_dir.mkdir()
    (run_dir / "splat.ply").write_bytes(b"p" * 2_000_000)
    run = {
        "run_id": "gs-a-1",
        "dataset": "abc",
        "gpu": "L40S",
        "steps": 7000,
        "args": ["--steps", "7000"],
        "git": {"commit": "3a2e6f4", "diff_sha256": "0" * 64},
        "started_at": iso(T0),
        "finished_at": iso(T0 + timedelta(hours=1)),
        "attempts": [{"started_at": iso(T0), "finished_at": iso(T0 + timedelta(hours=1))}],
    }
    metrics = {"psnr": 24.1, "ssim": 0.81, "splats": 1_200_000, "wall_s": 3500.0, "per_frame": [1, 2]}
    m = cloud.run_manifest(run, metrics, run_dir, data)
    assert m["params"]["trainer"] == "gsplat"  # publish_splats skips manifests without it
    assert m["inputs"] == cloud.pipeline_fingerprint([data / "transforms.json", data / "points.ply"])
    assert m["finished_at"] == run["finished_at"]
    s = m["summary"]
    assert (s["psnr"], s["splats"], s["ply_mb"], s["cost_usd"]) == (24.1, 1_200_000, 2.0, 2.40)
    assert "per_frame" not in s
    json.dumps(m)


def test_follow_lines_tracks_steps() -> None:
    lines = ["loading 16 frames\n", "step=   100 loss=0.3\n", "step=   200 loss=0.2\n", "done\n"]
    assert cloud.follow_lines(lines, 200, "gs-a-1") == 200


def test_fmt_duration() -> None:
    assert cloud.fmt_duration(3725.9) == "1:02:05"
    assert cloud.fmt_duration(None) == "-"


def test_cloud_train_passes_options_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gs_trainer import cli

    seen: dict[str, object] = {}

    def fake_train(data, out, name, steps, gpu, max_side, detach, extra):
        seen.update(data=data, out=out, name=name, steps=steps, gpu=gpu, max_side=max_side, detach=detach)
        seen["extra"] = extra
        return 0

    monkeypatch.setattr(cloud, "train", fake_train)
    argv = ["cloud", "train", "--data", str(tmp_path), "--out", str(tmp_path / "o"), "--name", "depth",
            "--steps", "7000", "--gpu", "H100", "--max-side", "960", "--detach",
            "--", "--depth-lambda", "0.5", "--no-prune"]  # fmt: skip
    r = CliRunner().invoke(cli.app, argv)
    assert r.exit_code == 0, r.output
    assert seen == {
        "data": tmp_path, "out": tmp_path / "o", "name": "depth", "steps": 7000, "gpu": "H100",
        "max_side": 960, "detach": True, "extra": ["--depth-lambda", "0.5", "--no-prune"],
    }  # fmt: skip


def test_cli_imports_without_torch() -> None:
    code = "import sys, gs_trainer.cli; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_modal_app_builds_offline() -> None:
    """Defining the app, image, volumes and functions must not need a Modal token or network."""
    import modal

    from gs_trainer import modal_app

    assert modal_app.app.name == "flat-gs"
    fns = (
        modal_app.prepare, modal_app.depth_prior, modal_app.pack_prior, modal_app.train, modal_app.evaluate,
        modal_app.gpu_test,
    )  # fmt: skip
    for fn in fns:
        assert isinstance(fn, modal.Function)


def test_cloud_gpu_test_passes_pytest_args(monkeypatch: pytest.MonkeyPatch) -> None:
    from gs_trainer import cli

    seen: list[list[str]] = []

    def fake_gpu_test(args):
        seen.append(list(args))
        return 3

    monkeypatch.setattr(cloud, "gpu_test", fake_gpu_test)
    r = CliRunner().invoke(cli.app, ["cloud", "gpu-test", "--", "-k", "render", "-x"])
    assert r.exit_code == 3, r.output
    assert seen == [["-k", "render", "-x"]]


def test_depth_target_of_train_args() -> None:
    assert cloud.depth_target_of([]) == "lidar"
    assert cloud.depth_target_of(["--depth-loss", "--depth-target", "fused"]) == "fused"
    assert cloud.depth_target_of(["--depth-target=lidar_only", "--depth-target", "da3_holes_only"]) == (
        "da3_holes_only"
    )
    assert cloud.depth_target_of(["--depth-target"]) == "lidar"


def test_fusion_constants_match_the_modules() -> None:
    from gs_trainer import da3, fusion

    assert cloud.FUSION_MODES == fusion.MODES
    assert cloud.DA3_MODEL == da3.DEFAULT_MODEL


def test_needs_depth_prior() -> None:
    full = {m: {} for m in cloud.FUSION_MODES} | {"version": 1}
    meta = {"model": cloud.DA3_MODEL}
    assert cloud.needs_depth_prior(None, None, cloud.DA3_MODEL)
    assert cloud.needs_depth_prior({"fused": {}}, meta, cloud.DA3_MODEL)
    assert not cloud.needs_depth_prior(full, meta, cloud.DA3_MODEL)
    assert cloud.needs_depth_prior(full, meta, "depth-anything/DA3-GIANT-1.1")
    assert cloud.fused_modes_of(full) == list(cloud.FUSION_MODES) and cloud.fused_modes_of(None) == []


def test_cloud_train_refuses_a_fused_target_without_priors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Before any GPU starts: a fusion target the volume doesn't have, or one that doesn't exist."""
    started: list[str] = []
    reports: dict[str, dict | None] = {"fusion": None}
    monkeypatch.setattr(cloud, "push", lambda data, *a: "abc123")
    monkeypatch.setattr(cloud, "report_running", lambda: None)
    monkeypatch.setattr(cloud, "data_volume", lambda: None)
    monkeypatch.setattr(cloud, "_read_json", lambda vol, path: reports["fusion"])
    monkeypatch.setattr(cloud, "start_train", lambda *a, **k: started.append("run") or ("gs-x", None))
    with pytest.raises(cloud.CloudError, match="push --da3"):
        cloud.train(tmp_path, None, "x", 10, "L40S", None, True, ["--depth-loss", "--depth-target", "fused"])
    with pytest.raises(cloud.CloudError, match="unknown --depth-target mesh"):
        cloud.check_depth_targets("abc123", [["--depth-target", "mesh"]])
    reports["fusion"] = {"fused": {}, "lidar_only": {}}
    with pytest.raises(cloud.CloudError, match="da3_holes_only"):
        cloud.check_depth_targets(
            "abc123", [["--depth-target", "fused"], ["--depth-target", "da3_holes_only"]]
        )
    assert started == []
    assert cloud.train(tmp_path, None, "x", 10, "L40S", None, True, ["--depth-target", "lidar_only"]) == 0
    assert cloud.train(tmp_path, None, "x", 10, "L40S", None, True, ["--depth-loss"]) == 0
    assert started == ["run", "run"]


def test_cloud_push_takes_da3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gs_trainer import cli

    seen: list[tuple] = []
    monkeypatch.setattr(cloud, "push", lambda *a: seen.append(a) or "abc123")
    assert CliRunner().invoke(cli.app, ["cloud", "push", "--data", str(tmp_path)]).exit_code == 0
    r = CliRunner().invoke(cli.app, ["cloud", "push", "--data", str(tmp_path), "--da3"])
    assert r.exit_code == 0 and "abc123" in r.output
    assert seen == [(tmp_path, False, cloud.DA3_MODEL), (tmp_path, True, cloud.DA3_MODEL)]


def test_upload_plan_and_marker(tmp_path: Path) -> None:
    from gs_trainer.datapack import TarStats

    data = make_dataset(tmp_path / "d")
    files = cloud.dataset_files(data)
    assert cloud.upload_plan(None) == "all"
    old = {"hash": "h", "files": 7, "bytes": 99, "from": "/x", "at": "then"}  # pushed before the tars
    assert cloud.upload_plan(old) == "bulk"
    m = cloud.complete_marker(old, "h", [], data, TarStats(3, 35), T0)
    assert (m["files"], m["bytes"], m["at"]) == (7, 99, "then") and m["bulk_tar"]["files"] == 3
    assert cloud.upload_plan(m) == "none"
    m = cloud.complete_marker(None, "h", files, data, TarStats(3, 35), T0)
    assert m["files"] == len(files) and m["bytes"] == sum(p.stat().st_size for p, _ in files)
    assert cloud.upload_plan(m) == "none"
    assert cloud.billing_group("flat-gs pack-prior abc") == "prepare / depth prior"


class FakeVolume:
    """The few `modal.Volume` calls `push` makes, on a dict of path -> bytes."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = dict(files or {})
        self.batches: list[list[str]] = []

    def batch_upload(self, force: bool = False):
        vol = self

        class Batch:
            def __enter__(self):
                self.paths: list[str] = []
                return self

            def put_file(self, src, dst: str) -> None:
                data = src.read() if hasattr(src, "read") else Path(src).read_bytes()
                vol.files[dst.lstrip("/")] = data
                self.paths.append(dst.lstrip("/"))

            def __exit__(self, *exc) -> None:
                vol.batches.append(self.paths)

        return Batch()


def _fake_push_env(monkeypatch: pytest.MonkeyPatch, vol: FakeVolume) -> list[str]:
    remote: list[str] = []
    monkeypatch.setattr(cloud, "data_volume", lambda: vol)
    monkeypatch.setattr(
        cloud, "_exists", lambda v, path: any(k == path or k.startswith(path + "/") for k in vol.files)
    )
    monkeypatch.setattr(
        cloud, "_read_json", lambda v, path: json.loads(vol.files[path]) if path in vol.files else None
    )

    def prepare(h: str, args=()) -> None:
        remote.append("prepare")
        for name in ("meta.json", "sigma.json"):
            vol.files[f"datasets/{h}/prior/{name}"] = b"{}"
        vol.files[f"datasets/{h}/prior.tar"] = b"tar"

    def pack_prior(h: str) -> None:
        remote.append("pack_prior")
        vol.files[f"datasets/{h}/prior.tar"] = b"tar"

    monkeypatch.setattr(cloud, "prepare_remote", prepare)
    monkeypatch.setattr(cloud, "pack_prior_remote", pack_prior)
    return remote


def test_push_uploads_files_and_bulk_tar_then_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import io
    import tarfile

    data = make_dataset(tmp_path / "d")
    vol = FakeVolume()
    remote = _fake_push_env(monkeypatch, vol)
    h = cloud.push(data)
    prefix = f"datasets/{h}"
    files, marker = vol.batches
    assert marker == [f"{prefix}/.complete"]  # last, on its own
    assert f"{prefix}/bulk.tar" in files and f"{prefix}/images/a.jpg" in files
    with tarfile.open(fileobj=io.BytesIO(vol.files[f"{prefix}/bulk.tar"]), mode="r:") as t:
        assert t.getnames() == ["images/a.jpg", "images/b.jpg", "depth/a.png"]
    assert json.loads(vol.files[f"{prefix}/.complete"])["bulk_tar"]["files"] == 3
    assert remote == ["prepare"]  # prepare writes prior.tar itself

    vol.batches.clear()
    assert cloud.push(data) == h  # nothing to do
    assert vol.batches == [] and remote == ["prepare"]


def test_push_adds_the_tars_to_a_dataset_from_before(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "d")
    h = cloud.dataset_hash(data)
    prefix = f"datasets/{h}"
    old = {"hash": h, "files": 7, "bytes": 52, "from": "/x", "at": "then"}
    vol = FakeVolume({
        f"{prefix}/.complete": json.dumps(old).encode(),
        f"{prefix}/transforms.json": b"{}",
        f"{prefix}/prior/meta.json": b"{}",
        f"{prefix}/prior/sigma.json": b"{}",
    })  # fmt: skip
    remote = _fake_push_env(monkeypatch, vol)
    assert cloud.push(data) == h
    assert vol.batches == [[f"{prefix}/bulk.tar"], [f"{prefix}/.complete"]]  # just the tar
    marker = json.loads(vol.files[f"{prefix}/.complete"])
    assert marker["files"] == 7 and marker["at"] == "then" and "bulk_tar" in marker
    assert remote == ["pack_prior"]
