from pathlib import Path

import pytest
from typer.testing import CliRunner

from gs_trainer import cli, cloud


@pytest.mark.parametrize("command", ["push", "train", "sweep", "pull", "eval"])
def test_cloud_commands_select_project(command: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def capture(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return [] if command == "sweep" else 0

    target = "evaluate" if command == "eval" else command
    monkeypatch.setattr(cloud, target, capture)
    monkeypatch.setattr(cloud, "report_running", lambda: None)
    extra = {"sweep": ["recipe.toml"], "pull": ["gs-run"], "eval": ["--ply", "example.ply"]}.get(command, [])
    result = CliRunner().invoke(cli.app, ["cloud", command, *extra, "--project", "fifth"])
    assert result.exit_code == 0, result.output
    expected = cloud.REPO / "data" / "fifth" / "splat" / "datasets" / "default"
    if command != "pull":
        assert seen["args"][0] == expected
    if command != "push":
        assert seen["kwargs"]["project"] == "fifth"


def test_project_settings_and_explicit_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (cloud.REPO / "settings.toml").write_text('project = "third"\n')
    seen: dict = {}

    def capture(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return 0

    monkeypatch.setattr(cloud, "train", capture)
    result = CliRunner().invoke(
        cli.app,
        ["cloud", "train", "--data", str(tmp_path), "--out", str(tmp_path / "custom"), "--project", "fifth"],
    )
    assert result.exit_code == 0, result.output
    assert seen["args"][:2] == (tmp_path, tmp_path / "custom")
    assert seen["kwargs"] == {"project": "fifth"}
    result = CliRunner().invoke(cli.app, ["cloud", "train"])
    assert result.exit_code == 0, result.output
    assert seen["args"][0] == cloud.project_dataset("third")
    assert seen["kwargs"] == {"project": "third"}


def test_no_environment_variable_or_dotenv_selects_a_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLAT_PROJECT", "third")
    (cloud.REPO / ".env").write_text("FLAT_PROJECT=third\n")
    assert cloud.selected_project() is None
    with pytest.raises(cloud.CloudError, match=r"settings\.toml"):
        cloud.project_dataset()


def test_project_eval_run_keeps_trained_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []
    monkeypatch.setattr(cloud, "evaluate", lambda *args, **kwargs: seen.append((args, kwargs)))
    result = CliRunner().invoke(cli.app, ["cloud", "eval", "--run", "gs-old", "--project", "fifth"])
    assert result.exit_code == 0, result.output
    assert seen[0][0][0] is None
    assert seen[0][1] == {"project": "fifth"}


def test_project_paths_and_local_passthrough() -> None:
    dataset = cloud.project_dataset("fifth")
    run = cloud.REPO / "data" / "fifth" / "splat" / "runs" / "gs-test"
    assert cloud.default_run_dir("gs-test", project="fifth") == run
    assert cloud.default_run_dir("gs-test", data=dataset) == run
    assert cli.local_args(["--steps", "1"], "fifth", output="gs-test") == [
        "--steps",
        "1",
        "--data",
        str(dataset),
        "--out",
        str(run),
    ]
    assert cli.local_args(["--data=x", "--out", "y"], "fifth", output="gs-test") == ["--data=x", "--out", "y"]
    assert cli.local_args(["--help"], "fifth", output="gs-test") == ["--help"]
    with pytest.raises(cloud.CloudError, match="simple directory"):
        cloud.project_dataset("../other")


def test_pull_restores_run_project_and_explicit_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cloud, "REPO", tmp_path)
    saved = tmp_path / "custom-output"
    meta = {"project": "fifth", "data_dir": str(cloud.project_dataset("fifth")), "local_out": str(saved)}
    monkeypatch.setattr(cloud, "_run_json", lambda _: meta)
    monkeypatch.setattr(cloud, "runs_volume", lambda: object())
    monkeypatch.setattr(cloud, "_download_dir", lambda *args, **kwargs: [])
    assert cloud.pull("gs-run") == saved
    assert cloud.pull("gs-run", project="third") == tmp_path / "data" / "third" / "splat" / "runs" / "gs-run"
    meta.pop("local_out")
    assert cloud.pull("gs-run") == tmp_path / "data" / "fifth" / "splat" / "runs" / "gs-run"


@pytest.mark.parametrize("command", ["train", "evaluate", "prepare"])
def test_local_commands_select_project(command: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from gs_trainer import calibrate, evaluate, priors, train

    seen: list[list[str]] = []

    def capture(args):
        seen.append(args)
        return 0

    module = {"train": train, "evaluate": evaluate, "prepare": priors}[command]
    monkeypatch.setattr(module, "main", capture)
    monkeypatch.setattr(calibrate, "main", capture)
    result = CliRunner().invoke(cli.app, [command, "--project", "fifth"])
    assert result.exit_code == 0, result.output
    assert seen[0][:2] == ["--data", str(cloud.project_dataset("fifth"))]


@pytest.mark.parametrize(
    ("old", "new", "dataset"),
    [
        ("derived/splat/dataset", "third/splat/datasets/default", True),
        ("derived/splat/dataset-refined", "third/splat/datasets/refined", True),
        ("derived/fourth/derived/splat/dataset", "fourth/splat/datasets/default", True),
        ("derived/fifth/derived/splat/dataset-refined-fused", "fifth/splat/datasets/refined-fused", True),
        ("derived/splat/gs-example", "third/splat/runs/gs-example", False),
        ("derived/fifth/derived/splat/brush", "fifth/splat/runs/brush", False),
    ],
)
def test_missing_stored_legacy_paths_resolve(old: str, new: str, dataset: bool) -> None:
    old_path = cloud.REPO / "data" / old
    assert cloud.stored_local_path(str(old_path), dataset=dataset) == cloud.REPO / "data" / new
    old_path.mkdir(parents=True)
    assert cloud.stored_local_path(str(old_path), dataset=dataset) == old_path


def test_unrelated_stored_paths_stay_unchanged() -> None:
    for path in (
        cloud.REPO / "custom",
        Path("/elsewhere/data/derived/splat/dataset"),
        cloud.REPO / "data/derived/other/splat/dataset",
    ):
        assert cloud.stored_local_path(str(path), dataset=True) == path
    assert cloud.stored_local_path(None, dataset=True) is None


def test_pull_uses_migrated_metadata_without_rewriting_explicit_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_data = cloud.REPO / "data/derived/fifth/derived/splat/dataset"
    legacy_out = cloud.REPO / "data/derived/fifth/derived/splat/gs-example"
    meta = {"data_dir": str(legacy_data), "local_out": str(legacy_out)}
    monkeypatch.setattr(cloud, "_run_json", lambda _: meta)
    monkeypatch.setattr(cloud, "runs_volume", lambda: object())

    def download(volume, run, dest, **kwargs):
        (dest / "splat.ply").touch()
        return []

    seen: list[Path] = []

    def manifest(run, metrics, dest, data):
        seen.append(data)
        return {}

    monkeypatch.setattr(cloud, "_download_dir", download)
    monkeypatch.setattr(cloud, "run_manifest", manifest)
    assert cloud.pull("gs-example") == cloud.REPO / "data/fifth/splat/runs/gs-example"
    assert seen[-1] == cloud.REPO / "data/fifth/splat/datasets/default"
    explicit_out = cloud.REPO / "explicit-output"
    assert cloud.pull("gs-example", out=explicit_out, data=legacy_data) == explicit_out
    assert seen[-1] == legacy_data


def test_conftest_isolates_the_repo() -> None:
    assert cloud.TRAINER_DIR.parent != cloud.REPO and not any(cloud.REPO.iterdir())
    assert cloud.selected_project() is None


def test_project_precedence_cli_then_settings() -> None:
    (cloud.REPO / "settings.toml").write_text('project = "third"\n\n[poses]\nsource = "arkit"\n')
    assert cloud.selected_project() == "third"
    assert cloud.selected_project("sixth") == "sixth"


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_project_values_count_as_unset(empty: str) -> None:
    (cloud.REPO / "settings.toml").write_text('project = "third"\n')
    assert cloud.selected_project(empty) == "third"
    (cloud.REPO / "settings.toml").write_text(f'project = "{empty}"\n')
    assert cloud.selected_project() is None
    (cloud.REPO / "settings.toml").write_text("[poses]\n")
    assert cloud.selected_project() is None


def test_no_project_is_an_error_where_one_is_needed(tmp_path: Path) -> None:
    with pytest.raises(cloud.CloudError, match=r"--project NAME or set `project` in settings\.toml"):
        cloud.project_dataset()
    with pytest.raises(cloud.CloudError, match=r"settings\.toml"):
        cloud.default_run_dir("gs-x", data=tmp_path / "somewhere")
    # A dataset inside a project's splat/datasets/ still says where its runs go.
    dataset = cloud.REPO / "data" / "fifth" / "splat" / "datasets" / "refined"
    assert (
        cloud.default_run_dir("gs-x", data=dataset)
        == cloud.REPO / "data" / "fifth" / "splat" / "runs" / "gs-x"
    )
    result = CliRunner().invoke(cli.app, ["cloud", "push"])
    assert result.exit_code != 0


@pytest.mark.parametrize(
    ("where", "text"),
    [
        ("cli", "../up"),
        ("cli", "a/b"),
        ("settings.toml", 'project = ".."\n'),
        ("settings.toml", "project = 5\n"),
        ("settings.toml", "project = [unclosed\n"),
    ],
)
def test_bad_project_values_are_cloud_errors(where: str, text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    project = text if where == "cli" else None
    if where != "cli":
        (cloud.REPO / where).write_text(text)
    with pytest.raises(cloud.CloudError):
        cloud.selected_project(project)
    # The CLI (`app()`, whose wrapper CliRunner bypasses) prints one line instead of a traceback.
    monkeypatch.setattr(cloud, "push", lambda *a, **k: "abc")
    argv = ["gs-trainer", "cloud", "push", *(["--project", text] if project else [])]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as exit_:
        cli.app()
    assert exit_.value.code == 1


def test_store_path_is_repo_relative_inside_the_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    inside = cloud.REPO / "data" / "fifth" / "splat" / "datasets" / "default"
    assert cloud.store_path(inside) == "data/fifth/splat/datasets/default"
    assert cloud.store_path(str(inside)) == "data/fifth/splat/datasets/default"
    outside = tmp_path / "elsewhere"
    assert cloud.store_path(outside) == outside.as_posix()
    assert cloud.store_path(None) is None
    monkeypatch.chdir(cloud.REPO)
    assert cloud.store_path(Path("data/x")) == "data/x"  # a relative CLI path: against the cwd
    # Inside a Modal container gs_trainer sits in /root (REPO would be /): paths are left alone.
    monkeypatch.setattr(cloud, "TRAINER_DIR", Path("/root"))
    monkeypatch.setattr(cloud, "REPO", Path("/"))
    assert cloud.local_repo() is None
    assert cloud.store_path(Path("/runs/gs-x/splat.ply")) == "/runs/gs-x/splat.ply"


def test_stored_paths_round_trip_and_old_absolute_ones_still_read(tmp_path: Path) -> None:
    for path in (
        cloud.REPO / "data" / "fifth" / "splat" / "datasets" / "refined",
        cloud.REPO / "data" / "fifth" / "splat" / "runs" / "gs-x",
        tmp_path / "outside" / "run",
    ):
        path.mkdir(parents=True)
        stored = cloud.store_path(path)
        assert stored is not None
        assert cloud.stored_local_path(stored, dataset=False) == path.resolve()
    # A run recorded on another checkout (relative) resolves against this one, existing or not.
    assert cloud.stored_local_path("data/sixth/splat/runs/gs-y", dataset=False) == (
        cloud.REPO / "data" / "sixth" / "splat" / "runs" / "gs-y"
    )
    # Relative pre-migration paths are mapped like absolute ones.
    assert cloud.stored_local_path("data/derived/splat/dataset", dataset=True) == (
        cloud.REPO / "data" / "third" / "splat" / "datasets" / "default"
    )
    # Old records with absolute paths read unchanged.
    old = tmp_path / "old-abs"
    assert cloud.stored_local_path(str(old), dataset=True) == old


def test_pull_reads_relative_run_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = {
        "project": "fifth",
        "data_dir": "data/fifth/splat/datasets/refined",
        "local_out": "data/fifth/splat/runs/mine",
    }
    monkeypatch.setattr(cloud, "_run_json", lambda _: meta)
    monkeypatch.setattr(cloud, "runs_volume", lambda: object())

    def download(volume, run, dest, **kwargs):
        (dest / "splat.ply").touch()
        return []

    seen: list[Path | None] = []
    monkeypatch.setattr(cloud, "_download_dir", download)
    monkeypatch.setattr(
        cloud, "run_manifest", lambda run, metrics, dest, data: seen.append(data) or {"inputs": None}
    )
    dest = cloud.pull("gs-run")
    assert dest == cloud.REPO / "data" / "fifth" / "splat" / "runs" / "mine"
    assert seen == [(cloud.REPO / "data" / "fifth" / "splat" / "datasets" / "refined").resolve()]
    assert str(cloud.REPO) not in (dest / "manifest.json").read_text()


def test_marker_records_repo_relative_source(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from gs_trainer.datapack import TarStats

    data = cloud.REPO / "data" / "fifth" / "splat" / "datasets" / "default"
    marker = cloud.complete_marker(
        None, "h", [], data, TarStats(files=0, bytes=0), datetime(2026, 10, 11, tzinfo=UTC)
    )
    assert marker["from"] == "data/fifth/splat/datasets/default"
