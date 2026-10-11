"""Scene-specific state can be isolated without copying the pipeline repository."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scanmodel import cli
from scanmodel.cache import fingerprint
from scanmodel.cli import app
from scanmodel.config import ConfigError, load_config


def test_project_paths(tmp_path: Path) -> None:
    (tmp_path / "settings.toml").write_text('project = "fifth"\n')
    cfg = load_config(tmp_path)
    project = tmp_path / "data/fifth"
    assert cfg.data == project and cfg.raw == project / "raw"
    assert cfg.scene_dir == project / "scene" and cfg.views_file == project / "scene/views.json"
    assert cfg.stage_dir("fuse") == project / "derived/fuse"
    assert cfg.stage_dir("build") == project / "build"
    assert cfg.stage_dir("export") == project / "export"
    assert cfg.dataset_dir() == project / "splat/datasets/default"
    assert cfg.dataset_dir("refined") == project / "splat/datasets/refined"
    assert cfg.runs_dir == project / "splat/runs"
    assert load_config(tmp_path, project="office").project == "office"


@pytest.mark.parametrize("args", [["status", "--project", "office"], ["--project", "office", "status"]])
def test_project_option_before_or_after_command(repo: Path, args: list[str]) -> None:
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "data/office/scene" in result.output.replace("\n", "")


@pytest.mark.parametrize("name", ["../third", "/tmp/third", ".", "a/b", "a\\b"])
def test_project_name_cannot_escape_data(tmp_path: Path, name: str) -> None:
    with pytest.raises(ConfigError, match="plain folder"):
        load_config(tmp_path, project=name)


def test_project_test_poses_and_open_commands(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = repo
    dataset = tmp_path / "data/office/splat/datasets/default"
    dataset.mkdir(parents=True)
    (dataset / "transforms.json").write_text("{}")
    solved: list[Path] = []
    opened: list[str] = []
    monkeypatch.setattr(cli.testposes, "write", lambda data, out: solved.append(data))
    monkeypatch.setattr(cli.typer, "launch", lambda path: opened.append(path))
    runner = CliRunner()
    for command in ("test-poses", "report", "blend"):
        result = runner.invoke(app, [command, "--project", "office"])
        assert result.exit_code == 0, result.output
    assert solved == [dataset]
    assert opened == [
        str(tmp_path / "data/office/derived/eval/report.html"),
        str(tmp_path / "data/office/build/model.blend"),
    ]


def test_project_view_uses_its_scene_file(repo: Path) -> None:
    tmp_path = repo
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["view", "add", "entry", "https://example.com/?cam=0,0,1&look=0,1,1&fov=60", "--project", "office"],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "data/office/scene/views.json").is_file()


def test_publish_project_runs_checks_project_dataset(tmp_path: Path) -> None:
    cfg = load_config(tmp_path, project="office")
    dataset = cfg.dataset_dir()
    dataset.mkdir(parents=True)
    (dataset / "transforms.json").write_text("{}")
    (dataset / "points.ply").write_text("points")
    run = cfg.runs_dir / "msplat"
    run.mkdir(parents=True)
    (run / "splat.ply").write_text("splat")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "params": {"trainer": "msplat"},
                "summary": {},
                "inputs": fingerprint([dataset / "transforms.json", dataset / "points.ply"]),
            }
        )
    )
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    entries = cli.publish_splats(cfg.runs_dir, viewer, "msplat", dataset_dir=dataset)
    assert len(entries) == 1 and entries[0]["current"] is True
    assert (viewer / "splats/msplat.ply").resolve() == run / "splat.ply"


def test_settings_precedence_and_empty_values(tmp_path: Path) -> None:
    """--project > settings.toml > defaults; an empty --project falls through; no project is an error."""
    with pytest.raises(ConfigError, match="no project"):
        load_config(tmp_path)
    (tmp_path / "settings.toml").write_text('project = "third"\npose_graph = false\nposes = "posegraph"\n')
    cfg = load_config(tmp_path)
    assert (cfg.project, cfg.project_source, cfg.pose_graph, cfg.poses) == (
        "third",
        "settings.toml",
        False,
        "posegraph",
    )
    cfg = load_config(tmp_path, project="office")
    assert (cfg.project, cfg.project_source) == ("office", "--project")
    assert load_config(tmp_path, project=" ").project == "third"
    assert (
        cfg.blender == Path("/Applications/Blender.app/Contents/MacOS/Blender")
        and cfg.msplat == "msplat-train"
    )
    (tmp_path / "settings.toml").write_text(
        'project = "third"\n[tools]\nblender = "/opt/blender"\nbrush = "brush"\n'
    )
    cfg = load_config(tmp_path)
    assert (cfg.blender, cfg.brush, cfg.msplat, cfg.pose_graph) == (
        Path("/opt/blender"),
        "brush",
        "msplat-train",
        True,
    )
    for bad, match in (
        ('projekt = "x"\n', "unknown setting"),
        ("pose_graph = 1\n", "must be a bool"),
        ('[tools]\nblendr = "x"\n', r"\[tools\] takes"),
    ):
        (tmp_path / "settings.toml").write_text(bad)
        with pytest.raises(ConfigError, match=match):
            load_config(tmp_path)


def test_missing_project_is_a_clear_error(repo: Path) -> None:
    r = CliRunner().invoke(app, ["status"])
    assert r.exit_code != 0 and "no project" in r.output


def test_repo_settings_toml_is_valid() -> None:
    """The tracked settings.toml: known keys, a project name, a pose source."""
    from scanmodel.config import CODE_ROOT, POSE_SOURCES, read_settings

    s = read_settings(CODE_ROOT / "settings.toml")
    assert s["project"] and s["poses"] in POSE_SOURCES


def test_rel_and_from_repo(tmp_path: Path) -> None:
    from scanmodel.config import from_repo, rel

    assert rel(tmp_path / "data/fifth/raw", tmp_path) == "data/fifth/raw"
    assert rel(tmp_path, tmp_path) == "."
    assert rel("/elsewhere/x.ply", tmp_path) == "/elsewhere/x.ply"
    assert rel("data/x", tmp_path) == "data/x"
    assert from_repo("data/fifth/raw", tmp_path) == tmp_path / "data/fifth/raw"
    assert from_repo("/elsewhere/x.ply", tmp_path) == Path("/elsewhere/x.ply")


def test_fingerprints_survive_moving_the_repo(tmp_path: Path) -> None:
    for name in ("a", "b"):
        f = tmp_path / name / "data/p/derived/x.json"
        f.parent.mkdir(parents=True)
        f.write_text("{}")
    a, b = tmp_path / "a", tmp_path / "b"
    assert fingerprint([a / "data/p/derived"], a) == fingerprint([b / "data/p/derived"], b)
    assert fingerprint([a / "data/p/derived"], a) != fingerprint([a / "data/p/derived"], b)


def test_status_names_the_project_and_its_source(repo: Path) -> None:
    (repo / "settings.toml").write_text('project = "third"\n')
    runner = CliRunner()
    assert "project   third (settings.toml)" in runner.invoke(app, ["status"]).output
    assert "project   office (--project)" in runner.invoke(app, ["status", "--project", "office"]).output


def test_publish_marks_runs_against_their_own_dataset(tmp_path: Path) -> None:
    """A run trained on a named dataset is current when that dataset is unchanged, whatever the default."""
    cfg = load_config(tmp_path, project="office")
    for name in ("default", "refined"):
        d = cfg.dataset_dir(name)
        d.mkdir(parents=True)
        (d / "transforms.json").write_text(name)
        (d / "points.ply").write_text("points")
    refined = cfg.dataset_dir("refined")
    run = cfg.runs_dir / "msplat-refined"
    run.mkdir(parents=True)
    (run / "splat.ply").write_text("splat")
    inputs = fingerprint([refined / "transforms.json", refined / "points.ply"], tmp_path)
    manifest = {"params": {"trainer": "msplat"}, "summary": {"dataset": cfg.rel(refined)}, "inputs": inputs}
    (run / "manifest.json").write_text(json.dumps(manifest))
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (entry,) = cli.publish_splats(
        cfg.runs_dir, viewer, "msplat", dataset_dir=cfg.dataset_dir(), root=tmp_path
    )
    assert entry["current"] is True
    (refined / "points.ply").write_text("changed")
    (entry,) = cli.publish_splats(
        cfg.runs_dir, viewer, "msplat", dataset_dir=cfg.dataset_dir(), root=tmp_path
    )
    assert entry["current"] is False


def test_rekey_marks_stages_current_without_running(repo: Path) -> None:
    tmp_path = repo
    cfg = load_config(tmp_path, project="office")
    poses = cfg.stage_dir("poses")
    poses.mkdir(parents=True)
    (poses / "manifest.json").write_text(
        json.dumps({"inputs": "old", "params": {"pose_graph": True}, "code": "old"})
    )
    stale = cli._poses_stage(cfg)  # pyright: ignore[reportPrivateUsage]
    runner = CliRunner()
    r = runner.invoke(app, ["rekey", "--project", "office", "--dry-run"])
    assert r.exit_code == 0 and "would rekey" in r.output
    assert json.loads((poses / "manifest.json").read_text())["inputs"] == "old"
    r = runner.invoke(app, ["rekey", "--project", "office"])
    assert r.exit_code == 0, r.output
    m = json.loads((poses / "manifest.json").read_text())
    assert m["params"] == {"pose_graph": True}
    assert {k: m[k] for k in ("inputs", "code")} == {k: stale.key()[k] for k in ("inputs", "code")}
    assert "up to date" in runner.invoke(app, ["rekey", "--project", "office"]).output
