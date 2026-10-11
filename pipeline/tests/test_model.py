"""`scanmodel model`: a project's scan -> its Blender model (the model stages only), and the stage runner."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from scanmodel import cli
from scanmodel.cache import Stage
from scanmodel.config import Config, load_config


def _scans(repo: Path) -> None:
    """Projects `third` and `fifth` with two sessions each, `empty` with none; settings.toml selects third."""
    (repo / "settings.toml").write_text('project = "third"\n')
    for scan in ("third", "fifth"):
        for session in ("20261009-124955-conservatory", "20261009-130514-lap"):
            d = repo / "data" / scan / "raw" / session
            d.mkdir(parents=True)
            (d / "session.json").write_text("{}")
    (repo / "data" / "empty" / "raw").mkdir(parents=True)


def test_model_stages_leave_splats_out(repo: Path) -> None:
    _scans(repo)
    cfg = load_config()
    names = [s.name for s in cli.model_stages(cfg, "refine")]
    assert names == ["ingest", "poses", "refine", "fuse", "structure", "eval", "build"]
    assert "refine" not in [s.name for s in cli.model_stages(cfg, "posegraph")]
    # The pose graph's leave-one-pair-out check is a tuning diagnostic: not in the default key or run.
    assert cli._poses_stage(cfg).params == {"pose_graph": True}  # pyright: ignore[reportPrivateUsage]
    assert cli._poses_stage(cfg, stability=True).params == {"pose_graph": True, "stability": True}  # pyright: ignore[reportPrivateUsage]


def test_stage_table() -> None:
    t = cli.stage_table([("ingest", True, 41.2), ("poses", False, 0.1), ("refine", True, 590.0)])
    assert "ingest" in t and "41.2 s" in t and "cached" in t and "9 min 50 s" in t
    assert t.splitlines()[-1].split() == ["total", "10", "min", "31", "s"]


def test_model_command_runs_the_model_stages_on_the_project(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scans(repo)
    ran: list[tuple[list[str], Config]] = []

    def fake_run(stages: list[Stage], cfg: Config, force: bool = False) -> list[tuple[str, bool, float]]:
        ran.append(([s.name for s in stages], cfg))
        return []

    monkeypatch.setattr(cli, "run_stages", fake_run)
    res = CliRunner().invoke(cli.app, ["model", "fifth", "--glb"])
    assert res.exit_code == 0, res.output
    names, cfg = ran[0]
    assert names == ["ingest", "poses", "refine", "fuse", "structure", "eval", "build", "export"]
    assert cfg.project == "fifth" and cfg.data == repo / "data" / "fifth"
    assert "model.blend" in res.output
    res = CliRunner().invoke(cli.app, ["model"])  # settings.toml's project
    assert res.exit_code == 0, res.output
    assert ran[-1][1].project == "third"
    res = CliRunner().invoke(cli.app, ["model", "empty"])
    said = " ".join(res.output.replace("│", " ").split())  # the error box wraps lines
    assert res.exit_code != 0 and "no session bundles" in said and "fifth, third" in said
