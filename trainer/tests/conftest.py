"""Shared fixtures: a tiny synthetic dataset v2 (`gs_trainer.synth`), built once per test session, and an
isolated environment for every test.

`synth_dataset` is shared by every test, so treat it as read-only. A test that writes into a dataset
(priors, for instance) builds its own copy with `synth.write_dataset(tmp_path / "data")`.

`isolated_environment` (autouse) makes the tests independent of the checkout they run in: `cloud.REPO`
(where the project selection reads `settings.toml`, and what stored paths are relative to) is an empty
folder, so the repo's own `settings.toml` is never read. A test that needs one writes it into `cloud.REPO`.
"""

from pathlib import Path

import pytest

from gs_trainer import cloud, synth


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo = tmp_path_factory.mktemp("repo")
    monkeypatch.setattr(cloud, "REPO", repo)
    return repo


@pytest.fixture(scope="session")
def synth_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synth.write_dataset(tmp_path_factory.mktemp("synth") / "dataset")
