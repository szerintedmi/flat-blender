"""Tests see tracked files only: no repo settings.toml, no local data/.

The pipeline finds its repo from the working directory, so every test runs inside an empty repo skeleton
(the `repo` fixture gives a test its own); tests that need settings write them there."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


def make_repo(root: Path) -> Path:
    """An empty repo skeleton default_root() recognises (pipeline/pyproject.toml, scene/)."""
    (root / "pipeline").mkdir(parents=True, exist_ok=True)
    (root / "pipeline" / "pyproject.toml").touch()
    (root / "scene").mkdir(exist_ok=True)
    return root


@pytest.fixture(scope="session", autouse=True)
def isolated_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Run the session in an empty skeleton, so a load_config() without a root never reads the checkout."""
    root = make_repo(tmp_path_factory.mktemp("repo"))
    old = Path.cwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(old)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """tmp_path as a repo skeleton and the working directory: the CLI's root."""
    monkeypatch.chdir(make_repo(tmp_path))
    return tmp_path
