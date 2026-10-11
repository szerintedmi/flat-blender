"""Registration across independently captured sessions uses their shared coordinate frame."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from scanmodel.geom import translation
from scanmodel.models import bundle as b
from scanmodel.stages import poses


def test_registration_targets_include_previous_session_transforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = [
        poses.SessionData(
            id=f"session-{i}",
            raw=tmp_path / str(i),
            session=b.Session(
                project_id="other-scene",
                session_id=f"session-{i}",
                type=b.SessionType.SCAN,
                device=b.Device(model="test", os_version="test", app_version="test"),
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            frames=[],
            stills=[],
            room=None,
            to_zup=np.eye(4),
            points=np.array([[10.0 * i, 0.0, 0.0]]),
            dropped=[],
        )
        for i in range(4)
    ]
    # The final capture resumes session 1's map, whose coordinates were registered above.
    sessions[3].session.world_map_in = "session-1"
    sessions[3].points = sessions[1].points.copy()
    targets: list[np.ndarray] = []

    def register(points: np.ndarray, target: np.ndarray) -> np.ndarray:
        targets.append(target.copy())
        return translation((-float(points[0, 0]), 0.0, 0.0))

    monkeypatch.setattr(poses, "_load", lambda _: sessions)
    monkeypatch.setattr(poses, "global_register", register)
    monkeypatch.setattr(poses, "icp_check", lambda *args: (0.0, np.eye(4), 1.0))
    monkeypatch.setattr(poses, "with_normals", lambda p: (p, np.tile([0.0, 0.0, 1.0], (len(p), 1))))
    monkeypatch.setattr(poses, "floor_height", lambda *args: 0.0)
    poses.run(tmp_path / "ingest", tmp_path / "overrides.json", tmp_path / "poses")
    assert len(targets) == 2
    np.testing.assert_allclose(targets[1], np.zeros((2, 3)))
    np.testing.assert_allclose(sessions[3].to_zup, sessions[1].to_zup)
