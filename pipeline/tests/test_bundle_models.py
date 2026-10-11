"""Bundle v1 models against what FlatBlender writes: app 0.4 fields, and older files that must still validate."""

import json
from datetime import UTC, datetime

import pytest

from scanmodel.models.bundle import (
    DepthFormat,
    Device,
    LockedCamera,
    Session,
    SessionType,
    Still,
    VideoFormat,
)

# session.json `camera` of scan 4 (app 0.3.0 (6)).
SCAN4_CAMERA = {
    "exposure_duration_s": 0.004166,
    "iso": 619.8161,
    "lens_position": 0.7058823,
    "locked": True,
    "metered": {"duration_s": 0.010002, "iso": 258.20508},
    "shutter_mode": "sharp",
    "underexposure_ev": 0,
    "white_balance": {"blue": 2.3164062, "green": 1, "red": 1.6132812},
}


def test_camera_flicker_fields_app_04() -> None:
    cam = LockedCamera.model_validate(
        {**SCAN4_CAMERA, "flicker_suspected": "50hz", "flicker_prompt": "accepted"}
    )
    assert cam.flicker_suspected == "50hz" and cam.flicker_prompt == "accepted"
    old = LockedCamera.model_validate(SCAN4_CAMERA)
    assert old.flicker_suspected is None and old.flicker_prompt is None
    assert old.metered is not None and old.metered.duration_s == pytest.approx(1 / 100, rel=0.01)
    with pytest.raises(ValueError):
        LockedCamera.model_validate({**SCAN4_CAMERA, "flicker_suspected": "100hz"})


def test_scan4_style_session_json_parses() -> None:
    """A session.json as app 0.3 wrote scan 4's (its camera block above), then as app 0.4 would."""
    raw = json.loads(
        Session(
            project_id="fourth",
            session_id="20261008-120000-kitchen",
            type=SessionType.SCAN,
            device=Device(model="iPad16,6", os_version="26.0", app_version="0.3.0 (6)"),
            video=VideoFormat(width=1920, height=1440, fps=60.0),
            depth=DepthFormat(),
            started_at=datetime(2026, 10, 8, 12, 0, tzinfo=UTC),
        ).model_dump_json(exclude_none=True)
    )
    raw["camera"] = SCAN4_CAMERA
    session = Session.model_validate(raw)
    assert session.camera is not None and session.camera.metered is not None
    assert session.camera.flicker_suspected is None  # written before app 0.4
    # The same file as app 0.4 would write it after the flicker prompt was shown and accepted.
    raw["camera"] |= {
        "flicker_suspected": "50hz",
        "flicker_prompt": "accepted",
        "shutter_mode": "flicker_50hz",
    }
    assert Session.model_validate(raw).camera.flicker_prompt == "accepted"  # type: ignore[union-attr]


def test_still_lines_with_and_without_sharpness() -> None:
    # App 0.4+: motion only.
    new = Still.model_validate_json(
        '{"id":1,"file":"stills/000001.jpg","t":1.5,"source":"auto",'
        '"motion":{"angular_speed_deg_s":2.5,"speed_m_s":0.02,"blur_px":0.4}}'
    )
    assert new.sharpness is None and new.flagged_blurry is None
    assert new.motion is not None and new.motion.frame_sharpness is None
    # App < 0.4 (scan 4): sharpness, stream sharpness and the blur flag still validate (Strict forbids extras).
    old = Still.model_validate_json(
        '{"id":7,"file":"stills/000007.jpg","t":1234.5,"source":"tap","sharpness":61.2,'
        '"motion":{"angular_speed_deg_s":21.3,"speed_m_s":0.12,"blur_px":5.1,"frame_sharpness":40.5,'
        '"median_sharpness":70.1},"flagged_blurry":true}'
    )
    assert old.flagged_blurry is True and old.motion is not None and old.motion.median_sharpness == 70.1
