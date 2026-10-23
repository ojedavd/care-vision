import json

import pytest

from care_vision.judge.__main__ import cloud_arguments
from care_vision.judge.state import Projection


def options():
    return {"experimental-s-only-5fps": True, "pose-shadow-config": "/private/scene.json",
            "experimental-seated-closure-seconds": 5}


def test_seated_profile_arguments_are_explicit_and_do_not_enable_roi_or_alerts():
    argv = cloud_arguments(options())
    assert argv == ["--experimental-s-only-5fps", "--pose-shadow-config", "/private/scene.json",
                    "--experimental-seated-closure-seconds", "5"]
    assert "--experimental-pose-observation" not in argv
    assert "--experimental-alert-after-seconds" not in argv
    assert cloud_arguments({}) == []


@pytest.mark.parametrize("bad", [True, False, 0, -1, "5", None, float("inf"), float("nan")])
def test_closure_config_rejects_unusable_durations(bad):
    with pytest.raises(ValueError):
        cloud_arguments(options() | {"experimental-seated-closure-seconds": bad})


@pytest.mark.parametrize("change", [
    {"experimental-s-only-5fps": False}, {"pose-shadow-config": ""},
    {"pose-shadow-config": None}, {"pose-shadow-config": 3},
])
def test_closure_cannot_be_enabled_without_required_profile(change):
    with pytest.raises(ValueError):
        cloud_arguments(options() | change)


def test_profile_is_confirmed_by_worker_and_exposes_no_private_configuration():
    p = Projection()
    assert p.snapshot()["profile"]["seated_closure_seconds"] is None
    p.accept({"event": "server_ready", "phase": 3, "pose_shadow_enabled": True,
              "seated_closure_seconds": 5, "pose_config": "/private/scene.json"})
    assert p.snapshot()["profile"] == {"pose_enabled": True, "seated_closure_seconds": 5}
    assert "/private" not in json.dumps(p.snapshot())
    p.accept({"event": "agent_state_transition", "to_state": "NORMAL",
              "reason": "experimental_seated_closure", "track_id": "private-track"})
    assert p.snapshot()["timeline"][-1] == {
        "stage": "episode_resolved", "action": "EXPERIMENTAL_SEATED_PATTERN"}
    assert "private-track" not in json.dumps(p.snapshot())


@pytest.mark.parametrize("fields", [
    {"pose_shadow_enabled": False}, {"phase": 1}, {"seated_closure_seconds": float("nan")},
    {"seated_closure_seconds": True}, {"seated_closure_seconds": 0},
])
def test_projection_does_not_advertise_unconfirmed_profile(fields):
    p = Projection()
    p.accept({"event": "server_ready", "phase": 3, "pose_shadow_enabled": True,
              "seated_closure_seconds": 5} | fields)
    assert p.snapshot()["profile"]["seated_closure_seconds"] is None
