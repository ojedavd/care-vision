import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from care_vision.aws_demo import START, run_aws_demo
from care_vision.cloud.controller import Controller, Policy
from care_vision.cloud.server import CloudServer
from care_vision.common.metrics import EventLog


def test_remote_start_propagates_private_state_and_bounded_disconnect(tmp_path, monkeypatch):
    release = tmp_path / "release"
    release.mkdir()
    state = tmp_path / "stable-state" / "episode.json"
    calls = []

    def start(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=123)

    monkeypatch.setattr("subprocess.Popen", start)
    monkeypatch.setattr(sys, "argv", [
        "driver", str(release), "3", "models/person.onnx", "yolox", "0.35", "0.3", "640",
        "", "0", "1", "20", "1", "1", str(state), "3",
    ])
    exec(compile(START, "remote-start", "exec"), {})
    command, kwargs = calls[0]
    assert "--experimental-episode-continuity" in command
    assert command[command.index("--episode-state-file") + 1] == str(state)
    assert command[command.index("--diagnostic-disconnect-after-episode-seconds") + 1] == "3"
    assert kwargs["env"]["PYTHONPATH"] == "/opt/cool/python_3.11/site-packages"
    assert state.parent != release / "run"


@pytest.mark.parametrize("key", ["../existing", "UPPER", "one/two"])
def test_state_namespace_validation_precedes_deployment(tmp_path, key):
    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="state key"):
        run_aws_demo("unused", Path("unused"), output, tmp_path,
                     experimental_s_only_5fps=True, experimental_episode_continuity=True,
                     episode_state_key=key)
    assert not output.exists()


def test_diagnostic_disconnect_is_one_shot_and_never_targets_restored_identity(tmp_path, identity):
    log = EventLog("cloud", tmp_path / "diagnostic.jsonl")
    try:
        cloud = CloudServer(log, experimental_s_only_5fps=True,
                            experimental_episode_continuity=True,
                            diagnostic_disconnect_after_episode_seconds=3)
        c = Controller(identity, Policy(experimental_s_only_5fps=True,
                                       experimental_episode_continuity=True))
        c._open_episode({"observation_id": "episode", "frame_id": "first",
                         "captured_mono_ns": 1_000_000_000, "width": 1280, "height": 720},
                        "track", [600, 40, 150, 500])
        assert not cloud.diagnostic_disconnect_due(c, {"valid": True,
                                                       "captured_mono_ns": 3_000_000_000})
        c.episode["identity_link_status"] = "unresolved_after_reconnect"
        assert not cloud.diagnostic_disconnect_due(c, {"valid": True,
                                                       "captured_mono_ns": 5_000_000_000})
        c.episode["identity_link_status"] = "session_track_only"
        assert cloud.diagnostic_disconnect_due(c, {"valid": True,
                                                  "captured_mono_ns": 5_000_000_000})
        assert not cloud.diagnostic_disconnect_due(c, {"valid": True,
                                                       "captured_mono_ns": 6_000_000_000})
    finally:
        log.close()
