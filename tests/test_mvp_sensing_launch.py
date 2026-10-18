"""Check the comparison arms before starting remote workers."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("launcher,mode", [
    ("run_mvp_sensing_comparison.py", "fixed20"),
    ("run_mvp_sensing_comparison.py", "adaptive"),
    ("run_mvp_fixed5_controls.py", "fixed5"),
])
def test_comparison_launch_keeps_shared_vision_and_isolates_actuation(
    tmp_path, monkeypatch, launcher, mode,
):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "comparison", tools / launcher,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=123)

    monkeypatch.setattr("subprocess.Popen", launch)
    monkeypatch.setattr(sys, "argv", ["launch", str(tmp_path), mode, "config.json", mode])
    exec(compile(module.START, "comparison-launch", "exec"), {})
    command, kwargs = calls[0]
    assert command[command.index("--person-input-size") + 1] == "640"
    assert "--experimental-s-only-5fps" in command
    assert "--experimental-weak-association" in command
    assert "--diagnostic-disconnect-after-episode-seconds" not in command
    assert kwargs["env"]["PYTHONPATH"].endswith(":/opt/cool/python_3.11/site-packages")
    if mode.startswith("fixed"):
        assert "care_vision.cloud" not in command
        assert str(tmp_path / "tools/fixed_sensing_cloud.py") in command
        assert "--pose-shadow-config" not in command
        assert "--experimental-pose-observation" not in command
        assert "--episode-state-file" not in command
    else:
        assert "care_vision.cloud" in command
        assert "--experimental-pose-observation" in command
        assert "--experimental-episode-continuity" in command
        assert command[command.index("--episode-state-file") + 1] == str(
            tmp_path / mode / "episode.json"
        )
    compile(module.STOP_EITHER, "comparison-stop", "exec")


def test_fixed5_launcher_sets_initial_and_reconnect_rate_without_changing_normal_edge(
    tmp_path, monkeypatch,
):
    from care_vision.common.metrics import EventLog
    from care_vision.edge.runtime import Edge

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from fixed5_sensing_edge import Fixed5Edge

    source = SimpleNamespace(width=1280, height=720, fps=25, boot_id="recorded")
    log = EventLog("edge", tmp_path / "edge.jsonl")
    try:
        fixed = Fixed5Edge(source, "ws://127.0.0.1:1", log, initial_transmit_fps=20)
        normal = Edge(source, "ws://127.0.0.1:1", log)
        assert fixed.initial_transmit_fps == fixed.sampler.fps == 5
        assert normal.initial_transmit_fps == normal.sampler.fps == 2
        assert fixed.network.degraded_rtt_ms == normal.network.degraded_rtt_ms == 250
    finally:
        log.close()
