import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def trial_tool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("run_tapo_spoken_trial")


@pytest.mark.parametrize("mode", [
    "seated-hold-60", "crouch-near-seat-60", "kneel-near-seat-60", "bend-near-seat-60",
    "sit-return-60",
])
def test_plan_only_never_starts_recording_or_cloud(
    trial_tool, mode, tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("plan-only must not start subprocesses")

    monkeypatch.setattr(trial_tool, "subprocess", SimpleNamespace(run=forbidden, Popen=forbidden))
    output = tmp_path / "unused"
    monkeypatch.setattr(sys, "argv", [
        "trial", "--host", "unused", "--identity", "missing-key", "--rtsp-ip", "192.0.2.1",
        "--output", str(output), "--mode", mode, "--plan-only",
        "--dataset-split", "evaluation_reserved",
    ])
    trial_tool.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["dataset_split"] == "evaluation_reserved"
    assert not plan["seated_shadow_enabled_live"]
    assert plan["cues_are_instructions_not_ground_truth"]
    assert max(t for t, _ in plan["planned_cues"]) < plan["duration_seconds"]
    assert not output.exists()


def test_benign_capture_reuses_s_only_profile_and_keeps_video_for_review(
    trial_tool, tmp_path, monkeypatch
):
    output = tmp_path / "trial"
    key = tmp_path / "key"
    key.write_text("test-placeholder")
    launched = []

    def process(command, **kwargs):
        launched.append(command)
        output.mkdir()
        return SimpleNamespace(stdout=(), poll=lambda: 0, returncode=0)

    monkeypatch.setattr(trial_tool, "subprocess", SimpleNamespace(
        run=lambda *a, **kw: None, Popen=process, PIPE=-1, STDOUT=-2
    ))
    monkeypatch.setattr(trial_tool.sys, "platform", "darwin")
    monkeypatch.setenv("CARE_VISION_RTSP_USERNAME", "test")
    monkeypatch.setenv("CARE_VISION_RTSP_PASSWORD", "test")
    monkeypatch.setattr(sys, "argv", [
        "trial", "--host", "unused", "--identity", str(key), "--rtsp-ip", "192.0.2.1",
        "--output", str(output), "--mode", "seated-hold-60", "--record-rtsp",
        "--experimental-weak-association",
    ])
    # Fixture must not depend on model artifacts existing on a test checkout.
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    trial_tool.main()
    command = launched[0]
    assert "--experimental-s-only-5fps" in command
    assert "--experimental-weak-association" in command
    assert "--record-rtsp" in command
    assert command[command.index("--baseline-resolution") + 1] == "1280x720"
    assert command[command.index("--person-input-size") + 1] == "640"
    assert command[command.index("--person-threshold") + 1] == "0.35"
    manifest = json.loads((output / "spoken-cues.json").read_text())
    assert manifest["scenario"] == trial_tool.BENIGN_TRIALS["seated-hold-60"][0]
    assert manifest["dataset_split"] == "development"
    assert manifest["cues_are_instructions_not_ground_truth"]
