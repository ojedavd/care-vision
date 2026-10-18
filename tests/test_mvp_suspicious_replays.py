"""Ensure the bounded experiment cannot silently enable alerts or accept unrelated evidence."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mode", ["baseline", "test20"])
def test_launcher_isolates_state_and_alert_opt_in(tmp_path, monkeypatch, mode):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from run_mvp_suspicious_replays import START

    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr("subprocess.Popen", launch)
    monkeypatch.setattr(sys, "argv", ["launch", str(tmp_path), mode, "config.json", mode])
    exec(compile(START, "suspicious-launch", "exec"), {})
    command = calls[0]
    assert "--experimental-pose-observation" in command
    assert "--experimental-episode-continuity" in command
    assert command[command.index("--episode-state-file") + 1] == str(
        tmp_path / mode / "episode.json"
    )
    alert = "--experimental-alert-after-seconds"
    assert (alert in command) == (mode == "test20")
    if mode == "test20":
        assert command[command.index(alert) + 1] == "20"


@pytest.mark.parametrize("fault", [None, "historical", "track", "frame", "invalid", "roi"])
def test_audit_rejects_uncausal_evidence(monkeypatch, fault):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from audit_mvp_suspicious_replays import causal

    obs = {"valid": True, "evidence_kind": "live", "frame_id": "f",
           "view_kind": "full_frame", "persons": [{"track_state": "confirmed", "track_id": 1}]}
    event = {"episode_id": "e", "observation_id": "o", "frame_id": "f", "track_id": 1}
    if fault == "historical":
        obs["evidence_kind"] = "historical"
    elif fault == "track":
        obs["persons"][0]["track_id"] = 2
    elif fault == "frame":
        obs["frame_id"] = "other"
    elif fault == "invalid":
        obs["valid"] = False
    elif fault == "roi":
        obs["view_kind"] = "roi"
    assert causal(event, {"o": obs}, {"e": {"track_id": 1}}) == (fault is None)
