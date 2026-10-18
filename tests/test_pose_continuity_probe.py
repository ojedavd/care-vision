"""The experiment cut must follow verified ROI evidence, never a mere request."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pose_continuity_cloud_probe",
    Path(__file__).resolve().parents[1] / "tools/pose_continuity_cloud_probe.py",
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("block", ["no_event", "wrong_event", "invalid", "full_frame",
                                   "historical", "restored", "unspent", "no_episode", None])
def test_phase_cut_requires_matched_evidence_and_is_one_shot(tmp_path, block):
    log = probe.ProbeLog("cloud", tmp_path / "cloud.jsonl")
    try:
        server = probe.ProbeServer(
            log, experimental_s_only_5fps=True, experimental_episode_continuity=True,
            episode_state_file=tmp_path / "episode.json", pose_shadow_factory=lambda _: None,
            experimental_pose_observation=True,
        )
        episode = {"episode_id": "episode", "identity_link_status": "session_track_only",
                   "pose_observation_requested": True}
        observation = {"observation_id": "new_roi", "valid": True,
                       "view_kind": "roi", "evidence_kind": "live"}
        if block != "no_event":
            log.emit("pose_observation_cycle", stage="roi_evidence_processed",
                     observation_id="other" if block == "wrong_event" else "new_roi")
        if block == "invalid":
            observation["valid"] = False
            observation["invalid_reason"] = "processing_stale"
        if block == "full_frame":
            observation["view_kind"] = "full_frame"
        if block == "historical":
            observation["evidence_kind"] = "historical"
        if block == "restored":
            episode["identity_link_status"] = "unresolved_after_reconnect"
        if block == "unspent":
            episode["pose_observation_requested"] = False
        controller = SimpleNamespace(episode=None if block == "no_episode" else episode)
        assert server.diagnostic_disconnect_due(controller, observation) is (block is None)
        assert not server.diagnostic_disconnect_due(controller, observation)
    finally:
        log.close()


def test_probe_rejects_non_opt_in_configuration(tmp_path):
    log = probe.ProbeLog("cloud", tmp_path / "cloud.jsonl")
    try:
        with pytest.raises(ValueError, match="probe requires"):
            probe.ProbeServer(log)
    finally:
        log.close()
