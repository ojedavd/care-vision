import importlib.util
from copy import deepcopy
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "stream_analysis", Path(__file__).parents[1] / "tools/analyze_pose_shadow_stream.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_same_count_different_payload_or_missing_processing_does_not_match():
    original = {
        "sent_fingerprint": [[1, "sha-a"], [9, "sha-b"]],
        "receive_sequences": [1, 9], "processed_sequences": [1, 9],
    }
    assert analysis.matched_inputs([original, deepcopy(original)])
    for key, replacement in [
        ("sent_fingerprint", [[1, "sha-a"], [9, "changed-image"]]),
        ("processed_sequences", [1]),
        ("receive_sequences", [1, 8]),
    ]:
        changed = {**original, key: replacement}
        assert not analysis.matched_inputs([original, changed])


def test_behavior_comparison_ignores_generated_ids_but_detects_changed_trigger_frame():
    def trial(boot, origin):
        return [
            {"event": "observation", "captured_mono_ns": origin},
            {"event": "agent_action", "type": "SET_TRANSMIT_FPS", "transmit_fps": 5,
             "reason": "suspicious", "command_id": boot, "based_on_frame_id": boot + ":23"},
            {"event": "incident_episode", "episode_id": boot, "reason": "episode_opened",
             "incident_status": "unresolved", "captured_mono_ns": origin + 12,
             "horizontal_observed": False, "test_alert_emitted": False},
        ]
    a, b = trial("boot-a", 100), trial("boot-b", 200)
    assert analysis.behavior_signature(a) == analysis.behavior_signature(b)
    b[1]["based_on_frame_id"] = "boot-b:24"
    assert analysis.behavior_signature(a) != analysis.behavior_signature(b)
