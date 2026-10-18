import copy

import pytest

from care_vision.check_loop import CheckFailed
from care_vision.check_states import EXPECTED_TRANSITIONS, check_state_records


def state_records():
    records = []
    for index, (before, after) in enumerate(EXPECTED_TRANSITIONS, 1):
        observation = {
            "event": "observation",
            "observation_id": f"O{index}",
            "frame_id": f"F{index}",
            "captured_mono_ns": index * 1_000,
            "local_mono_ns": index * 10_000,
            "evidence_kind": "live",
        }
        records.extend(
            [
                observation,
                {
                    "event": "agent_state_transition",
                    "from_state": before,
                    "to_state": after,
                    "reason": "test",
                    "observation_id": observation["observation_id"],
                    "frame_id": observation["frame_id"],
                    "captured_mono_ns": observation["captured_mono_ns"],
                    "local_mono_ns": observation["local_mono_ns"] + 1,
                },
            ]
        )
    records.extend(
        [
            {
                "event": "agent_action",
                "type": "SET_TRANSMIT_FPS",
                "transmit_fps": 20,
                "agent_state": "VERIFYING",
            },
            {
                "event": "agent_action",
                "type": "SET_TRANSMIT_FPS",
                "transmit_fps": 2,
                "agent_state": "RECOVERY",
            },
        ]
    )
    return records


def test_state_checker_verifies_causal_sequence():
    report = check_state_records(state_records())
    assert report["sequence"] == ["NORMAL", "SUSPICIOUS", "VERIFYING", "RECOVERY", "NORMAL"]
    assert not report["possible_fall_activated"]


def test_state_checker_rejects_skipped_state_and_wrong_observation():
    skipped = [
        record
        for record in state_records()
        if not (
            record.get("event") == "agent_state_transition"
            and record.get("to_state") == "SUSPICIOUS"
        )
    ]
    with pytest.raises(CheckFailed, match="state sequence"):
        check_state_records(skipped)
    mismatched = copy.deepcopy(state_records())
    next(record for record in mismatched if record.get("event") == "agent_state_transition")[
        "frame_id"
    ] = "wrong"
    with pytest.raises(CheckFailed, match="causal observation"):
        check_state_records(mismatched)
