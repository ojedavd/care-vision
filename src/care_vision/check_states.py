"""Verify the deterministic observation-state sequence and its causal evidence."""

from care_vision.check_loop import require

EXPECTED_TRANSITIONS = [
    ("NORMAL", "SUSPICIOUS"),
    ("SUSPICIOUS", "VERIFYING"),
    ("VERIFYING", "RECOVERY"),
    ("RECOVERY", "NORMAL"),
]


def check_state_records(cloud):
    transitions = [record for record in cloud if record.get("event") == "agent_state_transition"]
    require(
        [(record["from_state"], record["to_state"]) for record in transitions]
        == EXPECTED_TRANSITIONS,
        "unexpected deterministic state sequence",
    )
    observations = {
        record["observation_id"]: record
        for record in cloud
        if record.get("event") == "observation" and record.get("evidence_kind", "live") == "live"
    }
    for transition in transitions:
        observation = observations.get(transition["observation_id"])
        require(observation is not None, "state transition lacks live observation evidence")
        require(
            observation["frame_id"] == transition["frame_id"]
            and observation["captured_mono_ns"] == transition["captured_mono_ns"]
            and observation["local_mono_ns"] <= transition["local_mono_ns"],
            "state transition does not match its causal observation",
        )
    require(
        all(
            later["captured_mono_ns"] >= earlier["captured_mono_ns"]
            for earlier, later in zip(transitions, transitions[1:])
        ),
        "state transition time moved backwards",
    )
    fps_actions = [
        record
        for record in cloud
        if record.get("event") == "agent_action" and record.get("type") == "SET_TRANSMIT_FPS"
    ]
    require(
        [(record["transmit_fps"], record.get("agent_state")) for record in fps_actions]
        == [(20, "VERIFYING"), (2, "RECOVERY")],
        "FPS actions do not match VERIFYING and RECOVERY states",
    )
    return {
        "verified": True,
        "sequence": [transitions[0]["from_state"]] + [record["to_state"] for record in transitions],
        "transition_count": len(transitions),
        "possible_fall_activated": any(
            record["to_state"] == "POSSIBLE_FALL" for record in transitions
        ),
        "incident_activated": any(record["to_state"] == "INCIDENT" for record in transitions),
    }
