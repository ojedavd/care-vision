"""Restoration needs operator review, never inferred identity or recovery."""
import json

import pytest

from care_vision.judge.state import Projection


@pytest.mark.parametrize(
    "event", ["episode_restored", "episode_review_required", "incident_episode"]
)
def test_review_latches_independently_of_freshness_or_sensing_state(event):
    p = Projection()
    review = {
        "event": event, "review_required": True,
        "identity_link_status": "unresolved_after_reconnect",
        "review_reason": "identity_unresolved_after_reconnect",
        "episode_id": "private-id", "reference_bbox": [1, 2, 3, 4],
    }
    p.accept(review)
    assert p.snapshot()["review"]["required"]
    assert not p.snapshot()["vision_live"]
    p.accept({"event": "session_started"})
    p.accept({"event": "observation", "valid": True})
    p.accept({"event": "agent_state_transition", "to_state": "VERIFYING"})
    p.accept(review)
    p.accept({"event": "incident_episode", "review_required": False})
    state = p.snapshot()
    assert state["vision_live"] and state["review"]["required"]
    assert state["agent_state"] == "VERIFYING"
    assert sum(t["stage"] == "human_review_required" for t in state["timeline"]) == 1
    assert "private-id" not in json.dumps(state) and "reference_bbox" not in json.dumps(state)


def test_review_projection_rejects_free_form_reason():
    p = Projection()
    p.accept({
        "event": "episode_review_required", "review_required": True,
        "identity_link_status": "unresolved_after_reconnect", "review_reason": "private text",
    })
    assert p.snapshot()["review"] == {"required": False, "reason": None}
