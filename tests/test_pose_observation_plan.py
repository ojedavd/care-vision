from copy import deepcopy
from dataclasses import replace

import pytest
from test_pose_shadow import episode, processor
from test_seated_pose_candidate import observation

from care_vision.cloud.pose_observation_plan import PoseObservationPlanner, PoseObservationPolicy


def inputs():
    obs = observation(3)
    obs["frame_id"] = "frame-3"
    diagnostic = {
        **{k: obs[k] for k in ("observation_id", "frame_id", "capture_seq", "captured_mono_ns")},
        "episode_id": "e", "context_status": "current", "context_provenance": "synthetic",
        "pose_status": "ok", "seated_pose_candidate": True,
        "reason": "seat_support_unverified", "evidence_status": "SEATED_PATTERN_REQUIRES_SUPPORT",
    }
    p = processor()
    return PoseObservationPlanner(p.geometry, p.hips), obs, diagnostic, episode()


def test_plan_keeps_body_and_context_and_requires_fresh_scene_after_bounded_window():
    planner, obs, diagnostic, ep = inputs()
    original = deepcopy((obs, diagnostic, ep))
    result = planner.consider(obs, diagnostic, ep, result_age_seconds=.02)
    assert not result["dispatched"] and not result["would_close_episode"]
    plan = result["plan"]
    x, y, w, h = plan["request"]["roi"]
    for box in (obs["persons"][0]["tracked_bbox"], [384, 288, 192, 108]):
        assert x <= box[0] and y <= box[1]
        assert x + w >= box[0] + box[2] and y + h >= box[1] + box[3]
    assert plan["return"]["after_applied_ack_seconds"] == 2
    assert plan["return"]["or_after_roi_frames"] == 4
    assert plan["return"]["requires_applied_ack_and_new_full_frame"]
    assert (obs, diagnostic, ep) == original
    assert not hasattr(planner, "controller") and not hasattr(planner, "socket")


@pytest.mark.parametrize("fault", [
    "stale", "negative_age", "nan_age", "wrong_frame", "wrong_episode", "unknown_context",
    "identity", "horizontal", "alert", "historical", "roi", "weak", "multiple", "boundary",
    "pending", "inference_error", "unstable",
])
def test_uncertain_or_unmatched_inputs_never_consume_budget(fault):
    planner, obs, diagnostic, ep = inputs()
    age, pending = .02, False
    if fault == "stale":
        age = 1.01
    elif fault == "negative_age":
        age = -.01
    elif fault == "nan_age":
        age = float("nan")
    elif fault == "wrong_frame":
        diagnostic["frame_id"] = "different"
    elif fault == "wrong_episode":
        diagnostic["episode_id"] = "different"
    elif fault == "unknown_context":
        diagnostic["context_status"] = "unknown"
    elif fault == "identity":
        ep["identity_link_status"] = "unresolved_after_reconnect"
    elif fault == "horizontal":
        ep["horizontal_observed"] = True
    elif fault == "alert":
        ep["test_alert_emitted"] = True
    elif fault == "historical":
        obs["evidence_kind"] = "historical"
    elif fault == "roi":
        obs["view_kind"] = "roi"
    elif fault == "weak":
        obs["persons"][0]["association_only"] = True
    elif fault == "multiple":
        obs["persons"].append(deepcopy(obs["persons"][0]))
    elif fault == "boundary":
        obs["persons"][0]["bbox"] = [0, 200, 160, 300]
    elif fault == "pending":
        pending = True
    elif fault == "inference_error":
        diagnostic["pose_status"] = "inference_error"
    elif fault == "unstable":
        diagnostic["seated_pose_candidate"] = False
    assert planner.consider(obs, diagnostic, ep, result_age_seconds=age,
                            commands_pending=pending)["plan"] is None
    assert not planner.spent


def test_reconnect_does_not_reset_logical_episode_budget_and_capacity_fails_closed():
    planner, obs, diagnostic, ep = inputs()
    planner.policy = replace(PoseObservationPolicy(), maximum_episode_budgets=1)
    assert planner.consider(obs, diagnostic, ep, result_age_seconds=0)["plan"]
    obs["session_id"] = "reconnected"
    assert planner.consider(obs, diagnostic, ep, result_age_seconds=0)["reason"] == (
        "episode_observation_budget_spent"
    )
    ep["episode_id"] = diagnostic["episode_id"] = "second"
    assert planner.consider(obs, diagnostic, ep, result_age_seconds=0)["reason"] == (
        "planner_capacity_exhausted"
    )


def test_low_joint_response_requests_detail_but_model_failure_does_not_loop():
    planner, obs, diagnostic, ep = inputs()
    diagnostic.update(seated_pose_candidate=False, reason="pose_support_insufficient")
    assert planner.consider(obs, diagnostic, ep, result_age_seconds=0)["plan"]
    assert planner.consider(obs, diagnostic, ep, result_age_seconds=0)["plan"] is None
