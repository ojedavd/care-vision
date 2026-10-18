from copy import deepcopy

import numpy as np
import pytest

from care_vision.cloud.seated_candidate import SeatContext, SeatedCandidate
from care_vision.cloud.seated_pose_candidate import SeatedPoseCandidate, SeatPoseContext


def observation(t, box=(400, 200, 160, 300)):
    return {
        "observation_id": f"o-{t}",
        "captured_mono_ns": round(t * 1e9),
        "capture_seq": round(t * 25),
        "edge_id": "edge",
        "boot_id": "boot",
        "session_id": "session",
        "width": 1280,
        "height": 720,
        "source_width": 1280,
        "source_height": 720,
        "view_kind": "full_frame",
        "valid": True,
        "evidence_kind": "live",
        "person_detection_status": "ok",
        "person_tracking_status": "ok",
        "persons": [{"track_id": "a", "track_state": "confirmed", "tracked_bbox": list(box)}],
    }


def evaluator(**changes):
    episode = {
        "episode_id": "e",
        "track_id": "a",
        "started_ns": 1_000_000_000,
        "reference_bbox": [400 / 1280, 40 / 720, 160 / 1280, 500 / 720],
        "horizontal_observed": False,
        "test_alert_emitted": False,
    }
    episode.update(changes)
    context = SeatContext("edge", 1280, 720, (0.30, 0.4, 0.15, 0.15), "synthetic occupancy")
    geometry = SeatedCandidate(episode, observation(1), context)
    return SeatedPoseCandidate(
        geometry,
        SeatPoseContext((0.34, 0.40, 0.12, 0.12), "synthetic hip proximity, not seat contact"),
        identity_link_status="session_track_only",
    )


def pose():
    points = np.tile([450, 330, 0.8], (17, 1))
    points[[5, 6]] = [[470, 230, 0.8], [490, 230, 0.8]]
    points[[11, 12]] = [[470, 330, 0.8], [490, 330, 0.8]]
    points[[13, 14]] = [[410, 340, 0.8], [430, 340, 0.8]]
    points[[15, 16]] = [[410, 450, 0.8], [430, 450, 0.8]]
    return points


@pytest.mark.parametrize("step", [0.2, 0.5])
def test_stable_pattern_never_verifies_seat_support_or_closes(step):
    e = evaluator()
    before = pose().copy()
    for i in range(round(2 / step)):
        assert not e.update(observation(1 + i * step), before)["seated_pose_candidate"]
    item = observation(3)
    original = deepcopy(item)
    result = e.update(item, before)
    assert result["seated_pose_candidate"]
    assert result["pose_stable_seconds"] == 2
    assert result["evidence_status"] == "SEATED_PATTERN_REQUIRES_SUPPORT"
    assert not result["would_close_episode"] and not result["seat_support_verified"]
    assert result["suggested_observation"]["dispatched"] is False
    assert item == original and np.array_equal(before, pose())


@pytest.mark.parametrize(
    "change", ["missing", "low_score", "gap", "identity", "weak", "historical", "nan", "session"]
)
def test_insufficient_evidence_breaks_stability(change):
    e = evaluator()
    e.update(observation(1), pose())
    e.update(observation(1.5), pose())
    item, points = observation(2), pose()
    if change == "missing":
        points = None
    elif change == "low_score":
        points[15, 2] = 0.1
    elif change == "gap":
        item = observation(4)
    elif change == "identity":
        e.identity_link_status = "unresolved_after_reconnect"
    elif change == "weak":
        item["persons"][0]["association_only"] = True
    elif change == "historical":
        item["evidence_kind"] = "historical"
    elif change == "nan":
        points[0, 0] = float("nan")
    elif change == "session":
        item["session_id"] = "new"
    result = e.update(item, points)
    assert not result["seated_pose_candidate"] and not result["would_close_episode"]


def test_knee_flexion_with_horizontal_shanks_cannot_establish_sitting():
    e = evaluator()
    points = pose()
    points[[15, 16], :2] = [[520, 350], [540, 350]]
    for t in (1, 1.5, 2, 2.5, 3):
        result = e.update(observation(t), points)
    assert result["geometric_candidate"]
    assert not result["seated_pose_candidate"]
    assert result["reason"] == "pose_not_consistent_with_seated_pattern"


def test_horizontal_evidence_blocks_even_later_good_pose():
    e = evaluator()
    e.update(observation(1, (300, 400, 400, 150)), pose())
    for t in (2, 2.5, 3, 3.5, 4):
        result = e.update(observation(t), pose())
        assert result["evidence_status"] == "INCIDENT_FOLLOWUP_REQUIRED"
        assert not result["seated_pose_candidate"] and not result["would_close_episode"]


def test_missing_joint_support_requests_body_and_seat_region_inside_frame():
    e = evaluator()
    item = observation(1)
    points = pose()
    points[15, 2] = 0.1
    result = e.update(item, points)
    request = result["suggested_observation"]
    assert result["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert request["action"] == "REQUEST_ROI" and not request["dispatched"]
    assert 0 <= request["x"] < request["x"] + request["width"] <= 1280
    assert 0 <= request["y"] < request["y"] + request["height"] <= 720
    assert request["x"] <= 400 and request["y"] <= 200
    assert request["x"] + request["width"] >= 560
    assert request["y"] + request["height"] >= 500
    assert not result["would_close_episode"]


def test_unlinked_identity_requests_fresh_context_without_accepting_pose():
    e = evaluator()
    e.identity_link_status = "unresolved_after_reconnect"
    result = e.update(observation(1), pose())
    assert result["reason"] == "episode_identity_unlinked"
    assert result["suggested_observation"]["action"] == "REQUEST_FULL_FRAME"
    assert not result["suggested_observation"]["dispatched"]
    assert not result["seated_pose_candidate"] and not result["would_close_episode"]


def test_prior_alert_keeps_incident_followup_with_good_seated_pose():
    e = evaluator(test_alert_emitted=True)
    result = e.update(observation(1), pose())
    assert result["evidence_status"] == "INCIDENT_FOLLOWUP_REQUIRED"
    assert result["suggested_observation"]["action"] == "KEEP_OBSERVING"
    assert not result["suggested_observation"]["dispatched"]
    assert not result["would_close_episode"]


def test_displaced_valid_joints_restart_pose_stability():
    e = evaluator()
    for t in (1, 1.5, 2):
        e.update(observation(t), pose())
    moved = pose()
    moved[[13, 14, 15, 16], 0] -= 40.1
    changed = e.update(observation(2.5), moved)
    assert changed["pose_stable_seconds"] == 0
    assert not changed["seated_pose_candidate"]
    result = e.update(observation(3), moved)
    assert result["geometric_candidate"]
    assert result["pose_stable_seconds"] == 0.5
    assert not result["seated_pose_candidate"]
