from copy import deepcopy

import pytest

from care_vision.seated_pose_evaluation import score_pose_intervals


def inputs(label="confuser", candidate=False, reason="pose_not_consistent_with_seated_pattern"):
    condition = {
        "source_sha256": "log",
        "model": "RTMPose",
        "timeline": [
            {
                "episode_id": "e",
                "seconds_from_first_observation": i / 2,
                "seated_pose_candidate": candidate,
                "reason": reason,
                "seat_support_verified": False,
                "would_close_episode": False,
            }
            for i in range(7)
        ],
    }
    review = {
        "source_sha256": "log",
        "source_video_sha256": "video",
        "review_scope": "full_video",
        "reviewed_by": "reviewer",
        "prediction_blinded": True,
        "dataset_split": "evaluation_reserved",
        "intervals": [
            {
                "start_seconds": 0,
                "end_seconds": 4,
                "label": label,
                "body_visibility": "adequate",
                "seat_support": "occluded",
                "context_status": "current",
            }
        ],
    }
    return condition, review


def test_absent_pose_does_not_become_successful_confuser_rejection():
    condition, review = inputs(reason="pose_unavailable")
    scored = score_pose_intervals(condition, review)
    assert scored["intervals"][0]["assessment"] == "indeterminate_pose_evidence"
    assert not scored["live_activation_supported"]


def test_known_confuser_pattern_is_an_error_even_if_pose_is_occluded():
    condition, review = inputs(candidate=True)
    review["intervals"][0]["body_visibility"] = "occluded"
    scored = score_pose_intervals(condition, review)
    assert scored["intervals"][0]["assessment"] == "unexpected_pattern_requires_review"


def test_sitting_pattern_with_hidden_support_cannot_validate_closure():
    condition, review = inputs("seated", True, "seat_support_unverified")
    before = deepcopy((condition, review))
    result = score_pose_intervals(condition, review)
    assert (
        result["intervals"][0]["assessment"] == "seated_pattern_observed_support_still_unverified"
    )
    assert not result["normal_activity_recognition_validated"]
    assert (condition, review) == before


def test_changed_scene_and_unknown_posture_remain_indeterminate():
    condition, review = inputs("seated", True)
    review["intervals"][0]["context_status"] = "changed"
    assert score_pose_intervals(condition, review)["intervals"][0]["assessment"] == (
        "indeterminate_context_or_label"
    )
    review["intervals"][0].update(context_status="current", label="unknown")
    assert score_pose_intervals(condition, review)["intervals"][0]["assessment"] == (
        "indeterminate_context_or_label"
    )


def test_observation_gap_cannot_count_as_complete_negative_interval():
    condition, review = inputs()
    condition["timeline"] = [condition["timeline"][0], condition["timeline"][-1]]
    assert score_pose_intervals(condition, review)["intervals"][0]["assessment"] == (
        "interrupted_observation"
    )


@pytest.mark.parametrize("mutation", ["unblinded", "no_video", "unknown_visibility", "closure"])
def test_invalid_or_unsafe_review_inputs_rejected(mutation):
    condition, review = inputs()
    if mutation == "unblinded":
        review["prediction_blinded"] = False
    elif mutation == "no_video":
        review["source_video_sha256"] = None
    elif mutation == "unknown_visibility":
        del review["intervals"][0]["body_visibility"]
    else:
        condition["timeline"][0]["would_close_episode"] = True
    with pytest.raises(ValueError):
        score_pose_intervals(condition, review)
