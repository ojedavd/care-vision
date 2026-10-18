from copy import deepcopy

import pytest

from care_vision.seated_evaluation import score_intervals


def source():
    return {
        "source_sha256": "source-hash",
        "timeline": [
            {"seconds_from_first_observation": t, "seated_like_candidate": candidate,
             "reason": "test_signal", "episode_id": "episode"}
            for t, candidate in [(2, False), (3, True), (6, False), (7, True)]
        ],
    }


def review():
    return {
        "source_sha256": "source-hash", "review_scope": "full_video",
        "reviewed_by": "synthetic test reviewer", "dataset_split": "development",
        "intervals": [
            {"start_seconds": 1, "end_seconds": 4, "label": "seated"},
            {"start_seconds": 5, "end_seconds": 8, "label": "confuser"},
            {"start_seconds": 9, "end_seconds": 12, "label": "pending_incident"},
        ],
    }


def test_candidates_in_confusers_are_errors_and_unobserved_intervals_are_not_successes():
    result = score_intervals(source(), review())
    seated, confuser, missing = result["intervals"]
    assert seated["outcome"] == "candidate_found"
    assert seated["first_candidate_delay_seconds"] == 2
    assert confuser["outcome"] == "unexpected_seated_candidate"
    assert confuser["candidate_episode_ids"] == ["episode"]
    assert missing["outcome"] == "not_observed"
    assert not result["live_activation_supported"]


def test_unknown_labels_and_unannotated_samples_remain_explicit():
    annotation = review()
    annotation["intervals"] = [{"start_seconds": 1, "end_seconds": 4, "label": "unknown"}]
    result = score_intervals(source(), annotation)
    assert result["intervals"][0]["outcome"] == "indeterminate"
    assert result["unannotated_shadow_samples"] == 2


@pytest.mark.parametrize("change", ["hash", "cues", "reviewer", "overlap", "nan", "empty"])
def test_invalid_or_unreviewed_annotations_are_rejected(change):
    annotation = review()
    if change == "hash":
        annotation["source_sha256"] = "wrong"
    elif change == "cues":
        annotation["review_scope"] = "planned_cues"
    elif change == "reviewer":
        annotation["reviewed_by"] = ""
    elif change == "overlap":
        annotation["intervals"][1]["start_seconds"] = 3
    elif change == "nan":
        annotation["intervals"][0]["end_seconds"] = float("nan")
    else:
        annotation["intervals"] = []
    with pytest.raises(ValueError):
        score_intervals(source(), annotation)


def test_reordered_timeline_cannot_shift_candidates_into_different_intervals():
    run = deepcopy(source())
    run["timeline"].reverse()
    with pytest.raises(ValueError, match="chronological"):
        score_intervals(run, review())
