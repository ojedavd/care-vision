import pytest

from care_vision.evaluation import (
    identity_tracking_metrics,
    multi_object_tracking_metrics,
    positive_segments,
    primary_track_continuity,
)


def test_positive_segments_split_on_negatives_and_time_gaps():
    samples = [
        {"second": 0, "persons": []},
        {"second": 1, "persons": [{}]},
        {"second": 2, "persons": [{}]},
        {"second": 3, "persons": []},
        {"second": 5, "persons": [{}]},
    ]
    segments = positive_segments(samples)
    assert [[item["second"] for item in segment] for segment in segments] == [[1, 2], [5]]


def test_primary_track_continuity_counts_switches_across_detection_gaps():
    summary = primary_track_continuity(["track:1", None, "track:1", "track:2", "track:1"])

    assert summary == {
        "frames_with_confirmed_track": 4,
        "unique_primary_track_ids": 2,
        "primary_track_switches": 2,
        "primary_track_ids": ["track:1", "track:1", "track:2", "track:1"],
    }


def test_identity_tracking_metrics_report_coverage_fragmentation_and_switches():
    metrics = identity_tracking_metrics(
        [
            {"video_id": "v", "frame_index": 1, "identity_id": "alice", "track_id": "t1"},
            {"video_id": "v", "frame_index": 2, "identity_id": "alice", "track_id": None},
            {"video_id": "v", "frame_index": 3, "identity_id": "alice", "track_id": "t2"},
            {"video_id": "v", "frame_index": 1, "identity_id": "bob", "track_id": "t3"},
            {"video_id": "v", "frame_index": 2, "identity_id": "bob", "track_id": "t4"},
        ]
    )

    assert metrics["ground_truth_identities"] == 2
    assert metrics["annotated_identity_instances"] == 5
    assert metrics["matched_identity_instances"] == 4
    assert metrics["identity_coverage"] == 0.8
    assert metrics["unique_track_ids_across_identities"] == 4
    assert metrics["adjacent_track_switches"] == 1
    assert metrics["identities"][0]["unique_track_ids"] == 2
    assert metrics["identities"][0]["adjacent_track_switches"] == 0
    assert metrics["identities"][1]["adjacent_track_switches"] == 1


def test_identity_tracking_metrics_reject_empty_assignments():
    with pytest.raises(ValueError, match="must not be empty"):
        identity_tracking_metrics([])


def test_multi_object_tracking_metrics_include_global_identity_assignment():
    metrics = multi_object_tracking_metrics(
        [
            {
                "video_id": "v",
                "frame_index": 1,
                "expected_persons": 2,
                "confirmed_persons": 2,
                "matches": [
                    {"identity_id": "alice", "track_id": "t1", "iou": 0.8},
                    {"identity_id": "bob", "track_id": "t2", "iou": 0.6},
                ],
            },
            {
                "video_id": "v",
                "frame_index": 2,
                "expected_persons": 2,
                "confirmed_persons": 2,
                "matches": [
                    {"identity_id": "alice", "track_id": "t2", "iou": 0.7},
                ],
            },
        ]
    )

    assert metrics["matches"] == 3
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["identity_switches"] == 1
    assert metrics["mota"] == 0.25
    assert metrics["motp"] == pytest.approx(0.7)
    assert metrics["identity_true_positive"] == 2
    assert metrics["identity_false_positive"] == 2
    assert metrics["identity_false_negative"] == 2
    assert metrics["idf1"] == 0.5


def test_multi_object_tracking_metrics_reject_empty_samples():
    with pytest.raises(ValueError, match="must not be empty"):
        multi_object_tracking_metrics([])
