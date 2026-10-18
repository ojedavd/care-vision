import pytest

from care_vision.cloud.person import PersonDetection
from care_vision.evaluation import (
    frame_localization_diagnostics,
    intersection_over_union,
    match_detection_pairs,
    match_detections,
    person_count_metrics,
    person_detection_metrics,
)


def test_iou_uses_xywh_and_handles_disjoint_boxes():
    assert intersection_over_union((0, 0, 10, 10), (5, 5, 10, 10)) == 25 / 175
    assert intersection_over_union((0, 0, 10, 10), (20, 20, 2, 2)) == 0


def test_matching_is_one_to_one_and_respects_threshold():
    predictions = [
        PersonDetection((0, 0, 10, 10), 0.9),
        PersonDetection((1, 1, 8, 8), 0.8),
    ]
    matched_predictions, matched_truth, scores = match_detections(
        predictions, [{"bbox": [0, 0, 10, 10]}], 0.5
    )
    assert matched_predictions == {0}
    assert matched_truth == {0}
    assert scores == [1]


def test_matching_retains_prediction_and_identity_indexes():
    predictions = [
        PersonDetection((20, 0, 10, 10), 0.8),
        PersonDetection((0, 0, 10, 10), 0.9),
    ]
    truth = [
        {"identity_id": "left", "bbox": [0, 0, 10, 10]},
        {"identity_id": "right", "bbox": [20, 0, 10, 10]},
    ]

    assert match_detection_pairs(predictions, truth, 0.5) == [(1, 0, 1), (0, 1, 1)]


def test_person_detection_metrics_aggregate_instances_and_iou():
    metrics = person_detection_metrics(
        [
            {
                "expected_person_count": 2,
                "predicted_person_count": 2,
                "matched_ious": [0.8, 0.6],
            },
            {
                "expected_person_count": 1,
                "predicted_person_count": 2,
                "matched_ious": [0.7],
            },
        ]
    )

    assert metrics == {
        "true_positive": 3,
        "false_positive": 1,
        "false_negative": 0,
        "precision": 0.75,
        "recall": 1,
        "f1": 6 / 7,
        "matched_iou_median": 0.7,
    }


def test_person_detection_metrics_reject_empty_rows():
    with pytest.raises(ValueError, match="must not be empty"):
        person_detection_metrics([])


def test_localization_diagnostics_separate_presence_and_box_failures():
    metrics = frame_localization_diagnostics(
        [
            {"expected_person_count": 1, "predicted_person_count": 1, "matched_ious": [0.7]},
            {"expected_person_count": 1, "predicted_person_count": 1, "matched_ious": []},
            {"expected_person_count": 1, "predicted_person_count": 0, "matched_ious": []},
            {"expected_person_count": 0, "predicted_person_count": 1, "matched_ious": []},
            {"expected_person_count": 0, "predicted_person_count": 0, "matched_ious": []},
        ]
    )

    assert metrics == {
        "positive_frames": 3,
        "positive_frames_with_any_detection": 2,
        "positive_frames_with_iou_match": 1,
        "positive_frames_detection_without_iou_match": 1,
        "positive_frames_without_detection": 1,
        "negative_frames": 2,
        "negative_frames_with_detection": 1,
        "presence_recall": 2 / 3,
        "localized_frame_recall": 1 / 3,
    }


def test_localization_diagnostics_reject_empty_rows():
    with pytest.raises(ValueError, match="must not be empty"):
        frame_localization_diagnostics([])


def test_person_count_metrics_separate_presence_from_exact_count():
    metrics = person_count_metrics([0, 1, 2, 2], [1, 1, 1, 2])

    assert metrics["presence"] == {
        "true_positive": 3,
        "false_positive": 1,
        "true_negative": 0,
        "false_negative": 0,
        "precision": 0.75,
        "recall": 1,
        "f1": 6 / 7,
    }
    assert metrics["count"] == {
        "exact_match_frames": 2,
        "exact_match_rate": 0.5,
        "overcount_frames": 1,
        "undercount_frames": 1,
        "mean_absolute_error": 0.5,
        "maximum_absolute_error": 1,
    }


def test_person_count_metrics_reject_invalid_inputs():
    with pytest.raises(ValueError, match="same non-zero length"):
        person_count_metrics([], [])
    with pytest.raises(ValueError, match="non-negative integers"):
        person_count_metrics([1], [-1])
