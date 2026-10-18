import importlib.util
from pathlib import Path

import pytest


def tool(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / f"tools/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = tool("prepare_tracking_challenge_plan")
scorer = tool("score_tracking_challenge")


def test_schedule_uses_pts_and_never_duplicates_frames():
    frames = planner.schedule([10, 10.1, 10.49, 10.7, 11.02], 2)
    assert [f["frame_index"] for f in frames] == [0, 2, 4]
    assert frames[1]["captured_mono_ns"] == 1_490_000_000


@pytest.mark.parametrize("pts", [[], [0, 0], [0, 2, 1], [0, float("nan")]])
def test_invalid_media_pts_rejected(pts):
    with pytest.raises(ValueError):
        planner.schedule(pts, 2)


@pytest.mark.parametrize("fps", [0, -1, float("inf")])
def test_invalid_sampling_rate_rejected(fps):
    with pytest.raises(ValueError):
        planner.schedule([0, 1], fps)


def test_box_scaling_matches_stretched_image_without_mutating_labels():
    sample = {"persons": [{"identity_id": "a", "bbox": [10, 20, 30, 40]}]}
    truth = scorer.scaled_truth(sample, 640, 1440)
    assert truth == [{"identity_id": "a", "bbox": [20, 10, 60, 20]}]
    assert sample["persons"][0]["bbox"] == [10, 20, 30, 40]


def test_sparse_misses_and_identity_changes_are_separate():
    def row(identity=None, track=None):
        return {
            "expected": 1,
            "predicted": int(track is not None),
            "misses_with_raw_match": int(track is None),
            "misses_without_raw_match": 0,
            "matches": [] if track is None else [{"identity_id": identity, "track_id": track}],
        }

    result = scorer.score_rows([row("a", "1"), row(), row("a", "2"), row("b", "2")])
    assert result["matched_instances_iou05"] == 3
    assert result["misses_with_raw_detection_match"] == 1
    assert result["id_changes_between_matched_annotation_points"] == 1
    assert result["track_ids_matched_to_multiple_annotation_identities"] == 1
    assert result["distinct_track_ids_by_identity"] == {"a": 2, "b": 1}


def test_letterbox_label_offsets_and_scales_match_portrait_canvas():
    sample = {"persons": [{"identity_id": "a", "bbox": [0, 0, 1080, 1920]}]}
    result = scorer.scaled_truth(sample, 1080, 1920, [0.375, 0.375, 437, 0])
    assert result[0]["bbox"] == [437, 0, 405, 720]


def test_empty_annotation_frames_score_false_predictions():
    result = scorer.score_rows(
        [
            {
                "expected": 0,
                "predicted": 2,
                "matches": [],
                "misses_with_raw_match": 0,
                "misses_without_raw_match": 0,
            }
        ]
    )
    assert result["unmatched_predictions"] == 2
    assert result["expected_instances"] == 0
