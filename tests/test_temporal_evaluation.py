import copy
import json
from pathlib import Path

import pytest

from care_vision.temporal_evaluation import (
    event_signal_timing,
    phase_at,
    sample_indices,
    signal_metrics,
    validate_dataset,
    validate_evidence_manifest,
)


@pytest.fixture
def dataset():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "evaluation/phase7c-temporal.json").read_text())


def test_real_annotations_cover_every_frame_and_exclude_transition_boundaries(dataset):
    validate_dataset(dataset)
    video = dataset["videos"][0]
    assert phase_at(video, 179) == "uncertain"
    assert phase_at(video, 180) == "descending"
    assert phase_at(video, 255) == "floor_rest"
    with pytest.raises(ValueError):
        phase_at(video, video["frame_count"])


@pytest.mark.parametrize("change", ["overlap", "gap", "incomplete", "event_order", "duplicate"])
def test_reject_annotations_that_could_silently_corrupt_metrics(dataset, change):
    broken = copy.deepcopy(dataset)
    video = broken["videos"][0]
    if change == "overlap":
        video["segments"][1]["start_frame"] -= 1
    elif change == "gap":
        video["segments"][1]["start_frame"] += 1
    elif change == "incomplete":
        video["segments"].pop()
    elif change == "event_order":
        video["events"][0]["floor_contact_frames"] = [100, 110]
    else:
        broken["videos"].append(copy.deepcopy(video))
    with pytest.raises(ValueError):
        validate_dataset(broken)


def test_missing_positives_reduce_coverage_and_missing_negatives_are_not_true_negatives():
    rows = [
        {"phase": "descending", "primary_motion_direction": "downward"},
        {"phase": "descending", "primary_motion_direction": "stable"},
        {"phase": "descending", "primary_motion_direction": "unknown"},
        {"phase": "normal", "primary_motion_direction": "unknown"},
        {"phase": "normal", "primary_motion_direction": "downward"},
        {"phase": "normal", "primary_motion_direction": "stable"},
        {"phase": "uncertain", "primary_motion_direction": "downward"},
    ]
    result = signal_metrics(rows, "downward")
    assert result["positive_samples"] == result["negative_samples"] == 3
    assert result["tp"] == result["fp"] == result["tn"] == result["fn"] == 1
    assert result["excluded"] == 1
    assert result["recall_available"] == 0.5
    assert result["positive_coverage_including_missing"] == pytest.approx(1 / 3)
    assert result["availability"] == pytest.approx(4 / 6)


def test_undefined_metrics_are_null_and_recovery_is_not_a_floor_negative():
    rows = [{"phase": "recovering", "primary_orientation": "horizontal"}]
    result = signal_metrics(rows, "horizontal")
    assert result["excluded"] == 1
    assert result["fp"] == result["tn"] == 0
    assert result["recall_available"] is None
    assert result["precision_available"] is None
    assert result["false_positive_rate_available"] is None


def test_sampler_preserves_source_time_and_does_not_invent_frames():
    assert sample_indices(60, 30, 2) == [0, 15, 30, 45]
    indices = sample_indices(492, 23.998048939110642, 20)
    assert len(indices) == len(set(indices)) == 410
    assert indices[-1] < 492
    assert sample_indices(2, 30, 20) == [0]
    for invalid in (0, -1, float("nan"), float("inf"), 31):
        with pytest.raises(ValueError):
            sample_indices(60, 30, invalid)


def test_event_timing_ignores_early_false_positives_and_preserves_misses(dataset):
    video = dataset["videos"][0]
    rows = [
        {"frame_index": 100, "phase": "normal", "primary_motion_direction": "downward"},
        {"frame_index": 175, "phase": "uncertain", "primary_motion_direction": "downward"},
        {"frame_index": 195, "phase": "descending", "primary_motion_direction": "downward"},
        {"frame_index": 270, "phase": "floor_rest", "primary_orientation": None},
    ]
    result = {r["signal"]: r for r in event_signal_timing(video, rows)}
    assert result["downward"]["first_positive_frame"] == 195
    assert result["downward"]["delay_from_onset_bracket_seconds"] == pytest.approx(
        [15 / video["fps"], 30 / video["fps"]]
    )
    assert result["horizontal"]["phase_samples"] == 1
    assert result["horizontal"]["available_samples"] == 0
    assert result["horizontal"]["first_positive_frame"] is None
    assert result["horizontal"]["delay_from_onset_bracket_seconds"] is None


@pytest.fixture
def evidence_manifest(dataset):
    return {
        "schema_version": 1,
        "dataset_sha256": "d" * 64,
        "width": 1280,
        "height": 720,
        "source_hashes": {v["video_id"]: v["sha256"] for v in dataset["videos"]},
        "frames": [
            {
                "video_id": v["video_id"],
                "frame_index": i,
                "filename": f"{v['video_id']}-{i}.jpg",
                "sha256": "a" * 64,
            }
            for v in dataset["videos"]
            for i in sample_indices(v["frame_count"], v["fps"], 2)
        ],
    }


def test_evidence_preserves_exact_source_schedule(dataset, evidence_manifest):
    validate_evidence_manifest(evidence_manifest, dataset, "d" * 64, [2])
    # A 2 FPS pack cannot be silently reused as 20 FPS evidence.
    with pytest.raises(ValueError, match="missing scheduled"):
        validate_evidence_manifest(evidence_manifest, dataset, "d" * 64, [20])


@pytest.mark.parametrize("change", ["dataset", "source", "missing", "duplicate", "path"])
def test_evidence_rejects_misbound_missing_or_unsafe_frames(dataset, evidence_manifest, change):
    if change == "dataset":
        evidence_manifest["dataset_sha256"] = "b" * 64
    elif change == "source":
        evidence_manifest["source_hashes"][dataset["videos"][0]["video_id"]] = "b" * 64
    elif change == "missing":
        evidence_manifest["frames"].pop()
    elif change == "duplicate":
        evidence_manifest["frames"].append(evidence_manifest["frames"][0])
    else:
        evidence_manifest["frames"][0]["filename"] = "../outside.jpg"
    with pytest.raises(ValueError):
        validate_evidence_manifest(evidence_manifest, dataset, "d" * 64, [2])
