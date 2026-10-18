import copy
import json
from pathlib import Path

import pytest

from care_vision.cloud.temporal_candidate import (
    SequenceObservation,
    SequencePolicy,
    TemporalCandidate,
)
from care_vision.sequence_evaluation import (
    observation_from_row,
    replay_video,
    score_episodes,
    validate_observations,
)
from care_vision.temporal_evaluation import sample_indices


def obs(t, orientation="vertical", motion="stable", pose=False, track="test:track:1"):
    return SequenceObservation(t, round(t * 100), track, orientation, motion, pose)


def candidate(analyzer):
    assert not analyzer.update(obs(0, motion="downward"))["events"]
    assert not analyzer.update(obs(0.5, "horizontal"))["events"]
    assert not analyzer.update(obs(1, "horizontal"))["events"]
    assert analyzer.update(obs(1.5, "horizontal"))["events"] == ["candidate"]


def test_order_duration_and_pose_ablation():
    bbox, combined = TemporalCandidate(), TemporalCandidate(use_pose=True)
    for analyzer in (bbox, combined):
        candidate(analyzer)
    assert bbox.update(obs(2, "horizontal", pose=True))["events"] == []
    assert combined.update(obs(2, "horizontal", pose=True))["events"] == ["recovery_cue"]
    assert bbox.update(obs(2.5, "intermediate", "upward"))["events"] == ["recovery_cue"]
    assert combined.update(obs(2.5, "intermediate", "upward"))["events"] == []
    assert combined.episodes[0]["recovery_cue_frame"] == 200
    assert combined.episodes[0]["recovery_cue_sources"] == ["pose_straightening"]
    # A recovery cue keeps the episode open; it does not certify safety.
    assert combined.episodes[0]["end_reason"] is None


@pytest.mark.parametrize("mode", ["horizontal_only", "downward_while_horizontal", "squat"])
def test_horizontal_or_squat_without_ordered_sequence_cannot_activate(mode):
    analyzer = TemporalCandidate(use_pose=True)
    for i in range(15):
        o = obs(
            i / 2,
            "intermediate" if mode == "squat" else "horizontal",
            "stable" if mode == "horizontal_only" else "downward",
            pose=True,
        )
        assert not analyzer.update(o)["events"]
    assert analyzer.episodes == []


def test_descent_expires_and_interruption_restarts_horizontal_duration():
    analyzer = TemporalCandidate()
    analyzer.update(obs(0, motion="downward"))
    for i in range(1, 11):
        analyzer.update(obs(i / 2))
    for i in range(11, 15):
        analyzer.update(obs(i / 2, "horizontal"))
    assert not analyzer.episodes
    analyzer = TemporalCandidate()
    analyzer.update(obs(0, motion="downward"))
    analyzer.update(obs(0.5, "horizontal"))
    analyzer.update(obs(1, "intermediate"))
    analyzer.update(obs(1.5, "horizontal"))
    assert not analyzer.update(obs(2, "horizontal"))["events"]
    assert analyzer.update(obs(2.5, "horizontal"))["events"] == ["candidate"]


@pytest.mark.parametrize("loss", ["gap", "track", "missing", "orientation"])
@pytest.mark.parametrize("retain", [False, True])
def test_evidence_loss_aborts_active_episode_without_recovery(loss, retain):
    analyzer = TemporalCandidate(use_pose=True, retain_pending_descent=retain)
    candidate(analyzer)
    next_o = {
        "gap": obs(2.5, "horizontal", "upward", True),
        "track": obs(2, "horizontal", "upward", True, track="test:track:2"),
        "missing": obs(2, None, "unknown", None, track=None),
        "orientation": obs(2, None, "unknown", None),
    }[loss]
    result = analyzer.update(next_o)
    assert result["reset_reason"]
    assert not result["events"]
    assert analyzer.episodes[0]["recovery_cue_frame"] is None
    assert analyzer.episodes[0]["end_reason"] != "sustained_vertical_proxy"
    assert analyzer.active is None
    # Existing horizontality after reacquisition cannot replay the old descent.
    for i in range(1, 5):
        assert not analyzer.update(
            obs(
                next_o.timestamp_seconds + i / 2,
                "horizontal",
                track=next_o.track_id or "test:track:1",
            )
        )["events"]


def test_pose_unknown_does_not_invalidate_bbox_candidate_or_invent_a_cue():
    analyzer = TemporalCandidate(use_pose=True)
    candidate(analyzer)
    result = analyzer.update(obs(2, "horizontal", pose=None))
    assert result["state"] == "POST_DESCENT_HORIZONTAL"
    assert not result["pose_cue_available"]
    assert not result["events"]


def test_pending_descent_can_survive_short_missing_same_track_but_dwell_restarts():
    analyzer = TemporalCandidate(retain_pending_descent=True)
    analyzer.update(obs(0, motion="downward"))
    analyzer.update(obs(0.5, "horizontal"))
    missing = analyzer.update(obs(0.75, None, "unknown", None, track=None))
    assert missing["state"] == "UNKNOWN"
    assert missing["pending_descent_retained"]
    assert not missing["events"]
    assert analyzer.update(obs(1, "horizontal"))["pending_descent_retained"]
    for t in (1.5, 2):
        assert not analyzer.update(obs(t, "horizontal"))["events"]
    assert analyzer.update(obs(2.5, "horizontal"))["events"] == ["candidate"]
    assert analyzer.episodes[0]["horizontal_start_frame"] == 150


@pytest.mark.parametrize("different_track", [False, True])
def test_missing_ticks_do_not_extend_retention_and_identity_must_match(different_track):
    analyzer = TemporalCandidate(retain_pending_descent=True)
    analyzer.update(obs(0, motion="downward"))
    analyzer.update(obs(0.5, "horizontal"))
    analyzer.update(obs(0.75, None, "unknown", None, track=None))
    if different_track:
        reacquire = 1
        track = "test:track:2"
    else:
        for t in (1, 1.25, 1.5):
            analyzer.update(obs(t, None, "unknown", None, track=None))
        reacquire, track = 1.75, "test:track:1"
    for i in range(5):
        result = analyzer.update(obs(reacquire + i / 2, "horizontal", track=track))
        assert not result["events"]
    assert analyzer.descent is None


def test_recent_missing_does_not_extend_descent_memory():
    analyzer = TemporalCandidate(retain_pending_descent=True)
    analyzer.update(obs(0, motion="downward"))
    for i in range(1, 10):
        analyzer.update(obs(i / 2))
    analyzer.update(obs(4.75, None, "unknown", None, track=None))
    analyzer.update(obs(5, "horizontal"))
    for t in (5.5, 6, 6.5):
        assert not analyzer.update(obs(t, "horizontal"))["events"]
    assert not analyzer.episodes


def test_vertical_proxy_requires_uninterrupted_duration_and_allows_later_episode():
    analyzer = TemporalCandidate()
    candidate(analyzer)
    analyzer.update(obs(2))
    analyzer.update(obs(2.5))
    analyzer.update(obs(3, "intermediate"))
    for t in (3.5, 4, 4.5, 5):
        assert not analyzer.update(obs(t))["events"]
    assert analyzer.update(obs(5.5))["events"] == ["upright_proxy"]
    assert analyzer.episodes[0]["end_reason"] == "sustained_vertical_proxy"
    analyzer.update(obs(6, motion="downward"))
    analyzer.update(obs(6.5, "horizontal"))
    analyzer.update(obs(7, "horizontal"))
    assert analyzer.update(obs(7.5, "horizontal"))["events"] == ["candidate"]
    assert len(analyzer.episodes) == 2


def test_fast_vertical_close_clears_state_even_before_descent_memory_expires():
    analyzer = TemporalCandidate()
    candidate(analyzer)
    for t in (2, 2.5, 3, 3.5):
        analyzer.update(obs(t))
    result = analyzer.update(obs(4))
    assert result["events"] == ["upright_proxy"]
    assert result["state"] == "OBSERVING"


def test_gaps_do_not_count_as_horizontal_dwell_and_time_must_increase():
    analyzer = TemporalCandidate()
    analyzer.update(obs(0, motion="downward"))
    analyzer.update(obs(0.5, "horizontal"))
    assert not analyzer.update(obs(1.5, "horizontal"))["events"]
    for t in (1.5, 1.4):
        with pytest.raises(ValueError, match="increase"):
            analyzer.update(obs(t))


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf")])
def test_invalid_policy_rejected(invalid):
    with pytest.raises(ValueError):
        SequencePolicy(horizontal_seconds=invalid)


def row(o, phase="normal"):
    return {
        "frame_index": o.frame_index,
        "source_time_seconds": o.timestamp_seconds,
        "primary_track_id": o.track_id,
        "primary_orientation": o.orientation,
        "primary_motion_direction": o.motion,
        "pose_straightening": o.pose_straightening,
        "phase": phase,
    }


def test_annotations_cannot_affect_decisions_and_future_append_preserves_prefix():
    video = {"video_id": "test", "fps": 100, "frame_count": 400, "events": []}
    rows = [
        row(obs(0, motion="downward")),
        row(obs(0.5, "horizontal")),
        row(obs(1, "horizontal")),
        row(obs(1.5, "horizontal")),
    ]
    prefix = replay_video(video, rows, use_pose=True)
    poisoned = [{**r, "phase": "recovering", "event_id": "fabricated"} for r in rows]
    full = replay_video(video, poisoned + [row(obs(2, pose=True))], use_pose=True)
    assert full["timeline"][: len(rows)] == prefix["timeline"]
    assert prefix["episodes"][0]["end_reason"] == "end_of_recording_unresolved"
    assert prefix["episodes"][0]["recovery_cue_frame"] is None
    assert observation_from_row(poisoned[0]) == observation_from_row(rows[0])


def test_scoring_does_not_hide_early_cues_duplicates_misses_or_boundary_activations():
    event = {
        "event_id": "fall",
        "descent_onset_frames": [10, 15],
        "floor_contact_frames": [20, 25],
        "recovery_onset_frames": [50, 55],
        "upright_again_frames": [80, 85],
    }
    video = {"events": [event], "fps": 10}
    episodes = [
        {"episode_id": i, "candidate_frame": f, "recovery_cue_frame": 45}
        for i, f in enumerate((18, 22, 30, 35, 52, 90), 1)
    ]
    result = score_episodes(video, episodes)
    assert result["matched_events"] == 1
    assert result["candidate_counts"] == {
        "unmatched": 2,
        "uncertain_boundary": 2,
        "matched_floor_rest": 1,
        "duplicate": 1,
    }
    assert result["events"][0]["candidate_delay_from_floor_contact_seconds"] == [0.5, 1]
    assert result["events"][0]["recovery_cue"] == {"status": "early", "delay_seconds": [-1, -0.5]}
    missing = score_episodes(video, [])
    assert missing["missed_events"] == 1
    assert missing["events"][0]["recovery_cue"]["status"] == "no_matched_candidate"


@pytest.mark.parametrize(
    "frame,status",
    [
        (None, "missing"),
        (50, "uncertain_boundary"),
        (55, "in_recovery"),
        (80, "uncertain_boundary"),
        (85, "late"),
    ],
)
def test_recovery_cue_boundary_scoring(frame, status):
    from care_vision.sequence_evaluation import _cue_score

    event = {"recovery_onset_frames": [50, 55], "upright_again_frames": [80, 85]}
    assert _cue_score(event, frame, 10)["status"] == status


@pytest.fixture
def cached_report():
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads((root / "evaluation/phase7d-temporal.json").read_text())
    videos = []
    for video in dataset["videos"]:
        samples = []
        for i in sample_indices(video["frame_count"], video["fps"], 2):
            samples.append(
                row(
                    SequenceObservation(
                        i / video["fps"], i, "test:track:1", "vertical", "stable", None
                    )
                )
            )
        videos.append(
            {"video_id": video["video_id"], "source_sha256": video["sha256"], "samples": samples}
        )
    report = {
        "evaluation": "phase7d-offline-pose-comparison",
        "dataset_sha256": "dataset",
        "dataset_id": dataset["dataset_id"],
        "evidence_manifest_sha256": "manifest",
        "runs": [{"sample_fps": 2, "videos": videos}],
    }
    return dataset, report


@pytest.mark.parametrize(
    "corruption",
    [
        "hash",
        "manifest",
        "timestamp",
        "missing_frame",
        "duplicate_frame",
        "order",
        "missing_video",
        "duplicate_rate",
        "source_hash",
        "nan",
        "invalid_pose",
    ],
)
def test_replay_rejects_corrupted_or_incomplete_cache(cached_report, corruption):
    dataset, report = cached_report
    validate_observations(report, dataset, "dataset", "manifest")
    broken = copy.deepcopy(report)
    video = broken["runs"][0]["videos"][0]
    rows = video["samples"]
    if corruption == "hash":
        broken["dataset_sha256"] = "wrong"
    elif corruption == "manifest":
        broken["evidence_manifest_sha256"] = "wrong"
    elif corruption == "timestamp":
        rows[1]["source_time_seconds"] += 0.01
    elif corruption == "missing_frame":
        rows.pop()
    elif corruption == "duplicate_frame":
        rows.insert(1, rows[0])
    elif corruption == "order":
        rows.reverse()
    elif corruption == "missing_video":
        broken["runs"][0]["videos"].pop()
    elif corruption == "duplicate_rate":
        broken["runs"].append(copy.deepcopy(broken["runs"][0]))
    elif corruption == "source_hash":
        video["source_sha256"] = "wrong"
    elif corruption == "nan":
        rows[0]["source_time_seconds"] = float("nan")
    else:
        rows[0]["pose_straightening"] = "true"
    with pytest.raises(ValueError):
        validate_observations(broken, dataset, "dataset", "manifest")
