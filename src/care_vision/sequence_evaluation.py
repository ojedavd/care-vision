"""Replay cached vision observations, then score sequence candidates separately."""

import math
from collections import Counter
from dataclasses import asdict

from care_vision.cloud.temporal_candidate import (
    SequenceObservation,
    SequencePolicy,
    TemporalCandidate,
)
from care_vision.temporal_evaluation import sample_indices, validate_dataset


def observation_from_row(row):
    # Deliberately excludes row['phase'], video metadata and event boundaries.
    return SequenceObservation(
        timestamp_seconds=row["source_time_seconds"],
        frame_index=row["frame_index"],
        track_id=row["primary_track_id"],
        orientation=row["primary_orientation"],
        motion=row["primary_motion_direction"],
        pose_straightening=row["pose_straightening"],
    )


def validate_observations(report, dataset, dataset_sha256, manifest_sha256):
    validate_dataset(dataset)
    if (
        report.get("evaluation") != "phase7d-offline-pose-comparison"
        or report.get("dataset_sha256") != dataset_sha256
        or report.get("dataset_id") != dataset["dataset_id"]
        or report.get("evidence_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("observation provenance mismatch")
    sources = {v["video_id"]: v for v in dataset["videos"]}
    rates = []
    for run in report["runs"]:
        rate = run["sample_fps"]
        if not math.isfinite(rate) or rate <= 0 or rate in rates:
            raise ValueError("invalid or duplicate sampling rate")
        rates.append(rate)
        seen = set()
        for video in run["videos"]:
            video_id = video["video_id"]
            if video_id not in sources or video_id in seen:
                raise ValueError("unexpected or duplicate video")
            seen.add(video_id)
            source = sources[video_id]
            if video["source_sha256"] != source["sha256"]:
                raise ValueError("source hash mismatch")
            rows = video["samples"]
            indices = sample_indices(source["frame_count"], source["fps"], rate)
            if [r["frame_index"] for r in rows] != indices:
                raise ValueError("observations must cover the exact ordered sampling schedule")
            for row in rows:
                observation_from_row(row)  # validate every consumed field
                if not math.isclose(
                    row["source_time_seconds"],
                    row["frame_index"] / source["fps"],
                    abs_tol=1e-9,
                    rel_tol=0,
                ):
                    raise ValueError("source timestamp mismatch")
        if seen != set(sources):
            raise ValueError("missing video")
    if not rates:
        raise ValueError("no observation runs")


def _cue_score(event, frame, fps):
    if frame is None:
        return {"status": "missing", "delay_seconds": None}
    lo, hi = event["recovery_onset_frames"]
    upright_lo, upright_hi = event["upright_again_frames"]
    status = (
        "early"
        if frame < lo
        else "uncertain_boundary"
        if frame < hi
        else "in_recovery"
        if frame < upright_lo
        else "uncertain_boundary"
        if frame < upright_hi
        else "late"
    )
    return {"status": status, "delay_seconds": [(frame - hi) / fps, (frame - lo) / fps]}


def score_episodes(video, episodes):
    """One-to-one matching in definite floor rest, not a fall-detection metric.

    Boundary activations are retained as ambiguous; early/late/duplicate
    activations remain unmatched. Never cherry-pick later recovery cues.
    """
    matched, scored = {}, []
    for episode in episodes:
        frame = episode["candidate_frame"]
        status, event_id = "unmatched", None
        for event in video["events"]:
            lo, hi = event["floor_contact_frames"]
            recovery_lo, recovery_hi = event["recovery_onset_frames"]
            if hi <= frame < recovery_lo:
                event_id = event["event_id"]
                status = "duplicate" if event_id in matched else "matched_floor_rest"
                if status == "matched_floor_rest":
                    matched[event_id] = episode
                break
            if lo <= frame < hi or recovery_lo <= frame < recovery_hi:
                status, event_id = "uncertain_boundary", event["event_id"]
                break
        scored.append({**episode, "match_status": status, "annotated_event_id": event_id})
    events = []
    for event in video["events"]:
        episode = matched.get(event["event_id"])
        frame = episode["candidate_frame"] if episode else None
        lo, hi = event["floor_contact_frames"]
        events.append(
            {
                "event_id": event["event_id"],
                "matched_episode_id": episode["episode_id"] if episode else None,
                "candidate_frame": frame,
                "candidate_delay_from_floor_contact_seconds": [
                    (frame - hi) / video["fps"],
                    (frame - lo) / video["fps"],
                ]
                if episode
                else None,
                "recovery_cue": _cue_score(event, episode["recovery_cue_frame"], video["fps"])
                if episode
                else {"status": "no_matched_candidate", "delay_seconds": None},
            }
        )
    return {
        "annotated_events": len(events),
        "matched_events": len(matched),
        "missed_events": len(events) - len(matched),
        "candidate_counts": dict(Counter(e["match_status"] for e in scored)),
        "recovery_cue_counts": dict(Counter(e["recovery_cue"]["status"] for e in events)),
        "episodes": scored,
        "events": events,
    }


def replay_video(video, rows, *, use_pose=False, retain_pending_descent=False):
    analyzer = TemporalCandidate(use_pose=use_pose, retain_pending_descent=retain_pending_descent)
    timeline = [analyzer.update(observation_from_row(row)) for row in rows]
    # End of recording is censoring, never evidence of recovery or safety.
    episodes = [
        {**e, "end_reason": e["end_reason"] or "end_of_recording_unresolved"}
        for e in analyzer.episodes
    ]
    return {
        "video_id": video["video_id"],
        "source_duration_seconds": video["frame_count"] / video["fps"],
        "observations": len(rows),
        "unknown_bbox_observations": sum(not r["bbox_available"] for r in timeline),
        "unknown_motion_observations": sum(not r["motion_available"] for r in timeline),
        "unknown_pose_cue_observations": sum(not r["pose_cue_available"] for r in timeline),
        "reset_counts": dict(Counter(r["reset_reason"] for r in timeline if r["reset_reason"])),
        **score_episodes(video, episodes),
        "timeline": timeline,
    }


def evaluate_sequences(report, dataset):
    sources = {v["video_id"]: v for v in dataset["videos"]}
    runs = []
    for run in report["runs"]:
        for use_pose, retain_pending_descent in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            videos = [
                replay_video(
                    sources[v["video_id"]],
                    v["samples"],
                    use_pose=use_pose,
                    retain_pending_descent=retain_pending_descent,
                )
                for v in run["videos"]
            ]
            runs.append(
                {
                    "sample_fps": run["sample_fps"],
                    "variant": "bbox_plus_pose_cue" if use_pose else "bbox_only",
                    "gap_policy": "retain_pending_descent"
                    if retain_pending_descent
                    else "strict_reset",
                    "annotated_events": sum(v["annotated_events"] for v in videos),
                    "matched_events": sum(v["matched_events"] for v in videos),
                    "missed_events": sum(v["missed_events"] for v in videos),
                    "negative_clip_candidates": sum(
                        len(v["episodes"]) for v in videos if v["annotated_events"] == 0
                    ),
                    "negative_clip_duration_seconds": sum(
                        v["source_duration_seconds"] for v in videos if v["annotated_events"] == 0
                    ),
                    "videos": videos,
                }
            )
    return {
        "evaluation": "phase7e-offline-temporal-sequences",
        "policy": asdict(SequencePolicy()),
        "threshold_status": "development hypothesis, not calibrated or held-out validation",
        "fall_classification_enabled": False,
        "commands_enabled": False,
        "runs": runs,
    }
