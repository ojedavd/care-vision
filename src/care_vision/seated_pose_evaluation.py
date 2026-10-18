"""Score reserved seated-pose diagnostics, keeping missing evidence indeterminate."""

from collections import Counter

from care_vision.seated_evaluation import score_intervals

UNAVAILABLE_REASONS = {
    "pose_unavailable",
    "pose_invalid",
    "pose_support_insufficient",
    "evidence_unavailable",
    "context_or_stream_mismatch",
    "missing_or_ambiguous_person",
    "episode_identity_unlinked",
    "track_continuity_unavailable",
    "weak_association_cannot_establish_benign_posture",
    "invalid_bbox",
}


def score_pose_intervals(condition, annotation):
    if annotation.get("prediction_blinded") is not True:
        raise ValueError("review must be completed without consulting predictions")
    if annotation.get("dataset_split") not in {"development", "evaluation_reserved"}:
        raise ValueError("review requires dataset split")
    if not annotation.get("source_video_sha256"):
        raise ValueError("review must identify the recorded video")
    timeline = condition["timeline"]
    if any(row.get("would_close_episode") or row.get("seat_support_verified") for row in timeline):
        raise ValueError("this frozen shadow version cannot verify support or close episodes")
    adapted = {
        **condition,
        "timeline": [
            {**row, "seated_like_candidate": row["seated_pose_candidate"]} for row in timeline
        ],
    }
    for interval in annotation["intervals"]:
        if interval.get("body_visibility") not in {"adequate", "occluded", "unknown"}:
            raise ValueError("interval requires reviewed body visibility")
        if interval.get("seat_support") not in {"visible", "occluded", "unknown"}:
            raise ValueError("interval requires seat-support observability")
        if interval.get("context_status") not in {"current", "changed", "unknown"}:
            raise ValueError("interval requires scene-context status")
    result = score_intervals(adapted, annotation)
    for interval in result["intervals"]:
        samples = [
            r
            for r in timeline
            if interval["start_seconds"]
            <= r["seconds_from_first_observation"]
            < interval["end_seconds"]
        ]
        available = [r for r in samples if r["reason"] not in UNAVAILABLE_REASONS]
        interval["pose_evaluable_samples"] = len(available)
        interval["unavailable_samples"] = len(samples) - len(available)
        interval["support_observability_validated_by_model"] = False
        if interval["outcome"] == "unexpected_seated_candidate":
            interval["assessment"] = "unexpected_pattern_requires_review"
        elif interval["label"] == "unknown" or interval["context_status"] != "current":
            interval["assessment"] = "indeterminate_context_or_label"
        elif interval["body_visibility"] != "adequate" or not available:
            interval["assessment"] = "indeterminate_pose_evidence"
        elif len(samples) < 2 or (
            samples[-1]["seconds_from_first_observation"]
            - samples[0]["seconds_from_first_observation"]
            < 2
        ):
            interval["assessment"] = "insufficient_observed_duration"
        elif len(available) / len(samples) < 0.8:
            interval["assessment"] = "insufficient_pose_coverage"
        elif any(
            b["seconds_from_first_observation"] - a["seconds_from_first_observation"] > 0.8
            for a, b in zip(samples, samples[1:])
        ):
            interval["assessment"] = "interrupted_observation"
        elif interval["label"] == "seated" and interval["candidate_observations"]:
            interval["assessment"] = "seated_pattern_observed_support_still_unverified"
        elif interval["label"] == "seated":
            interval["assessment"] = "seated_pattern_not_found"
        else:
            interval["assessment"] = "no_pattern_in_evaluable_samples"
    result.update(
        model=condition["model"],
        source_video_sha256=annotation["source_video_sha256"],
        scope="Reviewed intervals within recorded openings; pose patterns only",
        normal_activity_recognition_validated=False,
        live_activation_supported=False,
    )
    result["assessment_counts"] = dict(Counter(r["assessment"] for r in result["intervals"]))
    return result
