"""Score shadow candidates against reviewed intervals, never against spoken cues.

Only recorded episode timelines are scored. Missing samples are unobserved,
not successful rejections. Counts are conditional on the recorded openings;
they are not full-stream precision/recall or live closure measurements.
"""

import math
from collections import Counter

LABELS = {"seated", "confuser", "pending_incident", "unknown"}


def score_intervals(run, annotation):
    if annotation["source_sha256"] != run["source_sha256"]:
        raise ValueError("annotation source hash mismatch")
    if annotation.get("review_scope") != "full_video" or not annotation.get("reviewed_by"):
        raise ValueError("intervals require full video review and reviewer attribution")
    intervals = annotation["intervals"]
    if not intervals:
        raise ValueError("no reviewed intervals")
    previous_end = 0.0
    clean = []
    for interval in intervals:
        start, end = interval["start_seconds"], interval["end_seconds"]
        if (
            not math.isfinite(start) or not math.isfinite(end)
            or start < previous_end or end <= start or interval["label"] not in LABELS
        ):
            raise ValueError("review intervals must be ordered, non-overlapping and finite")
        clean.append(interval)
        previous_end = end
    timeline = run["timeline"]
    if any(
        b["seconds_from_first_observation"] <= a["seconds_from_first_observation"]
        for a, b in zip(timeline, timeline[1:])
    ):
        raise ValueError("shadow timeline must be chronological")
    results = []
    annotated_samples = 0
    for interval in clean:
        start, end = interval["start_seconds"], interval["end_seconds"]
        samples = [r for r in timeline if start <= r["seconds_from_first_observation"] < end]
        candidates = [r for r in samples if r["seated_like_candidate"]]
        annotated_samples += len(samples)
        label = interval["label"]
        if label == "unknown":
            outcome = "indeterminate"
        elif not samples:
            outcome = "not_observed"
        elif candidates:
            outcome = "candidate_found" if label == "seated" else "unexpected_seated_candidate"
        else:
            outcome = "no_candidate_in_observed_samples"
        results.append({
            **interval, "outcome": outcome, "shadow_samples": len(samples),
            "candidate_observations": len(candidates),
            "candidate_episode_ids": sorted({r["episode_id"] for r in candidates}),
            "first_candidate_delay_seconds": (
                candidates[0]["seconds_from_first_observation"] - start if candidates else None
            ),
            "reason_counts": dict(Counter(r["reason"] for r in samples)),
        })
    return {
        "source_sha256": run["source_sha256"],
        "reviewed_by": annotation["reviewed_by"],
        "dataset_split": annotation["dataset_split"],
        "intervals": results,
        "outcome_counts": dict(Counter(r["outcome"] for r in results)),
        "unannotated_shadow_samples": len(timeline) - annotated_samples,
        "live_activation_supported": False,
        "scope": "Reviewed intervals within recorded episodes; candidate evidence only",
    }
