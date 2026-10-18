"""Offline scoring of posture proxies against provisional temporal annotations.

This module never decides incidents. Unknown evidence is reported explicitly;
it is not silently converted into a correct negative.
"""

import hashlib
import math
from pathlib import Path

PHASES = {"normal", "uncertain", "descending", "floor_rest", "recovering"}
SIGNALS = {
    "downward": ("descending", "primary_motion_direction", "downward"),
    "horizontal": ("floor_rest", "primary_orientation", "horizontal"),
    "near_floor": ("floor_rest", "primary_near_floor", True),
    "upward": ("recovering", "primary_motion_direction", "upward"),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(dataset):
    if dataset.get("annotation_schema_version") != 1 or not dataset.get("dataset_id"):
        raise ValueError("unsupported temporal annotation schema")
    if not dataset.get("review", {}).get("status") or not dataset.get("videos"):
        raise ValueError("review provenance and videos are required")
    video_ids = set()
    filenames = set()
    for video in dataset["videos"]:
        video_id, filename = video["video_id"], video["source_filename"]
        if video_id in video_ids or filename in filenames or Path(filename).name != filename:
            raise ValueError("duplicate video or invalid source filename")
        video_ids.add(video_id)
        filenames.add(filename)
        for key in ("frame_count", "width", "height"):
            if type(video[key]) is not int or video[key] <= 0:
                raise ValueError(f"invalid {key}")
        if not math.isfinite(video["fps"]) or video["fps"] <= 0:
            raise ValueError("invalid source FPS")
        digest = video["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid source hash")
        end = 0
        for segment in video["segments"]:
            start, stop = segment["start_frame"], segment["end_frame_exclusive"]
            if (
                type(start) is not int
                or type(stop) is not int
                or start != end
                or not start < stop <= video["frame_count"]
                or segment["phase"] not in PHASES
            ):
                raise ValueError("segments must cover the video without gaps or overlaps")
            end = stop
        if end != video["frame_count"]:
            raise ValueError("segments must cover the entire video")
        event_ids = set()
        for event in video["events"]:
            if event["event_id"] in event_ids:
                raise ValueError("duplicate event ID")
            event_ids.add(event["event_id"])
            previous = -1
            for key in (
                "descent_onset_frames",
                "floor_contact_frames",
                "recovery_onset_frames",
                "upright_again_frames",
            ):
                bounds = event[key]
                if (
                    len(bounds) != 2
                    or any(type(x) is not int for x in bounds)
                    or not 0 <= bounds[0] <= bounds[1] < video["frame_count"]
                    or bounds[0] < previous
                ):
                    raise ValueError("invalid or unordered event boundary")
                previous = bounds[1]


def sample_indices(frame_count, source_fps, sample_fps):
    """Nearest source frame to each fixed-rate tick, without annotation-driven inserts."""
    if not math.isfinite(sample_fps) or not 0 < sample_fps <= source_fps:
        raise ValueError("sample FPS must be positive and no greater than source FPS")
    return sorted(
        {
            round(tick * source_fps / sample_fps)
            for tick in range(math.ceil(frame_count / source_fps * sample_fps))
            if round(tick * source_fps / sample_fps) < frame_count
        }
    )


def phase_at(video, frame_index):
    for segment in video["segments"]:
        if segment["start_frame"] <= frame_index < segment["end_frame_exclusive"]:
            return segment["phase"]
    raise ValueError("frame outside annotated video")


def signal_value(row, signal):
    _, field, expected = SIGNALS[signal]
    value = row.get(field)
    if value is None or value == "unknown":
        return None
    return value == expected


def expected_signal(phase, signal):
    if phase == "uncertain" or (signal in {"horizontal", "near_floor"} and phase == "recovering"):
        return None
    return phase == SIGNALS[signal][0]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def signal_metrics(rows, signal):
    counts = dict(
        tp=0, fp=0, tn=0, fn=0, unavailable_positive=0, unavailable_negative=0, excluded=0
    )
    for row in rows:
        expected = expected_signal(row["phase"], signal)
        predicted = signal_value(row, signal)
        if expected is None:
            counts["excluded"] += 1
        elif predicted is None:
            counts["unavailable_positive" if expected else "unavailable_negative"] += 1
        else:
            counts[
                ("tp" if predicted else "fn") if expected else ("fp" if predicted else "tn")
            ] += 1
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    positives = tp + fn + counts["unavailable_positive"]
    negatives = fp + tn + counts["unavailable_negative"]
    return {
        **counts,
        "positive_samples": positives,
        "negative_samples": negatives,
        "availability": ratio(tp + fp + tn + fn, positives + negatives),
        "precision_available": ratio(tp, tp + fp),
        "recall_available": ratio(tp, tp + fn),
        "positive_coverage_including_missing": ratio(tp, positives),
        "false_positive_rate_available": ratio(fp, fp + tn),
    }


def event_signal_timing(video, rows):
    """First positive proxy in a definite annotated phase; not incident latency."""
    result = []
    boundaries = {
        "downward": ("descent_onset_frames", "floor_contact_frames"),
        "horizontal": ("floor_contact_frames", "recovery_onset_frames"),
        "near_floor": ("floor_contact_frames", "recovery_onset_frames"),
        "upward": ("recovery_onset_frames", "upright_again_frames"),
    }
    for event in video["events"]:
        for signal, (start_key, stop_key) in boundaries.items():
            onset_lo, onset_hi = event[start_key]
            stop = event[stop_key][0]
            candidates = [
                row
                for row in rows
                if onset_hi <= row["frame_index"] < stop
                and expected_signal(row["phase"], signal) is True
            ]
            first = next((row for row in candidates if signal_value(row, signal) is True), None)
            result.append(
                {
                    "event_id": event["event_id"],
                    "signal": signal,
                    "phase_samples": len(candidates),
                    "available_samples": sum(
                        signal_value(row, signal) is not None for row in candidates
                    ),
                    "observed_in_phase": first is not None,
                    "first_positive_frame": first["frame_index"] if first else None,
                    "delay_from_onset_bracket_seconds": [
                        (first["frame_index"] - onset_hi) / video["fps"],
                        (first["frame_index"] - onset_lo) / video["fps"],
                    ]
                    if first
                    else None,
                }
            )
    return result


def summarize(rows):
    return {signal: signal_metrics(rows, signal) for signal in SIGNALS}


def validate_evidence_manifest(manifest, dataset, dataset_sha256, rates):
    """Bind portable JPEGs to source indices without changing the sampling timeline."""
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset_sha256") != dataset_sha256
        or (manifest.get("width"), manifest.get("height")) != (1280, 720)
    ):
        raise ValueError("invalid evidence manifest or dataset mismatch")
    sources = {v["video_id"]: v for v in dataset["videos"]}
    if manifest.get("source_hashes") != {k: v["sha256"] for k, v in sources.items()}:
        raise ValueError("evidence source hashes mismatch")
    seen, names = set(), set()
    for frame in manifest["frames"]:
        key = (frame["video_id"], frame["frame_index"])
        name = frame["filename"]
        if (
            key in seen
            or name in names
            or Path(name).name != name
            or not name.endswith(".jpg")
            or key[0] not in sources
            or type(key[1]) is not int
            or not 0 <= key[1] < sources[key[0]]["frame_count"]
            or len(frame["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in frame["sha256"])
        ):
            raise ValueError("invalid or duplicate JPEG evidence frame")
        seen.add(key)
        names.add(name)
    for video_id, video in sources.items():
        for rate in rates:
            required = {
                (video_id, i) for i in sample_indices(video["frame_count"], video["fps"], rate)
            }
            if not required <= seen:
                raise ValueError("missing scheduled JPEG evidence")
