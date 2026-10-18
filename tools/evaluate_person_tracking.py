"""Evaluate temporal person tracking on a reviewed video timeline."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

from care_vision.cloud.nanodet import NanoDetPersonDetector
from care_vision.cloud.tracking import PersonTracker
from care_vision.common.environment import runtime_info
from care_vision.evaluation import (
    match_detections,
    positive_segments,
    primary_track_continuity,
)


def divide(numerator, denominator):
    return numerator / denominator if denominator else 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate CARE-Vision person tracking")
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracking-fps", type=float, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--phase", type=int, choices=[1, 3], default=1)
    args = parser.parse_args()
    if not 0 < args.tracking_fps <= 30:
        parser.error("tracking-fps must be in (0, 30]")

    annotations = json.loads(Path(args.annotations).read_text())
    samples_by_frame = {sample["frame_index"]: sample for sample in annotations["samples"]}
    detector = NanoDetPersonDetector(
        args.model,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    tracker = PersonTracker()
    capture = cv2.VideoCapture(args.video)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    last_frame = max(samples_by_frame)
    next_sample_seconds = 0.0
    inferred = 0
    latencies = []
    scored = []
    all_confirmed_ids = set()
    try:
        frame_index = 0
        while frame_index <= last_frame:
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {frame_index}")
            captured_seconds = frame_index / source_fps
            scheduled = captured_seconds + 0.5 / source_fps >= next_sample_seconds
            annotated = frame_index in samples_by_frame
            if scheduled or annotated:
                if scheduled:
                    while next_sample_seconds <= captured_seconds + 0.5 / source_fps:
                        next_sample_seconds += 1 / args.tracking_fps
                started = time.perf_counter_ns()
                detections = detector.detect(image)
                person_observation = {
                    "person_detection_status": "ok",
                    "person_detected": bool(detections),
                    "person_count": len(detections),
                    "persons": [
                        {
                            "detection_id": f"frame-{frame_index}:person:{index}",
                            "label": "person",
                            "confidence": detection.confidence,
                            "bbox": list(detection.bbox),
                        }
                        for index, detection in enumerate(detections)
                    ],
                }
                tracked = tracker.update(
                    person_observation,
                    round(captured_seconds * 1_000_000_000),
                    "evaluation",
                )
                latencies.append((time.perf_counter_ns() - started) / 1_000_000)
                inferred += 1
                all_confirmed_ids.update(
                    item["track_id"]
                    for item in tracked["tracks"]
                    if item["state"] in {"confirmed", "lost"}
                )
                if annotated:
                    sample = samples_by_frame[frame_index]
                    confirmed = [
                        person
                        for person in tracked["persons"]
                        if person["track_state"] == "confirmed"
                    ]
                    _, matched_truth, matched_ious = match_detections(
                        [SimpleNamespace(bbox=person["tracked_bbox"]) for person in confirmed],
                        sample["persons"],
                        0.5,
                    )
                    scored.append(
                        {
                            "second": sample["second"],
                            "frame_index": frame_index,
                            "expected_persons": len(sample["persons"]),
                            "confirmed_persons": len(confirmed),
                            "matched_truth": len(matched_truth),
                            "matched_ious": matched_ious,
                            "track_ids": [person["track_id"] for person in confirmed],
                            "primary_track_id": tracked["primary_track_id"],
                            "tracks": tracked["tracks"],
                        }
                    )
            frame_index += 1
    finally:
        capture.release()

    frame_tp = sum(row["expected_persons"] > 0 and row["confirmed_persons"] > 0 for row in scored)
    frame_fp = sum(row["expected_persons"] == 0 and row["confirmed_persons"] > 0 for row in scored)
    frame_tn = sum(row["expected_persons"] == 0 and row["confirmed_persons"] == 0 for row in scored)
    frame_fn = sum(row["expected_persons"] > 0 and row["confirmed_persons"] == 0 for row in scored)
    precision = divide(frame_tp, frame_tp + frame_fp)
    recall = divide(frame_tp, frame_tp + frame_fn)
    scored_by_second = {row["second"]: row for row in scored}
    continuity = []
    for segment in positive_segments(annotations["samples"]):
        ids = [scored_by_second[item["second"]]["primary_track_id"] for item in segment]
        continuity.append(
            {
                "from_second": segment[0]["second"],
                "to_second": segment[-1]["second"],
                "annotated_frames": len(segment),
                **primary_track_continuity(ids),
            }
        )

    frames_with_multiple_confirmed_tracks = sum(row["confirmed_persons"] > 1 for row in scored)
    confirmed_count_excess_frames = sum(
        row["confirmed_persons"] > row["expected_persons"] for row in scored
    )

    report = {
        "phase": args.phase,
        "runtime": runtime_info(args.phase),
        "dataset_id": annotations["dataset_id"],
        "backend": detector.backend,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "source_fps": source_fps,
        "tracking_fps": args.tracking_fps,
        "frames_inferred": inferred,
        "confirmed_track_ids_total": len(all_confirmed_ids),
        "tracking_diagnostics": {
            "frames_with_multiple_confirmed_tracks": frames_with_multiple_confirmed_tracks,
            "confirmed_count_excess_frames": confirmed_count_excess_frames,
            "maximum_confirmed_persons": max(row["confirmed_persons"] for row in scored),
            "primary_track_switches_total": sum(
                segment["primary_track_switches"] for segment in continuity
            ),
        },
        "confirmed_presence_metrics": {
            "true_positive": frame_tp,
            "false_positive": frame_fp,
            "true_negative": frame_tn,
            "false_negative": frame_fn,
            "precision": precision,
            "recall": recall,
            "f1": divide(2 * precision * recall, precision + recall),
        },
        "continuity": continuity,
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[round(0.95 * (len(latencies) - 1))],
            "maximum": max(latencies),
        },
        "samples": scored,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
