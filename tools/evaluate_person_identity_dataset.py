"""Evaluate detector and tracker identity continuity on a multi-video dataset."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

from care_vision.cloud.nanodet import NanoDetPersonDetector
from care_vision.cloud.tracking import PersonTracker
from care_vision.cloud.yolox import YoloXPersonDetector
from care_vision.common.environment import runtime_info
from care_vision.evaluation import (
    identity_tracking_metrics,
    match_detection_pairs,
    multi_object_tracking_metrics,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_video(filename, directories):
    matches = [
        directory / filename for directory in directories if (directory / filename).is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one source for {filename}, found {len(matches)}")
    return matches[0]


def validate_video(capture, video, source_path):
    actual = {
        "sha256": file_sha256(source_path),
        "frame_count": round(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    for key, value in actual.items():
        if value != video[key]:
            raise ValueError(f"{video['source_filename']} {key} mismatch: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-dir", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector", choices=["nanodet", "yolox"], default="nanodet")
    parser.add_argument("--detector-input-size", type=int)
    parser.add_argument("--tracking-fps", type=float, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--tracker-iou-threshold", type=float, default=0.2)
    parser.add_argument("--confirmation-hits", type=int, default=2)
    parser.add_argument("--max-gap-seconds", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--reacquire-seconds", type=float, default=0.25)
    parser.add_argument("--phase", type=int, choices=[1, 3], default=1)
    args = parser.parse_args()
    if not 0 < args.tracking_fps <= 30:
        parser.error("tracking-fps must be in (0, 30]")
    if not 0 < args.iou_threshold <= 1:
        parser.error("iou-threshold must be in (0, 1]")

    dataset = json.loads(Path(args.dataset).read_text())
    directories = [Path(item) for item in args.video_dir]
    detector_class = {
        "nanodet": NanoDetPersonDetector,
        "yolox": YoloXPersonDetector,
    }[args.detector]
    detector_options = {
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
    }
    if args.detector_input_size is not None:
        if args.detector != "yolox":
            parser.error("--detector-input-size is only supported by yolox")
        detector_options["input_size"] = args.detector_input_size
    detector = detector_class(args.model, **detector_options)
    assignments = []
    samples = []
    latencies = []
    frames_inferred = 0
    all_track_ids = set()
    for video in dataset["videos"]:
        source_path = locate_video(video["source_filename"], directories)
        samples_by_frame = {sample["frame_index"]: sample for sample in video["samples"]}
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {source_path}")
        validate_video(capture, video, source_path)
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        tracker = PersonTracker(
            iou_threshold=args.tracker_iou_threshold,
            confirmation_hits=args.confirmation_hits,
            max_gap_seconds=args.max_gap_seconds,
            alpha=args.alpha,
            reacquire_seconds=args.reacquire_seconds,
        )
        next_sample_seconds = 0.0
        last_frame = max(samples_by_frame)
        try:
            for frame_index in range(last_frame + 1):
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"could not decode {video['source_filename']} frame {frame_index}"
                    )
                captured_seconds = frame_index / source_fps
                scheduled = captured_seconds + 0.5 / source_fps >= next_sample_seconds
                annotated = frame_index in samples_by_frame
                if not scheduled and not annotated:
                    continue
                if scheduled:
                    while next_sample_seconds <= captured_seconds + 0.5 / source_fps:
                        next_sample_seconds += 1 / args.tracking_fps
                started = time.perf_counter_ns()
                detections = detector.detect(image)
                observation = {
                    "person_detection_status": "ok",
                    "person_detected": bool(detections),
                    "person_count": len(detections),
                    "persons": [
                        {
                            "detection_id": f"{video['video_id']}:{frame_index}:person:{index}",
                            "label": "person",
                            "confidence": detection.confidence,
                            "bbox": list(detection.bbox),
                        }
                        for index, detection in enumerate(detections)
                    ],
                }
                tracked = tracker.update(
                    observation,
                    round(captured_seconds * 1_000_000_000),
                    video["video_id"],
                )
                latencies.append((time.perf_counter_ns() - started) / 1_000_000)
                frames_inferred += 1
                all_track_ids.update(item["track_id"] for item in tracked["tracks"])
                if not annotated:
                    continue
                sample = samples_by_frame[frame_index]
                confirmed = [
                    person for person in tracked["persons"] if person["track_state"] == "confirmed"
                ]
                matches = match_detection_pairs(
                    [SimpleNamespace(bbox=person["tracked_bbox"]) for person in confirmed],
                    sample["persons"],
                    args.iou_threshold,
                )
                truth_to_track = {
                    truth_index: confirmed[prediction_index]["track_id"]
                    for prediction_index, truth_index, _ in matches
                }
                for truth_index, person in enumerate(sample["persons"]):
                    assignments.append(
                        {
                            "video_id": video["video_id"],
                            "frame_index": frame_index,
                            "second": sample["second"],
                            "identity_id": person["identity_id"],
                            "track_id": truth_to_track.get(truth_index),
                        }
                    )
                samples.append(
                    {
                        "video_id": video["video_id"],
                        "second": sample["second"],
                        "frame_index": frame_index,
                        "expected_persons": len(sample["persons"]),
                        "confirmed_persons": len(confirmed),
                        "matches": [
                            {
                                "identity_id": sample["persons"][truth_index]["identity_id"],
                                "track_id": confirmed[prediction_index]["track_id"],
                                "iou": iou,
                            }
                            for prediction_index, truth_index, iou in matches
                        ],
                    }
                )
        finally:
            capture.release()

    report = {
        "phase": args.phase,
        "runtime": runtime_info(args.phase),
        "dataset_id": dataset["dataset_id"],
        "backend": detector.backend,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "iou_threshold": args.iou_threshold,
        "tracking_fps": args.tracking_fps,
        "tracker_config": {
            "iou_threshold": args.tracker_iou_threshold,
            "confirmation_hits": args.confirmation_hits,
            "max_gap_seconds": args.max_gap_seconds,
            "alpha": args.alpha,
            "reacquire_seconds": args.reacquire_seconds,
        },
        "frames_inferred": frames_inferred,
        "track_ids_created": len(all_track_ids),
        "metrics": identity_tracking_metrics(assignments),
        "mot_metrics": multi_object_tracking_metrics(samples),
        "tracking_diagnostics": {
            "annotated_frames": len(samples),
            "negative_frames": sum(row["expected_persons"] == 0 for row in samples),
            "confirmed_on_negative_frames": sum(
                row["expected_persons"] == 0 and row["confirmed_persons"] > 0 for row in samples
            ),
            "confirmed_count_excess_frames": sum(
                row["confirmed_persons"] > row["expected_persons"] for row in samples
            ),
            "maximum_confirmed_persons": max(row["confirmed_persons"] for row in samples),
        },
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[round(0.95 * (len(latencies) - 1))],
            "maximum": max(latencies),
        },
        "assignments": assignments,
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"assignments", "samples"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
