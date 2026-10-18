"""Evaluate the NanoDet adapter against a small, reviewed CARE-Vision sample."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import cv2

from care_vision.cloud.nanodet import NanoDetPersonDetector
from care_vision.common.environment import runtime_info
from care_vision.evaluation import match_detections


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def divide(numerator, denominator):
    return numerator / denominator if denominator else 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate CARE-Vision person detection")
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--phase", type=int, choices=[1, 3], default=1)
    args = parser.parse_args()

    video = Path(args.video)
    annotations_path = Path(args.annotations)
    model = Path(args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    annotations = json.loads(annotations_path.read_text())
    actual_video_hash = sha256(video)
    if actual_video_hash != annotations["source_sha256"]:
        parser.error("video SHA-256 does not match annotations")

    detector = NanoDetPersonDetector(
        model,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    capture = cv2.VideoCapture(str(video))
    rows = []
    latencies = []
    true_positive = false_positive = false_negative = 0
    frame_tp = frame_fp = frame_tn = frame_fn = 0
    all_ious = []
    try:
        for sample in annotations["samples"]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_index"])
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {sample['frame_index']}")
            started = time.perf_counter_ns()
            predictions = detector.detect(image)
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000
            latencies.append(latency_ms)
            matched_predictions, matched_truth, matched_ious = match_detections(
                predictions, sample["persons"], args.iou_threshold
            )
            true_positive += len(matched_predictions)
            false_positive += len(predictions) - len(matched_predictions)
            false_negative += len(sample["persons"]) - len(matched_truth)
            all_ious.extend(matched_ious)
            expected_present = bool(sample["persons"])
            detected_present = bool(predictions)
            if expected_present and detected_present:
                frame_tp += 1
            elif expected_present:
                frame_fn += 1
            elif detected_present:
                frame_fp += 1
            else:
                frame_tn += 1
            rows.append(
                {
                    "second": sample["second"],
                    "frame_index": sample["frame_index"],
                    "expected": sample["persons"],
                    "predicted": [
                        {"bbox": list(item.bbox), "confidence": item.confidence}
                        for item in predictions
                    ],
                    "matched_ious": matched_ious,
                    "latency_ms": latency_ms,
                }
            )
    finally:
        capture.release()

    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    frame_precision = divide(frame_tp, frame_tp + frame_fp)
    frame_recall = divide(frame_tp, frame_tp + frame_fn)
    report = {
        "phase": args.phase,
        "runtime": runtime_info(args.phase),
        "dataset_id": annotations["dataset_id"],
        "video_sha256": actual_video_hash,
        "model_sha256": sha256(model),
        "backend": detector.backend,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "iou_threshold": args.iou_threshold,
        "sample_count": len(rows),
        "ground_truth_persons": sum(len(row["expected"]) for row in rows),
        "detection_metrics": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": divide(2 * precision * recall, precision + recall),
            "matched_iou_median": statistics.median(all_ious) if all_ious else None,
        },
        "frame_presence_metrics": {
            "true_positive": frame_tp,
            "false_positive": frame_fp,
            "true_negative": frame_tn,
            "false_negative": frame_fn,
            "precision": frame_precision,
            "recall": frame_recall,
            "f1": divide(2 * frame_precision * frame_recall, frame_precision + frame_recall),
        },
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[round(0.95 * (len(latencies) - 1))],
            "maximum": max(latencies),
        },
        "samples": rows,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
