"""Evaluate person boxes on a manually annotated multi-video dataset."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import cv2

from care_vision.cloud.mediapipe_person import MediaPipePersonDetector
from care_vision.cloud.nanodet import NanoDetPersonDetector
from care_vision.cloud.yolox import YoloXPersonDetector
from care_vision.common.environment import runtime_info
from care_vision.evaluation import (
    frame_localization_diagnostics,
    match_detection_pairs,
    person_count_metrics,
    person_detection_metrics,
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


def validate_annotations(video):
    seen_frames = set()
    for sample in video["samples"]:
        frame_index = sample["frame_index"]
        if frame_index in seen_frames:
            raise ValueError(f"duplicate frame_index {frame_index} in {video['video_id']}")
        seen_frames.add(frame_index)
        identities = set()
        for person in sample["persons"]:
            identity = person["identity_id"]
            if identity in identities:
                raise ValueError(f"duplicate identity {identity} at frame {frame_index}")
            identities.add(identity)
            bbox = person["bbox"]
            if len(bbox) != 4 or any(type(value) is not int for value in bbox):
                raise ValueError(f"invalid bbox at frame {frame_index}: {bbox}")
            x, y, width, height = bbox
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > video["width"]
                or y + height > video["height"]
            ):
                raise ValueError(f"out-of-bounds bbox at frame {frame_index}: {bbox}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-dir", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--detector", choices=["nanodet", "mediapipe", "yolox"], default="nanodet")
    parser.add_argument("--detector-input-size", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--phase", type=int, choices=[1, 3], default=1)
    args = parser.parse_args()
    if not 0 < args.iou_threshold <= 1:
        parser.error("iou-threshold must be in (0, 1]")

    dataset = json.loads(Path(args.dataset).read_text())
    directories = [Path(item) for item in args.video_dir]
    detector_class = {
        "nanodet": NanoDetPersonDetector,
        "mediapipe": MediaPipePersonDetector,
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
    rows = []
    latencies = []
    for video in dataset["videos"]:
        validate_annotations(video)
        source_path = locate_video(video["source_filename"], directories)
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {source_path}")
        try:
            validate_video(capture, video, source_path)
            for sample in video["samples"]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_index"])
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"could not decode {video['source_filename']} frame {sample['frame_index']}"
                    )
                started = time.perf_counter_ns()
                detections = detector.detect(image)
                latency_ms = (time.perf_counter_ns() - started) / 1_000_000
                latencies.append(latency_ms)
                matches = match_detection_pairs(detections, sample["persons"], args.iou_threshold)
                rows.append(
                    {
                        "video_id": video["video_id"],
                        "source_filename": video["source_filename"],
                        "second": sample["second"],
                        "frame_index": sample["frame_index"],
                        "scene_tags": sample["scene_tags"],
                        "expected_person_count": len(sample["persons"]),
                        "predicted_person_count": len(detections),
                        "expected": sample["persons"],
                        "predicted": [
                            {"bbox": list(item.bbox), "confidence": item.confidence}
                            for item in detections
                        ],
                        "matches": [
                            {
                                "prediction_index": prediction_index,
                                "truth_index": truth_index,
                                "identity_id": sample["persons"][truth_index]["identity_id"],
                                "iou": iou,
                            }
                            for prediction_index, truth_index, iou in matches
                        ],
                        "matched_ious": [iou for _, _, iou in matches],
                        "latency_ms": latency_ms,
                    }
                )
        finally:
            capture.release()

    tags = sorted({tag for row in rows for tag in row["scene_tags"]})
    report = {
        "phase": args.phase,
        "runtime": runtime_info(args.phase),
        "dataset_id": dataset["dataset_id"],
        "review": dataset["review"],
        "backend": detector.backend,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "iou_threshold": args.iou_threshold,
        "sample_count": len(rows),
        "ground_truth_persons": sum(row["expected_person_count"] for row in rows),
        "metrics": person_detection_metrics(rows),
        "frame_metrics": person_count_metrics(
            [row["expected_person_count"] for row in rows],
            [row["predicted_person_count"] for row in rows],
        ),
        "localization_diagnostics": frame_localization_diagnostics(rows),
        "metrics_by_scene_tag": {
            tag: person_detection_metrics([row for row in rows if tag in row["scene_tags"]])
            for tag in tags
        },
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[round(0.95 * (len(latencies) - 1))],
            "maximum": max(latencies),
        },
        "samples": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
