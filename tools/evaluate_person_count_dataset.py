"""Evaluate person presence and count on a manually reviewed multi-video dataset."""

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
from care_vision.evaluation import person_count_metrics


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


def validate_source(capture, video, source_path):
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
    parser.add_argument(
        "--detector",
        choices=["nanodet", "mediapipe", "yolox"],
        default="nanodet",
    )
    parser.add_argument("--detector-input-size", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--phase", type=int, choices=[1, 3], default=1)
    args = parser.parse_args()

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
        source_path = locate_video(video["source_filename"], directories)
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {source_path}")
        try:
            validate_source(capture, video, source_path)
            source_fps = capture.get(cv2.CAP_PROP_FPS)
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
                rows.append(
                    {
                        "video_id": video["video_id"],
                        "source_filename": video["source_filename"],
                        "second": sample["second"],
                        "frame_index": sample["frame_index"],
                        "source_time_seconds": sample["frame_index"] / source_fps,
                        "expected_person_count": sample["expected_person_count"],
                        "predicted_person_count": len(detections),
                        "scene_tags": sample["scene_tags"],
                        "detections": [
                            {"confidence": detection.confidence, "bbox": list(detection.bbox)}
                            for detection in detections
                        ],
                        "latency_ms": latency_ms,
                    }
                )
        finally:
            capture.release()

    expected = [row["expected_person_count"] for row in rows]
    predicted = [row["predicted_person_count"] for row in rows]
    tags = sorted({tag for row in rows for tag in row["scene_tags"]})
    report = {
        "phase": args.phase,
        "runtime": runtime_info(args.phase),
        "dataset_id": dataset["dataset_id"],
        "review": dataset["review"],
        "backend": detector.backend,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "metrics": person_count_metrics(expected, predicted),
        "metrics_by_scene_tag": {
            tag: person_count_metrics(
                [row["expected_person_count"] for row in rows if tag in row["scene_tags"]],
                [row["predicted_person_count"] for row in rows if tag in row["scene_tags"]],
            )
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
