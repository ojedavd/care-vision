"""Collect dense raw AWS detections for a separate multi-person association check."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from care_vision.cloud.yolox import YoloXPersonDetector
from care_vision.common.environment import runtime_info
from care_vision.edge.buffer import encode
from care_vision.temporal_evaluation import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    dataset = json.loads(args.dataset.read_text())
    video = next(v for v in dataset["videos"] if v["source_filename"] == args.video.name)
    if file_sha256(args.video) != video["sha256"]:
        raise ValueError("source hash mismatch")
    runtime = runtime_info(3)
    detector = YoloXPersonDetector(
        args.model, score_threshold=0.4, nms_threshold=0.3, input_size=416
    )
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError("cannot open video")
    for key, prop in (
        ("width", cv2.CAP_PROP_FRAME_WIDTH),
        ("height", cv2.CAP_PROP_FRAME_HEIGHT),
        ("frame_count", cv2.CAP_PROP_FRAME_COUNT),
    ):
        if round(cap.get(prop)) != video[key]:
            raise ValueError("video metadata mismatch")
    fps = cap.get(cv2.CAP_PROP_FPS)
    annotated = {s["frame_index"] for s in video["samples"]}
    # Nearest source frame to each 20 FPS tick, plus every annotated source frame.
    selected = {
        round(t * fps / 20) for t in range(round(max(annotated) / fps * 20) + 1)
    } | annotated
    rows, latencies = [], []
    try:
        for index in range(max(annotated) + 1):
            ok, image = cap.read()
            if not ok:
                raise ValueError("incomplete video")
            if index not in selected:
                continue
            payload = encode(image, 1280, 720)
            frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            started = time.perf_counter()
            detections = detector.detect(frame)
            latencies.append((time.perf_counter() - started) * 1000)
            rows.append(
                {
                    "frame_id": f"{video['video_id']}:{index}",
                    "capture_seq": index + 1,
                    "captured_mono_ns": round(index / fps * 1e9),
                    "available": True,
                    "persons": [
                        {"bbox": list(d.bbox), "confidence": d.confidence} for d in detections
                    ],
                }
            )
    finally:
        cap.release()
    report = {
        "video_id": video["video_id"],
        "source_sha256": video["sha256"],
        "dataset_sha256": file_sha256(args.dataset),
        "model_sha256": file_sha256(args.model),
        "runtime": runtime,
        "width": 1280,
        "height": 720,
        "rows": rows,
        "configuration": {
            "sample_fps": 20,
            "score_threshold": 0.4,
            "nms_threshold": 0.3,
            "detector_input_size": 416,
            "jpeg_quality": 80,
        },
        "inference_ms": {
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "scope": "Offline full-frame AWS inference; no ROI/live transport or incident decisions.",
    }
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"frames_inferred": len(rows), "inference_ms": report["inference_ms"]}))


if __name__ == "__main__":
    main()
