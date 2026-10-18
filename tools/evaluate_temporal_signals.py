"""Replay source videos through the cloud detector/tracker/trajectory at fixed rates."""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from care_vision.cloud.person import observe_people
from care_vision.cloud.pose_signals import PoseTrajectoryAnalyzer, pose_scoring_row
from care_vision.cloud.rtmpose import RTMPoseEstimator
from care_vision.cloud.tracking import PersonTracker
from care_vision.cloud.trajectory import PostureTrajectoryAnalyzer
from care_vision.cloud.yolox import YoloXPersonDetector
from care_vision.common.environment import runtime_info
from care_vision.temporal_evaluation import (
    event_signal_timing,
    file_sha256,
    phase_at,
    sample_indices,
    signal_metrics,
    summarize,
    validate_dataset,
    validate_evidence_manifest,
)


def observe_frame(
    image,
    video,
    index,
    header,
    detector,
    tracker,
    trajectory,
    pose_estimator=None,
    pose_analyzer=None,
):
    namespace = video["video_id"]
    started = time.perf_counter_ns()
    person = observe_people(detector, image, f"{namespace}:{index}")
    if person["person_detection_status"] != "ok":
        raise RuntimeError(f"detector failed: {person}")
    person = tracker.update(person, header["captured_mono_ns"], namespace)
    signals = trajectory.update(person, header, namespace)
    result = {
        "frame_index": index,
        "source_time_seconds": index / video["fps"],
        "phase": phase_at(video, index),
        "person_count": person["person_count"],
        "confirmed_person_count": person["confirmed_person_count"],
        "primary_track_id": person["primary_track_id"],
        **signals,
        "processing_ms": (time.perf_counter_ns() - started) / 1e6,
    }
    if pose_estimator is not None:
        pose_started = time.perf_counter_ns()
        bbox = person["primary_tracked_bbox"]
        points = pose_estimator.estimate(image, bbox) if bbox is not None else None
        result.update(
            pose_analyzer.update(
                points,
                person["primary_track_id"],
                header["captured_mono_ns"],
                image.shape,
                namespace,
            )
        )
        result["pose_inference_performed"] = bbox is not None
        result["pose_processing_ms"] = (time.perf_counter_ns() - pose_started) / 1e6
        result["combined_processing_ms"] = result["processing_ms"] + result["pose_processing_ms"]
    return result


def evaluate_evidence(video, directory, manifest, detector, sample_fps, pose_estimator=None):
    frames = {
        frame["frame_index"]: frame
        for frame in manifest["frames"]
        if frame["video_id"] == video["video_id"]
    }
    tracker, trajectory = PersonTracker(), PostureTrajectoryAnalyzer()
    pose_analyzer = PoseTrajectoryAnalyzer() if pose_estimator else None
    rows = []
    for index in sample_indices(video["frame_count"], video["fps"], sample_fps):
        frame = frames[index]
        path = directory / frame["filename"]
        if file_sha256(path) != frame["sha256"]:
            raise ValueError(f"JPEG evidence hash mismatch: {frame['filename']}")
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (manifest["height"], manifest["width"]):
            raise ValueError("JPEG evidence dimensions mismatch")
        header = {
            "width": manifest["width"],
            "height": manifest["height"],
            "source_width": manifest["width"],
            "source_height": manifest["height"],
            "view_kind": "full_frame",
            "captured_mono_ns": round(index / video["fps"] * 1e9),
        }
        rows.append(
            observe_frame(
                image,
                video,
                index,
                header,
                detector,
                tracker,
                trajectory,
                pose_estimator,
                pose_analyzer,
            )
        )
    return rows


def pose_report(rows, video=None):
    projected = [pose_scoring_row(row) for row in rows]
    straightening = [pose_scoring_row(row, straightening=True) for row in rows]
    called = [row for row in rows if row["pose_inference_performed"]]

    def latency(values):
        values = sorted(values)
        return (
            {
                "median": statistics.median(values),
                "p95": values[round(0.95 * (len(values) - 1))],
                "maximum": max(values),
            }
            if values
            else None
        )

    report = {
        "metrics": {
            signal: signal_metrics(projected, signal)
            for signal in ("downward", "horizontal", "upward")
        },
        "straightening_metrics": signal_metrics(straightening, "upward"),
        "inference_calls": len(called),
        "status_counts": {
            status: sum(row["pose_status"] == status for row in rows)
            for status in sorted({row["pose_status"] for row in rows})
        },
        "pose_processing_ms_when_called": latency([row["pose_processing_ms"] for row in called]),
        "combined_processing_ms_when_called": latency(
            [row["combined_processing_ms"] for row in called]
        ),
        "combined_processing_ms_all_frames": latency(
            [row["combined_processing_ms"] for row in rows]
        ),
    }
    if video is not None:
        report["event_signal_timing"] = [
            r for r in event_signal_timing(video, projected) if r["signal"] != "near_floor"
        ]
        report["straightening_event_timing"] = [
            {**r, "signal": "straightening"}
            for r in event_signal_timing(video, straightening)
            if r["signal"] == "upward"
        ]
    return report


def evaluate_video(video, path, detector, sample_fps):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open {path.name}")
        actual = {
            "sha256": file_sha256(path),
            "frame_count": round(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        if any(actual[key] != video[key] for key in actual):
            raise ValueError(f"source integrity or orientation mismatch: {path.name}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if abs(fps - video["fps"]) > 0.001:
            raise ValueError(f"source FPS mismatch: {path.name}")
        indices = set(sample_indices(video["frame_count"], fps, sample_fps))
        tracker, trajectory = PersonTracker(), PostureTrajectoryAnalyzer()
        rows = []
        # Decode sequentially; do not introduce seek/keyframe ambiguity.
        for index in range(video["frame_count"]):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"decode failed at {path.name}:{index}")
            if index not in indices:
                continue
            if image.shape[:2] != (video["height"], video["width"]):
                raise ValueError("decoded source shape mismatch")
            header = {
                "width": video["width"],
                "height": video["height"],
                "source_width": video["width"],
                "source_height": video["height"],
                "view_kind": "full_frame",
                "captured_mono_ns": round(index / fps * 1e9),
            }
            rows.append(observe_frame(image, video, index, header, detector, tracker, trajectory))
        return rows
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video-dir", type=Path, action="append")
    inputs.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, nargs="+", default=[2, 20])
    parser.add_argument("--phase", type=int, choices=[1, 3], default=1)
    args = parser.parse_args()
    if args.pose_model and not args.evidence_dir:
        parser.error("pose comparison requires identical --evidence-dir JPEG inputs")
    if args.output.exists():
        parser.error("output exists; use a new path to preserve evidence")
    dataset = json.loads(args.dataset.read_text())
    validate_dataset(dataset)
    manifest = None
    if args.evidence_dir:
        manifest = json.loads((args.evidence_dir / "manifest.json").read_text())
        validate_evidence_manifest(manifest, dataset, file_sha256(args.dataset), args.sample_fps)
    if len(set(args.sample_fps)) != len(args.sample_fps):
        parser.error("duplicate sample FPS")
    for rate in args.sample_fps:
        for video in dataset["videos"]:
            sample_indices(video["frame_count"], video["fps"], rate)
    runtime = runtime_info(args.phase)
    detector = YoloXPersonDetector(
        args.model, score_threshold=0.4, nms_threshold=0.3, input_size=416
    )
    pose_estimator = RTMPoseEstimator(args.pose_model) if args.pose_model else None
    runs = []
    for rate in args.sample_fps:
        videos = []
        for video in dataset["videos"]:
            if manifest is not None:
                rows = evaluate_evidence(
                    video, args.evidence_dir, manifest, detector, rate, pose_estimator
                )
            else:
                paths = {d.resolve() / video["source_filename"] for d in args.video_dir}
                paths = [path for path in paths if path.is_file()]
                if len(paths) != 1:
                    raise ValueError(f"expected exactly one source: {video['source_filename']}")
                rows = evaluate_video(video, paths[0], detector, rate)
            videos.append(
                {
                    "video_id": video["video_id"],
                    "source_sha256": video["sha256"],
                    "frames_inferred": len(rows),
                    "metrics": summarize(rows),
                    "event_signal_timing": event_signal_timing(video, rows),
                    "samples": rows,
                    **({"pose": pose_report(rows, video)} if pose_estimator else {}),
                }
            )
            print(f"{rate:g} FPS: {video['video_id']}: {len(rows)} observations", flush=True)
        all_rows = [row for video in videos for row in video["samples"]]
        latencies = sorted(row["processing_ms"] for row in all_rows)
        runs.append(
            {
                "sample_fps": rate,
                "frames_inferred": len(all_rows),
                "metrics_micro": summarize(all_rows),
                "videos": videos,
                "processing_ms": {
                    "median": statistics.median(latencies),
                    "p95": latencies[round(0.95 * (len(latencies) - 1))],
                    "maximum": max(latencies),
                },
                **({"pose": pose_report(all_rows)} if pose_estimator else {}),
            }
        )
    report = {
        "evaluation": "phase7d-offline-pose-comparison"
        if pose_estimator
        else "phase7c-offline-temporal-signals",
        "phase": args.phase,
        "runtime": runtime,
        "dataset_id": dataset["dataset_id"],
        "dataset_sha256": file_sha256(args.dataset),
        "review": dataset["review"],
        "model_sha256": file_sha256(args.model),
        "backend": detector.backend,
        "evidence_manifest_sha256": file_sha256(args.evidence_dir / "manifest.json")
        if manifest is not None
        else None,
        "configuration": {
            "score_threshold": 0.4,
            "nms_threshold": 0.3,
            "input_size": 416,
            "representation": "Edge encode, fixed 1280x720 full-frame JPEG, no adaptive policy"
            if manifest is not None
            else "native full-frame, no transport or JPEG reencoding",
            "tracker": "production defaults, reset per video and sampling rate",
            "signals": "unchanged Phase 7B experimental thresholds",
        },
        "limitations": [
            "Offline signal study, not closed-loop or end-to-end latency evaluation.",
            "Provisional assistant annotations; human review pending.",
            "One positive event; correlated frames are not independent trials.",
            "Missing signals are separate from true negatives.",
            "Primary person only; no per-person fall claim in multi-person scenes.",
        ],
        "fall_classification_enabled": False,
        "runs": runs,
    }
    if pose_estimator:
        report["pose_configuration"] = {
            "model_sha256": pose_estimator.model_sha256,
            "backend": pose_estimator.backend,
            "input_size": [192, 256],
            "crop": "confirmed primary tracked bbox, padding 1.25",
            "scope": "offline cloud experiment only; not connected to agent controller",
            "keypoint_min_response": PoseTrajectoryAnalyzer.keypoint_threshold,
            "horizontal_torso_degrees": PoseTrajectoryAnalyzer.horizontal_degrees,
            "vertical_torso_degrees": PoseTrajectoryAnalyzer.vertical_degrees,
            "vertical_velocity_threshold": PoseTrajectoryAnalyzer.velocity_threshold,
            "straightening_degrees_per_s": PoseTrajectoryAnalyzer.straightening_degrees_per_s,
            "threshold_status": "experimental, declared before evaluation, not calibrated",
        }
        report["limitations"].extend(
            [
                "No independently annotated keypoints; cannot measure pose localization accuracy.",
                "Torso angle is a 2D posture proxy, not proof of floor contact or a fall.",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
