"""AWS-only pose inference on frozen person detections; no live decisions."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from care_vision.cloud.accumulated_descent import AccumulatedDescent
from care_vision.cloud.incident_evidence import IncidentEvidence
from care_vision.cloud.pose_evidence import assess_pose_support
from care_vision.cloud.pose_signals import PoseTrajectoryAnalyzer
from care_vision.cloud.rtmpose import RTMPoseEstimator
from care_vision.cloud.temporal_candidate import SequenceObservation
from care_vision.cloud.trajectory import PostureTrajectoryAnalyzer
from care_vision.common.environment import runtime_info


def evaluate(root, plan):
    for name, expected in plan["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {name}")
    model = Path(plan["pose_model"]["remote_path"])
    if hashlib.sha256(model.read_bytes()).hexdigest() != plan["pose_model"]["sha256"]:
        raise ValueError("pose model hash mismatch")
    cv2.setNumThreads(1)
    runtime = runtime_info(3)  # Refuse to silently run inference outside Graviton + COOL.
    estimator = RTMPoseEstimator(model)
    runs = []
    latencies = []
    for job in plan["jobs"]:
        report = json.loads((root / job["report"]).read_text())
        stream = report["streams"][job["stream_index"]]
        namespace = job["job_id"]
        poses, geometry = {}, PostureTrajectoryAnalyzer()
        descent = AccumulatedDescent()
        ledgers = {name: IncidentEvidence() for name in (
            "bbox_reference", "torso_orientation", "supported_body_orientation",
        )}
        timelines = {name: [] for name in ledgers}
        rows, counts = [], Counter()
        if len(stream["rows"]) != len(job["images"]):
            raise ValueError("image schedule mismatch")
        for row, image_entry in zip(stream["rows"], job["images"]):
            t = row["source_time_seconds"]
            frame = row.get("frame_index", row.get("capture_seq", 0) - 1)
            if frame != image_entry["frame_index"] or t != image_entry["source_time_seconds"]:
                raise ValueError("frame timestamp mismatch")
            data = Path(image_entry["remote_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != image_entry["sha256"]:
                raise ValueError("JPEG hash mismatch")
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (720, 1280):
                raise ValueError("unexpected image")
            captured = row["captured_mono_ns"]
            header = {"width": 1280, "height": 720, "captured_mono_ns": captured}
            present = [p for p in row["persons"] if p["track_state"] == "confirmed"]
            visible_ids = {p["track_id"] for p in present}
            for track, analyzer in poses.items():
                if track not in visible_ids:
                    analyzer.update(None, track, captured, image.shape, namespace)
            added = {s["track_id"]: s for s in descent.update(
                t, frame, {p["track_id"]: p["tracked_bbox"] for p in present},
            )}
            people = []
            for person in present:
                track = person["track_id"]
                bbox = person["tracked_bbox"]
                signals = geometry.update(
                    {"primary_track_id": track, "primary_tracked_bbox": bbox}, header, namespace,
                )
                started = time.perf_counter_ns()
                points = estimator.estimate(image, bbox)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                latencies.append(elapsed)
                analyzer = poses.setdefault(track, PoseTrajectoryAnalyzer())
                pose = analyzer.update(points, track, captured, image.shape, namespace)
                support = assess_pose_support(
                    points, bbox, image.shape,
                    [p["bbox"] for p in row["persons"] if p is not person],
                )
                motion = signals["primary_motion_direction"]
                if added[track]["descent_signal"] and motion in ("stable", "unknown"):
                    motion = "downward"
                people.append({
                    "track_id": track, "bbox": bbox, "original_detection_bbox": person["bbox"],
                    "bbox_orientation": signals["primary_orientation"],
                    "accumulated_bbox_motion": motion, "pose_inference_ms": elapsed,
                    **pose, **support,
                })
                counts["pose_inferences"] += 1
                counts[support["body_support_status"]] += 1
            for name, ledger in ledgers.items():
                observations = []
                for person in people:
                    orientation = person["bbox_orientation"] if name == "bbox_reference" else (
                        person["pose_orientation"] if name == "torso_orientation"
                        or person["body_support_usable"] else None
                    )
                    observations.append(SequenceObservation(
                        t, frame, person["track_id"], orientation,
                        person["accumulated_bbox_motion"], None,
                    ))
                timelines[name].append(ledger.update(t, frame, observations))
            rows.append({
                "frame_index": frame, "source_time_seconds": t,
                "jpeg_sha256": image_entry["sha256"], "persons": people,
                "image_local_path": image_entry["local_path"],
                "person_count": len(people),
            })
        variants = {}
        for name, ledger in ledgers.items():
            episodes = ledger.finish()
            variants[name] = {
                "timeline": timelines[name], "episodes": episodes,
                "summary": {
                    "suspicions": len(episodes),
                    "candidates": sum(e["candidate_frame"] is not None for e in episodes),
                    "recovery_proxies": sum(e["recovery_frame"] is not None for e in episodes),
                    "unresolved": sum(e["outcome"] == "UNRESOLVED" for e in episodes),
                },
            }
        runs.append({
            "job_id": namespace, "video_id": job["video_id"], "model": report["profile"]["label"],
            "schedule": stream["name"], "samples": len(rows), "counts": dict(counts),
            "rows": rows, "variants": variants,
        })
        print(namespace, len(rows), "frames", dict(counts), flush=True)
    return {
        "experiment": "pose-occlusion-v1", "runtime": runtime, "plan": plan, "runs": runs,
        "pose_model_inferences": len(latencies), "new_person_detector_inferences": 0,
        "pose_estimator_ms": {
            "median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95)),
        },
        "camera_opened": False, "alerts_enabled": False, "live_controller_changed": False,
        "limitations": [
            "Score support does not establish visibility, occlusion, identity or joint accuracy.",
            "Crop ambiguity only considers other detections; "
            "an undetected bystander can contaminate.",
            "Orientation ablations keep bbox motion and all temporal thresholds unchanged.",
            "Pose is never inferred without a confirmed detector track; "
            "absences remain unavailable.",
            "Experimental body support requires both legs and torso; "
            "partial views may be rejected.",
            "Development videos and provisional annotations; not independent fall validation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.root, json.loads(args.plan.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
