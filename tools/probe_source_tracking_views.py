"""Offline Graviton probe on pre-rendered views; no command/Edge/camera session."""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

from care_vision.cloud.vision import MotionVision
from care_vision.cloud.yolox import YoloXPersonDetector
from care_vision.common.environment import runtime_info
from care_vision.common.image_geometry import ImageGeometry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(plan_path, output, model, baseline):
    environment = runtime_info(3)
    if environment["ec2"].get("architecture") != "arm64" or not environment["cool"]["verified"]:
        raise RuntimeError("real image inference requires Graviton and COOL")
    cv2.setNumThreads(8)
    plan = json.loads(plan_path.read_text())
    root = plan_path.parent
    for filename, expected in plan["frozen_files"].items():
        if sha(root / filename) != expected:
            raise ValueError(f"probe source changed: {filename}")
    spec = importlib.util.spec_from_file_location("baseline_vision", baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    detector = YoloXPersonDetector(model, input_size=640, score_threshold=.2, nms_threshold=.3)

    class CachedDetector:
        backend = detector.backend
        detections = []

        def detect(self, image):
            return self.detections

    cached = CachedDetector()
    visions = [old.MotionVision(person_detector=cached, experimental_weak_association=True),
               MotionVision(person_detector=cached, experimental_weak_association=True)]
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for frame in plan["frames"]:
        jpeg = (root / frame["jpeg"]).read_bytes()
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        cached.detections = detector.detect(image)
        observations = [vision.observe(frame["header"], jpeg) for vision in visions]
        geometry = ImageGeometry.from_header(frame["header"])
        errors = []
        for obs in observations:
            if len(obs["persons"]) != 1:
                errors.append(None)
                continue
            p = obs["persons"][0]
            raw = geometry.to_source(p["bbox"])
            tracked = p.get("tracked_source_bbox")
            if tracked is None:
                tracked = geometry.to_source(p["tracked_bbox"])
            errors.append(max(abs(a - b) for a, b in zip(raw, tracked)))
        rows.append({"capture_seq": frame["header"]["capture_seq"],
                     "view_kind": frame["header"]["view_kind"],
                     "raw_detections": [{"bbox": d.bbox, "confidence": d.confidence}
                                        for d in cached.detections],
                     "baseline": observations[0], "source_tracking": observations[1],
                     "tracked_vs_raw_source_max_component_pixels": errors})
    (output / "observations.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
    )
    summaries = {}
    for mode in ("baseline", "source_tracking"):
        summaries[mode] = {
            "persons_per_frame": [len(r[mode]["persons"]) for r in rows],
            "track_ids": sorted({p["track_id"] for r in rows for p in r[mode]["persons"]}),
            "tracking_status": sorted({r[mode]["person_tracking_status"] for r in rows}),
            "first_confirmed": next((r[mode]["primary_track_id"] for r in rows
                                     if r[mode]["primary_track_id"]), None),
            "last_confirmed": rows[-1][mode]["primary_track_id"],
        }
    deviations = {}
    for kind in ("full_frame", "roi"):
        deviations[kind] = {
            mode: statistics.median(values) if values else None
            for i, mode in enumerate(("baseline", "source_tracking"))
            if (values := [r["tracked_vs_raw_source_max_component_pixels"][i]
                           for r in rows if r["view_kind"] == kind
                           and r["tracked_vs_raw_source_max_component_pixels"][i] is not None])
        }
    report = {
        "scope": "Offline real YOLOX detections on recorded JPEGs and Edge-rendered ROIs",
        "new_model_inferences": len(rows), "frames": len(rows),
        "detections_shared_between_trackers": True, "summaries": summaries,
        "tracked_vs_raw_source_deviation_median_pixels": deviations,
        "deviation_is_accuracy_or_identity_ground_truth": False,
        "runtime": environment, "opencv_threads": cv2.getNumThreads(),
        "plan_sha256": sha(plan_path), "commands_sent": 0, "new_camera": False,
        "pose_actuation_tested": False, "full_demo_passed": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"frames": len(rows), "summaries": summaries, "deviation": deviations}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    probe(args.plan, args.output, args.model, args.baseline)
