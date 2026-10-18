"""AWS-only causal ROI replay with fresh NanoDet inference on transmitted crops."""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from care_vision.cloud.nanodet import NanoDetPersonDetector
from care_vision.cloud.occlusion_fragments import OcclusionFragments
from care_vision.cloud.occlusion_roi import incident_roi
from care_vision.cloud.person import observe_people
from care_vision.cloud.tracking import PersonTracker
from care_vision.common.environment import runtime_info

base = Path(sys.argv[1])
plan = json.loads((base / "plan.json").read_text())
cv2.setNumThreads(1)
runtime = runtime_info(3)
assert hashlib.sha256(Path(plan["model"]).read_bytes()).hexdigest() == plan["model_sha256"]
results = []
for mode in plan.get("modes", ["full", "tight", "expanded"]):
    detector = NanoDetPersonDetector(plan["model"])
    people_tracker = PersonTracker()
    tracker = OcclusionFragments()
    roi = None
    rows = []
    commands = []
    total_bytes = 0
    roi_frames = 0
    semantic = set(plan["semantic_frames"])
    debug = base / "debug" / mode
    debug.mkdir(parents=True)
    for f in plan["frames"]:
        active = [g for g in tracker.groups if not g["ledger"].terminal]
        if f["frame_index"] not in semantic and not active:
            continue
        data = Path(f["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == f["sha256"]
        full = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        # Simulated gateway: crop + identical JPEG recompression for every variant.
        effective_roi = None if mode == "hybrid" and f["frame_index"] in semantic else roi
        x, y, w, h = effective_roi or [0, 0, full.shape[1], full.shape[0]]
        ok, encoded = cv2.imencode(
            ".jpg", full[y : y + h, x : x + w], [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        assert ok
        total_bytes += encoded.size
        received = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        # Canvas represents coordinates only. Hidden pixels are zero and masked out.
        canvas = np.zeros_like(full)
        canvas[y : y + h, x : x + w] = received
        people = None
        detection = None
        if f["frame_index"] in semantic:
            detection = observe_people(detector, received, str(f["frame_index"]))
            assert detection["person_detection_status"] == "ok"
            for p in detection["persons"]:
                p["bbox"][0] += x
                p["bbox"][1] += y
            if detection["primary_person_bbox"] is not None:
                detection["primary_person_bbox"] = list(detection["persons"][0]["bbox"])
            detection = people_tracker.update(detection, round(f["time"] * 1e9), "roi-test")
            people = detection["persons"]
        state, small, foreground = tracker.update(
            canvas, f["time"], f["frame_index"], people, visible_roi=effective_roi
        )
        state.update(
            roi=effective_roi,
            encoded_bytes=int(encoded.size),
            detection=detection,
            periodic_context=mode == "hybrid" and roi is not None and effective_roi is None,
        )
        rows.append(state)
        if effective_roi:
            roi_frames += 1
        if f["frame_index"] in semantic and (active or state["episodes"]):
            for ep in state["episodes"]:
                if ep["fragment_box"]:
                    bx, by, bw, bh = ep["fragment_box"]
                    cv2.rectangle(small, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.putText(
                small,
                f"{mode} {f['time']:.2f}s",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imwrite(
                str(debug / f"{f['frame_index']:06d}.jpg"),
                np.hstack([small, cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)]),
            )
        # Command applies only to the NEXT frame. No event-time annotations are inputs.
        pending = [
            g
            for g in tracker.groups
            if not g["ledger"].terminal and g["ledger"].bound_track is None
        ]
        next_roi = None
        if mode != "full" and pending:
            reference = np.array(pending[0]["ledger"].reference_bbox) * (
                [full.shape[1] / 640, full.shape[0] / 360] * 2
            )
            next_roi = incident_roi(reference, full.shape, expanded=mode in {"expanded", "hybrid"})
        if next_roi != roi:
            commands.append(
                {
                    "time": f["time"],
                    "action": "REQUEST_ROI" if next_roi else "REQUEST_FULL_FRAME",
                    "roi": next_roi,
                    "reason": "verify_descent"
                    if next_roi
                    else "reassociation_or_evidence_exhausted",
                }
            )
        roi = next_roi
    episodes = [g["ledger"].finish() for g in tracker.groups]
    results.append(
        {
            "mode": mode,
            "rows": rows,
            "commands": commands,
            "episodes": episodes,
            "selected_frames": len(rows),
            "roi_frames": roi_frames,
            "encoded_bytes": total_bytes,
        }
    )
    print(mode, len(rows), roi_frames, total_bytes, json.dumps(episodes), flush=True)
(base / "result.json").write_text(json.dumps({"runtime": runtime, "results": results}) + "\n")
