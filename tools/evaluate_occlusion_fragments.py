"""Replay dense pixels on Graviton/COOL; use cached matched sparse detections."""

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from care_vision.cloud.occlusion_fragments import OcclusionFragments
from care_vision.common.environment import runtime_info

base = Path(sys.argv[1])
plan = json.loads((base / "plan.json").read_text())
cv2.setNumThreads(1)
runtime = runtime_info(3)
results = []
for job in plan["jobs"]:
    tracker = OcclusionFragments()
    rows = []
    detections = {r["frame_index"]: r["persons"] for r in job["detections"]}
    debug = base / "debug" / job["id"]
    debug.mkdir(parents=True)
    selected_bytes = 0
    for frame in job["frames"]:
        if job.get("mode") == "fixed2" and frame["frame_index"] not in detections:
            continue
        if (
            job.get("mode") == "adaptive"
            and frame["frame_index"] not in detections
            and not any(not g["ledger"].terminal for g in tracker.groups)
        ):
            continue
        data = Path(frame["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == frame["sha256"]
        selected_bytes += len(data)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        tic = time.perf_counter()
        state, small, foreground = tracker.update(
            image, frame["time"], frame["frame_index"], detections.get(frame["frame_index"])
        )
        state["processing_ms"] = (time.perf_counter() - tic) * 1000
        rows.append(state)
        if frame["frame_index"] in detections and any(
            not g["ledger"].terminal for g in tracker.groups
        ):
            for episode in state["episodes"]:
                box = episode["fragment_box"]
                if box:
                    x, y, w, h = box
                    cv2.rectangle(small, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                small,
                f"{frame['time']:.2f}s",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imwrite(
                str(debug / f"{frame['frame_index']:06d}.jpg"),
                np.hstack([small, cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)]),
            )
    episodes = [g["ledger"].finish() for g in tracker.groups]
    results.append(
        {
            "id": job["id"],
            "rows": rows,
            "episodes": episodes,
            "selected_frames": len(rows),
            "selected_jpeg_bytes": selected_bytes,
        }
    )
    print(job["id"], json.dumps(episodes), flush=True)
(base / "result.json").write_text(json.dumps({"runtime": runtime, "jobs": results}) + "\n")
