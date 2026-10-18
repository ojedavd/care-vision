"""Offline PersonTracker/ByteTrack comparison over one hashed detection cache.

Reports output continuity only, never IDF1/HOTA/identity accuracy without labels.
The upstream NumPy path is used with explicit compatibility-only source edits.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

COMMIT = "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_numpy_adapter(base):
    source = base / "bytetrack-upstream"
    adapted = base / "bytetrack-numpy"
    adapted.mkdir(mode=0o700)
    (adapted / "yolox/tracker").mkdir(parents=True)
    edits = []
    for path in sorted((source / "yolox/tracker").glob("*.py")):
        original = path.read_text()
        # torch is imported but unused in the Nx5 NumPy branch selected below.
        text = original.replace("import torch\n", "").replace(
            "import torch.nn.functional as F\n", ""
        )
        text = re.sub(r"\bnp\.float\b", "float", text)
        destination = adapted / "yolox/tracker" / path.name
        destination.write_text(text)
        edits.append(
            {"file": path.name, "upstream_sha256": sha(path), "adapted_sha256": sha(destination)}
        )
    for filename in ("yolox/__init__.py", "yolox/tracker/__init__.py"):
        (adapted / filename).write_text("")
    sys.path.insert(0, str(adapted))
    return edits


def load_cache(base):
    plan = json.loads((base / "plan.json").read_text())
    runtime = json.loads((base / "cache-runtime.json").read_text())
    if sha(base / "detections.jsonl") != runtime["cache_sha256"]:
        raise ValueError("cache digest mismatch")
    if sha(base / "plan.json") != runtime["plan_sha256"]:
        raise ValueError("plan digest mismatch")
    rows = [json.loads(line) for line in (base / "detections.jsonl").read_text().splitlines()]
    if len(rows) != len(plan["frames"]):
        raise ValueError("incomplete detection cache")
    result = {}
    for expected, row in zip(plan["frames"], rows, strict=True):
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError("cache frame metadata mismatch")
        if row["observation"]["person_detection_status"] != "ok":
            raise ValueError("invalid detection status")
        key = (row["run"], row["frame_index"])
        if key in result:
            raise ValueError("duplicate detection frame")
        result[key] = row
    return plan, runtime, result


def summarize(timeline):
    ids = {p["track_id"] for row in timeline for p in row["visible_tracks"]}
    with_output = sum(bool(row["visible_tracks"]) for row in timeline)
    return {
        "frames": len(timeline),
        "frames_with_visible_confirmed_track": with_output,
        "frames_without_visible_confirmed_track": len(timeline) - with_output,
        "distinct_visible_track_ids": len(ids),
        "frames_with_multiple_visible_tracks": sum(len(r["visible_tracks"]) > 1 for r in timeline),
        "scope": "output counts; not physical identity accuracy or annotated fragmentation",
    }


def run(base):
    import numpy as np

    from care_vision.cloud.tracking import PersonTracker
    from care_vision.common.environment import runtime_info

    plan, cache_runtime, cache = load_cache(base)
    runtime = runtime_info(3)
    if not runtime["cool"]["verified"] or runtime["ec2"]["architecture"] != "arm64":
        raise RuntimeError("comparison requires verified Graviton/COOL host")
    edits = install_numpy_adapter(base)
    from yolox.tracker.basetrack import BaseTrack
    from yolox.tracker.byte_tracker import BYTETracker

    report = {
        "schema_version": 1,
        "scope": "offline development tracking; no production change",
        "upstream": "https://github.com/FoundationVision/ByteTrack/tree/" + COMMIT,
        "upstream_commit": COMMIT,
        "compatibility_edits": edits,
        "compatibility_scope": (
            "remove unused torch imports; np.float to builtin float; empty package initializers"
        ),
        "runtime": runtime,
        "cache_runtime": cache_runtime,
        "cache_sha256": sha(base / "detections.jsonl"),
        "worker_sha256": sha(Path(__file__)),
        "person_tracker_sha256": sha(base / "care_vision/cloud/tracking.py"),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "lap", "cython-bbox")
        },
        "config": {
            "PersonTracker": {
                "association_confidence": 0.2,
                "new_track_confidence": 0.35,
                "confirmation_hits": 2,
                "max_gap_seconds": 2,
            },
            "ByteTrack": {
                "track_thresh": 0.35,
                "new_track_thresh": 0.45,
                "match_thresh": 0.8,
                "track_buffer": 60,
                "mot20": False,
                "weak_input_floor": 0.2,
            },
        },
        "limitations": [
            "No manual person/bbox labels: no identity accuracy, IDF1, HOTA or true ID switches",
            "ByteTrack advances one Kalman step per frame; timestamp jitter retained in logs",
            "ByteTrack retention uses frame counts; PersonTracker uses real nanoseconds",
            "ByteTrack starts at 0.45 and can activate on frame 1; "
            "PersonTracker starts at 0.35 and requires two hits",
            "Raw YOLOX post-NMS cache floor 0.2 excludes ByteTrack scores between 0.1 and 0.2",
        ],
        "conditions": [],
    }
    with (base / "tracking-timelines.jsonl").open("x") as output:
        for video in plan["runs"]:
            for schedule in video["schedules"]:
                fps = schedule["requested_sample_fps"]
                namespace = f"{video['name']}:{fps}fps"
                person = PersonTracker(association_confidence=0.2)
                BaseTrack._count = 0
                byte = BYTETracker(
                    SimpleNamespace(
                        track_thresh=0.35, match_thresh=0.8, track_buffer=60, mot20=False
                    ),
                    frame_rate=fps,
                )
                timelines = {"PersonTracker": [], "ByteTrack": []}
                inputs = []
                for frame in schedule["frames"]:
                    row = cache[(video["name"], frame["frame_index"])]
                    observation = row["observation"]
                    persons = observation["persons"]
                    inputs.append({"frame_id": row["frame_id"], "persons": persons})
                    ours = person.update(
                        copy.deepcopy(observation), row["captured_mono_ns"], namespace
                    )
                    boxes = np.asarray(
                        [
                            [
                                p["bbox"][0],
                                p["bbox"][1],
                                p["bbox"][0] + p["bbox"][2],
                                p["bbox"][1] + p["bbox"][3],
                                p["confidence"],
                            ]
                            for p in persons
                        ],
                        dtype=np.float64,
                    ).reshape(-1, 5)
                    theirs = byte.update(
                        boxes, (row["height"], row["width"]), (row["height"], row["width"])
                    )
                    outputs = {
                        "PersonTracker": [
                            {"track_id": p["track_id"], "bbox": p["tracked_bbox"]}
                            for p in ours["persons"]
                            if p["track_state"] == "confirmed"
                        ],
                        "ByteTrack": [
                            {"track_id": str(p.track_id), "bbox": p.tlwh.tolist()} for p in theirs
                        ],
                    }
                    for tracker, tracks in outputs.items():
                        item = {
                            "condition": namespace,
                            "tracker": tracker,
                            "frame_id": row["frame_id"],
                            "capture_seq": row["capture_seq"],
                            "captured_mono_ns": row["captured_mono_ns"],
                            "input_detections": len(persons),
                            "visible_tracks": tracks,
                        }
                        timelines[tracker].append(item)
                        output.write(json.dumps(item, allow_nan=False) + "\n")
                report["conditions"].append(
                    {
                        "condition": namespace,
                        "requested_sample_fps": fps,
                        "input_frames": len(inputs),
                        "input_detections": sum(len(r["persons"]) for r in inputs),
                        "weak_input_detections": sum(
                            p["confidence"] < 0.35 for r in inputs for p in r["persons"]
                        ),
                        "shared_input_sha256": hashlib.sha256(
                            json.dumps(inputs, sort_keys=True, allow_nan=False).encode()
                        ).hexdigest(),
                        "byte_max_time_lost_frames": byte.max_time_lost,
                        "outputs": {k: summarize(v) for k, v in timelines.items()},
                    }
                )
                print(f"compared {namespace}: {len(inputs)} shared frames", flush=True)
    report["timeline_sha256"] = sha(base / "tracking-timelines.jsonl")
    (base / "tracking-comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    run(parser.parse_args().base)
