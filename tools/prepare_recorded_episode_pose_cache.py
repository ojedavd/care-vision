"""Prepare exact logged frame/track regions for conditional episode pose replay.

Post-tracker logs are used only as fixed episode inputs, never tracker benchmark inputs.
Reencoded AVI images are not original transmitted JPEGs. No local image inference.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--shared-jpegs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (args.output / "images").mkdir(mode=0o700)
    old = json.loads(args.source_plan.read_text())
    shared = json.loads((args.shared_jpegs / "plan.json").read_text())
    images = {r["frame_id"]: r for r in shared["frames"]}
    plan = {
        "schema_version": 1,
        "region_source": "recorded cloud tracked_bbox",
        "scope": "exact logged frames/regions for recorded episode replay; post-tracker inputs",
        "source_plan_sha256": sha(args.source_plan),
        "frames": [],
        "runs": old["runs"],
    }
    detections = []
    reused = 0
    for run in old["runs"]:
        for filename, digest in run["source_files"].items():
            if sha(root / filename) != digest:
                raise ValueError("recorded source digest mismatch")
        cloud = next(root / f for f in run["source_files"] if f.endswith("cloud.jsonl"))
        video = next(root / f for f in run["source_files"] if f.endswith("demo.avi"))
        capture_file = next(root / f for f in run["source_files"] if f.endswith("captures.jsonl"))
        captures = list(map(json.loads, capture_file.read_text().splitlines()))
        capture_lookup = {c["capture_seq"]: (i, c) for i, c in enumerate(captures)}
        logged = [
            r
            for r in map(json.loads, cloud.read_text().splitlines())
            if r["event"] == "observation"
        ]
        wanted = {}
        for row in logged:
            index, capture = capture_lookup[row["capture_seq"]]
            if capture["captured_mono_ns"] != row["captured_mono_ns"]:
                raise ValueError("logged observation/capture timestamp mismatch")
            if index in wanted:
                raise ValueError("duplicate logged capture")
            wanted[index] = row
        decoder = cv2.VideoCapture(str(video))
        try:
            index = 0
            while wanted:
                ok, image = decoder.read()
                if not ok:
                    raise ValueError("AVI ended before logged frame")
                if index in wanted:
                    row = wanted.pop(index)
                    if image.shape[:2] != (720, 1280) or len(row["persons"]) != 1:
                        raise ValueError(
                            "this conditional experiment requires one person/full-frame"
                        )
                    frame_id = f"{run['name']}:{row['capture_seq']}"
                    filename = f"images/{run['name']}-{row['capture_seq']:06d}.jpg"
                    path = args.output / filename
                    if frame_id in images:
                        previous = images[frame_id]
                        source = args.shared_jpegs / previous["filename"]
                        if sha(source) != previous["jpeg_sha256"]:
                            raise ValueError("shared JPEG digest mismatch")
                        path.write_bytes(source.read_bytes())
                        reused += 1
                    else:
                        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if not ok:
                            raise ValueError("JPEG encode failed")
                        path.write_bytes(encoded.tobytes())
                    frame = {
                        "run": run["name"],
                        "frame_id": frame_id,
                        "frame_index": index,
                        "capture_seq": row["capture_seq"],
                        "captured_mono_ns": row["captured_mono_ns"],
                        "seconds_from_first_observation": (
                            row["captured_mono_ns"] - logged[0]["captured_mono_ns"]
                        )
                        / 1e9,
                        "filename": filename,
                        "jpeg_sha256": sha(path),
                        "width": 1280,
                        "height": 720,
                        "pose_bbox": row["persons"][0]["tracked_bbox"],
                        "logged_observation_id": row["observation_id"],
                    }
                    plan["frames"].append(frame)
                    detections.append(
                        {
                            **frame,
                            "observation": {
                                "person_detection_status": row["person_detection_status"],
                                "persons": row["persons"],
                            },
                        }
                    )
                index += 1
        finally:
            decoder.release()
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (args.output / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in detections))
    result = {
        "cache_sha256": sha(args.output / "detections.jsonl"),
        "plan_sha256": sha(args.output / "plan.json"),
        "frames": len(detections),
        "shared_jpeg_bytes_reused": reused,
        "new_jpeg_encodes": len(detections) - reused,
        "new_detector_inferences": 0,
        "scope": "recorded post-tracker observations; not pre-tracker detection cache",
    }
    (args.output / "cache-runtime.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
