"""Summarize offline pose responses and provisional side-unknown visual anchors.

No activity classifier, COCO metric or interpolation of visibility annotations.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_errors(points, anchors):
    if points is None:
        return [
            {**a, "closest_family_error_pixels": None, "selected_joint_index": None}
            for a in anchors
        ]
    values = np.asarray(points, dtype=np.float64)
    if values.shape != (17, 3) or not np.all(np.isfinite(values)):
        raise ValueError("expected 17 finite pose points")
    results = []
    for anchor in anchors:
        indices = anchor["acceptable_indices"]
        if not indices or any(i < 0 or i >= 17 for i in indices):
            raise ValueError("invalid anchor indices")
        errors = np.linalg.norm(values[indices, :2] - anchor["xy"], axis=1)
        closest = indices[int(np.argmin(errors))]
        results.append(
            {
                **anchor,
                "closest_family_error_pixels": float(errors.min()),
                "selected_joint_index": closest,
                "selected_joint_response": float(values[closest, 2]),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite results")
    runtime = json.loads((args.base / "pose-runtime.json").read_text())
    plan = json.loads((args.base / "pose-plan.json").read_text())
    if sha(args.base / "poses.jsonl") != runtime["poses_sha256"]:
        raise ValueError("pose cache digest mismatch")
    if sha(args.base / "pose-plan.json") != runtime["plan_sha256"]:
        raise ValueError("pose plan digest mismatch")
    poses = list(map(json.loads, (args.base / "poses.jsonl").read_text().splitlines()))
    if len(poses) != len(plan["frames"]):
        raise ValueError("incomplete pose cache")
    for row, frame in zip(poses, plan["frames"], strict=True):
        if any(row[k] != v for k, v in frame.items()):
            raise ValueError("pose source metadata mismatch")
    annotations = json.loads(args.annotations.read_text())
    lookup = {r["frame_id"]: r for r in poses}
    if len(lookup) != len(poses):
        raise ValueError("duplicate pose frame")
    summaries, reviewed = [], []
    for name in plan["models"]:
        for run in plan["runs"]:
            rows = [r["predictions"][name] for r in poses if r["run"] == run["name"]]
            summaries.append(
                {
                    "model": name,
                    "run": run["name"],
                    "frames": len(rows),
                    "statuses": dict(Counter(r["status"] for r in rows)),
                    "response_supported_body_frames": sum(
                        r.get("support", {}).get("body_support_usable", False) for r in rows
                    ),
                    "response_supported_torso_frames": sum(
                        r.get("support", {}).get("torso_supported", False) for r in rows
                    ),
                    "response_supported_leg_frames": sum(
                        r.get("support", {}).get("legs_supported", False) for r in rows
                    ),
                    "physical_visibility_validated": False,
                }
            )
    errors = {name: [] for name in plan["models"]}
    for annotation in annotations["frames"]:
        if annotation["frame_id"] not in lookup:
            raise ValueError("annotation absent from common cache")
        row = lookup[annotation["frame_id"]]
        if annotation["jpeg_sha256"] != row["jpeg_sha256"]:
            raise ValueError("annotation source hash mismatch")
        predictions = {}
        for name, prediction in row["predictions"].items():
            scored = anchor_errors(prediction["points"], annotation["visible_anchors"])
            errors[name].extend(
                r["closest_family_error_pixels"]
                for r in scored
                if r["closest_family_error_pixels"] is not None
            )
            predictions[name] = {
                "status": prediction["status"],
                "support": prediction.get("support"),
                "anchors": scored,
            }
        reviewed.append(
            {
                "frame_id": annotation["frame_id"],
                "posture_visual": annotation["posture_visual"],
                "predictions": predictions,
            }
        )
    result = {
        "schema_version": 1,
        "poses_sha256": runtime["poses_sha256"],
        "annotation_sha256": sha(args.annotations),
        "annotation_review": annotations["review"],
        "scorer_sha256": sha(Path(__file__)),
        "frames": len(poses),
        "source_exclusions": plan["exclusions"],
        "models": plan["models"],
        "per_run": summaries,
        "reviewed_frames": reviewed,
        "anchor_diagnostics": {
            name: {
                "count": len(values),
                "median_closest_family_error_pixels": float(np.median(values)) if values else None,
                "maximum_closest_family_error_pixels": max(values) if values else None,
                "scope": (
                    "optimistic side-unknown lower bound; provisional anchors, "
                    "not COCO/PCK accuracy"
                ),
            }
            for name, values in errors.items()
        },
        "decision": (
            "keep both offline; response support or knee angle alone cannot close a benign episode"
        ),
        "normal_activity_recognition_validated": False,
        "would_close_episode": False,
        "joint_visibility_accuracy_validated": False,
        "immobility_validated": False,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["anchor_diagnostics"], indent=2))


if __name__ == "__main__":
    main()
