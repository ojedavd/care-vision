"""Score an anchored geometric segment only at pre-existing identity annotations."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from care_vision.cloud.context_association import anchored_segment
from care_vision.common.cool import validate_evidence
from care_vision.evaluation import match_detection_pairs
from care_vision.temporal_evaluation import file_sha256


def evaluate(detections, video):
    rows = detections["rows"]
    by_index = {r["capture_seq"] - 1: r for r in rows}
    scale = min(detections["width"] / video["width"], detections["height"] / video["height"])
    left = (detections["width"] - round(video["width"] * scale)) // 2
    top = (detections["height"] - round(video["height"] * scale)) // 2
    matches = {}
    for sample in video["samples"]:
        truth = []
        for p in sample["persons"]:
            x, y, w, h = p["bbox"]
            truth.append({**p, "bbox": [x * scale + left, y * scale + top, w * scale, h * scale]})
        row = by_index[sample["frame_index"]]
        pairs = match_detection_pairs([SimpleNamespace(**p) for p in row["persons"]], truth, 0.5)
        matches[sample["frame_index"]] = {pred: truth[gt]["identity_id"] for pred, gt, _ in pairs}
    results = []
    for sample in video["samples"]:
        anchor = by_index[sample["frame_index"]]
        for person in sample["persons"]:
            indices = [
                i
                for i, identity in matches[sample["frame_index"]].items()
                if identity == person["identity_id"]
            ]
            result = {
                "anchor_frame": sample["frame_index"],
                "identity_id": person["identity_id"],
                "anchor_matched": bool(indices),
                "targets": [],
            }
            if indices:
                window = [
                    r
                    for r in rows
                    if abs(r["captured_mono_ns"] - anchor["captured_mono_ns"]) <= 8e9
                ]
                association = anchored_segment(window, anchor["frame_id"], indices[0])
                result["association"] = association
                result["context_frames"] = len(window)
                for target in video["samples"]:
                    row = by_index[target["frame_index"]]
                    if row == anchor or row not in window:
                        continue
                    if not any(
                        p["identity_id"] == person["identity_id"] for p in target["persons"]
                    ):
                        continue
                    chosen = association["selected"].get(row["frame_id"])
                    actual = matches[target["frame_index"]].get(chosen)
                    status = (
                        "abstained"
                        if chosen is None
                        else (
                            "correct"
                            if actual == person["identity_id"]
                            else "wrong_identity"
                            if actual is not None
                            else "unmatched_detection"
                        )
                    )
                    result["targets"].append(
                        {"frame_index": target["frame_index"], "status": status}
                    )
            results.append(result)
    counts = {
        status: sum(t["status"] == status for r in results for t in r["targets"])
        for status in ("correct", "wrong_identity", "unmatched_detection", "abstained")
    }
    return {
        "anchors": results,
        "anchor_count": len(results),
        "matched_anchors": sum(r["anchor_matched"] for r in results),
        "target_counts": counts,
        "limitations": [
            "Ground truth selects correct anchor detections only for this conditional diagnostic.",
            "Targets exclude the anchor; sparse labels do not verify intermediate identities.",
            "Previously used development video and provisional labels, not held-out validation.",
            "No incident accuracy, cross-view inference or live-loop performance claim.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    d = json.loads(args.detections.read_text())
    dataset = json.loads(args.dataset.read_text())
    video = next(v for v in dataset["videos"] if v["video_id"] == d["video_id"])
    if d["source_sha256"] != video["sha256"] or d["dataset_sha256"] != file_sha256(args.dataset):
        raise ValueError("identity dataset/source mismatch")
    validate_evidence(d["runtime"]["cool"], d["runtime"]["opencv_version"])
    report = evaluate(d, video)
    report.update(
        detections_sha256=file_sha256(args.detections), dataset_sha256=file_sha256(args.dataset)
    )
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "anchors"}, indent=2))


if __name__ == "__main__":
    main()
