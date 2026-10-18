"""Score only exact historical annotated frames; never interpolate identity labels."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from care_vision.evaluation import match_detection_pairs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scaled_truth(sample, width, height, transform=None):
    sx, sy, left, top = transform or [1280 / width, 720 / height, 0, 0]
    return [
        {
            "identity_id": p["identity_id"],
            "bbox": [
                p["bbox"][0] * sx + left,
                p["bbox"][1] * sy + top,
                p["bbox"][2] * sx,
                p["bbox"][3] * sy,
            ],
        }
        for p in sample["persons"]
    ]


def score_rows(rows):
    identities = defaultdict(list)
    assigned_ids = defaultdict(set)
    for row in rows:
        for match in row["matches"]:
            identities[match["identity_id"]].append(match["track_id"])
            assigned_ids[match["track_id"]].add(match["identity_id"])
    return {
        "annotated_frames_scored": len(rows),
        "expected_instances": sum(r["expected"] for r in rows),
        "matched_instances_iou05": sum(len(r["matches"]) for r in rows),
        "unmatched_instances": sum(r["expected"] - len(r["matches"]) for r in rows),
        "unmatched_predictions": sum(r["predicted"] - len(r["matches"]) for r in rows),
        "misses_with_raw_detection_match": sum(r["misses_with_raw_match"] for r in rows),
        "misses_without_raw_detection_match": sum(r["misses_without_raw_match"] for r in rows),
        "distinct_track_ids_by_identity": {k: len(set(v)) for k, v in identities.items()},
        "id_changes_between_matched_annotation_points": sum(
            sum(a != b for a, b in zip(ids, ids[1:])) for ids in identities.values()
        ),
        "track_ids_matched_to_multiple_annotation_identities": sum(
            len(ids) > 1 for ids in assigned_ids.values()
        ),
        "scope": "sparse exact-frame diagnostics; changes are not standard MOT ID switches",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite scores")
    manifest = json.loads(args.manifest.read_text())
    annotation_path = Path(manifest["annotation_path"])
    if sha(annotation_path) != manifest["annotation_sha256"]:
        raise ValueError("annotation digest mismatch")
    dataset = json.loads(annotation_path.read_text())
    plan = json.loads((args.base / "plan.json").read_text())
    if sha(args.manifest) != plan["manifest_sha256"]:
        raise ValueError("manifest digest mismatch")
    comparison = json.loads((args.base / "tracking-comparison.json").read_text())
    if sha(args.base / "tracking-timelines.jsonl") != comparison["timeline_sha256"]:
        raise ValueError("timeline digest mismatch")
    if sha(args.base / "detections.jsonl") != comparison["cache_sha256"]:
        raise ValueError("cache digest mismatch")
    raw = {
        r["frame_id"]: r
        for r in map(json.loads, (args.base / "detections.jsonl").read_text().splitlines())
    }
    timelines = {}
    for row in map(json.loads, (args.base / "tracking-timelines.jsonl").read_text().splitlines()):
        key = (row["condition"], row["tracker"], row["frame_id"])
        if key in timelines:
            raise ValueError("duplicate timeline row")
        timelines[key] = row
    results = []
    runs = {r["name"]: r for r in manifest["runs"]}
    for video in dataset["videos"]:
        for fps in (2, 5):
            condition = f"{video['video_id']}:{fps}fps"
            for tracker in ("PersonTracker", "ByteTrack"):
                rows, omitted = [], []
                for sample in video["samples"]:
                    frame_id = f"{video['video_id']}:{sample['frame_index'] + 1}"
                    timeline = timelines.get((condition, tracker, frame_id))
                    if timeline is None:
                        omitted.append(sample["frame_index"])
                        continue
                    truth = scaled_truth(
                        sample,
                        video["width"],
                        video["height"],
                        runs[video["video_id"]].get("box_transform"),
                    )
                    predicted = timeline["visible_tracks"]
                    matches = match_detection_pairs(
                        [SimpleNamespace(bbox=p["bbox"]) for p in predicted], truth, 0.5
                    )
                    raw_matches = match_detection_pairs(
                        [
                            SimpleNamespace(bbox=p["bbox"])
                            for p in raw[frame_id]["observation"]["persons"]
                        ],
                        truth,
                        0.5,
                    )
                    matched_truth = {j for _, j, _ in matches}
                    raw_truth = {j for _, j, _ in raw_matches}
                    missing = set(range(len(truth))) - matched_truth
                    rows.append(
                        {
                            "frame_id": frame_id,
                            "frame_index": sample["frame_index"],
                            "scene_tags": sample["scene_tags"],
                            "expected": len(truth),
                            "predicted": len(predicted),
                            "matches": [
                                {
                                    "identity_id": truth[j]["identity_id"],
                                    "track_id": predicted[i]["track_id"],
                                    "iou": iou,
                                }
                                for i, j, iou in matches
                            ],
                            "misses_with_raw_match": len(missing & raw_truth),
                            "misses_without_raw_match": len(missing - raw_truth),
                        }
                    )
                results.append(
                    {
                        "condition": condition,
                        "tracker": tracker,
                        "summary": score_rows(rows),
                        "rows": rows,
                        "unscored_annotation_frame_indices": omitted,
                    }
                )
    result = {
        "schema_version": 1,
        "annotation_sha256": sha(annotation_path),
        "annotation_review": dataset["review"],
        "new_ground_truth": False,
        "comparison_sha256": sha(args.base / "tracking-comparison.json"),
        "scorer_sha256": sha(Path(__file__)),
        "scope": "historical development annotations, no interpolation or full-stream metrics",
        "iou_threshold": 0.5,
        "conditions": results,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("sparse annotation scoring complete")


if __name__ == "__main__":
    main()
