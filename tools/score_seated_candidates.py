"""Compare an immutable shadow replay with manually reviewed video intervals."""

import argparse
import hashlib
import json
from pathlib import Path

from care_vision.seated_evaluation import score_intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve earlier evidence")
    replay = json.loads(args.replay.read_text())
    annotations = json.loads(args.annotations.read_text())
    if replay.get("mode") != "offline_shadow" or annotations.get("schema_version") != 1:
        parser.error("requires offline shadow replay and annotation schema v1")
    by_hash = {r["source_sha256"]: r for r in replay["runs"]}
    reviews = annotations["runs"]
    if not reviews or len({r["source_sha256"] for r in reviews}) != len(reviews):
        parser.error("reviews must contain unique, nonempty source hashes")
    if any(r.get("dataset_split") not in {"development", "evaluation_reserved"} for r in reviews):
        parser.error("each review must declare its dataset split")
    result = {
        "experiment": "reviewed-seated-candidates-v1",
        "replay_sha256": hashlib.sha256(args.replay.read_bytes()).hexdigest(),
        "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "runs": [score_intervals(by_hash[r["source_sha256"]], r) for r in reviews],
        "unreviewed_replay_source_hashes": sorted(set(by_hash) - {
            r["source_sha256"] for r in reviews
        }),
        "live_activation_supported": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps([r["outcome_counts"] for r in result["runs"]], indent=2))


if __name__ == "__main__":
    main()
