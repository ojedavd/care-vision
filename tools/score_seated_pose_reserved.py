"""Score blinded full-video reviews against frozen original-episode pose replays."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_seated_pose_reserved import verify_freeze

from care_vision.seated_pose_evaluation import score_pose_intervals


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads(args.freeze.read_text())
    verify_freeze(freeze, root)
    replay = json.loads(args.replay.read_text())
    reviews = json.loads(args.annotations.read_text())
    if reviews.get("template_only") or not reviews.get("runs"):
        raise ValueError("annotation template is not reviewed evidence")
    if reviews["freeze_sha256"] != sha(args.freeze):
        raise ValueError("review belongs to different frozen version")
    if replay.get("experiment") != "seated-pose-recorded-episodes-shadow-v1":
        raise ValueError("requires original recorded-episode pose replay")
    seen = set()
    results = []
    development = set(freeze["development_source_sha256"])
    for review in reviews["runs"]:
        digest = review["source_sha256"]
        if digest in seen:
            raise ValueError("duplicate review source")
        seen.add(digest)
        if review["dataset_split"] == "evaluation_reserved" and (
            digest in development or review["source_video_sha256"] in development
        ):
            raise ValueError("previous recording cannot be relabeled reserved")
        matches = [c for c in replay["conditions"] if c["source_sha256"] == digest]
        if not matches:
            raise ValueError("review source missing from replay")
        for condition in matches:
            if condition.get("source_video_sha256") != review["source_video_sha256"]:
                raise ValueError("review video hash differs from replay source")
            results.append(score_pose_intervals(condition, review))
    result = {
        "schema_version": 1,
        "freeze_sha256": sha(args.freeze),
        "replay_sha256": sha(args.replay),
        "annotations_sha256": sha(args.annotations),
        "runs": results,
        "unreviewed_source_sha256": sorted(
            {r["source_sha256"] for r in replay["conditions"]} - seen
        ),
        "live_activation_supported": False,
        "scope": (
            "sparse episode coverage/full-video reviews; no automatic benign closure validation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"scored {len(results)} model/source conditions; live activation unsupported")


if __name__ == "__main__":
    main()
