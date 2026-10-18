"""Audit an actual file-based adaptive loop against its exact source annotations."""

import argparse
import json
from pathlib import Path

from care_vision.adaptive_evaluation import audit_adaptive_run
from care_vision.check_loop import read_log
from care_vision.temporal_evaluation import file_sha256, validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    dataset = json.loads(args.dataset.read_text())
    validate_dataset(dataset)
    video = next(v for v in dataset["videos"] if v["video_id"] == args.video_id)
    deployment = json.loads((args.run / "deployment.json").read_text())
    if (
        file_sha256(args.run / "demo.avi") != video["sha256"]
        or deployment.get("input_video_sha256") != video["sha256"]
    ):
        raise ValueError("run source does not match annotated video")
    report = audit_adaptive_run(
        read_log(args.run / "edge.jsonl"), read_log(args.run / "cloud.jsonl"), video
    )
    report.update(
        dataset_sha256=file_sha256(args.dataset),
        source_sha256=video["sha256"],
        review=dataset["review"],
        deployment=deployment,
        log_hashes={name: file_sha256(args.run / name) for name in ("edge.jsonl", "cloud.jsonl")},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        target.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "actions": report["actions"],
                "original_demo_check_passed": report["original_demo_check"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
