"""Compare fixed association policies on a dense provisional identity dataset."""

import argparse
import json
import platform
from pathlib import Path

from care_vision.common.cool import validate_evidence
from care_vision.dense_identity_evaluation import evaluate_dense
from care_vision.temporal_evaluation import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    detections = json.loads(args.detections.read_text())
    dataset = json.loads(args.dataset.read_text())
    validate_evidence(detections["runtime"]["cool"], detections["runtime"]["opencv_version"])
    result = evaluate_dense(detections, dataset)
    result.update(
        dataset_sha256=file_sha256(args.dataset),
        detections_sha256=file_sha256(args.detections),
        review=dataset["review"],
        replay_runtime={"system": platform.system(), "machine": platform.machine()},
    )
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "coverage": result["coverage"],
                "policies": {
                    k: {field: v[field] for field in ("adjacent_counts", "chain_counts")}
                    for k, v in result["policies"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
