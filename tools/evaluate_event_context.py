"""Evaluate event context on validated existing loop logs, without new image inference."""

import argparse
import json
import platform
from pathlib import Path

from care_vision.adaptive_evaluation import audit_adaptive_run
from care_vision.check_loop import read_log
from care_vision.event_context_evaluation import replay_event_contexts
from care_vision.sequence_evaluation import score_episodes
from care_vision.temporal_evaluation import file_sha256, validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--association-policy", choices=["single_person", "mutual_unique"], default="single_person"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    dataset = json.loads(args.dataset.read_text())
    validate_dataset(dataset)
    video = next(v for v in dataset["videos"] if v["video_id"] == args.video_id)
    deployment = json.loads((args.run / "deployment.json").read_text())
    if deployment.get("input_video_sha256") != video["sha256"]:
        raise ValueError("annotated source does not match deployment")
    edge, cloud = read_log(args.run / "edge.jsonl"), read_log(args.run / "cloud.jsonl")
    audit = audit_adaptive_run(edge, cloud, video)
    report = replay_event_contexts(cloud, association_policy=args.association_policy)
    for summary in report["summaries"]:
        summary["final_annotation_score"] = score_episodes(video, summary["final"]["episodes"])
    report.update(
        association_policy=args.association_policy,
        dataset_sha256=file_sha256(args.dataset),
        review=dataset["review"],
        log_hashes={n: file_sha256(args.run / n) for n in ("edge.jsonl", "cloud.jsonl")},
        vision_runtime=audit["runtime"],
        source_sha256=video["sha256"],
        replay_runtime={
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        limitations=[
            "Causal offline replay, not connected to live decisions; no new image inference.",
            "Geometric association is experimental, not verified physical identity.",
            "Compare rebuilt combined/live-only trajectories; never concatenate tracker IDs.",
            "Earlier estimates can be revised when delayed evidence arrives; they are not alerts.",
            "History can be incomplete; post-event context is limited to eight capture seconds.",
            "Provisional annotations and one development event; no held-out validation.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        target.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {k: v for k, v in s.items() if k not in {"final", "final_annotation_score"}}
                for s in report["summaries"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
