"""Evaluate causal sequence hypotheses from cached Phase 7D vision observations."""

import argparse
import json
import platform
from pathlib import Path

from care_vision.sequence_evaluation import evaluate_sequences, validate_observations
from care_vision.temporal_evaluation import file_sha256, validate_evidence_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite existing evidence")
    observations = json.loads(args.observations.read_text())
    dataset = json.loads(args.dataset.read_text())
    manifest = json.loads(args.manifest.read_text())
    dataset_hash, manifest_hash = file_sha256(args.dataset), file_sha256(args.manifest)
    validate_observations(observations, dataset, dataset_hash, manifest_hash)
    validate_evidence_manifest(
        manifest, dataset, dataset_hash, [r["sample_fps"] for r in observations["runs"]]
    )
    report = evaluate_sequences(observations, dataset)
    report.update(
        observations_sha256=file_sha256(args.observations),
        dataset_sha256=dataset_hash,
        evidence_manifest_sha256=manifest_hash,
        model_sha256=observations["model_sha256"],
        pose_configuration=observations["pose_configuration"],
        vision_runtime=observations["runtime"],
        replay_runtime={
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        review=dataset["review"],
        limitations=[
            "Cached inference only; no new image inference, transport or active perception run.",
            "The policy consumes only current/past observations, never annotation phases.",
            "Development clips already used in earlier experiments; one staged positive event.",
            "All annotation/provenance limitations of Phase 7D remain; human review pending.",
            "Matching means a sequence candidate during floor rest, not a confirmed fall.",
            "Missing evidence and end of clip do not prove absence of an incident or recovery.",
            "A primary-track change aborts the sequence; no all-person safety coverage.",
            "Recovery cue latency is source-video time, not wall-clock alert latency.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for run in report["runs"]:
        print(
            f"{run['sample_fps']:g} FPS {run['gap_policy']} {run['variant']}: "
            f"matched {run['matched_events']}/{run['annotated_events']}, "
            f"negative-clip candidates {run['negative_clip_candidates']}"
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
