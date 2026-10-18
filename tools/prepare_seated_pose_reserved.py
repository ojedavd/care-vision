"""Freeze the current seated-pose diagnostic and prepare unseen capture slots.

Plan-only: no camera, speech, credentials, AWS writes or image inference.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from care_vision.cloud.seated_candidate import SeatedPolicy
from care_vision.cloud.seated_pose_candidate import SeatedPosePolicy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_freeze(freeze, root):
    required = {
        "src/care_vision/cloud/seated_pose_candidate.py",
        "src/care_vision/cloud/seated_candidate.py",
        "tools/run_tapo_spoken_trial.py",
    }
    if (
        freeze.get("schema_version") != 1
        or freeze.get("shadow_only") is not True
        or not required.issubset(freeze.get("file_sha256", {}))
    ):
        raise ValueError("requires complete shadow-only frozen manifest")
    changed = [
        name
        for name, digest in freeze["file_sha256"].items()
        if not (root / name).is_file() or sha(root / name) != digest
    ]
    if changed:
        raise ValueError("frozen inputs changed: " + ", ".join(changed))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.verify:
        verify_freeze(json.loads(args.verify.read_text()), root)
        print("frozen inputs unchanged")
        return
    if args.output is None:
        parser.error("select --output or --verify")
    os.umask(0o077)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    paths = [
        "src/care_vision/cloud/seated_candidate.py",
        "src/care_vision/cloud/seated_pose_candidate.py",
        "src/care_vision/cloud/pose_evidence.py",
        "src/care_vision/cloud/rtmpose.py",
        "src/care_vision/cloud/tracking.py",
        "src/care_vision/cloud/yolox.py",
        "src/care_vision/cloud/person.py",
        "src/care_vision/cloud/controller.py",
        "src/care_vision/cloud/server.py",
        "src/care_vision/aws_demo.py",
        "tools/run_tapo_spoken_trial.py",
        "tools/compare_pose_cache.py",
        "tools/yolo_pose_candidate.py",
        "tools/replay_seated_pose_candidates.py",
        "tools/prepare_recorded_episode_pose_cache.py",
        "tools/score_seated_pose_reserved.py",
        "tools/prepare_seated_pose_reserved.py",
        "src/care_vision/seated_pose_evaluation.py",
        "artifacts/seated-shadow-20261014/seat-context.json",
        "evaluation/seated-pose-20261014/recorded-plan.json",
        "artifacts/phase7d-pose-20260921/models/end2end.onnx",
        "artifacts/pose-comparison-20261014/models/yolo11n-pose.onnx",
        "artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx",
    ]
    freeze = {
        "schema_version": 1,
        "version": "seated-pose-reserved-v1-20261014",
        "file_sha256": {name: sha(root / name) for name in paths},
        "geometry_policy": asdict(SeatedPolicy()),
        "pose_policy": asdict(SeatedPosePolicy()),
        "capture_profile": {
            "target_capture_fps": 25,
            "measured_capture_fps": None,
            "transmit_fps": [2, 5],
            "resolution": [1280, 720],
            "detector": "YOLOX-S640",
            "strong_confidence": 0.35,
            "weak_association": True,
            "rtsp_ffmpeg_nobuffer": False,
            "cloud_threads": 8,
        },
        "shadow_only": True,
        "support_producer_validated": False,
        "live_activation_supported": False,
        "scope": "new recordings of known scene; no camera/participant generalization claim",
        "capture_started": False,
    }
    known = subprocess.run(
        ["rg", "--files", "artifacts"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    previous_files = [
        root / name for name in known if Path(name).name in {"demo.avi", "cloud.jsonl"}
    ]
    freeze["development_source_sha256"] = sorted({sha(p) for p in previous_files})
    cases = [
        ("sit-remain-1", "seated-hold-60", "seated", "current", "remain seated through end"),
        ("crouch-near-1", "crouch-near-seat-60", "confuser", "current", "beside seat, not seated"),
        ("kneel-near-1", "kneel-near-seat-60", "confuser", "current", "kneeling beside seat"),
        ("bend-near-1", "bend-near-seat-60", "confuser", "current", "reach object beside seat"),
        (
            "sit-occluded-1",
            "seated-hold-60",
            "seated",
            "current",
            "record actual occlusion, mark unobservable joints/support",
        ),
        (
            "seat-context-changed-1",
            "seated-hold-60",
            "unknown",
            "changed",
            "move chair before capture; old band is unverified",
        ),
    ]
    slots = []
    for case_id, mode, intended, context, description in cases:
        command = [
            sys.executable,
            str(root / "tools/run_tapo_spoken_trial.py"),
            "--host",
            "unused",
            "--identity",
            "unused",
            "--rtsp-ip",
            "192.0.2.1",
            "--output",
            "unused",
            "--mode",
            mode,
            "--dataset-split",
            "evaluation_reserved",
            "--experimental-weak-association",
            "--plan-only",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        slots.append(
            {
                "case_id": case_id,
                "capture_mode": mode,
                "status": "not_captured",
                "planned_family": intended,
                "planned_context_status": context,
                "description": description,
                "trial_plan": json.loads(result.stdout),
                "reviewed_intervals": [],
                "source_sha256": None,
                "reviewed_by": None,
                "review_scope": "pending",
                "posture_labels_are_planned_not_measured": True,
            }
        )
    freeze_path = args.output / "freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "freeze_sha256": sha(freeze_path),
        "capture_started": False,
        "cases": slots,
        "admission_rules": [
            "Check frozen file hashes before capture, inference and replay",
            "New video/log hashes must not occur in development recordings",
            "Review full video and actual timestamps before reading predictions",
            "Spoken cues are instructions, never labels",
            "Freeze changes or tuning reclassify affected evaluation as development",
            "Unobserved/unknown/occluded support cannot count as successful rejection",
        ],
    }
    (args.output / "trial-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "annotations-template.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "freeze_sha256": sha(freeze_path),
                "runs": [],
                "template_only": True,
                "review_scope": "pending",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"prepared {len(slots)} reserved slots; no captures executed")


if __name__ == "__main__":
    main()
