"""Compare recovery guards on saved observations; no image inference or transport."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import replay_incident_episode as replay_module

from care_vision.cloud.controller import Controller


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(baseline, paths):
    old_path = baseline / "src/care_vision/cloud/controller.py"
    frozen = json.loads((baseline / "freeze.json").read_text())
    if sha(old_path) != frozen["file_sha256"]["src/care_vision/cloud/controller.py"]:
        raise ValueError("archived controller digest mismatch")
    spec = importlib.util.spec_from_file_location("baseline_controller", old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    runs = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        observations = [r for r in rows if r["event"] == "observation"]
        results = {}
        for name, cls in (("baseline", old.Controller), ("guard_v2", Controller)):
            replay_module.Controller = cls
            result = replay_module.replay(observations)
            results[name] = {
                **result,
                "closures": [e for e in result["events"]
                             if e["reason"] == "same_track_sustained_reference_posture"],
                "blocks": [e for e in result["events"]
                           if e["reason"] == "reference_posture_evidence_insufficient"],
            }
        replay_module.Controller = Controller
        runs.append({"source": str(path), "source_sha256": sha(path), **results})
    return {
        "schema_version": 1,
        "baseline_controller_sha256": sha(old_path),
        "guard_controller_sha256": sha(Path(sys.modules[Controller.__module__].__file__)),
        "guard_module_sha256": sha(
            Path(sys.modules[Controller.__module__].__file__).with_name("reference_posture.py")
        ),
        "runs": runs,
        "scope": (
            "Saved post-tracker observations with simulated ACKs and original sampling. "
            "Counterfactual controller replay, not fresh inference, live sensing, or "
            "transport verification. Data used for this fix are development."
        ),
        "new_image_inferences": 0,
        "normal_activity_recognition_validated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--logs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.output.exists():
        raise ValueError("refusing to overwrite experiment")
    result = compare(args.baseline, args.logs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for run in result["runs"]:
        print(Path(run["source"]).parent.name, {
            name: {"closures": len(run[name]["closures"]), "state": run[name]["final_state"],
                   "blocks": len(run[name]["blocks"])} for name in ("baseline", "guard_v2")
        })


if __name__ == "__main__":
    main()
