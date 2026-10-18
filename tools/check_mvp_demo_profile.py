"""Check the local frozen demo package; never opens camera, network, or inference."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/mvp-demo-preparation-20261016/profile.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check(manifest):
    profile = json.loads(manifest.read_text())
    failures = []
    for name, expected in profile["files_sha256"].items():
        path = ROOT / name
        if not path.is_file():
            failures.append({"path": name, "reason": "missing"})
        elif digest(path) != expected:
            failures.append({"path": name, "reason": "sha256_mismatch"})
    assessment_path = ROOT / profile["assessment"]
    historical_pass = False
    if assessment_path.is_file():
        assessment = json.loads(assessment_path.read_text())
        historical_pass = (
            assessment.get("passed") is True
            and len(assessment.get("runs", [])) == 5
            and all(run.get("passed") is True for run in assessment["runs"])
            and sum(run["sent_received_processed_dropped"][2]
                    for run in assessment["runs"]) == 759
        )
    return {
        "passed": not failures and historical_pass and bool(profile["files_sha256"]),
        "files_checked": len(profile["files_sha256"]),
        "failures": failures,
        "historical_five_replays_passed": historical_pass,
        "scope": "Local integrity only; no fresh AWS, camera, inference, or demo acceptance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = check(args.manifest)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
