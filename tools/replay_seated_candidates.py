"""Inspect seated-like candidates in recorded cloud logs; no live decisions.

Region configuration is development input, never an outcome annotation.
Recorded episode openings supply the existing policy's reference/track only.
This does not simulate new episode openings or suppression after a closure.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from care_vision.cloud.seated_candidate import SeatContext, SeatedCandidate, SeatedPolicy


def replay(path, context):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    observations = {r["observation_id"]: r for r in rows if r["event"] == "observation"}
    origin = next(iter(observations.values()))["captured_mono_ns"]
    active = None
    episodes = []
    timeline = []

    def inspect(row):
        result = active.update(row)
        result["seconds_from_first_observation"] = (row["captured_mono_ns"] - origin) / 1e9
        timeline.append(result)
        if result["seated_like_candidate"] and episodes[-1]["first_candidate_seconds"] is None:
            episodes[-1]["first_candidate_seconds"] = result["seconds_from_first_observation"]

    for row in rows:
        if row["event"] == "observation" and active is not None:
            inspect(row)
        elif row["event"] == "incident_episode":
            if row["reason"] == "episode_opened":
                start = observations[row["observation_id"]]
                active = SeatedCandidate(row, start, context)
                episodes.append({
                    "episode_id": row["episode_id"], "track_id": row["track_id"],
                    "opened_seconds": (row["started_ns"] - origin) / 1e9,
                    "first_candidate_seconds": None,
                    "recorded_closure_seconds": None,
                })
                inspect(start)
            elif active is not None and row["episode_id"] == active.episode_id:
                active.blocked |= row["horizontal_observed"] or row["test_alert_emitted"]
                if row["incident_status"] in {"recovery_observed", "posture_change_cleared"}:
                    episodes[-1]["recorded_closure_seconds"] = (
                        row["captured_mono_ns"] - origin
                    ) / 1e9
                    active = None
    return {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "observations": len(observations),
        "episodes": episodes,
        "reason_counts": dict(Counter(r["reason"] for r in timeline)),
        "timeline": timeline,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Refuse to overwrite evidence from another experiment.
    if args.output.exists():
        parser.error("output already exists; choose a new evidence path")
    context = SeatContext(**json.loads(args.context.read_text()))
    result = {
        "experiment": "seated-like-shadow-v1", "mode": "offline_shadow",
        "thresholds_calibrated": False, "new_inferences": 0,
        "live_episode_closures": 0, "notifications_sent": 0,
        "context": asdict(context), "policy": asdict(SeatedPolicy()),
        "context_sha256": hashlib.sha256(args.context.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1]
             / "src/care_vision/cloud/seated_candidate.py").read_bytes()
        ).hexdigest(),
        "runs": [replay(path, context) for path in args.logs],
        "scope": (
            "Known development observations, recorded episode anchors and manual scene context. "
            "Geometric candidates only; no validated benign recognition, recovery or live closure."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{"source": r["source"], "episodes": r["episodes"]}
                      for r in result["runs"]], indent=2))


if __name__ == "__main__":
    main()
