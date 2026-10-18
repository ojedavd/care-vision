"""Audit measured S-only transport and episode test-alert/recovery causality."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_s_only_service import audit, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--expect-alert", action="store_true")
    parser.add_argument(
        "--outcome", choices=["recovery", "open-alert", "negative"], default="recovery"
    )
    args = parser.parse_args()
    folder = args.folder
    transport = audit(folder)
    cloud = load(folder / "cloud.jsonl")
    observations = {r["observation_id"]: r for r in cloud if r["event"] == "observation"}
    events = [r for r in cloud if r["event"] == "incident_episode"]
    alerts = [r for r in cloud if r["event"] == "test_alert"]
    recoveries = [r for r in events if r["incident_status"] == "recovery_observed"]
    threshold = json.loads((folder / "deployment.json").read_text())[
        "experimental_alert_after_seconds"
    ]
    assert bool(alerts) == args.expect_alert
    closures = [
        r for r in events if r["incident_status"] in {"recovery_observed", "posture_change_cleared"}
    ]
    if args.outcome == "recovery":
        assert recoveries, "No measured same-track recovery event"
    elif args.outcome == "negative":
        assert not alerts and not recoveries, "Negative produced alert or horizontal recovery"
        assert any(r["incident_status"] == "posture_change_cleared" for r in closures)
    else:
        assert args.expect_alert, "open-alert requires --expect-alert"
        assert not closures, "Episode unexpectedly closed"
        assert len(alerts) == 1, "Expected one experimental alert"
        final = events[-1]
        assert final["reason"] == "evidence_ended_episode_open" and final["episode_open"]
        assert final["episode_id"] == alerts[0]["episode_id"]
        assert final["track_id"] == alerts[0]["track_id"]
        assert final["test_alert_emitted"]
    assert len({r["episode_id"] for r in alerts}) == len(alerts), "Duplicate test alert"
    for event in alerts + closures:
        observation = observations[event["observation_id"]]
        assert observation["valid"] and observation.get("evidence_kind", "live") == "live"
        assert event["frame_id"] == observation["frame_id"]
        assert any(
            p["track_id"] == event["track_id"] and p["track_state"] == "confirmed"
            for p in observation["persons"]
        )
    for alert in alerts:
        assert alert["test_only"] and not alert["notification_sent"]
        assert alert["horizontal_evidence_seconds"] >= threshold
        opened = next(
            r
            for r in events
            if r["episode_id"] == alert["episode_id"] and r["reason"] == "episode_opened"
        )
        assert opened["track_id"] == alert["track_id"]
        if args.outcome == "recovery":
            recovery = next(r for r in recoveries if r["episode_id"] == alert["episode_id"])
            assert recovery["captured_mono_ns"] > alert["captured_mono_ns"]
            assert recovery["test_alert_emitted"]
    for recovery in closures:
        opened = next(
            r
            for r in events
            if r["episode_id"] == recovery["episode_id"] and r["reason"] == "episode_opened"
        )
        assert opened["track_id"] == recovery["track_id"]
        if not args.expect_alert:
            assert recovery["horizontal_evidence_seconds"] < threshold
    transitions = [r for r in cloud if r["event"] == "agent_state_transition"]
    assert any(r["to_state"] == "ASSISTANCE_REQUIRED" for r in transitions) == args.expect_alert
    assert transitions[-1]["to_state"] == (
        "ASSISTANCE_REQUIRED" if args.outcome == "open-alert" else "NORMAL"
    ), "Wrong final episode state"
    origin = next(iter(observations.values()))["captured_mono_ns"]
    result = {
        "passed": True,
        "threshold_seconds": threshold,
        "expected_outcome": args.outcome,
        "transport": transport,
        "alerts": len(alerts),
        "recoveries": len(recoveries),
        "events": [
            {
                "reason": r["reason"],
                "status": r["incident_status"],
                "seconds": round((r["captured_mono_ns"] - origin) / 1e9, 3),
                "episode_id": r["episode_id"],
                "track_id": r["track_id"],
                "horizontal_evidence_seconds": r["horizontal_evidence_seconds"],
            }
            for r in events
            if "captured_mono_ns" in r
        ],
        "final_state": transitions[-1]["to_state"],
        "sha256": {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("cloud.jsonl", "edge.jsonl", "source.tar.gz")
        },
        "scope": (
            "Measured cloud test alert only; no third-party notification or validated fall decision"
        ),
    }
    (folder / "episode-audit-v2.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
