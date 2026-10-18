"""Audit the two-run Graviton continuity experiment, separately from full demo.

Recorded-pixel replay and structured incident state only. This does not validate
physical identity, fall detection, WAN packet loss or preservation of every frame.
"""

import argparse
import hashlib
import json
import stat
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def snapshot(path):
    data = json.loads(path.read_text())
    actual = hashlib.sha256(json.dumps(
        data["snapshot"], sort_keys=True, allow_nan=False
    ).encode()).hexdigest()
    if data["sha256"] != actual:
        raise ValueError("snapshot digest mismatch")
    return data["snapshot"]["episode"]


def audit(root):
    before = snapshot(root / "state-before-restart.json")
    after = snapshot(root / "state-after-restart.json")
    cleanup = json.loads((root / "service-cleanup.json").read_text())
    runs = []
    followups = []
    episode_ids = set()
    manifests = []
    compatible_new_tracks = 0
    for name in ("disconnect-1", "restart-1"):
        folder = root / name
        cloud, edge = rows(folder / "cloud.jsonl"), rows(folder / "edge.jsonl")
        observations = [r for r in cloud if r["event"] == "observation"]
        actions = [r for r in cloud if r["event"] == "agent_action"]
        by_observation = {r["observation_id"]: r for r in observations}
        cloud_acks = {r["command_id"]: r for r in cloud if r["event"] == "ack_received"}
        edge_acks = {r["command_id"]: r for r in edge if r["event"] == "command_ack"}
        for action in actions:
            if action["reason"] != "unresolved_episode_requires_more_observation":
                continue
            ack = cloud_acks.get(action["command_id"])
            edge_ack = edge_acks.get(action["command_id"])
            cause = by_observation[action["caused_by_observation_id"]]
            evidence = [r for r in observations
                        if r.get("evidence_kind", "live") == "live" and r["valid"]
                        and r["session_id"] == action["session_id"]
                        and r["applied_command_id"] == action["command_id"]
                        and r["target_transmit_fps"] == 5
                        and edge_ack and r["capture_seq"] >= edge_ack["effective_from_capture_seq"]]
            followups.append({
                "run": name, "command_id": action["command_id"],
                "session_id": action["session_id"], "fresh_valid_observations": len(evidence),
                "causal_ack_and_new_perception": bool(
                    cause["valid"] and cause["frame_id"] == action["based_on_frame_id"]
                    and ack and edge_ack and ack["result"] == edge_ack["result"] == "applied"
                    and ack["session_id"] == edge_ack["session_id"] == action["session_id"]
                    and evidence
                ),
            })
        received = [r for r in cloud if r["event"] == "frame_received"]
        observed_keys = {(r["session_id"], r["frame_id"], r.get("evidence_kind", "live"))
                         for r in observations}
        unprocessed = [{k: r[k] for k in ("session_id", "frame_id", "capture_seq")}
                       for r in received if (
                           r["session_id"], r["frame_id"], r.get("evidence_kind", "live")
                       ) not in observed_keys]
        ready = next(r for r in cloud if r["event"] == "server_ready")
        incidents = [r for r in cloud if r["event"] == "incident_episode"]
        episode_ids.update(r["episode_id"] for r in incidents)
        _, reference_y, _, reference_h = before["reference_bbox"]
        for r in observations:
            if not r["valid"] or len(r["persons"]) != 1:
                continue
            p = r["persons"][0]
            if p["track_state"] != "confirmed" or p["track_id"] == before["track_id"]:
                continue
            _, y, w, h = p["tracked_bbox"]
            if (w / h <= .8 and h / r["height"] >= .85 * reference_h
                    and y / r["height"] <= reference_y + .15 * reference_h):
                compatible_new_tracks += 1
        runs.append({
            "run": name, "sent": sum(r["event"] == "frame_sent" for r in edge),
            "received": len(received), "observed": len(observations),
            "unprocessed_received_frames": unprocessed,
            "sessions": sum(r["event"] == "session_started" for r in cloud),
            "diagnostic_disconnects": sum(r["event"] == "diagnostic_disconnect" for r in cloud),
            "restores": sum(r["event"] == "episode_restored" for r in cloud),
            "closures": sum(r["incident_status"] in {"recovery_observed", "posture_change_cleared"}
                            for r in incidents),
            "test_alerts": sum(r["event"] == "test_alert" for r in cloud),
            "drain_confirmed": any(r["event"] == "drain_confirmed" for r in cloud),
            "runtime_verified": (ready["cool_verified"]
                                 and ready["runtime"]["ec2"]["architecture"] == "arm64"),
            "budget_exhaustion_events": sum(
                r.get("reason") == "episode_observation_budget_exhausted" for r in incidents
            ),
            "original_report_passed": json.loads((folder / "report.json").read_text())["passed"],
            "source_hashes": {f: hashlib.sha256((folder / f).read_bytes()).hexdigest()
                              for f in ("cloud.jsonl", "edge.jsonl", "source.tar.gz")},
        })
        manifests.append(json.loads((folder / "source-manifest.json").read_text()))
    core = ["controller", "server", "vision", "tracking", "episode_store", "yolox"]
    checks = {
        "graviton_cool_verified": all(r["runtime_verified"] for r in runs),
        "same_core_pipeline": all(manifests[0][f"src/care_vision/cloud/{f}.py"]
                                  == manifests[1][f"src/care_vision/cloud/{f}.py"] for f in core),
        "controlled_disconnect_once": runs[0]["diagnostic_disconnects"] == 1
                                      and runs[1]["diagnostic_disconnects"] == 0,
        "restored_in_both_runs": all(r["restores"] == 1 for r in runs),
        "logical_episode_preserved": episode_ids == {before["episode_id"]} == {after["episode_id"]},
        "structured_evidence_preserved": all(before[k] == after[k] for k in [
            "origin_identity", "track_id", "reference_bbox", "horizontal_evidence_seconds",
            "horizontal_observed", "test_alert_emitted",
        ]),
        "identity_stays_unresolved": after["identity_link_status"] == "unresolved_after_reconnect",
        "fresh_observation_loop": len(followups) == 3
                                  and all(f["causal_ack_and_new_perception"] for f in followups),
        "no_false_closure_or_alert": all(r["closures"] == r["test_alerts"] == 0 for r in runs),
        "new_track_upright_not_recovery": compatible_new_tracks > 0,
        "budget_persisted": before["followup_requests"] == 2 and after["followup_requests"] == 3
                            and after["followup_budget_reported"]
                            and runs[1]["budget_exhaustion_events"] == 1,
        "final_sessions_drained": all(r["drain_confirmed"] for r in runs),
        "distinct_service_processes_stopped": len({p["pid"] for p in cleanup["services"]}) == 2
            and not any(p["running_target_cloud"] for p in cleanup["services"]),
        "private_snapshot": cleanup["state_mode"] == "0o600"
            and all(stat.S_IMODE((root / f).stat().st_mode) == 0o600 for f in [
                "state-before-restart.json", "state-after-restart.json"]),
    }
    return {
        "milestone_passed": all(checks.values()), "checks": checks, "runs": runs,
        "followups": followups,
        "horizontal_evidence_seconds": before["horizontal_evidence_seconds"],
        "new_track_reference_posture_observations": compatible_new_tracks,
        "transport_complete": not any(r["unprocessed_received_frames"] for r in runs),
        "full_demo_passed": False,
        "scope": "Two recorded development replays; logical episode/structured evidence only; "
                 "controlled WebSocket close and new cloud process, "
                 "not WAN outage or physical ReID",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["milestone_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
