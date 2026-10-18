"""Audit recorded ROI interruption/reconnection and process restart separately from demo."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_aws_episode_continuity import rows, snapshot


def key(row):
    return row["session_id"], row["frame_id"], row.get("evidence_kind", "live")


def audit(root):
    runs, followups, all_cycles, all_incidents, states, pids = [], [], [], [], [], []
    for name in ("cut-roi", "restart-from-disk"):
        folder = root / name
        cloud, edge = rows(folder / "cloud.jsonl"), rows(folder / "edge.jsonl")
        state = snapshot(folder / "episode.json")
        states.append(state)
        pids.append(json.loads((folder / "deployment.json").read_text())["pid"])
        observations = [r for r in cloud if r["event"] == "observation"]
        restored_sessions = {r["session_id"] for r in cloud if r["event"] == "episode_restored"}
        by_observation = {r["observation_id"]: r for r in observations}
        actions = [r for r in cloud if r["event"] == "agent_action"]
        acknowledgments = {r["command_id"]: r for r in cloud if r["event"] == "ack_received"}
        edge_acks = {r["command_id"]: r for r in edge if r["event"] == "command_ack"}
        for action in actions:
            if action["reason"] != "unresolved_episode_requires_more_observation":
                continue
            cid, sid = action["command_id"], action["session_id"]
            ack, edge_ack = acknowledgments.get(cid), edge_acks.get(cid)
            cause = by_observation.get(action["caused_by_observation_id"])
            evidence = [r for r in observations if r["valid"]
                        and r.get("evidence_kind", "live") == "live" and r["session_id"] == sid
                        and r["applied_command_id"] == cid and r["target_transmit_fps"] == 5
                        and edge_ack and r["capture_seq"] >= edge_ack["effective_from_capture_seq"]]
            returns = [a for a in actions if a["type"] == "SET_TRANSMIT_FPS"
                       and a["transmit_fps"] == 2 and a["session_id"] == sid
                       and a["local_mono_ns"] > action["local_mono_ns"]]
            restored = returns[0] if returns else {}
            rid = restored.get("command_id")
            baseline_evidence = [r for r in observations if r["valid"]
                                 and r["session_id"] == sid and r["applied_command_id"] == rid
                                 and r["target_transmit_fps"] == 2]
            followups.append({
                "run": name, "session_id": sid, "command_id": cid,
                "fresh_valid_5fps": len(evidence), "fresh_valid_2fps": len(baseline_evidence),
                "causal_loop_passed": bool(
                    cause and cause["valid"] and cause["frame_id"] == action["based_on_frame_id"]
                    and ack and edge_ack and ack["result"] == edge_ack["result"] == "applied"
                    and ack["session_id"] == edge_ack["session_id"] == sid and evidence
                    and rid in acknowledgments and rid in edge_acks
                    and acknowledgments[rid]["result"] == edge_acks[rid]["result"] == "applied"
                    and baseline_evidence
                ),
            })
        cycles = [r for r in cloud if r["event"] == "pose_observation_cycle"]
        incidents = [r for r in cloud if r["event"] == "incident_episode"]
        all_cycles.extend(cycles)
        all_incidents.extend(incidents)
        sent = [r for r in edge if r["event"] == "frame_sent"]
        received = [r for r in cloud if r["event"] == "frame_received"]
        sent_keys, received_keys, observed_keys = (set(map(key, rs)) for rs in
                                                   (sent, received, observations))
        sessions = []
        for start in (r for r in cloud if r["event"] == "session_started"):
            sid = start["session_id"]
            parts = [[r for r in rs if r["session_id"] == sid] for rs in
                     (sent, received, observations)]
            sessions.append({
                "session_id": sid,
                "live_sent_received_observed": [sum(r.get("evidence_kind", "live") == "live"
                                                    for r in rs) for rs in parts],
                "historical_sent_received_observed": [
                    sum(r.get("evidence_kind", "live") != "live" for r in rs) for rs in parts],
                "finished": [r for r in cloud if r["event"] == "session_finished"
                             and r["session_id"] == sid],
            })
        ready = next(r for r in cloud if r["event"] == "server_ready")
        runs.append({
            "run": name, "sessions": sessions, "sent": len(sent), "received": len(received),
            "observed": len(observations),
            "sent_not_received": sorted(sent_keys - received_keys),
            "received_not_observed": sorted(received_keys - observed_keys),
            "observed_not_received": sorted(observed_keys - received_keys),
            "received_not_sent": sorted(received_keys - sent_keys),
            "duplicate_frame_records": [len(rs) - len(set(map(key, rs)))
                                        for rs in (sent, received, observations)],
            "drops": [r for r in cloud if r["event"] == "frame_dropped"],
            "roi_requests": sum(r["type"] == "REQUEST_ROI" for r in actions),
            "restored_full_frame_only": bool(restored_sessions) and all(
                r["view_kind"] == "full_frame" for r in observations
                if r["session_id"] in restored_sessions and r.get("evidence_kind", "live") == "live"
            ),
            "cuts": sum(r["event"] == "diagnostic_disconnect" for r in cloud),
            "probe_triggers": [r for r in cloud if r["event"] == "pose_continuity_probe_trigger"],
            "restores": [r for r in cloud if r["event"] == "episode_restored"],
            "closures": sum(r["incident_status"] in {"recovery_observed", "posture_change_cleared"}
                            for r in incidents),
            "alerts": sum(r["event"] == "test_alert" for r in cloud),
            "pose_closures": sum(bool(r.get("would_close_episode")) for r in cloud
                                 if r["event"] == "pose_shadow"),
            "stale": sum(r.get("invalid_reason") == "processing_stale" for r in observations),
            "errors": [r for r in cloud if r["event"] in {"session_error", "pose_shadow_error"}],
            "drains": [r for r in cloud if r["event"] == "drain_confirmed"],
            "runtime_verified": ready["cool_verified"]
                and ready["runtime"]["ec2"]["architecture"] == "arm64",
            "state_metadata": json.loads((folder / "state-metadata.json").read_text()),
            "cleanup": json.loads((folder / "cleanup.json").read_text()),
        })
    deployment = json.loads((root / "deployment.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    project = Path(__file__).resolve().parents[1]
    def matches(path, digest):
        return hashlib.sha256((project / path).read_bytes()).hexdigest() == digest

    checks = {
        "graviton_cool": all(r["runtime_verified"] for r in runs),
        "one_cut_in_first_run": [r["cuts"] for r in runs] == [1, 0],
        "first_roi_interrupted": sum(r["stage"] == "roi_evidence_processed"
                                     for r in all_cycles) == 1
            and sum(r["stage"] == "session_ended_cycle_incomplete" for r in all_cycles) == 1,
        "no_cycle_misreported_complete": not any(r["stage"] == "cycle_complete"
                                                 for r in all_cycles),
        "one_roi_total": [r["roi_requests"] for r in runs] == [1, 0],
        "restored_sessions_full_frame": all(r["restored_full_frame_only"] for r in runs),
        "both_restores": all(len(r["restores"]) == 1 for r in runs),
        "logical_episode_preserved": {r["episode_id"] for r in all_incidents}
            == {s["episode_id"] for s in states} == {states[0]["episode_id"]},
        "evidence_and_roi_budget_preserved": all(states[0][k] == states[1][k] for k in (
            "origin_identity", "reference_bbox", "track_id", "horizontal_observed",
            "horizontal_evidence_seconds", "test_alert_emitted", "pose_observation_requested",
        )) and all(s["pose_observation_requested"] for s in states),
        "identity_unresolved": all(s["identity_link_status"] == "unresolved_after_reconnect"
                                   and s["incident_status"] == "unresolved" for s in states),
        "followup_budget_monotonic": states[0]["followup_requests"]
            < states[1]["followup_requests"] <= 3,
        "fresh_loops_in_both_runs": {f["run"] for f in followups} == {r["run"] for r in runs}
            and all(f["causal_loop_passed"] for f in followups),
        "no_closure_alert_or_stale": all(r["closures"] == r["alerts"]
                                        == r["pose_closures"] == r["stale"] == 0 for r in runs),
        "only_expected_disconnect_error": len(runs[0]["errors"]) == 1
            and runs[0]["errors"][0]["error_type"] == "ConnectionClosedError"
            and "1012" in runs[0]["errors"][0]["reason"]
            and "controlled continuity test" in runs[0]["errors"][0]["reason"]
            and not runs[1]["errors"],
        "final_sessions_drained": all(len(r["drains"]) == 1 for r in runs),
        "distinct_processes_stopped": len(set(pids)) == 2
            and all(r["cleanup"]["confirmed_stopped"] for r in runs),
        "private_snapshots": all(r["state_metadata"]["mode"] == "0o600" for r in runs),
        "prior_snapshot_preserved": json.loads((root / "previous-state-before.json").read_text())
            == json.loads((root / "previous-state-after.json").read_text()),
        "workspace_removed": deployment["cleanup"]["workspace_removed"],
        "frozen_sources_and_inputs": all(matches(p, digest) for p, digest in
                                         {**plan["code_sha256"], **plan["inputs"]}.items()),
    }
    return {
        "milestone_passed": all(checks.values()), "checks": checks, "runs": runs,
        "followups": followups, "cycles": all_cycles, "final_states": states,
        "transport_complete": all(not r[k] for r in runs for k in (
            "sent_not_received", "received_not_observed", "observed_not_received",
            "received_not_sent", "drops",
        )),
        "full_demo_passed": False,
        "scope": "Archived development video twice; controlled WS close and orderly process "
                 "restart; no live camera, physical ReID, seated closure, "
                 "fall validation or WAN loss",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"milestone_passed": result["milestone_passed"],
                      "transport_complete": result["transport_complete"],
                      "checks": result["checks"]}, indent=2))
    if not result["milestone_passed"]:
        raise SystemExit(1)
