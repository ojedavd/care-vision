"""Measured workload comparison; never derives incident accuracy from resource savings."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {"median": statistics.median(values),
            "p95": values[math.ceil(.95 * len(values)) - 1], "max": max(values)}


def resource_window(records, start_ns, end_ns):
    """Integrate sampled CPU proportionally at boundaries; this is an estimate."""
    cpu, coverage, rss = 0., 0., []
    for r in records:
        end = r["local_mono_ns"] / 1e9
        begin = end - r["interval_seconds"]
        overlap = max(0., min(end, end_ns / 1e9) - max(begin, start_ns / 1e9))
        cpu += overlap * r["cpu_percent_one_core"] / 100
        coverage += overlap
        if overlap and r.get("rss_bytes") is not None:
            rss.append(r["rss_bytes"])
    seconds = (end_ns - start_ns) / 1e9
    return {"window_seconds": seconds, "sample_coverage_seconds": coverage,
            "cpu_seconds_estimate": cpu, "mean_cpu_percent_one_core_estimate": 100 * cpu / seconds,
            "rss_bytes": distribution(rss),
            "method": "first frame receipt to last observation/pose result; proportional "
                      "CPU integration over ~1s samples; excludes pre-session model startup"}


def audit(root, adaptive_reference=None):
    project = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "plan.json").read_text())
    results = []
    cases = [(root, case) for case in plan["conditions"]]
    if adaptive_reference is not None:
        reference_plan = json.loads((adaptive_reference / "plan.json").read_text())
        cases += [(adaptive_reference, case) for case in reference_plan["conditions"]
                  if case["mode"] == "adaptive"]
    for case_root, case in cases:
        source, mode = case["source"], case["mode"]
        folder = case_root / (source + "-" + mode)
        original = project / "artifacts/pose-observation-live-20261014" / source
        captures = rows(original / "demo.captures.jsonl")
        stamps = {r["capture_seq"]: r["captured_mono_ns"] for r in captures}
        duration = (captures[-1]["captured_mono_ns"] - captures[0]["captured_mono_ns"]) / 1e9
        edge, cloud = rows(folder / "edge.jsonl"), rows(folder / "cloud.jsonl")
        sent = [r for r in edge if r["event"] == "frame_sent"]
        received = [r for r in cloud if r["event"] == "frame_received"]
        observed = [r for r in cloud if r["event"] == "observation"]
        dropped = [r for r in cloud if r["event"] == "frame_dropped"]
        ids = [set(r["frame_id"] for r in group) for group in (sent, received, observed, dropped)]
        actions = [r for r in cloud if r["event"] == "agent_action"]
        acks = {r["command_id"]: r for r in cloud if r["event"] == "ack_received"}
        edge_acks = {r["command_id"]: r for r in edge if r["event"] == "command_ack"}
        by_observation = {r["observation_id"]: r for r in observed}
        action_audit = []
        for action in actions:
            field = ("applied_command_id" if action["type"] == "SET_TRANSMIT_FPS"
                     else "applied_visual_command_id")
            cid = action["command_id"]
            cause = by_observation.get(action["caused_by_observation_id"])
            ack, edge_ack = acks.get(cid), edge_acks.get(cid)
            evidence = [r for r in observed if r.get(field) == cid and edge_ack
                        and r["capture_seq"] >= edge_ack.get("effective_from_capture_seq", 0)]
            action_audit.append({
                "command_id": cid, "type": action["type"], "fps": action.get("transmit_fps"),
                "new_observations": len(evidence),
                "new_valid_observations": sum(r["valid"] for r in evidence),
                "passed": bool(cause and cause["frame_id"] == action["based_on_frame_id"]
                               and ack and edge_ack
                               and ack["result"] == edge_ack["result"] == "applied"
                               and ack["session_id"] == edge_ack["session_id"]
                               == action["session_id"]
                               and evidence),
            })
        cycles = [r for r in cloud if r["event"] == "pose_observation_cycle"]
        incidents = [r for r in cloud if r["event"] == "incident_episode"]
        closures = []
        for r in incidents:
            if r["incident_status"] not in {"posture_change_cleared", "recovery_observed"}:
                continue
            observation = by_observation.get(r.get("observation_id"))
            if observation:
                t = (stamps[observation["capture_seq"]] - captures[0]["captured_mono_ns"]) / 1e9
                closures.append({"source_seconds": t, "reason": r["reason"]})
        review = json.loads((original / "activity-review.json").read_text())
        phase_rows = []
        for phase in review["intervals_seconds_conservative"]:
            group = [r for r in observed if phase["from"] <= (
                stamps[r["capture_seq"]] - captures[0]["captured_mono_ns"]
            ) / 1e9 < phase["to"]]
            times = [(stamps[r["capture_seq"]] - captures[0]["captured_mono_ns"]) / 1e9
                     for r in group]
            boundaries = [phase["from"], *sorted(times), phase["to"]]
            phase_rows.append({**phase, "observations": len(group),
                               "person_present": sum(bool(r["persons"]) for r in group),
                               "max_observation_gap_seconds": max(b-a for a, b in
                                                                  zip(boundaries, boundaries[1:])),
                               "scope": "sampled visual review, not dense ground truth/recall"})
        resource_rows = [r for r in cloud if r["event"] == "process_resources"]
        evidence_end = max(r["local_mono_ns"] for r in cloud
                           if r["event"] in {"observation", "pose_shadow"})
        resource = resource_window(resource_rows, received[0]["local_mono_ns"], evidence_end)
        runtime = json.loads((folder / "runtime.json").read_text())
        mapping = rows(folder / "replay-captures.jsonl")
        stop = next(r for r in edge if r["event"] == "source_stopped")
        check = {
            "capture_complete": stop["captured"] == len(captures) and not stop["source_error"],
            "source_timestamps_preserved": len(mapping) == len(captures) and all(
                r["original_captured_mono_ns"] == stamps[r["capture_seq"]] for r in mapping
            ) and len({r["replay_captured_mono_ns"] - r["original_captured_mono_ns"]
                       for r in mapping}) == 1,
            "one_session": sum(r["event"] == "session_started" for r in cloud) == 1,
            "graviton_cool_8threads": runtime["cool_verified"]
                and runtime["runtime"]["ec2"]["architecture"] == "arm64"
                and runtime["opencv_num_threads"] == 8,
            "ledger_reconciled": ids[0] == ids[1] == ids[2] | ids[3]
                and not ids[2] & ids[3],
            "unique_frame_records": all(len(group) == len(index) for group, index in
                                        zip((sent, received, observed, dropped), ids)),
            "drained": any(r["event"] == "drain_confirmed" for r in cloud),
            "no_session_pose_error": not any(r["event"] in {"session_error", "pose_shadow_error"}
                                              for r in cloud),
            "no_network_cap": not any(r.get("network_limited") for r in sent),
            "cpu_sample_coverage": abs(resource["sample_coverage_seconds"]
                                       - resource["window_seconds"]) < .1,
            "worker_stopped": json.loads((folder / "cleanup.json").read_text())
                ["confirmed_stopped"],
        }
        rates = sorted({r["target_transmit_fps"] for r in sent})
        if mode.startswith("fixed"):
            expected_rate = int(mode.removeprefix("fixed"))
            check["fixed_rate_applied_without_agent"] = (
                rates == [expected_rate] and not actions and not incidents
            )
        else:
            check.update(
                adaptive_rates=rates == [2, 5] and sent[-1]["target_transmit_fps"] == 2,
                roi_cycle_complete=sum(r["stage"] == "cycle_complete" for r in cycles) == 1,
                actions_ack_new_evidence=bool(actions) and all(r["passed"] for r in action_audit),
                no_alert_or_pose_closure=not any(r["event"] == "test_alert"
                                                or r.get("would_close_episode") for r in cloud),
                no_seated_closure=all(any(p["activity"] == "standing"
                                         and p["from"] <= c["source_seconds"] <= p["to"]
                                         for p in review["intervals_seconds_conservative"])
                                      for c in closures),
            )
        results.append({
            **case, "folder": str(folder.relative_to(project)),
            "checks": check, "audit_passed": all(check.values()),
            "source_frames": len(captures), "source_seconds": duration,
            "sent": len(sent), "received": len(received), "processed": len(observed),
            "cloud_dropped": len(dropped), "unprocessed_unaccounted": sorted(ids[1]-ids[2]-ids[3]),
            "jpeg_bytes": sum(r["payload_bytes"] for r in sent),
            "network_limited_frames": sum(bool(r.get("network_limited")) for r in sent),
            "effective_target_fps_counts": dict(Counter(
                r["effective_target_transmit_fps"] for r in sent)),
            "network_cap_events": [r for r in edge
                                   if r["event"] == "network_transmission_cap_changed"],
            "network_rtt_ms_from_status": distribution([
                r["network_rtt_ms"] for r in edge if r["event"] == "edge_status"
                and r.get("network_rtt_ms") is not None]),
            "target_fps": rates, "effective_sent_fps_over_source": len(sent) / duration,
            "effective_processed_fps_over_source": len(observed) / duration,
            "processing_ms": distribution([r["processing_ms"] for r in observed]),
            "queue_wait_ms": distribution([r["cloud_queue_wait_ms"] for r in observed]),
            "stale_observations": sum(r.get("invalid_reason") == "processing_stale"
                                      for r in observed),
            "source_lateness_ms": distribution([r["lateness_ms"] for r in mapping]),
            "resources_cloud": resource, "actions": action_audit, "closures": closures,
            "phase_coverage": phase_rows,
            "pose_inferences": sum(r["event"] == "pose_shadow" and r.get("pose_status") == "ok"
                                   for r in cloud),
            "checkpoint_ms": distribution([r["checkpoint_ms"] for r in cloud
                                            if r["event"] == "episode_checkpoint"]),
            "annotation_sha256": hashlib.sha256((original / "activity-review.json").read_bytes())
                .hexdigest(),
        })
    comparisons = []
    for source in dict.fromkeys(r["source"] for r in results):
        fixed = next(r for r in results if r["source"] == source and r["mode"].startswith("fixed"))
        adaptive = next(r for r in results if r["source"] == source and r["mode"] == "adaptive")
        reduction = {key: 100 * (1 - adaptive[key] / fixed[key])
                     for key in ("sent", "processed", "jpeg_bytes")}
        reduction["cpu_seconds_estimate"] = 100 * (
            1 - adaptive["resources_cloud"]["cpu_seconds_estimate"]
            / fixed["resources_cloud"]["cpu_seconds_estimate"])
        comparisons.append({"source": source, "reference_mode": fixed["mode"],
                            "reduction_percent": reduction,
                            "fixed_overload": fixed["cloud_dropped"] > 0,
                            "constant_reference_valid": fixed["checks"]["no_network_cap"],
                            "comparison_scope": "observed traffic/process load; if capped, "
                                                "not a constant-rate comparison",
                            "incident_performance_equivalence": None})
    provenance = {p: hashlib.sha256((project / p).read_bytes()).hexdigest() == digest
                  for p, digest in {**plan["code_sha256"], **plan["input_sha256"],
                                    **plan.get("adaptive_log_sha256", {})}.items()}
    checks = {
        "all_conditions_audited": all(r["audit_passed"] for r in results),
        "inputs_code_unchanged": all(provenance.values()),
        "old_snapshots_unchanged": json.loads((root / "existing-state-before.json").read_text())
            == json.loads((root / "existing-state-after.json").read_text()),
        "workspace_removed": json.loads((root / "deployment.json").read_text())["cleanup"]
            ["workspace_removed"],
    }
    return {"audit_passed": all(checks.values()), "checks": checks, "runs": results,
            "comparisons": comparisons, "clinical_or_incident_accuracy_validated": False,
            "cool_speedup_isolated": False, "aws_cost_savings_measured": False,
            "limits": plan["limits"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaptive-reference", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    result = audit(args.root.resolve(), args.adaptive_reference.resolve()
                   if args.adaptive_reference else None)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["audit_passed"], "checks": result["checks"],
                      "comparisons": result["comparisons"]}, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)
