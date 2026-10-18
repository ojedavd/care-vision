"""Validate causal evidence and measured rates, not just printed desired settings."""

import argparse
import json
import statistics
from pathlib import Path

from care_vision.common.cool import validate_evidence


class CheckFailed(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise CheckFailed(reason)


def read_log(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def rate_windows(records, target, duration=5):
    """Count actual events in complete windows on this process's monotonic clock."""
    require(len(records) >= 2, f"missing frames at {target} FPS")
    times = [record["local_mono_ns"] / 1e9 for record in records]
    require(all(b > a for a, b in zip(times, times[1:])), "non-increasing event times")
    start = times[0] + 0.25
    rates = []
    while start + duration <= times[-1]:
        rate = sum(start <= t < start + duration for t in times) / duration
        require(
            abs(rate - target) <= target * 0.1 + 1e-6,
            f"achieved {rate:.2f} FPS; requested {target} FPS",
        )
        rates.append(round(rate, 3))
        start += duration
    require(rates, f"need a complete {duration}s stable window at {target} FPS")
    # A large replay burst cannot stand in for steady live sampling.
    for index, t in enumerate(times):
        count = sum(t <= later < t + 1 for later in times[index:])
        require(count <= target + 2, f"burst detected at {target} FPS")
    return rates


def check_records(edge, cloud, *, escalation_fps=20, capture_fps=30):
    require(escalation_fps in {5, 20}, "unsupported escalation FPS")
    require(0 < capture_fps <= 30, "invalid capture FPS")
    ready = [r for r in cloud if r["event"] == "server_ready"]
    require(len(ready) == 1, "expected exactly one cloud startup record")
    environment = ready[0].get("runtime", {})
    phase = ready[0].get("phase", 1)
    require(phase in {1, 2, 3}, "unsupported execution phase")
    if phase in {2, 3}:
        require(
            environment.get("system") == "Linux"
            and environment.get("machine") in {"aarch64", "arm64"}
            and environment.get("ec2", {}).get("architecture") == "arm64",
            "remote phase requires EC2 ARM64 runtime evidence",
        )
    if phase == 3:
        try:
            validate_evidence(environment.get("cool", {}), environment.get("opencv_version", ""))
        except RuntimeError as exc:
            raise CheckFailed(str(exc)) from exc
        require(ready[0].get("cool_verified") is True, "COOL startup verification missing")
    all_sent = [r for r in edge if r["event"] == "frame_sent"]
    all_received = [r for r in cloud if r["event"] == "frame_received"]
    all_observations = [r for r in cloud if r["event"] == "observation"]

    def is_live(r):
        return r.get("evidence_kind", "live") == "live"

    sent = [r for r in all_sent if is_live(r)]
    received = [r for r in all_received if is_live(r)]
    observations = [r for r in all_observations if is_live(r)]
    actions = [r for r in cloud if r["event"] == "agent_action" and r["type"] == "SET_TRANSMIT_FPS"]
    require(
        [r["transmit_fps"] for r in actions] == [escalation_fps, 2],
        f"expected exactly 2 → {escalation_fps} → 2",
    )
    require(len({r["session_id"] for r in sent}) == 1, "demo must finish in one healthy session")
    sent_by_id = {r["frame_id"]: r for r in all_sent}
    received_by_id = {r["frame_id"]: r for r in all_received}
    obs_by_id = {r["observation_id"]: r for r in observations}
    require(len(sent_by_id) == len(all_sent), "duplicate transmitted frame")
    require(len(received_by_id) == len(all_received), "duplicate received frame")
    require(set(received_by_id) <= set(sent_by_id), "received frame has no send evidence")
    for record in all_received:
        original = sent_by_id[record["frame_id"]]
        for key in (
            "session_id",
            "boot_id",
            "capture_seq",
            "captured_mono_ns",
            "applied_command_id",
            "policy_seq",
            "target_transmit_fps",
        ):
            require(original[key] == record[key], f"frame metadata mismatch: {key}")
        for key in (
            "evidence_kind",
            "buffer_request_id",
            "history_index",
            "history_fps",
            "view_kind",
            "source_width",
            "source_height",
            "roi_x",
            "roi_y",
            "roi_width",
            "roi_height",
            "visual_policy_seq",
            "applied_visual_command_id",
        ):
            require(original.get(key) == record.get(key), f"frame metadata mismatch: {key}")
    require(
        all(b["capture_seq"] > a["capture_seq"] for a, b in zip(sent, sent[1:])),
        "capture sequence must increase",
    )
    for observation in all_observations:
        if phase == 3:
            require(
                observation.get("runtime_id") == environment["cool"].get("runtime_id")
                and observation.get("runtime_id") is not None,
                "observation is not linked to the verified COOL runtime",
            )
        require(observation["frame_id"] in received_by_id, "observation without received frame")
        frame = received_by_id[observation["frame_id"]]
        require(
            observation["local_mono_ns"] >= frame["local_mono_ns"], "observation predates receipt"
        )
        require(
            observation.get("evidence_kind", "live") == frame.get("evidence_kind", "live")
            and observation.get("buffer_request_id") == frame.get("buffer_request_id"),
            "observation evidence kind/request mismatch",
        )
        require(
            observation["applied_command_id"] == frame["applied_command_id"],
            "observation policy mismatch",
        )

    command_ids = [r["command_id"] for r in actions]
    command_timings = []
    for index, action in enumerate(actions):
        trigger = obs_by_id.get(action["caused_by_observation_id"])
        require(trigger is not None and trigger["valid"], "command without valid observation")
        require(trigger["frame_id"] == action["based_on_frame_id"], "wrong command trigger frame")
        require(trigger["local_mono_ns"] <= action["local_mono_ns"], "command predates observation")
        require(
            trigger["applied_command_id"] == (None if index == 0 else command_ids[0]),
            "return command not caused by escalated evidence",
        )
        acks = [
            r
            for r in edge
            if r["event"] == "command_ack"
            and r["command_id"] == action["command_id"]
            and r["result"] == "applied"
        ]
        require(len(acks) == 1, "command must be applied exactly once")
        ack = acks[0]
        require(ack["active_transmit_fps"] == action["transmit_fps"], "ACK rate mismatch")
        require(ack["active_policy_seq"] == action["command_seq"], "ACK sequence mismatch")
        require(
            any(
                r["event"] == "ack_received"
                and r["command_id"] == action["command_id"]
                and r["result"] in {"applied", "duplicate"}
                for r in cloud
            ),
            "cloud never received ACK",
        )
        following = [r for r in sent if r["applied_command_id"] == action["command_id"]]
        require(following, "ACK without new frames")
        for frame in following:
            require(
                frame["capture_seq"] >= ack["effective_from_capture_seq"],
                "post-command frame predates command application",
            )
            require(
                frame["captured_mono_ns"] >= ack["applied_at_edge_mono_ns"],
                "post-command capture timestamp predates application",
            )
        new_evidence = [
            r
            for r in observations
            if r["valid"] and r["applied_command_id"] == action["command_id"]
        ]
        require(len(new_evidence) >= 10, "missing second perception cycle")
        # Both endpoints of these durations are logged by the cloud process.
        # Never subtract an edge clock from an AWS clock.
        ack_received = next(
            r
            for r in cloud
            if r["event"] == "ack_received"
            and r["command_id"] == action["command_id"]
            and r["result"] in {"applied", "duplicate"}
        )
        command_timings.append(
            {
                "command_id": action["command_id"],
                "ack_round_trip_ms": round(
                    (ack_received["local_mono_ns"] - action["local_mono_ns"]) / 1e6, 3
                ),
                "new_perception_ms": round(
                    (new_evidence[0]["local_mono_ns"] - action["local_mono_ns"]) / 1e6, 3
                ),
            }
        )

    rates = []
    for command_id, target in ((None, 2), (command_ids[0], escalation_fps), (command_ids[1], 2)):
        outbound = [r for r in sent if r["applied_command_id"] == command_id]
        inbound = [r for r in received if r["applied_command_id"] == command_id]
        require(all(r["target_transmit_fps"] == target for r in outbound), "wrong target metadata")
        rates.append(
            {
                "command_id": command_id,
                "target_fps": target,
                "transmit_fps_windows": rate_windows(outbound, target),
                "receive_fps_windows": rate_windows(inbound, target),
            }
        )
    capture_rates = [
        r["capture_fps"]
        for r in edge
        if r["event"] == "edge_status"
        and r["source_state"] == "running"
        and 0.9 <= r["window_seconds"] <= 1.5
        and r["local_mono_ns"] - r["window_seconds"] * 1e9 >= sent[0]["local_mono_ns"]
    ]
    require(len(capture_rates) >= 10, "missing independent capture measurements")
    require(
        all(capture_fps * 0.9 <= rate <= capture_fps * 1.1 for rate in capture_rates),
        f"capture did not remain near {capture_fps} FPS",
    )
    require(
        not any(r["event"] in {"session_error", "frame_dropped"} for r in cloud),
        "healthy-loop demo had a server error or dropped evidence",
    )
    history_report = check_history(
        edge,
        cloud,
        all_sent,
        all_received,
        all_observations,
        ready[0].get("temporal_buffer_enabled", False)
        and not (escalation_fps == 5 and ready[0].get("experimental_s_only_5fps", False)),
    )
    return {
        "passed": True,
        "temporal_buffer": history_report,
        "phase": phase,
        "cool_verified": phase == 3,
        "cloud_runtime": environment,
        "capture_fps_median": round(statistics.median(capture_rates), 3),
        "frames_sent": len(all_sent),
        "live_frames_sent": len(sent),
        "frames_received": len(all_received),
        "observations": len(all_observations),
        "rate_windows": rates,
        "command_chain": command_ids,
        "command_timings": command_timings,
        "processing_ms_median": round(
            statistics.median(r["processing_ms"] for r in observations), 3
        ),
    }


def check_history(edge, cloud, sent, received, observations, required):
    requests = [
        r for r in cloud if r["event"] == "agent_action" and r["type"] == "REQUEST_RECENT_BUFFER"
    ]
    historical = [r for r in sent if r.get("evidence_kind") == "historical"]
    if not required and not requests and not historical:
        return {"verified": False}
    pre_event = [r for r in requests if r.get("reason") == "recover_pre_event_evidence"]
    require(len(pre_event) == 1, "expected one pre-event buffer request")
    recovery = [r for r in requests if r not in pre_event]
    require(
        all(r.get("reason") == "recover_offline_evidence" for r in recovery),
        "unrecognized historical request",
    )
    if recovery:
        # Validate every additional request and final drain, not just the demo window.
        from care_vision.check_history_delivery import check_history_delivery

        check_history_delivery(edge, cloud)
    request = pre_event[0]
    rid = request["command_id"]
    historical = [r for r in historical if r.get("buffer_request_id") == rid]
    trigger = next(
        (r for r in observations if r["observation_id"] == request["caused_by_observation_id"]),
        None,
    )
    require(
        trigger is not None and trigger["valid"] and trigger.get("evidence_kind", "live") == "live",
        "buffer request lacks live trigger",
    )
    require(
        request["based_on_frame_id"] == trigger["frame_id"]
        and request["to_edge_mono_ns"] == trigger["captured_mono_ns"]
        and request["to_edge_mono_ns"] - request["from_edge_mono_ns"] == 5_000_000_000,
        "buffer request is not anchored at T minus five seconds",
    )
    require(request["local_mono_ns"] >= trigger["local_mono_ns"], "buffer request predates trigger")
    acks = [
        r
        for r in edge
        if r["event"] == "command_ack" and r["command_id"] == rid and r["result"] == "applied"
    ]
    require(len(acks) == 1, "buffer request must apply exactly once")
    ack = acks[0]
    require(
        any(
            r["event"] == "ack_received"
            and r["command_id"] == rid
            and r["result"] in {"applied", "duplicate"}
            for r in cloud
        ),
        "cloud never received buffer ACK",
    )
    require(ack["coverage"] == "full", "demo needs full pre-event coverage")
    require(len(historical) == ack["selected_frames"] > 0, "missing historical send evidence")
    require(
        sum(r["payload_bytes"] for r in historical) == ack["selected_bytes"],
        "historical byte count mismatch",
    )
    require(
        95 <= len(historical) + ack["deduplicated_frames"] <= 101,
        "five-second historical sample is incomplete",
    )
    for index, record in enumerate(historical):
        require(
            record["buffer_request_id"] == rid and record["history_index"] == index,
            "historical request or index mismatch",
        )
        require(
            request["from_edge_mono_ns"] <= record["captured_mono_ns"] < request["to_edge_mono_ns"],
            "historical evidence outside requested window",
        )
        require(
            record["policy_seq"] == 0 and record["applied_command_id"] is None,
            "historical frame claims live policy",
        )
        require(record["local_mono_ns"] >= ack["local_mono_ns"], "history sent before ACK")
    require(
        all(
            b["capture_seq"] > a["capture_seq"] and b["captured_mono_ns"] > a["captured_mono_ns"]
            for a, b in zip(historical, historical[1:])
        ),
        "historical order mismatch",
    )
    require(
        historical[0]["captured_mono_ns"] - request["from_edge_mono_ns"] < 200_000_000,
        "beginning of requested pre-event window missing",
    )
    ids = {r["frame_id"] for r in historical}
    inbound = [r for r in received if r.get("buffer_request_id") == rid]
    evidence = [r for r in observations if r.get("buffer_request_id") == rid]
    require({r["frame_id"] for r in inbound} == ids, "missing historical receipt")
    require(
        len(evidence) == len(ids) and {r["frame_id"] for r in evidence} == ids,
        "missing historical perception",
    )
    require(any(r["valid"] for r in evidence), "no valid historical temporal observation")
    require(
        all(not r["valid"] or r["reference_frame_id"] in ids for r in evidence),
        "historical perception mixed with live timeline",
    )
    complete = [
        r for r in cloud if r["event"] == "buffer_processed" and r["buffer_request_id"] == rid
    ]
    require(
        len(complete) == 1 and complete[0]["frames_processed"] == len(ids),
        "historical processing never completed",
    )
    require(not any(r["event"] == "buffer_incomplete" for r in cloud), "incomplete buffer transfer")
    statuses = [r for r in edge if r["event"] == "edge_status"]
    require(
        statuses
        and all(
            0 <= r["buffer_bytes"] <= 32 * 1024 * 1024 and 0 <= r["buffer_frames"] <= 300
            for r in statuses
        ),
        "ring buffer exceeded memory/frame limits",
    )
    return {
        "verified": True,
        "request_id": rid,
        "coverage": ack["coverage"],
        "frames": len(ids),
        "deduplicated_frames": ack["deduplicated_frames"],
        "payload_bytes": ack["selected_bytes"],
        "requested_seconds": 5,
        "earliest_pre_event_seconds": round(
            (request["to_edge_mono_ns"] - historical[0]["captured_mono_ns"]) / 1e9, 3
        ),
        "buffer_peak_bytes": max(r["buffer_bytes"] for r in statuses),
        "buffer_peak_frames": max(r["buffer_frames"] for r in statuses),
    }


def main():
    parser = argparse.ArgumentParser(description="Check CARE-Vision local causal loop logs")
    parser.add_argument("--edge-log", required=True)
    parser.add_argument("--cloud-log", required=True)
    parser.add_argument("--escalation-fps", type=int, choices=[5, 20], default=20)
    parser.add_argument("--capture-fps", type=float, default=30)
    args = parser.parse_args()
    try:
        report = check_records(
            read_log(args.edge_log),
            read_log(args.cloud_log),
            escalation_fps=args.escalation_fps,
            capture_fps=args.capture_fps,
        )
    except (CheckFailed, KeyError, ValueError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
