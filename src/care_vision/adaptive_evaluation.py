"""Audit actual adaptive-loop evidence against source-video annotations.

The loop keeps making its own decisions. This offline audit never merges live
and historical tracks, invents observations, or treats missing evidence as safe.
"""

from collections import Counter, defaultdict

from care_vision.check_loop import CheckFailed, check_records
from care_vision.cloud.temporal_candidate import SequenceObservation, TemporalCandidate
from care_vision.common.cool import validate_evidence
from care_vision.sequence_evaluation import score_episodes
from care_vision.temporal_evaluation import phase_at


def _unique(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key}")
    return result


def audit_adaptive_run(edge, cloud, video):
    sent = _unique([r for r in edge if r["event"] == "frame_sent"], "frame_id")
    received = _unique([r for r in cloud if r["event"] == "frame_received"], "frame_id")
    observations = [r for r in cloud if r["event"] == "observation"]
    observed = _unique(observations, "frame_id")
    by_observation = _unique(observations, "observation_id")
    requests = {
        r["command_id"]: r
        for r in cloud
        if r["event"] == "agent_action" and r["type"] == "REQUEST_RECENT_BUFFER"
    }
    if not sent or len({r["boot_id"] for r in sent.values()}) != 1:
        raise ValueError("audit requires one capture boot with transmitted frames")
    stops = [r for r in edge if r["event"] == "source_stopped"]
    if len(stops) != 1 or stops[0]["captured"] != video["frame_count"] or stops[0]["source_error"]:
        raise ValueError("source must decode every annotated frame exactly once")
    ready = [r for r in cloud if r["event"] == "server_ready"]
    if len(ready) != 1:
        raise ValueError("expected one cloud runtime record")
    runtime = ready[0]["runtime"]
    if ready[0]["phase"] == 3:
        validate_evidence(runtime.get("cool", {}), runtime.get("opencv_version", ""))
        if ready[0].get("cool_verified") is not True or runtime.get("machine") not in {
            "arm64",
            "aarch64",
        }:
            raise ValueError("missing verified COOL/ARM runtime")
        if any(r.get("runtime_id") != runtime["cool"]["runtime_id"] for r in observations):
            raise ValueError("observation runtime mismatch")
    if any(b["local_mono_ns"] < a["local_mono_ns"] for a, b in zip(cloud, cloud[1:])):
        raise ValueError("cloud log must retain processing order")
    for row in sent.values():
        if (
            type(row["capture_seq"]) is not int
            or not 1 <= row["capture_seq"] <= video["frame_count"]
        ):
            raise ValueError("capture sequence outside source")
        if row.get("evidence_kind", "live") not in {"live", "historical"}:
            raise ValueError("unsupported evidence kind")
        if row.get("evidence_kind") == "historical":
            request = requests.get(row.get("buffer_request_id"))
            if (
                request is None
                or not (
                    request["from_edge_mono_ns"]
                    <= row["captured_mono_ns"]
                    < request["to_edge_mono_ns"]
                )
                or request["session_id"] != row["session_id"]
            ):
                raise ValueError("historical evidence outside its requested session/window")
    for rows, originals in ((received, sent), (observed, received)):
        for frame_id, row in rows.items():
            if frame_id not in originals:
                raise ValueError("evidence without preceding send/receipt")
            original = originals[frame_id]
            for field in (
                "capture_seq",
                "captured_mono_ns",
                "session_id",
                "evidence_kind",
                "buffer_request_id",
                "applied_command_id",
            ):
                if row.get(field) != original.get(field):
                    raise ValueError(f"evidence metadata mismatch: {field}")
            if originals is received and row["local_mono_ns"] < original["local_mono_ns"]:
                raise ValueError("observation precedes receipt on cloud clock")

    # Preserve the established demonstration checker, including negative results.
    try:
        demo = {"passed": True, "report": check_records(edge, cloud)}
    except CheckFailed as exc:
        demo = {"passed": False, "reason": str(exc)}

    timeline, streams = [], defaultdict(list)
    latest_live = None
    for row in observations:
        index = row["capture_seq"] - 1  # VideoSource reads one source frame per capture.
        historical = row.get("evidence_kind", "live") == "historical"
        if not historical:
            if latest_live is not None and index <= latest_live:
                raise ValueError("live source order must increase")
            latest_live = index
        stream = (
            f"historical:{row['buffer_request_id']}" if historical else f"live:{row['session_id']}"
        )
        if historical and not row.get("buffer_request_id"):
            raise ValueError("historical evidence needs its own request namespace")
        streams[stream].append(row)
        timeline.append(
            {
                "frame_id": row["frame_id"],
                "source_frame_index": index,
                "source_time_seconds": index / video["fps"],
                "phase": phase_at(video, index),
                "stream": stream,
                "cloud_processed_mono_ns": row["local_mono_ns"],
                "latest_live_source_frame_at_processing": latest_live,
                "primary_track_id": row["primary_track_id"],
                "orientation": row["primary_orientation"],
                "motion": row["primary_motion_direction"],
            }
        )

    sequences = []
    for stream, rows in streams.items():
        analyzer = TemporalCandidate(retain_pending_descent=True)
        states = []
        origin = rows[0]["captured_mono_ns"]
        for row in rows:
            stale = row.get("invalid_reason") == "processing_stale"
            state = analyzer.update(
                SequenceObservation(
                    (row["captured_mono_ns"] - origin) / 1e9,
                    row["capture_seq"] - 1,
                    row["primary_track_id"],
                    None if stale else row["primary_orientation"],
                    "unknown" if stale else row["primary_motion_direction"],
                    None,
                )
            )
            states.append(state)
        episodes = [
            {**e, "end_reason": e["end_reason"] or "end_of_recording_unresolved"}
            for e in analyzer.episodes
        ]
        sequences.append({"stream": stream, **score_episodes(video, episodes), "states": states})

    phase_coverage = []
    for segment in video["segments"]:
        start, stop = segment["start_frame"], segment["end_frame_exclusive"]
        for kind in ("live", "historical"):

            def in_segment(row):
                return (
                    start <= row["capture_seq"] - 1 < stop
                    and row.get("evidence_kind", "live") == kind
                )

            phase_obs = [r for r in observations if in_segment(r)]
            indices = sorted(r["capture_seq"] - 1 for r in phase_obs)
            boundaries = [start, *indices, stop]
            phase_coverage.append(
                {
                    **segment,
                    "evidence_kind": kind,
                    "sent_frames": sum(in_segment(r) for r in sent.values()),
                    "processed_frames": len(phase_obs),
                    "bbox_available_frames": sum(
                        r["primary_orientation"] is not None for r in phase_obs
                    ),
                    "motion_counts": dict(
                        Counter(r["primary_motion_direction"] for r in phase_obs)
                    ),
                    "orientation_counts": dict(
                        Counter(r["primary_orientation"] or "unknown" for r in phase_obs)
                    ),
                    "view_counts": dict(
                        Counter(r.get("view_kind", "full_frame") for r in phase_obs)
                    ),
                    "max_source_gap_seconds_including_phase_edges": max(
                        b - a for a, b in zip(boundaries, boundaries[1:])
                    )
                    / video["fps"],
                }
            )

    actions = []
    for command in [r for r in cloud if r["event"] == "agent_action"]:
        trigger = by_observation.get(command["caused_by_observation_id"])
        if (
            trigger is None
            or trigger["frame_id"] != command["based_on_frame_id"]
            or trigger.get("evidence_kind", "live") != "live"
            or trigger["local_mono_ns"] > command["local_mono_ns"]
        ):
            raise ValueError("action lacks preceding live observation")
        ack = next(
            (
                r
                for r in cloud
                if r["event"] == "ack_received" and r["command_id"] == command["command_id"]
            ),
            None,
        )
        if ack is not None and ack["local_mono_ns"] < command["local_mono_ns"]:
            raise ValueError("ACK precedes command on cloud clock")
        actions.append(
            {
                "type": command["type"],
                "command_id": command["command_id"],
                "transmit_fps": command.get("transmit_fps"),
                "trigger_frame_index": trigger["capture_seq"] - 1,
                "trigger_phase": phase_at(video, trigger["capture_seq"] - 1),
                "trigger_source_time_seconds": (trigger["capture_seq"] - 1) / video["fps"],
                "ack_result": ack.get("result") if ack else None,
                "ack_round_trip_ms": (ack["local_mono_ns"] - command["local_mono_ns"]) / 1e6
                if ack
                else None,
            }
        )
        if command["type"] in {"SET_TRANSMIT_FPS", "REQUEST_ROI", "REQUEST_FULL_FRAME"}:
            field = (
                "applied_command_id"
                if command["type"] == "SET_TRANSMIT_FPS"
                else "applied_visual_command_id"
            )
            following = [
                r
                for r in observations
                if r.get("evidence_kind", "live") == "live"
                and r.get(field) == command["command_id"]
            ]
            if any(r["local_mono_ns"] < command["local_mono_ns"] for r in following):
                raise ValueError("updated observation precedes command")
            valid = [r for r in following if r.get("valid") is True]
            actions[-1].update(
                updated_live_observations=len(following),
                first_updated_source_frame_index=following[0]["capture_seq"] - 1
                if following
                else None,
                first_valid_new_perception_after_command_ms=(
                    valid[0]["local_mono_ns"] - command["local_mono_ns"]
                )
                / 1e6
                if valid
                else None,
            )
        if command["type"] == "REQUEST_RECENT_BUFFER":
            rows = streams.get(f"historical:{command['command_id']}", [])
            complete = next(
                (
                    r
                    for r in cloud
                    if r["event"] == "buffer_processed"
                    and r["buffer_request_id"] == command["command_id"]
                ),
                None,
            )
            actions[-1].update(
                requested_history_fps=command["history_fps"],
                selected_frames=ack.get("selected_frames") if ack else None,
                coverage=ack.get("coverage") if ack else None,
                processed_frames=len(rows),
                first_history_processing_after_request_ms=(
                    rows[0]["local_mono_ns"] - command["local_mono_ns"]
                )
                / 1e6
                if rows
                else None,
                completed_after_request_ms=(complete["local_mono_ns"] - command["local_mono_ns"])
                / 1e6
                if complete
                else None,
                complete=complete is not None,
                latest_live_source_frame_at_first_history_processing=next(
                    (
                        r["latest_live_source_frame_at_processing"]
                        for r in timeline
                        if r["stream"] == f"historical:{command['command_id']}"
                    ),
                    None,
                ),
            )
    return {
        "evaluation": "phase7f-real-adaptive-loop-audit",
        "sessions": len({r["session_id"] for r in sent.values()}),
        "video_id": video["video_id"],
        "runtime": runtime,
        "original_demo_check": demo,
        "counts": {
            "sent": len(sent),
            "received": len(received),
            "processed": len(observed),
            "sent_without_receipt": len(sent.keys() - received.keys()),
            "received_without_observation": len(received.keys() - observed.keys()),
        },
        "payload_bytes_by_kind": {
            kind: sum(
                r["payload_bytes"] for r in sent.values() if r.get("evidence_kind", "live") == kind
            )
            for kind in ("live", "historical")
        },
        "actions": actions,
        "phase_coverage": phase_coverage,
        "sequences": sequences,
        "processing_timeline": timeline,
        "recorded_failures": [
            r
            for r in edge + cloud
            if r["event"]
            in {"connection_error", "buffer_transfer_aborted", "buffer_incomplete", "session_error"}
        ],
        "fall_classification_enabled": False,
        "limitations": [
            "One development clip, provisional annotations; not held-out incident validation.",
            "Sequence scoring is offline and never affects the running controller.",
            "Historical and live namespaces are separate; their tracks cannot be concatenated.",
            "Capture sequence maps to source index; capture clock controls sequence dwell.",
            "Durations use same-host clocks; source-video times are not end-to-end latency.",
            "Different resolution/ROI/sampling from Phase 7D; no isolated FPS or model comparison.",
            "Payload bytes exclude transport overhead and do not measure privacy or cost savings.",
        ],
    }
