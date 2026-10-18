"""Evidence-only check-in summary. All latency arithmetic stays on one clock."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from care_vision.check_history_delivery import check_history_delivery
from care_vision.check_loop import CheckFailed, check_records, read_log, require
from care_vision.check_resolution import check_resolution_records
from care_vision.check_roi import check_roi_records
from care_vision.check_states import check_state_records


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "max": values[-1],
    }


def resource_summary(rows):
    samples = [r for r in rows if r["event"] == "process_resources"]
    duration = sum(r["interval_seconds"] for r in samples)
    return {
        "samples": len(samples),
        "sampled_seconds": duration,
        "cpu_percent_one_core_mean": sum(
            r["cpu_percent_one_core"] * r["interval_seconds"] for r in samples
        ) / duration if duration else None,
        "cpu_percent_one_core": distribution([r["cpu_percent_one_core"] for r in samples]),
        "rss_bytes": distribution([r["rss_bytes"] for r in samples if r["rss_bytes"] is not None]),
        "peak_rss_bytes": max((r["peak_rss_bytes"] for r in samples), default=None),
    }


def summarize(edge, cloud):
    sent = [r for r in edge if r["event"] == "frame_sent"]
    obs = [r for r in cloud if r["event"] == "observation"]
    live = [r for r in obs if r.get("evidence_kind", "live") == "live"]
    return {
        "jpeg_payload_bytes": sum(r["payload_bytes"] for r in sent),
        "live_jpeg_payload_bytes": sum(
            r["payload_bytes"] for r in sent if r.get("evidence_kind", "live") == "live"
        ),
        "historical_jpeg_payload_bytes": sum(
            r["payload_bytes"] for r in sent if r.get("evidence_kind") == "historical"
        ),
        "live_processing_ms": distribution([r["processing_ms"] for r in live]),
        "historical_processing_ms": distribution([
            r["processing_ms"] for r in obs if r.get("evidence_kind") == "historical"
        ]),
        "live_queue_wait_ms": distribution([
            r["cloud_queue_wait_ms"] for r in live if "cloud_queue_wait_ms" in r
        ]),
        "edge_resources": resource_summary(edge),
        "cloud_resources": resource_summary(cloud),
    }


def replay_rows(cloud):
    """Use AWS availability order; capture_seq only selects the matching source image."""
    first = next(r for r in cloud if r["event"] == "frame_received")["local_mono_ns"]
    fields = {
        "observation": ["capture_seq", "evidence_kind", "frame_id", "processing_ms",
                        "width", "height", "source_width", "source_height", "source_roi",
                        "primary_source_bbox", "primary_person_confidence", "motion_score",
                        "primary_motion_direction", "primary_orientation", "view_kind",
                        "applied_command_id", "applied_visual_command_id"],
        "agent_state_transition": ["to_state", "reason"],
        "agent_action": ["type", "command_id", "reason", "transmit_fps"],
        "ack_received": ["command_id", "command_type", "result"],
        "status_received": ["capture_fps", "transmit_fps", "target_transmit_fps",
                            "effective_target_transmit_fps", "network_quality", "network_rtt_ms",
                            "pipeline_quality", "cloud_feedback_state", "cloud_live_dropped",
                            "cloud_live_stale", "cloud_queue_depth", "cloud_feedback_age_ms",
                            "bytes_sent", "source_state"],
        "process_resources": ["cpu_percent_one_core", "rss_bytes"],
        "buffer_processed": ["frames_processed"],
        "drain_confirmed": [],
    }
    return [
        {"t": (r["local_mono_ns"] - first) / 1e9, "event": r["event"],
         **{k: r.get(k) for k in fields[r["event"]]}}
        for r in cloud if r["event"] in fields and r["local_mono_ns"] >= first
    ]


def validate_bundle(run):
    run = Path(run)
    edge, cloud = read_log(run / "edge.jsonl"), read_log(run / "cloud.jsonl")
    checks = {}
    for name, check in (
        ("loop", lambda: check_records(edge, cloud)),
        ("roi", lambda: check_roi_records(edge, cloud)),
        ("resolution", lambda: check_resolution_records(edge, cloud)),
        ("states", lambda: check_state_records(cloud)),
        ("history", lambda: check_history_delivery(edge, cloud)),
    ):
        try:
            checks[name] = {"passed": True, "report": check()}
        except CheckFailed as exc:
            checks[name] = {"passed": False, "error": str(exc)}
    deployment = json.loads((run / "deployment.json").read_text())
    for name, field in (("source.tar.gz", "source_archive_sha256"),
                        ("source-manifest.json", "source_manifest_sha256"),
                        ("demo.avi", "input_video_sha256")):
        require(hashlib.sha256((run / name).read_bytes()).hexdigest() == deployment.get(field),
                f"bundle hash mismatch: {name}")
    metrics = summarize(edge, cloud)
    checks["resource_samples"] = {"passed": all(
        metrics[f"{side}_resources"]["samples"] >= 2 for side in ("edge", "cloud")
    )}
    checks["aws_cool"] = {"passed": bool(
        checks["loop"]["passed"] and checks["loop"]["report"]["cool_verified"]
    )}
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "metrics": metrics,
        "source_kind": deployment.get("source_kind", "video_file"),
        "scope": "Architecture demo; incident classification and alerts disabled",
        "limitations": [
            "JPEG payload bytes exclude WebSocket, TCP and SSH overhead; not billed network usage.",
            "CPU is process-only: 100% equals one core; startup and drain are included.",
            "RSS is sampled on Linux; peak RSS is a process lifetime high-water mark.",
            "Replay uses cloud monotonic availability time. Edge clocks are never subtracted.",
            "No fall accuracy, savings, cost or COOL speedup is established by this run.",
        ],
    }, replay_rows(cloud)
