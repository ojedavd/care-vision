"""Correlate Phase 4 live-frame timing across Edge and cloud logs.

Monotonic clocks from different hosts are never subtracted directly. Transport
variation is the change in receive spacing relative to send spacing, anchored at
the first live frame at the selected FPS. This is relative variation, not
absolute one-way network latency; it also includes endpoint scheduling effects.
"""

import argparse
import csv
import json
from pathlib import Path


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def event_map(records: list[dict], event: str) -> dict[str, dict]:
    return {
        record["frame_id"]: record
        for record in records
        if record["event"] == event and record.get("evidence_kind", "live") == "live"
    }


def maximum_one_second_window(records: list[dict]) -> dict:
    best_count, best_start, best_end = 0, None, None
    for index, record in enumerate(records):
        start = record["local_mono_ns"]
        window = [item for item in records[index:] if item["local_mono_ns"] < start + 1_000_000_000]
        if len(window) > best_count:
            best_count = len(window)
            best_start = record["frame_id"]
            best_end = window[-1]["frame_id"]
    return {"frames": best_count, "first_frame_id": best_start, "last_frame_id": best_end}


def replacement_map(cloud: list[dict]) -> dict[str, str]:
    """A drop event follows the receive event that replaced the pending frame."""
    incoming = None
    replacements = {}
    for record in cloud:
        if record["event"] == "frame_received" and record.get("evidence_kind", "live") == "live":
            incoming = record["frame_id"]
        elif record["event"] == "frame_dropped":
            replacements[record["frame_id"]] = incoming
    return replacements


def analyze(edge: list[dict], cloud: list[dict], target_fps=20) -> tuple[list[dict], dict]:
    sent = event_map(edge, "frame_sent")
    received = event_map(cloud, "frame_received")
    observed = event_map(cloud, "observation")
    replacements = replacement_map(cloud)
    relevant = sorted(
        (record for record in sent.values() if record["target_transmit_fps"] == target_fps),
        key=lambda record: record["capture_seq"],
    )
    if not relevant:
        raise ValueError(f"no {target_fps} FPS live frames")
    missing = set(sent) - set(received)
    if missing:
        raise ValueError(f"sent frames missing at cloud: {sorted(missing)}")

    first_send = relevant[0]["local_mono_ns"]
    first_receive = received[relevant[0]["frame_id"]]["local_mono_ns"]
    rows = []
    previous_send = previous_receive = None
    for record in relevant:
        frame_id = record["frame_id"]
        receive = received[frame_id]
        observation = observed.get(frame_id)
        send_ns = record["local_mono_ns"]
        receive_ns = receive["local_mono_ns"]
        rows.append(
            {
                "frame_id": frame_id,
                "capture_seq": record["capture_seq"],
                "captured_mono_ns": record["captured_mono_ns"],
                "sampled_mono_ns": record["sampled_mono_ns"],
                "edge_sent_mono_ns": send_ns,
                "cloud_received_mono_ns": receive_ns,
                "cloud_observed_mono_ns": observation["local_mono_ns"] if observation else None,
                "capture_to_sample_ms": round(
                    (record["sampled_mono_ns"] - record["captured_mono_ns"]) / 1e6, 6
                ),
                "sample_to_send_ms": round((send_ns - record["sampled_mono_ns"]) / 1e6, 6),
                "edge_send_gap_ms": (
                    round((send_ns - previous_send) / 1e6, 6) if previous_send else None
                ),
                "cloud_receive_gap_ms": (
                    round((receive_ns - previous_receive) / 1e6, 6) if previous_receive else None
                ),
                "relative_transport_drift_ms": round(
                    ((receive_ns - first_receive) - (send_ns - first_send)) / 1e6, 6
                ),
                "receive_to_observation_ms": (
                    round((observation["local_mono_ns"] - receive_ns) / 1e6, 6)
                    if observation
                    else None
                ),
                "payload_bytes": record["payload_bytes"],
                "disposition": (
                    "queue_replaced" if frame_id in replacements
                    else "observed" if observation else "unprocessed"
                ),
                "replaced_by_frame_id": replacements.get(frame_id),
            }
        )
        previous_send, previous_receive = send_ns, receive_ns

    edge_window = maximum_one_second_window(relevant)
    cloud_relevant = [received[record["frame_id"]] for record in relevant]
    cloud_window = maximum_one_second_window(cloud_relevant)
    dropped = [row for row in rows if row["disposition"] == "queue_replaced"]
    summary = {
        f"live_{target_fps}fps_frames_sent": len(relevant),
        f"live_{target_fps}fps_frames_received": len(cloud_relevant),
        f"live_{target_fps}fps_frames_observed": sum(
            row["disposition"] == "observed" for row in rows
        ),
        "unprocessed_without_drop_record": sum(
            row["disposition"] == "unprocessed" for row in rows
        ),
        "transport_frame_loss": len(missing),
        "cloud_queue_replacements": len(dropped),
        "edge_peak_one_second_window": edge_window,
        "cloud_peak_one_second_window": cloud_window,
        "relative_transport_drift_ms": {
            "minimum": min(row["relative_transport_drift_ms"] for row in rows),
            "maximum": max(row["relative_transport_drift_ms"] for row in rows),
        },
        "dropped_frames": [
            {
                "frame_id": row["frame_id"],
                "capture_seq": row["capture_seq"],
                "replaced_by_frame_id": row["replaced_by_frame_id"],
                "capture_to_sample_ms": row["capture_to_sample_ms"],
                "sample_to_send_ms": row["sample_to_send_ms"],
                "edge_send_gap_ms": row["edge_send_gap_ms"],
                "cloud_receive_gap_ms": row["cloud_receive_gap_ms"],
                "relative_transport_drift_ms": row["relative_transport_drift_ms"],
            }
            for row in dropped
        ],
    }
    return rows, summary


def write_analysis(run: Path, output: Path, target_fps=20) -> dict:
    rows, summary = analyze(
        read_log(run / "edge.jsonl"), read_log(run / "cloud.jsonl"), target_fps
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"live-{target_fps}fps-frame-correlation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "transport-analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, choices=(2, 20), default=20)
    args = parser.parse_args()
    output = args.output or args.run / "transport-analysis"
    print(json.dumps(write_analysis(args.run, output, args.fps), indent=2))


if __name__ == "__main__":
    main()
