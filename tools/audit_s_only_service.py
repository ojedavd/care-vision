"""Audit one experimental S-only 2→5→2 WebSocket replay."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(folder):
    cloud, edge = load(folder / "cloud.jsonl"), load(folder / "edge.jsonl")
    ready = next(row for row in cloud if row["event"] == "server_ready")
    assert ready["cool_verified"] and ready["opencv_num_threads"] == 8
    assert ready["runtime"]["ec2"]["architecture"] == "arm64"
    sent = [row for row in edge if row["event"] == "frame_sent"]
    received = [row for row in cloud if row["event"] == "frame_received"]
    observed = [row for row in cloud if row["event"] == "observation"]
    assert sent and len(sent) == len(received) == len(observed)
    assert [row["frame_id"] for row in sent] == [row["frame_id"] for row in received]
    assert [row["frame_id"] for row in sent] == [row["frame_id"] for row in observed]
    assert not [row for row in cloud if row["event"] in {"frame_dropped", "session_error"}]
    assert not [row for row in observed if row.get("invalid_reason") == "processing_stale"]
    assert all((row["width"], row["height"]) == (1280, 720) for row in observed)
    assert all("s-2022nov" in row["person_detector_backend"] for row in observed)
    assert all(row["person_detection_status"] == "ok" for row in observed)
    actions = [row for row in cloud if row["event"] == "agent_action"]
    edge_acks = [row for row in edge if row["event"] == "command_ack"]
    cloud_acks = [row for row in cloud if row["event"] == "ack_received"]
    assert actions and len(actions) == len(edge_acks) == len(cloud_acks)
    assert all(row["type"] == "SET_TRANSMIT_FPS" for row in actions)
    assert [row["transmit_fps"] for row in actions] == [
        5 if i % 2 == 0 else 2 for i in range(len(actions))
    ]
    assert [row["command_id"] for row in actions] == [row["command_id"] for row in edge_acks]
    assert [row["command_id"] for row in actions] == [row["command_id"] for row in cloud_acks]
    assert all(row["result"] == "applied" for row in edge_acks + cloud_acks)
    acks = {row["command_id"]: row for row in edge_acks}
    for row in observed:
        command = row.get("applied_command_id")
        if command:
            assert row["capture_seq"] >= acks[command]["effective_from_capture_seq"]
    assert any(row["event"] == "drain_confirmed" for row in cloud)
    assert any(row["event"] == "drain_complete" for row in edge)
    outcomes = [row for row in cloud if row["event"] == "loss_verification_outcome"]
    assert len(outcomes) == len(actions) // 2
    assert all(row["incident_status"] == "unresolved" for row in outcomes)
    rates = Counter(row["target_transmit_fps"] for row in observed)
    origin = observed[0]["captured_mono_ns"]
    events = []
    for action in actions:
        source = next(row for row in observed if row["frame_id"] == action["based_on_frame_id"])
        events.append(
            {
                "source_seconds": round((source["captured_mono_ns"] - origin) / 1e9, 3),
                "rate": action["transmit_fps"],
                "reason": action["reason"],
            }
        )
    return {
        "passed": True,
        "captured": max(row.get("captured", 0) for row in edge if row["event"] == "edge_status"),
        "sent": len(sent),
        "received": len(received),
        "processed": len(observed),
        "dropped": 0,
        "rate_counts": dict(rates),
        "completed_loops": len(actions) // 2,
        "verification_open_at_end": len(actions) % 2 == 1,
        "commands": len(actions),
        "acks": len(edge_acks),
        "events": events,
        "posture_outcomes": Counter(row["outcome"] for row in outcomes),
        "processing_median_ms": {
            str(rate): round(
                statistics.median(
                    row["processing_ms"] for row in observed if row["target_transmit_fps"] == rate
                ),
                3,
            )
            for rate in rates
        },
        "cool_verified": True,
        "instance_type": ready["runtime"]["ec2"]["instanceType"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    folder = parser.parse_args().folder
    result = audit(folder)
    (folder / "s-only-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
