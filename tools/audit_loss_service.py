"""Audit the experimental WebSocket 2→5→2 loss-verification loop."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def percentile(values, fraction):
    ordered = sorted(values)
    return round(ordered[int(fraction * (len(ordered) - 1))], 3)


def audit(folder):
    cloud, edge = records(folder / "cloud.jsonl"), records(folder / "edge.jsonl")
    deployment = json.loads((folder / "deployment.json").read_text())
    source = (
        "live RTSP camera"
        if deployment["source_kind"] == "rtsp_camera"
        else "real-time recorded-source"
    )
    ready = next(row for row in cloud if row["event"] == "server_ready")
    assert ready["cool_verified"] and ready["experimental_loss_5fps"]
    assert ready["opencv_num_threads"] == 8
    assert ready["runtime"]["ec2"]["architecture"] == "arm64"
    sent = [row for row in edge if row["event"] == "frame_sent"]
    received = [row for row in cloud if row["event"] == "frame_received"]
    observed = [row for row in cloud if row["event"] == "observation"]
    assert len(sent) == len(received) == len(observed)
    assert [row["frame_id"] for row in sent] == [row["frame_id"] for row in received]
    assert [row["frame_id"] for row in sent] == [row["frame_id"] for row in observed]
    assert not [row for row in cloud if row["event"] == "frame_dropped"]
    assert not [row for row in observed if row.get("invalid_reason") == "processing_stale"]
    actions = [row for row in cloud if row["event"] == "agent_action"]
    edge_acks = [row for row in edge if row["event"] == "command_ack"]
    cloud_acks = [row for row in cloud if row["event"] == "ack_received"]
    assert len(actions) == len(edge_acks) == len(cloud_acks)
    assert [row["command_id"] for row in actions] == [row["command_id"] for row in edge_acks]
    assert [row["command_id"] for row in actions] == [row["command_id"] for row in cloud_acks]
    assert all(row["result"] == "applied" for row in edge_acks + cloud_acks)
    acks = {row["command_id"]: row for row in edge_acks}
    by_id = {row["command_id"]: row for row in actions}
    assert len(by_id) == len(actions)
    assert [row["command_seq"] for row in actions] == list(range(1, len(actions) + 1))
    rates = [row for row in actions if row["type"] == "SET_TRANSMIT_FPS"]
    assert [row["transmit_fps"] for row in rates] == [
        5 if index % 2 == 0 else 2 for index in range(len(rates))
    ]
    assert len(actions) == len(rates) * 2
    assert all(actions[i]["type"] == "SET_TRANSMIT_FPS" for i in range(0, len(actions), 2))
    assert all(actions[i]["type"] == "REQUEST_FULL_FRAME" for i in range(1, len(actions), 2))
    for rate, visual in zip(actions[::2], actions[1::2]):
        assert visual["output_width"] == (1280 if rate["transmit_fps"] == 5 else 320)
        assert visual["output_height"] == (720 if rate["transmit_fps"] == 5 else 180)
        assert rate["based_on_frame_id"] == visual["based_on_frame_id"]
    for row in observed:
        rate = row["target_transmit_fps"]
        expected = (1280, 720) if rate == 5 else (320, 180)
        assert (row["width"], row["height"]) == expected
        assert row["person_detection_status"] == "ok"
        assert ("s-2022nov" in row["person_detector_backend"]) == (rate == 5)
        for command_id in (row.get("applied_command_id"), row.get("applied_visual_command_id")):
            if command_id is not None:
                assert row["capture_seq"] >= acks[command_id]["effective_from_capture_seq"]
                assert by_id[command_id]["command_seq"] <= max(
                    row["policy_seq"], row["visual_policy_seq"]
                )
    assert not [row for row in cloud if row["event"] == "session_error"]
    assert any(row["event"] == "drain_confirmed" for row in cloud)
    assert any(row["event"] == "drain_complete" for row in edge)
    rate_counts = Counter(row["target_transmit_fps"] for row in observed)
    s = [row for row in observed if row["target_transmit_fps"] == 5]
    nano = [row for row in observed if row["target_transmit_fps"] == 2]
    origin = observed[0]["captured_mono_ns"]
    episodes = []
    for start, end in zip(rates[::2], rates[1::2]):
        start_obs = next(row for row in observed if row["frame_id"] == start["based_on_frame_id"])
        end_obs = next(row for row in observed if row["frame_id"] == end["based_on_frame_id"])
        window = [
            row
            for row in s
            if start_obs["captured_mono_ns"]
            < row["captured_mono_ns"]
            <= end_obs["captured_mono_ns"]
        ]
        assert window
        episodes.append(
            {
                "start_source_seconds": round((start_obs["captured_mono_ns"] - origin) / 1e9, 3),
                "end_source_seconds": round((end_obs["captured_mono_ns"] - origin) / 1e9, 3),
                "s_frames": len(window),
                "s_person_detected_frames": sum(row["person_detected"] for row in window),
            }
        )
    return {
        "passed": True,
        "loop_completed": len(rates) >= 2,
        "completed_loops": len(rates) // 2,
        "verification_open_at_capture_end": len(rates) % 2 == 1,
        "scope": (
            f"{source} WebSocket loop; no fall/alert validation"
            if rates
            else f"{source} run without escalation; no fall validation"
        ),
        "captured": max(row.get("captured", 0) for row in edge if row["event"] == "edge_status"),
        "sent": len(sent),
        "received": len(received),
        "processed": len(observed),
        "dropped": 0,
        "rate_counts": dict(rate_counts),
        "commands": len(actions),
        "acks": len(edge_acks),
        "episodes": episodes,
        "s_person_detected_frames": sum(row["person_detected"] for row in s),
        "nano_processing_median_ms": round(
            statistics.median(row["processing_ms"] for row in nano), 3
        ),
        "s_processing_median_ms": round(statistics.median(row["processing_ms"] for row in s), 3)
        if s
        else None,
        "s_processing_p95_ms": percentile([row["processing_ms"] for row in s], 0.95) if s else None,
        "s_queue_wait_p95_ms": percentile([row["cloud_queue_wait_ms"] for row in s], 0.95)
        if s
        else None,
        "cool_verified": True,
        "instance_type": ready["runtime"]["ec2"]["instanceType"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    (args.folder / "loss-loop-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
