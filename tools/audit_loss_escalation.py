"""Audit the causal Edge command/ACK/frame sequence in loss-escalation replays."""

import argparse
import json
import statistics
from pathlib import Path


def audit(folder):
    result = json.loads((folder / "result.json").read_text())
    runtime = json.loads((folder / "runtime.json").read_text())
    assert runtime["runtime"]["cool"]["verified"]
    assert runtime["runtime"]["ec2"]["architecture"] == "arm64"
    rows, sent, commands, acks = (
        result["rows"],
        result["sent"],
        result["commands"],
        result["acks"],
    )
    assert len(rows) == len(sent)
    assert len(commands) == len(acks)
    assert [c["command_id"] for c in commands] == [a["command_id"] for a in acks]
    assert [c["command_seq"] for c in commands] == list(range(1, len(commands) + 1))
    frames = {row["header"]["frame_id"]: row for row in rows}
    by_id = dict(zip((c["command_id"] for c in commands), commands))
    ack_by_id = dict(zip((a["command_id"] for a in acks), acks))
    assert len(frames) == len(rows)
    for row, record in zip(rows, sent):
        header = row["header"]
        assert header["frame_id"] == record["frame_id"]
        assert row["jpeg_sha256"] == record["sha256"]
        assert header["payload_bytes"] == record["payload_bytes"]
        assert header["capture_seq"] == record["capture_seq"]
        if header["stream_role"] == "verification":
            assert (header["width"], header["height"], header["target_transmit_fps"]) == (
                1280,
                720,
                5,
            )
        else:
            assert header["stream_role"] == "baseline"
            assert (header["width"], header["height"], header["target_transmit_fps"]) == (
                320,
                180,
                2,
            )
        for command_id in (header["applied_command_id"], header["applied_visual_command_id"]):
            if command_id is None:
                continue
            command, ack = by_id[command_id], ack_by_id[command_id]
            assert ack["result"] == "applied"
            assert header["capture_seq"] >= ack["effective_from_capture_seq"]
            assert (
                frames[command["based_on_frame_id"]]["header"]["capture_seq"]
                < header["capture_seq"]
            )
    rates = [c for c in commands if c["type"] == "SET_TRANSMIT_FPS"]
    ups = [c for c in rates if c["transmit_fps"] == 5]
    downs = [c for c in rates if c["transmit_fps"] == 2]
    assert len(ups) == len(downs)
    episodes = []
    for up, down in zip(ups, downs):
        start = frames[up["based_on_frame_id"]]["source_seconds"]
        end = frames[down["based_on_frame_id"]]["source_seconds"]
        evidence = [
            row
            for row in rows
            if row["header"]["stream_role"] == "verification"
            and start < row["source_seconds"] <= end
        ]
        assert evidence
        deltas = [b["source_seconds"] - a["source_seconds"] for a, b in zip(evidence, evidence[1:])]
        episodes.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "s_frames": len(evidence),
                "s_person_detected_frames": sum(
                    row["people"]["person_detected"] for row in evidence
                ),
                "median_interframe_seconds": round(statistics.median(deltas), 3)
                if deltas
                else None,
                "s_jpeg_bytes": sum(row["header"]["payload_bytes"] for row in evidence),
                "s_inference_median_ms": round(
                    statistics.median(row["people"]["person_detection_ms"] for row in evidence),
                    1,
                ),
            }
        )
    return {
        "status": "passed",
        "scope": "sequential replay, not wall-paced service or fall confirmation",
        "frames": len(rows),
        "baseline_frames": sum(row["header"]["stream_role"] == "baseline" for row in rows),
        "s_frames": sum(row["header"]["stream_role"] == "verification" for row in rows),
        "jpeg_bytes": sum(record["payload_bytes"] for record in sent),
        "commands": len(commands),
        "acks": len(acks),
        "episodes": episodes,
        "wall_seconds": result["wall_seconds"],
        "cool_verified": True,
        "ec2": runtime["runtime"]["ec2"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    args = parser.parse_args()
    for folder in args.folders:
        result = audit(folder)
        (folder / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(folder, result["status"], result["frames"], result["episodes"])


if __name__ == "__main__":
    main()
