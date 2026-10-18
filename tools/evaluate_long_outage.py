"""Measure retained and evicted offline evidence after a long cloud outage."""

import argparse
import asyncio
import json
import socket
from pathlib import Path

from websockets.asyncio.server import serve

from care_vision.check_loop import read_log
from care_vision.cloud.server import CloudServer
from care_vision.common.metrics import EventLog
from care_vision.edge.runtime import Edge
from care_vision.edge.source import VideoSource
from care_vision.fixture import make_video
from care_vision.temporal_evaluation import file_sha256


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def scenario(folder):
    video = make_video(folder / "static.avi", static=True, seconds=22)
    edge_log = EventLog("edge", folder / "edge.jsonl")
    cloud_log = EventLog("cloud", folder / "cloud.jsonl")
    port = free_port()
    source = VideoSource(str(video))
    task = asyncio.create_task(Edge(source, f"ws://127.0.0.1:{port}", edge_log).run())
    try:
        await asyncio.sleep(12)
        async with serve(CloudServer(cloud_log).handler, "127.0.0.1", port):
            await asyncio.wait_for(task, timeout=30)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        edge_log.close()
        cloud_log.close()
    edge, cloud = read_log(folder / "edge.jsonl"), read_log(folder / "cloud.jsonl")
    connected = next(r for r in edge if r["event"] == "connected")
    initial = next(r for r in cloud if r["event"] == "status_received" and r.get("initial"))
    gap = next(r for r in cloud if r["event"] == "offline_evidence_gap")
    recovery = next(r for r in cloud if r["event"] == "offline_recovery_requested")
    ack = next(
        r
        for r in cloud
        if r["event"] == "ack_received" and r["command_id"] == recovery["buffer_request_id"]
    )
    drain = next(r for r in cloud if r["event"] == "drain_confirmed")
    historical = [
        r for r in cloud if r["event"] == "observation" and r["evidence_kind"] == "historical"
    ]
    if not (
        source.count == 660
        and connected["captured_while_offline"] > 300
        and initial["captured_while_offline"] == connected["captured_while_offline"]
        and initial["offline_frames_retained_at_reconnect"] <= 300
        and initial["offline_frames_lost_at_reconnect"] > 0
        and initial["offline_frames_retained_at_reconnect"]
        + initial["offline_frames_lost_at_reconnect"]
        == initial["captured_while_offline"]
        and initial["recovery_truncated"] is True
        and gap["offline_frames_lost_at_reconnect"] == initial["offline_frames_lost_at_reconnect"]
        and ack["selected_frames"] == len(historical) > 0
        and drain["historical_received"] == drain["historical_processed"] == len(historical)
        and not any(r["event"] == "cloud_processing_unconfirmed_at_eof" for r in edge)
    ):
        raise AssertionError("long-outage evidence accounting or drain failed")
    return {
        "capture_frames": source.count,
        "offline_seconds": connected["offline_seconds"],
        "captured_while_offline": connected["captured_while_offline"],
        "offline_frames_retained_at_reconnect": initial["offline_frames_retained_at_reconnect"],
        "offline_frames_lost_at_reconnect": initial["offline_frames_lost_at_reconnect"],
        "recovery_truncated": initial["recovery_truncated"],
        "historical_selected_received_processed": len(historical),
        "cloud_gap_event_emitted": True,
        "drain_confirmed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    args.output.mkdir(parents=True)
    result = asyncio.run(scenario(args.output))
    names = ("static.avi", "edge.jsonl", "cloud.jsonl")
    report = {
        "scope": "Synthetic static video, local OpenCV, long startup cloud outage",
        "cloud_start_delay_seconds": 12,
        "result": result,
        "limits": [
            "Retained counts describe the reconnection snapshot, not guaranteed delivery.",
            "The ring buffer is volatile; a process restart loses it.",
            "No incident detection or AWS network benchmark is claimed.",
        ],
        "file_sha256": {name: file_sha256(args.output / name) for name in names},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
