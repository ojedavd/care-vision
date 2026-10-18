"""Stop and restart cloud mid-motion to audit the reconnect evidence gap."""

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
    video = make_video(folder / "moving.avi", seconds=22)
    edge_log = EventLog("edge", folder / "edge.jsonl")
    cloud_log = EventLog("cloud", folder / "cloud.jsonl")
    source = VideoSource(str(video))
    port = free_port()
    task = None
    try:
        async with serve(CloudServer(cloud_log).handler, "127.0.0.1", port) as server:
            task = asyncio.create_task(Edge(source, f"ws://127.0.0.1:{port}", edge_log).run())
            await asyncio.sleep(10)
            server.close()
            await server.wait_closed()
        await asyncio.sleep(3)
        async with serve(CloudServer(cloud_log).handler, "127.0.0.1", port):
            await asyncio.wait_for(task, timeout=30)
    finally:
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        edge_log.close()
        cloud_log.close()
    edge, cloud = read_log(folder / "edge.jsonl"), read_log(folder / "cloud.jsonl")
    sessions = [r for r in cloud if r["event"] == "session_started"]
    connected = [r for r in edge if r["event"] == "connected"]
    recovery = [r for r in cloud if r["event"] == "offline_recovery_requested"]
    pre_event = [
        r
        for r in cloud
        if r["event"] == "agent_action" and r.get("reason") == "recover_pre_event_evidence"
    ]
    aborted = [r for r in edge if r["event"] == "buffer_transfer_aborted"]
    incomplete = [r for r in cloud if r["event"] == "buffer_incomplete"]
    disconnects = [r for r in edge if r["event"] == "offline"]
    acknowledgements = (
        [
            r
            for r in cloud
            if r["event"] == "ack_received"
            and r["command_type"] == "REQUEST_RECENT_BUFFER"
            and r["command_id"] == recovery[-1]["buffer_request_id"]
        ]
        if recovery
        else []
    )
    final = [r for r in cloud if r["event"] == "drain_confirmed"]
    if not (
        source.count == 660
        and len(sessions) >= 2
        and len(connected) >= 2
        and connected[-1]["captured_while_offline"] > 0
        and recovery
        and aborted
        and incomplete
        and disconnects
        and pre_event
        and recovery[-1]["includes_unconfirmed_context"] is True
        and recovery[-1]["from_edge_mono_ns"] < disconnects[0]["local_mono_ns"]
        and recovery[-1]["available_frames"] > connected[-1]["captured_while_offline"]
        and acknowledgements
        and acknowledgements[-1]["selected_frames"] > 0
        and final
        and final[-1]["historical_processed"] > 0
    ):
        raise AssertionError("mid-event outage did not reconnect and recover evidence")
    return {
        "capture_frames": source.count,
        "session_count": len(sessions),
        "offline_seconds": connected[-1]["offline_seconds"],
        "captured_while_offline": connected[-1]["captured_while_offline"],
        "recovery_available_frames": recovery[-1]["available_frames"],
        "recovery_includes_unconfirmed_context": recovery[-1]["includes_unconfirmed_context"],
        "recovery_truncated": recovery[-1]["truncated"],
        "recovery_selected_frames": acknowledgements[-1]["selected_frames"],
        "pre_event_context_lost_prefix_ms": max(
            0,
            (recovery[-1]["from_edge_mono_ns"] - pre_event[0]["from_edge_mono_ns"]) / 1_000_000,
        ),
        "aborted_pending_history_records": aborted[-1]["pending_records"],
        "first_session_historical_received_before_failure": incomplete[-1]["frames_received"],
        "final_live_received": final[-1]["live_received"],
        "final_live_processed": final[-1]["live_processed"],
        "final_historical_received": final[-1]["historical_received"],
        "final_historical_processed": final[-1]["historical_processed"],
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
    names = ("moving.avi", "edge.jsonl", "cloud.jsonl")
    report = {
        "scope": "Local cloud service stopped mid-motion for three seconds; no AWS benchmark",
        "fault": {"cloud_stop_at_seconds": 10, "cloud_unavailable_seconds": 3},
        "result": result,
        "limits": [
            "This tests a complete service outage, not packet loss.",
            "The synthetic motion fixture is not a fall-detection validation.",
        ],
        "file_sha256": {name: file_sha256(args.output / name) for name in names},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
