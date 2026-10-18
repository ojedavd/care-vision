"""Reproduce controlled startup outage and unconfirmed EOF on the local loop."""

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


async def recover_case(folder):
    video = make_video(folder / "recover.avi", static=True, seconds=4)
    edge_log, cloud_log = (
        EventLog("edge", folder / "recover-edge.jsonl"),
        EventLog("cloud", folder / "recover-cloud.jsonl"),
    )
    port = free_port()
    source = VideoSource(str(video))
    task = asyncio.create_task(Edge(source, f"ws://127.0.0.1:{port}", edge_log).run())
    try:
        await asyncio.sleep(1)
        async with serve(CloudServer(cloud_log).handler, "127.0.0.1", port):
            await asyncio.wait_for(task, timeout=8)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        edge_log.close()
        cloud_log.close()
    edge, cloud = read_log(folder / "recover-edge.jsonl"), read_log(folder / "recover-cloud.jsonl")
    connected = next(r for r in edge if r["event"] == "connected")
    status = next(r for r in cloud if r["event"] == "status_received")
    if (
        source.count != 120
        or connected["captured_while_offline"] <= 0
        or status["captured_while_offline"] != connected["captured_while_offline"]
        or not any(r["event"] == "drain_complete" for r in edge)
        or any(r["event"] == "cloud_processing_unconfirmed_at_eof" for r in edge)
    ):
        raise AssertionError("outage recovery was not confirmed")
    return {
        "source_frames": source.count,
        "offline_seconds_before_connect": connected["offline_seconds"],
        "captured_while_offline": connected["captured_while_offline"],
        "cloud_status_received_outage_count": status["captured_while_offline"],
        "network_qualities": sorted(
            {r["network_quality"] for r in edge if r["event"] == "network_transition"}
        ),
        "drain_confirmed": True,
    }


async def unavailable_case(folder):
    video = make_video(folder / "unavailable.avi", static=True, seconds=2)
    log = EventLog("edge", folder / "unavailable-edge.jsonl")
    source = VideoSource(str(video))
    error = None
    try:
        await asyncio.wait_for(Edge(source, f"ws://127.0.0.1:{free_port()}", log).run(), 6)
    except RuntimeError as exc:
        error = str(exc)
    finally:
        log.close()
    records = read_log(folder / "unavailable-edge.jsonl")
    if (
        error != "source ended without cloud processing confirmation"
        or source.count != 60
        or not any(r["event"] == "cloud_processing_unconfirmed_at_eof" for r in records)
    ):
        raise AssertionError("unavailable-cloud EOF was not an explicit failure")
    return {"source_frames": source.count, "failure": error, "cloud_analysis_available": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    args.output.mkdir(parents=True)
    recovered = asyncio.run(recover_case(args.output))
    unavailable = asyncio.run(unavailable_case(args.output))
    names = (
        "recover.avi",
        "recover-edge.jsonl",
        "recover-cloud.jsonl",
        "unavailable.avi",
        "unavailable-edge.jsonl",
    )
    report = {
        "scope": "Synthetic video, local OpenCV, loopback network faults; no AWS vision benchmark",
        "startup_outage_then_recovery": recovered,
        "cloud_unavailable_until_eof": unavailable,
        "file_sha256": {name: file_sha256(args.output / name) for name in names},
        "limits": [
            "Outage frames remain only in the bounded rolling buffer; no store-and-forward yet.",
            "RTT state is telemetry only; no FPS/resolution adaptation yet.",
            "No restricted bandwidth, packet loss or AWS tunnel measurements in this run.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
