"""Replay archived offline samples through a delayed, byte-paced TCP proxy."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from evaluate_network_proxy import proxy_pipe
from evaluate_offline_spool import offline_run
from websockets.asyncio.server import serve

from care_vision.check_loop import read_log
from care_vision.cloud.server import CloudServer
from care_vision.common.metrics import EventLog
from care_vision.edge.runtime import Edge
from care_vision.edge.source import VideoSource
from care_vision.edge.spool import EncryptedSpool
from care_vision.fixture import make_video
from care_vision.temporal_evaluation import file_sha256


async def scenario(folder, video, key, first_count):
    edge_log = EventLog("edge", folder / "edge-reconnected.jsonl")
    cloud_log = EventLog("cloud", folder / "cloud.jsonl")
    source = VideoSource(str(video), spool=EncryptedSpool(folder / "private-spool", key))
    assert source.count == first_count
    counters = {"paced_bytes": 0, "unpaced_bytes": 0}
    try:
        async with serve(CloudServer(cloud_log).handler, "127.0.0.1", 0) as cloud:
            cloud_port = cloud.sockets[0].getsockname()[1]
            started = time.monotonic()

            async def relay(inbound_reader, inbound_writer):
                outbound_reader, outbound_writer = await asyncio.open_connection(
                    "127.0.0.1", cloud_port
                )
                await asyncio.gather(
                    proxy_pipe(inbound_reader, outbound_writer, started, counters),
                    proxy_pipe(outbound_reader, inbound_writer, started, counters),
                )

            async with await asyncio.start_server(relay, "127.0.0.1", 0) as proxy:
                port = proxy.sockets[0].getsockname()[1]
                await asyncio.wait_for(
                    Edge(source, f"ws://127.0.0.1:{port}", edge_log).run(), timeout=65
                )
    finally:
        edge_log.close()
        cloud_log.close()
    return source.count, counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    folder = args.output.resolve()
    folder.mkdir(parents=True)
    first_video = make_video(folder / "offline.avi", static=True, seconds=12)
    second_video = make_video(folder / "reconnected.avi", static=True, seconds=18)
    key = os.urandom(32)
    first_count, _ = offline_run(folder, first_video, 1, key)
    total_count, counters = asyncio.run(scenario(folder, second_video, key, first_count))
    edge = read_log(folder / "edge-reconnected.jsonl")
    cloud = read_log(folder / "cloud.jsonl")
    budgets = [r for r in edge if r["event"] == "history_transmission_budget_changed"]
    archive_sent = [
        r
        for r in edge
        if r["event"] == "frame_sent"
        and r["evidence_kind"] == "historical"
        and r["capture_seq"] <= first_count
    ]
    observed = [
        r
        for r in cloud
        if r["event"] == "observation"
        and r["evidence_kind"] == "historical"
        and r["capture_seq"] <= first_count
    ]
    degraded_at = next(r["local_mono_ns"] for r in budgets if r["wire_max_fps"] == 2)
    recovered_at = next(r["local_mono_ns"] for r in budgets if r["wire_max_fps"] == 20)
    paced_archived = [
        r for r in archive_sent if degraded_at <= r["send_started_mono_ns"] < recovered_at
    ]
    intervals = [
        (b["send_started_mono_ns"] - a["send_started_mono_ns"]) / 1e9
        for a, b in zip(paced_archived, paced_archived[1:])
    ]
    drained = next(r for r in cloud if r["event"] == "drain_confirmed")
    if not (
        first_count == 360
        and total_count == 900
        and counters["paced_bytes"] > 0
        and counters["unpaced_bytes"] > 0
        and [r["wire_max_fps"] for r in budgets] == [2, 20]
        and len(paced_archived) >= 2
        and all(seconds >= 0.45 for seconds in intervals)
        and {r["frame_id"] for r in archive_sent} == {r["frame_id"] for r in observed}
        and drained["historical_processed"] == drained["historical_received"]
        and not any(r["event"] == "cloud_processing_unconfirmed_at_eof" for r in edge)
    ):
        raise AssertionError("degraded archive pacing or cloud confirmation failed")
    names = (
        "offline.avi",
        "reconnected.avi",
        "edge-1.jsonl",
        "edge-reconnected.jsonl",
        "cloud.jsonl",
        "private-spool/offline.sqlite3",
    )
    report = {
        "scope": "Two Edge processes, local delayed and byte-paced TCP proxy",
        "offline_captured": first_count,
        "total_captured": total_count,
        "paced_bytes_both_directions": counters["paced_bytes"],
        "archive_frames_processed_after_restart": len(observed),
        "archive_frames_sent_under_measured_degradation": len(paced_archived),
        "minimum_archive_send_interval_seconds_under_degradation": round(min(intervals), 3),
        "history_wire_fps_limits": [r["wire_max_fps"] for r in budgets],
        "drain_confirmed": True,
        "limits": [
            "Local proxy, not a physical home network or AWS path.",
            "No packet loss, power failure, or incident-detection claim.",
            "2 FPS and 64 KiB/s historical budgets are experimental, not calibrated.",
            "The ephemeral spool key is not saved in the artifact.",
        ],
        "file_sha256": {name: file_sha256(folder / name) for name in names},
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
