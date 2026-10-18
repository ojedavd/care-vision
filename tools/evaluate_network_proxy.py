"""Run Edge and cloud through a paced TCP proxy with measured WebSocket RTT."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from websockets.asyncio.server import serve

from care_vision.check_loop import read_log
from care_vision.cloud.server import CloudServer
from care_vision.common.metrics import EventLog
from care_vision.edge.runtime import Edge
from care_vision.edge.source import VideoSource
from care_vision.fixture import make_video
from care_vision.temporal_evaluation import file_sha256

DEGRADED_SECONDS = 12
ONE_WAY_DELAY_SECONDS = 0.16
BYTES_PER_SECOND = 128 * 1024


async def proxy_pipe(reader, writer, started, counters):
    due = 0.0
    try:
        while data := await reader.read(65536):
            degraded = time.monotonic() - started < DEGRADED_SECONDS
            if degraded:
                await asyncio.sleep(ONE_WAY_DELAY_SECONDS)
                await asyncio.sleep(max(0, due - time.monotonic()))
                due = time.monotonic() + len(data) / BYTES_PER_SECOND
                counters["paced_bytes"] += len(data)
            else:
                counters["unpaced_bytes"] += len(data)
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()


async def scenario(folder):
    video = make_video(folder / "moving.avi", seconds=22)
    edge_log = EventLog("edge", folder / "edge.jsonl")
    cloud_log = EventLog("cloud", folder / "cloud.jsonl")
    counters = {"paced_bytes": 0, "unpaced_bytes": 0}
    source = VideoSource(str(video))
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
                proxy_port = proxy.sockets[0].getsockname()[1]
                await asyncio.wait_for(
                    Edge(source, f"ws://127.0.0.1:{proxy_port}", edge_log).run(),
                    timeout=45,
                )
    finally:
        edge_log.close()
        cloud_log.close()
    edge, cloud = read_log(folder / "edge.jsonl"), read_log(folder / "cloud.jsonl")
    changes = [r for r in edge if r["event"] == "network_transmission_cap_changed"]
    limited = [r for r in edge if r["event"] == "frame_sent" and r.get("network_limited") is True]
    unrestricted_high = [
        r
        for r in edge
        if r["event"] == "frame_sent"
        and r.get("target_transmit_fps") == 20
        and r.get("network_limited") is False
    ]
    actions = [r for r in cloud if r["event"] == "agent_action" and r["type"] == "SET_TRANSMIT_FPS"]
    sent_live = {
        r["frame_id"] for r in edge if r["event"] == "frame_sent" and r["evidence_kind"] == "live"
    }
    sent_history = {
        r["frame_id"]
        for r in edge
        if r["event"] == "frame_sent" and r["evidence_kind"] == "historical"
    }
    received_live = {
        r["frame_id"]
        for r in cloud
        if r["event"] == "frame_received" and r["evidence_kind"] == "live"
    }
    received_history = {
        r["frame_id"]
        for r in cloud
        if r["event"] == "frame_received" and r["evidence_kind"] == "historical"
    }
    observed_history = {
        r["frame_id"]
        for r in cloud
        if r["event"] == "observation" and r["evidence_kind"] == "historical"
    }
    acknowledgements = {
        r["command_id"]
        for r in cloud
        if r["event"] == "ack_received" and r["result"] in {"applied", "duplicate"}
    }
    rtts = [
        r["network_rtt_ms"]
        for r in edge
        if r["event"] == "edge_status" and r.get("network_rtt_ms") is not None
    ]
    drain = [r for r in cloud if r["event"] == "drain_confirmed"]
    if not (
        source.count == 660
        and counters["paced_bytes"] > 0
        and counters["unpaced_bytes"] > 0
        and any(r["cap_fps"] == 5 for r in changes)
        and any(r["cap_fps"] is None for r in changes)
        and limited
        and unrestricted_high
        and any(r["transmit_fps"] == 20 for r in actions)
        and all(r["command_id"] in acknowledgements for r in actions)
        and all(r["effective_target_transmit_fps"] == 5 for r in limited)
        and all(r["effective_target_transmit_fps"] == 20 for r in unrestricted_high)
        and sent_live == received_live
        and sent_history == received_history == observed_history
        and rtts
        and max(rtts) >= 250
        and min(rtts) <= 150
        and drain
        and drain[-1]["live_received"] == len(sent_live)
        and drain[-1]["historical_received"] == len(sent_history)
        and drain[-1]["historical_processed"] == len(observed_history)
        and not any(r["event"] == "cloud_processing_unconfirmed_at_eof" for r in edge)
    ):
        raise AssertionError("paced proxy loop did not show cap, recovery, and cloud drain")
    return {
        "capture_frames": source.count,
        "paced_bytes_both_directions": counters["paced_bytes"],
        "unpaced_bytes_both_directions": counters["unpaced_bytes"],
        "cap_changes": [r["cap_fps"] for r in changes],
        "max_measured_rtt_ms": round(max(rtts), 1),
        "min_measured_rtt_ms": round(min(rtts), 1),
        "live_frames_sent_while_limited": len(limited),
        "live_frames_sent_after_cap_removed_at_20fps": len(unrestricted_high),
        "network_limited_frames": max(
            r["network_limited_frames"] for r in edge if r["event"] == "edge_status"
        ),
        "agent_fps_actions": [r["transmit_fps"] for r in actions],
        "cloud_live_received": drain[-1]["live_received"],
        "cloud_live_processed": drain[-1]["live_processed"],
        "cloud_historical_received": drain[-1]["historical_received"],
        "cloud_historical_processed": drain[-1]["historical_processed"],
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
        "scope": "Local TCP proxy, actual WebSocket bytes and ping RTT; OpenCV is local",
        "proxy": {
            "first_seconds_paced": DEGRADED_SECONDS,
            "one_way_delay_ms": ONE_WAY_DELAY_SECONDS * 1000,
            "bytes_per_second_each_direction": BYTES_PER_SECOND,
        },
        "result": result,
        "limits": [
            "This proxy is not a physical home network or an AWS Graviton path.",
            "The experiment does not inject packet loss or complete outages.",
            "The fixture tests motion-driven sensing, not fall detection.",
        ],
        "file_sha256": {name: file_sha256(args.output / name) for name in names},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
