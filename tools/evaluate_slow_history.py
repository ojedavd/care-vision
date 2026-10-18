"""Real local WebSocket transfer with the historical wire budget held at its minimum."""

import argparse
import asyncio
import json
from pathlib import Path

from websockets.asyncio.server import serve

from care_vision.check_history_delivery import check_history_delivery
from care_vision.check_loop import read_log
from care_vision.cloud.server import CloudServer
from care_vision.common.history_budget import (
    MIN_HISTORY_BYTES_PER_SECOND,
    MIN_HISTORY_FPS,
)
from care_vision.common.metrics import EventLog
from care_vision.edge.runtime import Edge
from care_vision.edge.source import VideoSource
from care_vision.temporal_evaluation import file_sha256


class SlowHistoryEdge(Edge):
    def history_limits(self):
        return MIN_HISTORY_FPS, MIN_HISTORY_BYTES_PER_SECOND


async def scenario(video, output):
    edge_log = EventLog("edge", output / "edge.jsonl")
    cloud_log = EventLog("cloud", output / "cloud.jsonl")
    source = VideoSource(str(video), width=1280, height=720)
    try:
        async with serve(CloudServer(cloud_log).handler, "127.0.0.1", 0) as cloud:
            port = cloud.sockets[0].getsockname()[1]
            await asyncio.wait_for(SlowHistoryEdge(
                source, f"ws://127.0.0.1:{port}", edge_log,
                output_width=320, output_height=180,
            ).run(), timeout=240)
    finally:
        edge_log.close()
        cloud_log.close()
    edge, cloud = read_log(output / "edge.jsonl"), read_log(output / "cloud.jsonl")
    result = check_history_delivery(edge, cloud)
    if not any(r["completed_after_request_ms"] > 30_000 for r in result["history"]):
        raise AssertionError("transfer did not exercise the former 30 second deadline")
    return {
        "scope": "Local OpenCV, real WebSocket, fixed minimum historical pacing; no RTT injection",
        "passed": True,
        "source_sha256": file_sha256(video),
        "history_fps_limit": MIN_HISTORY_FPS,
        "history_bytes_per_second": MIN_HISTORY_BYTES_PER_SECOND,
        "result": result,
        "deadlines": [r for r in cloud if r["event"] == "history_deadline"],
        "file_sha256": {name: file_sha256(output / name) for name in ("edge.jsonl", "cloud.jsonl")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = asyncio.run(scenario(args.source_video, args.output))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
