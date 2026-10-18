"""Verify encrypted samples from a previous Edge process reach cloud vision."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from evaluate_offline_spool import free_port, offline_run
from websockets.asyncio.server import serve

from care_vision.check_loop import read_log
from care_vision.cloud.server import CloudServer
from care_vision.common.metrics import EventLog
from care_vision.edge.runtime import Edge
from care_vision.edge.source import VideoSource
from care_vision.edge.spool import EncryptedSpool
from care_vision.fixture import make_video
from care_vision.temporal_evaluation import file_sha256


async def reconnect(folder, video, key, first_count):
    edge_log = EventLog("edge", folder / "edge-reconnected.jsonl")
    cloud_log = EventLog("cloud", folder / "cloud.jsonl")
    source = VideoSource(str(video), spool=EncryptedSpool(folder / "private-spool", key))
    assert source.count == first_count
    port = free_port()
    try:
        async with serve(CloudServer(cloud_log).handler, "127.0.0.1", port):
            await asyncio.wait_for(
                Edge(source, f"ws://127.0.0.1:{port}", edge_log).run(), timeout=35
            )
    finally:
        edge_log.close()
        cloud_log.close()
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    folder = args.output.resolve()
    folder.mkdir(parents=True)
    first_video = make_video(folder / "offline.avi", static=True, seconds=12)
    second_video = make_video(folder / "reconnected.avi", static=True, seconds=8)
    key = os.urandom(32)
    first_count, _ = offline_run(folder, first_video, 1, key)
    source = asyncio.run(reconnect(folder, second_video, key, first_count))
    edge = read_log(folder / "edge-reconnected.jsonl")
    cloud = read_log(folder / "cloud.jsonl")
    initial = next(r for r in cloud if r["event"] == "status_received" and r.get("initial"))
    planned = next(r for r in cloud if r["event"] == "offline_recovery_planned")
    requests = [r for r in cloud if r["event"] == "offline_recovery_requested"]
    processed = [
        r for r in cloud if r["event"] == "observation" and r["evidence_kind"] == "historical"
    ]
    live = [r for r in cloud if r["event"] == "observation" and r["evidence_kind"] == "live"]
    archived = [r for r in processed if r["capture_seq"] <= first_count]
    drained = next(r for r in cloud if r["event"] == "drain_confirmed")
    confirmed = [r for r in edge if r["event"] == "spool_frames_confirmed"]
    if not (
        first_count == 360
        and source.count == 600
        and initial["spool_frames"] > 0
        and planned["archive_windows"] >= 1
        and requests[0]["window_kind"] == "archive"
        and len(requests) == planned["windows"]
        and archived
        and live
        and min(r["captured_mono_ns"] for r in archived) < min(r["captured_mono_ns"] for r in live)
        and len({r["frame_id"] for r in processed}) == len(processed)
        and drained["historical_received"] == drained["historical_processed"] == len(processed)
        and sum(r["frames_removed"] for r in confirmed) > 0
        and not any(r["event"] == "cloud_processing_unconfirmed_at_eof" for r in edge)
    ):
        raise AssertionError("archived evidence was not processed and confirmed after restart")
    names = (
        "offline.avi",
        "reconnected.avi",
        "edge-1.jsonl",
        "edge-reconnected.jsonl",
        "cloud.jsonl",
        "private-spool/offline.sqlite3",
    )
    report = {
        "scope": "Two Edge processes, encrypted offline capture, local cloud on reconnect",
        "first_process_captured": first_count,
        "cumulative_captured": source.count,
        "archive_windows_requested": planned["archive_windows"],
        "archived_frames_processed_from_first_process": len(archived),
        "historical_frames_processed_total": len(processed),
        "spool_frames_removed_after_cloud_confirmation": sum(
            r["frames_removed"] for r in confirmed
        ),
        "drain_confirmed": True,
        "limits": [
            "Local synthetic video and OpenCV; no AWS/Graviton/COOL performance claim.",
            "Encrypted spool samples offline capture at ~2 FPS, not every 30 FPS frame.",
            "The key is ephemeral and not saved with the artifact.",
            "A process restart was tested, not a power failure or operating-system reboot.",
        ],
        "file_sha256": {name: file_sha256(folder / name) for name in names},
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
