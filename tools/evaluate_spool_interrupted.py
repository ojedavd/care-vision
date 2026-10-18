"""Interrupt an encrypted historical transfer, reconnect, and verify replay."""

import argparse
import asyncio
import json
import os
from pathlib import Path

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


class InterruptingCloudLog(EventLog):
    def __init__(self, path):
        super().__init__("cloud", path)
        self.cloud = None
        self.historical_count = 0
        self.interrupted_session_id = None

    def emit(self, event, **fields):
        super().emit(event, **fields)
        if event != "frame_received" or fields.get("evidence_kind") != "historical":
            return
        if self.interrupted_session_id is not None:
            return
        self.historical_count += 1
        if self.historical_count == 4:
            websocket, identity = self.cloud.active
            self.interrupted_session_id = identity["session_id"]
            super().emit(
                "fault_injected",
                session_id=self.interrupted_session_id,
                after_historical_frames=self.historical_count,
                reason="close_cloud_socket_before_buffer_complete",
            )
            asyncio.create_task(
                websocket.close(code=1012, reason="injected historical interruption")
            )


async def scenario(folder, video, key, first_count):
    edge_log = EventLog("edge", folder / "edge-reconnected.jsonl")
    cloud_log = InterruptingCloudLog(folder / "cloud.jsonl")
    cloud = CloudServer(cloud_log)
    cloud_log.cloud = cloud
    source = VideoSource(str(video), spool=EncryptedSpool(folder / "private-spool", key))
    assert source.count == first_count
    try:
        async with serve(cloud.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            await asyncio.wait_for(
                Edge(source, f"ws://127.0.0.1:{port}", edge_log).run(), timeout=40
            )
    finally:
        edge_log.close()
        cloud_log.close()
    return source.count, cloud_log.interrupted_session_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    folder = args.output.resolve()
    folder.mkdir(parents=True)
    first_video = make_video(folder / "offline.avi", static=True, seconds=12)
    second_video = make_video(folder / "reconnected.avi", static=True, seconds=16)
    key = os.urandom(32)
    first_count, _ = offline_run(folder, first_video, 1, key)
    total_count, interrupted_session = asyncio.run(scenario(folder, second_video, key, first_count))
    edge = read_log(folder / "edge-reconnected.jsonl")
    cloud = read_log(folder / "cloud.jsonl")
    sessions = [r for r in cloud if r["event"] == "session_started"]
    first_received = {
        r["frame_id"]
        for r in cloud
        if r["event"] == "frame_received"
        and r["evidence_kind"] == "historical"
        and r["session_id"] == interrupted_session
    }
    second_id = sessions[-1]["session_id"]
    second_observed = {
        r["frame_id"]
        for r in cloud
        if r["event"] == "observation"
        and r["evidence_kind"] == "historical"
        and r["session_id"] == second_id
    }
    second_old = {
        r["frame_id"]
        for r in cloud
        if r["event"] == "observation"
        and r["evidence_kind"] == "historical"
        and r["session_id"] == second_id
        and r["capture_seq"] <= first_count
    }
    first_incomplete = [
        r
        for r in cloud
        if r["event"] == "buffer_incomplete" and r.get("session_id") == interrupted_session
    ]
    drains = [r for r in cloud if r["event"] == "drain_confirmed"]
    confirmations = [r for r in edge if r["event"] == "spool_frames_confirmed"]
    if not (
        first_count == 360
        and total_count == 840
        and len(sessions) >= 2
        and sessions[0]["session_id"] == interrupted_session
        and len(first_received) >= 4
        and first_received <= second_observed
        and second_old
        and first_incomplete
        and not any(r["session_id"] == interrupted_session for r in drains)
        and drains[-1]["session_id"] == second_id
        and drains[-1]["historical_received"] == drains[-1]["historical_processed"]
        and confirmations
        and all(r["session_id"] != interrupted_session for r in confirmations)
        and any(r["event"] == "buffer_transfer_aborted" for r in edge)
        and not any(r["event"] == "cloud_processing_unconfirmed_at_eof" for r in edge)
    ):
        raise AssertionError("interrupted encrypted transfer was not safely replayed")
    names = (
        "offline.avi",
        "reconnected.avi",
        "edge-1.jsonl",
        "edge-reconnected.jsonl",
        "cloud.jsonl",
        "private-spool/offline.sqlite3",
    )
    report = {
        "scope": "Local cloud socket closed after four archived frames; Edge reconnects",
        "offline_captured": first_count,
        "total_captured": total_count,
        "sessions": len(sessions),
        "first_session_historical_received_before_interruption": len(first_received),
        "first_session_historical_transfer_incomplete": True,
        "replayed_first_session_frames_processed_in_second": len(first_received),
        "old_archived_frames_processed_in_second": len(second_old),
        "spool_frames_removed_only_after_second_session_drain": sum(
            r["frames_removed"] for r in confirmations
        ),
        "final_drain_confirmed": True,
        "limits": [
            "Local socket interruption, not packet loss or a physical network fault.",
            "The spool samples at approximately 2 FPS and retains at most 60 seconds by default.",
            "The ephemeral spool key is not saved in the artifact.",
            "No AWS/Graviton/COOL performance or incident-detection result is claimed.",
        ],
        "file_sha256": {name: file_sha256(folder / name) for name in names},
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
