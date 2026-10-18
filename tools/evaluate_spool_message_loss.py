"""Omit one complete historical WebSocket message, then verify safe replay."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from evaluate_offline_spool import offline_run
from websockets.asyncio.server import serve

import care_vision.cloud.server as cloud_module
from care_vision.check_loop import read_log
from care_vision.cloud.server import CloudServer, Session
from care_vision.common.contracts import loads, unpack_frame
from care_vision.common.metrics import EventLog
from care_vision.edge.runtime import Edge
from care_vision.edge.source import VideoSource
from care_vision.edge.spool import EncryptedSpool
from care_vision.fixture import make_video
from care_vision.temporal_evaluation import file_sha256


class OmitHistoricalMessage:
    def __init__(self, websocket, fault, log):
        self.websocket, self.fault, self.log = websocket, fault, log

    async def send(self, data):
        await self.websocket.send(data)

    async def __aiter__(self):
        pending_historical = None
        async for raw in self.websocket:
            if not self.fault["omitted"] and isinstance(raw, bytes):
                header, _ = unpack_frame(raw)
                if header.get("evidence_kind") == "historical":
                    if pending_historical is not None:
                        yield pending_historical
                    pending_historical = raw
                    continue
            if (
                not self.fault["omitted"]
                and isinstance(raw, str)
                and loads(raw)["type"] == "BUFFER_COMPLETE"
                and pending_historical is not None
            ):
                header, _ = unpack_frame(pending_historical)
                self.fault["omitted"] = True
                self.fault["session_id"] = header["session_id"]
                self.fault["frame_id"] = header["frame_id"]
                self.log.emit(
                    "fault_injected",
                    **self.fault,
                    reason="last_historical_message_omitted_before_buffer_complete",
                )
                pending_historical = None
            yield raw


async def scenario(folder, video, key, first_count):
    edge_log = EventLog("edge", folder / "edge-reconnected.jsonl")
    cloud_log = EventLog("cloud", folder / "cloud.jsonl")
    source = VideoSource(str(video), spool=EncryptedSpool(folder / "private-spool", key))
    assert source.count == first_count
    fault = {"omitted": False, "session_id": None, "frame_id": None}

    class LossySession(Session):
        def __init__(self, websocket, *args, **kwargs):
            if not fault["omitted"]:
                websocket = OmitHistoricalMessage(websocket, fault, cloud_log)
            super().__init__(websocket, *args, **kwargs)

    original_session = cloud_module.Session
    cloud_module.Session = LossySession
    try:
        async with serve(CloudServer(cloud_log).handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            await asyncio.wait_for(
                Edge(source, f"ws://127.0.0.1:{port}", edge_log).run(), timeout=45
            )
    finally:
        cloud_module.Session = original_session
        edge_log.close()
        cloud_log.close()
    return source.count, fault


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    folder = args.output.resolve()
    folder.mkdir(parents=True)
    first_video = make_video(folder / "offline.avi", static=True, seconds=12)
    second_video = make_video(folder / "reconnected.avi", static=True, seconds=20)
    key = os.urandom(32)
    first_count, _ = offline_run(folder, first_video, 1, key)
    total_count, fault = asyncio.run(scenario(folder, second_video, key, first_count))
    edge = read_log(folder / "edge-reconnected.jsonl")
    cloud = read_log(folder / "cloud.jsonl")
    sessions = [r for r in cloud if r["event"] == "session_started"]
    final_id = sessions[-1]["session_id"]
    errors = [r for r in cloud if r["event"] == "session_error"]
    final_historical = {
        r["frame_id"]
        for r in cloud
        if r["event"] == "observation"
        and r["evidence_kind"] == "historical"
        and r["session_id"] == final_id
    }
    drains = [r for r in cloud if r["event"] == "drain_confirmed"]
    confirmed = [r for r in edge if r["event"] == "spool_frames_confirmed"]
    if not (
        first_count == 360
        and total_count == 960
        and fault["omitted"]
        and len(sessions) >= 2
        and sessions[0]["session_id"] == fault["session_id"]
        and any("incomplete historical transfer" in r["reason"] for r in errors)
        and fault["frame_id"]
        not in {
            r["frame_id"]
            for r in cloud
            if r["event"] == "frame_received" and r["session_id"] == fault["session_id"]
        }
        and fault["frame_id"] in final_historical
        and not any(r["session_id"] == fault["session_id"] for r in drains)
        and drains[-1]["session_id"] == final_id
        and drains[-1]["historical_received"] == drains[-1]["historical_processed"]
        and confirmed
        and all(r["session_id"] == final_id for r in confirmed)
        and not any(r["event"] == "cloud_processing_unconfirmed_at_eof" for r in edge)
    ):
        raise AssertionError("omitted historical message was not detected and replayed")
    names = (
        "offline.avi",
        "reconnected.avi",
        "edge-1.jsonl",
        "edge-reconnected.jsonl",
        "cloud.jsonl",
        "private-spool/offline.sqlite3",
    )
    report = {
        "scope": "One complete historical message omitted at cloud receive; local WebSocket",
        "offline_captured": first_count,
        "total_captured": total_count,
        "cloud_sessions": len(sessions),
        "first_session_rejected_incomplete_transfer": True,
        "omitted_frame_processed_after_reconnect": True,
        "spool_frames_removed_after_final_confirmation": sum(
            r["frames_removed"] for r in confirmed
        ),
        "final_drain_confirmed": True,
        "limits": [
            "This injects application-message omission, not IP packet loss.",
            "TCP/WebSocket normally retransmits lost packets; this omits a full message.",
            "The ephemeral spool key is not saved in the artifact.",
            "No AWS/Graviton/COOL or incident-detection performance claim.",
        ],
        "file_sha256": {name: file_sha256(folder / name) for name in names},
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
