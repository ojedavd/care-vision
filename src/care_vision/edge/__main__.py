import argparse
import asyncio
import base64
import getpass
import os

from care_vision.common.metrics import EventLog
from care_vision.common.resources import monitor_resources
from care_vision.edge.recorded_source import RecordedSource
from care_vision.edge.runtime import Edge
from care_vision.edge.source import CameraSource, RTSPSource, VideoSource
from care_vision.edge.spool import EncryptedSpool


def main():
    parser = argparse.ArgumentParser(description="CARE-Vision file/camera/RTSP gateway")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video")
    inputs.add_argument("--camera", type=int)
    inputs.add_argument("--rtsp-ip", help="local IPv4 address of an IP camera")
    parser.add_argument("--rtsp-stream", type=int, choices=[1, 2], default=1)
    parser.add_argument("--rtsp-ffmpeg-nobuffer", action="store_true")
    parser.add_argument("--camera-seconds", type=float, default=45)
    parser.add_argument("--record-camera", help="new local AVI for replay; no audio")
    parser.add_argument("--replay-captures", help="recorded timestamps for video input")
    parser.add_argument("--replay-sampling-phase-ms", type=int)
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--capture-fps", type=float, default=30)
    parser.add_argument("--initial-transmit-fps", type=int, choices=[2, 20], default=2)
    parser.add_argument("--capture-resolution", choices=["640x360", "1280x720"], default="640x360")
    parser.add_argument(
        "--baseline-resolution", choices=["320x180", "640x360", "1280x720"], default="640x360"
    )
    parser.add_argument("--edge-id", default="edge-1")
    parser.add_argument("--log")
    parser.add_argument("--offline-spool", help="encrypted local spool directory (experimental)")
    args = parser.parse_args()
    if args.replay_captures and (not args.video or args.offline_spool):
        parser.error("replay-captures requires video input without offline spool")
    if args.replay_sampling_phase_ms is not None and (
        not args.replay_captures or not 0 <= args.replay_sampling_phase_ms < 500
    ):
        parser.error("replay sampling phase requires recorded timestamps and 0-499 ms")
    if (args.camera is not None or args.rtsp_ip) and args.offline_spool:
        parser.error("live camera spool replay mapping is not supported yet")
    if args.video and args.record_camera:
        parser.error("record-camera requires camera or RTSP input")
    if args.rtsp_ffmpeg_nobuffer and not args.rtsp_ip:
        parser.error("rtsp-ffmpeg-nobuffer requires RTSP input")
    source_width, source_height = map(int, args.capture_resolution.split("x"))
    output_width, output_height = map(int, args.baseline_resolution.split("x"))
    spool = None
    if args.offline_spool:
        encoded_key = os.environ.get("CARE_VISION_SPOOL_KEY")
        if not encoded_key:
            parser.error("CARE_VISION_SPOOL_KEY is required with --offline-spool")
        try:
            key = base64.b64decode(encoded_key, validate=True)
        except ValueError:
            parser.error("CARE_VISION_SPOOL_KEY must be valid base64")
        if len(key) != 32:
            parser.error("CARE_VISION_SPOOL_KEY must decode to 32 bytes")
        spool = EncryptedSpool(args.offline_spool, key)
    log = EventLog("edge", args.log)
    if args.camera is not None:
        source = CameraSource(
            args.camera,
            args.camera_seconds,
            args.capture_fps,
            source_width,
            source_height,
            args.record_camera,
        )
    elif args.rtsp_ip:
        username = os.environ.get("CARE_VISION_RTSP_USERNAME") or input("Camera account username: ")
        password = os.environ.get("CARE_VISION_RTSP_PASSWORD") or getpass.getpass(
            "Camera account password: "
        )
        source = RTSPSource(
            args.rtsp_ip,
            username,
            password,
            args.rtsp_stream,
            args.camera_seconds,
            args.capture_fps,
            source_width,
            source_height,
            ffmpeg_nobuffer=args.rtsp_ffmpeg_nobuffer,
            record=args.record_camera,
        )
        log.emit("rtsp_capture_configuration", ffmpeg_nobuffer=args.rtsp_ffmpeg_nobuffer)
    elif args.replay_captures:
        from pathlib import Path

        source = RecordedSource(
            args.video,
            args.replay_captures,
            args.capture_fps,
            source_width,
            source_height,
            Path(args.log).with_name("replay-captures.jsonl") if args.log else None,
        )
    else:
        source = VideoSource(args.video, args.capture_fps, source_width, source_height, spool)

    async def run():
        async with monitor_resources(log):
            edge = Edge(
                source,
                args.url,
                log,
                args.edge_id,
                output_width,
                output_height,
                initial_transmit_fps=args.initial_transmit_fps,
                replay_sampling_phase_ms=args.replay_sampling_phase_ms,
            )
            await edge.run()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        log.close()


if __name__ == "__main__":
    main()
