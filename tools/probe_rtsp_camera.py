"""Check a local RTSP camera without contacting AWS or printing credentials."""

import argparse
import getpass
import os
import time

from care_vision.edge.source import RTSPSource


def main():
    parser = argparse.ArgumentParser(description="Probe a local RTSP camera")
    parser.add_argument("--ip", required=True, help="camera IPv4 address")
    parser.add_argument("--stream", type=int, choices=[1, 2], default=1)
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    username = os.environ.get("CARE_VISION_RTSP_USERNAME") or input(
        "Camera account username: "
    )
    password = os.environ.get("CARE_VISION_RTSP_PASSWORD") or getpass.getpass(
        "Camera account password: "
    )
    source = RTSPSource(
        args.ip, username, password, stream=args.stream, seconds=args.seconds,
        width=1280, height=720,
    )
    started = time.monotonic()
    source.start()
    try:
        source.thread.join(timeout=args.seconds + 15)
        if source.thread.is_alive():
            raise RuntimeError("RTSP read did not finish in time")
        _, count, _, error = source.snapshot()
        if error:
            raise RuntimeError(error)
        duration = time.monotonic() - started
        status = source.buffer_status()
        span_ns = (
            status["buffer_newest_mono_ns"] - status["buffer_oldest_mono_ns"]
            if status["buffer_frames"] >= 2 else 0
        )
        observed_fps = (
            (status["buffer_frames"] - 1) * 1e9 / span_ns if span_ns > 0 else None
        )
        print(f"Frames captured: {count}")
        print(f"Wall time including connection: {duration:.1f} s")
        if observed_fps is not None:
            print(f"Capture rate while receiving: {observed_fps:.1f} FPS")
        print(f"Ring buffer: {status['buffer_frames']} frames, {status['buffer_bytes']} bytes")
        print("No frames were sent to AWS.")
    finally:
        source.close()


if __name__ == "__main__":
    main()
