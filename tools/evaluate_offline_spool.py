"""Verify encrypted offline samples survive two separate Edge processes."""

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from care_vision.check_loop import read_log
from care_vision.edge.spool import EncryptedSpool
from care_vision.fixture import make_video
from care_vision.temporal_evaluation import file_sha256


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def offline_run(folder, video, index, key):
    log_path = folder / f"edge-{index}.jsonl"
    environment = {**os.environ, "CARE_VISION_SPOOL_KEY": base64.b64encode(key).decode()}
    with (folder / f"edge-{index}.stderr").open("w") as stderr:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "care_vision.edge",
                "--video",
                str(video),
                "--url",
                f"ws://127.0.0.1:{free_port()}",
                "--offline-spool",
                str(folder / "private-spool"),
                "--log",
                str(log_path),
            ],
            stdout=stderr,
            stderr=subprocess.STDOUT,
            env=environment,
            timeout=22,
            check=False,
        )
    records = read_log(log_path)
    stopped = next(r for r in records if r["event"] == "source_stopped")
    if result.returncode == 0 or not any(
        r["event"] == "cloud_processing_unconfirmed_at_eof" for r in records
    ):
        raise AssertionError("offline process did not fail explicitly at EOF")
    return stopped["captured"], records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    folder = args.output.resolve()
    folder.mkdir(parents=True)
    first_video = make_video(folder / "first.avi", static=True, seconds=12)
    second_video = make_video(folder / "second.avi", static=True, seconds=2)
    key = os.urandom(32)
    first_count, first = offline_run(folder, first_video, 1, key)
    second_count, second = offline_run(folder, second_video, 2, key)
    spool = EncryptedSpool(folder / "private-spool", key)
    try:
        status = spool.status()
        old = spool.select(
            status["spool_oldest_mono_ns"],
            status["spool_newest_mono_ns"] + 1,
            30,
        )
        first_boot = next(r["boot_id"] for r in first if r["event"] == "offline")
        second_boot = next(r["boot_id"] for r in second if r["event"] == "offline")
        second_started_ns = min(r["local_mono_ns"] for r in second if r["event"] == "edge_status")
        retained_before_restart = [c for c in old if c.mono_ns < second_started_ns]
        if not (
            first_count == 360
            and second_count == 420
            and first_boot == second_boot == spool.boot_id
            and spool.last_seq == 420
            and 20 <= status["spool_frames"] <= 31
            and retained_before_restart
            and all(c.jpeg.startswith(b"\xff\xd8") for c in old)
            and status["spool_oldest_mono_ns"] < second_started_ns - 10_000_000_000
        ):
            raise AssertionError("offline evidence was not retained across Edge processes")
    finally:
        spool.close()
    names = (
        "first.avi",
        "second.avi",
        "edge-1.jsonl",
        "edge-2.jsonl",
        "private-spool/offline.sqlite3",
    )
    report = {
        "scope": "Two separate local Edge processes; cloud unreachable in both",
        "first_capture_frames": first_count,
        "second_cumulative_capture_frames": second_count,
        "spool_frames_after_restart": status["spool_frames"],
        "spool_encrypted_bytes": status["spool_encrypted_bytes"],
        "frames_from_first_process_still_decryptable": len(retained_before_restart),
        "oldest_spool_frame_more_than_10_seconds_before_second_process": True,
        "boot_identity_persisted": True,
        "cloud_processing_unconfirmed_at_eof": True,
        "limits": [
            "The key is ephemeral and not saved with the evidence artifact.",
            "This verifies local persistence; the cloud does not yet request older spool windows.",
            "Process restart was tested, not an operating-system reboot or power failure.",
        ],
        "file_sha256": {name: file_sha256(folder / name) for name in names},
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
