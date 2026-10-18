"""Record bounded dense RTSP evidence and capture timestamps, without inference."""

import argparse
import getpass
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import cv2
from run_tapo_spoken_trial import NATURAL_CUES

from care_vision.common.contracts import utc
from care_vision.edge.source import RTSPSource

CONTROLLED_CUES = [
    (0, "Comenzó la prueba. Quedate de pie y quieto, con todo el cuerpo visible."),
    (15, "Caminá normalmente unos pasos y volvé al lugar preparado."),
    (30, "Quedate de pie y quieto. Preparáte para bajar lentamente al suelo."),
    (40, "Bajá lentamente al suelo y acostate de costado. No te dejes caer."),
    (60, "Permanecé acostado y visible hasta la siguiente indicación."),
    (85, "Incorporate despacio, a tu ritmo, y quedate de pie."),
    (105, "Quedate de pie y quieto hasta el final."),
    (116, "Quedan cuatro segundos. Seguí quieto."),
]
LATERAL_CUES = [
    (0, "Comenzó la prueba. Quedate de pie y quieto, con todo el cuerpo visible."),
    (15, "Caminá normalmente hacia otra zona despejada del encuadre y quedate allí."),
    (30, "Quedate quieto. Preparáte para bajar lentamente en esta nueva posición."),
    (40, "Bajá lentamente al suelo y acostate de costado. No te dejes caer."),
    (60, "Permanecé acostado y visible hasta la siguiente indicación."),
    (85, "Incorporate despacio, a tu ritmo, y quedate de pie."),
    (105, "Quedate de pie y quieto hasta el final."),
    (116, "Quedan cuatro segundos. Seguí quieto."),
]
DAILY_NEGATIVE_CUES = [
    (0, "Comenzó la prueba cotidiana. Quedate de pie y quieto, con todo el cuerpo visible."),
    (15, "Caminá a tu ritmo habitual unos pasos y volvé."),
    (30, "Sentate normalmente en la silla y permanecé sentado."),
    (50, "Levantate despacio y quedate de pie."),
    (65, "Agachate normalmente, sin acostarte, y volvé a levantarte."),
    (85, "Caminá normalmente unos pasos y volvé."),
    (105, "Quedate quieto hasta el final."),
    (116, "Quedan cuatro segundos. Seguí quieto."),
]
SCENARIOS = {
    "natural": (90, NATURAL_CUES),
    "controlled_episode": (120, CONTROLLED_CUES),
    "controlled_lateral": (120, LATERAL_CUES),
    "daily_negative": (120, DAILY_NEGATIVE_CUES),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtsp-ip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtsp-ffmpeg-nobuffer", action="store_true")
    parser.add_argument("--scenario", choices=SCENARIOS, default="natural")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    seconds, cues = SCENARIOS[args.scenario]
    username = os.getenv("CARE_VISION_RTSP_USERNAME") or input("Camera account username: ")
    password = os.getenv("CARE_VISION_RTSP_PASSWORD") or getpass.getpass(
        "Camera account password: "
    )
    source = RTSPSource(
        args.rtsp_ip,
        username,
        password,
        seconds=seconds,
        fps=25,
        ffmpeg_nobuffer=args.rtsp_ffmpeg_nobuffer,
    )
    args.output.mkdir(parents=True)
    video_path = args.output / "dense.avi"
    manifest = {
        "schema_version": 1,
        "scenario": args.scenario,
        "duration_requested_seconds": seconds,
        "physical_action": "controlled activity only; never fall or drop",
        "cues_are_instructions_not_ground_truth": True,
        "capture_fps_target": 25,
        "ffmpeg_nobuffer": args.rtsp_ffmpeg_nobuffer,
        "inference_performed": False,
        "audio_recorded": False,
        "timestamps_scope": "Edge decoded-frame read, not camera exposure",
        "avi_fps_is_nominal": True,
        "planned_cues": cues,
        "delivered_cues": [],
        "complete": False,
    }
    cap = writer = voice = None
    rows = []
    error = None
    try:
        subprocess.run(
            [
                "/usr/bin/say",
                "-v",
                "Paulina",
                "Prueba de audio. Prepará el encuadre. Grabaremos "
                + ("noventa" if seconds == 90 else "ciento veinte")
                + " segundos.",
            ],
            check=True,
        )
        cap = source._open()
        if not cap.isOpened():
            raise RuntimeError("cannot open RTSP camera")
        subprocess.run(
            [
                "/usr/bin/say",
                "-v",
                "Paulina",
                "La grabación comienza en cinco segundos. Quedate quieto.",
            ],
            check=True,
        )
        # Drain RTSP during countdown, retaining only frames from the recorded scene.
        until = time.monotonic() + 5
        while time.monotonic() < until:
            if not cap.read()[0]:
                raise RuntimeError("RTSP read failed during countdown")
        started = None
        next_cue = 0
        with (args.output / "captures.jsonl").open("x") as log:
            while started is None or time.monotonic_ns() - started < seconds * 1_000_000_000:
                ok, image = cap.read()
                captured = time.monotonic_ns()
                if not ok:
                    raise RuntimeError("RTSP read failed before scene completed")
                if started is None:
                    height, width = image.shape[:2]
                    manifest["source_width"] = width
                    manifest["source_height"] = height
                    manifest["recording_resolution"] = [width, height]
                    print(f"Recording native RTSP resolution: {width}x{height}", flush=True)
                    started = captured
                    writer = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError("cannot open dense video writer")
                elif image.shape[:2] != (height, width):
                    raise RuntimeError("RTSP resolution changed during recording")
                row = {
                    "capture_seq": len(rows) + 1,
                    "captured_mono_ns": captured,
                    "captured_at_utc": utc(),
                }
                writer.write(image)
                log.write(json.dumps(row) + "\n")
                rows.append(row)
                elapsed = (captured - started) / 1e9
                if next_cue < len(cues) and elapsed >= cues[next_cue][0]:
                    planned, spoken = cues[next_cue]
                    if voice is not None and voice.poll() is None:
                        voice.terminate()
                        voice.wait(timeout=3)
                    voice = subprocess.Popen(["/usr/bin/say", "-v", "Paulina", "-r", "145", spoken])
                    manifest["delivered_cues"].append(
                        {"planned_seconds": planned, "actual_seconds": elapsed, "text": spoken}
                    )
                    print(f"Cue {planned}s: {spoken}", flush=True)
                    next_cue += 1
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if voice is not None and voice.poll() is None:
            voice.terminate()
            voice.wait(timeout=3)
        decoded = 0
        if video_path.exists():
            check = cv2.VideoCapture(str(video_path))
            while True:
                ok, decoded_image = check.read()
                if not ok:
                    break
                if decoded_image.shape[:2] != (manifest["source_height"], manifest["source_width"]):
                    error = "RecordedResolutionMismatch"
                decoded += 1
            check.release()
            manifest["video_sha256"] = hashlib.sha256(video_path.read_bytes()).hexdigest()
        intervals = [
            (b["captured_mono_ns"] - a["captured_mono_ns"]) / 1e9 for a, b in zip(rows, rows[1:])
        ]
        manifest.update(
            {
                "recorded_frames": len(rows),
                "decoded_frames": decoded,
                "capture_fps_median": 1 / statistics.median(intervals) if intervals else None,
                "maximum_capture_gap_seconds": max(intervals, default=None),
                "duration_seconds": sum(intervals),
                "error_type": error,
                "captures_sha256": hashlib.sha256(
                    (args.output / "captures.jsonl").read_bytes()
                ).hexdigest()
                if rows
                else None,
            }
        )
        manifest["complete"] = (
            error is None
            and decoded == len(rows)
            and decoded > 0
            and sum(intervals) >= seconds - 1
            and len(manifest["delivered_cues"]) == len(cues)
        )
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["complete"]:
        raise RuntimeError("dense capture verification failed; preserve and inspect artifacts")
    print(json.dumps(manifest, indent=2))
    print(f"Dense capture verified: {args.output.resolve()}")


if __name__ == "__main__":
    main()
