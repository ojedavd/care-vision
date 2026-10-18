"""Run one bounded, spoken Mac-camera trial through the existing Graviton loop.

This runner does not classify the video. The usual cloud pipeline runs live; the
experimental visibility/recovery variant can evaluate the saved video later.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

SCENARIOS = {
    "reserved_episode": {
        "seconds": 120,
        "cues": [
            (
                0,
                "baseline",
                "Comienza la grabación. Quedate de pie, quieto y con todo el cuerpo visible.",
            ),
            (15, "sit", "Sentate normalmente en la silla."),
            (35, "stand", "Levantate normalmente y quedate de pie."),
            (50, "bend", "Inclinate para recoger un objeto liviano y volvé a estar de pie."),
            (68, "prepare", "Quedate de pie. Preparáte para un descenso lento y controlado."),
            (75, "controlled_descent", "Bajá lentamente al suelo. No te dejes caer."),
            (90, "floor_rest", "Permanecé acostado hasta la siguiente indicación."),
            (101, "rise", "Incorporate despacio y quedate de pie dentro del encuadre."),
            (115, "finish", "Quedate de pie unos segundos más."),
        ],
    },
    "negative": {
        "seconds": 75,
        "cues": [
            (0, "baseline", "Comienza la grabación. Quedate de pie con todo el cuerpo visible."),
            (12, "sit", "Sentate normalmente en la silla y permanecé sentado."),
            (27, "stand", "Levantate normalmente y quedate de pie."),
            (38, "bend", "Inclinate para recoger un objeto liviano y volvé a estar de pie."),
            (50, "walk", "Caminá despacio por detrás de la silla y volvé a tu lugar."),
            (63, "baseline", "Quedate de pie y quieto hasta el final."),
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Current, previously trusted user@host")
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New trial directory")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("physical Mac camera and spoken cues require macOS")
    if not args.identity.is_file():
        parser.error("SSH identity file does not exist")
    if args.output.exists():
        parser.error("output directory must be new")

    root = Path(__file__).resolve().parents[1]
    model = root / "artifacts/phase5g-compact-detectors-20260920/models/yolox_nano.onnx"
    if not model.is_file():
        parser.error("YOLOX Nano model is missing")
    scenario = SCENARIOS[args.scenario]
    output = args.output.resolve()
    output.mkdir(parents=True)
    run = output / "live"
    frozen_files = [
        *sorted((root / "src/care_vision/cloud").glob("*.py")),
        root / "tools/occlusion_command_cloud.py",
        root / "tools/evaluate_visibility_command_loop.py",
    ]
    protocol = {
        "scenario": args.scenario,
        "planned_duration_seconds": scenario["seconds"],
        "capture_fps": 30,
        "capture_resolution": "1280x720",
        "baseline_transmit_resolution": "320x180",
        "audio_recorded": False,
        "alerts_enabled": False,
        "physical_action": "controlled descent only; never fall or drop",
        "cues_are_instructions_not_ground_truth": True,
        "live_pipeline": "standard Graviton/OpenCV 5+COOL controller, not experimental V4",
        "experimental_evaluation": "saved video after live trial; no tuning on this recording",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "frozen_source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in frozen_files
        },
        "cues": [
            {"at_seconds": at, "action": action, "text": spoken}
            for at, action, spoken in scenario["cues"]
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    command = [
        sys.executable,
        "-u",
        "-m",
        "care_vision.aws_demo",
        "--host",
        args.host,
        "--identity",
        str(args.identity),
        "--phase",
        "3",
        "--camera",
        "0",
        "--camera-seconds",
        str(scenario["seconds"]),
        "--tcp-telemetry",
        "--capture-resolution",
        "1280x720",
        "--baseline-resolution",
        "320x180",
        "--person-model",
        str(model),
        "--person-detector",
        "yolox",
        "--person-threshold",
        "0.4",
        "--person-input-size",
        "416",
        "--output",
        str(run),
    ]
    capture_start = None
    next_cue = 0
    voice = None
    deadline = time.monotonic() + 600
    with (
        (output / "runner.log").open("x") as log,
        (output / "cue-events.jsonl").open("x") as events,
    ):
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("trial exceeded 10-minute bound")
                capture_log = run / "demo.captures.jsonl"
                if capture_start is None and capture_log.exists():
                    with capture_log.open() as source:
                        first = source.readline()
                    if first.endswith("\n"):
                        capture_start = json.loads(first)["captured_mono_ns"]
                        print("Camera capture started; follow the spoken cues.", flush=True)
                if capture_start is not None and next_cue < len(scenario["cues"]):
                    at, action, spoken = scenario["cues"][next_cue]
                    now = time.monotonic_ns()
                    if (now - capture_start) / 1e9 >= at:
                        if voice is not None and voice.poll() is None:
                            voice.terminate()
                            voice.wait(timeout=3)
                        record = {
                            "event": "spoken_cue",
                            "action": action,
                            "planned_seconds": at,
                            "actual_seconds": (now - capture_start) / 1e9,
                            "local_mono_ns": now,
                            "text": spoken,
                        }
                        events.write(json.dumps(record, ensure_ascii=False) + "\n")
                        events.flush()
                        voice = subprocess.Popen(
                            ["/usr/bin/say", "-v", "Paulina", "-r", "145", spoken]
                        )
                        print(f"{action}: {spoken}", flush=True)
                        next_cue += 1
                time.sleep(0.1)
            if voice is not None:
                voice.wait(timeout=15)
            (output / "runner-result.json").write_text(
                json.dumps(
                    {
                        "returncode": process.returncode,
                        "capture_started": capture_start is not None,
                        "cues_started": next_cue,
                        "live_report_exists": (run / "report.json").exists(),
                    },
                    indent=2,
                )
                + "\n"
            )
            if process.returncode:
                raise RuntimeError(f"live trial failed; inspect {output / 'runner.log'}")
            print(f"Live trial complete: {run}", flush=True)
        finally:
            if voice is not None and voice.poll() is None:
                voice.terminate()
                voice.wait(timeout=3)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
