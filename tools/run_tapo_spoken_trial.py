"""Guide one bounded Tapo or Mac camera trial to Graviton with Spanish cues."""

import argparse
import getpass
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ADAPTIVE_CUES = [
    (0, "La captura comenzó. Permanecé quieto y con el cuerpo completo visible."),
    (11, "Ahora caminá normalmente de lado a lado durante unos segundos."),
    (21, "Detenete. Quedate quieto unos segundos."),
    (29, "Sentate normalmente en la silla."),
    (38, "Levantate normalmente y quedate de pie."),
    (46, "Quedate quieto hasta que termine la prueba."),
    (56, "Quedan cuatro segundos. Seguí quieto."),
]
FIXED_20_CUES = [
    (0, "Comenzó la prueba de transmisión fija a veinte cuadros por segundo. Quedate quieto."),
    (15, "Vamos por la mitad. Seguí quieto."),
    (26, "Quedan cuatro segundos. Seguí quieto."),
]
NATURAL_CUES = [
    (0, "Comenzó la prueba cotidiana. Quedate de pie y quieto, con todo el cuerpo visible."),
    (15, "Caminá a tu ritmo habitual por la escena y volvé a tu lugar."),
    (30, "Sentate normalmente en la silla y permanecé sentado."),
    (45, "Levantate normalmente y quedate de pie."),
    (55, "Caminá normalmente unos pasos y volvé a tu lugar."),
    (65, "Quedate quieto hasta el final. Ahora observamos si vuelve a la tasa basal."),
    (86, "Quedan cuatro segundos. Seguí quieto."),
]
MOTION_CUES = [
    (
        0,
        "Comenzó la prueba del ciclo adaptativo. Quedate quieto con el cuerpo visible. "
        "Prepará una mano cerca de la cámara.",
    ),
    (
        12,
        "Pasá la palma abierta frente al lente de lado a lado, sin tocarlo, hasta que te avise.",
    ),
    (
        32,
        "Alejá la mano y quedate quieto. "
        "Ahora mediremos si el envío vuelve a dos cuadros por segundo.",
    ),
    (45, "Seguimos observando. Quedate quieto hasta el final."),
    (56, "Quedan cuatro segundos. Seguí quieto."),
]
CONTROLLED_FALL_CUES = [
    (0, "Comenzó la prueba. Quedate de pie y con el cuerpo completo visible."),
    (12, "Caminá normalmente unos pasos y volvé al centro del encuadre."),
    (22, "Quedate quieto. Prepará el descenso controlado."),
    (
        28,
        "Si es seguro para vos, bajá lentamente hasta quedar acostado en el suelo. "
        "No te dejes caer.",
    ),
    (38, "Permanecé acostado y quieto unos segundos."),
    (48, "Incorporate despacio y volvé a estar de pie, si te resulta cómodo."),
    (56, "Quedan cuatro segundos. Quedate quieto."),
]

EPISODE_ALERT_CUES = [
    (0, "La captura comenzó. Quedate de pie, quieto y con el cuerpo completo visible."),
    (12, "Caminá unos pasos y volvé al centro del encuadre."),
    (22, "Ahora bajá de forma controlada hasta quedar acostado. No te dejes caer."),
    (35, "Permanecé acostado. Estamos observando la postura y esperando la alerta de prueba."),
    (60, "Seguí acostado hasta la indicación de incorporarte. La prueba todavía continúa."),
    (85, "Ahora incorporate despacio y volvé a estar de pie."),
    (105, "Quedate de pie hasta el final para observar la recuperación."),
    (116, "Quedan cuatro segundos."),
]


EPISODE_60_CUES = [
    (0, "Comenzó la captura. Quedate de pie y con el cuerpo completo visible."),
    (10, "Bajá de forma controlada hasta quedar acostado. No te dejes caer."),
    (22, "Permanecé acostado hasta la indicación de incorporarte."),
    (42, "Incorporate despacio y volvé a estar de pie."),
    (56, "Quedan cuatro segundos. Quedate de pie."),
]
RECOVERY_60_CUES = [
    (0, "Comenzó la prueba de recuperación. Quedate de pie con el cuerpo visible."),
    (10, "Bajá de forma controlada hasta quedar acostado. No te dejes caer."),
    (22, "Incorporate despacio y volvé a estar de pie."),
    (42, "Quedate de pie hasta el final. Observamos el cierre del mismo episodio."),
    (56, "Quedan cuatro segundos."),
]
NO_RECOVERY_60_CUES = [
    (0, "Comenzó la prueba sin recuperación. Quedate de pie con el cuerpo visible."),
    (10, "Bajá de forma controlada hasta quedar acostado. No te dejes caer."),
    (22, "Permanecé acostado hasta que termine la captura."),
    (42, "Seguimos observando la alerta experimental. Permanecé acostado."),
    (56, "Quedan cuatro segundos. Esperá el aviso de fin para incorporarte."),
]
SITTING_60_CUES = [
    (0, "Comenzó el control negativo. Quedate de pie con el cuerpo visible."),
    (10, "Sentate normalmente en la silla."),
    (22, "Permanecé sentado. Esta actividad no debería generar alerta."),
    (42, "Levantate normalmente y quedate de pie."),
    (56, "Quedan cuatro segundos. Quedate de pie."),
]


BENIGN_TRIALS = {
    "sit-return-60": (
        "stand, sit and remain seated, then return upright",
        [
            (0, "Comenzó la prueba. Quedate de pie junto a la silla, con todo el cuerpo visible."),
            (10, "Sentate normalmente en la silla."),
            (22, "Permanecé sentado hasta que te indique levantarte."),
            (42, "Ahora levantate normalmente y permanecé de pie junto a la silla."),
            (56, "Quedan cuatro segundos. Seguí de pie."),
        ],
    ),
    "seated-hold-60": (
        "sit in chair and remain seated through capture end",
        [
            (0, "Comenzó la prueba. Quedate de pie junto a la silla, con todo el cuerpo visible."),
            (10, "Sentate normalmente en la silla."),
            (22, "Permanecé sentado a tu ritmo habitual. No hace falta levantarse."),
            (42, "Seguí sentado hasta el aviso de fin."),
            (56, "Quedan cuatro segundos. Seguí sentado."),
        ],
    ),
    "crouch-near-seat-60": (
        "crouch beside chair without sitting; return upright",
        [
            (0, "Comenzó la prueba. Quedate de pie al lado de la silla, con el cuerpo visible."),
            (10, "Agachate al lado de la silla, sin sentarte ni acostarte, si te resulta cómodo."),
            (18, "Volvé a estar de pie a tu ritmo."),
            (30, "Repetí el agachamiento junto a la silla, sin sentarte."),
            (38, "Volvé a estar de pie a tu ritmo."),
            (56, "Quedan cuatro segundos. Quedate de pie."),
        ],
    ),
    "kneel-near-seat-60": (
        "kneel beside chair without sitting in it; return upright",
        [
            (0, "Comenzó la prueba. Quedate de pie junto a la silla, con el cuerpo visible."),
            (10, "Arrodillate despacio al lado de la silla, sin sentarte en ella, si es cómodo."),
            (18, "Incorporate despacio a tu ritmo."),
            (30, "Quedate de pie. No hace falta repetir el movimiento."),
            (56, "Quedan cuatro segundos. Quedate de pie."),
        ],
    ),
    "bend-near-seat-60": (
        "bend beside chair to reach an object; return upright",
        [
            (0, "Comenzó la prueba. Quedate de pie junto a la silla, con todo el cuerpo visible."),
            (10, "Inclinate para alcanzar un objeto junto a la silla, sin sentarte."),
            (18, "Volvé a estar de pie a tu ritmo."),
            (30, "Repetí el movimiento para alcanzar el objeto, sin sentarte."),
            (38, "Volvé a estar de pie a tu ritmo."),
            (56, "Quedan cuatro segundos. Quedate de pie."),
        ],
    ),
}


def benign_trial_plan(mode, split):
    description, cues = BENIGN_TRIALS[mode]
    return {
        "mode": mode,
        "scenario": description,
        "dataset_split": split,
        "duration_seconds": 60,
        "planned_cues": cues,
        "cues_are_instructions_not_ground_truth": True,
        "capture_fps_requested": 25,
        "baseline_transmit_fps": 2,
        "verification_transmit_fps": 5,
        "baseline_resolution": [1280, 720],
        "person_detector": "YOLOX-S640",
        "person_threshold": 0.35,
        "controller": "existing S-only policy; no seated closure hook",
        "seated_shadow_enabled_live": False,
        "experimental_alert_after_seconds": 20,
        "notifications_sent_by_design": False,
        "requires_video_review": True,
    }


def first_capture_ns(path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # The writer may still be finishing this line.
            if row.get("event") == "frame_sent":
                return row["captured_mono_ns"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rtsp-ip")
    source.add_argument("--camera", type=int, help="Local Mac camera index")
    parser.add_argument("--rtsp-ffmpeg-nobuffer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "adaptive",
            "fixed20",
            "motion",
            "natural",
            "controlled-fall",
            "episode-alert",
            "episode-60",
            "sitting-60",
            "recovery-60",
            "no-recovery-60",
            *BENIGN_TRIALS,
        ],
        default="adaptive",
    )
    parser.add_argument("--record-rtsp", action="store_true")
    parser.add_argument("--experimental-weak-association", action="store_true")
    parser.add_argument(
        "--dataset-split", choices=["development", "evaluation_reserved"], default="development"
    )
    parser.add_argument(
        "--plan-only", action="store_true", help="Print a new benign trial plan only"
    )
    parser.add_argument(
        "--evaluation-freeze",
        type=Path,
        help="Verify a frozen shadow evaluation before any camera/audio activity",
    )
    args = parser.parse_args()
    evaluation_freeze_sha256 = evaluation_freeze_version = None
    if args.evaluation_freeze:
        from prepare_seated_pose_reserved import verify_freeze

        if args.dataset_split != "evaluation_reserved" or args.mode not in BENIGN_TRIALS:
            parser.error("evaluation freeze requires a reserved benign trial")
        frozen = json.loads(args.evaluation_freeze.read_text())
        verify_freeze(frozen, Path(__file__).resolve().parents[1])
        profile = frozen["capture_profile"]
        if args.experimental_weak_association != profile["weak_association"]:
            parser.error("weak-association option differs from frozen capture profile")
        if args.rtsp_ffmpeg_nobuffer != profile["rtsp_ffmpeg_nobuffer"]:
            parser.error("RTSP buffering option differs from frozen capture profile")
        import hashlib

        evaluation_freeze_sha256 = hashlib.sha256(args.evaluation_freeze.read_bytes()).hexdigest()
        evaluation_freeze_version = frozen["version"]
    episode_modes = {"episode-alert", "episode-60", "sitting-60", "recovery-60", "no-recovery-60"}
    episode_modes |= BENIGN_TRIALS.keys()
    if args.plan_only:
        if args.mode not in BENIGN_TRIALS:
            parser.error("--plan-only requires a benign activity trial mode")
        plan = benign_trial_plan(args.mode, args.dataset_split)
        plan["experimental_weak_association"] = args.experimental_weak_association
        plan["evaluation_freeze_sha256"] = evaluation_freeze_sha256
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.mode in BENIGN_TRIALS and (args.rtsp_ip is None or not args.record_rtsp):
        parser.error("benign activity trials require --rtsp-ip and --record-rtsp for visual review")
    if args.experimental_weak_association and args.mode not in episode_modes:
        parser.error("weak association requires an S-only episode trial mode")
    if sys.platform != "darwin":
        parser.error("spoken cues require macOS")
    if args.rtsp_ffmpeg_nobuffer and args.rtsp_ip is None:
        parser.error("--rtsp-ffmpeg-nobuffer requires --rtsp-ip")
    if not args.identity.is_file():
        parser.error("SSH identity file does not exist")
    if args.output.exists():
        parser.error("output directory must be new")
    root = Path(__file__).resolve().parents[1]
    model = root / "artifacts/phase5g-compact-detectors-20260920/models/yolox_nano.onnx"
    if not model.is_file():
        parser.error("YOLOX Nano model is missing")
    verification_model = (
        root / "artifacts/phase5c-dataset-review-20260920/models/"
        "object_detection_yolox_2022nov.onnx"
    )
    if args.mode in episode_modes | {"controlled-fall"} and not verification_model.is_file():
        parser.error("YOLOX-S verification model is missing")

    env = os.environ.copy()
    if args.rtsp_ip is not None:
        username = os.environ.get("CARE_VISION_RTSP_USERNAME") or input("Camera account username: ")
        password = os.environ.get("CARE_VISION_RTSP_PASSWORD") or getpass.getpass(
            "Camera account password: "
        )
        env["CARE_VISION_RTSP_USERNAME"] = username
        env["CARE_VISION_RTSP_PASSWORD"] = password
        del password
    subprocess.run(
        ["/usr/bin/say", "-v", "Paulina", "Prueba de audio. Prepará el encuadre."],
        check=True,
    )
    if args.mode == "controlled-fall":
        print("Prepare a safe, clear area for one slow, voluntary descent to the floor.")
    else:
        print("Place yourself in frame before capture. No fall or abrupt descent is needed.")
    print("Starting the cloud service; spoken cues will begin with real camera capture.")
    output = args.output.resolve()
    cues_by_mode = {
        "adaptive": ADAPTIVE_CUES,
        "fixed20": FIXED_20_CUES,
        "motion": MOTION_CUES,
        "natural": NATURAL_CUES,
        "controlled-fall": CONTROLLED_FALL_CUES,
        "episode-alert": EPISODE_ALERT_CUES,
        "episode-60": EPISODE_60_CUES,
        "sitting-60": SITTING_60_CUES,
        "recovery-60": RECOVERY_60_CUES,
        "no-recovery-60": NO_RECOVERY_60_CUES,
    }
    cues_by_mode.update({mode: trial[1] for mode, trial in BENIGN_TRIALS.items()})
    cues = cues_by_mode[args.mode]
    seconds = {"fixed20": 30, "natural": 90, "episode-alert": 120}.get(args.mode, 60)
    if any(at_seconds >= seconds for at_seconds, _ in cues):
        parser.error("a spoken cue is scheduled after camera capture ends")
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
        "--camera-seconds",
        str(seconds),
        "--capture-fps",
        "25",
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
        str(output),
    ]
    if args.rtsp_ip is not None:
        command.extend(["--rtsp-ip", args.rtsp_ip, "--rtsp-stream", "1"])
    else:
        command.extend(["--camera", str(args.camera)])
    if args.mode == "fixed20":
        command.extend(["--initial-transmit-fps", "20"])
    if args.rtsp_ffmpeg_nobuffer:
        command.append("--rtsp-ffmpeg-nobuffer")
    if args.mode == "controlled-fall":
        command.extend(
            ["--experimental-loss-5fps", "--verification-model", str(verification_model)]
        )
    if args.record_rtsp:
        command.append("--record-rtsp")
    if args.mode in episode_modes:
        # Keep the measured S-only representation and detector settings.
        for flag, value in (
            ("--baseline-resolution", "1280x720"),
            ("--person-model", str(verification_model)),
            ("--person-threshold", "0.35"),
            ("--person-input-size", "640"),
        ):
            command[command.index(flag) + 1] = value
        command.extend(["--experimental-s-only-5fps", "--experimental-alert-after-seconds", "20"])
    if args.experimental_weak_association:
        command.append("--experimental-weak-association")
    lines = queue.Queue()
    process = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def relay():
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.put(line)

    reader = threading.Thread(target=relay, daemon=True)
    reader.start()
    capture_start = None
    capture_end_announced = False
    next_cue = 0
    voice = None
    cue_events = []
    deadline = time.monotonic() + 600
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("trial exceeded 10 minutes")
            while not lines.empty():
                line = lines.get_nowait()
                if "CAMERA_START_IN_5_SECONDS" in line:
                    subprocess.Popen(
                        [
                            "/usr/bin/say",
                            "-v",
                            "Paulina",
                            "-r",
                            "170",
                            "La cámara comienza en cinco segundos. Quedate quieto.",
                        ]
                    )
            if capture_start is None:
                capture_start = first_capture_ns(output / "edge.jsonl")
            if capture_start is not None and next_cue < len(cues):
                at_seconds, spoken = cues[next_cue]
                elapsed = (time.monotonic_ns() - capture_start) / 1e9
                if at_seconds <= elapsed < seconds:
                    if voice is not None and voice.poll() is None:
                        voice.terminate()
                        voice.wait(timeout=3)
                    voice = subprocess.Popen(
                        [
                            "/usr/bin/say",
                            "-v",
                            "Paulina",
                            "-r",
                            "145",
                            spoken,
                        ]
                    )
                    cue_events.append(
                        {
                            "planned_seconds": at_seconds,
                            "actual_seconds": round(elapsed, 3),
                            "text": spoken,
                        }
                    )
                    print(f"Cue {at_seconds}s: {spoken}", flush=True)
                    next_cue += 1
            if capture_start is not None and not capture_end_announced:
                edge_log = output / "edge.jsonl"
                if edge_log.exists() and '"event": "drain_started"' in edge_log.read_text():
                    if voice is not None and voice.poll() is None:
                        voice.terminate()
                        voice.wait(timeout=3)
                    voice = subprocess.Popen(
                        [
                            "/usr/bin/say",
                            "-v",
                            "Paulina",
                            "Terminó la captura. Podés incorporarte despacio y descansar.",
                        ]
                    )
                    capture_end_announced = True
            time.sleep(0.1)
        reader.join(timeout=2)
        if output.exists():
            (output / "spoken-cues.json").write_text(
                json.dumps(
                    {
                        "scenario": (
                            {
                                "adaptive": "normal activity; no simulated fall",
                                "fixed20": "fixed 20 FPS transport/load diagnostic; remain still",
                                "motion": (
                                    "near-lens hand motion to exercise adaptive sensing; no fall"
                                ),
                                "natural": (
                                    "daily activity: walking, sitting, standing, quiet recovery; "
                                    "no fall"
                                ),
                                "episode-60": (
                                    "controlled descent, horizontal posture, recovery; no impact"
                                ),
                                "sitting-60": "negative control: sitting and standing; no fall",
                                "recovery-60": (
                                    "controlled horizontal posture and early recovery; no impact"
                                ),
                                "no-recovery-60": (
                                    "controlled horizontal posture until capture ends; "
                                    "test alert only"
                                ),
                                "episode-alert": (
                                    "controlled horizontal posture, test alert, then recovery; "
                                    "no impact"
                                ),
                                "controlled-fall": (
                                    "one voluntary controlled descent to the floor, "
                                    "lying still, then rising"
                                ),
                            }.get(args.mode, BENIGN_TRIALS.get(args.mode, (None,))[0])
                        ),
                        "capture_start_mono_ns": capture_start,
                        "planned_cues": cues,
                        "delivered_cues": cue_events,
                        "audio_recorded": False,
                        "source_kind": "rtsp_camera" if args.rtsp_ip is not None else "mac_camera",
                        "capture_fps_requested": 25,
                        "dataset_split": args.dataset_split,
                        "evaluation_freeze_sha256": evaluation_freeze_sha256,
                        "evaluation_freeze_version": evaluation_freeze_version,
                        "cues_are_instructions_not_ground_truth": True,
                        "benign_trial_plan": (
                            benign_trial_plan(args.mode, args.dataset_split)
                            if args.mode in BENIGN_TRIALS
                            else None
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
        if process.returncode:
            raise RuntimeError(f"cloud trial failed; inspect {output}")
        print(f"Spoken trial complete: {output}")
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
