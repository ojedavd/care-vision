"""Sequential causal 2→5→2 replay for a lost Nano track, with S on Graviton.

This experimental harness is not the production WebSocket service or a fall classifier.
"""

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def worker(base):
    import cv2
    import numpy as np

    from care_vision.cloud.loss_probe import LossProbe
    from care_vision.cloud.person import observe_people
    from care_vision.cloud.tracking import PersonTracker
    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.contracts import envelope, uid, utc, validate
    from care_vision.common.environment import runtime_info

    config = json.loads((base / "worker-config.json").read_text())
    cv2.setNumThreads(config["opencv_threads"])
    detectors = {}
    for name, profile in config["profiles"].items():
        path = base / profile["filename"]
        assert sha(path.read_bytes()) == profile["sha256"]
        detectors[name] = YoloXPersonDetector(
            path,
            input_size=profile["input_size"],
            score_threshold=profile["threshold"],
            nms_threshold=0.3,
        )
    tracker, probe = PersonTracker(), LossProbe()
    mode, verification_start = "baseline", None
    first_s_top = maximum_s_top = None
    commands, rows, pending = [], [], {}
    command_seq = 0
    origin = None

    def reply(value):
        print(json.dumps(value), flush=True)

    reply(
        {
            "event": "ready",
            "runtime": runtime_info(3),
            "opencv_num_threads": cv2.getNumThreads(),
            "model_hashes": {
                name: profile["sha256"] for name, profile in config["profiles"].items()
            },
        }
    )
    for line in sys.stdin:
        packet = json.loads(line)
        if packet.get("control") == "finish":
            assert not pending
            reply(
                {"event": "finished", "frames": len(rows), "commands": len(commands), "mode": mode}
            )
            break
        if packet.get("type") == "COMMAND_ACK":
            validate(packet)
            command = pending.pop(packet["command_id"])
            assert packet["command_type"] == command["type"] and packet["result"] == "applied"
            if command["type"] == "SET_TRANSMIT_FPS":
                assert packet["active_transmit_fps"] == command["transmit_fps"]
            reply({"acknowledged": packet["command_id"]})
            continue
        assert not pending
        header, jpeg = packet["header"], base64.b64decode(packet["jpeg"], validate=True)
        validate(header)
        assert sha(jpeg) == packet["sha256"]
        assert header["stream_role"] == mode
        expected_size = (320, 180) if mode == "baseline" else (1280, 720)
        assert (header["width"], header["height"]) == expected_size
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.shape[:2] == expected_size[::-1]
        if origin is None:
            origin = header["captured_mono_ns"]
        t = (header["captured_mono_ns"] - origin) / 1e9
        detector = detectors["nano" if mode == "baseline" else "s"]
        people = observe_people(detector, image, header["frame_id"])
        assert people["person_detection_status"] == "ok"
        issued = []
        observation_id = uid()
        identity = {key: header[key] for key in ("edge_id", "boot_id", "session_id")}

        def issue(kind, reason, **fields):
            nonlocal command_seq
            command_seq += 1
            command = envelope(
                kind,
                identity,
                command_id=uid(),
                command_seq=command_seq,
                issued_at_utc=utc(),
                caused_by_observation_id=observation_id,
                based_on_frame_id=header["frame_id"],
                valid_until_edge_mono_ns=header["captured_mono_ns"] + 5_000_000_000,
                reason=reason,
                **fields,
            )
            validate(command)
            issued.append(command)
            commands.append(command)
            pending[command["command_id"]] = command

        if mode == "baseline":
            people = tracker.update(people, header["captured_mono_ns"], "loss-escalation")
            lost = probe.observe(header["captured_mono_ns"], people)
            if lost:
                issue("SET_TRANSMIT_FPS", "confirmed_person_missing_verify", transmit_fps=5)
                issue(
                    "REQUEST_FULL_FRAME",
                    "confirmed_person_missing_need_scene_context",
                    output_width=1280,
                    output_height=720,
                )
                mode, verification_start = "verification", header["captured_mono_ns"]
                first_s_top = maximum_s_top = None
        else:
            box = people["primary_person_bbox"]
            if box is not None:
                if first_s_top is None:
                    first_s_top = box[1]
                maximum_s_top = max(maximum_s_top or box[1], box[1])
            elapsed = (header["captured_mono_ns"] - verification_start) / 1e9
            downward_extension = (
                config["extend_on_downward"]
                and first_s_top is not None
                and maximum_s_top - first_s_top >= 150
            )
            limit = 10 if downward_extension else config["verification_seconds"]
            if elapsed >= limit:
                issue(
                    "REQUEST_FULL_FRAME",
                    "bounded_verification_ended",
                    output_width=320,
                    output_height=180,
                )
                issue("SET_TRANSMIT_FPS", "return_to_baseline_after_evidence", transmit_fps=2)
                mode, verification_start = "baseline", None

        row = {
            "event": "observation",
            "header": header,
            "source_seconds": t,
            "jpeg_sha256": sha(jpeg),
            "people": people,
            "issued_command_ids": [c["command_id"] for c in issued],
        }
        rows.append(row)
        reply({"row": row, "commands": issued})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--verification-seconds", type=int, default=3, choices=(3, 6))
    parser.add_argument("--extend-on-downward", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not all((args.capture, args.output, args.host, args.identity)) or args.output.exists():
        parser.error("capture, output, host and identity required; output must be new")

    import cv2

    from care_vision.common.contracts import envelope, validate
    from care_vision.edge.buffer import encode
    from care_vision.edge.commands import CommandHandler
    from care_vision.edge.representation import VisualPolicy
    from care_vision.edge.sampler import Sampler

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((args.capture / "manifest.json").read_text())
    video_bytes = (args.capture / "dense.avi").read_bytes()
    captures_bytes = (args.capture / "captures.jsonl").read_bytes()
    assert manifest["complete"]
    assert sha(video_bytes) == manifest["video_sha256"]
    assert sha(captures_bytes) == manifest["captures_sha256"]
    captures = [json.loads(line) for line in captures_bytes.splitlines()]
    assert len(captures) == manifest["decoded_frames"]
    assert all(
        b["captured_mono_ns"] > a["captured_mono_ns"] for a, b in zip(captures, captures[1:])
    )
    args.output.mkdir(parents=True)
    models = {
        "nano": (
            "nano.onnx",
            "artifacts/phase5g-compact-detectors-20260920/models/yolox_nano.onnx",
            416,
            0.4,
        ),
        "s": (
            "s.onnx",
            "artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx",
            640,
            0.35,
        ),
    }
    config = {
        "opencv_threads": 8,
        "verification_seconds": args.verification_seconds,
        "extend_on_downward": args.extend_on_downward,
        "profiles": {
            name: {
                "filename": filename,
                "sha256": sha((root / path).read_bytes()),
                "input_size": size,
                "threshold": threshold,
            }
            for name, (filename, path, size, threshold) in models.items()
        },
    }
    (args.output / "worker-config.json").write_text(json.dumps(config, indent=2) + "\n")
    archive = args.output / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(Path(__file__), arcname="worker.py")
        tar.add(args.output / "worker-config.json", arcname="worker-config.json")
        for path in sorted((root / "src/care_vision").rglob("*.py")):
            tar.add(path, arcname=str(path.relative_to(root / "src")))
        for _, (filename, path, _, _) in models.items():
            tar.add(root / path, arcname=filename)
    remote = f"/tmp/care-vision-loss-{uuid.uuid4().hex}"
    options = [
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=10",
        "-i",
        str(args.identity),
    ]
    ssh = ["ssh", *options, args.host]
    subprocess.run([*ssh, f"mkdir {shlex.quote(remote)}"], check=True)
    subprocess.run(
        ["scp", *options, str(archive), f"{args.host}:{remote}/package.tar.gz"], check=True
    )
    command = (
        f"tar -xzf {remote}/package.tar.gz -C {remote} && "
        f"PYTHONPATH={remote}:/opt/cool/python_3.11/site-packages "
        f"LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib "
        f"timeout 900 /usr/bin/python3.11 -u {remote}/worker.py --worker {remote}"
    )
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "source_video_sha256": manifest["video_sha256"],
                "source_captures_sha256": manifest["captures_sha256"],
                "package_sha256": sha(archive.read_bytes()),
                "baseline": "Nano 320x180 at 2 FPS",
                "trigger": "confirmed track missing for 1-2 s; 8 s request cooldown",
                "verification": (
                    f"S 1280x720 at 5 FPS for {args.verification_seconds} s, "
                    "then return to Nano 2 FPS"
                ),
                "exploratory_extension": (
                    "extend to 10 s if S box top descends >=150 px from its initial position"
                    if args.extend_on_downward
                    else None
                ),
                "scope": (
                    "sequential SSH replay with real Edge command/ACK and causal frames; "
                    "not paced, WebSocket or incident decision"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    source_width, source_height = manifest["source_width"], manifest["source_height"]
    identity = {
        "edge_id": "loss-escalation-replay",
        "boot_id": uuid.uuid4().hex,
        "session_id": uuid.uuid4().hex,
    }
    visual = VisualPolicy(source_width, source_height, 320, 180)
    sampler = Sampler(2)
    handler = CommandHandler(identity, sampler, 25, visual_policy=visual)
    rows, commands, acks, sent = [], [], [], []
    video = cv2.VideoCapture(str(args.capture / "dense.avi"))
    started = time.perf_counter()
    with (
        (args.output / "cloud.stderr.log").open("w") as err,
        (args.output / "cloud.jsonl").open("w") as cloud_log,
    ):
        process = subprocess.Popen(
            [*ssh, command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            bufsize=1,
        )

        def receive():
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("cloud worker closed; inspect cloud.stderr.log")
            cloud_log.write(line)
            cloud_log.flush()
            return json.loads(line)

        def exchange(packet):
            process.stdin.write(json.dumps(packet) + "\n")
            process.stdin.flush()
            return receive()

        try:
            ready = receive()
            assert ready["event"] == "ready" and ready["runtime"]["cool"]["verified"]
            (args.output / "runtime.json").write_text(json.dumps(ready, indent=2) + "\n")
            for capture in captures:
                ok, image = video.read()
                assert ok and image.shape[:2] == (source_height, source_width)
                ns, seq = capture["captured_mono_ns"], capture["capture_seq"]
                if not sampler.select(seq, ns):
                    continue
                jpeg, visual_metadata = visual.render(
                    encode(image, source_width, source_height), ns
                )
                role = "baseline" if sampler.fps == 2 else "verification"
                header = envelope(
                    "FRAME",
                    identity,
                    frame_id=f"{identity['boot_id']}:{seq}",
                    capture_seq=seq,
                    captured_mono_ns=ns,
                    sampled_mono_ns=ns,
                    captured_at_utc=capture["captured_at_utc"],
                    encoding="jpeg",
                    width=visual.output_width,
                    height=visual.output_height,
                    payload_bytes=len(jpeg),
                    evidence_kind="live",
                    target_transmit_fps=sampler.fps,
                    policy_seq=handler.policy_seq,
                    applied_command_id=handler.command_id,
                    stream_role=role,
                    **visual_metadata,
                )
                validate(header)
                result = exchange(
                    {"header": header, "jpeg": base64.b64encode(jpeg).decode(), "sha256": sha(jpeg)}
                )
                row = result["row"]
                assert row["jpeg_sha256"] == sha(jpeg) and row["header"] == header
                rows.append(row)
                sent.append(
                    {
                        "frame_id": header["frame_id"],
                        "capture_seq": seq,
                        "sha256": sha(jpeg),
                        "payload_bytes": len(jpeg),
                        "role": role,
                    }
                )
                for request in result["commands"]:
                    ack = handler.apply(request, ns, seq)
                    assert ack["result"] == "applied", ack
                    validate(ack)
                    assert exchange(ack)["acknowledged"] == request["command_id"]
                    commands.append(request)
                    acks.append(ack)
                    value = request.get("transmit_fps", request.get("output_width"))
                    print(
                        f"{row['source_seconds']:.3f}s {request['type']} {value} "
                        f"ACK={ack['result']}",
                        flush=True,
                    )
            assert not video.read()[0]
            finished = exchange({"control": "finish"})
            process.stdin.close()
            assert process.wait(timeout=30) == 0
        finally:
            video.release()
            if process.poll() is None:
                process.kill()
                process.wait()
    report = {
        "rows": rows,
        "commands": commands,
        "acks": acks,
        "sent": sent,
        "finished": finished,
        "wall_seconds": time.perf_counter() - started,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"COMPLETE {len(rows)} frames {len(commands)} commands {len(acks)} ACK", flush=True)


if __name__ == "__main__":
    main()
