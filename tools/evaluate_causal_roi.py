"""Isolated sequential cloud-selected ROI experiment; no live service or alerts."""

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

    from care_vision.cloud.occlusion_roi import incident_roi
    from care_vision.cloud.person import observe_people
    from care_vision.cloud.tracking import PersonTracker
    from care_vision.cloud.visibility_probe import VisibilityProbe
    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.contracts import envelope, uid, utc, validate
    from care_vision.common.environment import runtime_info

    config = json.loads((base / "worker-config.json").read_text())
    cv2.setNumThreads(1)
    assert cv2.getNumThreads() == 1
    detectors = []
    for profile in config["profiles"]:
        path = base / profile["filename"]
        assert sha(path.read_bytes()) == profile["sha256"]
        detectors.append(
            YoloXPersonDetector(
                path,
                input_size=profile["input_size"],
                score_threshold=profile["threshold"],
                nms_threshold=0.3,
            )
        )
    tracker, probe = PersonTracker(), VisibilityProbe()
    rows, commands, acks, references = [], [], [], {}
    active_start = active_command = pending = origin = None
    seq = 0

    def reply(value):
        print(json.dumps(value), flush=True)

    reply({"event": "ready", "runtime": runtime_info(3), "opencv_num_threads": 1})
    for line in sys.stdin:
        packet = json.loads(line)
        if packet.get("control") == "finish":
            assert pending is None
            reply(
                {
                    "event": "finished",
                    "probes": probe.finish(),
                    "frames": len(rows),
                    "commands": len(commands),
                    "acks": len(acks),
                }
            )
            break
        if packet.get("type") == "COMMAND_ACK":
            validate(packet)
            assert pending is not None and pending["command_id"] == packet["command_id"]
            assert packet["result"] == "applied"
            assert packet["applied_visual_command_id"] == pending["command_id"]
            acks.append(packet)
            pending = None
            reply({"acknowledged": packet["command_id"]})
            continue
        header, jpeg = packet["header"], base64.b64decode(packet["jpeg"], validate=True)
        validate(header)
        assert pending is None and sha(jpeg) == packet["sha256"]
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.shape[:2] == (header["height"], header["width"])
        if origin is None:
            origin = header["captured_mono_ns"]
        t = (header["captured_mono_ns"] - origin) / 1e9
        observation_id = uid()
        issued = []
        if header["stream_role"] == "baseline":
            people = observe_people(detectors[0], image, header["frame_id"])
            assert people["person_detection_status"] == "ok"
            people = tracker.update(people, header["captured_mono_ns"], "causal-roi")
            state = probe.update(t, header["capture_seq"] - 1, people["persons"])
            references[t] = header["capture_seq"]
            active = next(
                (p for p in state["probes"] if p["state"] == "VERIFYING_VISIBILITY"), None
            )
            desired = active["start"] if active else None
            if desired != active_start:
                seq += 1
                identity = {k: header[k] for k in ("edge_id", "boot_id", "session_id")}
                common = dict(
                    command_id=uid(),
                    issued_at_utc=utc(),
                    command_seq=seq,
                    caused_by_observation_id=observation_id,
                    based_on_frame_id=header["frame_id"],
                    based_on_capture_seq=header["capture_seq"],
                    valid_until_edge_mono_ns=header["captured_mono_ns"] + 5_000_000_000,
                    issued_source_seconds=t,
                    output_width=1280,
                    output_height=720,
                    source_width=header["source_width"],
                    source_height=header["source_height"],
                )
                if active:
                    reference = active["reference_bbox"]
                    scales = [
                        header["source_width"] / header["width"],
                        header["source_height"] / header["height"],
                    ]
                    native = [
                        reference[0] * scales[0],
                        reference[1] * scales[1],
                        reference[2] * scales[0],
                        reference[3] * scales[1],
                    ]
                    roi = incident_roi(native, (header["source_height"], header["source_width"]))
                    common.update(
                        roi_x=roi[0],
                        roi_y=roi[1],
                        roi_width=roi[2],
                        roi_height=roi[3],
                        reference_source_seconds=active["reference_time"],
                        reference_capture_seq=references[active["reference_time"]],
                        reference_bbox_native=native,
                        verifier_interval_seconds=1,
                        reason="existing_visibility_probe_requires_more_evidence",
                    )
                    pending = envelope("REQUEST_ROI", identity, **common)
                    active_command = pending
                else:
                    pending = envelope(
                        "REQUEST_FULL_FRAME",
                        identity,
                        reason="bounded_visibility_probe_ended",
                        **common,
                    )
                    active_command = None
                validate(pending)
                commands.append(pending)
                issued.append(pending)
                active_start = desired
        else:
            assert header["stream_role"] == "verification" and active_command is not None
            assert header["applied_visual_command_id"] == active_command["command_id"]
            assert header["capture_seq"] > active_command["based_on_capture_seq"]
            people = observe_people(detectors[1], image, header["frame_id"])
            assert people["person_detection_status"] == "ok"
            state = None
        row = {
            "header": header,
            "source_seconds": t,
            "jpeg_sha256": sha(jpeg),
            "observation_id": observation_id,
            "people": people,
            "probe_state": state,
        }
        rows.append(row)
        reply({"event": "observation", "row": row, "commands": issued})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not all([args.capture, args.output, args.host, args.identity]) or args.output.exists():
        parser.error("capture, host, identity and new output required")
    import cv2
    from replay_tapo_fixed_rates import select_indices

    from care_vision.common.contracts import envelope, utc
    from care_vision.edge.buffer import encode
    from care_vision.edge.commands import CommandHandler
    from care_vision.edge.representation import VisualPolicy
    from care_vision.edge.sampler import Sampler

    manifest = json.loads((args.capture / "manifest.json").read_text())
    assert manifest["complete"]
    assert sha((args.capture / "dense.avi").read_bytes()) == manifest["video_sha256"]
    assert sha((args.capture / "captures.jsonl").read_bytes()) == manifest["captures_sha256"]
    captures = [
        json.loads(line) for line in (args.capture / "captures.jsonl").read_text().splitlines()
    ]
    stamps = [r["captured_mono_ns"] for r in captures]
    assert len(captures) == manifest["decoded_frames"]
    assert all(b > a for a, b in zip(stamps, stamps[1:]))
    selected = set(select_indices(stamps, 2))
    args.output.mkdir(parents=True)
    (args.output / "roi-evidence").mkdir()
    (args.output / "source-previews").mkdir()
    root = Path(__file__).resolve().parents[1]
    models = [
        (
            "nano.onnx",
            "artifacts/phase5g-compact-detectors-20260920/models/yolox_nano.onnx",
            416,
            0.4,
        ),
        (
            "s.onnx",
            "artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx",
            640,
            0.35,
        ),
    ]
    config = {
        "profiles": [
            {
                "filename": name,
                "sha256": sha((root / path).read_bytes()),
                "input_size": size,
                "threshold": threshold,
            }
            for name, path, size, threshold in models
        ]
    }
    (args.output / "worker-config.json").write_text(json.dumps(config, indent=2) + "\n")
    archive = args.output / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(Path(__file__), arcname="worker.py")
        tar.add(args.output / "worker-config.json", arcname="worker-config.json")
        for path in sorted((root / "src/care_vision").rglob("*.py")):
            tar.add(path, arcname=str(path.relative_to(root / "src")))
        for name, path, _, _ in models:
            tar.add(root / path, arcname=name)
    remote = f"/tmp/care-vision-causal-roi-{uuid.uuid4().hex}"
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
        "LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib "
        f"timeout 600 /usr/bin/python3.11 -u {remote}/worker.py --worker {remote}"
    )
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "source_manifest": manifest,
                "package_sha256": sha(archive.read_bytes()),
                "remote": remote,
                "baseline_fps": 2,
                "verification_fps": 1,
                "roi_policy": "unchanged VisibilityProbe + incident_roi(expanded=True)",
                "scope": (
                    "Sequential SSH command/ACK/new-image loop; no realtime FPS, no fall decisions"
                ),
                "worker_receives": (
                    "model config and one current image at a time; "
                    "no cues, source video, future images or manual ROI"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    identity = {
        "edge_id": "causal-roi-replay",
        "boot_id": uuid.uuid4().hex,
        "session_id": uuid.uuid4().hex,
    }
    width, height = manifest["source_width"], manifest["source_height"]
    visual = VisualPolicy(width, height, 1280, 720)
    handler = CommandHandler(identity, Sampler(2), 25, visual_policy=visual)
    rows, commands, acks, sent = [], [], [], []
    last_roi_ns = None
    started = time.perf_counter()
    video = cv2.VideoCapture(str(args.capture / "dense.avi"))
    with (
        (args.output / "cloud.stderr.log").open("w") as err,
        (args.output / "cloud.jsonl").open("w") as log,
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
            log.write(line)
            log.flush()
            return json.loads(line)

        def exchange(value):
            process.stdin.write(json.dumps(value) + "\n")
            process.stdin.flush()
            return receive()

        def frame(image, jpeg, capture, role, metadata):
            nonlocal last_roi_ns
            header = envelope(
                "FRAME",
                identity,
                frame_id=f"{identity['boot_id']}:{capture['capture_seq']}",
                capture_seq=capture["capture_seq"],
                captured_mono_ns=capture["captured_mono_ns"],
                sampled_mono_ns=capture["captured_mono_ns"],
                captured_at_utc=capture["captured_at_utc"],
                sent_at_utc=utc(),
                encoding="jpeg",
                width=320 if role == "baseline" else 1280,
                height=180 if role == "baseline" else 720,
                payload_bytes=len(jpeg),
                evidence_kind="live",
                target_transmit_fps=2 if role == "baseline" else 1,
                policy_seq=0,
                applied_command_id=None,
                stream_role=role,
                **metadata,
            )
            response = exchange(
                {"header": header, "jpeg": base64.b64encode(jpeg).decode(), "sha256": sha(jpeg)}
            )
            assert response["row"]["jpeg_sha256"] == sha(jpeg)
            assert response["row"]["header"] == header
            rows.append(response["row"])
            sent.append({"header": header, "sha256": sha(jpeg)})
            for command in response["commands"]:
                ack = handler.apply(command, capture["captured_mono_ns"], capture["capture_seq"])
                assert ack["result"] == "applied"
                assert exchange(ack)["acknowledged"] == command["command_id"]
                commands.append(command)
                acks.append(ack)
                last_roi_ns = None
                print(
                    f"{response['row']['source_seconds']:.3f}s {command['type']} ROI={visual.roi}",
                    flush=True,
                )

        try:
            ready = receive()
            assert ready["event"] == "ready" and ready["runtime"]["cool"]["verified"]
            (args.output / "runtime.json").write_text(json.dumps(ready, indent=2) + "\n")
            for i, capture in enumerate(captures):
                ok, image = video.read()
                assert ok and image.shape[:2] == (height, width)
                ns = capture["captured_mono_ns"]
                if i in selected:
                    frame(
                        image,
                        encode(image, 320, 180),
                        capture,
                        "baseline",
                        dict(
                            view_kind="full_frame",
                            source_width=width,
                            source_height=height,
                            visual_policy_seq=0,
                            applied_visual_command_id=None,
                        ),
                    )
                if (
                    i not in selected
                    and visual.view_kind == "roi"
                    and capture["capture_seq"] >= visual.effective_from_capture_seq
                    and (last_roi_ns is None or ns - last_roi_ns >= 1_000_000_000)
                ):
                    jpeg, metadata = visual.render(encode(image, width, height), ns)
                    (
                        args.output / "roi-evidence" / f"{capture['capture_seq']:06d}.jpg"
                    ).write_bytes(jpeg)
                    (
                        args.output / "source-previews" / f"{capture['capture_seq']:06d}.jpg"
                    ).write_bytes(encode(image, 1280, 720))
                    frame(image, jpeg, capture, "verification", metadata)
                    last_roi_ns = ns
                if (
                    i in selected
                    and len([r for r in rows if r["header"]["stream_role"] == "baseline"]) % 60 == 0
                ):
                    print(f"Baseline source time {(ns - stamps[0]) / 1e9:.1f}s", flush=True)
            assert not video.read()[0]
            end = exchange({"control": "finish"})
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
        "finish": end,
        "wall_seconds": time.perf_counter() - started,
        "baseline_samples": len(selected),
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("COMPLETE: commands applied before subsequent ROI evidence", flush=True)


if __name__ == "__main__":
    main()
