"""Paced JPEG replay over SSH to Graviton; fixed sensing, no agent actuation."""

import argparse
import base64
import hashlib
import json
import queue
import shlex
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path


def select_indices(timestamps, fps):
    """Select first available capture at each due time, without duplicating frames."""
    selected = []
    due = timestamps[0]
    period = 1e9 / fps
    for i, captured in enumerate(timestamps):
        if captured >= due:
            selected.append(i)
            due += period
            if due <= captured:
                due += (int((captured - due) // period) + 1) * period
    return selected


def cloud_worker(model):
    import cv2

    from care_vision.cloud.vision import MotionVision
    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.environment import runtime_info

    cv2.setNumThreads(1)
    runtime = runtime_info(3)
    runtime["opencv_num_threads"] = cv2.getNumThreads()
    if runtime["opencv_num_threads"] != 1:
        raise RuntimeError("fixed replay requires one OpenCV thread")
    detector = YoloXPersonDetector(model, score_threshold=0.4, nms_threshold=0.3, input_size=416)
    vision = MotionVision(backend="opencv-COOL-fixed-rate-replay", person_detector=detector)
    print(
        json.dumps(
            {
                "event": "ready",
                "runtime": runtime,
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        ),
        flush=True,
    )
    count = payload_bytes = 0
    for line in sys.stdin:
        packet = json.loads(line)
        received = time.monotonic_ns()
        jpeg = base64.b64decode(packet["jpeg"], validate=True)
        if hashlib.sha256(jpeg).hexdigest() != packet["sha256"]:
            raise ValueError("JPEG integrity mismatch")
        observation = vision.observe(packet["header"], jpeg)
        count += 1
        payload_bytes += len(jpeg)
        print(
            json.dumps(
                {
                    "event": "observation",
                    "received_cloud_mono_ns": received,
                    "completed_cloud_mono_ns": time.monotonic_ns(),
                    "jpeg_sha256": packet["sha256"],
                    **observation,
                }
            ),
            flush=True,
        )
    print(
        json.dumps({"event": "complete", "processed": count, "jpeg_payload_bytes": payload_bytes}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-worker", type=Path, metavar="MODEL")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rates", type=int, nargs="+", default=[2, 10, 15, 20])
    parser.add_argument("--resolution", choices=["320x180", "1280x720"], default="320x180")
    parser.add_argument("--remote-python", default="/usr/bin/python3.11")
    args = parser.parse_args()
    if args.cloud_worker:
        cloud_worker(args.cloud_worker)
        return
    if not all([args.capture, args.host, args.identity, args.output]):
        parser.error("capture, host, identity and output are required")
    if args.output.exists() or any(r not in {2, 10, 15, 20} for r in args.rates):
        parser.error("output must be new; supported rates: 2, 10, 15, 20")
    import cv2

    from care_vision.edge.buffer import encode

    output_width, output_height = map(int, args.resolution.split("x"))
    manifest = json.loads((args.capture / "manifest.json").read_text())
    video = args.capture / "dense.avi"
    captures_path = args.capture / "captures.jsonl"
    if not manifest["complete"]:
        raise ValueError("capture was not verified complete")
    source_width = manifest.get("source_width", 1280)
    source_height = manifest.get("source_height", 720)
    for path, key in [(video, "video_sha256"), (captures_path, "captures_sha256")]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("capture integrity mismatch")
    captures = [json.loads(line) for line in captures_path.read_text().splitlines()]
    timestamps = [row["captured_mono_ns"] for row in captures]
    if len(captures) != manifest["decoded_frames"] or any(
        b <= a for a, b in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("capture timeline mismatch")
    schedules = {rate: select_indices(timestamps, rate) for rate in args.rates}
    wanted = set().union(*map(set, schedules.values()))
    jpegs = {}
    cap = cv2.VideoCapture(str(video))
    try:
        for i in range(len(captures)):
            ok, image = cap.read()
            if not ok or image.shape[:2] != (source_height, source_width):
                raise ValueError("source decode mismatch")
            if i in wanted:
                jpegs[i] = encode(image, output_width, output_height)
        if cap.read()[0]:
            raise ValueError("unexpected extra video frame")
    finally:
        cap.release()
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    model = root / "artifacts/phase5g-compact-detectors-20260920/models/yolox_nano.onnx"
    archive = args.output / "worker-source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted((root / "src/care_vision").rglob("*.py")):
            tar.add(path, arcname=str(path.relative_to(root / "src")))
        tar.add(Path(__file__), arcname="worker.py")
        tar.add(model, arcname="model.onnx")
    remote = f"/tmp/care-vision-fixed-{uuid.uuid4().hex}"
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
        ["scp", *options, str(archive), f"{args.host}:{remote}/source.tar.gz"], check=True
    )
    subprocess.run([*ssh, f"tar -xzf {remote}/source.tar.gz -C {remote}"], check=True)
    protocol = {
        "capture_manifest": manifest,
        "rates_in_order": args.rates,
        "source_sha256": manifest["video_sha256"],
        "resolution": [output_width, output_height],
        "detector_input_size": 416,
        "detector_score_threshold": 0.4,
        "detector_nms_threshold": 0.3,
        "worker_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "scope": "Paced fixed full-frame JPEG via SSH stdin; no agent, ROI or history",
        "transport": "base64 JSON over SSH, not production WebSocket",
        "limitations": [
            "MJPEG recording adds re-encoding versus direct camera JPEG",
            "JPEG payload bytes exclude base64, SSH and control overhead",
            "Source clocks preserved; cloud and Edge clocks not subtracted",
            "No incident decisions or alerts",
        ],
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries = []
    for rate, indices in schedules.items():
        run = args.output / f"fps-{rate}"
        run.mkdir()
        command = (
            f"PYTHONPATH={remote}:/opt/cool/python_3.11/site-packages "
            "LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib "
            f"{shlex.quote(args.remote_python)} -u {remote}/worker.py "
            f"--cloud-worker {remote}/model.onnx"
        )
        with (run / "stderr.log").open("w") as errors, (run / "cloud.jsonl").open("w") as log:
            process = subprocess.Popen(
                [*ssh, command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                bufsize=1,
            )
            ready = queue.Queue()
            received_rows = []

            def read_output():
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        row = json.loads(line)
                        received_rows.append(row)
                        if row["event"] == "ready":
                            ready.put(row)
                finally:
                    ready.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            try:
                runtime = ready.get(timeout=45)
                if runtime is None:
                    raise RuntimeError(f"cloud startup failed; inspect {run}")
                if runtime["model_sha256"] != hashlib.sha256(model.read_bytes()).hexdigest():
                    raise ValueError("remote model mismatch")
                if runtime["runtime"].get("opencv_num_threads") != 1:
                    raise ValueError("remote OpenCV thread configuration mismatch")
                identity = {
                    "edge_id": "fixed-replay",
                    "boot_id": uuid.uuid4().hex,
                    "session_id": uuid.uuid4().hex,
                }
                started = time.monotonic_ns()
                with (run / "sent.jsonl").open("w") as sent:
                    for index in indices:
                        deadline = started + timestamps[index] - timestamps[0]
                        time.sleep(max(0, (deadline - time.monotonic_ns()) / 1e9))
                        capture = captures[index]
                        jpeg = jpegs[index]
                        header = {
                            **identity,
                            **capture,
                            "frame_id": f"{identity['boot_id']}:{capture['capture_seq']}",
                            "applied_command_id": None,
                            "policy_seq": 0,
                            "width": output_width,
                            "height": output_height,
                            "view_kind": "full_frame",
                            "source_width": source_width,
                            "source_height": source_height,
                            "evidence_kind": "live",
                        }
                        digest = hashlib.sha256(jpeg).hexdigest()
                        before = time.monotonic_ns()
                        process.stdin.write(
                            json.dumps(
                                {
                                    "header": header,
                                    "sha256": digest,
                                    "jpeg": base64.b64encode(jpeg).decode(),
                                }
                            )
                            + "\n"
                        )
                        process.stdin.flush()
                        sent.write(
                            json.dumps(
                                {
                                    "frame_id": header["frame_id"],
                                    "source_index": index,
                                    "scheduled_mono_ns": deadline,
                                    "write_started_mono_ns": before,
                                    "write_finished_mono_ns": time.monotonic_ns(),
                                    "sha256": digest,
                                }
                            )
                            + "\n"
                        )
                process.stdin.close()
                process.wait(timeout=45)
                reader.join(timeout=5)
                observations = [r for r in received_rows if r["event"] == "observation"]
                completions = [r for r in received_rows if r["event"] == "complete"]
                expected_ids = [
                    f"{identity['boot_id']}:{captures[i]['capture_seq']}" for i in indices
                ]
                if (
                    process.returncode
                    or reader.is_alive()
                    or not completions
                    or ([r["frame_id"] for r in observations] != expected_ids)
                ):
                    raise RuntimeError(f"incomplete replay; inspect {run}")
                sent_rows = [
                    json.loads(line) for line in (run / "sent.jsonl").read_text().splitlines()
                ]
                if completions[-1]["jpeg_payload_bytes"] != sum(len(jpegs[i]) for i in indices):
                    raise ValueError("cloud payload accounting mismatch")
                summary = {
                    "rate": rate,
                    "selected": len(indices),
                    "processed": len(observations),
                    "integrity_verified": True,
                    "send_lateness_ms_max": max(
                        (r["write_started_mono_ns"] - r["scheduled_mono_ns"]) / 1e6
                        for r in sent_rows
                    ),
                    "write_block_ms_max": max(
                        (r["write_finished_mono_ns"] - r["write_started_mono_ns"]) / 1e6
                        for r in sent_rows
                    ),
                    "jpeg_payload_bytes": sum(len(jpegs[i]) for i in indices),
                    "processing_ms_median": statistics.median(
                        r["processing_ms"] for r in observations
                    ),
                    "runtime": runtime,
                    "wall_seconds": (time.monotonic_ns() - started) / 1e9,
                }
                summaries.append(summary)
                (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(f"{rate} FPS: {len(observations)}/{len(indices)} processed", flush=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                reader.join(timeout=5)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
