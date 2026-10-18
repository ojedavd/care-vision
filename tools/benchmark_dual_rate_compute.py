"""Paced Graviton compute benchmark of Nano 2 FPS plus selective S 1 FPS.

Inputs are saved Tapo images. The worker paces their original source timestamps;
it does not measure camera, Edge encoding, network transport, or controller latency.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import shlex
import subprocess
import tarfile
import time
import uuid
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parallel_worker(base, role, clock, gate, messages):
    import cv2
    import numpy as np

    from care_vision.cloud.person import observe_people
    from care_vision.cloud.yolox import YoloXPersonDetector

    plan = json.loads((base / "plan.json").read_text())
    threads = plan.get("verification_threads", 1) if role == "verification" else 1
    cv2.setNumThreads(threads)
    assert cv2.getNumThreads() == threads
    profile = next(p for p in plan["models"] if p["role"] == role)
    model = base / profile["filename"]
    assert sha(model) == profile["sha256"]
    detector = YoloXPersonDetector(
        model,
        input_size=profile["input_size"],
        score_threshold=profile["threshold"],
        nms_threshold=0.3,
    )
    images = {}
    frames = [r for r in plan["frames"] if r["role"] == role]
    for item in frames:
        path = base / item["filename"]
        assert sha(path) == item["sha256"]
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        assert image is not None
        images[item["filename"]] = image
    warmup_start = time.perf_counter()
    observe_people(detector, images[frames[0]["filename"]], f"warmup:{role}")
    warmup = time.perf_counter() - warmup_start
    messages.put({"kind": "ready", "role": role, "warmup_seconds": warmup})
    assert gate.wait(timeout=30)
    start_wall = clock.value
    start_cpu = time.process_time()
    events = []
    for item in frames:
        due = start_wall + item["due_seconds"]
        remaining = due - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        started = time.perf_counter()
        result = observe_people(detector, images[item["filename"]], item["frame_id"])
        completed = time.perf_counter()
        assert result["person_detection_status"] == "ok"
        with open("/proc/self/statm") as source:
            resident_pages = int(source.read().split()[1])
        events.append(
            {
                "role": role,
                "capture_seq": item["capture_seq"],
                "due_seconds": item["due_seconds"],
                "start_seconds": started - start_wall,
                "complete_seconds": completed - start_wall,
                "queue_wait_seconds": started - due,
                "processing_seconds": completed - started,
                "rss_bytes": resident_pages * os.sysconf("SC_PAGE_SIZE"),
                "person_count": result["person_count"],
            }
        )
    messages.put(
        {
            "kind": "finished",
            "role": role,
            "events": events,
            "cpu_seconds": time.process_time() - start_cpu,
            "opencv_threads": threads,
        }
    )


def run_parallel(base, plan, runtime):
    context = multiprocessing.get_context("spawn")
    clock = context.Value("d", 0.0)
    gate = context.Event()
    messages = context.Queue()
    processes = [
        context.Process(target=parallel_worker, args=(base, role, clock, gate, messages))
        for role in ("baseline", "verification")
    ]
    for process in processes:
        process.start()
    ready = [messages.get(timeout=60) for _ in processes]
    assert {r["role"] for r in ready} == {"baseline", "verification"}
    assert all(r["kind"] == "ready" for r in ready)
    clock.value = time.perf_counter() + 0.5
    gate.set()
    finished = [messages.get(timeout=90) for _ in processes]
    assert all(r["kind"] == "finished" for r in finished)
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    events = [event for group in finished for event in group["events"]]
    events.sort(key=lambda r: (r["due_seconds"], r["role"]))
    payload = {
        "runtime": runtime,
        "opencv_num_threads_by_role": {r["role"]: r["opencv_threads"] for r in finished},
        "worker_processes": 2,
        "warmup_seconds": {r["role"]: r["warmup_seconds"] for r in ready},
        "events": events,
        "wall_seconds": max(r["complete_seconds"] for r in events),
        "cpu_seconds": sum(r["cpu_seconds"] for r in finished),
        "cpu_seconds_by_role": {r["role"]: r["cpu_seconds"] for r in finished},
        "peak_rss_bytes_by_role": {
            r["role"]: max(e["rss_bytes"] for e in r["events"]) for r in finished
        },
        "scope": "paced compute only; two AWS processes, preloaded images, no Edge or transport",
    }
    (base / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("COMPLETE paced parallel cloud compute", flush=True)


def worker(base):
    import cv2
    import numpy as np

    from care_vision.cloud.person import observe_people
    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.environment import runtime_info

    plan = json.loads((base / "plan.json").read_text())
    cv2.setNumThreads(1)
    assert cv2.getNumThreads() == 1
    runtime = runtime_info(3)
    if plan["execution"] == "parallel":
        run_parallel(base, plan, runtime)
        return
    detectors = {}
    for profile in plan["models"]:
        model = base / profile["filename"]
        assert sha(model) == profile["sha256"]
        detectors[profile["role"]] = YoloXPersonDetector(
            model,
            input_size=profile["input_size"],
            score_threshold=profile["threshold"],
            nms_threshold=0.3,
        )
    images = {}
    for item in plan["frames"]:
        path = base / item["filename"]
        assert sha(path) == item["sha256"]
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        assert image is not None
        images[item["filename"]] = image
    # Warm each graph once, outside the paced interval.
    warmup = {}
    for role in ("baseline", "verification"):
        item = next(x for x in plan["frames"] if x["role"] == role)
        start = time.perf_counter()
        observe_people(detectors[role], images[item["filename"]], f"warmup:{role}")
        warmup[role] = time.perf_counter() - start
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    events = []
    for item in plan["frames"]:
        due = start_wall + item["due_seconds"]
        remaining = due - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        started = time.perf_counter()
        result = observe_people(detectors[item["role"]], images[item["filename"]], item["frame_id"])
        completed = time.perf_counter()
        assert result["person_detection_status"] == "ok"
        with open("/proc/self/statm") as source:
            resident_pages = int(source.read().split()[1])
        events.append(
            {
                "role": item["role"],
                "capture_seq": item["capture_seq"],
                "due_seconds": item["due_seconds"],
                "start_seconds": started - start_wall,
                "complete_seconds": completed - start_wall,
                "queue_wait_seconds": started - due,
                "processing_seconds": completed - started,
                "rss_bytes": resident_pages * os.sysconf("SC_PAGE_SIZE"),
                "person_count": result["person_count"],
            }
        )
    elapsed = time.perf_counter() - start_wall
    payload = {
        "runtime": runtime,
        "opencv_num_threads": 1,
        "warmup_seconds": warmup,
        "events": events,
        "wall_seconds": elapsed,
        "cpu_seconds": time.process_time() - start_cpu,
        "scope": "paced compute only; preloaded frames, no network or Edge pipeline",
    }
    (base / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("COMPLETE paced cloud compute", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--prior-replay", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--start", type=float, default=52)
    parser.add_argument("--end", type=float, default=82)
    parser.add_argument("--execution", choices=["serial", "parallel"], default="serial")
    parser.add_argument("--verification-threads", type=int, default=1)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not all([args.capture, args.prior_replay, args.output, args.host, args.identity]):
        parser.error("capture, prior-replay, output, host and identity are required")
    if args.output.exists() or not 0 <= args.start < args.end:
        parser.error("output must be new and window must have 0 <= START < END")
    if args.verification_threads < 1 or (
        args.execution == "serial" and args.verification_threads != 1
    ):
        parser.error("verification threads must be positive; serial mode requires one thread")
    import cv2
    from replay_tapo_fixed_rates import select_indices

    from care_vision.edge.buffer import encode

    manifest = json.loads((args.capture / "manifest.json").read_text())
    assert manifest["complete"]
    assert sha(args.capture / "dense.avi") == manifest["video_sha256"]
    assert sha(args.capture / "captures.jsonl") == manifest["captures_sha256"]
    captures = [json.loads(x) for x in (args.capture / "captures.jsonl").read_text().splitlines()]
    stamps = [x["captured_mono_ns"] for x in captures]
    assert len(captures) == manifest["decoded_frames"]
    prior = json.loads((args.prior_replay / "result.json").read_text())
    roi_rows = [
        r
        for r in prior["rows"]
        if r["header"]["stream_role"] == "verification"
        and args.start <= r["source_seconds"] < args.end
    ]
    assert roi_rows
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True)
    images = args.output / "images"
    images.mkdir()
    frames = []
    origin = stamps[0]
    selected = set(select_indices(stamps, 2))
    video = cv2.VideoCapture(str(args.capture / "dense.avi"))
    try:
        for i, capture in enumerate(captures):
            ok, image = video.read()
            assert ok and image.shape[:2] == (
                manifest["source_height"],
                manifest["source_width"],
            )
            source_seconds = (capture["captured_mono_ns"] - origin) / 1e9
            if i in selected and args.start <= source_seconds < args.end:
                name = f"images/baseline-{capture['capture_seq']:06d}.jpg"
                (args.output / name).write_bytes(encode(image, 320, 180))
                frames.append(
                    {
                        "role": "baseline",
                        "frame_id": f"baseline:{capture['capture_seq']}",
                        "capture_seq": capture["capture_seq"],
                        "due_seconds": source_seconds - args.start,
                        "filename": name,
                        "sha256": sha(args.output / name),
                    }
                )
        assert not video.read()[0]
    finally:
        video.release()
    for row in roi_rows:
        seq = row["header"]["capture_seq"]
        source = args.prior_replay / "roi-evidence" / f"{seq:06d}.jpg"
        assert sha(source) == row["jpeg_sha256"]
        name = f"images/verification-{seq:06d}.jpg"
        (args.output / name).write_bytes(source.read_bytes())
        frames.append(
            {
                "role": "verification",
                "frame_id": f"verification:{seq}",
                "capture_seq": seq,
                "due_seconds": row["source_seconds"] - args.start,
                "filename": name,
                "sha256": sha(args.output / name),
            }
        )
    frames.sort(key=lambda r: (r["due_seconds"], r["role"]))
    assert len({r["capture_seq"] for r in frames}) == len(frames)
    models = [
        (
            "baseline",
            "nano.onnx",
            "artifacts/phase5g-compact-detectors-20260920/models/yolox_nano.onnx",
            416,
            0.4,
        ),
        (
            "verification",
            "s.onnx",
            "artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx",
            640,
            0.35,
        ),
    ]
    profiles = [
        {
            "role": role,
            "filename": name,
            "sha256": sha(root / path),
            "input_size": size,
            "threshold": threshold,
        }
        for role, name, path, size, threshold in models
    ]
    plan = {
        "source_video_sha256": manifest["video_sha256"],
        "source_timestamps_sha256": manifest["captures_sha256"],
        "window_seconds": [args.start, args.end],
        "models": profiles,
        "frames": frames,
        "scheduling": (
            "single non-preemptive FIFO worker ordered by capture timestamp"
            if args.execution == "serial"
            else "two independent one-thread AWS processes sharing a monotonic start clock"
        ),
        "execution": args.execution,
        "verification_threads": args.verification_threads,
        "scope": "preloaded cloud compute only; paced 2 FPS baseline + approximately 1 FPS ROI",
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    archive = args.output / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(Path(__file__), arcname="worker.py")
        tar.add(args.output / "plan.json", arcname="plan.json")
        tar.add(images, arcname="images")
        for path in sorted((root / "src/care_vision").rglob("*.py")):
            tar.add(path, arcname=str(path.relative_to(root / "src")))
        for _, name, path, _, _ in models:
            tar.add(root / path, arcname=name)
    remote = f"/tmp/care-vision-dual-rate-{uuid.uuid4().hex}"
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
        ["scp", *options, str(archive), f"{args.host}:{remote}/package.tar.gz"],
        check=True,
    )
    command = (
        f"tar -xzf {remote}/package.tar.gz -C {remote} && "
        f"PYTHONPATH={remote}:/opt/cool/python_3.11/site-packages "
        "LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib "
        f"timeout 180 /usr/bin/python3.11 -u {remote}/worker.py --worker {remote}"
    )
    (args.output / "deployment.json").write_text(
        json.dumps({"remote": remote, "package_sha256": sha(archive)}, indent=2) + "\n"
    )
    with (args.output / "execution.log").open("w") as log:
        with subprocess.Popen(
            [*ssh, command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError("remote compute benchmark failed; inspect execution.log")
    subprocess.run(
        ["scp", *options, f"{args.host}:{remote}/result.json", str(args.output / "result.json")],
        check=True,
    )
    result = json.loads((args.output / "result.json").read_text())
    assert len(result["events"]) == len(frames)
    assert [(r["role"], r["capture_seq"]) for r in result["events"]] == [
        (r["role"], r["capture_seq"]) for r in frames
    ]
    assert result["runtime"]["cool"]["verified"]
    print("Verified all paced frames and COOL runtime", flush=True)


if __name__ == "__main__":
    main()
