"""Measure YOLOX-S ROI latency versus OpenCV thread count on Graviton.

Reuses the exact hashed images/model from a prior dual-rate compute package.
This is preloaded cloud inference only, not an end-to-end or fall test.
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(base, threads):
    import cv2
    import numpy as np

    from care_vision.cloud.person import observe_people
    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.environment import runtime_info

    plan = json.loads((base / "plan.json").read_text())
    profile = next(m for m in plan["models"] if m["role"] == "verification")
    frames = [f for f in plan["frames"] if f["role"] == "verification"]
    assert len(frames) == 30
    cv2.setNumThreads(threads)
    assert cv2.getNumThreads() == threads
    model = base / profile["filename"]
    assert sha(model) == profile["sha256"]
    images = []
    for item in frames:
        path = base / item["filename"]
        assert sha(path) == item["sha256"]
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        assert image is not None
        images.append(image)
    detector = YoloXPersonDetector(
        model,
        input_size=profile["input_size"],
        score_threshold=profile["threshold"],
        nms_threshold=0.3,
    )
    for index in (0, 1):
        observe_people(detector, images[index], f"warmup:{index}")
    start_cpu = time.process_time()
    observations = []
    for item, image in zip(frames, images, strict=True):
        start = time.perf_counter()
        result = observe_people(detector, image, item["frame_id"])
        elapsed = time.perf_counter() - start
        assert result["person_detection_status"] == "ok"
        observations.append(
            {
                "capture_seq": item["capture_seq"],
                "person_count": result["person_count"],
                "seconds": elapsed,
            }
        )
    with open("/proc/self/statm") as source:
        resident_pages = int(source.read().split()[1])
    print(
        json.dumps(
            {
                "threads": threads,
                "opencv_threads": cv2.getNumThreads(),
                "runtime": runtime_info(3),
                "observations": observations,
                "cpu_seconds": time.process_time() - start_cpu,
                "rss_bytes_end": resident_pages * os.sysconf("SC_PAGE_SIZE"),
            }
        ),
        flush=True,
    )


def worker(base):
    results = []
    script = Path(__file__)
    for threads in (1, 2, 4, 8):
        completed = subprocess.run(
            [
                "/usr/bin/python3.11",
                str(script),
                "--worker-base",
                str(base),
                "--threads",
                str(threads),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": f"{base}:/opt/cool/python_3.11/site-packages"},
        )
        results.append(json.loads(completed.stdout))
        print(f"threads={threads}: 30/30", flush=True)
    assert all(r["runtime"]["cool"]["verified"] for r in results)
    assert all(r["runtime"]["ec2"]["architecture"] == "arm64" for r in results)
    baseline = [(o["capture_seq"], o["person_count"]) for o in results[0]["observations"]]
    assert all(
        [(o["capture_seq"], o["person_count"]) for o in r["observations"]] == baseline
        for r in results
    )
    (base / "thread-result.json").write_text(
        json.dumps({"conditions": results, "same_person_counts": True}, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-base", type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker_base:
        if args.threads is not None:
            run_one(args.worker_base, args.threads)
        else:
            worker(args.worker_base)
        return
    if not all((args.prior, args.host, args.identity, args.output)):
        parser.error("prior, host, identity and output are required")
    if args.output.exists():
        parser.error("output must be new")
    prior_plan = json.loads((args.prior / "plan.json").read_text())
    assert len([f for f in prior_plan["frames"] if f["role"] == "verification"]) == 30
    assert (args.prior / "package.tar.gz").exists()
    args.output.mkdir(parents=True)
    remote = f"/tmp/care-vision-yolox-s-threads-{uuid.uuid4().hex}"
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
        [
            "scp",
            *options,
            str(args.prior / "package.tar.gz"),
            str(Path(__file__)),
            f"{args.host}:{remote}/",
        ],
        check=True,
    )
    (args.output / "deployment.json").write_text(
        json.dumps(
            {
                "remote": remote,
                "prior_package_sha256": sha(args.prior / "package.tar.gz"),
                "worker_sha256": sha(Path(__file__)),
            },
            indent=2,
        )
        + "\n"
    )
    command = (
        f"tar -xzf {remote}/package.tar.gz -C {remote} && "
        f"PYTHONPATH={remote}:/opt/cool/python_3.11/site-packages "
        "LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib "
        f"timeout 240 /usr/bin/python3.11 -u {remote}/benchmark_yolox_s_threads.py "
        f"--worker-base {remote}"
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
                raise RuntimeError("remote benchmark failed; inspect execution.log")
    subprocess.run(
        [
            "scp",
            *options,
            f"{args.host}:{remote}/thread-result.json",
            str(args.output / "result.json"),
        ],
        check=True,
    )
    result = json.loads((args.output / "result.json").read_text())
    assert result["same_person_counts"]
    assert len(result["conditions"]) == 4


if __name__ == "__main__":
    main()
