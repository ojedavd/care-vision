"""Prepare shared JPEGs locally and raw YOLOX-S detections on Graviton/COOL.

Offline experiment only. JPEGs are re-encoded from videos, not original payloads.
No tracker runs here; weak boxes retained by the detector remain in the cache.
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tarfile
import time
import uuid
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    path.chmod(0o600)


def selected_frames(run):
    """Validate schedules and deduplicate by source frame without losing cadence."""
    frames = {}
    for schedule in run["schedules"]:
        previous = -1
        for frame in schedule["frames"]:
            index = frame["frame_index"]
            if index <= previous or frame["captured_mono_ns"] <= 0:
                raise ValueError("schedule must have increasing frame indices and positive times")
            previous = index
            if index in frames and frames[index] != frame:
                raise ValueError("conflicting source frame metadata")
            frames[index] = frame
        times = [f["captured_mono_ns"] for f in schedule["frames"]]
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("capture timestamps must increase")
    return [frames[index] for index in sorted(frames)]


def prepare(root, manifest_path, output):
    import cv2

    manifest = json.loads(manifest_path.read_text())
    profile_path = manifest_path.parent / "profile.json"
    profile = json.loads(profile_path.read_text())
    output.mkdir(parents=True, mode=0o700)
    (output / "images").mkdir(mode=0o700)
    plan = {
        "schema_version": 1,
        "scope": "reencoded recorded video; raw pre-tracker detections; development only",
        "manifest_sha256": sha(manifest_path),
        "profile_sha256": sha(profile_path),
        "model_sha256": profile["person_model_sha256"],
        "jpeg_quality": 80,
        "detector": {"input_size": 640, "score_threshold": 0.2, "nms_threshold": 0.3},
        "threads": 8,
        "frames": [],
        "runs": manifest["runs"],
    }
    for run in manifest["runs"]:
        for name, digest in run["source_files"].items():
            if sha(root / name) != digest:
                raise ValueError(f"source digest mismatch: {name}")
        video_path = root / run.get(
            "video_path",
            next(
                p for p in run["source_files"] if Path(p).suffix.lower() in (".avi", ".mov", ".mp4")
            ),
        )
        wanted = {f["frame_index"]: f for f in selected_frames(run)}
        capture = cv2.VideoCapture(str(video_path))
        try:
            index = 0
            while wanted:
                ok, image = capture.read()
                if not ok:
                    raise ValueError("AVI ended before selected frame")
                if index in wanted:
                    frame = wanted.pop(index)
                    if "source_dimensions" in run:
                        if list(image.shape[:2][::-1]) != run["source_dimensions"]:
                            raise ValueError("source dimension mismatch")
                    if run.get("resize_to"):
                        if run.get("resize_mode") == "letterbox":
                            sx, sy, left, top = run["box_transform"]
                            height, width = image.shape[:2]
                            rw, rh = round(width * sx), round(height * sy)
                            image = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
                            image = cv2.copyMakeBorder(
                                image,
                                top,
                                720 - rh - top,
                                left,
                                1280 - rw - left,
                                cv2.BORDER_CONSTANT,
                                value=(114, 114, 114),
                            )
                        else:
                            image = cv2.resize(
                                image, tuple(run["resize_to"]), interpolation=cv2.INTER_LINEAR
                            )
                    if image.shape[:2] != (720, 1280):
                        raise ValueError("unexpected source dimensions")
                    frame_id = f"{run['name']}:{frame['capture_seq']}"
                    filename = f"images/{run['name']}-{frame['capture_seq']:06d}.jpg"
                    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise ValueError("JPEG encoding failed")
                    path = output / filename
                    path.write_bytes(encoded.tobytes())
                    path.chmod(0o600)
                    plan["frames"].append(
                        {
                            **frame,
                            "run": run["name"],
                            "frame_id": frame_id,
                            "filename": filename,
                            "jpeg_sha256": sha(path),
                            "width": 1280,
                            "height": 720,
                        }
                    )
                index += 1
        finally:
            capture.release()
    save(output / "plan.json", plan)
    return plan


def worker(base):
    import cv2
    import numpy as np

    from care_vision.cloud.person import observe_people
    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.environment import runtime_info

    plan = json.loads((base / "plan.json").read_text())
    runtime = runtime_info(3)
    if not runtime["cool"]["verified"] or runtime["ec2"]["architecture"] != "arm64":
        raise RuntimeError("requires verified Graviton + COOL")
    cv2.setNumThreads(plan["threads"])
    model = base / "model.onnx"
    if sha(model) != plan["model_sha256"]:
        raise ValueError("model digest mismatch")
    detector = YoloXPersonDetector(model, **plan["detector"])
    cache = base / "detections.jsonl"
    with cache.open("x") as stream:
        cache.chmod(0o600)
        for index, frame in enumerate(plan["frames"]):
            path = base / frame["filename"]
            if sha(path) != frame["jpeg_sha256"]:
                raise ValueError("JPEG digest mismatch")
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (frame["height"], frame["width"]):
                raise ValueError("JPEG decoding/dimension mismatch")
            started = time.perf_counter()
            observation = observe_people(detector, image, frame["frame_id"])
            if observation["person_detection_status"] != "ok":
                raise RuntimeError(f"detection failed: {frame['frame_id']}")
            stream.write(
                json.dumps(
                    {
                        **frame,
                        "observation": observation,
                        "inference_seconds": time.perf_counter() - started,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            if (index + 1) % 100 == 0:
                print(f"raw detections {index + 1}/{len(plan['frames'])}", flush=True)
    save(
        base / "cache-runtime.json",
        {
            "runtime": runtime,
            "opencv_threads": cv2.getNumThreads(),
            "plan_sha256": sha(base / "plan.json"),
            "model_sha256": sha(model),
            "cache_sha256": sha(cache),
            "frames": len(plan["frames"]),
            "scope": "offline cache generation, not a latency benchmark or detection validation",
            "code_sha256": {
                p: sha(base / p)
                for p in (
                    "prepare_tracking_cache.py",
                    "care_vision/cloud/yolox.py",
                    "care_vision/cloud/person.py",
                )
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-base", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    parser.add_argument(
        "--remote-python",
        default=(
            "/home/ubuntu/care-vision-phase3/releases/20261014-115106-d0341185/.venv/bin/python"
        ),
    )
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker_base:
        worker(args.worker_base)
        return
    if not all((args.manifest, args.output, args.host, args.identity)):
        parser.error("manifest, output, host and identity required")
    if args.output.exists():
        parser.error("output must be new; failed runs must be retained")
    root = Path(__file__).resolve().parents[1]
    plan = prepare(root, args.manifest, args.output)
    model = root / (
        "artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx"
    )
    if sha(model) != plan["model_sha256"]:
        raise ValueError("local model digest mismatch")
    package = args.output / "package.tar.gz"
    with tarfile.open(package, "w:gz") as archive:
        archive.add(args.output / "plan.json", arcname="plan.json")
        archive.add(args.output / "images", arcname="images")
        archive.add(model, arcname="model.onnx")
        archive.add(Path(__file__), arcname="prepare_tracking_cache.py")
        for path in sorted((root / "src/care_vision").rglob("*.py")):
            archive.add(path, arcname=str(path.relative_to(root / "src")))
    remote = f"/tmp/care-vision-tracking-cache-{uuid.uuid4().hex}"
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
    save(
        args.output / "deployment.json",
        {
            "remote": remote,
            "package_sha256": sha(package),
            "remote_python": args.remote_python,
            "worker_sha256": sha(Path(__file__)),
        },
    )
    subprocess.run([*ssh, f"umask 077; mkdir {shlex.quote(remote)}"], check=True)
    subprocess.run(["scp", *options, str(package), f"{args.host}:{remote}/"], check=True)
    command = (
        f"tar -xzf {remote}/package.tar.gz -C {remote} && "
        f"PYTHONNOUSERSITE=1 PYTHONPATH={remote}:/opt/cool/python_3.11/site-packages "
        "LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib "
        f"timeout 900 {shlex.quote(args.remote_python)} -u {remote}/prepare_tracking_cache.py "
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
                raise RuntimeError("remote cache generation failed; execution.log retained")
    for filename in ("detections.jsonl", "cache-runtime.json"):
        subprocess.run(
            ["scp", *options, f"{args.host}:{remote}/{filename}", str(args.output / filename)],
            check=True,
        )
    result = json.loads((args.output / "cache-runtime.json").read_text())
    if sha(args.output / "detections.jsonl") != result["cache_sha256"]:
        raise ValueError("downloaded cache digest mismatch")
    print(f"cache ready: {result['frames']} frames", flush=True)


if __name__ == "__main__":
    main()
