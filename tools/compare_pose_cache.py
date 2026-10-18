"""Offline RTMPose/YOLO11 pose comparison on common frozen YOLOX-S regions.

JPEG acquisition/encoding is local; all image inference is Graviton/OpenCV DNN.
No posture labels inferred from scripts, no live incident decisions.
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
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    path.chmod(0o600)


def rtm_on_crop(estimator, crop):
    import cv2
    import numpy as np

    resized = cv2.resize(crop, estimator.input_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = np.ascontiguousarray(((rgb - estimator.mean) / estimator.std).transpose(2, 0, 1)[None])
    estimator.net.setInput(blob)
    outputs = estimator.net.forward(estimator.net.getUnconnectedOutLayersNames())
    height, width = crop.shape[:2]
    return estimator.decode(outputs, np.array([width / 2, height / 2]), np.array([width, height]))


def worker(base):
    import cv2
    import numpy as np
    from yolo_pose_candidate import YoloPoseCandidate, common_crop, restore_points

    from care_vision.cloud.pose_evidence import assess_pose_support
    from care_vision.cloud.rtmpose import RTMPoseEstimator
    from care_vision.common.environment import runtime_info

    plan = json.loads((base / "pose-plan.json").read_text())
    runtime = runtime_info(3)
    if not runtime["cool"]["verified"] or runtime["ec2"]["architecture"] != "arm64":
        raise RuntimeError("requires verified Graviton/COOL")
    cv2.setNumThreads(8)
    estimators, errors = {}, {}
    for name, constructor in (("RTMPose", RTMPoseEstimator), ("YOLO11n-pose", YoloPoseCandidate)):
        item = plan["models"][name]
        path = base / item["filename"]
        if sha(path) != item["sha256"]:
            raise ValueError("pose model digest mismatch")
        try:
            estimators[name] = constructor(path)
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    statuses = {name: Counter() for name in plan["models"]}
    latencies = {name: [] for name in plan["models"]}
    warmed = set()
    output = base / "poses.jsonl"
    with output.open("x") as stream:
        output.chmod(0o600)
        for index, item in enumerate(plan["frames"]):
            path = base / item["filename"]
            if sha(path) != item["jpeg_sha256"]:
                raise ValueError("JPEG digest mismatch")
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (720, 1280):
                raise ValueError("unexpected JPEG")
            crop_started = time.perf_counter()
            crop, box = common_crop(image, item["bbox"])
            crop_seconds = time.perf_counter() - crop_started
            x, y, w, h = item["bbox"]
            target = [
                (x - box[0]) * crop.shape[1] / box[2],
                (y - box[1]) * crop.shape[0] / box[3],
                w * crop.shape[1] / box[2],
                h * crop.shape[0] / box[3],
            ]
            predictions = {}
            for name in plan["models"]:
                if name not in estimators:
                    result = {"status": "model_load_error", "error": errors[name], "points": None}
                else:
                    estimator = estimators[name]
                    started = time.perf_counter()
                    try:

                        def estimate():
                            if name == "RTMPose":
                                return rtm_on_crop(estimator, crop), {"status": "ok"}
                            return estimator.estimate(crop, target)

                        if name not in warmed:
                            for _ in range(2):
                                estimate()
                            warmed.add(name)
                            started = time.perf_counter()
                        points, metadata = estimate()
                        elapsed = time.perf_counter() - started
                        points = None if points is None else restore_points(points, box, crop.shape)
                        support = assess_pose_support(points, item["bbox"], image.shape)
                        result = {
                            **metadata,
                            "points": None if points is None else points.tolist(),
                            "support": support,
                            "inference_seconds": elapsed,
                        }
                        latencies[name].append(elapsed)
                    except Exception as error:
                        result = {
                            "status": "inference_error",
                            "error": f"{type(error).__name__}: {error}",
                            "points": None,
                        }
                statuses[name][result["status"]] += 1
                predictions[name] = result
            stream.write(
                json.dumps(
                    {
                        **item,
                        "common_crop_bbox": box,
                        "common_crop_shape": list(crop.shape),
                        "common_crop_seconds": crop_seconds,
                        "predictions": predictions,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            if (index + 1) % 100 == 0:
                print(f"pose pairs {index + 1}/{len(plan['frames'])}", flush=True)
    save(
        base / "pose-runtime.json",
        {
            "runtime": runtime,
            "threads": cv2.getNumThreads(),
            "numpy_version": np.__version__,
            "numpy_module": np.__file__,
            "plan_sha256": sha(base / "pose-plan.json"),
            "poses_sha256": sha(output),
            "frames": len(plan["frames"]),
            "model_load_errors": errors,
            "statuses": {k: dict(v) for k, v in statuses.items()},
            "latency_seconds": {
                k: {
                    "count": len(v),
                    "median": float(np.median(v)) if v else None,
                    "p95": float(np.percentile(v, 95)) if v else None,
                }
                for k, v in latencies.items()
            },
            "warmup_calls_per_loaded_model": 2,
            "scope": (
                "offline common-crop DNN compatibility and diagnostics; "
                "interleaved timing; not isolated benchmark, visibility accuracy or validation"
            ),
            "code_sha256": {
                n: sha(base / n)
                for n in (
                    "compare_pose_cache.py",
                    "yolo_pose_candidate.py",
                    "care_vision/cloud/rtmpose.py",
                    "care_vision/cloud/pose_evidence.py",
                )
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-base", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker_base:
        worker(args.worker_base)
        return
    if not all((args.cache, args.models, args.output, args.host, args.identity)):
        parser.error("cache, models, output, host and identity required")
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    cached_plan = json.loads((args.cache / "plan.json").read_text())
    cached_runtime = json.loads((args.cache / "cache-runtime.json").read_text())
    if sha(args.cache / "detections.jsonl") != cached_runtime["cache_sha256"]:
        raise ValueError("raw cache digest mismatch")
    if sha(args.cache / "plan.json") != cached_runtime["plan_sha256"]:
        raise ValueError("cache plan digest mismatch")
    raw = list(map(json.loads, (args.cache / "detections.jsonl").read_text().splitlines()))
    if len(raw) != len(cached_plan["frames"]):
        raise ValueError("incomplete detector cache")
    selected, exclusions = [], []
    for expected, row in zip(cached_plan["frames"], raw, strict=True):
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError("raw frame metadata mismatch")
        people = row["observation"]["persons"]
        if row["observation"]["person_detection_status"] != "ok" or len(people) != 1:
            exclusions.append({"frame_id": row["frame_id"], "reason": "not_one_raw_person"})
        elif people[0]["confidence"] < 0.35:
            exclusions.append({"frame_id": row["frame_id"], "reason": "weak_source_detection"})
        else:
            selected.append(
                {
                    **expected,
                    "bbox": row.get("pose_bbox", people[0]["bbox"]),
                    "source_confidence": people[0]["confidence"],
                }
            )
    provenance = json.loads((args.models.parent / "model-provenance.json").read_text())
    model_paths = {
        "RTMPose": root / "artifacts/phase7d-pose-20260921/models/end2end.onnx",
        "YOLO11n-pose": args.models / "yolo11n-pose.onnx",
    }
    if (
        sha(model_paths["RTMPose"])
        != "a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f"
    ):
        raise ValueError("RTMPose digest mismatch")
    if sha(model_paths["YOLO11n-pose"]) != provenance["onnx_sha256"]:
        raise ValueError("YOLO pose digest mismatch")
    plan = {
        "schema_version": 1,
        "cache_sha256": sha(args.cache / "detections.jsonl"),
        "cache_plan_sha256": sha(args.cache / "plan.json"),
        "runs": cached_plan["runs"],
        "frames": selected,
        "exclusions": exclusions,
        "models": {
            name: {"filename": name + ".onnx", "sha256": sha(path)}
            for name, path in model_paths.items()
        },
        "yolo_provenance": provenance,
        "common_roi": (
            cached_plan.get("region_source", "YOLOX-S raw bbox")
            + "; padding1.25, expanded to3:4 aspect; native-size affine"
        ),
        "rtmpose_preprocess": "commonROI resize192x256, RGB/SDK mean/std; no second bbox expansion",
        "yolo_preprocess": "commonROI letterbox640, RGB/255; target IoU>=0.3, margin>=0.1",
        "support_threshold": 0.3,
        "scope": "development/offline; no activity labels or episode decisions",
    }
    save(args.output / "pose-plan.json", plan)
    package = args.output / "package.tar.gz"
    with tarfile.open(package, "w:gz") as archive:
        archive.add(args.output / "pose-plan.json", arcname="pose-plan.json")
        for item in selected:
            archive.add(args.cache / item["filename"], arcname=item["filename"])
        for name, path in model_paths.items():
            archive.add(path, arcname=name + ".onnx")
        for filename in ("compare_pose_cache.py", "yolo_pose_candidate.py"):
            archive.add(Path(__file__).with_name(filename), arcname=filename)
        for path in (root / "src/care_vision").rglob("*.py"):
            archive.add(path, arcname=str(path.relative_to(root / "src")))
    remote = "/tmp/care-vision-pose-cache-" + uuid.uuid4().hex
    python = "/home/ubuntu/care-vision-phase3/releases/20261014-115106-d0341185/.venv/bin/python"
    save(
        args.output / "deployment.json",
        {"remote": remote, "package_sha256": sha(package), "remote_python": python},
    )
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
    subprocess.run([*ssh, f"umask 077; mkdir {remote}"], check=True)
    subprocess.run(["scp", *options, str(package), f"{args.host}:{remote}/"], check=True)
    command = (
        f"tar -xzf {remote}/package.tar.gz -C {remote} && PYTHONNOUSERSITE=1 "
        f"PYTHONPATH={remote}:/opt/cool/python_3.11/site-packages "
        "LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib "
        f"timeout 900 {shlex.quote(python)} -u {remote}/compare_pose_cache.py "
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
                raise RuntimeError("pose worker failed; evidence retained")
    for filename in ("poses.jsonl", "pose-runtime.json"):
        subprocess.run(
            ["scp", *options, f"{args.host}:{remote}/{filename}", str(args.output / filename)],
            check=True,
        )
    report = json.loads((args.output / "pose-runtime.json").read_text())
    if sha(args.output / "poses.jsonl") != report["poses_sha256"]:
        raise ValueError("pose download digest mismatch")
    subprocess.run([*ssh, f"rm -rf {remote} && test ! -e {remote}"], check=True)
    save(
        args.output / "cleanup.json",
        {
            "temporary_workdir_removed": True,
            "offline_worker_exited": True,
            "instance_left_running": True,
            "pending_episode_snapshot_untouched": True,
        },
    )
    print("pose comparison ready", report["statuses"], flush=True)


if __name__ == "__main__":
    main()
