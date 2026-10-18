"""Offline Graviton detector comparison on identical native-recording JPEGs."""

import argparse
import hashlib
import json
import shlex
import statistics
import subprocess
import tarfile
import time
import uuid
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(base):
    import cv2
    import numpy as np

    from care_vision.cloud.nanodet import NanoDetPersonDetector
    from care_vision.cloud.person import observe_people
    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.environment import runtime_info

    plan = json.loads((base / "plan.json").read_text())
    cv2.setNumThreads(1)
    assert cv2.getNumThreads() == 1
    runtime = runtime_info(3)
    reports = []
    print("Graviton + COOL verified; one OpenCV thread", flush=True)
    for profile in plan["profiles"]:
        model = base / profile["filename"]
        assert digest(model) == profile["sha256"]
        options = dict(score_threshold=profile["score_threshold"], nms_threshold=0.3)
        if profile.get("family", "yolox") == "nanodet":
            detector = NanoDetPersonDetector(model, **options)
        else:
            detector = YoloXPersonDetector(model, input_size=profile["input_size"], **options)
        rows = []
        started = time.perf_counter()
        cpu_started = time.process_time()
        for frame in plan["frames"]:
            path = base / frame["filename"]
            assert digest(path) == frame["sha256"]
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            assert image is not None and image.shape[:2] == (720, 1280)
            rotation = profile.get("rotation_degrees_clockwise", 0)
            if rotation:
                image = cv2.rotate(
                    image,
                    {
                        90: cv2.ROTATE_90_CLOCKWISE,
                        180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                    }[rotation],
                )
            if profile.get("channel_order", "rgb") == "bgr":
                # Diagnostic only: cancel the production adapter's BGR -> RGB swap.
                # JPEG evidence and production preprocessing remain unchanged.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            people = observe_people(detector, image, f"{profile['label']}:{frame['capture_seq']}")
            assert people["person_detection_status"] == "ok"
            rows.append({**frame, **people, "runtime_id": runtime["cool"]["runtime_id"]})
            if len(rows) % 40 == 0:
                print(f"{profile['label']}: {len(rows)}/{len(plan['frames'])}", flush=True)
        times = sorted(row["person_detection_ms"] for row in rows)
        reports.append(
            {
                "profile": profile,
                "rows": rows,
                "processed": len(rows),
                "detector_ms_median": statistics.median(times),
                "detector_ms_p95": times[int(len(times) * 0.95 + 0.999999) - 1],
                "cpu_seconds": time.process_time() - cpu_started,
                "wall_seconds": time.perf_counter() - started,
            }
        )
    result = {"runtime": runtime, "opencv_num_threads": 1, "profiles": reports}
    (base / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("COMPLETE: all selected profiles processed identical evidence", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--s-threshold", type=float, choices=[0.4, 0.35], default=0.4)
    parser.add_argument("--floor-only", action="store_true")
    parser.add_argument("--window-seconds", type=float, nargs=2, metavar=("START", "END"))
    parser.add_argument("--source-roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--channel-orders", nargs="+", choices=["rgb", "bgr"], default=["rgb"])
    parser.add_argument("--rotations", nargs="+", type=int, choices=[0, 90, 180, 270], default=[0])
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=["yolox-nano", "yolox-tiny", "yolox-s", "nanodet"],
        default=["yolox-nano", "yolox-s"],
    )
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if "nanodet" in args.detectors and args.channel_orders != ["rgb"]:
        parser.error("NanoDet uses its native preprocessing; channel audit applies only to YOLOX")
    if args.window_seconds and (
        args.floor_only or not 0 <= args.window_seconds[0] < args.window_seconds[1]
    ):
        parser.error("window must have 0 <= START < END and cannot be combined with floor-only")
    if not all([args.capture, args.output, args.host, args.identity]) or args.output.exists():
        parser.error("capture, host, identity and new output are required")
    import cv2
    from replay_tapo_fixed_rates import select_indices

    from care_vision.edge.buffer import encode

    manifest = json.loads((args.capture / "manifest.json").read_text())
    assert manifest["complete"]
    assert digest(args.capture / "dense.avi") == manifest["video_sha256"]
    assert digest(args.capture / "captures.jsonl") == manifest["captures_sha256"]
    if args.source_roi:
        x, y, width, height = args.source_roi
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > manifest["source_width"]
            or y + height > manifest["source_height"]
        ):
            parser.error("source ROI must fit the native recording")
    captures = [
        json.loads(line) for line in (args.capture / "captures.jsonl").read_text().splitlines()
    ]
    timestamps = [row["captured_mono_ns"] for row in captures]
    assert len(captures) == manifest["decoded_frames"]
    assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
    selected = set(select_indices(timestamps, 2))
    if args.floor_only:
        selected = {i for i in selected if 52 <= (timestamps[i] - timestamps[0]) / 1e9 < 85}
    if args.window_seconds:
        start, end = args.window_seconds
        selected = {i for i in selected if start <= (timestamps[i] - timestamps[0]) / 1e9 < end}
    if not selected:
        parser.error("requested time window contains no samples")
    args.output.mkdir(parents=True)
    evidence = args.output / "evidence"
    evidence.mkdir()
    frames = []
    video = cv2.VideoCapture(str(args.capture / "dense.avi"))
    try:
        for index, capture in enumerate(captures):
            ok, image = video.read()
            assert ok and image.shape[:2] == (manifest["source_height"], manifest["source_width"])
            if index in selected:
                if args.source_roi:
                    x, y, width, height = args.source_roi
                    image = image[y : y + height, x : x + width]
                jpeg = encode(image, 1280, 720)
                filename = f"evidence/{capture['capture_seq']:06d}.jpg"
                (args.output / filename).write_bytes(jpeg)
                frames.append(
                    {
                        **capture,
                        "filename": filename,
                        "source_seconds": (capture["captured_mono_ns"] - timestamps[0]) / 1e9,
                        "sha256": hashlib.sha256(jpeg).hexdigest(),
                    }
                )
        assert not video.read()[0]
    finally:
        video.release()
    root = Path(__file__).resolve().parents[1]
    models = [
        (
            "nanodet",
            "artifacts/phase5-person-evaluation/models/object_detection_nanodet_2022nov.onnx",
            416,
        ),
        ("yolox-nano", "artifacts/phase5g-compact-detectors-20260920/models/yolox_nano.onnx", 416),
        ("yolox-tiny", "artifacts/phase5g-compact-detectors-20260920/models/yolox_tiny.onnx", 416),
        (
            "yolox-s",
            "artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx",
            640,
        ),
    ]
    models = [model for model in models if model[0] in args.detectors]
    profiles = [
        {
            "label": label,
            "filename": f"{label}.onnx",
            "sha256": digest(root / path),
            "input_size": size,
            "family": "nanodet" if label == "nanodet" else "yolox",
            "score_threshold": (
                args.s_threshold
                if label == "yolox-s"
                else 0.35
                if label in {"yolox-tiny", "nanodet"}
                else 0.4
            ),
            "nms_threshold": 0.3,
        }
        for label, path, size in models
    ]
    profiles = [
        {
            **profile,
            "label": (
                profile["label"]
                if args.channel_orders == ["rgb"]
                else f"{profile['label']}-{order}"
            ),
            "channel_order": "native" if profile["family"] == "nanodet" else order,
        }
        for profile in profiles
        for order in args.channel_orders
    ]
    profiles = [
        {
            **profile,
            "label": (
                profile["label"] if args.rotations == [0] else f"{profile['label']}-cw{rotation}"
            ),
            "rotation_degrees_clockwise": rotation,
            "bbox_coordinate_space": "rotated_image",
        }
        for profile in profiles
        for rotation in args.rotations
    ]
    plan = {
        "source_video_sha256": manifest["video_sha256"],
        "source_timestamps_sha256": manifest["captures_sha256"],
        "frames": frames,
        "profiles": profiles,
        "sampling_fps": 2,
        "jpeg_resolution": [1280, 720],
        "source_resolution": [manifest["source_width"], manifest["source_height"]],
        "source_roi": args.source_roi,
        "sample_window_seconds": args.window_seconds or ([52, 85] if args.floor_only else None),
        "view_kind": "experimental_fixed_roi" if args.source_roi else "full_frame",
        "scope": "Offline identical-JPEG detector comparison; not paced 2 FPS, no agent or alerts",
        "limits": [
            "Input sizes differ by profile; equal scores are not calibrated equivalent confidences",
            "One development scene, no independent box ground truth or fall validation",
            "Fixed ROI, when supplied, is chosen for diagnosis; it is not an agent command",
        ],
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    archive = args.output / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(args.output / "plan.json", arcname="plan.json")
        tar.add(evidence, arcname="evidence")
        tar.add(Path(__file__), arcname="worker.py")
        for path in sorted((root / "src/care_vision").rglob("*.py")):
            tar.add(path, arcname=str(path.relative_to(root / "src")))
        for label, path, _ in models:
            tar.add(root / path, arcname=f"{label}.onnx")
    remote = f"/tmp/care-vision-detectors-{uuid.uuid4().hex}"
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
    (args.output / "deployment.json").write_text(
        json.dumps({"remote": remote, "package_sha256": digest(archive)}, indent=2) + "\n"
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
                raise RuntimeError(
                    "remote comparison failed; preserve output and inspect execution.log"
                )
    subprocess.run(
        ["scp", *options, f"{args.host}:{remote}/result.json", str(args.output / "result.json")],
        check=True,
    )
    result = json.loads((args.output / "result.json").read_text())
    for profile in result["profiles"]:
        assert [(r["capture_seq"], r["sha256"]) for r in profile["rows"]] == [
            (r["capture_seq"], r["sha256"]) for r in frames
        ]
    print("Local verification: planned IDs and JPEG hashes in all selected profiles", flush=True)


if __name__ == "__main__":
    main()
