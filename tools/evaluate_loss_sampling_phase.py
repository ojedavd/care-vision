"""Diagnostic Nano 2 FPS sampling-phase sweep on Graviton, without S or WebSocket.

The preparation side selects, resizes and encodes recorded frames once. The
WebSocket Edge uses a second JPEG encode; diagnostic timings from this tool
must not be treated as exact service predictions. Detection, tracking and the
loss trigger execute in the remote COOL runtime.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import tarfile
import uuid
from pathlib import Path

SCENES = {
    "lateral": "artifacts/tapo-heldout-20261004/controlled-lateral-1",
    "natural": "artifacts/tapo-fixed-rate-evaluation-20261003/natural-2",
    "controlled1": "artifacts/tapo-fixed-rate-evaluation-20261003/controlled-1",
}
PHASE_MS = (0, 100, 200, 300, 400)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root, output):
    import cv2

    from care_vision.edge.buffer import encode
    from care_vision.edge.sampler import Sampler

    output.mkdir(parents=True)
    manifest = {"phases_ms": PHASE_MS, "scenes": {}, "scope": "known development videos"}
    for scene, relative in SCENES.items():
        folder = root / relative
        source = json.loads((folder / "manifest.json").read_text())
        video_path, captures_path = folder / "dense.avi", folder / "captures.jsonl"
        assert source["complete"] and sha(video_path) == source["video_sha256"]
        assert sha(captures_path) == source["captures_sha256"]
        captures = [json.loads(line) for line in captures_path.read_text().splitlines()]
        assert len(captures) == source["decoded_frames"]
        origin = captures[0]["captured_mono_ns"]
        samplers = {}
        for phase in PHASE_MS:
            sampler = Sampler(2)
            sampler.set_rate(2, origin + phase * 1_000_000)
            samplers[phase] = sampler
        phases = {str(phase): [] for phase in PHASE_MS}
        video = cv2.VideoCapture(str(video_path))
        assert video.isOpened() and round(video.get(cv2.CAP_PROP_FRAME_COUNT)) == len(captures)
        try:
            for capture in captures:
                ok, image = video.read()
                assert ok and image.shape[:2] == (
                    source["source_height"],
                    source["source_width"],
                )
                seq, stamp = capture["capture_seq"], capture["captured_mono_ns"]
                selected = [
                    phase for phase, sampler in samplers.items() if sampler.select(seq, stamp)
                ]
                if not selected:
                    continue
                relative_image = f"images/{scene}/{seq}.jpg"
                image_path = output / relative_image
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(encode(image, 320, 180))
                entry = {
                    "capture_seq": seq,
                    "captured_mono_ns": stamp,
                    "source_seconds": round((stamp - origin) / 1e9, 3),
                    "image": relative_image,
                    "sha256": sha(image_path),
                }
                for phase in selected:
                    phases[str(phase)].append(entry)
        finally:
            video.release()
        manifest["scenes"][scene] = {
            "video_sha256": source["video_sha256"],
            "captures_sha256": source["captures_sha256"],
            "phases": phases,
        }
    (output / "phase-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def worker(base):
    import cv2
    import numpy as np

    from care_vision.cloud.loss_probe import LossProbe
    from care_vision.cloud.person import observe_people
    from care_vision.cloud.tracking import PersonTracker
    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.environment import runtime_info

    manifest = json.loads((base / "phase-manifest.json").read_text())
    model = base / "nano.onnx"
    assert sha(model) == manifest["model_sha256"]
    cv2.setNumThreads(8)
    detector = YoloXPersonDetector(model, input_size=416, score_threshold=0.4, nms_threshold=0.3)
    runtime = runtime_info(3)
    assert runtime["cool"]["verified"] and runtime["machine"] == "aarch64"
    print(
        json.dumps({"event": "ready", "runtime": runtime, "threads": cv2.getNumThreads()}),
        flush=True,
    )
    for scene, detail in manifest["scenes"].items():
        for phase, entries in detail["phases"].items():
            tracker, probe = PersonTracker(), LossProbe()
            triggers = []
            detected = 0
            for entry in entries:
                path = base / entry["image"]
                jpeg = path.read_bytes()
                assert hashlib.sha256(jpeg).hexdigest() == entry["sha256"]
                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                assert image is not None and image.shape[:2] == (180, 320)
                people = observe_people(detector, image, str(entry["capture_seq"]))
                people = tracker.update(people, entry["captured_mono_ns"], f"phase:{scene}:{phase}")
                detected += bool(people["person_detected"])
                lost = probe.observe(entry["captured_mono_ns"], people)
                if lost:
                    triggers.append(
                        {
                            "source_seconds": entry["source_seconds"],
                            "capture_seq": entry["capture_seq"],
                            "last_seen_source_seconds": round(
                                (lost["captured_mono_ns"] - entries[0]["captured_mono_ns"]) / 1e9
                                + entries[0]["source_seconds"],
                                3,
                            ),
                        }
                    )
            print(
                json.dumps(
                    {
                        "event": "phase_result",
                        "scene": scene,
                        "phase_ms": int(phase),
                        "sampled": len(entries),
                        "detected": detected,
                        "triggers": triggers,
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not all((args.output, args.host, args.identity)) or args.output.exists():
        parser.error("new output directory, host and SSH identity are required")
    root = Path(__file__).resolve().parents[1]
    manifest = prepare(root, args.output)
    model_path = root / "artifacts/phase5g-compact-detectors-20260920/models/yolox_nano.onnx"
    manifest["model_sha256"] = sha(model_path)
    (args.output / "phase-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = args.output / "phase-package.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(Path(__file__), arcname="worker.py")
        tar.add(args.output / "phase-manifest.json", arcname="phase-manifest.json")
        tar.add(model_path, arcname="nano.onnx")
        for path in sorted((root / "src/care_vision").rglob("*.py")):
            tar.add(path, arcname=str(path.relative_to(root / "src")))
        for path in sorted((args.output / "images").rglob("*.jpg")):
            tar.add(path, arcname=str(path.relative_to(args.output)))
    remote = f"/tmp/care-vision-phase-{uuid.uuid4().hex}"
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
    subprocess.run(["ssh", *options, args.host, f"mkdir {shlex.quote(remote)}"], check=True)
    subprocess.run(
        ["scp", *options, str(archive), f"{args.host}:{remote}/phase-package.tar.gz"],
        check=True,
    )
    command = (
        f"tar -xzf {remote}/phase-package.tar.gz -C {remote} && "
        f"PYTHONPATH={remote}:/opt/cool/python_3.11/site-packages "
        f"LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib "
        f"timeout 900 /usr/bin/python3.11 -u {remote}/worker.py --worker {remote}"
    )
    with (args.output / "cloud.jsonl").open("w") as log:
        process = subprocess.Popen(
            ["ssh", *options, args.host, command],
            stdout=log,
            stderr=(args.output / "cloud.stderr").open("w"),
            text=True,
        )
        if process.wait() != 0:
            raise RuntimeError("Graviton phase worker failed; see cloud.stderr")
    rows = [json.loads(line) for line in (args.output / "cloud.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "ready" and rows[0]["runtime"]["cool"]["verified"]
    results = [row for row in rows if row["event"] == "phase_result"]
    assert len(results) == len(SCENES) * len(PHASE_MS)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
