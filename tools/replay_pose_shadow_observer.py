"""Exercise the optional observer in Graviton with recorded observations/JPEGs.

Sequential drain guarantees coverage, not live performance or detector parity.
"""

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import cv2

from care_vision.cloud.pose_shadow import shadow_factory
from care_vision.common.environment import runtime_info
from care_vision.common.metrics import EventLog


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def replay(plan_path, config_path, output):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(plan_path.read_text())
    environment = runtime_info(3)
    if environment["ec2"].get("architecture") != "arm64" or not environment["cool"]["verified"]:
        raise RuntimeError("observer image replay requires Graviton and COOL")
    cv2.setNumThreads(8)
    create = shadow_factory(config_path)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    results = []
    for source in plan["runs"]:
        log_path = root / source["cloud_path"]
        if sha(log_path) != source["cloud_sha256"]:
            raise ValueError("recorded observation source changed")
        inputs = root / source["input_path"]
        if sha(inputs / "plan.json") != source["input_plan_sha256"]:
            raise ValueError("JPEG input plan changed")
        frames = {f["capture_seq"]: f for f in json.loads(
            (inputs / "plan.json").read_text()
        )["frames"] if f["run"] == source["run"]}
        log = EventLog("cloud-pose-shadow", output / (source["run"] + ".jsonl"))
        observer, active, pending = create(log), None, None

        async def flush():
            if pending is None:
                return
            frame = frames[pending["capture_seq"]]
            if frame["captured_mono_ns"] != pending["captured_mono_ns"]:
                raise ValueError("pose shadow frame timestamp differs")
            path = inputs / frame["filename"]
            if sha(path) != frame["jpeg_sha256"]:
                raise ValueError("pose shadow JPEG differs")
            observer.submit(pending, path.read_bytes(), active)
            # Replay coverage, deliberately not a throughput/live backpressure test.
            await observer.queue.join()

        try:
            for row in map(json.loads, log_path.read_text().splitlines()):
                if row["event"] == "observation":
                    await flush()
                    pending = row
                elif row["event"] == "incident_episode":
                    if row["incident_status"] in {"recovery_observed", "posture_change_cleared"}:
                        active = None
                    else:
                        active = row
            await flush()
        finally:
            await observer.close()
            log.close()
        records = [json.loads(line) for line in (
            output / (source["run"] + ".jsonl")
        ).read_text().splitlines()]
        decisions = [r for r in records if r["event"] == "pose_shadow"]
        if any(r["would_close_episode"] or r["seat_support_verified"] for r in decisions):
            raise ValueError("shadow closure invariant violated")
        results.append({
            "run": source["run"], "source_sha256": source["cloud_sha256"],
            "decisions": len(decisions),
            "pose_status": dict(Counter(r["pose_status"] for r in decisions)),
            "reason_counts": dict(Counter(r["reason"] for r in decisions)),
            "candidates": sum(r["seated_pose_candidate"] for r in decisions),
            "log_sha256": sha(output / (source["run"] + ".jsonl")),
        })
    report = {
        "scope": "Sequential recorded-observation observer; no new detector or live test",
        "preprocessing": "RTMPose direct affine192x256; differs from common-ROI comparison",
        "runtime": environment, "threads": cv2.getNumThreads(),
        "plan_sha256": sha(plan_path), "config_sha256": sha(config_path),
        "code_sha256": {str(p.relative_to(root)): sha(p) for p in [
            root / "src/care_vision/cloud/pose_shadow.py", Path(__file__),
        ]}, "runs": results, "live_activation_supported": False,
        "all_suggestions_passive": all(
            r.get("suggested_observation") is None
            or r["suggested_observation"]["dispatched"] is False
            for source in plan["runs"]
            for r in map(json.loads, (output / (source["run"] + ".jsonl")).read_text().splitlines())
            if r["event"] == "pose_shadow"
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    asyncio.run(replay(args.plan, args.config, args.output))


if __name__ == "__main__":
    main()
