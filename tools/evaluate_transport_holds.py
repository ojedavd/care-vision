"""Control, 250 ms high-rate hold, 1.5 s baseline hold; preserve every outcome."""

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from care_vision.aws_demo import SSH, run_aws_demo, source_archive
from care_vision.check_history_delivery import check_history_delivery
from care_vision.check_loop import CheckFailed, check_records, read_log
from care_vision.check_resolution import check_resolution_records
from care_vision.check_roi import check_roi_records
from care_vision.check_states import check_state_records
from care_vision.temporal_evaluation import file_sha256
from care_vision.transport_probe import probe_process

SCENARIOS = [("control", 12, 0), ("high-rate-250ms", 12, .25), ("baseline-1500ms", 28, 1.5)]


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def local_run(args, folder, at, duration):
    folder.mkdir(exist_ok=False)
    shutil.copyfile(args.source_video, folder / "demo.avi")
    with (folder / "cloud.stderr").open("w") as error:
        cloud = subprocess.Popen([
            sys.executable, "-m", "care_vision.cloud", "--port", "0", "--run-seconds", "300",
            "--log", str(folder / "cloud.jsonl"), "--person-model", str(args.person_model),
            "--person-detector", "yolox", "--person-threshold", ".4",
            "--person-input-size", "416",
        ], stdout=error, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            ready = None
            while time.monotonic() < deadline and cloud.poll() is None:
                log = folder / "cloud.jsonl"
                if log.exists():
                    try:
                        ready = next(
                            (r for r in read_log(log) if r["event"] == "server_ready"), None
                        )
                    except json.JSONDecodeError:
                        pass
                if ready:
                    break
                time.sleep(.05)
            if not ready:
                raise RuntimeError("local cloud readiness failed")
            with probe_process(ready["port"], folder, at, duration) as port:
                with (folder / "edge.stderr").open("w") as edge_error:
                    subprocess.run([
                        sys.executable, "-m", "care_vision.edge",
                        "--video", str(folder / "demo.avi"),
                        "--url", f"ws://127.0.0.1:{port}", "--log", str(folder / "edge.jsonl"),
                        "--capture-resolution", "1280x720", "--baseline-resolution", "320x180",
                    ], stdout=edge_error, stderr=subprocess.STDOUT, timeout=240, check=True)
        finally:
            stop(cloud)


def evaluate(folder, duration):
    edge, cloud = read_log(folder / "edge.jsonl"), read_log(folder / "cloud.jsonl")
    relay = read_log(folder / "transport.jsonl")
    checks = {}
    for name, check in (
        ("loop", lambda: check_records(edge, cloud)),
        ("roi", lambda: check_roi_records(edge, cloud)),
        ("resolution", lambda: check_resolution_records(edge, cloud)),
        ("states", lambda: check_state_records(cloud)),
        ("history", lambda: check_history_delivery(edge, cloud)),
    ):
        try:
            check()
            checks[name] = {"passed": True}
        except CheckFailed as exc:
            checks[name] = {"passed": False, "reason": str(exc)}
    def ids(rows, event):
        return {r["frame_id"] for r in rows if r["event"] == event}
    sent, received = ids(edge, "frame_sent"), ids(cloud, "frame_received")
    processed, dropped = ids(cloud, "observation"), ids(cloud, "frame_dropped")
    starts = [r for r in relay if r["event"] == "hold_started"]
    ends = [r for r in relay if r["event"] == "hold_released"]
    hold_ms = (
        (ends[0]["local_mono_ns"] - starts[0]["local_mono_ns"]) / 1e6
        if starts and ends else None
    )
    active = [r for r in edge if r["event"] == "frame_sent" and r["evidence_kind"] == "live"
              and starts and r["local_mono_ns"] <= starts[0]["local_mono_ns"]]
    return {
        "healthy_demo_acceptance": all(c["passed"] for c in checks.values()),
        "checks": checks,
        "hold_exercised": (len(starts) == len(ends) == 1 and hold_ms >= duration * 1000)
        if duration else not starts and not ends,
        "actual_hold_ms": hold_ms,
        "live_target_at_hold": active[-1]["target_transmit_fps"] if active else None,
        "frames_sent": len(sent), "frames_received": len(received),
        "frames_processed": len(processed), "frames_dropped": len(dropped),
        "all_frames_accounted": sent == received == processed | dropped and not processed & dropped,
        "all_frames_processed": sent == received == processed and not dropped,
        "drain_confirmed": any(r["event"] == "drain_confirmed" for r in cloud),
        "connections": sum(r["event"] == "connected" for r in edge),
        "event_loop_lag_ms_max": {
            side: max((r["event_loop_lag_ms_max"] for r in rows
                       if r["event"] == "process_resources"
                       and r.get("event_loop_lag_ms_max") is not None), default=None)
            for side, rows in (("edge", edge), ("cloud", cloud))
        },
        "cloud_runtime": next(r["runtime"] for r in cloud if r["event"] == "server_ready"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["local", "aws"], required=True)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--person-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "aws":
        if not args.host or not args.identity:
            parser.error("AWS mode requires --host and --identity")
        SSH(args.host, args.identity).python("print('SSH_READY')")
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.tar.gz").write_bytes(source_archive(root, [
        (Path(__file__), "tools/evaluate_transport_holds.py"),
    ]))
    report = {"mode": args.mode, "complete": False, "scenarios": [],
              "scope": "Controlled upstream TCP holds; not IP loss, fall accuracy or savings.",
              "source_sha256": file_sha256(args.source_video),
              "model_sha256": file_sha256(args.person_model)}
    for name, at, duration in SCENARIOS:
        print(f"SCENARIO {name}", flush=True)
        folder = (args.output / name).resolve()
        result = {"name": name, "pause_at_seconds": at, "requested_hold_seconds": duration}
        try:
            if args.mode == "local":
                local_run(args, folder, at, duration)
            else:
                run_aws_demo(
                    args.host, args.identity, folder, root, phase=3,
                    person_model=args.person_model, person_detector="yolox", person_threshold=.4,
                    person_input_size=416, source_video=args.source_video,
                    capture_resolution="1280x720", baseline_resolution="320x180",
                    transport_pause=(at, duration),
                )
        except Exception as exc:
            result["runner_error"] = f"{type(exc).__name__}: {exc}"
        try:
            result.update(evaluate(folder, duration))
        except Exception as exc:
            result["analysis_error"] = f"{type(exc).__name__}: {exc}"
        result["target_phase_exercised"] = (
            not duration or result.get("live_target_at_hold") == (20 if at == 12 else 2)
        )
        result["file_sha256"] = {f.name: file_sha256(f) for f in folder.iterdir() if f.is_file()}
        report["scenarios"].append(result)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(r.get("analysis_error") or not r.get("hold_exercised")
           or not r.get("target_phase_exercised")
           or not r.get("all_frames_accounted") or not r.get("drain_confirmed")
           for r in report["scenarios"]):
        parser.exit(1, "Incomplete experiment; inspect report.json\n")
    print("Experiment completed; inspect healthy-demo acceptance separately.", flush=True)


if __name__ == "__main__":
    main()
