"""Run the real edge and cloud processes, then check the resulting evidence."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from care_vision.check_loop import check_records, read_log
from care_vision.fixture import make_video


def run_demo(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    video = make_video(output / "demo.avi")
    edge_log, cloud_log = output / "edge.jsonl", output / "cloud.jsonl"
    with (output / "cloud.stderr").open("w") as cloud_err:
        cloud = subprocess.Popen(
            [sys.executable, "-m", "care_vision.cloud", "--port", "0", "--log", str(cloud_log)],
            stdout=cloud_err,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 10
            port = None
            while time.monotonic() < deadline:
                if cloud.poll() is not None:
                    raise RuntimeError(f"cloud exited; inspect {output / 'cloud.stderr'}")
                if cloud_log.exists():
                    try:
                        ready = [r for r in read_log(cloud_log) if r["event"] == "server_ready"]
                    except json.JSONDecodeError:
                        ready = []
                    if ready:
                        port = ready[-1]["port"]
                        break
                time.sleep(0.05)
            if port is None:
                raise RuntimeError("cloud readiness timeout")
            print(
                f"Local vision worker ready on port {port}; 34 s video, 30 FPS capture.", flush=True
            )
            with (output / "edge.stderr").open("w") as edge_err:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "care_vision.edge",
                        "--video",
                        str(video),
                        "--url",
                        f"ws://127.0.0.1:{port}",
                        "--log",
                        str(edge_log),
                    ],
                    stdout=edge_err,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=240,
                )
            # The final observation is logged before checking; no cross-host clock arithmetic.
            time.sleep(0.2)
            report = check_records(read_log(edge_log), read_log(cloud_log))
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
        finally:
            cloud.terminate()
            try:
                cloud.wait(timeout=3)
            except subprocess.TimeoutExpired:
                cloud.kill()
                cloud.wait()


def main():
    parser = argparse.ArgumentParser(description="CARE-Vision local demonstration: 2 → 20 → 2 FPS")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts") / time.strftime("%Y%m%d-%H%M%S")
    )
    args = parser.parse_args()
    # Avoid silently mixing or overwriting earlier evidence.
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be new or empty")
    try:
        report = run_demo(args.output.resolve())
    except Exception as exc:
        parser.exit(1, f"FAIL: {exc}\nLogs: {args.output.resolve()}\n")
    print(json.dumps(report, indent=2))
    print(f"PASS: local loop verified. Evidence: {args.output.resolve()}")


if __name__ == "__main__":
    main()
