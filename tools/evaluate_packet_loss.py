"""One AWS trial each at 0%, 1%, 3% netem loss; preserve failed trials."""

import argparse
import json
from pathlib import Path

from evaluate_transport_holds import evaluate

from care_vision.aws_demo import run_aws_demo
from care_vision.check_loop import read_log
from care_vision.temporal_evaluation import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--person-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    report = {"complete": False, "scenarios": [],
              "scope": "Kernel packet drops on isolated inner TCP legs after SSH; not WAN loss",
              "source_sha256": file_sha256(args.source_video),
              "model_sha256": file_sha256(args.person_model)}
    for loss in (0, 1, 3):
        folder = args.output / f"loss-{loss}pct"
        result = {"name": folder.name, "configured_loss_percent": loss}
        print(f"SCENARIO {folder.name}", flush=True)
        try:
            run_aws_demo(args.host, args.identity, folder.resolve(), root, phase=3,
                         person_model=args.person_model, person_detector="yolox",
                         person_threshold=.4, person_input_size=416,
                         source_video=args.source_video, capture_resolution="1280x720",
                         baseline_resolution="320x180", transport_pause=(12, 0), packet_loss=loss)
        except Exception as exc:
            result["runner_error"] = f"{type(exc).__name__}: {exc}"
        try:
            result.update(evaluate(folder, 0))
            netem = read_log(folder / "netem.jsonl")
            final = next(r for r in netem if r["event"] == "final")
            qdisc = next(q for q in final["qdisc"] if q["kind"] == "netem")
            result["kernel_qdisc_final"] = qdisc
            result["kernel_drops"] = qdisc["drops"]
            result["loss_exercised"] = qdisc["drops"] > 0 if loss else qdisc["drops"] == 0
            result["cleanup_confirmed"] = any(
                r["event"] == "cleanup" and r["success"] for r in netem)
        except Exception as exc:
            result["analysis_error"] = f"{type(exc).__name__}: {exc}"
        result["file_sha256"] = {f.name: file_sha256(f) for f in folder.iterdir() if f.is_file()}
        report["scenarios"].append(result)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if not result.get("cleanup_confirmed"):
            parser.exit(1, "Stopped: inspect diagnostic cleanup before another trial\n")
    report["complete"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(s.get("analysis_error") or not s.get("loss_exercised") for s in report["scenarios"]):
        parser.exit(1, "Loss experiment incomplete; inspect retained evidence\n")
    print("All loss conditions exercised; healthy-demo acceptance is separate.", flush=True)


if __name__ == "__main__":
    main()
