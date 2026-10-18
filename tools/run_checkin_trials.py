"""Sequential AWS rehearsals; preserve every failure instead of retrying to success."""

import argparse
import json
from pathlib import Path

from care_vision.aws_demo import run_aws_demo
from care_vision.checkin import validate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--person-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Repeated architecture rehearsals on the same development video, not incidents",
        "planned_trials": args.trials,
        "complete": False,
        "attempts": [],
    }

    def save():
        count = sum(row["passed"] for row in report["attempts"])
        report["passed_trials"] = count
        report["note"] = (
            f'{count}/{len(report["attempts"])} ensayos de arquitectura aprobados; '
            f'{args.trials} planificados. Repeticiones del mismo video de desarrollo. '
            'No valida detección de caídas ni disponibilidad garantizada.'
        )
        (args.output / "trials.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for trial in range(1, args.trials + 1):
        folder = (args.output / f"trial-{trial}").resolve()
        result = {"run": str(folder), "passed": False}
        print(f"TRIAL {trial}/{args.trials}: {folder}", flush=True)
        try:
            run_aws_demo(
                args.host, args.identity, folder, root, phase=3,
                person_model=args.person_model, person_detector="yolox", person_threshold=0.4,
                person_input_size=416, source_video=args.source_video,
                capture_resolution="1280x720", baseline_resolution="320x180",
            )
            summary, _ = validate_bundle(folder)
            (folder / "checkin-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            result.update(passed=summary["passed"], metrics=summary["metrics"])
            result["checks"] = {
                key: {k: v for k, v in value.items() if k != "report"}
                for key, value in summary["checks"].items()
            }
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc) or repr(exc))
        report["attempts"].append(result)
        save()
        print(f"TRIAL {trial}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    report["complete"] = True
    save()
    if report["passed_trials"] != args.trials:
        parser.exit(1, f"Some trials failed; see {args.output / 'trials.json'}\n")


if __name__ == "__main__":
    main()
