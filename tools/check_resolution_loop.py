import argparse
import json

from care_vision.check_loop import CheckFailed, check_records, read_log
from care_vision.check_resolution import check_resolution_records
from care_vision.check_roi import check_roi_records


def main():
    parser = argparse.ArgumentParser(description="Verify CARE-Vision resolution and ROI loop")
    parser.add_argument("--edge-log", required=True)
    parser.add_argument("--cloud-log", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    edge, cloud = read_log(args.edge_log), read_log(args.cloud_log)
    try:
        report = check_records(edge, cloud)
        report["roi"] = check_roi_records(edge, cloud)
        report["resolution"] = check_resolution_records(edge, cloud)
    except CheckFailed as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        with open(args.report, "x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
