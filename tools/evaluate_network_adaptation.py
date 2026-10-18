"""Evaluate the deterministic Edge transmission cap under injected RTT samples."""

import argparse
import json
from pathlib import Path

from care_vision.check_loop import read_log
from care_vision.common.contracts import dumps
from care_vision.common.metrics import EventLog
from care_vision.common.transport import Outbox
from care_vision.edge.runtime import Edge
from care_vision.edge.source import VideoSource
from care_vision.temporal_evaluation import file_sha256


def selected(edge, first_seq, last_seq, start_ns):
    return sum(
        edge.select_live_capture(seq, start_ns + (seq - 1) * 1_000_000_000 // 30)
        for seq in range(first_seq, last_seq + 1)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    args.output.mkdir(parents=True)
    log = EventLog("edge", args.output / "edge.jsonl")
    edge = Edge(VideoSource(str(args.output / "unused.avi")), "ws://127.0.0.1:1", log)
    edge.outbox = Outbox()  # Connected test harness; no video source or socket is started.
    start = 100_000_000_000
    try:
        edge.network.connected(0)
        edge.sampler.set_rate(20, start)
        normal = selected(edge, 1, 30, start)
        edge.observe_network_rtt(0.3, start + 1_000_000_000)
        after_one_high = edge.network.transmission_cap_fps
        edge.observe_network_rtt(0.3, start + 1_000_000_000)
        limited_status = edge.status()
        dumps(limited_status)
        limited = selected(edge, 31, 60, start)
        limited_frames = edge.network_limited_frames
        edge.observe_network_rtt(0.1, start + 2_000_000_000)
        edge.observe_network_rtt(0.1, start + 2_000_000_000)
        recovered_status = edge.status()
        dumps(recovered_status)
        recovered = selected(edge, 61, 90, start)
        edge.sampler.set_rate(2, start + 3_000_000_000)
        edge.observe_network_rtt(0.3, start + 3_000_000_000)
        edge.observe_network_rtt(0.3, start + 3_000_000_000)
        baseline_status = edge.status()
        baseline = selected(edge, 91, 120, start)
    finally:
        log.close()
    changes = [
        r
        for r in read_log(args.output / "edge.jsonl")
        if r["event"] == "network_transmission_cap_changed"
    ]
    if not (
        after_one_high is None
        and 18 <= normal <= 21
        and 4 <= limited <= 6
        and limited_frames >= 12
        and 18 <= recovered <= 21
        and 1 <= baseline <= 3
        and limited_status["target_transmit_fps"] == 20
        and limited_status["effective_target_transmit_fps"] == 5
        and recovered_status["effective_target_transmit_fps"] == 20
        and baseline_status["effective_target_transmit_fps"] == 2
        and [r["cap_fps"] for r in changes] == [5, None, 5]
    ):
        raise AssertionError("network transmission cap did not follow the expected trace")
    report = {
        "scope": "Deterministic 30 FPS timestamps and injected RTT; no real bandwidth impairment",
        "capture_frames_per_window": 30,
        "requested_fps": 20,
        "selected_normal": normal,
        "selected_measured_degraded": limited,
        "network_limited_frames": limited_frames,
        "selected_recovered": recovered,
        "selected_baseline_under_degradation": baseline,
        "cap_after_one_high_rtt": after_one_high,
        "cap_after_two_high_rtt": limited_status["network_cap_fps"],
        "effective_target_degraded": limited_status["effective_target_transmit_fps"],
        "effective_target_recovered": recovered_status["effective_target_transmit_fps"],
        "effective_target_baseline": baseline_status["effective_target_transmit_fps"],
        "cap_transitions": len(changes),
        "limits": [
            "RTT was injected; no packet loss, bandwidth restriction, or AWS path was measured.",
            "The 250/150 ms thresholds and 5 FPS cap are experimental, not calibrated.",
            "No fall-detection or incident-performance claim follows from this trace.",
        ],
        "file_sha256": {"edge.jsonl": file_sha256(args.output / "edge.jsonl")},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
