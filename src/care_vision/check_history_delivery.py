"""Verify complete history delivery and cloud-confirmed EOF, independently of demo choreography."""

import argparse
import json

from care_vision.check_loop import read_log, require


def check_history_delivery(edge, cloud):
    sent = [r for r in edge if r["event"] == "frame_sent"]
    received = [r for r in cloud if r["event"] == "frame_received"]
    observations = [r for r in cloud if r["event"] == "observation"]
    for rows in (sent, received, observations):
        require(len({r["frame_id"] for r in rows}) == len(rows), "duplicate delivery evidence")
    require(
        {r["frame_id"] for r in sent} == {r["frame_id"] for r in received},
        "transmitted frames missing from cloud",
    )
    require(
        {r["frame_id"] for r in received} == {r["frame_id"] for r in observations},
        "received frames missing observations",
    )
    failures = {
        "buffer_transfer_aborted",
        "buffer_incomplete",
        "connection_error",
        "session_error",
        "drain_timeout",
        "frame_dropped",
    }
    require(
        not any(r["event"] in failures for r in edge + cloud), "delivery has recorded failure/drop"
    )
    requests = [
        r for r in cloud if r["event"] == "agent_action" and r["type"] == "REQUEST_RECENT_BUFFER"
    ]
    require(requests, "no history request exercised")
    reports = []
    for request in requests:
        request_id = request["command_id"]
        ack = [
            r
            for r in cloud
            if r["event"] == "ack_received"
            and r["command_id"] == request_id
            and r["result"] == "applied"
        ]
        done = [
            r
            for r in cloud
            if r["event"] == "buffer_processed" and r["buffer_request_id"] == request_id
        ]
        rows = [r for r in observations if r.get("buffer_request_id") == request_id]
        require(len(ack) == len(done) == 1, "history ACK/completion missing or duplicated")
        require(
            ack[0]["selected_frames"] == len(rows) == done[0]["frames_processed"],
            "selected history was not fully processed",
        )
        return_to_low = next(
            (
                r
                for r in cloud
                if r["event"] == "agent_action"
                and r["type"] == "SET_TRANSMIT_FPS"
                and r["transmit_fps"] == 2
                and r["local_mono_ns"] > request["local_mono_ns"]
            ),
            None,
        )
        reports.append(
            {
                "buffer_request_id": request_id,
                "selected_and_processed": len(rows),
                "first_processing_after_request_ms": (
                    rows[0]["local_mono_ns"] - request["local_mono_ns"]
                )
                / 1e6
                if rows
                else None,
                "first_processing_before_return_to_low": bool(
                    rows
                    and (
                        return_to_low is None
                        or rows[0]["local_mono_ns"] < return_to_low["local_mono_ns"]
                    )
                ),
                "completed_after_request_ms": (done[0]["local_mono_ns"] - request["local_mono_ns"])
                / 1e6,
            }
        )
    endings = [r for r in edge if r["event"] == "drain_complete"]
    require(len(endings) == 1, "expected one confirmed EOF drain")
    end = endings[0]
    confirmations = [
        r for r in cloud if r["event"] == "drain_confirmed" and r["drain_id"] == end["drain_id"]
    ]
    require(len(confirmations) == 1, "cloud drain confirmation missing")
    for key in (
        "session_id",
        "live_received",
        "live_processed",
        "live_dropped",
        "historical_received",
        "historical_processed",
    ):
        require(end[key] == confirmations[0][key], f"drain ACK mismatch: {key}")
    live = [
        r
        for r in sent
        if r["session_id"] == end["session_id"] and r.get("evidence_kind", "live") == "live"
    ]
    historical = [
        r
        for r in sent
        if r["session_id"] == end["session_id"] and r.get("evidence_kind") == "historical"
    ]
    require(
        len(live) == end["live_received"] == end["live_processed"] and end["live_dropped"] == 0,
        "EOF live evidence incomplete",
    )
    require(
        len(historical) == end["historical_received"] == end["historical_processed"],
        "EOF historical evidence incomplete",
    )
    starts = [r for r in edge if r["event"] == "drain_started"]
    require(len(starts) == 1, "EOF drain start missing")
    elapsed = (end["local_mono_ns"] - starts[0]["local_mono_ns"]) / 1e9
    require(0 <= elapsed <= starts[0]["timeout_seconds"], "EOF deadline exceeded")
    require(
        any(r["first_processing_before_return_to_low"] for r in reports),
        "history never processed before returning to low rate",
    )
    return {
        "passed": True,
        "frames": len(sent),
        "history": reports,
        "eof_drain_seconds": elapsed,
        "live_dropped": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-log", required=True)
    parser.add_argument("--cloud-log", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = check_history_delivery(read_log(args.edge_log), read_log(args.cloud_log))
    with open(args.report, "x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
