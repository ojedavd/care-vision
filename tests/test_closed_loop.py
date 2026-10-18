import copy

import pytest

from care_vision.check_loop import CheckFailed, check_records, read_log
from care_vision.demo import run_demo


@pytest.mark.integration
def test_separate_processes_and_checker_rejects_false_success(tmp_path):
    report = run_demo(tmp_path)
    assert report["passed"]
    assert report["temporal_buffer"]["verified"]
    assert report["frames_received"] == report["observations"]
    edge, cloud = read_log(tmp_path / "edge.jsonl"), read_log(tmp_path / "cloud.jsonl")
    # Simulate unrelated machine clock origins. Transport rates and round trips
    # must still pass without synchronizing edge and cloud monotonic clocks.
    shifted_edge = copy.deepcopy(edge)
    for record in shifted_edge:
        record["local_mono_ns"] += 600_000_000_000
    shifted_report = check_records(shifted_edge, cloud)
    assert shifted_report["command_timings"] == report["command_timings"]
    # An ACK alone must never pass the causal-loop acceptance check.
    no_perception = [r for r in cloud if r["event"] != "observation"]
    with pytest.raises(CheckFailed):
        check_records(edge, no_perception)
    duplicated = edge + [next(r for r in edge if r["event"] == "frame_sent")]
    with pytest.raises(CheckFailed, match="duplicate transmitted"):
        check_records(duplicated, cloud)
    # Correct-looking target metadata must not mask incorrect actual transmission timing.
    slow = copy.deepcopy(edge)
    for record in slow:
        if record["event"] == "frame_sent" and record["target_transmit_fps"] == 20:
            record["local_mono_ns"] *= 10
    with pytest.raises(CheckFailed, match="achieved"):
        check_records(slow, cloud)
    missing_ack = [r for r in cloud if r["event"] != "ack_received"]
    with pytest.raises(CheckFailed, match="never received ACK"):
        check_records(edge, missing_ack)

    no_history = [
        r
        for r in cloud
        if not (r["event"] == "observation" and r.get("evidence_kind") == "historical")
    ]
    with pytest.raises(CheckFailed, match="historical perception"):
        check_records(edge, no_history)
    no_completion = [r for r in cloud if r["event"] != "buffer_processed"]
    with pytest.raises(CheckFailed, match="never completed"):
        check_records(edge, no_completion)
    wrong_window = copy.deepcopy(cloud)
    next(r for r in wrong_window if r.get("type") == "REQUEST_RECENT_BUFFER")[
        "to_edge_mono_ns"
    ] += 1
    with pytest.raises(CheckFailed, match="anchored"):
        check_records(edge, wrong_window)

    # A second recognized recovery request must be checked, never silently ignored.
    recovery = copy.deepcopy(next(r for r in cloud if r.get("type") == "REQUEST_RECENT_BUFFER"))
    recovery.update(reason="recover_offline_evidence", command_id="missing-recovery-delivery")
    with pytest.raises(CheckFailed, match="ACK/completion"):
        check_records(edge, cloud + [recovery])
