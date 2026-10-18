"""Acceptance regression against retained measured transport logs."""

from pathlib import Path

import pytest

from care_vision.check_loop import CheckFailed, check_records, read_log


def test_five_fps_profile_still_rejects_wrong_rates_and_missing_perception():
    folder = (
        Path(__file__).resolve().parents[1]
        / "artifacts/tapo-recorded-60-20261004-235727/sitting-60"
    )
    if not folder.exists():
        pytest.skip("retained experimental logs unavailable")
    edge, cloud = read_log(folder / "edge.jsonl"), read_log(folder / "cloud.jsonl")
    assert check_records(edge, cloud, escalation_fps=5, capture_fps=25)["passed"]
    with pytest.raises(CheckFailed, match="expected exactly"):
        check_records(edge, cloud)
    with pytest.raises(CheckFailed, match="valid observation"):
        check_records(
            edge,
            [r for r in cloud if r["event"] != "observation"],
            escalation_fps=5,
            capture_fps=25,
        )
