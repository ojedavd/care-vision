"""Resource units and partial sampling windows must not invent CPU savings."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "sensing_audit", Path(__file__).resolve().parents[1] / "tools/audit_mvp_sensing_comparison.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_cpu_is_core_seconds_and_partial_intervals_are_proportional():
    records = [
        {"local_mono_ns": 1_000_000_000, "interval_seconds": 1,
         "cpu_percent_one_core": 100, "rss_bytes": 1000},
        {"local_mono_ns": 2_000_000_000, "interval_seconds": 1,
         "cpu_percent_one_core": 200, "rss_bytes": 2000},
        {"local_mono_ns": 3_000_000_000, "interval_seconds": 1,
         "cpu_percent_one_core": 900, "rss_bytes": 9000},
    ]
    result = audit.resource_window(records, 500_000_000, 2_000_000_000)
    assert result["cpu_seconds_estimate"] == 2.5
    assert result["sample_coverage_seconds"] == result["window_seconds"] == 1.5
    assert result["mean_cpu_percent_one_core_estimate"] == pytest.approx(166.6666667)
    assert result["rss_bytes"]["max"] == 2000


def test_missing_samples_are_visible_as_incomplete_coverage():
    result = audit.resource_window([], 0, 1_000_000_000)
    assert result["sample_coverage_seconds"] == 0
    assert result["rss_bytes"] is None
