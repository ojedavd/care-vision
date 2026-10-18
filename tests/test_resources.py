import asyncio
import json
import time

import pytest

from care_vision.common.metrics import EventLog
from care_vision.common.resources import interval_metrics, monitor_resources


def test_cpu_is_interval_based_and_can_exceed_one_core():
    previous = {"mono": 100, "cpu_seconds": 20}
    current = {"mono": 102, "cpu_seconds": 23, "rss_bytes": 123, "peak_rss_bytes": 456}
    result = interval_metrics(previous, current)
    assert result["cpu_percent_one_core"] == 150
    assert result["rss_bytes"] == 123
    with pytest.raises(ValueError):
        interval_metrics(current, current)


def test_monitor_flushes_final_sample_on_failure(tmp_path):
    path = tmp_path / "resources.jsonl"
    log = EventLog("test", path)

    async def run():
        async with monitor_resources(log):
            raise RuntimeError("service failed")

    with pytest.raises(RuntimeError, match="service failed"):
        asyncio.run(run())
    log.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "process_resources"
    assert rows[0]["interval_seconds"] > 0
    assert rows[0]["peak_rss_bytes"] > 0


def test_monitor_distinguishes_an_event_loop_stall_from_regular_cpu_samples(tmp_path):
    path = tmp_path / "stall.jsonl"
    log = EventLog("test", path)

    async def run():
        async with monitor_resources(log, interval=1, heartbeat_interval=.01):
            await asyncio.sleep(.025)
            time.sleep(.06)  # Deliberate local stall, not network latency.
            await asyncio.sleep(.025)

    try:
        asyncio.run(run())
    finally:
        log.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]["event_loop_lag_samples"] >= 2
    assert rows[-1]["event_loop_lag_ms_max"] >= 40
    assert rows[-1]["event_loop_probe_interval_seconds"] == .01
