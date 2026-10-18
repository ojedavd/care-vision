"""Process-only resource samples; CPU 100% means one fully occupied core."""

import asyncio
import os
import resource
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path


def snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = None
    if sys.platform.startswith("linux"):
        rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return {
        "mono": time.monotonic(),
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "rss_bytes": rss,
        "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
    }


def interval_metrics(previous, current):
    elapsed = current["mono"] - previous["mono"]
    if elapsed <= 0:
        raise ValueError("resource sample interval must be positive")
    return {
        "interval_seconds": elapsed,
        "cpu_seconds": current["cpu_seconds"],
        "cpu_percent_one_core": 100 * (current["cpu_seconds"] - previous["cpu_seconds"]) / elapsed,
        "rss_bytes": current["rss_bytes"],
        "peak_rss_bytes": current["peak_rss_bytes"],
        "logical_cpus": os.cpu_count(),
    }


@asynccontextmanager
async def monitor_resources(log, interval=1.0, heartbeat_interval=0.1):
    """Bound the sampler to service lifetime, including a final partial interval."""
    previous = snapshot()
    lag_max_ms = 0.0
    lag_samples = 0

    def sample():
        nonlocal previous, lag_max_ms, lag_samples
        current = snapshot()
        log.emit(
            "process_resources", **interval_metrics(previous, current),
            event_loop_lag_ms_max=lag_max_ms if lag_samples else None,
            event_loop_lag_samples=lag_samples,
            event_loop_probe_interval_seconds=heartbeat_interval,
        )
        previous = current
        lag_max_ms = 0.0
        lag_samples = 0

    async def heartbeat():
        nonlocal lag_max_ms, lag_samples
        while True:
            due = time.monotonic() + heartbeat_interval
            await asyncio.sleep(heartbeat_interval)
            lag_max_ms = max(lag_max_ms, max(0, time.monotonic() - due) * 1000)
            lag_samples += 1

    async def monitor():
        while True:
            await asyncio.sleep(interval)
            sample()

    task = asyncio.create_task(monitor())
    probe = asyncio.create_task(heartbeat())
    try:
        yield
    finally:
        task.cancel()
        probe.cancel()
        await asyncio.gather(task, probe, return_exceptions=True)
        sample()
