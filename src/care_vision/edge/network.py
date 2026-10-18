"""Edge-side connectivity signal; never infer cloud vision from local capture."""

import math
import statistics
import time


class NetworkMonitor:
    """Classify observed WebSocket pong RTT with bounded hysteresis."""

    degraded_rtt_ms = 250.0
    severe_rtt_ms = 1000.0
    recovered_rtt_ms = 150.0
    consecutive_samples = 2
    baseline_samples = 5

    def __init__(self):
        self.quality = "OFFLINE"
        self.rtt_ms = None
        self.high = self.low = 0
        self.measured_degradation = False
        self.offline_since = time.monotonic()
        self.offline_capture_count = 0
        self.last_outage = {"offline_seconds": 0.0, "captured_while_offline": 0}
        self.baseline_rtts = []

    @property
    def recovery_threshold_ms(self):
        # Learn only before measured degradation; never normalize a sustained
        # bad connection. Preserve at least 20% hysteresis below the high limit.
        if len(self.baseline_rtts) < self.baseline_samples:
            return self.recovered_rtt_ms
        return min(
            0.8 * self.degraded_rtt_ms,
            max(self.recovered_rtt_ms, 1.2 * statistics.median(self.baseline_rtts)),
        )

    def connected(self, captured, now=None):
        now = time.monotonic() if now is None else now
        duration = max(0.0, now - self.offline_since) if self.offline_since is not None else 0.0
        missed = max(0, captured - self.offline_capture_count)
        # Connectivity is established, but RTT has not yet been measured.
        self.quality = "DEGRADED"
        self.rtt_ms = None
        self.high = self.low = 0
        self.measured_degradation = False
        self.baseline_rtts = []
        self.offline_since = None
        self.last_outage = {"offline_seconds": duration, "captured_while_offline": missed}
        return dict(self.last_outage)

    def outage(self, captured, now=None):
        if self.offline_since is None:
            return dict(self.last_outage)
        now = time.monotonic() if now is None else now
        return {
            "offline_seconds": max(0.0, now - self.offline_since),
            "captured_while_offline": max(0, captured - self.offline_capture_count),
        }

    def observed_rtt(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid round-trip time")
        if self.quality == "OFFLINE":
            raise RuntimeError("pong cannot arrive while offline")
        self.rtt_ms = seconds * 1000
        if (
            not self.measured_degradation
            and self.rtt_ms < self.degraded_rtt_ms
            and len(self.baseline_rtts) < self.baseline_samples
        ):
            self.baseline_rtts.append(self.rtt_ms)
        if self.rtt_ms >= self.degraded_rtt_ms:
            self.high += 1
            self.low = 0
            if self.high >= self.consecutive_samples or self.rtt_ms >= self.severe_rtt_ms:
                self.quality = "DEGRADED"
                self.measured_degradation = True
        elif self.rtt_ms <= self.recovery_threshold_ms:
            self.low += 1
            self.high = 0
            if self.low >= self.consecutive_samples:
                self.quality = "GOOD"
                self.measured_degradation = False
        else:
            self.high = self.low = 0
        return self.quality

    @property
    def transmission_cap_fps(self):
        return 5 if self.measured_degradation and self.quality == "DEGRADED" else None

    def disconnected(self, captured, now=None):
        now = time.monotonic() if now is None else now
        if self.offline_since is None:
            self.offline_since = now
            self.offline_capture_count = captured
        self.quality = "OFFLINE"
        self.rtt_ms = None
        self.high = self.low = 0
        self.measured_degradation = False
