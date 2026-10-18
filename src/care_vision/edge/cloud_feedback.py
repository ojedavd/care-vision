"""Session-scoped cloud evidence accounting, independent of link RTT."""

import time

from care_vision.common.contracts import ProtocolError, validate

COUNTERS = ("live_received", "live_processed", "live_dropped", "live_stale")
FEEDBACK_TTL_SECONDS = 3.0  # Experimental display freshness, not an alert/FPS threshold.


class CloudFeedback:
    def __init__(self, identity):
        self.identity = dict(identity)
        self.latest = None
        self.received_at = None

    def accept(self, message, now=None):
        validate(message)
        if message["type"] != "CLOUD_STATUS" or any(
            message.get(k) != v for k, v in self.identity.items()
        ):
            raise ProtocolError("cloud status belongs to another session")
        if self.latest and (
            message["status_seq"] <= self.latest["status_seq"]
            or any(message[k] < self.latest[k] for k in COUNTERS)
        ):
            raise ProtocolError("cloud status sequence or counters regressed")
        self.latest = dict(message)
        self.received_at = time.monotonic() if now is None else now

    def fields(self, network_quality, now=None):
        now = time.monotonic() if now is None else now
        age = max(0, now - self.received_at) if self.received_at is not None else None
        freshness = ("OFFLINE" if network_quality == "OFFLINE" else
                     "UNKNOWN" if age is None else
                     "STALE" if age > FEEDBACK_TTL_SECONDS else "FRESH")
        latest = self.latest or {}
        incomplete = bool(latest.get("live_dropped", 0) or latest.get("live_stale", 0))
        quality = ("OFFLINE" if network_quality == "OFFLINE" else
                   "DEGRADED" if incomplete or network_quality == "DEGRADED" else
                   "UNKNOWN" if freshness != "FRESH" or not latest.get("live_processed") else
                   "GOOD")
        return {
            "pipeline_quality": quality,
            "cloud_feedback_state": freshness,
            "cloud_feedback_age_ms": age * 1000 if age is not None else None,
            "cloud_status_seq": latest.get("status_seq"),
            **{"cloud_" + k: latest.get(k) for k in (*COUNTERS, "queue_depth", "queue_capacity")},
        }
