"""Experimental request trigger when a previously confirmed person disappears.

This produces a request for more evidence, never an incident decision.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LossProbePolicy:
    minimum_missing_seconds: float = 1.0
    maximum_missing_seconds: float = 2.0
    cooldown_seconds: float = 8.0


class LossProbe:
    def __init__(self, policy: LossProbePolicy | None = None):
        self.policy = policy or LossProbePolicy()
        self.last_seen = None
        self.last_request_ns = None
        self.requested_track_id = None

    def observe(self, captured_ns: int, people: dict) -> dict | None:
        if people["person_detection_status"] != "ok" or people["person_tracking_status"] != "ok":
            return None
        confirmed = [p for p in people["persons"] if p["track_state"] == "confirmed"]
        previous_track_id = self.last_seen["track_id"] if self.last_seen else None
        matched = next(
            (p for p in confirmed if p["track_id"] == previous_track_id), None
        )
        if matched is not None or (self.last_seen is None and len(confirmed) == 1):
            person = matched if matched is not None else confirmed[0]
            self.last_seen = {
                "track_id": person["track_id"],
                "captured_mono_ns": captured_ns,
                "bbox": person["tracked_bbox"],
            }
            if self.requested_track_id == person["track_id"]:
                self.requested_track_id = None
            return None
        if self.last_seen is None:
            return None
        age = (captured_ns - self.last_seen["captured_mono_ns"]) / 1e9
        if age > self.policy.maximum_missing_seconds:
            self.last_seen = None
            if len(confirmed) == 1:
                self.observe(captured_ns, people)
            return None
        if age < self.policy.minimum_missing_seconds:
            return None
        if self.requested_track_id == self.last_seen["track_id"]:
            return None
        if (
            self.last_request_ns is not None
            and (captured_ns - self.last_request_ns) / 1e9 < self.policy.cooldown_seconds
        ):
            return None
        self.last_request_ns = captured_ns
        self.requested_track_id = self.last_seen["track_id"]
        return dict(self.last_seen)
