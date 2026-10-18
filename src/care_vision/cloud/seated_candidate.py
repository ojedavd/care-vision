"""Offline shadow evidence for a seated-like posture change, not a safe closure.

A manually reviewed seat occupancy region and bbox history are only proxies.
There is no pose inference, furniture detection, command, alert or live hook.
Use one instance per recorded episode and full-frame stream.
"""

import math
from collections import deque
from dataclasses import dataclass


def checked_box(box):
    if len(box) != 4 or any(not math.isfinite(v) for v in box):
        raise ValueError("box must have four finite values")
    x, y, w, h = box
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
        raise ValueError("box must be normalized and inside the full frame")
    return tuple(box)


@dataclass(frozen=True)
class SeatContext:
    edge_id: str
    source_width: int
    source_height: int
    occupancy_region: tuple[float, float, float, float]
    provenance: str

    def __post_init__(self):
        if not self.edge_id or not self.provenance:
            raise ValueError("context requires camera identity and provenance")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("source dimensions must be positive")
        object.__setattr__(self, "occupancy_region", checked_box(self.occupancy_region))


@dataclass(frozen=True)
class SeatedPolicy:
    stable_seconds: float = 2.0
    maximum_gap_seconds: float = 0.8
    minimum_height_ratio: float = 0.45
    maximum_height_ratio: float = 0.85
    minimum_top_drop: float = 0.15
    maximum_aspect: float = 0.8
    horizontal_aspect: float = 1.2
    maximum_box_variation: float = 0.08

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("thresholds must be finite and positive")
        if not self.minimum_height_ratio < self.maximum_height_ratio < 1:
            raise ValueError("height ratios must describe contraction")
        if self.maximum_aspect >= self.horizontal_aspect:
            raise ValueError("seated and horizontal gates must not overlap")
        if self.maximum_gap_seconds >= self.stable_seconds:
            raise ValueError("stability requires multiple contiguous observations")


class SeatedCandidate:
    def __init__(self, episode, origin, context, *, policy=SeatedPolicy()):
        self.context, self.policy = context, policy
        self.episode_id, self.track_id = episode["episode_id"], episode["track_id"]
        self.reference = checked_box(episode["reference_bbox"])
        self.identity = tuple(origin[k] for k in ("edge_id", "boot_id", "session_id"))
        self.started_ns = episode["started_ns"]
        self.blocked = bool(episode["horizontal_observed"] or episode["test_alert_emitted"])
        self.samples = deque()
        self.last_ns = self.last_seq = None

    def update(self, observation):
        now, seq = observation["captured_mono_ns"], observation["capture_seq"]
        if self.last_ns is not None and (now <= self.last_ns or seq <= self.last_seq):
            raise ValueError("shadow input must have increasing capture times and sequences")
        gap = self.last_ns is not None and (
            now - self.last_ns > self.policy.maximum_gap_seconds * 1e9
        )
        self.last_ns, self.last_seq = now, seq
        result = {
            "episode_id": self.episode_id,
            "observation_id": observation["observation_id"],
            "captured_mono_ns": now,
            "seated_like_candidate": False,
            "would_close_episode": False,
            "mode": "offline_shadow",
            "stable_seconds": 0.0,
        }

        def reject(reason):
            self.samples.clear()
            return {**result, "reason": reason}

        c = self.context
        if (
            tuple(observation.get(k) for k in ("edge_id", "boot_id", "session_id"))
            != self.identity
            or observation.get("edge_id") != c.edge_id
            or (observation.get("source_width"), observation.get("source_height"))
            != (c.source_width, c.source_height)
            or (observation.get("width"), observation.get("height"))
            != (c.source_width, c.source_height)
            or observation.get("view_kind") != "full_frame"
        ):
            return reject("context_or_stream_mismatch")
        if (
            observation.get("valid") is not True
            or observation.get("evidence_kind") != "live"
            or observation.get("person_detection_status") != "ok"
            or observation.get("person_tracking_status") != "ok"
            or now < self.started_ns
        ):
            return reject("evidence_unavailable")
        people = observation.get("persons", [])
        if len(people) != 1:
            return reject("missing_or_ambiguous_person")
        person = people[0]
        if person.get("track_id") != self.track_id or person.get("track_state") != "confirmed":
            return reject("track_continuity_unavailable")
        try:
            box = checked_box(tuple(
                value / (c.source_width if i % 2 == 0 else c.source_height)
                for i, value in enumerate(person["tracked_bbox"])
            ))
        except (KeyError, TypeError, ValueError):
            return reject("invalid_bbox")
        x, y, w, h = box
        aspect = w * c.source_width / (h * c.source_height)
        # Horizontal evidence blocks this benign branch for the entire episode,
        # including association-only observations. Recovery needs a separate rule.
        if aspect >= self.policy.horizontal_aspect:
            self.blocked = True
        if self.blocked:
            return reject("prior_horizontal_or_alert_requires_incident_followup")
        if person.get("association_only", False):
            return reject("weak_association_cannot_establish_benign_posture")
        _, ry, rw, rh = self.reference
        if rw * c.source_width / (rh * c.source_height) > self.policy.maximum_aspect:
            return reject("reference_not_upright_like")
        rx, sy, sw, sh = c.occupancy_region
        # Bbox midpoint is an occupancy proxy, not a measured hip or seat contact.
        in_region = rx <= x + w / 2 <= rx + sw and sy <= y + h / 2 <= sy + sh
        if not in_region:
            return reject("outside_seat_occupancy_region")
        if not (
            self.policy.minimum_height_ratio <= h / rh <= self.policy.maximum_height_ratio
            and (y - ry) / rh >= self.policy.minimum_top_drop
            and aspect <= self.policy.maximum_aspect
        ):
            return reject("posture_change_not_seated_like")
        if gap:
            self.samples.clear()
        self.samples.append((now, box))
        # Keep the sample immediately before the window boundary so duration is
        # measured by timestamps, never by assuming a fixed number of frames.
        cutoff = now - self.policy.stable_seconds * 1e9
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        variation = max(
            (max(b[i] for _, b in self.samples) - min(b[i] for _, b in self.samples))
            * (c.source_width / c.source_height if i % 2 == 0 else 1) / rh
            for i in range(4)
        )
        if variation > self.policy.maximum_box_variation:
            self.samples.clear()
            self.samples.append((now, box))
        seconds = (now - self.samples[0][0]) / 1e9
        candidate = seconds >= self.policy.stable_seconds
        return {
            **result,
            "seated_like_candidate": candidate,
            "stable_seconds": seconds,
            "reason": "seated_like_requires_review" if candidate else "awaiting_stable_evidence",
        }
