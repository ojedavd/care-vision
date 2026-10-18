"""Experimental sensing trigger, separate from a fall or identity decision.

Keep an immutable recent reference for a bounded visual search after mild posture
change. Missing observations never extend the reference or confirm recovery.
Production execution belongs in AWS; saved-observation replay needs no pixels.
"""

import math
from dataclasses import dataclass

from care_vision.cloud.accumulated_descent import AccumulatedDescent, DescentPolicy


@dataclass(frozen=True)
class VisibilityProbePolicy:
    # Development hypotheses: request more evidence sooner than incident logic.
    minimum_top_drop: float = 0.08
    minimum_center_drop: float = 0.02
    minimum_height_contraction: float = 0.08
    probe_seconds: float = 8.0
    upright_dwell: float = 2.0
    maximum_gap: float = 0.8

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("probe parameters must be finite and positive")


class VisibilityProbe:
    def __init__(self, *, policy=VisibilityProbePolicy()):
        self.policy = policy
        self.change = AccumulatedDescent(
            policy=DescentPolicy(
                minimum_top_drop=policy.minimum_top_drop,
                minimum_center_drop=policy.minimum_center_drop,
                minimum_height_contraction=policy.minimum_height_contraction,
            )
        )
        self.probes = []
        self.previous_changes = set()

    def update(self, t, frame_index, people):
        """Call only for fresh semantic observations, including empty detections.

        A new ID never supplies upright evidence for an earlier probe. Expiration
        bounds high-rate transmission; it does not resolve the possible incident.
        """
        boxes = {
            p["track_id"]: p["tracked_bbox"] for p in people if p["track_state"] == "confirmed"
        }
        signals = self.change.update(t, frame_index, boxes)
        current = {s["track_id"] for s in signals if s["descent_signal"]}
        started = []
        for s in signals:
            if (
                s["descent_signal"]
                and s["track_id"] not in self.previous_changes
                and not any(
                    p["source_track"] == s["track_id"] and p["state"] == "VERIFYING_VISIBILITY"
                    for p in self.probes
                )
            ):
                p = {
                    "source_track": s["track_id"],
                    "start": t,
                    "reference_time": s["reference_time"],
                    "reference_bbox": list(s["reference_bbox"]),
                    "expires_at": t + self.policy.probe_seconds,
                    "state": "VERIFYING_VISIBILITY",
                    "last_seen": None,
                    "upright_since": None,
                    "missing_observations": 0,
                    "ended_at": None,
                    "incident_confirmed": False,
                    "recovery_observed": False,
                    "reason": "mild_posture_change_requires_new_evidence",
                }
                self.probes.append(p)
                started.append(dict(p))
        self.previous_changes = current
        for p in self.probes:
            if p["state"] != "VERIFYING_VISIBILITY":
                continue
            if t >= p["expires_at"]:
                p.update(state="UNRESOLVED", ended_at=t, reason="visual_search_budget_expired")
                continue
            box = boxes.get(p["source_track"])
            if box is None:
                p["missing_observations"] += 1
                p["upright_since"] = p["last_seen"] = None
                continue
            _, y, w, h = box
            _, ry, _, rh = p["reference_bbox"]
            upright = w / h <= 0.8 and h >= 0.85 * rh and y <= ry + 0.15 * rh
            if p["last_seen"] is None or t - p["last_seen"] > self.policy.maximum_gap:
                p["upright_since"] = None
            p["last_seen"] = t
            p["upright_since"] = (
                (t if p["upright_since"] is None else p["upright_since"]) if upright else None
            )
            if (
                p["upright_since"] is not None
                and t - p["upright_since"] + 1e-9 >= self.policy.upright_dwell
            ):
                p.update(
                    state="POSTURE_CHANGE_CLEARED",
                    ended_at=t,
                    reason="same_track_sustained_reference_posture",
                )
        return {
            "time": t,
            "started": started,
            "probes": [dict(p) for p in self.probes],
            "request_more_evidence": any(p["state"] == "VERIFYING_VISIBILITY" for p in self.probes),
        }

    def finish(self):
        for p in self.probes:
            if p["state"] == "VERIFYING_VISIBILITY":
                p.update(
                    state="UNRESOLVED",
                    ended_at=self.change.last_time,
                    reason="end_of_saved_observations",
                )
        return [dict(p) for p in self.probes]
