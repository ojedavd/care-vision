"""Experimental offline evidence ledger for a continuously visible body fragment.

Image-plane geometry is a development hypothesis, not calibrated floor geometry.
The fragment extractor must stop on discontinuity/ambiguous component merging.
This module never invents a hidden pose or emits caregiver alerts.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OcclusionPolicy:
    maximum_gap: float = 0.8
    low_dwell: float = 2.0
    upright_dwell: float = 2.0
    seed_window: float = 5.0
    rise_fraction: float = 0.2
    rise_memory: float = 5.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("policy values must be finite and positive")


class OcclusionEvidence:
    def __init__(
        self, track, start, reference_bbox, *, policy=OcclusionPolicy(), trigger="descent"
    ):
        if trigger not in {"descent", "visibility_uncertainty"}:
            raise ValueError("invalid verification trigger")
        if not isinstance(track, str) or not track or not math.isfinite(start):
            raise ValueError("invalid track or timestamp")
        if (
            len(reference_bbox) != 4
            or any(not math.isfinite(v) for v in reference_bbox)
            or reference_bbox[2] <= 0
            or reference_bbox[3] <= 0
        ):
            raise ValueError("invalid reference box")
        self.track = track
        self.start = start
        self.reference_bbox = tuple(reference_bbox)
        self.policy = policy
        self.trigger = trigger
        self.state = "VERIFYING_DESCENT" if trigger == "descent" else "VERIFYING_VISIBILITY"
        self.low_since = self.last_seen = self.last_time = None
        self.candidate_at = None
        self.bound_track = None
        self.proposed_track = self.proposed_at = self.reappearance_box = None
        self.regional_rise_reference = self.regional_reference_at = None
        self.last_person = self.upright_since = None
        self.rise_seen = False
        self.rise_at = None
        self.original_upright_since = self.original_seen = None
        self.events = []

    @property
    def terminal(self):
        return self.state in {
            "RECOVERY_OBSERVED",
            "UNRESOLVED",
            "NO_LOW_EVIDENCE",
            "POSTURE_CHANGE_CLEARED",
        }

    def transition(self, t, state, reason):
        self.state = state
        self.events.append({"time": t, "state": state, "reason": reason})

    def update(self, t, *, fragment_box=None, matches=(), people=None, ambiguous=False):
        if (
            not math.isfinite(t)
            or t < self.start
            or (self.last_time is not None and t <= self.last_time)
        ):
            raise ValueError("timestamps must strictly increase")
        self.last_time = t
        if self.terminal:
            return self.snapshot()
        if ambiguous:
            self.transition(t, "UNRESOLVED", "ambiguous_fragment_or_identity")
            return self.snapshot()
        if self.candidate_at is None and people is not None:
            original = next(
                (p for p in people if p["track"] == self.track and p["state"] == "confirmed"), None
            )
            if original is not None:
                _, y, w, h = original["bbox"]
                _, ry, _, rh = self.reference_bbox
                upright = w / h <= 0.8 and h >= 0.85 * rh and y <= ry + 0.15 * rh
                if self.original_seen is None or t - self.original_seen > self.policy.maximum_gap:
                    self.original_upright_since = None
                self.original_upright_since = (
                    (t if self.original_upright_since is None else self.original_upright_since)
                    if upright
                    else None
                )
                self.original_seen = t
                if (
                    self.original_upright_since is not None
                    and t - self.original_upright_since + 1e-9 >= self.policy.upright_dwell
                ):
                    self.transition(
                        t,
                        "POSTURE_CHANGE_CLEARED",
                        "original_person_remains_upright_without_low_candidate",
                    )
                    return self.snapshot()
            else:
                self.original_upright_since = self.original_seen = None
        if self.bound_track is not None:
            if people is not None:
                person = next(
                    (
                        p
                        for p in people
                        if p["track"] == self.bound_track and p["state"] == "confirmed"
                    ),
                    None,
                )
                if person is None or (
                    self.last_person is not None and t - self.last_person > self.policy.maximum_gap
                ):
                    self.transition(t, "UNRESOLVED", "reassociated_person_lost")
                else:
                    self._recovery(t, person["bbox"])
            elif self.last_person is not None and t - self.last_person > self.policy.maximum_gap:
                self.transition(t, "UNRESOLVED", "semantic_observation_timeout")
            return self.snapshot()
        if self.last_seen is not None and t - self.last_seen > self.policy.maximum_gap:
            self.transition(t, "UNRESOLVED", "visible_fragment_continuity_lost")
            return self.snapshot()
        if self.last_seen is None and t - self.start > self.policy.seed_window:
            self.transition(t, "NO_LOW_EVIDENCE", "descent_without_visible_low_fragment")
            return self.snapshot()
        if fragment_box is None:
            self.low_since = None
            return self.snapshot()
        self.last_seen = t
        _, ry, _, rh = self.reference_bbox
        _, y, w, h = fragment_box
        # A visible fragment can lie above the earlier foot line in perspective.
        # This opt-in trigger is a posture candidate, not calibrated floor contact.
        bottom_fraction = 0.85 if self.trigger == "visibility_uncertainty" else 1.05
        low = w / h >= 1.5 and y >= ry + 0.65 * rh and y + h >= ry + bottom_fraction * rh
        self.low_since = (t if self.low_since is None else self.low_since) if low else None
        if (
            self.candidate_at is None
            and self.low_since is not None
            and t - self.low_since + 1e-9 >= self.policy.low_dwell
        ):
            self.candidate_at = t
            self.transition(
                t,
                "OCCLUDED_INCIDENT_CANDIDATE",
                (
                    "descent_then_continuous_low_fragment"
                    if self.trigger == "descent"
                    else "uncertain_posture_then_continuous_low_fragment"
                ),
            )
        # Only a unique repeated semantic match can bridge detector IDs.
        if people is not None and self.candidate_at is not None:
            if len(matches) > 1:
                self.transition(t, "UNRESOLVED", "multiple_people_match_fragment")
            elif len(matches) == 1:
                match = matches[0]
                # A detector box covering only a low fragment is not a returning
                # full person. Keep pixel continuity until the visible extent rises.
                _, match_y, _, match_h = match["bbox"]
                if match_h < 0.6 * rh or match_y > ry + 0.65 * rh:
                    self.proposed_track = self.proposed_at = self.reappearance_box = None
                    return self.snapshot()
                if (
                    self.proposed_track == match["track"]
                    and self.proposed_at is not None
                    and t - self.proposed_at <= self.policy.maximum_gap
                    and match["state"] == "confirmed"
                ):
                    if self.trigger == "visibility_uncertainty":
                        # Keep pixel continuity until a full upright person is
                        # corroborated. Partial reappearance may lose its detector ID.
                        # Two unique matches corroborate a regional box, not an ID bridge.
                        if (
                            self.regional_reference_at is not None
                            and t - self.regional_reference_at > self.policy.rise_memory
                            and not self.rise_seen
                        ):
                            self.regional_rise_reference = self.regional_reference_at = None
                        if self.regional_rise_reference is None:
                            self.regional_rise_reference = self.reappearance_box
                            self.regional_reference_at = t
                        _, r_y, _, r_h = self.regional_rise_reference
                        _, m_y, m_w, m_h = match["bbox"]
                        if (
                            r_y - m_y >= self.policy.rise_fraction * rh
                            and m_h - r_h >= self.policy.rise_fraction * rh
                        ):
                            self.rise_seen = True
                            if self.rise_at is None:
                                self.rise_at = t
                        upright = m_w / m_h <= 0.8 and m_h >= 0.85 * rh
                        if not upright:
                            self.proposed_at = t
                            return self.snapshot()
                    self.bound_track = match["track"]
                    self.transition(t, "PERSON_REASSOCIATED", "repeated_unique_fragment_overlap")
                    self._recovery(t, match["bbox"])
                else:
                    self.proposed_track = match["track"]
                    self.reappearance_box = tuple(match["bbox"])
                self.proposed_at = t
            else:
                self.proposed_track = self.proposed_at = self.reappearance_box = None
        return self.snapshot()

    def _recovery(self, t, box):
        self.last_person = t
        _, y0, _, h0 = self.reappearance_box
        _, y, w, h = box
        rh = self.reference_bbox[3]
        if y0 - y >= self.policy.rise_fraction * rh and h - h0 >= self.policy.rise_fraction * rh:
            self.rise_seen = True
            if self.rise_at is None:
                self.rise_at = t
        if self.rise_at is not None and t - self.rise_at > self.policy.rise_memory:
            self.transition(t, "UNRESOLVED", "rise_evidence_expired_without_sustained_upright")
            return
        upright = w / h <= 0.8 and h >= 0.85 * rh
        self.upright_since = (
            (t if self.upright_since is None else self.upright_since) if upright else None
        )
        if (
            self.rise_seen
            and self.upright_since is not None
            and t - self.upright_since + 1e-9 >= self.policy.upright_dwell
        ):
            self.transition(t, "RECOVERY_OBSERVED", "reassociated_person_rises_and_remains_upright")

    def finish(self):
        if not self.terminal:
            self.transition(
                self.last_time if self.last_time is not None else self.start,
                "UNRESOLVED",
                "end_of_evidence",
            )
        return self.snapshot()

    def snapshot(self):
        return {
            "source_track": self.track,
            "state": self.state,
            "candidate_at": self.candidate_at,
            "bound_track": self.bound_track,
            "rise_seen": self.rise_seen,
            "rise_at": self.rise_at,
            "upright_since": self.upright_since,
            "events": list(self.events),
            "hidden_pose_observed": False,
            "alerts_enabled": False,
            "verification_trigger": self.trigger,
            "regional_rise_reference": self.regional_rise_reference,
        }
