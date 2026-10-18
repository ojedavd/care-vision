"""AWS OpenCV foreground evidence for offline chair-occlusion experiments.

No floor-plane calibration, pose reconstruction, or production alert decision.
Call with dense full-frame pixels and sparse, same-camera person detections.
"""

import math

import cv2
import numpy as np

from care_vision.cloud.accumulated_descent import AccumulatedDescent
from care_vision.cloud.occlusion_evidence import OcclusionEvidence
from care_vision.cloud.visibility_probe import VisibilityProbe


class OcclusionFragments:
    def __init__(self, *, visibility_probes=False):
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=25, detectShadows=False
        )
        self.descent = AccumulatedDescent()
        self.previous_t = None
        self.groups = []
        self.previous_descents = set()
        self.visibility = VisibilityProbe() if visibility_probes else None
        self.visibility_state = None

    def update(self, image, t, frame_index, detections=None, *, visible_roi=None):
        if self.previous_t is not None and t <= self.previous_t:
            raise ValueError("timestamps must strictly increase")
        height, width = image.shape[:2]
        scale = np.array([640 / width, 360 / height, 640 / width, 360 / height])
        small = cv2.resize(image, (640, 360))
        active = any(not g["ledger"].terminal for g in self.groups)
        dt = 0 if self.previous_t is None else t - self.previous_t
        learning = 1 if self.previous_t is None else 1 - math.exp(-dt / 10)
        foreground = self.background.apply(small, learningRate=0 if active else learning)
        # Pixels outside a transmitted ROI are unavailable, never fresh evidence.
        visibility = np.ones((360, 640), np.uint8) * 255
        if visible_roi is not None:
            x, y, w, h = visible_roi
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise ValueError("ROI must be inside the full-frame coordinate canvas")
            visibility[:] = 0
            x0, y0 = math.ceil(x * 640 / width), math.ceil(y * 360 / height)
            x1, y1 = math.floor((x + w) * 640 / width), math.floor((y + h) * 360 / height)
            visibility[y0:y1, x0:x1] = 255
        foreground &= visibility
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        connected = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        connected &= visibility
        _, labels, stats, _ = cv2.connectedComponentsWithStats(connected)
        people = (
            None
            if detections is None
            else [
                {
                    "track": p["track_id"],
                    "state": p["track_state"],
                    "bbox": (np.array(p["bbox"]) * scale).tolist(),
                }
                for p in detections
            ]
        )
        if detections is not None:
            signals = self.descent.update(
                t,
                frame_index,
                {
                    p["track_id"]: p["tracked_bbox"]
                    for p in detections
                    if p["track_state"] == "confirmed"
                },
            )
            current_descents = {s["track_id"] for s in signals if s["descent_signal"]}
            for signal in signals:
                track = signal["track_id"]
                if (
                    signal["descent_signal"]
                    and track not in self.previous_descents
                    and not any(
                        not g["ledger"].terminal
                        and track in {g["ledger"].track, g["ledger"].bound_track}
                        for g in self.groups
                    )
                ):
                    reference = np.array(signal["reference_bbox"]) * scale
                    self.groups.append(
                        {"ledger": OcclusionEvidence(track, t, reference), "mask": None}
                    )
            self.previous_descents = current_descents
            if self.visibility is not None:
                self.visibility_state = self.visibility.update(t, frame_index, detections)
                for probe in self.visibility_state["started"]:
                    track = probe["source_track"]
                    if not any(
                        not g["ledger"].terminal
                        and track in {g["ledger"].track, g["ledger"].bound_track}
                        for g in self.groups
                    ):
                        self.groups.append(
                            {
                                "ledger": OcclusionEvidence(
                                    track,
                                    t,
                                    np.array(probe["reference_bbox"]) * scale,
                                    trigger="visibility_uncertainty",
                                ),
                                "mask": None,
                            }
                        )
        states = []
        for group in self.groups:
            ledger = group["ledger"]
            fragment_box, matches = None, []
            ambiguous = False
            # End pixel association permanently once a semantic identity has been bound.
            if not ledger.terminal and ledger.bound_track is None:
                x, y, w, h = ledger.reference_bbox
                search_region = np.zeros((360, 640), np.uint8)
                sx0 = max(0, round(x + w / 2 - 0.75 * h))
                sx1 = min(640, round(x + w / 2 + 0.75 * h))
                sy0 = max(0, math.ceil(y + 0.65 * h))
                sy1 = min(360, round(y + 1.6 * h))
                search_region[sy0:sy1, sx0:sx1] = 255
                if (
                    ledger.last_seen is not None
                    and t - ledger.last_seen > ledger.policy.maximum_gap
                ):
                    group["mask"] = (
                        None  # Ledger closes the episode; never reseed an expired identity.
                    )
                elif group["mask"] is None:
                    region = np.zeros((360, 640), np.uint8)
                    x0, x1 = (
                        max(0, round(x + w / 2 - 0.75 * h)),
                        min(640, round(x + w / 2 + 0.75 * h)),
                    )
                    lower_fraction = 0.65 if ledger.trigger == "visibility_uncertainty" else 1.05
                    y0, y1 = (
                        max(0, math.ceil(y + lower_fraction * h)),
                        min(360, round(y + 1.6 * h)),
                    )
                    region[y0:y1, x0:x1] = 255
                    lower = foreground & region
                    ys, xs = np.where(lower > 0)
                    if (
                        len(xs) >= 0.015 * w * h
                        and len(xs)
                        and xs.max() - xs.min() + 1 >= 0.25 * h
                        and (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) >= 1.5
                    ):
                        selected = set(labels[lower > 0]) - {0}
                        selected = {
                            n for n in selected if np.sum((labels == n) & (lower > 0)) >= 10
                        }
                        mask = np.isin(labels, list(selected)).astype(np.uint8) * 255
                        if ledger.trigger == "visibility_uncertainty":
                            mask &= search_region
                        if mask.any():
                            group["mask"] = mask
                else:
                    radius = min(21, max(3, round(h * dt)))
                    predicted = cv2.dilate(
                        group["mask"], np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
                    )
                    selected = set(labels[predicted > 0]) - {0}
                    selected = {
                        n
                        for n in selected
                        if np.sum((labels == n) & (predicted > 0)) >= 10
                        and stats[n, cv2.CC_STAT_AREA] >= 20
                    }
                    mask = np.isin(labels, list(selected)).astype(np.uint8) * 255
                    if ledger.trigger == "visibility_uncertainty" and ledger.candidate_at is None:
                        mask &= search_region
                    previous_area = np.count_nonzero(group["mask"])
                    current_area = np.count_nonzero(mask)
                    ambiguous = current_area > max(previous_area * 4, w * h)
                    group["mask"] = mask if current_area else None
                    # A missing fragment is not eligible for a new seed.
                    if not current_area:
                        ambiguous = True
                if group["mask"] is not None:
                    fy, fx = np.where(group["mask"] > 0)
                    fragment_box = [
                        int(fx.min()),
                        int(fy.min()),
                        int(fx.max() - fx.min() + 1),
                        int(fy.max() - fy.min() + 1),
                    ]
                    if people is not None:
                        for person in people:
                            bx, by, bw, bh = person["bbox"]
                            fraction = np.mean(
                                (fx >= bx) & (fx <= bx + bw) & (fy >= by) & (fy <= by + bh)
                            )
                            if fraction >= 0.6:
                                matches.append({**person, "fraction": float(fraction)})
            state = ledger.update(
                t, fragment_box=fragment_box, matches=matches, people=people, ambiguous=ambiguous
            )
            if ledger.terminal or ledger.bound_track is not None:
                group["mask"] = None
            states.append(
                {
                    **state,
                    "start": ledger.start,
                    "reference_bbox": list(ledger.reference_bbox),
                    "fragment_box": fragment_box,
                    "matches": matches,
                }
            )
        self.previous_t = t
        return (
            {
                "time": t,
                "frame_index": frame_index,
                "episodes": states,
                "visibility_probe": self.visibility_state,
            },
            small,
            foreground,
        )
