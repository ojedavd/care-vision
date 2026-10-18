"""Deterministic temporal association for normalized person detections."""

import math
from dataclasses import dataclass

from care_vision.evaluation import intersection_over_union


@dataclass
class _Track:
    number: int
    bbox: tuple[float, float, float, float]
    first_seen_ns: int
    last_seen_ns: int
    hits: int = 1
    misses: int = 0
    visible: bool = True


class PersonTracker:
    """Associate detections by IoU without inferring a person's identity."""

    def __init__(
        self,
        iou_threshold=0.2,
        confirmation_hits=2,
        max_gap_seconds=2.0,
        alpha=0.65,
        reacquire_seconds=0.25,
        association_confidence=None,
        new_track_confidence=0.35,
    ):
        if not 0 < iou_threshold <= 1:
            raise ValueError("tracking IoU threshold must be in (0, 1]")
        if type(confirmation_hits) is not int or confirmation_hits < 1:
            raise ValueError("tracking confirmation_hits must be a positive integer")
        if max_gap_seconds <= 0 or not 0 < alpha <= 1 or reacquire_seconds <= 0:
            raise ValueError("tracking gap and smoothing parameters are invalid")
        if association_confidence is not None and not (
            0 < association_confidence < new_track_confidence <= 1
        ):
            raise ValueError("association confidence must be below new-track confidence")
        self.association_confidence = association_confidence
        self.new_track_confidence = new_track_confidence
        self.iou_threshold = iou_threshold
        self.confirmation_hits = confirmation_hits
        self.max_gap_ns = round(max_gap_seconds * 1_000_000_000)
        self.alpha = alpha
        self.reacquire_ns = round(reacquire_seconds * 1_000_000_000)
        self.namespace = None
        self.tracks = {}
        self.next_number = 1
        self.last_update_ns = None

    def reset(self, namespace=None):
        self.namespace = namespace
        self.tracks = {}
        self.next_number = 1
        self.last_update_ns = None

    def _track_id(self, track):
        return f"{self.namespace}:track:{track.number}"

    def _state(self, track):
        if track.hits < self.confirmation_hits:
            return "tentative"
        return "confirmed" if track.visible else "lost"

    @staticmethod
    def _rounded_bbox(track):
        return [round(value) for value in track.bbox]

    def _association(self, track, bbox, captured_ns):
        overlap = intersection_over_union(track.bbox, bbox)
        tx, ty, tw, th = track.bbox
        x, y, width, height = bbox
        center_distance = math.hypot(
            (tx + tw / 2) - (x + width / 2),
            (ty + th / 2) - (y + height / 2),
        ) / max(tw, th, width, height)
        track_area = tw * th
        detection_area = width * height
        size_ratio = min(track_area, detection_area) / max(track_area, detection_area)
        geometrically_close = center_distance <= 0.75 and size_ratio >= 0.15
        recently_confirmed = (
            track.hits >= self.confirmation_hits
            and captured_ns - track.last_seen_ns <= self.reacquire_ns
        )
        short_gap_reacquisition = (
            recently_confirmed and center_distance <= 1.5 and size_ratio >= 0.05
        )
        valid = overlap >= self.iou_threshold or geometrically_close or short_gap_reacquisition
        score = overlap + 0.25 * max(0, 1 - center_distance) + 0.05 * size_ratio
        return valid, score, overlap

    def update(self, person_observation: dict, captured_ns: int, namespace: str) -> dict:
        result = dict(person_observation)
        if self.namespace != namespace:
            self.reset(namespace)
        detection_status = result["person_detection_status"]
        if detection_status != "ok":
            snapshot_ns = max(captured_ns, self.last_update_ns or captured_ns)
            result.update(
                person_tracking_status=detection_status,
                active_track_count=len(self.tracks),
                tracks=self._active_tracks(snapshot_ns),
                confirmed_person_detected=False,
                confirmed_person_count=0,
                primary_track_id=None,
                primary_tracked_bbox=None,
            )
            return result
        if self.last_update_ns is not None and captured_ns <= self.last_update_ns:
            raise ValueError("tracking capture time must increase")
        self.last_update_ns = captured_ns

        self.tracks = {
            number: track
            for number, track in self.tracks.items()
            if captured_ns - track.last_seen_ns <= self.max_gap_ns
        }
        for track in self.tracks.values():
            track.visible = False
            track.misses += 1

        persons = [dict(person) for person in result["persons"]]
        weak_indices = set()
        if self.association_confidence is not None:
            persons = [p for p in persons if p["confidence"] >= self.association_confidence]
            weak_indices = {
                i for i, p in enumerate(persons) if p["confidence"] < self.new_track_confidence
            }
        candidates = []
        for number, track in self.tracks.items():
            for index, person in enumerate(persons):
                valid, score, overlap = self._association(track, person["bbox"], captured_ns)
                if index in weak_indices:
                    # Weak evidence only sustains a unique confirmed track with overlap.
                    valid = valid and track.hits >= self.confirmation_hits and overlap >= 0.3
                if valid:
                    candidates.append((index not in weak_indices, score, overlap, number, index))
        ambiguous = {index for index in weak_indices if sum(c[-1] == index for c in candidates) > 1}
        candidates = [c for c in candidates if c[-1] not in ambiguous]
        candidates.sort(reverse=True)
        matched_tracks = set()
        matched_detections = set()
        assignments = {}
        for _, _, overlap, number, index in candidates:
            if number in matched_tracks or index in matched_detections:
                continue
            matched_tracks.add(number)
            matched_detections.add(index)
            assignments[index] = (number, overlap)

        for index, person in enumerate(persons):
            assignment = assignments.get(index)
            new_track = assignment is None
            if new_track and index in weak_indices:
                continue
            if new_track:
                number = self.next_number
                self.next_number += 1
                bbox = tuple(float(value) for value in person["bbox"])
                self.tracks[number] = _Track(number, bbox, captured_ns, captured_ns)
                match_iou = None
            else:
                number, match_iou = assignment
                track = self.tracks[number]
                track.bbox = tuple(
                    self.alpha * current + (1 - self.alpha) * previous
                    for current, previous in zip(person["bbox"], track.bbox)
                )
                track.last_seen_ns = captured_ns
                track.hits += 1
                track.misses = 0
                track.visible = True
            track = self.tracks[number]
            person.update(
                track_id=self._track_id(track),
                track_state=self._state(track),
                tracked_bbox=self._rounded_bbox(track),
                track_hits=track.hits,
                track_age_ms=(captured_ns - track.first_seen_ns) / 1_000_000,
                track_new=new_track,
                track_match_iou=match_iou,
            )

            if self.association_confidence is not None:
                person["association_only"] = index in weak_indices

        persons = [person for person in persons if "track_id" in person]
        if self.association_confidence is not None:
            detected_primary = max(persons, key=lambda p: p["confidence"], default=None)
            result.update(
                person_detected=bool(persons),
                person_count=len(persons),
                primary_person_bbox=detected_primary["bbox"] if detected_primary else None,
                primary_person_confidence=(
                    detected_primary["confidence"] if detected_primary else None
                ),
            )
        confirmed = [person for person in persons if person["track_state"] == "confirmed"]
        primary = max(
            confirmed,
            key=lambda person: (person["track_hits"], person["confidence"]),
            default=None,
        )
        result.update(
            persons=persons,
            person_tracking_status="ok",
            active_track_count=len(self.tracks),
            tracks=self._active_tracks(captured_ns),
            confirmed_person_detected=bool(confirmed),
            confirmed_person_count=len(confirmed),
            primary_track_id=primary["track_id"] if primary else None,
            primary_tracked_bbox=primary["tracked_bbox"] if primary else None,
        )
        return result

    def _active_tracks(self, captured_ns):
        return [
            {
                "track_id": self._track_id(track),
                "state": self._state(track),
                "visible": track.visible,
                "bbox": self._rounded_bbox(track),
                "hits": track.hits,
                "misses": track.misses,
                "age_ms": (captured_ns - track.first_seen_ns) / 1_000_000,
                "since_last_seen_ms": (captured_ns - track.last_seen_ns) / 1_000_000,
            }
            for track in sorted(self.tracks.values(), key=lambda item: item.number)
        ]
