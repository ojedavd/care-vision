"""Person-detection contract independent from a specific model backend."""

import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

MAX_PERSONS = 32


@dataclass(frozen=True)
class PersonDetection:
    """One person detection in the received image's pixel coordinates."""

    bbox: tuple[int, int, int, int]
    confidence: float


class PersonDetector(Protocol):
    backend: str

    def detect(self, image: np.ndarray) -> list[PersonDetection]: ...


def _validated(detections, width: int, height: int) -> list[PersonDetection]:
    if not isinstance(detections, list) or len(detections) > MAX_PERSONS:
        raise ValueError("invalid person detections collection")
    result = []
    for detection in detections:
        if not isinstance(detection, PersonDetection):
            raise ValueError("person detector returned an unsupported item")
        bbox = detection.bbox
        if (
            not isinstance(bbox, tuple)
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
        ):
            raise ValueError("invalid person bbox")
        x, y, box_width, box_height = bbox
        if (
            x < 0
            or y < 0
            or box_width <= 0
            or box_height <= 0
            or x + box_width > width
            or y + box_height > height
        ):
            raise ValueError("person bbox is outside the frame")
        confidence = detection.confidence
        if type(confidence) not in {int, float} or not math.isfinite(confidence):
            raise ValueError("invalid person confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("person confidence is outside [0, 1]")
        result.append(PersonDetection(bbox, float(confidence)))
    return sorted(result, key=lambda item: item.confidence, reverse=True)


def observe_people(detector: PersonDetector | None, image, frame_id: str) -> dict:
    """Return explicit availability and normalized person detections."""
    if image is None:
        return {
            "person_detection_status": "invalid_image",
            "person_detector_backend": getattr(detector, "backend", None),
            "person_detection_ms": None,
            "person_detected": False,
            "person_count": 0,
            "persons": [],
            "primary_person_bbox": None,
            "primary_person_confidence": None,
        }
    if detector is None:
        return {
            "person_detection_status": "not_configured",
            "person_detector_backend": None,
            "person_detection_ms": None,
            "person_detected": False,
            "person_count": 0,
            "persons": [],
            "primary_person_bbox": None,
            "primary_person_confidence": None,
        }

    started = time.monotonic_ns()
    try:
        detections = _validated(detector.detect(image), image.shape[1], image.shape[0])
    except Exception as exc:
        return {
            "person_detection_status": "error",
            "person_detection_error": f"{type(exc).__name__}: {exc}",
            "person_detector_backend": detector.backend,
            "person_detection_ms": (time.monotonic_ns() - started) / 1_000_000,
            "person_detected": False,
            "person_count": 0,
            "persons": [],
            "primary_person_bbox": None,
            "primary_person_confidence": None,
        }

    persons = [
        {
            "detection_id": f"{frame_id}:person:{index}",
            "label": "person",
            "confidence": detection.confidence,
            "bbox": list(detection.bbox),
        }
        for index, detection in enumerate(detections)
    ]
    primary = persons[0] if persons else None
    return {
        "person_detection_status": "ok",
        "person_detector_backend": detector.backend,
        "person_detection_ms": (time.monotonic_ns() - started) / 1_000_000,
        "person_detected": bool(persons),
        "person_count": len(persons),
        "persons": persons,
        "primary_person_bbox": primary["bbox"] if primary else None,
        "primary_person_confidence": primary["confidence"] if primary else None,
    }
