"""Pixel-based motion probe for the communication MVP; not fall inference."""

import time
from collections import deque

import cv2
import numpy as np

from care_vision.cloud.person import observe_people
from care_vision.cloud.source_tracking import SourceProjectionError, track_in_source
from care_vision.cloud.tracking import PersonTracker
from care_vision.cloud.trajectory import PostureTrajectoryAnalyzer
from care_vision.common.contracts import uid, utc
from care_vision.common.image_geometry import ImageGeometry


class MotionVision:
    def __init__(
        self,
        backend=None,
        runtime_id=None,
        person_detector=None,
        timeline="live",
        experimental_weak_association=False,
    ):
        self.history = deque(maxlen=64)
        self.backend = backend or f"opencv-{cv2.__version__}-motion-probe-NOT-COOL"
        self.runtime_id = runtime_id
        self.person_detector = person_detector
        self.timeline = timeline
        self.person_tracker = PersonTracker(
            association_confidence=0.2 if experimental_weak_association else None,
            new_track_confidence=0.35,
        )
        self.trajectory = PostureTrajectoryAnalyzer()
        self.visual_identity = None
        self.tracking_source_identity = None
        self.tracking_geometry_epoch = 0

    def observe(self, header: dict, jpeg: bytes) -> dict:
        started = time.monotonic_ns()
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        reason = None
        score = interval = reference_id = None
        captured = header["captured_mono_ns"]
        image_valid = image is not None and image.shape[:2] == (
            header["height"],
            header["width"],
        )
        person = observe_people(
            self.person_detector,
            image if image_valid else None,
            header["frame_id"],
        )
        namespace = (
            f"historical:{header.get('buffer_request_id')}"
            if self.timeline == "historical"
            else f"live:{header['session_id']}"
        )
        geometry_error = None
        try:
            try:
                geometry = ImageGeometry.from_header(header)
            except (KeyError, TypeError, ValueError) as error:
                raise SourceProjectionError("invalid source geometry") from error
            source_identity = (namespace, geometry.source_width, geometry.source_height)
            if self.tracking_source_identity is not None and (
                source_identity != self.tracking_source_identity
            ):
                self.tracking_geometry_epoch += 1
            self.tracking_source_identity = source_identity
            if self.tracking_geometry_epoch:
                namespace += f":source-geometry:{self.tracking_geometry_epoch}"
            person = track_in_source(self.person_tracker, person, header, namespace, geometry)
        except SourceProjectionError:
            geometry_error = "source_projection_unavailable"
            # Do not update tracker state from invalid geometry or padding-only
            # detections. Keep raw detection diagnostics but expose no identity.
            person.update(
                person_tracking_status=geometry_error, persons=[], tracks=[],
                active_track_count=0, confirmed_person_detected=False,
                confirmed_person_count=0, primary_track_id=None, primary_tracked_bbox=None,
                primary_tracked_source_bbox=None, tracking_coordinate_space="unavailable",
            )
        person.update(self.trajectory.update(person, header, namespace))
        if not image_valid:
            self.history.clear()
            reason = "invalid_image"
        else:
            visual_identity = (
                header.get("view_kind", "full_frame"),
                header.get("applied_visual_command_id"),
                header["width"],
                header["height"],
                header.get("source_width", header["width"]),
                header.get("source_height", header["height"]),
                tuple(header.get(k) for k in ("roi_x", "roi_y", "roi_width", "roi_height"))
                if header.get("view_kind") == "roi" else None,
            )
            if self.visual_identity is not None and visual_identity != self.visual_identity:
                self.history.clear()
            self.visual_identity = visual_identity
            gray = cv2.cvtColor(cv2.resize(image, (160, 90)), cv2.COLOR_BGR2GRAY)
            while self.history and captured - self.history[0][0] > 1_000_000_000:
                self.history.popleft()
            candidates = [
                item for item in self.history if 400_000_000 <= captured - item[0] <= 800_000_000
            ]
            if candidates:
                previous, reference_id, reference = min(
                    candidates, key=lambda item: abs(captured - item[0] - 500_000_000)
                )
                score = float(cv2.mean(cv2.absdiff(gray, reference))[0] / 255)
                interval = (captured - previous) / 1_000_000
            else:
                reason = "insufficient_temporal_context"
            self.history.append((captured, header["frame_id"], gray))
        if geometry_error:
            reason = geometry_error
            self.history.clear()
        return {
            "type": "OBSERVATION",
            "observation_schema_version": 1,
            "observation_id": uid(),
            **{
                key: header[key]
                for key in (
                    "edge_id",
                    "boot_id",
                    "session_id",
                    "frame_id",
                    "capture_seq",
                    "captured_mono_ns",
                    "captured_at_utc",
                    "applied_command_id",
                    "policy_seq",
                    "width",
                    "height",
                )
            },
            "view_kind": header.get("view_kind", "full_frame"),
            "source_width": header.get("source_width", header["width"]),
            "source_height": header.get("source_height", header["height"]),
            "visual_policy_seq": header.get("visual_policy_seq", 0),
            "applied_visual_command_id": header.get("applied_visual_command_id"),
            "source_roi": (
                [
                    header["roi_x"],
                    header["roi_y"],
                    header["roi_width"],
                    header["roi_height"],
                ]
                if header.get("view_kind") == "roi"
                else None
            ),
            "evidence_kind": header.get("evidence_kind", "live"),
            "target_transmit_fps": header.get("target_transmit_fps"),
            "buffer_request_id": header.get("buffer_request_id"),
            "reference_frame_id": reference_id,
            "processed_at_utc": utc(),
            "processing_ms": (time.monotonic_ns() - started) / 1_000_000,
            "vision_backend": self.backend,
            "runtime_id": self.runtime_id,
            "valid": reason is None,
            "invalid_reason": reason,
            "motion_score": score,
            "comparison_interval_ms": interval,
            **person,
        }
