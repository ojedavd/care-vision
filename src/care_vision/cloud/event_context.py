"""Experimental causal event-context replay; no alerts or controller commands.

Rebuild tracking from source-coordinate detections, never concatenate the IDs of
live and historical trackers. Revisions may change as delayed evidence arrives.
"""

import math

from care_vision.cloud.context_association import anchored_segment
from care_vision.cloud.temporal_candidate import SequenceObservation, TemporalCandidate
from care_vision.cloud.tracking import PersonTracker
from care_vision.cloud.trajectory import PostureTrajectoryAnalyzer, map_bbox_to_source


def source_frame(observation):
    """Extract bounded raw detections and metadata, excluding cached temporal labels."""
    o = observation
    width, height = o["source_width"], o["source_height"]
    if any(type(v) is not int or v <= 0 for v in (width, height, o["width"], o["height"])):
        raise ValueError("invalid image dimensions")
    header = {k: o[k] for k in ("width", "height", "source_width", "source_height", "view_kind")}
    if o["view_kind"] == "roi":
        roi = o.get("source_roi")
        if (
            not isinstance(roi, list)
            or len(roi) != 4
            or any(type(v) is not int for v in roi)
            or roi[0] < 0
            or roi[1] < 0
            or roi[2] <= 0
            or roi[3] <= 0
            or roi[0] + roi[2] > width
            or roi[1] + roi[3] > height
        ):
            raise ValueError("invalid source ROI")
        header.update(zip(("roi_x", "roi_y", "roi_width", "roi_height"), roi))
    elif o["view_kind"] != "full_frame":
        raise ValueError("unsupported view")
    persons = o["persons"]
    if not isinstance(persons, list) or len(persons) > 32 or o["person_count"] != len(persons):
        raise ValueError("invalid person collection")
    projected, anchor_indices = [], []
    for index, person in enumerate(persons):
        bbox, confidence = person["bbox"], person["confidence"]
        if (
            len(bbox) != 4
            or not all(math.isfinite(v) for v in bbox)
            or bbox[0] < 0
            or bbox[1] < 0
            or bbox[2] <= 0
            or bbox[3] <= 0
            or bbox[0] + bbox[2] > o["width"]
            or bbox[1] + bbox[3] > o["height"]
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("invalid raw detection")
        box = map_bbox_to_source(bbox, header)
        if box[2] <= 0 or box[3] <= 0:
            raise ValueError("detection does not intersect source view")
        projected.append({"bbox": box, "confidence": confidence})
        if o["primary_track_id"] is not None and person.get("track_id") == o["primary_track_id"]:
            anchor_indices.append(index)
    return {
        **{
            k: o[k]
            for k in (
                "frame_id",
                "capture_seq",
                "captured_mono_ns",
                "session_id",
                "boot_id",
                "edge_id",
                "observation_id",
            )
        },
        "width": width,
        "height": height,
        "persons": projected,
        "primary_index": anchor_indices[0] if len(anchor_indices) == 1 else None,
        "available": o["person_detection_status"] == "ok"
        and o.get("invalid_reason") != "processing_stale",
        "evidence_kind": o.get("evidence_kind", "live"),
        "buffer_request_id": o.get("buffer_request_id"),
    }


class EventContext:
    """One bounded request window; capture time and availability time stay separate."""

    maximum_frames = 512
    post_seconds = 8

    def __init__(self, request, trigger, *, association_policy="single_person"):
        if association_policy not in {"single_person", "mutual_unique"}:
            raise ValueError("unsupported association policy")
        self.association_policy = association_policy
        if (
            request["based_on_frame_id"] != trigger["frame_id"]
            or request["caused_by_observation_id"] != trigger["observation_id"]
            or trigger.get("evidence_kind", "live") != "live"
            or request["to_edge_mono_ns"] != trigger["captured_mono_ns"]
            or not 0 < request["to_edge_mono_ns"] - request["from_edge_mono_ns"] <= 10**10
        ):
            raise ValueError("invalid causal request anchor/window")
        self.request = dict(request)
        self.anchor = source_frame(trigger)
        self.frames = {}
        self.complete = False
        self.overflow = False
        self.revision = 0
        self.last_available_ns = request["local_mono_ns"]
        self.add(trigger)

    def add(self, observation):
        """Return whether evidence or the explicit capacity status changed."""
        row = source_frame(observation)
        for key in ("session_id", "boot_id", "edge_id"):
            if row[key] != self.request[key]:
                raise ValueError("event context cannot cross sessions")
        if (row["width"], row["height"]) != (self.anchor["width"], self.anchor["height"]):
            raise ValueError("capture coordinate system changed")
        start, end = self.request["from_edge_mono_ns"], self.request["to_edge_mono_ns"]
        if row["evidence_kind"] == "historical":
            if (
                row["buffer_request_id"] != self.request["command_id"]
                or not start <= row["captured_mono_ns"] < end
            ):
                raise ValueError("history belongs to another request/window")
        elif row["evidence_kind"] != "live" or row["buffer_request_id"] is not None:
            raise ValueError("invalid event evidence kind")
        if not start <= row["captured_mono_ns"] <= end + self.post_seconds * 10**9:
            return False
        previous = self.frames.get(row["frame_id"])
        if previous is not None:
            if previous != row:
                raise ValueError("conflicting evidence for the same frame")
            return False
        if len(self.frames) >= self.maximum_frames:
            changed = not self.overflow
            self.overflow = True
            return changed
        self.frames[row["frame_id"]] = row
        return True

    def snapshot(self, available_at_cloud_mono_ns, *, include_history=True):
        if available_at_cloud_mono_ns < self.last_available_ns:
            raise ValueError("context availability cannot move backwards")
        self.last_available_ns = available_at_cloud_mono_ns
        self.revision += 1
        rows = sorted(
            (r for r in self.frames.values() if include_history or r["evidence_kind"] == "live"),
            key=lambda r: r["captured_mono_ns"],
        )
        if any(
            b["captured_mono_ns"] <= a["captured_mono_ns"] or b["capture_seq"] <= a["capture_seq"]
            for a, b in zip(rows, rows[1:])
        ):
            raise ValueError("conflicting capture chronology")
        result = {
            "request_id": self.request["command_id"],
            "revision": self.revision,
            "available_at_cloud_mono_ns": available_at_cloud_mono_ns,
            "variant": "live_plus_history" if include_history else "live_only_rebuilt",
            "frame_count": len(rows),
            "historical_frames": sum(r["evidence_kind"] == "historical" for r in rows),
            "history_transfer_complete": self.complete,
            "identity_status": "experimental_anchor_association",
            "anchor_frame_id": self.anchor["frame_id"],
            "anchor_context_track_id": None,
            "episodes": [],
            "signal_counts": {},
            "commands_enabled": False,
        }
        if self.overflow:
            result["identity_status"] = "capacity_exceeded"
            return result
        if self.association_policy == "single_person" and any(len(r["persons"]) > 1 for r in rows):
            result["identity_status"] = "ambiguous_multiple_detections"
            return result
        if self.anchor["primary_index"] is None or not self.anchor["available"]:
            result["identity_status"] = "anchor_unavailable"
            return result
        tracker = PersonTracker()
        namespace = f"context:{self.request['command_id']}"
        tracked, anchor_id = [], None
        if self.association_policy == "mutual_unique":
            association = anchored_segment(
                rows, self.anchor["frame_id"], self.anchor["primary_index"]
            )
            result["association"] = association
            result["identity_status"] = "experimental_unique_segment"
            selected = association["selected"]
            smoothed, hits = None, 0
            for row in rows:
                people = []
                if row["frame_id"] in selected:
                    person = row["persons"][selected[row["frame_id"]]]
                    box = person["bbox"]
                    smoothed = (
                        list(box)
                        if smoothed is None
                        else [
                            0.65 * value + 0.35 * previous for value, previous in zip(box, smoothed)
                        ]
                    )
                    hits += 1
                    people = [
                        {
                            **person,
                            "track_id": namespace + ":unique",
                            "track_state": "confirmed" if hits >= 2 else "tentative",
                            "tracked_bbox": [round(v) for v in smoothed],
                        }
                    ]
                    if row["frame_id"] == self.anchor["frame_id"] and hits >= 2:
                        anchor_id = people[0]["track_id"]
                tracked.append((row, {"persons": people}))
        if self.association_policy == "single_person":
            for row in rows:
                people = tracker.update(
                    {
                        "person_detection_status": "ok" if row["available"] else "unavailable",
                        "persons": row["persons"],
                    },
                    row["captured_mono_ns"],
                    namespace,
                )
                tracked.append((row, people))
                if row["frame_id"] == self.anchor["frame_id"]:
                    person = people["persons"][self.anchor["primary_index"]]
                    if person.get("track_state") == "confirmed":
                        anchor_id = person["track_id"]
        if anchor_id is None:
            result["identity_status"] = "anchor_unconfirmed_in_context"
            return result
        result["anchor_context_track_id"] = anchor_id
        trajectory, sequence = (
            PostureTrajectoryAnalyzer(),
            TemporalCandidate(retain_pending_descent=True),
        )
        counts = {
            "downward": 0,
            "horizontal": 0,
            "unknown": 0,
            "historical_downward": 0,
            "pre_anchor_downward": 0,
        }
        for row, people in tracked:
            person = (
                next(
                    (
                        p
                        for p in people["persons"]
                        if p.get("track_id") == anchor_id and p.get("track_state") == "confirmed"
                    ),
                    None,
                )
                if row["available"]
                else None
            )
            signals = trajectory.update(
                {
                    "primary_track_id": anchor_id if person else None,
                    "primary_tracked_bbox": person["tracked_bbox"] if person else None,
                },
                {
                    "width": row["width"],
                    "height": row["height"],
                    "captured_mono_ns": row["captured_mono_ns"],
                },
                namespace,
            )
            sequence.update(
                SequenceObservation(
                    (row["captured_mono_ns"] - self.request["from_edge_mono_ns"]) / 1e9,
                    row["capture_seq"] - 1,
                    anchor_id if person else None,
                    signals["primary_orientation"],
                    signals["primary_motion_direction"],
                    None,
                )
            )
            counts["downward"] += signals["primary_motion_direction"] == "downward"
            counts["historical_downward"] += (
                signals["primary_motion_direction"] == "downward"
                and row["evidence_kind"] == "historical"
            )
            counts["pre_anchor_downward"] += (
                signals["primary_motion_direction"] == "downward"
                and row["captured_mono_ns"] < self.anchor["captured_mono_ns"]
            )
            counts["horizontal"] += signals["primary_orientation"] == "horizontal"
            counts["unknown"] += person is None
        result["signal_counts"] = counts
        result["episodes"] = [dict(e) for e in sequence.episodes]
        return result
