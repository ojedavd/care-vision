"""Verify the causal REQUEST_ROI and REQUEST_FULL_FRAME perception loop."""

from care_vision.check_loop import require


def _events(records, event, *, command_type=None):
    selected = [record for record in records if record.get("event") == event]
    if command_type is not None:
        selected = [record for record in selected if record.get("type") == command_type]
    return selected


def _one(records, reason):
    require(len(records) == 1, reason)
    return records[0]


def check_roi_records(edge, cloud):
    """Return measured ROI evidence or raise CheckFailed on any broken link."""
    roi = _one(
        _events(cloud, "agent_action", command_type="REQUEST_ROI"),
        "expected exactly one ROI request",
    )
    full = _one(
        _events(cloud, "agent_action", command_type="REQUEST_FULL_FRAME"),
        "expected exactly one full-frame return",
    )
    require(roi["session_id"] == full["session_id"], "visual commands span sessions")
    session_id = roi["session_id"]
    require(
        len({record.get("session_id") for record in _events(edge, "connected")}) == 1,
        "ROI demo must finish in one healthy session",
    )
    require(
        not any(
            record.get("event")
            in {"session_error", "frame_dropped", "buffer_incomplete", "buffer_transfer_aborted"}
            for record in edge + cloud
        ),
        "ROI demo contains an error, dropped frame, or incomplete history",
    )

    observations = _events(cloud, "observation")
    observations_by_id = {record["observation_id"]: record for record in observations}
    trigger = observations_by_id.get(roi["caused_by_observation_id"])
    require(trigger is not None, "ROI request has no trigger observation")
    require(
        trigger.get("valid")
        and trigger.get("evidence_kind", "live") == "live"
        and trigger.get("view_kind") == "full_frame"
        and trigger.get("confirmed_person_detected")
        and trigger.get("primary_tracked_bbox"),
        "ROI trigger is not a valid confirmed-person full-frame observation",
    )
    require(trigger["frame_id"] == roi["based_on_frame_id"], "ROI trigger frame mismatch")
    require(trigger["session_id"] == session_id, "ROI trigger belongs to another session")
    source_width, source_height = trigger["source_width"], trigger["source_height"]
    requested_roi = [roi[key] for key in ("roi_x", "roi_y", "roi_width", "roi_height")]
    x, y, width, height = requested_roi
    require(
        roi["source_width"] == source_width
        and roi["source_height"] == source_height
        and x >= 0
        and y >= 0
        and width > 0
        and height > 0
        and x + width <= source_width
        and y + height <= source_height,
        "ROI request is outside the source image",
    )

    edge_roi_ack = _one(
        [
            record
            for record in _events(edge, "command_ack")
            if record.get("command_id") == roi["command_id"] and record.get("result") == "applied"
        ],
        "ROI request must be applied exactly once",
    )
    cloud_roi_ack = _one(
        [
            record
            for record in _events(cloud, "ack_received")
            if record.get("command_id") == roi["command_id"]
            and record.get("result") in {"applied", "duplicate"}
        ],
        "cloud did not receive the ROI ACK exactly once",
    )
    for ack in (edge_roi_ack, cloud_roi_ack):
        require(
            ack.get("active_view_kind") == "roi"
            and ack.get("active_visual_policy_seq") == roi["command_seq"]
            and ack.get("applied_visual_command_id") == roi["command_id"],
            "ROI ACK does not identify the active visual policy",
        )

    sent = _events(edge, "frame_sent")
    received = _events(cloud, "frame_received")
    roi_sent = [record for record in sent if record.get("view_kind") == "roi"]
    roi_received = [record for record in received if record.get("view_kind") == "roi"]
    roi_observations = [record for record in observations if record.get("view_kind") == "roi"]
    require(len(roi_sent) >= 10, "missing sustained ROI transmission")
    roi_ids = {record["frame_id"] for record in roi_sent}
    require(
        {record["frame_id"] for record in roi_received} == roi_ids
        and {record["frame_id"] for record in roi_observations} == roi_ids,
        "ROI frames were not all received and perceived",
    )
    for record in roi_sent + roi_received:
        require(record.get("evidence_kind", "live") == "live", "historical frame claims ROI")
        require(
            record.get("session_id") == session_id
            and [record.get(key) for key in ("roi_x", "roi_y", "roi_width", "roi_height")]
            == requested_roi
            and record.get("visual_policy_seq") == roi["command_seq"]
            and record.get("applied_visual_command_id") == roi["command_id"]
            and record.get("width") == roi["output_width"]
            and record.get("height") == roi["output_height"],
            "ROI frame or observation metadata does not match the request",
        )
    for record in roi_observations:
        require(record.get("evidence_kind", "live") == "live", "historical frame claims ROI")
        require(
            record.get("session_id") == session_id
            and record.get("source_roi") == requested_roi
            and record.get("visual_policy_seq") == roi["command_seq"]
            and record.get("applied_visual_command_id") == roi["command_id"]
            and record.get("width") == roi["output_width"]
            and record.get("height") == roi["output_height"],
            "ROI frame or observation metadata does not match the request",
        )
    require(
        min(record["capture_seq"] for record in roi_sent)
        >= edge_roi_ack["effective_from_capture_seq"],
        "ROI frame predates command application",
    )

    full_trigger = observations_by_id.get(full["caused_by_observation_id"])
    require(
        full_trigger is not None
        and full_trigger.get("valid")
        and full_trigger.get("view_kind") == "roi"
        and full_trigger["frame_id"] == full["based_on_frame_id"],
        "full-frame return was not caused by valid ROI evidence",
    )
    edge_full_ack = _one(
        [
            record
            for record in _events(edge, "command_ack")
            if record.get("command_id") == full["command_id"] and record.get("result") == "applied"
        ],
        "full-frame return must be applied exactly once",
    )
    _one(
        [
            record
            for record in _events(cloud, "ack_received")
            if record.get("command_id") == full["command_id"]
            and record.get("result") in {"applied", "duplicate"}
        ],
        "cloud did not receive the full-frame ACK exactly once",
    )
    require(
        edge_full_ack.get("active_view_kind") == "full_frame"
        and edge_full_ack.get("active_visual_policy_seq") == full["command_seq"]
        and edge_full_ack.get("applied_visual_command_id") == full["command_id"],
        "full-frame ACK does not identify the active visual policy",
    )
    restored = [
        record
        for record in observations
        if record.get("view_kind") == "full_frame"
        and record.get("applied_visual_command_id") == full["command_id"]
    ]
    require(len(restored) >= 10, "missing sustained perception after full-frame return")
    require(
        all(
            record.get("source_roi") is None
            and record.get("visual_policy_seq") == full["command_seq"]
            for record in restored
        ),
        "restored full-frame observations retain ROI metadata",
    )

    return {
        "verified": True,
        "session_id": session_id,
        "roi_command_id": roi["command_id"],
        "roi": requested_roi,
        "output_resolution": [roi["output_width"], roi["output_height"]],
        "roi_frames_sent": len(roi_sent),
        "roi_frames_received": len(roi_received),
        "roi_observations": len(roi_observations),
        "confirmed_person_roi_observations": sum(
            bool(record.get("confirmed_person_detected")) for record in roi_observations
        ),
        "full_frame_command_id": full["command_id"],
        "restored_full_frame_observations": len(restored),
        "roi_ack_round_trip_ms": round(
            (cloud_roi_ack["local_mono_ns"] - roi["local_mono_ns"]) / 1e6, 3
        ),
    }
