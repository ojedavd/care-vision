"""Verify high-resolution capture with reduced full-frame and detailed ROI output."""

import statistics

from care_vision.check_loop import require


def check_resolution_records(edge, cloud):
    roi = [
        record
        for record in cloud
        if record.get("event") == "agent_action" and record.get("type") == "REQUEST_ROI"
    ]
    full = [
        record
        for record in cloud
        if record.get("event") == "agent_action" and record.get("type") == "REQUEST_FULL_FRAME"
    ]
    require(len(roi) == len(full) == 1, "expected one ROI and one full-frame command")
    roi, full = roi[0], full[0]
    require(
        (roi["source_width"], roi["source_height"]) == (1280, 720),
        "ROI was not produced from the high-resolution capture buffer",
    )
    require(
        (roi["output_width"], roi["output_height"]) == (640, 360),
        "ROI output is not 640x360",
    )
    require(
        (full["output_width"], full["output_height"]) == (320, 180),
        "full-frame return did not restore the reduced baseline",
    )

    sent = [record for record in edge if record.get("event") == "frame_sent"]
    live = [record for record in sent if record.get("evidence_kind", "live") == "live"]
    baseline = [
        record
        for record in live
        if record.get("view_kind", "full_frame") == "full_frame"
        and (record.get("width"), record.get("height")) == (320, 180)
    ]
    roi_frames = [record for record in live if record.get("view_kind") == "roi"]
    historical = [record for record in sent if record.get("evidence_kind") == "historical"]
    deferred = [record for record in edge if record.get("event") == "history_transmission_deferred"]
    resumed = [record for record in edge if record.get("event") == "history_transmission_resumed"]
    require(len(baseline) >= 10, "missing reduced full-frame baseline")
    require(len(roi_frames) >= 10, "missing detailed ROI frames")
    require(historical, "missing high-resolution historical evidence")
    # Since 7G history may run during VERIFYING. Scheduling/completeness belongs
    # to check_history_delivery; representation checks must not require old deferral.
    require(
        all((record["width"], record["height"]) == (640, 360) for record in roi_frames),
        "ROI frames did not use the requested resolution",
    )
    require(
        all(
            (record.get("source_width"), record.get("source_height")) == (1280, 720)
            for record in baseline + roi_frames
        ),
        "live frame metadata does not identify the high-resolution source",
    )
    require(
        all((record["width"], record["height"]) == (1280, 720) for record in historical),
        "historical evidence did not retain capture resolution",
    )
    require(
        all(record.get("view_kind", "full_frame") != "roi" for record in historical),
        "historical evidence was cropped by a later ROI policy",
    )

    baseline_bytes = statistics.mean(record["payload_bytes"] for record in baseline)
    historical_bytes = statistics.mean(record["payload_bytes"] for record in historical)
    require(baseline_bytes < historical_bytes, "reduced baseline did not reduce encoded bytes")
    baseline_scale = 320 / roi["source_width"]
    roi_scale = 640 / roi["roi_width"]
    require(roi_scale > baseline_scale, "ROI did not increase spatial sampling of its region")
    return {
        "verified": True,
        "capture_resolution": [1280, 720],
        "baseline_resolution": [320, 180],
        "roi_resolution": [640, 360],
        "baseline_frames": len(baseline),
        "roi_frames": len(roi_frames),
        "historical_frames": len(historical),
        "history_deferred_ms": round(
            (resumed[0]["local_mono_ns"] - deferred[0]["local_mono_ns"]) / 1e6, 3
        ) if len(deferred) == len(resumed) == 1 else None,
        "baseline_mean_payload_bytes": round(baseline_bytes),
        "high_resolution_history_mean_payload_bytes": round(historical_bytes),
        "baseline_payload_reduction_percent": round(
            (1 - baseline_bytes / historical_bytes) * 100, 2
        ),
        "baseline_horizontal_scale": round(baseline_scale, 4),
        "roi_horizontal_scale": round(roi_scale, 4),
        "roi_sampling_gain": round(roi_scale / baseline_scale, 2),
    }
