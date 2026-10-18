"""Version 1 wire contracts; image metadata and JPEG share one binary message."""

import json
import math
import struct
from datetime import datetime, timezone
from uuid import uuid4

MAX_MESSAGE = 1_048_576
MAX_HEADER = 8192
RATES = (2, 5, 20)
OUTPUT_RESOLUTIONS = ((320, 180), (640, 360), (1280, 720))


class ProtocolError(ValueError):
    pass


def uid() -> str:
    return str(uuid4())


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def envelope(kind: str, identity: dict, **fields) -> dict:
    return {"schema_version": 1, "type": kind, **identity, "sent_at_utc": utc(), **fields}


def number(message: dict, name: str, minimum=0, maximum=2**63 - 1, integer=False):
    value = message.get(name)
    types = (int,) if integer else (int, float)
    if type(value) not in types or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ProtocolError(f"invalid {name}")
    return value


def string(message: dict, name: str):
    value = message.get(name)
    if not isinstance(value, str) or not 0 < len(value) <= 256:
        raise ProtocolError(f"invalid {name}")
    return value


def timestamp(message: dict, name: str):
    try:
        value = datetime.fromisoformat(string(message, name))
        if value.utcoffset() is None:
            raise ValueError("timezone required")
    except ValueError as exc:
        raise ProtocolError(f"invalid {name}") from exc


def validate(message: dict) -> dict:
    if not isinstance(message, dict):
        raise ProtocolError("expected object")
    if type(message.get("schema_version")) is not int or message["schema_version"] != 1:
        raise ProtocolError("unsupported schema_version")
    for field in ("edge_id", "boot_id", "session_id"):
        string(message, field)
    timestamp(message, "sent_at_utc")
    kind = message.get("type")
    if kind == "FRAME":
        string(message, "frame_id")
        number(message, "capture_seq", 1, integer=True)
        if message["frame_id"] != f"{message['boot_id']}:{message['capture_seq']}":
            raise ProtocolError("frame_id does not match capture identity")
        for field in ("captured_mono_ns", "sampled_mono_ns", "policy_seq"):
            number(message, field, integer=True)
        timestamp(message, "captured_at_utc")
        if message["sampled_mono_ns"] < message["captured_mono_ns"]:
            raise ProtocolError("sample predates capture")
        for field, limit in (("width", 1920), ("height", 1080), ("payload_bytes", MAX_MESSAGE)):
            number(message, field, 1, limit, integer=True)
        evidence = message.get("evidence_kind", "live")
        if evidence not in {"live", "historical"}:
            raise ProtocolError("invalid evidence_kind")
        if evidence == "historical":
            string(message, "buffer_request_id")
            number(message, "history_index", 0, 299, integer=True)
            number(message, "history_fps", 1, 30)
            if message["policy_seq"] != 0 or message.get("applied_command_id") is not None:
                raise ProtocolError("historical frame cannot claim a live policy")
        elif message.get("buffer_request_id") is not None:
            raise ProtocolError("live frame cannot belong to a buffer request")
        if message.get("encoding") not in {"jpeg", "webp"}:
            raise ProtocolError("only jpeg or webp supported")
        number(message, "target_transmit_fps", 1, 30)
        if "network_limited" in message and "effective_target_transmit_fps" not in message:
            raise ProtocolError("network limiting flag requires effective target")
        if "effective_target_transmit_fps" in message:
            effective = number(message, "effective_target_transmit_fps", 1, 30)
            if effective > message["target_transmit_fps"]:
                raise ProtocolError("effective rate exceeds requested rate")
            if type(message.get("network_limited")) is not bool:
                raise ProtocolError("invalid network limiting flag")
            if message["network_limited"] != (effective < message["target_transmit_fps"]):
                raise ProtocolError("network limiting flag contradicts effective rate")
        if message.get("applied_command_id") is not None:
            string(message, "applied_command_id")
        if (message["policy_seq"] == 0) != (message.get("applied_command_id") is None):
            raise ProtocolError("inconsistent policy identity")
        view_kind = message.get("view_kind", "full_frame")
        if view_kind not in {"full_frame", "roi"}:
            raise ProtocolError("invalid view_kind")
        visual_seq = message.get("visual_policy_seq", 0)
        number({"value": visual_seq}, "value", integer=True)
        visual_command = message.get("applied_visual_command_id")
        if visual_command is not None:
            string(message, "applied_visual_command_id")
        if (visual_seq == 0) != (visual_command is None):
            raise ProtocolError("inconsistent visual policy identity")
        if view_kind == "roi":
            source_width = number(message, "source_width", 1, 4096, integer=True)
            source_height = number(message, "source_height", 1, 4096, integer=True)
            x = number(message, "roi_x", 0, source_width - 1, integer=True)
            y = number(message, "roi_y", 0, source_height - 1, integer=True)
            width = number(message, "roi_width", 1, source_width, integer=True)
            height = number(message, "roi_height", 1, source_height, integer=True)
            if x + width > source_width or y + height > source_height:
                raise ProtocolError("ROI exceeds source dimensions")
            if evidence == "historical":
                raise ProtocolError("historical frame must retain full-frame context")
    elif kind in {
        "SET_TRANSMIT_FPS",
        "REQUEST_RECENT_BUFFER",
        "REQUEST_ROI",
        "REQUEST_FULL_FRAME",
    }:
        for field in ("command_id", "caused_by_observation_id", "based_on_frame_id", "reason"):
            string(message, field)
        number(message, "command_seq", 1, integer=True)
        number(message, "valid_until_edge_mono_ns", integer=True)
        if kind == "SET_TRANSMIT_FPS":
            number(message, "transmit_fps", 1, 30)
        elif kind == "REQUEST_RECENT_BUFFER":
            start = number(message, "from_edge_mono_ns", integer=True)
            end = number(message, "to_edge_mono_ns", 1, integer=True)
            if not 0 < end - start <= 10_000_000_000:
                raise ProtocolError("buffer window must be in (0, 10] seconds")
            number(message, "history_fps", 1, 30)
        else:
            output = (
                number(message, "output_width", 1, 1920, integer=True),
                number(message, "output_height", 1, 1080, integer=True),
            )
            if output not in OUTPUT_RESOLUTIONS:
                raise ProtocolError("unsupported output resolution")
            if kind == "REQUEST_ROI":
                if "context_fps" in message:
                    number(message, "context_fps", 1, 10)
                source_width = number(message, "source_width", 1, 4096, integer=True)
                source_height = number(message, "source_height", 1, 4096, integer=True)
                x = number(message, "roi_x", 0, source_width - 1, integer=True)
                y = number(message, "roi_y", 0, source_height - 1, integer=True)
                width = number(message, "roi_width", 1, source_width, integer=True)
                height = number(message, "roi_height", 1, source_height, integer=True)
                if x + width > source_width or y + height > source_height:
                    raise ProtocolError("ROI exceeds source dimensions")
        timestamp(message, "issued_at_utc")
    elif kind == "COMMAND_ACK":
        string(message, "command_id")
        number(message, "command_seq", 1, integer=True)
        if message.get("result") not in {"applied", "duplicate", "rejected", "expired", "stale"}:
            raise ProtocolError("invalid result")
        number(message, "active_transmit_fps", 1, 30)
        number(message, "active_policy_seq", integer=True)
        if message.get("command_type") == "REQUEST_RECENT_BUFFER" and message["result"] in {
            "applied",
            "duplicate",
        }:
            number(message, "selected_frames", 0, 300, integer=True)
            number(message, "selected_bytes", 0, 32 * 1024 * 1024, integer=True)
            number(message, "deduplicated_frames", 0, 300, integer=True)
            if message.get("coverage") not in {"full", "partial", "empty"}:
                raise ProtocolError("invalid buffer coverage")
        if "active_context_fps" in message:
            number(message, "active_context_fps", 0, 10)
        if message.get("command_type") in {"REQUEST_ROI", "REQUEST_FULL_FRAME"}:
            if message.get("active_view_kind") not in {"full_frame", "roi"}:
                raise ProtocolError("invalid active view")
            number(message, "active_visual_policy_seq", integer=True)
            number(message, "active_output_width", 1, 1920, integer=True)
            number(message, "active_output_height", 1, 1080, integer=True)
        if message["result"] in {"applied", "duplicate"}:
            number(message, "applied_at_edge_mono_ns", integer=True)
            number(message, "effective_from_capture_seq", 1, integer=True)
    elif kind == "DRAIN_REQUEST":
        string(message, "drain_id")
        number(message, "live_frames_sent", integer=True)
    elif kind == "DRAINED":
        string(message, "drain_id")
        for field in (
            "live_received",
            "live_processed",
            "live_dropped",
            "historical_received",
            "historical_processed",
        ):
            number(message, field, integer=True)
        if message["live_processed"] + message["live_dropped"] != message["live_received"]:
            raise ProtocolError("incomplete live drain")
        if message["historical_processed"] != message["historical_received"]:
            raise ProtocolError("incomplete historical drain")
    elif kind == "BUFFER_COMPLETE":
        string(message, "buffer_request_id")
        number(message, "frames_sent", 0, 300, integer=True)
    elif kind == "CLOUD_STATUS":
        number(message, "status_seq", 1, integer=True)
        for field in ("live_received", "live_processed", "live_dropped", "live_stale"):
            number(message, field, integer=True)
        capacity = number(message, "queue_capacity", 1, 64, integer=True)
        depth = number(message, "queue_depth", 0, capacity, integer=True)
        pending = message["live_received"] - message["live_processed"] - message["live_dropped"]
        if pending - depth not in (0, 1) or message["live_stale"] > message["live_processed"]:
            raise ProtocolError("inconsistent cloud status accounting")
    elif kind == "EDGE_STATUS":
        if "cloud_status_version" in message:
            number(message, "cloud_status_version", 1, 1, integer=True)
        if "pipeline_quality" in message:
            if message["pipeline_quality"] not in {"GOOD", "DEGRADED", "UNKNOWN", "OFFLINE"}:
                raise ProtocolError("invalid pipeline quality")
            if message.get("cloud_feedback_state") not in {"FRESH", "STALE", "UNKNOWN", "OFFLINE"}:
                raise ProtocolError("invalid cloud feedback state")
            for field in ("cloud_status_seq", "cloud_live_received", "cloud_live_processed",
                          "cloud_live_dropped", "cloud_live_stale", "cloud_queue_depth",
                          "cloud_queue_capacity"):
                if field not in message:
                    raise ProtocolError("incomplete cloud feedback fields")
                if message[field] is not None:
                    number(message, field, integer=True)
            if message.get("cloud_feedback_age_ms") is not None:
                number(message, "cloud_feedback_age_ms")
            feedback_fields = ("live_received", "live_processed", "live_dropped", "live_stale",
                               "queue_depth", "queue_capacity")
            if message["cloud_status_seq"] is None:
                if (any(message["cloud_" + f] is not None for f in feedback_fields)
                        or message.get("cloud_feedback_age_ms") is not None
                        or message["cloud_feedback_state"] not in {"UNKNOWN", "OFFLINE"}):
                    raise ProtocolError("unknown cloud feedback contains measured values")
            else:
                number(message, "cloud_feedback_age_ms")
                validate({**message, "type": "CLOUD_STATUS",
                          "status_seq": message["cloud_status_seq"],
                          **{f: message["cloud_" + f] for f in feedback_fields}})
            if message["pipeline_quality"] == "GOOD" and (
                message.get("network_quality") != "GOOD"
                or message["cloud_feedback_state"] != "FRESH"
                or not message["cloud_live_processed"]
                or message["cloud_live_dropped"] or message["cloud_live_stale"]
            ):
                raise ProtocolError("GOOD pipeline contradicts evidence")
            if (message["pipeline_quality"] == "OFFLINE") != (
                message.get("network_quality") == "OFFLINE"
            ):
                raise ProtocolError("pipeline availability contradicts network")
        for field in ("capture_fps", "transmit_fps", "window_seconds"):
            number(message, field)
        number(message, "target_transmit_fps", 1, 30)
        if "network_cap_fps" in message and "effective_target_transmit_fps" not in message:
            raise ProtocolError("network cap requires effective target")
        if "effective_target_transmit_fps" in message:
            effective = number(message, "effective_target_transmit_fps", 1, 30)
            cap = message.get("network_cap_fps")
            if cap is not None:
                number(message, "network_cap_fps", 1, 30)
            if effective != min(message["target_transmit_fps"], cap or 30):
                raise ProtocolError("effective target contradicts network cap")
        for field in ("captured", "selected", "sent", "dropped", "bytes_sent", "policy_seq"):
            number(message, field, integer=True)
        if "network_limited_frames" in message:
            number(message, "network_limited_frames", integer=True)
        if message.get("connection") not in {"connected", "offline"}:
            raise ProtocolError("invalid connection state")
        if "network_quality" in message:
            if message["network_quality"] not in {"GOOD", "DEGRADED", "OFFLINE"}:
                raise ProtocolError("invalid network quality")
            rtt = message.get("network_rtt_ms")
            if rtt is not None:
                number(message, "network_rtt_ms", 0, 60_000)
            if type(message.get("cloud_analysis_available")) is not bool:
                raise ProtocolError("invalid cloud analysis availability")
            if (message["network_quality"] == "OFFLINE") == message["cloud_analysis_available"]:
                raise ProtocolError("network quality contradicts cloud availability")
            number(message, "offline_seconds")
            number(message, "captured_while_offline", integer=True)
            offline_counts = {
                "offline_frames_retained_at_reconnect",
                "offline_frames_lost_at_reconnect",
                "recovery_window_available",
            }
            if offline_counts & message.keys():
                if not offline_counts <= message.keys():
                    raise ProtocolError("incomplete offline recovery accounting")
                retained = number(message, "offline_frames_retained_at_reconnect", integer=True)
                lost = number(message, "offline_frames_lost_at_reconnect", integer=True)
                if type(message["recovery_window_available"]) is not bool:
                    raise ProtocolError("invalid recovery window availability")
                if retained + lost != message["captured_while_offline"]:
                    raise ProtocolError("offline recovery counts do not reconcile")
                if message["recovery_window_available"] != (
                    "recovery_from_edge_mono_ns" in message
                ):
                    raise ProtocolError("recovery window availability contradicts bounds")
            recovery_fields = {
                "recovery_from_edge_mono_ns",
                "recovery_to_edge_mono_ns",
                "recovery_available_frames",
                "recovery_truncated",
            }
            if recovery_fields & message.keys():
                if not recovery_fields <= message.keys():
                    raise ProtocolError("incomplete recovery window")
                start = number(message, "recovery_from_edge_mono_ns", integer=True)
                end = number(message, "recovery_to_edge_mono_ns", integer=True)
                if not 0 < end - start <= 10_000_000_000:
                    raise ProtocolError("invalid recovery window")
                number(message, "recovery_available_frames", 1, 300, integer=True)
                if offline_counts <= message.keys() and (
                    message["recovery_available_frames"]
                    < message["offline_frames_retained_at_reconnect"]
                ):
                    raise ProtocolError("recovery window smaller than offline evidence")
                if type(message["recovery_truncated"]) is not bool:
                    raise ProtocolError("invalid recovery truncation flag")
                if (
                    "recovery_includes_unconfirmed_context" in message
                    and type(message["recovery_includes_unconfirmed_context"]) is not bool
                ):
                    raise ProtocolError("invalid recovery context flag")
            spool_fields = {
                "spool_frames",
                "spool_encrypted_bytes",
                "spool_evicted",
                "spool_oldest_mono_ns",
                "spool_newest_mono_ns",
            }
            if spool_fields & message.keys():
                if not spool_fields <= message.keys():
                    raise ProtocolError("incomplete encrypted spool status")
                count = number(message, "spool_frames", 0, 3600, integer=True)
                number(message, "spool_encrypted_bytes", 0, 512 * 1024 * 1024, integer=True)
                number(message, "spool_evicted", integer=True)
                oldest, newest = (
                    message["spool_oldest_mono_ns"],
                    message["spool_newest_mono_ns"],
                )
                if count:
                    if oldest is None or newest is None:
                        raise ProtocolError("nonempty spool requires time bounds")
                    number(message, "spool_oldest_mono_ns", integer=True)
                    number(message, "spool_newest_mono_ns", integer=True)
                    if oldest > newest:
                        raise ProtocolError("invalid spool time bounds")
                elif oldest is not None or newest is not None:
                    raise ProtocolError("empty spool must not have time bounds")
    else:
        raise ProtocolError("unsupported message type")
    return message


def dumps(message: dict) -> str:
    validate(message)
    return json.dumps(message, separators=(",", ":"), allow_nan=False)


def loads(data: str) -> dict:
    if len(data) > MAX_HEADER:
        raise ProtocolError("control message too large")
    try:
        message = json.loads(data)
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("invalid JSON") from exc
    return validate(message)


def pack_frame(header: dict, jpeg: bytes) -> bytes:
    raw = dumps(header).encode()
    if len(raw) > MAX_HEADER or len(raw) + len(jpeg) + 4 > MAX_MESSAGE:
        raise ProtocolError("frame too large")
    if len(jpeg) != header["payload_bytes"]:
        raise ProtocolError("payload length mismatch")
    return struct.pack("!I", len(raw)) + raw + jpeg


def unpack_frame(data: bytes) -> tuple[dict, bytes]:
    if not 4 < len(data) <= MAX_MESSAGE:
        raise ProtocolError("invalid frame size")
    size = struct.unpack("!I", data[:4])[0]
    if not 0 < size <= MAX_HEADER or 4 + size >= len(data):
        raise ProtocolError("invalid header size")
    try:
        header = loads(data[4 : 4 + size].decode())
    except UnicodeError as exc:
        raise ProtocolError("invalid header encoding") from exc
    jpeg = data[4 + size :]
    if header["type"] != "FRAME" or len(jpeg) != header["payload_bytes"]:
        raise ProtocolError("invalid frame payload")
    return header, jpeg
