import json
import struct

import pytest
from conftest import command, frame, roi_command

from care_vision.common.contracts import ProtocolError, loads, pack_frame, unpack_frame, validate


def test_frame_roundtrip(identity):
    header, jpeg = frame(identity)
    assert unpack_frame(pack_frame(header, jpeg)) == (header, jpeg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("transmit_fps", float("nan")),
        ("transmit_fps", float("inf")),
        ("transmit_fps", True),
        ("transmit_fps", -1),
        ("command_seq", 1.5),
        ("session_id", ""),
        ("issued_at_utc", "yesterday"),
    ],
)
def test_invalid_command_fields(identity, field, value):
    message = command(identity)
    message[field] = value
    with pytest.raises(ProtocolError):
        validate(message)


@pytest.mark.parametrize("payload", [b"", b"1234", struct.pack("!I", 9000) + b"garbage"])
def test_malformed_binary(payload):
    with pytest.raises(ProtocolError):
        unpack_frame(payload)


def test_truncated_payload_and_wrong_identity(identity):
    header, jpeg = frame(identity)
    with pytest.raises(ProtocolError):
        unpack_frame(pack_frame(header, jpeg)[:-1])
    header["frame_id"] = "different"
    with pytest.raises(ProtocolError):
        validate(header)


@pytest.mark.parametrize("value", ["null", "[]", "{", json.dumps({"type": "unknown"})])
def test_invalid_control(value):
    with pytest.raises(ProtocolError):
        loads(value)


def test_roi_contract_rejects_out_of_bounds_and_unsupported_resolution(identity):
    message = roi_command(identity)
    assert validate(message) == message
    with pytest.raises(ProtocolError, match="exceeds"):
        validate({**message, "roi_x": 500})
    with pytest.raises(ProtocolError, match="resolution"):
        validate({**message, "output_width": 416, "output_height": 416})


def test_roi_frame_requires_consistent_visual_identity(identity):
    header, _ = frame(identity)
    header.update(
        view_kind="roi",
        source_width=640,
        source_height=360,
        roi_x=160,
        roi_y=40,
        roi_width=240,
        roi_height=300,
        visual_policy_seq=2,
        applied_visual_command_id="R1",
    )
    assert validate(header) == header
    with pytest.raises(ProtocolError, match="visual policy"):
        validate({**header, "applied_visual_command_id": None})


def test_high_resolution_mixed_roi_contract(identity):
    message = {
        **roi_command(identity),
        "output_width": 1280,
        "output_height": 720,
        "context_fps": 2,
    }
    assert validate(message) == message
    for value in [0, 11, True, float("nan")]:
        with pytest.raises(ProtocolError):
            validate({**message, "context_fps": value})
