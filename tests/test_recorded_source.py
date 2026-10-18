import json

import cv2
import numpy as np
import pytest

from care_vision.edge.recorded_source import RecordedSource
from care_vision.edge.sampler import Sampler


def test_recorded_intervals_and_ring_are_preserved(tmp_path):
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 360))
    offsets = [0, 40_000_000, 90_000_000, 180_000_000, 210_000_000, 300_000_000]
    rows = []
    for i, offset in enumerate(offsets):
        writer.write(np.full((360, 640, 3), i * 30, dtype=np.uint8))
        rows.append({"capture_seq": i + 1, "captured_mono_ns": 123_000_000_000 + offset})
    writer.release()
    captures = tmp_path / "captures.jsonl"
    captures.write_text("".join(json.dumps(r) + "\n" for r in rows))
    mapping = tmp_path / "mapping.jsonl"
    source = RecordedSource(video, captures, width=640, height=360, mapping=mapping)
    source.start()
    source.thread.join(3)
    source.close()
    assert source.done and source.error is None and source.count == len(rows)
    mapped = [json.loads(line) for line in mapping.read_text().splitlines()]
    first = mapped[0]["replay_captured_mono_ns"]
    assert source.replay_origin_ns == first
    assert [r["replay_captured_mono_ns"] - first for r in mapped] == offsets
    assert all(r["available_mono_ns"] >= r["replay_captured_mono_ns"] for r in mapped)
    selected, status = source.recent(first, first + offsets[-1] + 1, 20)
    assert selected and selected[0].seq == 1 and selected[-1].seq == len(rows)
    assert status["buffer_frames"] == len(rows)


def test_replay_origin_defines_a_repeatable_two_fps_phase(tmp_path):
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 360))
    offsets = [0, 100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000]
    for index, _ in enumerate(offsets):
        writer.write(np.full((360, 640, 3), index * 30, dtype=np.uint8))
    writer.release()
    captures = tmp_path / "captures.jsonl"
    captures.write_text(
        "".join(
            json.dumps({"capture_seq": i + 1, "captured_mono_ns": 123_000_000_000 + offset}) + "\n"
            for i, offset in enumerate(offsets)
        )
    )
    source = RecordedSource(video, captures)
    source.start()
    source.thread.join(3)
    source.close()
    sampler = Sampler(2)
    sampler.set_rate(2, source.replay_origin_ns + 200_000_000)
    frames = list(source.buffer.frames)
    assert [frame.seq for frame in frames if sampler.select(frame.seq, frame.mono_ns)] == [3]


@pytest.mark.parametrize("stamps", [[2, 2], [3, 2], [0, 121_000_000_000]])
def test_recorded_source_rejects_invalid_timeline(tmp_path, stamps):
    captures = tmp_path / "captures.jsonl"
    captures.write_text(
        "".join(
            json.dumps({"capture_seq": i + 1, "captured_mono_ns": t}) + "\n"
            for i, t in enumerate(stamps)
        )
    )
    with pytest.raises(ValueError):
        RecordedSource(tmp_path / "missing.avi", captures)
