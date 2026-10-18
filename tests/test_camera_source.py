import json
import os

import cv2
import numpy as np
import pytest

from care_vision.edge.source import (
    RTSP_FFMPEG_NOBUFFER,
    CameraSource,
    RTSPSource,
    VideoSource,
)


class FakeCamera:
    def __init__(self, fail=False, opened=True):
        self.fail, self.opened = fail, opened
        self.released = False
        self.settings = []

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.settings.append((prop, value))
        return True

    def get(self, prop):
        return dict(self.settings).get(prop, 0)

    def getBackendName(self):
        return "TEST"

    def read(self):
        return (False, None) if self.fail else (True, np.zeros((360, 640, 3), dtype=np.uint8))

    def release(self):
        self.released = True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seconds": 0},
        {"seconds": 121},
        {"seconds": float("nan")},
        {"index": -1},
        {"index": True},
        {"index": 17},
    ],
)
def test_camera_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        CameraSource(**kwargs)


def test_camera_duration_recording_and_release(monkeypatch, tmp_path):
    camera = FakeCamera()
    # Keep the real decoder for verifying the resulting recording.
    decoder = cv2.VideoCapture
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    recording = tmp_path / "camera.avi"
    source = CameraSource(seconds=1, width=640, height=360, record=recording)
    source.start()
    source.thread.join(5)
    source.close()
    assert source.done and source.error is None and camera.released
    assert not source.thread.is_alive()
    assert len(camera.settings) == 3  # Fixed once, never controller-driven.
    rows = [
        json.loads(line)
        for line in recording.with_suffix(".captures.jsonl").read_text().splitlines()
    ]
    assert 2 <= len(rows) == source.count <= 32
    assert [r["capture_seq"] for r in rows] == list(range(1, source.count + 1))
    assert all(a["captured_mono_ns"] < b["captured_mono_ns"] for a, b in zip(rows, rows[1:]))
    video = decoder(str(recording))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == source.count
    video.release()
    assert json.loads(recording.with_suffix(".source.json").read_text())["audio"] is False
    with pytest.raises(ValueError, match="must be new"):
        CameraSource(record=recording)


@pytest.mark.parametrize("opened", [True, False])
def test_camera_failure_is_not_clean_eof_and_releases(monkeypatch, opened):
    camera = FakeCamera(fail=True, opened=opened)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    source = CameraSource(seconds=1)
    source.start()
    source.thread.join(5)
    source.close()
    assert source.done and source.error and camera.released
    assert source.count == 0


def test_file_eof_remains_clean(monkeypatch):
    camera = FakeCamera(fail=True)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    source = VideoSource("fixture.avi")
    source.start()
    source.thread.join(5)
    source.close()
    assert source.done and source.error is None and camera.released


def test_rtsp_source_keeps_credentials_out_of_status_and_reports_failure(monkeypatch):
    camera = FakeCamera(fail=True)
    calls = []

    def open_stream(url, backend):
        calls.append((url, backend))
        return camera

    monkeypatch.setattr(cv2, "VideoCapture", open_stream)
    source = RTSPSource("192.168.1.20", "camera-user", "secret@word", seconds=1)
    assert source.path == "rtsp://192.168.1.20:554/stream1"
    source.start()
    source.thread.join(5)
    source.close()
    assert calls == [
        (('rtsp' + '://camera-user:secret%40word@192.168.1.20:554/stream1'), cv2.CAP_FFMPEG)
    ]
    assert source.done and source.error == "RTSP camera read failed before the requested duration"
    assert camera.released
    assert "secret" not in source.error + source.path


def test_rtsp_capture_populates_ring_buffer(monkeypatch):
    camera = FakeCamera()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    source = RTSPSource("192.168.1.20", "user", "secret", seconds=1)
    source.start()
    source.thread.join(5)
    source.close()
    assert source.done and source.error is None and camera.released
    assert source.count > 0
    assert source.buffer_status()["buffer_frames"] == source.count
    latest, _, _, _ = source.snapshot()
    assert latest.jpeg.startswith(b"\xff\xd8")


def test_rtsp_nobuffer_applies_only_during_capture_open(monkeypatch):
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    seen = []

    def open_stream(_url, _backend):
        seen.append(os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"))
        return object()

    monkeypatch.setattr(cv2, "VideoCapture", open_stream)
    source = RTSPSource("192.168.1.20", "user", "secret", ffmpeg_nobuffer=True)
    source._open()
    assert seen == [RTSP_FFMPEG_NOBUFFER]
    assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ

    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp")
    with pytest.raises(RuntimeError, match="conflicting FFmpeg capture options"):
        source._open()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"host": "camera.local"},
        {"host": "192.168.1.999"},
        {"username": ""},
        {"password": ""},
        {"stream": 3},
        {"seconds": 0},
        {"seconds": 121},
        {"seconds": float("nan")},
    ],
)
def test_rtsp_rejects_invalid_input(kwargs):
    params = {"host": "192.168.1.20", "username": "user", "password": "secret"}
    params.update(kwargs)
    with pytest.raises(ValueError):
        RTSPSource(**params)


def test_rtsp_recording_preserves_frame_sequence_without_credentials(monkeypatch, tmp_path):
    camera = FakeCamera()
    decoder = cv2.VideoCapture
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    recording = tmp_path / "rtsp.avi"
    source = RTSPSource("192.168.0.81", "private-user", "private-password",
                        seconds=1, width=640, height=360, record=recording)
    source.start()
    source.thread.join(5)
    source.close()
    assert source.error is None and camera.released
    rows = recording.with_suffix(".captures.jsonl").read_text().splitlines()
    video = decoder(str(recording))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == len(rows) == source.count
    video.release()
    metadata = recording.with_suffix(".source.json").read_text()
    assert json.loads(metadata)["kind"] == "rtsp_camera"
    assert "private-user" not in metadata and "private-password" not in metadata
    with pytest.raises(ValueError, match="must be new"):
        RTSPSource("192.168.0.81", "u", "p", record=recording)
