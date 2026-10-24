import json
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from care_vision.fixture import make_video
from care_vision.lab.live import LocalPreviewCamera, run_camera
from care_vision.lab.preview import HEADER, PreviewReader, PreviewWriter
from care_vision.lab.session import LabSessions


def test_camera_configuration_never_accepts_remote_destination_or_invalid_values(tmp_path):
    sessions = LabSessions(tmp_path)
    for options in ({'source': 'aws'}, {'source': []}, {'source': 'camera', 'camera_index': -1},
                    {'source': 'camera', 'seconds': 121}, {'preview': 'true'}):
        with pytest.raises(ValueError):
            sessions.start(**options)
    assert sessions.process is None
    state = sessions.snapshot()
    assert state['processing_location'] == 'THIS_COMPUTER'
    assert state['aws_connected'] is False
    assert not state['source']['preview_enabled']
    assert sessions.preview() is None


def test_preview_is_bounded_memory_only_and_expires(monkeypatch):
    read_fd, write_fd = os.pipe()
    reader, writer = PreviewReader(read_fd), PreviewWriter(write_fd)
    now = time.monotonic_ns()
    try:
        writer.publish(np.zeros((360, 640, 3), dtype=np.uint8),
                       SimpleNamespace(seq=4, mono_ns=now))
        deadline = time.monotonic() + 3
        while reader.frame() is None and time.monotonic() < deadline:
            time.sleep(.02)
        frame = reader.frame()
        assert frame and frame[0].startswith(b'\xff\xd8\xff') and frame[1] == 4
        monkeypatch.setattr(time, 'monotonic_ns', lambda: now + 4_000_000_000)
        assert reader.frame() is None
    finally:
        writer.close()
        reader.close()
    assert reader.latest is None and not reader.thread.is_alive()


def test_preview_rejects_oversized_packets():
    read_fd, write_fd = os.pipe()
    reader = PreviewReader(read_fd)
    os.write(write_fd, HEADER.pack(2_000_000, 1, time.monotonic_ns()))
    os.close(write_fd)
    reader.thread.join(timeout=2)
    assert reader.frame() is None and not reader.thread.is_alive()


def test_camera_preview_is_opt_in_and_is_cleared_on_stop(tmp_path):
    sessions = LabSessions(tmp_path)
    sessions.source_kind = 'camera'
    sessions.preview_enabled = False
    sessions.process = SimpleNamespace(poll=lambda: None)
    sessions.preview_reader = SimpleNamespace(
        frame=lambda: (b'\xff\xd8\xffdata', 2, time.monotonic_ns()))
    assert sessions.preview() is None
    sessions.preview_enabled = True
    assert sessions.preview().startswith(b'\xff\xd8\xff')
    sessions.process = SimpleNamespace(poll=lambda: 0)
    assert sessions.preview() is None


def test_camera_worker_uses_generated_test_device_without_recording(monkeypatch, tmp_path):
    # A virtual input exercises the real worker/Edge/local vision path. No hardware opens.
    fixture = make_video(tmp_path / 'generated-test-device.avi', seconds=6)
    monkeypatch.setattr(LocalPreviewCamera, '_open', lambda self: cv2.VideoCapture(str(fixture)))
    run = tmp_path / 'run'
    run_camera(run, index=0, seconds=3)
    report = json.loads((run / 'report.json').read_text())
    assert report['kind'] == 'LOCAL_CAMERA_DIAGNOSTIC' and report['camera_completed']
    assert report['frames_captured'] > 20 and report['cool_verified'] is False
    assert report['images_saved'] is False and report['audit_performed'] is False
    assert 'passed' not in report
    assert not any(p.suffix in {'.jpg', '.png', '.avi', '.mp4'} for p in run.rglob('*'))
    records = [json.loads(line) for line in (run / 'cloud.jsonl').read_text().splitlines()]
    assert any(r['event'] == 'observation' for r in records)
    ready = next(r for r in records if r['event'] == 'server_ready')
    assert ready['phase'] == 1 and ready['cool_verified'] is False
    with pytest.raises(ValueError, match='new or empty'):
        run_camera(run, index=0, seconds=3)


def test_camera_capture_never_encodes_preview_on_the_capture_thread(monkeypatch):
    camera = LocalPreviewCamera(0, 1)
    monkeypatch.setattr(cv2, 'imencode', lambda *args: pytest.fail('Unexpected preview encoding'))
    camera._record(np.zeros((3, 3, 3), dtype=np.uint8), SimpleNamespace(seq=1, mono_ns=1))
    assert camera.record is None and camera.writer is None


def test_session_launch_is_local_and_preview_is_opt_in(monkeypatch, tmp_path):
    calls = []
    process = SimpleNamespace(poll=lambda: None, returncode=None)
    def spawn(argv, **options):
        calls.append((argv, options))
        return process
    monkeypatch.setattr('care_vision.lab.session.subprocess.Popen', spawn)
    session = LabSessions(tmp_path)
    session.start(source='camera', camera_index=1, seconds=30, preview=False)
    argv, options = calls[0]
    assert 'care_vision.lab.live' in argv and '--camera' in argv
    assert '--preview-fd' not in argv and '--url' not in argv and '--host' not in argv
    assert options['start_new_session'] and not options['pass_fds']
    state = session.snapshot()
    assert state['source'] == {'kind': 'camera', 'camera_index': 1, 'preview_enabled': False}
    assert state['mode'] == 'LOCAL_CAMERA_NOT_COOL' and state['aws_connected'] is False
    assert not state['preview_available']
    process.poll = lambda: 0
    process.returncode = 0
    session.stop()
