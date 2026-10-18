"""File or bounded camera acquisition on a capture thread; no visual inference."""

import json
import os
import platform
import threading
import time
from ipaddress import IPv4Address
from pathlib import Path
from urllib.parse import quote

import cv2

from care_vision.common.contracts import utc
from care_vision.edge.buffer import Capture, RingBuffer, encode, fit_to_canvas
from care_vision.edge.spool import EncryptedSpool

RTSP_FFMPEG_NOBUFFER = "rtsp_transport;tcp|fflags;nobuffer"


class VideoSource:
    def __init__(
        self, path: str, fps=30.0, width=640, height=360, spool: EncryptedSpool | None = None
    ):
        if not 0 < fps <= 30:
            raise ValueError("capture FPS must be in (0, 30]")
        if (width, height) not in {(640, 360), (1280, 720)}:
            raise ValueError("unsupported capture resolution")
        self.path, self.fps = path, fps
        self.width, self.height = width, height
        self.spool = spool
        self.boot_id = spool.boot_id if spool else None
        self.offline = True
        self.latest: Capture | None = None
        self.buffer = RingBuffer()
        self.count = spool.last_seq if spool else 0
        self.done = False
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture, daemon=True)

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.latest, self.count, self.done, self.error

    def recent(self, start_ns, end_ns, fps):
        with self.lock:
            status = self.buffer.status()
            ring = self.buffer.select(start_ns, end_ns, fps)
            if not self.spool:
                return ring, status
            archived = self.spool.select(start_ns, end_ns, fps)
            selected = {frame.seq: frame for frame in (*archived, *ring)}
            status["archive_sparse"] = bool(
                archived
                and (
                    status["buffer_oldest_mono_ns"] is None
                    or start_ns < status["buffer_oldest_mono_ns"]
                )
            )
            return tuple(sorted(selected.values(), key=lambda frame: frame.mono_ns)), status

    def buffer_status(self):
        with self.lock:
            return {
                **self.buffer.status(),
                **(self.spool.status() if self.spool else {}),
            }

    def acknowledge_captures(self, seqs):
        with self.lock:
            return self.spool.acknowledge(seqs) if self.spool else 0

    def set_offline(self, offline: bool):
        with self.lock:
            self.offline = offline

    def recovery_window(self, first_seq, last_seq, context_from_ns=None):
        """Return retained outage frames and optional unconfirmed pre-outage context."""
        with self.lock:
            available = [
                frame
                for frame in self.buffer.frames
                if frame.seq <= last_seq
                and (
                    frame.mono_ns >= context_from_ns
                    if context_from_ns is not None
                    else frame.seq >= first_seq
                )
            ]
            offline_total = max(0, last_seq - first_seq + 1)
            if not available:
                return {
                    "recovery_window_available": False,
                    "offline_frames_retained_at_reconnect": 0,
                    "offline_frames_lost_at_reconnect": offline_total,
                }
            last_ns = available[-1].mono_ns
            available = [frame for frame in available if frame.mono_ns > last_ns - 10_000_000_000]
            offline_retained = sum(frame.seq >= first_seq for frame in available)
            context_evicted = (
                context_from_ns is not None and self.buffer.frames[0].mono_ns > context_from_ns
            )
            return {
                "recovery_window_available": True,
                "offline_frames_retained_at_reconnect": offline_retained,
                "offline_frames_lost_at_reconnect": offline_total - offline_retained,
                "recovery_from_edge_mono_ns": available[0].mono_ns,
                "recovery_to_edge_mono_ns": last_ns + 1,
                "recovery_available_frames": len(available),
                "recovery_truncated": (
                    offline_retained < last_seq - first_seq + 1 or context_evicted
                ),
                "recovery_includes_unconfirmed_context": context_from_ns is not None,
            }

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        if self.spool:
            self.spool.close()

    def _capture(self):
        video = None
        try:
            video = self._open()
            if not video.isOpened():
                raise RuntimeError(f"cannot open video: {self.path}")
            deadline = time.monotonic()
            started = deadline
            while not self.stop_event.is_set():
                if self._finished(started):
                    break
                if self.stop_event.wait(max(0, deadline - time.monotonic())):
                    break
                ok, image = video.read()
                if not ok:
                    self._read_failed()
                    break
                captured_ns, captured_utc = time.monotonic_ns(), utc()
                capture = Capture(
                    self.count + 1,
                    captured_ns,
                    captured_utc,
                    encode(image, self.width, self.height),
                )
                self._record(image, capture)
                with self.lock:
                    if self.spool:
                        self.spool.append(capture, retain=self.offline)
                    self.buffer.append(capture)
                    self.latest = capture
                    self.count = capture.seq
                # Don't burst through the file when the machine was suspended or stalled.
                deadline = max(deadline + 1 / self.fps, time.monotonic())
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if video is not None:
                video.release()
            self._finish_recording()
            with self.lock:
                self.done = True

    def _open(self):
        return cv2.VideoCapture(self.path)

    def _finished(self, started):
        return False

    def _read_failed(self):
        pass

    def _record(self, image, capture):
        pass

    def _finish_recording(self):
        pass


class CameraSource(VideoSource):
    """Fixed capture configuration; the agent changes transmission, not hardware FPS."""

    def __init__(self, index=0, seconds=45, fps=30, width=1280, height=720, record=None):
        if type(index) is not int or not 0 <= index <= 16:
            raise ValueError("camera index must be an integer from 0 to 16")
        if not 1 <= seconds <= 120:
            raise ValueError("camera duration must be between 1 and 120 seconds")
        super().__init__(f"camera:{index}", fps, width, height)
        self.index, self.seconds = index, seconds
        self.record = Path(record) if record else None
        if self.record and any(
            p.exists()
            for p in (
                self.record,
                self.record.with_suffix(".captures.jsonl"),
                self.record.with_suffix(".source.json"),
            )
        ):
            raise ValueError("camera recording must be new")
        self.writer = self.capture_log = None

    def _open(self):
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        camera = cv2.VideoCapture(self.index, backend)
        try:
            if not camera.isOpened():
                raise RuntimeError("camera unavailable; check macOS camera permission and device")
            applied = {
                "width": camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width),
                "height": camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                "fps": camera.set(cv2.CAP_PROP_FPS, self.fps),
            }
            if self.record:
                self.record.parent.mkdir(parents=True, exist_ok=True)
                self.writer = cv2.VideoWriter(
                    str(self.record),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    self.fps,
                    (self.width, self.height),
                )
                if not self.writer.isOpened():
                    raise RuntimeError("cannot open camera recording")
                self.capture_log = self.record.with_suffix(".captures.jsonl").open("x")
                self.record.with_suffix(".source.json").write_text(
                    json.dumps(
                        {
                            "kind": "physical_camera",
                            "camera_index": self.index,
                            "backend": camera.getBackendName(),
                            "requested_fps": self.fps,
                            "reported_fps": camera.get(cv2.CAP_PROP_FPS),
                            "reported_width": camera.get(cv2.CAP_PROP_FRAME_WIDTH),
                            "reported_height": camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            "configuration_applied": applied,
                            "duration_seconds": self.seconds,
                            "audio": False,
                            "recording_fps_is_playback_rate": True,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            return camera
        except Exception:
            camera.release()
            raise

    def _finished(self, started):
        return time.monotonic() - started >= self.seconds

    def _read_failed(self):
        raise RuntimeError("camera read failed before the requested duration")

    def _record(self, image, capture):
        if self.writer:
            self.writer.write(fit_to_canvas(image, self.width, self.height))
            self.capture_log.write(
                json.dumps(
                    {
                        "capture_seq": capture.seq,
                        "captured_mono_ns": capture.mono_ns,
                        "captured_at_utc": capture.at_utc,
                    }
                )
                + "\n"
            )
            self.capture_log.flush()

    def _finish_recording(self):
        if self.writer:
            self.writer.release()
        if self.capture_log:
            self.capture_log.close()


class RTSPSource(VideoSource):
    """Bounded IP-camera acquisition; sampling and inference remain unchanged."""

    def __init__(
        self, host, username, password, stream=1, seconds=45, fps=30,
        width=1280, height=720, ffmpeg_nobuffer=False, record=None,
    ):
        try:
            host = str(IPv4Address(host))
        except ValueError as exc:
            raise ValueError("RTSP host must be the camera's IPv4 address") from exc
        if not username or not password:
            raise ValueError("RTSP camera account credentials are required")
        if stream not in (1, 2):
            raise ValueError("RTSP stream must be 1 or 2")
        if not 1 <= seconds <= 120:
            raise ValueError("RTSP duration must be between 1 and 120 seconds")
        super().__init__(f"rtsp://{host}:554/stream{stream}", fps, width, height)
        self.seconds = seconds
        self.ffmpeg_nobuffer = ffmpeg_nobuffer
        self.record = Path(record) if record else None
        if self.record and any(p.exists() for p in (
            self.record, self.record.with_suffix('.captures.jsonl'),
            self.record.with_suffix('.source.json'),
        )):
            raise ValueError('RTSP recording must be new')
        self.writer = self.capture_log = None
        self._url = (
            f"rtsp://{quote(username, safe='')}:{quote(password, safe='')}"
            f"@{host}:554/stream{stream}"
        )

    _record = CameraSource._record
    _finish_recording = CameraSource._finish_recording

    def _open(self):
        camera = self._open_rtsp()
        if not self.record:
            return camera
        try:
            if not camera.isOpened():
                raise RuntimeError('RTSP camera unavailable')
            if self.record:
                self.record.parent.mkdir(parents=True, exist_ok=True)
                self.writer = cv2.VideoWriter(
                    str(self.record), cv2.VideoWriter_fourcc(*'MJPG'),
                    self.fps, (self.width, self.height),
                )
                if not self.writer.isOpened():
                    raise RuntimeError('cannot open RTSP recording')
                self.capture_log = self.record.with_suffix('.captures.jsonl').open('x')
                self.record.with_suffix('.source.json').write_text(json.dumps({
                    'kind': 'rtsp_camera', 'duration_seconds': self.seconds,
                    'requested_fps': self.fps, 'recording_fps_is_playback_rate': True,
                    'width': self.width, 'height': self.height, 'audio': False,
                }, indent=2) + '\n')
            return camera
        except Exception:
            camera.release()
            raise

    def _open_rtsp(self):
        if not self.ffmpeg_nobuffer:
            return cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        prior = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        if prior and prior != RTSP_FFMPEG_NOBUFFER:
            raise RuntimeError("conflicting FFmpeg capture options for RTSP")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_FFMPEG_NOBUFFER
        try:
            return cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        finally:
            if prior is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prior

    def _finished(self, started):
        return time.monotonic() - started >= self.seconds

    def _read_failed(self):
        raise RuntimeError("RTSP camera read failed before the requested duration")
