"""Bounded, memory-only preview pipe. Never block the camera capture thread."""

import os
import queue
import struct
import threading
import time

import cv2

from care_vision.edge.buffer import fit_to_canvas

HEADER = struct.Struct('!IQQ')
MAX_PREVIEW_BYTES = 1_048_576


class PreviewWriter:
    def __init__(self, fd):
        self.fd = fd
        self.frames = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.last_publish = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def publish(self, image, capture):
        now = time.monotonic()
        if self.stopping.is_set() or now - self.last_publish < .5:
            return
        self.last_publish = now
        try:
            self.frames.put_nowait((image, capture.seq, capture.mono_ns))
        except queue.Full:
            pass  # Drop the preview, never the acquisition frame.

    def run(self):
        try:
            while not self.stopping.is_set():
                try:
                    image, seq, captured_ns = self.frames.get(timeout=.2)
                except queue.Empty:
                    continue
                ok, data = cv2.imencode('.jpg', fit_to_canvas(image, 640, 360),
                                        [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok or len(data) > MAX_PREVIEW_BYTES:
                    continue
                packet = memoryview(HEADER.pack(len(data), seq, captured_ns) + data.tobytes())
                while packet and not self.stopping.is_set():
                    written = os.write(self.fd, packet)
                    packet = packet[written:]
        except (OSError, ValueError):
            pass  # A closed viewer must not interrupt acquisition or inference.
        finally:
            os.close(self.fd)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=1)


class PreviewReader:
    def __init__(self, fd):
        self.fd = fd
        self.lock = threading.Lock()
        self.latest = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def exact(self, length):
        data = bytearray()
        while len(data) < length:
            part = os.read(self.fd, length - len(data))
            if not part:
                return None
            data.extend(part)
        return bytes(data)

    def run(self):
        try:
            while True:
                header = self.exact(HEADER.size)
                if header is None:
                    break
                length, seq, captured_ns = HEADER.unpack(header)
                if not 1 <= length <= MAX_PREVIEW_BYTES:
                    break
                payload = self.exact(length)
                if payload is None:
                    break
                if payload.startswith(b'\xff\xd8\xff'):
                    with self.lock:
                        self.latest = (payload, seq, captured_ns)
        except OSError:
            pass
        finally:
            os.close(self.fd)

    def frame(self):
        with self.lock:
            if self.latest and 0 <= time.monotonic_ns() - self.latest[2] < 3_000_000_000:
                return self.latest
            return None

    def clear(self):
        with self.lock:
            self.latest = None

    def close(self):
        # Call after the owned process group closes its inherited write descriptor.
        self.thread.join(timeout=1)
        self.clear()
