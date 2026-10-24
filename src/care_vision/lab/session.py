"""Reuse the bounded demo worker; expose only its synthetic source preview."""

import json

import cv2

from care_vision.judge.session import Sessions
from care_vision.judge.state import LogTail
from care_vision.lab.telemetry import EdgeTelemetry, VisionTelemetry


class LabSessions(Sessions):
    def __init__(self, root):
        super().__init__(root, synthetic=True, seconds=90)
        self.edge = EdgeTelemetry()
        self.edge_tail = None
        self.preview_cache = None

    def start(self):
        with self.lock:
            super().start()
            self.projection = VisionTelemetry()
            self.tail.projection = self.projection
            self.edge = EdgeTelemetry()
            self.edge_tail = LogTail(self.tail.path.with_name('edge.jsonl'), self.edge)
            self.preview_cache = None

    def snapshot(self, operator=False):
        with self.lock:
            result = super().snapshot(operator=False)
            if self.edge_tail:
                self.edge_tail.poll()
            running = result['session_state'] in {'STARTING', 'RUNNING'}
            result['edge'] = self.edge.snapshot(running)
            if isinstance(self.projection, VisionTelemetry):
                result.update(self.projection.lab_snapshot())
            else:
                result.update(observation={}, commands=[], transitions=[])
            result['image_policy'] = 'OWN_GENERATED_SOURCE_PREVIEW_ONLY'
            result['report'] = None
            if self.tail and result['session_state'] == 'COMPLETED':
                try:
                    report = json.loads(self.tail.path.with_name('report.json').read_text())
                    result['report'] = {
                        key: report.get(key) for key in
                        ('passed', 'frames_sent', 'frames_received',
                         'observations', 'cool_verified')
                    }
                    result['report']['buffer_verified'] = (
                        report.get('temporal_buffer', {}).get('verified') is True
                    )
                except (OSError, ValueError, TypeError):
                    pass
            return result

    def preview(self):
        with self.lock:
            seq = self.edge.last_capture_seq
            if not self.tail or type(seq) not in {int, float} or not 1 <= seq <= 1020:
                return None
            seq = int(seq)
            if self.preview_cache and self.preview_cache[0] == seq:
                return self.preview_cache[1]
            # Fixed file owned by this generated run; no uploads or arbitrary paths.
            capture = cv2.VideoCapture(str(self.tail.path.with_name('demo.avi')))
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, seq - 1)
                ok, frame = capture.read()
                if not ok:
                    return None
                ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    return None
                payload = data.tobytes()
                self.preview_cache = seq, payload
                return payload
            finally:
                capture.release()
