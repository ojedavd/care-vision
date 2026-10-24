"""Own bounded synthetic/camera diagnostics; never connect to a shared AWS host."""

import json
import os
import subprocess
import time
from uuid import uuid4

import cv2

from care_vision.judge.session import Sessions
from care_vision.judge.state import LogTail
from care_vision.lab.preview import PreviewReader
from care_vision.lab.telemetry import EdgeTelemetry, VisionTelemetry


class LabSessions(Sessions):
    def __init__(self, root):
        super().__init__(root, synthetic=True, seconds=90)
        self.edge = EdgeTelemetry()
        self.edge_tail = None
        self.preview_cache = None
        self.preview_reader = None
        self.source_kind = 'synthetic'
        self.camera_index = None
        self.preview_enabled = False

    def start(self, source='synthetic', camera_index=0, seconds=45, preview=False):
        if not isinstance(source, str) or source not in {'synthetic', 'camera'}:
            raise ValueError(
                'Only synthetic or local camera sources are supported; no AWS targets.')
        if type(camera_index) is not int or not 0 <= camera_index <= 16:
            raise ValueError('Camera index must be an integer from 0 to 16.')
        if type(seconds) is not int or not 1 <= seconds <= 120 or type(preview) is not bool:
            raise ValueError('Duration must be 1..120 seconds and preview must be a boolean.')
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError('session already running')
            if self.preview_reader:
                self.preview_reader.close()
                self.preview_reader = None
            if self.output:
                self.output.close()
            self.source_kind = source
            self.camera_index = camera_index if source == 'camera' else None
            self.preview_enabled = source == 'camera' and preview
            if source == 'synthetic':
                self.seconds = 90
                super().start()
            else:
                self._start_camera(camera_index, seconds, preview)
            self.projection = VisionTelemetry()
            self.tail.projection = self.projection
            self.edge = EdgeTelemetry()
            self.edge_tail = LogTail(self.tail.path.with_name('edge.jsonl'), self.edge)
            self.preview_cache = None

    def _start_camera(self, index, seconds, preview):
        run = self.root / uuid4().hex
        run.mkdir(mode=0o700)
        self.tail = LogTail(run / 'demo/cloud.jsonl', self.projection)
        self.seconds = seconds + 20
        self.output = (run / 'worker.stderr').open('wb')
        read_fd, write_fd = os.pipe() if preview else (None, None)
        argv = [self.python, '-u', '-m', 'care_vision.lab.live',
                '--output', str(run / 'demo'), '--camera', str(index), '--seconds', str(seconds)]
        if preview:
            argv.extend(('--preview-fd', str(write_fd)))
        env = {k: v for k, v in os.environ.items() if not k.startswith('CARE_JUDGE_')}
        try:
            self.process = subprocess.Popen(
                argv, env=env, stdin=subprocess.DEVNULL, stdout=self.output, stderr=self.output,
                pass_fds=(write_fd,) if preview else (), start_new_session=True,
            )
        except Exception:
            if read_fd is not None:
                os.close(read_fd)
            self.output.close()
            self.state = 'FAILED'
            raise
        finally:
            if write_fd is not None:
                os.close(write_fd)
        if read_fd is not None:
            self.preview_reader = PreviewReader(read_fd)
        self.started = time.monotonic()
        self.state = 'STARTING'

    def stop(self):
        with self.lock:
            super().stop()
            if self.preview_reader:
                self.preview_reader.close()
                self.preview_reader = None

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
            result['source'] = {
                'kind': self.source_kind, 'camera_index': self.camera_index,
                'preview_enabled': self.preview_enabled,
            }
            result['processing_location'] = 'THIS_COMPUTER'
            result['aws_connected'] = False
            if self.source_kind == 'camera':
                result['mode'] = 'LOCAL_CAMERA_NOT_COOL'
                result['image_policy'] = 'OPT_IN_LOCAL_CAMERA_MEMORY_PREVIEW'
                frame = self.preview_reader.frame() if self.preview_reader and running else None
                result['preview_available'] = frame is not None
                result['preview_capture_seq'] = frame[1] if frame else None
                if self.preview_reader and not running:
                    self.preview_reader.clear()
            else:
                result['image_policy'] = 'OWN_GENERATED_SOURCE_PREVIEW_ONLY'
                result['preview_available'] = self.edge.last_capture_seq is not None

            result['report'] = None
            if self.tail and result['session_state'] == 'COMPLETED':
                try:
                    report = json.loads(self.tail.path.with_name('report.json').read_text())
                    result['report'] = {
                        key: report.get(key) for key in
                        ('passed', 'frames_sent', 'frames_received',
                         'observations', 'cool_verified')
                    }
                    if report.get('kind') == 'LOCAL_CAMERA_DIAGNOSTIC':
                        result['report'] = {
                            'kind': report['kind'], 'camera_completed': True,
                            'audit_performed': False, 'cool_verified': False,
                        }
                    result['report']['buffer_verified'] = (
                        report.get('temporal_buffer', {}).get('verified') is True
                    )
                except (OSError, ValueError, TypeError):
                    pass
            return result

    def preview(self):
        with self.lock:
            if self.source_kind == 'camera':
                running = self.process is not None and self.process.poll() is None
                if not running or not self.preview_enabled or not self.preview_reader:
                    return None
                frame = self.preview_reader.frame()
                return frame[0] if frame else None
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
