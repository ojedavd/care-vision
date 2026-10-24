"""Own bounded local or explicitly configured evaluator Graviton sessions."""

import json
import os
import subprocess
import time
from uuid import uuid4

import cv2

from care_vision.judge.session import Sessions
from care_vision.judge.state import LogTail
from care_vision.lab.cloud_profile import load_profile
from care_vision.lab.models import available, default_directory, model_paths
from care_vision.lab.preview import PreviewReader
from care_vision.lab.telemetry import EdgeTelemetry, VisionTelemetry


class LabSessions(Sessions):
    def __init__(self, root, *, models=None, cloud_profile=None, video=None):
        super().__init__(root, synthetic=True, seconds=90)
        self.edge = EdgeTelemetry()
        self.edge_tail = None
        self.preview_cache = None
        self.preview_reader = None
        self.source_kind = 'synthetic'
        self.camera_index = None
        self.preview_enabled = False
        self.models = models or default_directory()
        self.cloud_profile = cloud_profile
        self.video = video
        self.target, self.pipeline = 'local', 'motion'
        self.input_seconds = 45

    def start(self, source='synthetic', camera_index=0, seconds=45, preview=False,
              target='local', pipeline='motion'):
        if not isinstance(source, str) or source not in {'synthetic', 'camera', 'video'}:
            raise ValueError(
                'Choose generated input, a local camera or a configured local video.')
        if (not isinstance(target, str) or target not in {'local', 'graviton'}
                or not isinstance(pipeline, str) or pipeline not in {'motion', 'full'}):
            raise ValueError('Choose a supported processing mode.')
        if source == 'video' and not self.video:
            raise ValueError('Launch care-lab --video with an explicit local file first.')
        if target == 'graviton':
            if not self.cloud_profile or pipeline != 'full':
                raise ValueError('My AWS Graviton requires an explicit private cloud profile.')
            load_profile(self.cloud_profile)
        elif pipeline == 'full':
            model_paths(self.models)
        if pipeline == 'full' and source == 'synthetic':
            raise ValueError('Choose a local camera or configured video for the full pipeline.')
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
            self.target, self.pipeline = target, pipeline
            self.input_seconds = seconds
            self.camera_index = camera_index if source == 'camera' else None
            self.preview_enabled = source in {'camera', 'video'} and preview
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
        self.seconds = seconds + 100  # Runtime checks, capture, bounded drain and SSH cleanup.
        self.output = (run / 'worker.stderr').open('wb')
        read_fd, write_fd = os.pipe() if preview else (None, None)
        argv = [self.python, '-u', '-m', 'care_vision.lab.live',
                '--output', str(run / 'demo'), '--camera', str(index), '--seconds', str(seconds)]
        if self.pipeline == 'full':
            argv.extend(('--pipeline', 'full', '--target', self.target))
            if self.target == 'local':
                argv.extend(('--models', str(self.models)))
            else:
                argv.extend(('--cloud-profile', str(self.cloud_profile)))
        if self.source_kind == 'video':
            argv.extend(('--video', str(self.video)))
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
            result['target'], result['pipeline'] = self.target, self.pipeline
            result['input_seconds'] = self.input_seconds
            result['capabilities'] = {
                'local_models_present': available(self.models),
                'graviton_profile_configured': self.cloud_profile is not None,
                'local_video_configured': self.video is not None,
            }
            if self.source_kind in {'camera', 'video'}:
                result['mode'] = 'LOCAL_CAMERA_NOT_COOL'
                result['image_policy'] = 'OPT_IN_LOCAL_CAMERA_MEMORY_PREVIEW'
                frame = self.preview_reader.frame() if self.preview_reader and running else None
                result['preview_available'] = frame is not None
                result['preview_capture_seq'] = frame[1] if frame else None
                if self.preview_reader and not running:
                    self.preview_reader.clear()
                if self.pipeline == 'full':
                    result['mode'] = ('MY_GRAVITON_FULL_PIPELINE' if self.target == 'graviton'
                                      else 'LOCAL_FULL_PIPELINE_NOT_COOL')
                    result['processing_location'] = ('AWS_GRAVITON' if self.target == 'graviton'
                                                     else 'LOCAL_SERVICE')
                    result['aws_connected'] = bool(self.target == 'graviton' and running
                                                  and result['runtime']['cool_verified']
                                                  and result['edge_connected'])
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
                    if report.get('kind') in {'LOCAL_CAMERA_DIAGNOSTIC', 'LOCAL_FULL_PIPELINE',
                                               'GRAVITON_FULL_PIPELINE'}:
                        result['report'] = {
                            'kind': report['kind'], 'camera_completed': True,
                            'audit_performed': False,
                            'cool_verified': result['runtime']['cool_verified'],
                        }
                    result['report']['buffer_verified'] = (
                        report.get('temporal_buffer', {}).get('verified') is True
                    )
                except (OSError, ValueError, TypeError):
                    pass
            result['failure'] = (
                'Run failed. Inspect the private worker/SSH log; verify runtime, models and input.'
                if result['session_state'] == 'FAILED' else None)
            return result

    def preview(self):
        with self.lock:
            if self.source_kind in {'camera', 'video'}:
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
