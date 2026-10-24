"""Bounded Edge input with explicit local or evaluator-owned Graviton processing."""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2

from care_vision.common.metrics import EventLog
from care_vision.common.resources import monitor_resources
from care_vision.edge.runtime import Edge
from care_vision.edge.source import CameraSource, VideoSource
from care_vision.lab.models import model_paths
from care_vision.lab.preview import PreviewWriter
from care_vision.lab.remote import RemoteService


def read_records(path):
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except ValueError:
        return []  # A writer may still be appending its final newline.


class LocalPreviewCamera(CameraSource):
    def __init__(self, index, seconds, preview=None, *, cloud=False):
        super().__init__(index=index, seconds=seconds, fps=25 if cloud else 30,
                         width=1280 if cloud else 640, height=720 if cloud else 360)
        self.preview = preview

    def _record(self, image, capture):
        if self.preview:
            self.preview.publish(image, capture)


class LocalPreviewVideo(VideoSource):
    def __init__(self, path, seconds, preview=None, *, cloud=False):
        probe = cv2.VideoCapture(str(path))
        try:
            fps = probe.get(cv2.CAP_PROP_FPS)
        finally:
            probe.release()
        super().__init__(str(path), fps=fps if 0 < fps <= 30 else 30,
                         width=1280 if cloud else 640, height=720 if cloud else 360)
        self.seconds, self.preview = seconds, preview

    def _finished(self, started):
        return time.monotonic() - started >= self.seconds

    def _record(self, image, capture):
        if self.preview:
            self.preview.publish(image, capture)


def run_camera(output, index, seconds, preview_fd=None, *, target='local', pipeline='motion',
               models=None, cloud_profile=None, video=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Camera run output must be new or empty.')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    cloud_log = output / 'cloud.jsonl'
    edge_log = EventLog('edge', output / 'edge.jsonl')
    preview = PreviewWriter(preview_fd) if preview_fd is not None else None
    cloud = None
    remote = None
    cloud_err = (output / 'cloud.stderr').open('wb')
    try:
        if target not in {'local', 'graviton'} or pipeline not in {'motion', 'full'}:
            raise ValueError('Unsupported execution mode.')
        if target == 'graviton':
            if pipeline != 'full' or not cloud_profile:
                raise ValueError('Graviton requires a private profile and full pipeline.')
            remote = RemoteService(cloud_profile, output, seconds)
            url = remote.start()
        else:
            if pipeline == 'full':
                model_paths(models)
                command = [sys.executable, '-m', 'care_vision.lab.reference',
                           '--models', str(models), '--log', str(cloud_log),
                           '--seconds', str(seconds + 45)]
            else:
                command = [sys.executable, '-m', 'care_vision.cloud', '--phase', '1',
                           '--port', '0', '--log', str(cloud_log),
                           '--run-seconds', str(seconds + 20)]
            cloud = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=cloud_err,
                                     stderr=cloud_err, close_fds=True)
            deadline = time.monotonic() + (45 if pipeline == 'full' else 10)
            port = None
            while time.monotonic() < deadline and cloud.poll() is None:
                ready = [r for r in read_records(cloud_log) if r['event'] == 'server_ready']
                if ready:
                    port = ready[-1]['port']
                    break
                time.sleep(.05)
            if port is None:
                raise RuntimeError('Local vision worker did not become ready.')
            url = f'ws://127.0.0.1:{port}'
        # All model/runtime/source checks finish before the camera capture thread starts.
        source = (LocalPreviewVideo(video, seconds, preview, cloud=pipeline == 'full') if video
                  else LocalPreviewCamera(index, seconds, preview, cloud=pipeline == 'full'))

        async def run():
            async with monitor_resources(edge_log):
                await Edge(source, url, edge_log).run()

        asyncio.run(run())
        if source.error:
            raise RuntimeError('Input acquisition failed; check source and OS permission.')
        report = {
            'kind': ('GRAVITON_FULL_PIPELINE' if target == 'graviton' else
                     'LOCAL_FULL_PIPELINE' if pipeline == 'full' else 'LOCAL_CAMERA_DIAGNOSTIC'),
            'camera_completed': True, 'frames_captured': source.count,
            'cool_verified': bool(remote and remote.ready['cool_verified']),
            'audit_performed': False, 'images_saved': False,
        }
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        edge_log.close()
        if preview:
            preview.close()
        if remote:
            remote.close()
        if cloud and cloud.poll() is None:
            cloud.terminate()
            try:
                cloud.wait(timeout=3)
            except subprocess.TimeoutExpired:
                cloud.kill()
                cloud.wait(timeout=3)
        cloud_err.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--camera', type=int, required=True)
    parser.add_argument('--seconds', type=int, default=45)
    parser.add_argument('--preview-fd', type=int)
    parser.add_argument('--target', choices=['local', 'graviton'], default='local')
    parser.add_argument('--pipeline', choices=['motion', 'full'], default='motion')
    parser.add_argument('--models', type=Path)
    parser.add_argument('--cloud-profile', type=Path)
    parser.add_argument('--video', type=Path)
    args = parser.parse_args()
    if not 0 <= args.camera <= 16 or not 1 <= args.seconds <= 120:
        parser.error('Camera index must be 0..16 and duration 1..120 seconds.')
    os.umask(0o077)
    def terminate(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    run_camera(args.output, args.camera, args.seconds, args.preview_fd,
               target=args.target, pipeline=args.pipeline, models=args.models,
               cloud_profile=args.cloud_profile, video=args.video)


if __name__ == '__main__':
    main()
