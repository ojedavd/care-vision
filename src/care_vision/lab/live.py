"""Bounded local camera diagnostic. No AWS, models or image/video persistence."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from care_vision.common.metrics import EventLog
from care_vision.common.resources import monitor_resources
from care_vision.edge.runtime import Edge
from care_vision.edge.source import CameraSource
from care_vision.lab.preview import PreviewWriter


def read_records(path):
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except ValueError:
        return []  # A writer may still be appending its final newline.


class LocalPreviewCamera(CameraSource):
    def __init__(self, index, seconds, preview=None):
        super().__init__(index=index, seconds=seconds, fps=30, width=640, height=360)
        self.preview = preview

    def _record(self, image, capture):
        if self.preview:
            self.preview.publish(image, capture)


def run_camera(output, index, seconds, preview_fd=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Camera run output must be new or empty.')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    cloud_log = output / 'cloud.jsonl'
    edge_log = EventLog('edge', output / 'edge.jsonl')
    preview = PreviewWriter(preview_fd) if preview_fd is not None else None
    cloud = None
    cloud_err = (output / 'cloud.stderr').open('wb')
    try:
        # Destination is a freshly created local worker, never a caller-provided AWS URL.
        cloud = subprocess.Popen(
            [sys.executable, '-m', 'care_vision.cloud', '--phase', '1', '--port', '0',
             '--log', str(cloud_log), '--run-seconds', str(seconds + 20)],
            stdin=subprocess.DEVNULL, stdout=cloud_err, stderr=cloud_err, close_fds=True,
        )
        deadline = time.monotonic() + 10
        port = None
        while time.monotonic() < deadline and cloud.poll() is None:
            ready = [r for r in read_records(cloud_log) if r['event'] == 'server_ready']
            if ready:
                port = ready[-1]['port']
                break
            time.sleep(.05)
        if port is None:
            raise RuntimeError('Local vision worker did not become ready.')
        source = LocalPreviewCamera(index, seconds, preview)

        async def run():
            async with monitor_resources(edge_log):
                await Edge(source, f'ws://127.0.0.1:{port}', edge_log).run()

        asyncio.run(run())
        if source.error:
            raise RuntimeError('Camera acquisition failed; check device and OS permission.')
        report = {
            'kind': 'LOCAL_CAMERA_DIAGNOSTIC', 'camera_completed': True,
            'frames_captured': source.count, 'cool_verified': False,
            'audit_performed': False, 'images_saved': False,
        }
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        edge_log.close()
        if preview:
            preview.close()
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
    args = parser.parse_args()
    if not 0 <= args.camera <= 16 or not 1 <= args.seconds <= 120:
        parser.error('Camera index must be 0..16 and duration 1..120 seconds.')
    os.umask(0o077)
    run_camera(args.output, args.camera, args.seconds, args.preview_fd)


if __name__ == '__main__':
    main()
