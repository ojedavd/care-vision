"""Own one authenticated SSH worker and one loopback tunnel, never shared AWS state."""

import json
import os
import select
import shlex
import socket
import subprocess
import time
from pathlib import Path

from care_vision.lab.cloud_profile import load_profile, source_manifest, ssh_args


def records(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def remote_command(profile):
    # OpenSSH passes a command through a remote shell: quote every argument once.
    return shlex.join([
        'env', 'PYTHONNOUSERSITE=1',
        'PYTHONPATH=' + profile['remote_root'] + '/src:/opt/cool/python_3.11/site-packages',
        'LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib',
        profile['remote_python'], '-u', '-m', 'care_vision.lab.remote_worker',
    ])


class RemoteService:
    def __init__(self, profile_path, output, seconds, *, check_only=False):
        self.profile = load_profile(profile_path)
        self.output, self.seconds, self.check_only = Path(output), seconds, check_only
        self.worker = self.tunnel = None
        self.stream = self.errors = None
        self.ready = None

    def start(self):
        configuration = {key: value for key, value in self.profile.items()
                         if key not in {'ssh_host', 'ssh_identity'}}
        root = Path(__file__).resolve().parents[3]
        configuration.update(source_manifest=source_manifest(root), seconds=self.seconds,
                             check_only=self.check_only)
        self.stream = (self.output / 'cloud.jsonl').open('wb')
        self.errors = (self.output / 'ssh.stderr').open('wb')
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('CARE_JUDGE_', 'AWS_'))}
        self.worker = subprocess.Popen(
            [*ssh_args(self.profile), self.profile['ssh_host'], remote_command(self.profile)],
            stdin=subprocess.PIPE, stdout=self.stream, stderr=self.errors, env=env,
        )
        self.worker.stdin.write((json.dumps(configuration) + '\n').encode())
        self.worker.stdin.flush()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            rows = records(self.output / 'cloud.jsonl')
            checked = any(row.get('event') == 'cloud_preflight_verified'
                          and row.get('cool_verified') is True
                          and row.get('full_pipeline') is True for row in rows)
            if self.check_only and checked:
                if self.worker.wait(timeout=5) != 0:
                    raise RuntimeError('Remote preflight failed.')
                return None
            ready = next((row for row in rows if row.get('event') == 'server_ready'), None)
            if ready and checked:
                if not (ready.get('phase') == 3 and ready.get('cool_verified') is True
                        and ready.get('pose_shadow_enabled') is True
                        and ready.get('pose_observation_enabled') is True
                        and ready.get('experimental_s_only_5fps') is True
                        and type(ready.get('port')) is int and 1 <= ready['port'] <= 65535):
                    raise RuntimeError('Remote service did not verify the full COOL pipeline.')
                self.ready = ready
                return self.forward(ready['port'])
            if self.worker.poll() is not None:
                raise RuntimeError('Remote preflight failed; inspect private ssh.stderr.')
            time.sleep(.1)
        raise RuntimeError('Remote startup timed out; input was not opened.')

    def forward(self, remote_port):
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        marker = 'import sys;print("TUNNEL_READY",flush=True);sys.stdin.buffer.read(1)'
        self.tunnel = subprocess.Popen(
            [*ssh_args(self.profile, forwarding=True), '-o', 'ExitOnForwardFailure=yes',
             '-L', f'127.0.0.1:{port}:127.0.0.1:{remote_port}', self.profile['ssh_host'],
             shlex.join([self.profile['remote_python'], '-u', '-c', marker])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors,
            env={k: v for k, v in os.environ.items()
                 if not k.startswith(('CARE_JUDGE_', 'AWS_'))},
        )
        readable, _, _ = select.select([self.tunnel.stdout], [], [], 15)
        if not readable or self.tunnel.stdout.readline().strip() != b'TUNNEL_READY':
            raise RuntimeError('Private tunnel failed; input was not opened.')
        return f'ws://127.0.0.1:{port}'

    def close(self):
        # EOF tells the remote supervisor to terminate only its owned process group.
        for process in (self.tunnel, self.worker):
            if process is None:
                continue
            try:
                process.stdin.close()
            except (OSError, BrokenPipeError):
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        for stream in (self.stream, self.errors):
            if stream:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def check_cloud(profile, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    with RemoteService(profile, output, 1, check_only=True) as service:
        service.start()
