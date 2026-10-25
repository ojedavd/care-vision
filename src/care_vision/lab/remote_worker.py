"""Bounded cloud supervisor; JSON on stdin, cloud telemetry on stdout, no media files.

This module runs on the evaluator's Graviton. Closing the SSH input stops only its
owned worker. The independent deadline also bounds a lost connection.
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from care_vision.common.metrics import EventLog
from care_vision.lab.cloud_profile import seated_duration


def verify(configuration):
    from care_vision.cloud.pose_shadow import shadow_factory
    from care_vision.cloud.rtmpose import RTMPoseEstimator
    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.environment import runtime_info
    from care_vision.lab.cloud_profile import source_manifest

    root = Path(configuration['remote_root'])
    if source_manifest(root) != configuration['source_manifest']:
        raise ValueError('Local/remote source or dependency lock mismatch.')
    if 'OPENCV_FORCE_DNN_ENGINE' in os.environ:
        raise ValueError('DNN environment override is not accepted.')
    environment = runtime_info(3)  # EC2 ARM64 + actual native COOL operation checks.
    person = Path(configuration['person_model'])
    pose_config = Path(configuration['pose_config'])
    for path, expected in ((person, configuration['person_sha256']),
                           (pose_config, configuration['pose_config_sha256'])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Configured model or scene configuration hash mismatch.')
    shadow_factory(pose_config)  # Validates scene context and the external pose model hash.
    config = json.loads(pose_config.read_text())
    geometry = config['geometry']
    if (geometry['edge_id'] != 'edge-1'
            or (geometry['source_width'], geometry['source_height']) != (1280, 720)):
        raise ValueError('Lab scene config must match edge-1 and 1280x720 input.')
    import numpy as np

    detector = YoloXPersonDetector(person, input_size=640, score_threshold=.2,
                                   nms_threshold=.3, dnn_engine='classic')
    detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    estimator = RTMPoseEstimator(pose_config.parent / config['model_path'])
    estimator.estimate(np.zeros((720, 1280, 3), dtype=np.uint8), [200, 20, 400, 680])
    return environment, config['context_status']


def command(configuration):
    seconds = configuration['seconds']
    if type(seconds) is not int or not 1 <= seconds <= 120:
        raise ValueError('Input duration must be 1..120 seconds.')
    argv = [sys.executable, '-u', '-m', 'care_vision.cloud', '--phase', '3',
            '--host', '127.0.0.1', '--port', '0', '--run-seconds', str(seconds + 45),
            '--person-model', configuration['person_model'], '--person-detector', 'yolox',
            '--person-input-size', '640', '--experimental-s-only-5fps',
            '--experimental-weak-association', '--experimental-yolox-classic-engine',
            '--experimental-episode-continuity', '--pose-shadow-config',
            configuration['pose_config']]
    seated = seated_duration(configuration)
    if seated is not None:
        argv.extend(['--experimental-seated-closure-seconds', str(seated)])
    else:
        argv.append('--experimental-pose-observation')
    return argv


def stop_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def supervise(configuration, *, verify_fn=verify, spawn=subprocess.Popen):
    argv = command(configuration)
    signal.alarm(55)  # Also bounds runtime/model preparation before an owned worker exists.
    try:
        environment, context = verify_fn(configuration)
    finally:
        signal.alarm(0)
    if environment.get('cool', {}).get('verified') is not True:
        raise RuntimeError('Native COOL verification is required; no fallback is allowed.')
    seated = seated_duration(configuration)
    if seated is not None and context != 'current':
        raise ValueError('Seated closure requires current, independently reviewed scene context.')
    log = EventLog('cloud')
    log.emit('cloud_preflight_verified', cool_verified=environment['cool']['verified'],
             person_model_verified=True, pose_model_verified=True, source_verified=True,
             context_status=context, full_pipeline=True, seated_closure_seconds=seated)
    if configuration.get('check_only') is True:
        return 0
    # No SSH agent, portal credential or AWS account credential reaches inference.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('CARE_JUDGE_', 'AWS_')) and k != 'SSH_AUTH_SOCK'}
    process = spawn(argv, cwd=configuration['remote_root'], env=env,
                    stdin=subprocess.DEVNULL, start_new_session=True)
    disconnected = threading.Event()

    def watch_input():
        sys.stdin.buffer.read(1)
        disconnected.set()

    try:
        threading.Thread(target=watch_input, daemon=True).start()
        deadline = time.monotonic() + configuration['seconds'] + 50
        while process.poll() is None and not disconnected.wait(.1):
            if time.monotonic() > deadline:
                break
    finally:
        stop_owned(process)
    log.emit('cloud_supervisor_stopped', worker_stopped=process.poll() is not None)
    return 0 if disconnected.is_set() else process.returncode


def main():
    os.umask(0o077)
    def terminate(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    # The profile is delivered through a private pipe, not a URL or command line.
    line = sys.stdin.buffer.readline(1024 * 1024)
    if not line.endswith(b'\n'):
        raise ValueError('Bounded JSON configuration required.')
    configuration = json.loads(line)
    try:
        return supervise(configuration)
    except (OSError, ValueError, RuntimeError, KeyError):
        EventLog('cloud').emit('cloud_preflight_failed',
                               reason='Inspect private SSH stderr; no input was started.')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
