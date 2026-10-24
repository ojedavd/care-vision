import hashlib
import io
import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from care_vision.lab.cloud_profile import load_profile, source_manifest, ssh_args
from care_vision.lab.live import run_camera
from care_vision.lab.models import available, default_directory
from care_vision.lab.pose_config import create
from care_vision.lab.remote import RemoteService, remote_command
from care_vision.lab.remote_worker import command, supervise
from care_vision.lab.session import LabSessions
from care_vision.lab.telemetry import VisionTelemetry


def profile(tmp_path):
    key = tmp_path / 'ssh-identity'
    key.write_text('test placeholder, no real credential')
    key.chmod(0o600)
    value = {
        'ssh_host': 'evaluator-graviton', 'ssh_identity': str(key),
        'remote_root': '/opt/evaluator/care-vision',
        'remote_python': '/opt/evaluator/care-vision/.venv-cloud/bin/python',
        'person_model': '/opt/evaluator/person.onnx', 'person_sha256': 'a' * 64,
        'pose_config': '/opt/evaluator/pose.json', 'pose_config_sha256': 'b' * 64,
    }
    path = tmp_path / 'cloud-profile.json'
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path, value


def test_profile_is_explicit_private_and_remote_arguments_are_quoted(tmp_path):
    path, value = profile(tmp_path)
    loaded = load_profile(path)
    assert loaded == value
    args = ssh_args(loaded)
    assert 'StrictHostKeyChecking=yes' in args and 'ForwardAgent=no' in args
    assert 'ClearAllForwardings=yes' in args
    assert 'ClearAllForwardings=no' in ssh_args(loaded, forwarding=True)
    loaded['remote_root'] = '/opt/a space/$(untrusted)'
    words = shlex.split(remote_command(loaded))
    assert 'PYTHONPATH=/opt/a space/$(untrusted)/src:/opt/cool/python_3.11/site-packages' in words
    path.chmod(0o644)
    with pytest.raises(ValueError, match='owner-only'):
        load_profile(path)


@pytest.mark.parametrize(('field', 'value'), [
    ('ssh_host', '-oProxyCommand=anything'), ('remote_root', '/opt/../etc'),
    ('remote_python', 'python3'), ('person_sha256', 'unchecked'), ('pose_config', '/a\nb'),
])
def test_invalid_configuration_fails_before_ssh(tmp_path, field, value):
    path, content = profile(tmp_path)
    content[field] = value
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_profile(path)


def test_modes_require_models_or_private_profile_without_starting_input(tmp_path):
    session = LabSessions(tmp_path / 'sessions', models=tmp_path / 'absent-models')
    for args in ({'source': 'camera', 'pipeline': 'full'},
                 {'source': 'camera', 'pipeline': 'full', 'target': 'graviton'},
                 {'source': 'video'}, {'target': 'some-other-cloud'}, {'pipeline': []}):
        with pytest.raises(ValueError):
            session.start(**args)
    assert session.process is None and not session.snapshot()['aws_connected']
    assert not session.snapshot()['capabilities']['graviton_profile_configured']


def test_graviton_launch_uses_only_startup_profile_and_has_no_local_fallback(monkeypatch, tmp_path):
    path, _ = profile(tmp_path)
    calls = []
    process = SimpleNamespace(poll=lambda: None, returncode=None)
    monkeypatch.setattr('care_vision.lab.session.subprocess.Popen',
                        lambda argv, **kw: calls.append((argv, kw)) or process)
    session = LabSessions(tmp_path / 'sessions', cloud_profile=path)
    session.start(source='camera', pipeline='full', target='graviton')
    argv, _ = calls[0]
    assert argv[argv.index('--cloud-profile') + 1] == str(path)
    assert '--models' not in argv and '--host' not in argv and '--url' not in argv
    state = session.snapshot()
    assert state['processing_location'] == 'AWS_GRAVITON'
    assert not state['aws_connected'] and not state['runtime']['cool_verified']
    assert str(path) not in json.dumps(state)
    process.poll = lambda: 0
    process.returncode = 0
    session.stop()


def test_remote_preflight_failure_never_opens_camera_or_local_vision(monkeypatch, tmp_path):
    closed = []
    class Failure:
        def __init__(self, *_):
            pass

        def start(self):
            raise RuntimeError('synthetic remote failure')

        def close(self):
            closed.append(True)
    monkeypatch.setattr('care_vision.lab.live.RemoteService', Failure)
    monkeypatch.setattr('care_vision.lab.live.LocalPreviewCamera',
                        lambda *a, **k: pytest.fail('Input opened before preflight'))
    monkeypatch.setattr('care_vision.lab.live.subprocess.Popen',
                        lambda *a, **k: pytest.fail('Unexpected local fallback'))
    with pytest.raises(RuntimeError, match='remote failure'):
        run_camera(tmp_path / 'run', 0, 1, target='graviton', pipeline='full',
                   cloud_profile=tmp_path / 'profile')
    assert closed == [True]
    assert not (tmp_path / 'run/report.json').exists()


@pytest.mark.parametrize('invalid', ['phase', 'cool_verified', 'pose_shadow_enabled',
                                    'pose_observation_enabled', 'port', None])
def test_tunnel_gate_requires_verified_complete_cloud_ready(monkeypatch, tmp_path, invalid):
    path, _ = profile(tmp_path)
    output = tmp_path / 'output'
    output.mkdir()
    ready = {'event': 'server_ready', 'phase': 3, 'cool_verified': True,
             'pose_shadow_enabled': True, 'pose_observation_enabled': True,
             'experimental_s_only_5fps': True, 'port': 18765}
    if invalid:
        ready[invalid] = {'phase': 1, 'port': 70000}.get(invalid, False)
    def spawn(argv, **kw):
        rows = [{'event': 'cloud_preflight_verified', 'cool_verified': True,
                 'full_pipeline': True}, ready]
        kw['stdout'].write(('\n'.join(json.dumps(row) for row in rows) + '\n').encode())
        kw['stdout'].flush()
        return SimpleNamespace(stdin=io.BytesIO(), poll=lambda: None, wait=lambda **_: 0)
    monkeypatch.setattr('care_vision.lab.remote.subprocess.Popen', spawn)
    forwarded = []
    monkeypatch.setattr(RemoteService, 'forward',
                        lambda self, port: forwarded.append(port) or 'ws://127.0.0.1:18765')
    with RemoteService(path, output, 1) as service:
        if invalid:
            with pytest.raises(RuntimeError, match='full COOL pipeline'):
                service.start()
            assert not forwarded
        else:
            assert service.start() == 'ws://127.0.0.1:18765'
            assert forwarded == [18765]


def test_remote_disconnect_stops_only_owned_worker_with_real_process(monkeypatch, capsys):
    configuration = {'remote_root': '/', 'seconds': 1,
                     'person_model': '/external/person.onnx', 'pose_config': '/external/pose.json'}
    argv = command(configuration)
    assert argv[argv.index('--phase') + 1] == '3'
    assert '--experimental-pose-observation' in argv
    assert '--experimental-alert-after-seconds' not in argv
    processes = []
    canary = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'],
                             start_new_session=True)
    monkeypatch.setattr(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(b'')))
    def spawn(_argv, **kw):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], **kw)
        processes.append(child)
        return child
    try:
        result = supervise(configuration, verify_fn=lambda _: ({'cool': {'verified': True}},
                                                               'unknown'), spawn=spawn)
        assert result == 0 and processes[0].poll() is not None
        assert canary.poll() is None
        assert 'cloud_supervisor_stopped' in capsys.readouterr().out
    finally:
        canary.terminate()
        canary.wait(timeout=3)
        for child in processes:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=3)


def test_check_only_and_failed_preflight_cannot_launch_worker(capsys):
    configuration = {'remote_root': '/', 'seconds': 1, 'person_model': '/person.onnx',
                     'pose_config': '/pose.json', 'check_only': True}
    def spawn(*args, **kwargs):
        pytest.fail('Preflight launched a service')
    assert supervise(configuration, verify_fn=lambda _: ({'cool': {'verified': True}},
                                                         'unknown'), spawn=spawn) == 0
    assert 'cloud_preflight_verified' in capsys.readouterr().out
    def broken(_):
        raise RuntimeError('native verification failed')
    with pytest.raises(RuntimeError):
        supervise(configuration, verify_fn=broken, spawn=spawn)
    with pytest.raises(RuntimeError, match='COOL'):
        supervise(configuration, verify_fn=lambda _: ({'cool': {'verified': False}}, 'unknown'),
                  spawn=spawn)


def test_manifest_binds_source_and_lock_and_rejects_symlinks(tmp_path):
    (tmp_path / 'src/care_vision').mkdir(parents=True)
    for name in ('pyproject.toml', 'uv.lock', 'src/care_vision/module.py'):
        (tmp_path / name).write_text('test')
    initial = source_manifest(tmp_path)
    (tmp_path / 'uv.lock').write_text('changed')
    assert source_manifest(tmp_path) != initial
    (tmp_path / 'src/care_vision/module.py').unlink()
    (tmp_path / 'src/care_vision/module.py').symlink_to(tmp_path / 'uv.lock')
    with pytest.raises(ValueError, match='symlink'):
        source_manifest(tmp_path)


def test_scene_configuration_is_unknown_without_explicit_review_and_never_overwrites(tmp_path):
    model = tmp_path / 'pose.onnx'
    model.write_bytes(b'test model')
    output = tmp_path / 'scene.json'
    create(model, output)
    config = json.loads(output.read_text())
    assert config['context_status'] == 'unknown'
    assert config['model_sha256'] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert config['geometry']['source_width'] == 1280
    assert not output.stat().st_mode & 0o077
    with pytest.raises(FileExistsError):
        create(model, output)
    with pytest.raises(ValueError, match='both'):
        create(model, tmp_path / 'bad.json', reviewed=True)


def test_pose_projection_is_bounded_and_does_not_expose_identifiers_or_paths(monkeypatch):
    p = VisionTelemetry()
    p.accept({'event': 'observation', 'evidence_kind': 'live', 'observation_id': 'private-id',
              'width': 1280, 'height': 720, 'person_count': 1, 'confirmed_person_count': 1,
              'person_detection_status': 'ok', 'private_path': '/private/hidden'})
    p.accept({'event': 'pose_shadow', 'observation_id': 'private-id', 'capture_seq': 10,
              'pose_status': 'ok', 'points': [[640, 360, .8]] * 17,
              'response_support': {'torso_supported': True}, 'seat_support_verified': True})
    result = p.lab_snapshot()
    assert result['pose']['points'][0] == [.5, .5, .8]
    assert result['pose']['seat_support_verified'] is False
    assert result['observation']['person_count'] == 1
    assert 'private-id' not in json.dumps(result) and '/private/hidden' not in json.dumps(result)
    p.accept({'event': 'pose_shadow_error'})
    assert p.lab_snapshot()['pose']['points'] is None


@pytest.mark.integration
@pytest.mark.skipif(not available(default_directory()), reason='External models not installed')
def test_full_local_worker_runs_real_models_on_generated_file_without_aws(tmp_path):
    from care_vision.fixture import make_video
    video = make_video(tmp_path / 'test-input.avi', seconds=5)
    run = tmp_path / 'run'
    run_camera(run, 0, 3, pipeline='full', models=default_directory(), video=video)
    rows = [json.loads(line) for line in (run / 'cloud.jsonl').read_text().splitlines()]
    ready = next(row for row in rows if row['event'] == 'server_ready')
    assert ready['phase'] == 1 and not ready['cool_verified'] and ready['pose_shadow_enabled']
    observations = [r for r in rows if r['event'] == 'observation']
    assert observations and all(r['person_detection_status'] == 'ok' for r in observations)
    assert all((r['width'], r['height']) == (1280, 720) for r in observations)
    report = json.loads((run / 'report.json').read_text())
    assert report['kind'] == 'LOCAL_FULL_PIPELINE' and not report['cool_verified']
    assert not report['audit_performed'] and not report['images_saved']
    assert not any(p.suffix in {'.avi', '.jpg', '.png', '.mp4'} for p in run.rglob('*'))
