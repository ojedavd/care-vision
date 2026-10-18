import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = str(Path(__file__).parents[1] / 'tools')
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
runtime = importlib.import_module('portable_cloud_runtime')
runner = importlib.import_module('run_portable_classic_replays')
auditor = importlib.import_module('audit_portable_classic_replays')


def test_local_asset_or_pipeline_changes_rejected_before_connection(tmp_path):
    root, assets = tmp_path / 'code', tmp_path / 'assets'
    (root / 'src').mkdir(parents=True)
    assets.mkdir()
    source, model = root / 'src/a.py', assets / 'model.onnx'
    source.write_bytes(b'code')
    model.write_bytes(b'model')
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'files_sha256': {
        'src/a.py': hashlib.sha256(b'code').hexdigest(),
        'model.onnx': hashlib.sha256(b'model').hexdigest(),
    }}))
    assert len(runner.verify_inputs(root, assets, profile, [model])) == 2
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='asset mismatch'):
        runner.verify_inputs(root, assets, profile, [model])
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='pipeline mismatch'):
        runner.verify_inputs(root, assets, profile, [model])
    source.unlink()
    with pytest.raises(ValueError, match='source file set changed'):
        runner.verify_inputs(root, assets, profile, [model])


@pytest.mark.parametrize('failure', [None, 'architecture', 'sdk', 'python', 'uv', 'override'])
def test_locked_cloud_setup_rejects_incompatible_runtime(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(os, 'environ', {})
    python, uv = tmp_path / 'python', tmp_path / 'uv'
    python.touch()
    uv.touch()
    monkeypatch.setattr(sys, 'argv', ['remote', str(tmp_path), str(python), str(uv), '0.11.32'])
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64' if failure == 'architecture'
                        else 'aarch64')
    original = Path.is_dir
    monkeypatch.setattr(Path, 'is_dir', lambda p: failure != 'sdk'
                        if str(p) == '/opt/cool/python_3.11/site-packages/cv2' else original(p))
    monkeypatch.delenv('OPENCV_FORCE_DNN_ENGINE', raising=False)
    if failure == 'override':
        monkeypatch.setenv('OPENCV_FORCE_DNN_ENGINE', '')
    calls = []

    def output(command, **kwargs):
        if command[0] == str(python):
            return '[3,12]' if failure == 'python' else '[3,11]'
        if command[0] == str(uv):
            return 'uv 0.0.0' if failure == 'uv' else 'uv 0.11.32'
        assert command[0] == str(tmp_path / '.venv/bin/python')
        assert '/opt/cool/' in kwargs['env']['PYTHONPATH']
        return json.dumps({'cool': {'verified': True}, 'classic_available': True})

    monkeypatch.setattr(subprocess, 'check_output', output)
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: calls.append((cmd, kw)))
    if failure:
        with pytest.raises(SystemExit):
            exec(runtime.PREPARE_RUNTIME, {})
        assert not calls
    else:
        exec(runtime.PREPARE_RUNTIME, {})
        cmd, kw = calls[0]
        assert '--locked' in cmd and '--no-default-groups' in cmd
        assert kw['env']['UV_PROJECT_ENVIRONMENT'] == str(tmp_path / '.venv')
        assert kw['env']['UV_PYTHON_DOWNLOADS'] == 'never'
        assert 'PYTHONPATH' not in kw['env']


@pytest.mark.parametrize('mode', ['baseline', 'pose', 'test20'])
def test_cloud_launch_uses_workspace_python_and_original_flags(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(os, 'environ', {})
    python = tmp_path / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(sys, 'argv', ['remote', str(tmp_path), 'case', 'pose.json', mode])
    monkeypatch.delenv('OPENCV_FORCE_DNN_ENGINE', raising=False)
    calls = []

    def popen(cmd, **kw):
        calls.append((cmd, kw))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(subprocess, 'Popen', popen)
    exec(runtime.START, {})
    cmd, kw = calls[0]
    assert cmd[0] == str(python)
    assert '--experimental-yolox-classic-engine' in cmd
    assert '--experimental-episode-continuity' in cmd
    assert '--experimental-pose-observation' in cmd
    assert ('--experimental-alert-after-seconds' in cmd) == (mode == 'test20')
    assert 'OPENCV_FORCE_DNN_ENGINE' not in kw['env']
    assert not any('releases/2026' in value for value in cmd)


@pytest.mark.parametrize('field,value', [
    ('classic_available', False), ('machine', 'x86_64'), ('uv_version', 'wrong'),
    ('python_environment', '/old-release/python'), ('sdk_installed_by_tool', True),
])
def test_auditor_rejects_preflight_mismatches(field, value):
    evidence = {'classic_available': True, 'cool': {'verified': True}, 'machine': 'aarch64',
                'uv_version': 'pinned', 'python_requested': '/python',
                'python_environment': '/work/.venv/bin/python', 'sdk_installed_by_tool': False}
    plan = {'portable_runtime': {'uv_version': 'pinned', 'python': '/python'}}
    assert auditor.prepared_runtime_matches(evidence, plan, '/work')
    evidence[field] = value
    assert not auditor.prepared_runtime_matches(evidence, plan, '/work')


def test_remote_paths_do_not_accept_traversal_or_relative_values():
    assert runner.remote_path('/home/operator/') == '/home/operator'
    for value in ('relative', '/home/../private', '/home\nother'):
        with pytest.raises(argparse.ArgumentTypeError):
            runner.remote_path(value)


@pytest.mark.parametrize('tamper', [False, True])
def test_preparation_and_hash_failure_never_contact_ssh(tmp_path, monkeypatch, tamper):
    code, assets = tmp_path / 'code', tmp_path / 'assets'
    paths = ['tools/run_portable_classic_replays.py', 'tools/run_pose_observation_loop.py',
             'tools/portable_cloud_runtime.py', 'src/app.py', 'pyproject.toml',
             'uv.lock', '.python-version']
    for name in paths:
        p = code / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('fixture')
    config = 'artifacts/pose-observation-live-20261014/preparation-hold-1/pose-config.json'
    inputs = [config,
              'artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx',
              'artifacts/phase7d-pose-20260921/models/end2end.onnx']
    for name, folder in [('recovery-2', 'tapo-weak-live-20261005'),
                         ('no-recovery-1', 'tapo-weak-live-20261005'),
                         ('sit-return-live-2', 'pose-observation-live-20261014'),
                         ('sit-hold-live-1', 'pose-observation-live-20261014')]:
        inputs.extend(f'artifacts/{folder}/{name}/{f}' for f in
                      ('demo.avi', 'demo.captures.jsonl', 'activity-review.json'))
    for name in inputs:
        p = assets / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'context_provenance': 'fixture'}) if name == config else 'asset')
    hashes = {name: hashlib.sha256((assets / name).read_bytes()).hexdigest() for name in inputs}
    hashes['src/app.py'] = hashlib.sha256((code / 'src/app.py').read_bytes()).hexdigest()
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'files_sha256': hashes}))
    monkeypatch.setattr(runner, '__file__', str(code / 'tools/run_portable_classic_replays.py'))
    monkeypatch.setattr(runner, 'SSH', lambda *_: pytest.fail('SSH must not be contacted'))
    args = SimpleNamespace(asset_root=assets, output=tmp_path / 'output', prepare_only=True,
                           profile=profile, remote_python=None, remote_uv=None)
    if tamper:
        (assets / inputs[1]).write_text('tampered')
        with pytest.raises(ValueError, match='asset mismatch'):
            runner.run(args)
        assert not (args.output / 'package.tar.gz').exists()
    else:
        runner.run(args)
        with tarfile.open(args.output / 'package.tar.gz') as archive:
            names = archive.getnames()
        assert 'uv.lock' in names and 'models/person.onnx' in names
        assert not any(name.endswith('.avi') for name in names)


def test_ported_auditor_preserves_historical_case_results(tmp_path):
    project = Path(__file__).parents[1]
    historical = project / 'artifacts/yolox-classic-integration-20261015/run-1'
    if not historical.exists():
        pytest.skip('historical evidence is separate from the code-only package')
    expected = json.loads((historical / 'assessment.json').read_text())
    plan = json.loads((historical / 'plan.json').read_text())
    plan['portable_runtime'] = {'uv_version': 'fixture', 'python': '/fixture/python'}
    frozen = json.loads((project /
                         'artifacts/mvp-demo-preparation-20261016/profile.json').read_text())
    frozen = {p: h for p, h in frozen['files_sha256'].items()
              if p.startswith('src/') or p.endswith(('.onnx', '.avi', '.captures.jsonl'))}
    plan['frozen_asset_paths'] = [p for p in frozen if not p.startswith('src/')]
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    for condition in plan['conditions']:
        label = condition['source'] + '-' + condition['mode']
        (tmp_path / label).symlink_to(historical / label, target_is_directory=True)
    deployment = json.loads((historical / 'deployment.json').read_text())
    (tmp_path / 'deployment.json').write_text(json.dumps(deployment))
    (tmp_path / 'local-preflight.json').write_text(json.dumps({'files_sha256': frozen}))
    # Fixture exercises path/audit compatibility; not evidence of fresh remote preparation.
    (tmp_path / 'remote-preflight.json').write_text(json.dumps({
        'cool': {'verified': True}, 'classic_available': True, 'machine': 'aarch64',
        'sdk_installed_by_tool': False, 'uv_version': 'fixture',
        'python_requested': '/fixture/python',
        'python_environment': str(Path(deployment['workspace']) / '.venv/bin/python'),
    }))
    result = auditor.audit(tmp_path, project)
    assert result['passed']
    assert result['runs'] == expected['runs']
