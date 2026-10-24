import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('repo_hygiene', ROOT / 'tools/check_repository_hygiene.py')
HYGIENE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HYGIENE)


@pytest.mark.parametrize('name', [
    'recording.MP4', 'videos/new_scene.mov', 'docs/clip.webm', 'src/face.jpg',
    'tests/fixture.PNG', 'tools/camera.pem', '.env', '.env.local', '.aws/credentials',
    'docs/credentials.json', 'artifacts/trace.py', 'evaluation/observations.json',
    'models/person.onnx', 'docs/state.jsonl', 'docs/data.zip', '../outside.py',
])
def test_private_paths_cannot_enter_source(name):
    assert HYGIENE.inspect_payload(name, b'private example')


@pytest.mark.parametrize('content', [
    b'....ftypisom', b'RIFFabcdAVI ', b'RIFFabcdWEBP', b'\x89PNGdata',
    b'\xff\xd8\xffJPEG', b'\x1aE\xdf\xa3webm', b'PK\x03\x04zip', b'text\0binary',
])
def test_renamed_media_and_binary_are_rejected(content):
    assert HYGIENE.inspect_payload('src/renamed.py', content)


@pytest.mark.parametrize('content', [
    b'-----BEGIN ' + b'PRIVATE KEY-----',
    b'AK' + b'IA' + b'ABCD1234EFGH5678',
    b'gh' + b'p_' + b'aBcD1234' * 5,
    b'rtsp' + b'://' + b'user:example-password@example.invalid/stream',
    b'/' + b'Users' + b'/example/private/',
])
def test_obvious_credentials_and_personal_paths_are_rejected(content):
    assert HYGIENE.inspect_payload('src/example.py', content)


@pytest.mark.parametrize('name,content', [
    ('src/example.py', b'import os\nkey = os.environ.get("APP_TOKEN")\n'),
    ('README.md', b'No personal recordings are distributed.\n'),
    ('.githooks/pre-push', b'#!/bin/sh\npython3 tools/check_repository_hygiene.py --history\n'),
])
def test_reviewed_text_and_environment_based_configuration_are_allowed(name, content):
    assert HYGIENE.inspect_payload(name, content) == []


def test_symlinks_and_large_source_are_rejected():
    assert HYGIENE.inspect_payload('src/link.py', b'', '120000')
    assert HYGIENE.inspect_payload('src/large.py', b'x' * 1_048_577)


def test_ignored_recordings_include_new_scenes_and_uppercase_extensions():
    names = ['new.MP4', 'recordings/scene.mov', 'docs/portrait.JPG', 'src/hidden.webm', '.env', 'notes.txt']
    result = subprocess.run(['git', 'check-ignore', '--stdin'], input='\n'.join(names) + '\n',
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert set(result.stdout.splitlines()) == set(names)


def test_deleted_private_blob_is_still_rejected_in_history(tmp_path):
    def git(*args):
        return subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True)
    git('init', '-b', 'main')
    git('config', 'user.name', 'Synthetic Test')
    git('config', 'user.email', 'test@example.invalid')
    (tmp_path / 'src').mkdir()
    private = tmp_path / 'src/renamed.py'
    private.write_bytes(b'....ftypisom')
    git('add', 'src/renamed.py');git('commit', '-m', 'Synthetic media fixture')
    git('rm', 'src/renamed.py');git('commit', '-m', 'Remove fixture')
    result = subprocess.run(['python3', str(ROOT/'tools/check_repository_hygiene.py'), '--history'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'media/archive signature' in result.stdout
    assert 'ftypisom' not in result.stdout


@pytest.mark.parametrize('name', [
    'docs/images/proposal-architecture.jpg',
    'docs/images/illustrated-architecture.png',
])
def test_only_exact_reviewed_diagrams_are_allowed(name):
    content = (ROOT / name).read_bytes()
    assert HYGIENE.inspect_payload(name, content) == []
    assert HYGIENE.inspect_payload(name, content + b'changed')
    assert HYGIENE.inspect_payload('docs/images/other.jpg', content)
    assert HYGIENE.inspect_payload(name, content, '120000')
