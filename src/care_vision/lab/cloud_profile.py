"""Explicit private configuration for an evaluator-owned Graviton host."""

import hashlib
import json
import re
from pathlib import Path

FIELDS = {'ssh_host', 'ssh_identity', 'remote_root', 'remote_python', 'person_model',
          'person_sha256', 'pose_config', 'pose_config_sha256'}


def load_profile(path):
    path = Path(path).expanduser()
    if path.stat().st_mode & 0o077:
        raise ValueError('Cloud profile must have owner-only permissions (chmod 600).')
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError('Cloud profile fields do not match the documented profile.')
    if any(not isinstance(item, str) for item in value.values()):
        raise ValueError('Cloud profile fields must be strings.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]{0,250}', value['ssh_host']):
        raise ValueError('Use a trusted SSH alias or user@host.')
    key = Path(value['ssh_identity']).expanduser()
    if not key.is_file() or key.stat().st_mode & 0o077:
        raise ValueError('SSH identity must exist with owner-only permissions.')
    value['ssh_identity'] = str(key.absolute())
    for name in ('remote_root', 'remote_python', 'person_model', 'pose_config'):
        p = Path(value[name])
        if not p.is_absolute() or '..' in p.parts or any(ord(c) < 32 for c in value[name]):
            raise ValueError('Remote paths must be absolute, without traversal/control characters.')
    for name in ('person_sha256', 'pose_config_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', value[name]):
            raise ValueError('Expected SHA-256 model/configuration hashes.')
    return value


def source_manifest(root):
    root = Path(root)
    paths = [root / 'pyproject.toml', root / 'uv.lock',
             *sorted((root / 'src' / 'care_vision').rglob('*.py'))]
    if any(p.is_symlink() for p in paths):
        raise ValueError('Source symlinks are not accepted.')
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def ssh_args(profile, *, forwarding=False):
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=5',
            '-o', 'ServerAliveCountMax=2', '-o', 'ForwardAgent=no',
            '-o', 'ClearAllForwardings=' + ('no' if forwarding else 'yes'),
            '-o', 'GatewayPorts=no', '-i', profile['ssh_identity']]
