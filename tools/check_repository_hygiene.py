"""Reject private data, media, binary files and obvious credentials in Git content.

The default checks tracked working files. Hooks check staged blobs and every
committed tree, so removing a private file in a later commit cannot hide it.
Gitleaks supplies the complementary full secret scan. Findings never print data.
"""

import argparse
import hashlib
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT_FILES = {
    '.gitignore', '.python-version', 'AGENTS.md', 'README.md', 'SECURITY.md',
    'THIRD_PARTY_NOTICES.md', 'pyproject.toml', 'uv.lock',
}
ROOT_DIRS = {'src', 'tests', 'tools', 'docs', '.github', '.githooks'}
PRIVATE_DIRS = {
    'artifacts', 'evaluation', 'recordings', 'videos', 'captures', 'data', 'datasets',
    'logs', 'states', 'snapshots', 'models', 'private', '.aws', '.ssh', '.venv',
    '__pycache__', '.pytest_cache', '.ruff_cache',
}
TEXT_SUFFIXES = {'.css', '.py', '.html', '.js', '.md', '.toml', '.lock', '.json', '.yml', '.yaml'}
# Exact reviewed architecture figures; other media remain prohibited.
PUBLIC_DIAGRAMS = {
    'docs/images/illustrated-architecture.png':
        '4a92bc8d51bae319bca8409ea568da0d4b5fe07e521f66d4b24562341010f583',
    'docs/images/proposal-architecture.jpg':
        '0925117e969f322e235f31210cb92ba53d946c9fee81881a8f603b9917f84efc',
}
SECRET_FILES = {'credentials', 'credentials.json', 'id_rsa', 'id_ed25519', 'id_ecdsa', '.netrc', '.npmrc'}
PATTERNS = {
    'private key marker': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'AWS access key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'GitHub token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'credential-bearing URL': re.compile(rb'(?i)(?:rtsp|https?|postgres(?:ql)?|mysql)://[^\s/\x22\x27@{}]+:[^\s\x22\x27@{}]+@'),
    'embedded media': re.compile(rb'data:(?:image|video|audio)/[a-zA-Z0-9+.-]+;base64,[A-Za-z0-9+/]{80,}'),
    'personal macOS path': re.compile(rb'/Users/(?!Shared(?:/|\b))[A-Za-z0-9_.-]+/'),
}


def inspect_payload(name, content, mode='100644'):
    """Return redacted reason labels; never include the matched payload."""
    path = PurePosixPath(name)
    issues = []
    if path.is_absolute() or '..' in path.parts or any(ord(c) < 32 for c in name):
        issues.append('unsafe path')
    if mode not in {'100644', '100755'}:
        issues.append('non-regular Git entry')
    if not path.parts or (len(path.parts) == 1 and name not in ROOT_FILES):
        issues.append('root file outside allowlist')
    if len(path.parts) > 1 and path.parts[0] not in ROOT_DIRS:
        issues.append('directory outside allowlist')
    if any(p.lower() in PRIVATE_DIRS for p in path.parts):
        issues.append('private/generated directory')
    if path.name.lower() in SECRET_FILES or path.name.lower().startswith('.env'):
        issues.append('credential/configuration file')
    public_diagram = PUBLIC_DIAGRAMS.get(name) == hashlib.sha256(content).hexdigest()
    hook = name in {'.githooks/pre-commit', '.githooks/pre-push'}
    if name not in ROOT_FILES and not hook and path.suffix.lower() not in TEXT_SUFFIXES and not public_diagram:
        issues.append('unsupported file type; media/archives/weights are excluded')
    if len(content) > (3_145_728 if public_diagram else 1_048_576):
        issues.append('file exceeds source-only size limit')
    if public_diagram:
        return issues
    if (content.startswith((b'\x89PNG', b'\xff\xd8\xff', b'GIF87a', b'GIF89a', b'\x1aE\xdf\xa3', b'PK\x03\x04'))
            or content[4:8] == b'ftyp'
            or (content[:4] == b'RIFF' and content[8:12] in {b'AVI ', b'WAVE', b'WEBP'})):
        issues.append('media/archive signature, including renamed files')
    if b'\0' in content:
        issues.append('binary data')
    try:
        content.decode('utf-8')
    except UnicodeDecodeError:
        issues.append('non-UTF8 data')
    for label, pattern in PATTERNS.items():
        if pattern.search(content):
            issues.append(label)
    return issues


def git(*args):
    return subprocess.check_output(['git', *args])


def entries(staged=False, history=False):
    if history:
        seen = set()
        for commit in git('rev-list', '--all').decode().splitlines():
            for row in git('ls-tree', '-r', '-z', commit).split(b'\0'):
                if not row:
                    continue
                metadata, name = row.split(b'\t', 1)
                mode, kind, oid = metadata.decode().split()
                path = name.decode('utf-8')
                identity = (path, mode, oid)
                if identity in seen:
                    continue
                seen.add(identity)
                yield path, git('cat-file', '-p', oid), mode
    elif staged:
        changed = set(git('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z').split(b'\0'))
        for row in git('ls-files', '--stage', '-z').split(b'\0'):
            if not row:
                continue
            metadata, name = row.split(b'\t', 1)
            if name not in changed:
                continue
            mode, oid, stage = metadata.decode().split()
            yield name.decode(), git('cat-file', '-p', oid), mode
    else:
        for row in git('ls-files', '--stage', '-z').split(b'\0'):
            if not row:
                continue
            metadata, name = row.split(b'\t', 1)
            mode, oid, stage = metadata.decode().split()
            path = Path(name.decode())
            if path.is_symlink():
                yield path.as_posix(), b'', '120000'
            else:
                yield path.as_posix(), path.read_bytes(), mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--staged', action='store_true')
    group.add_argument('--history', action='store_true')
    args = parser.parse_args()
    failed = checked = 0
    for name, content, mode in entries(args.staged, args.history):
        checked += 1
        issues = inspect_payload(name, content, mode)
        if issues:
            failed += 1
            print(f'{name}: {", ".join(issues)}')
    print(f'Repository hygiene: {checked} entries checked; {failed} rejected.')
    raise SystemExit(bool(failed))


if __name__ == '__main__':
    main()
