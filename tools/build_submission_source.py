"""Build a deterministic code-only judge-review archive; never uploads or runs AWS."""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = ("pyproject.toml", "uv.lock", ".python-version", "AGENTS.md")
DOCS = ("submission-source-package.md", "submission-dependency-review-20261017.md",
        "portable-cloud-runtime-20261017.md")
SUFFIXES = {".py", ".html", ".js"}
KEY = re.compile(rb"(?m)^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s*$")


def collect(root):
    selected = {Path(name) for name in TOP}
    selected |= {Path("docs") / name for name in DOCS}
    for folder in ("src", "tests", "tools"):
        for path in (root / folder).rglob("*"):
            if path.is_symlink():
                raise ValueError(f"symlink rejected: {path.relative_to(root)}")
            if path.is_file() and path.suffix in SUFFIXES and not any(
                part.startswith(".") or part == "__pycache__"
                for part in path.relative_to(root).parts
            ):
                selected.add(path.relative_to(root))
    payloads = {}
    for relative in sorted(selected):
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"external path rejected: {relative}")
        payload = path.read_bytes()
        if KEY.search(payload):
            raise ValueError(f"private-key header rejected: {relative}")
        payloads[relative.as_posix()] = payload
    payloads["README.md"] = payloads["docs/submission-source-package.md"]
    return payloads


def build(root, output):
    payloads = collect(root)
    manifest = {
        "scope": "Code-only review candidate; no models, videos, states, SDK or credentials",
        "release_ready": False,
        "files_sha256": {name: hashlib.sha256(data).hexdigest()
                         for name, data in payloads.items()},
        "excluded": ["artifacts", "evaluation", ".venv", "models", "private state"],
        "remaining": ["rights review", "Graviton SDK provisioning", "recorded data bundle",
                      "judge access", "full recorded AWS reproduction"],
    }
    payloads["source-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo("care-vision-source/" + name, (2026, 10, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return {"files": len(payloads), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "release_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(ROOT, args.output), indent=2))


if __name__ == "__main__":
    main()
