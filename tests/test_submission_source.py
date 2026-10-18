import importlib.util
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "submission_source", Path(__file__).parents[1] / "tools/build_submission_source.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def tree(tmp_path):
    root = tmp_path / "project"
    for name in (*BUILDER.TOP, *("docs/" + n for n in BUILDER.DOCS), "src/app.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n")
    return root


def test_archive_excludes_private_data_and_is_deterministic(tmp_path):
    root = tree(tmp_path)
    for name in ("artifacts/episode.json", ".env", "src/key.pem", "src/__pycache__/x.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("EXCLUDED_PAYLOAD_SENTINEL_61017\n")
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    assert BUILDER.build(root, a) == BUILDER.build(root, b)
    with zipfile.ZipFile(a) as archive:
        assert all("EXCLUDED_PAYLOAD_SENTINEL_61017" not in archive.read(n).decode()
                   for n in archive.namelist())
    with pytest.raises(FileExistsError):
        BUILDER.build(root, a)


def test_symlink_cannot_pull_external_content(tmp_path):
    root = tree(tmp_path)
    outside = tmp_path / "private.py"
    outside.write_text("private\n")
    (root / "src/link.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        BUILDER.build(root, tmp_path / "out.zip")
    assert not (tmp_path / "out.zip").exists()


def test_key_content_in_allowed_source_is_rejected(tmp_path):
    root = tree(tmp_path)
    (root / "src/app.py").write_text("-----BEGIN " + "PRIVATE KEY-----\nexample\n")
    with pytest.raises(ValueError, match="private-key header"):
        BUILDER.build(root, tmp_path / "out.zip")
