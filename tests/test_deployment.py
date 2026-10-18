import io
import tarfile

import pytest

from care_vision.aws_demo import SSH, prepare_video, source_archive
from care_vision.check_loop import CheckFailed, check_records


def test_upload_contains_only_source_and_lock(tmp_path):
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        (tmp_path / name).write_text("test")
    source = tmp_path / "src/care_vision"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "private.pem").write_text("must not upload")
    (tmp_path / "secret.pem").write_text("must not upload")
    with tarfile.open(fileobj=io.BytesIO(source_archive(tmp_path)), mode="r:gz") as archive:
        assert set(archive.getnames()) == {
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "src/care_vision/__init__.py",
        }


@pytest.mark.parametrize(
    "host", ["-oProxyCommand=evil", "host;echo evil", "$(evil)", "host\nother"]
)
def test_ssh_destination_cannot_be_shell_code(tmp_path, host):
    key = tmp_path / "test.pem"
    key.touch()
    with pytest.raises(ValueError, match="expected an SSH"):
        SSH(host, key)


def test_phase_label_alone_does_not_prove_aws():
    with pytest.raises(CheckFailed, match="EC2 ARM64 runtime evidence"):
        check_records([], [{"event": "server_ready", "phase": 2}])


def test_source_video_can_be_preserved_for_remote_demo(tmp_path):
    source = tmp_path / "person.avi"
    source.write_bytes(b"real-person-fixture")
    destination = tmp_path / "run" / "demo.avi"
    destination.parent.mkdir()

    result = prepare_video(destination, source)

    assert result == destination
    assert destination.read_bytes() == source.read_bytes()
