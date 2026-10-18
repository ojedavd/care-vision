import pytest

from care_vision.common import environment


def test_local_startup_does_not_query_ec2(monkeypatch):
    def unexpected():
        raise AssertionError("local mode must not contact instance metadata")

    monkeypatch.setattr(environment, "ec2_identity", unexpected)
    assert "opencv_version" in environment.runtime_info(1)


def test_phase_two_rejects_non_linux_arm(monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(environment.platform, "machine", lambda: "arm64")
    with pytest.raises(RuntimeError, match="requires Linux ARM64"):
        environment.runtime_info(2)


def test_phase_two_records_instance_identity(monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "machine", lambda: "aarch64")
    identity = {
        "instanceId": "i-test",
        "instanceType": "c8g.2xlarge",
        "region": "us-east-1",
        "imageId": "ami-test",
        "architecture": "arm64",
    }
    monkeypatch.setattr(environment, "ec2_identity", lambda: identity)
    assert environment.runtime_info(2)["ec2"] == identity


def test_phase_two_rejects_metadata_mismatch(monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(environment, "ec2_identity", lambda: {"architecture": "x86_64"})
    with pytest.raises(RuntimeError, match="does not identify"):
        environment.runtime_info(2)
