import json
import sys
from types import SimpleNamespace

import cv2
import pytest

from care_vision.cloud import __main__ as cli
from care_vision.cloud.yolox import YoloXPersonDetector, validate_dnn_engine


@pytest.mark.parametrize("engine", ["auto", "classic"])
def test_named_engine_selection_and_observation_backend(tmp_path, monkeypatch, engine):
    calls = []
    net = SimpleNamespace(setPreferableBackend=lambda _: None, setPreferableTarget=lambda _: None)
    monkeypatch.delenv("OPENCV_FORCE_DNN_ENGINE", raising=False)
    monkeypatch.setattr(cv2.dnn, "ENGINE_CLASSIC", 1234, raising=False)

    def read(path, **kwargs):
        calls.append((path, kwargs))
        return net

    monkeypatch.setattr(cv2.dnn, "readNet", read)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    detector = YoloXPersonDetector(model, dnn_engine=engine)
    assert calls == [(str(model), {"engine": 1234} if engine == "classic" else {})]
    assert detector.dnn_engine_requested == engine
    assert detector.backend.endswith(":engine-classic") == (engine == "classic")


@pytest.mark.parametrize("fault", ["unsupported", "environment", "load_failure"])
def test_explicit_classic_never_silently_falls_back(tmp_path, monkeypatch, fault):
    monkeypatch.delenv("OPENCV_FORCE_DNN_ENGINE", raising=False)
    monkeypatch.setattr(cv2.dnn, "ENGINE_CLASSIC", 1234, raising=False)
    if fault == "unsupported":
        monkeypatch.delattr(cv2.dnn, "ENGINE_CLASSIC")
    elif fault == "environment":
        monkeypatch.setenv("OPENCV_FORCE_DNN_ENGINE", "2")
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("SDK failed to load explicit classic")

    monkeypatch.setattr(cv2.dnn, "readNet", read)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    with pytest.raises(RuntimeError if fault == "load_failure" else ValueError):
        YoloXPersonDetector(model, dnn_engine="classic")
    assert len(calls) == (1 if fault == "load_failure" else 0)


def test_auto_keeps_legacy_environment_behavior_and_rejects_unknown_engine(monkeypatch):
    monkeypatch.setenv("OPENCV_FORCE_DNN_ENGINE", "2")
    validate_dnn_engine("auto")
    with pytest.raises(ValueError, match="auto or classic"):
        validate_dnn_engine("new")


@pytest.mark.parametrize("classic", [False, True])
def test_cloud_cli_passes_opt_in_and_logs_request(tmp_path, monkeypatch, classic):
    monkeypatch.delenv("OPENCV_FORCE_DNN_ENGINE", raising=False)
    monkeypatch.setattr(cv2.dnn, "ENGINE_CLASSIC", 1234, raising=False)
    model, log = tmp_path / "model.onnx", tmp_path / "cloud.jsonl"
    model.write_bytes(b"model")
    calls = []

    class Server:
        def __init__(self, eventlog, detector_factory, *args, **kwargs):
            calls.append(detector_factory.keywords)

        async def run(self, *args):
            return None

    monkeypatch.setattr(cli, "CloudServer", Server)
    argv = ["cloud", "--phase", "3", "--experimental-s-only-5fps", "--person-detector",
            "yolox", "--person-model", str(model), "--person-input-size", "640", "--log", str(log)]
    if classic:
        argv.append("--experimental-yolox-classic-engine")
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls[0].get("dnn_engine", "auto") == ("classic" if classic else "auto")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    config = next(r for r in rows if r["event"] == "person_detector_configuration")
    assert config["dnn_engine_requested"] == ("classic" if classic else "auto")


def test_cli_rejects_classic_outside_opt_in_cloud_profile(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cloud", "--experimental-yolox-classic-engine"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "requires phase 3" in capsys.readouterr().err
