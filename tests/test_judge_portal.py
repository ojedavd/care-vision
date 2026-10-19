import http.client
import json
import secrets
import threading
import time

import pytest

from care_vision.judge.__main__ import Auth, Portal
from care_vision.judge.session import Sessions
from care_vision.judge.state import LogTail, Projection


def credentials():
    return {name: secrets.token_urlsafe(32) for name in ("viewer", "operator")}


@pytest.fixture
def portal(tmp_path):
    keys = credentials()
    sessions = Sessions(tmp_path / "sessions", synthetic=True, seconds=45)
    server = Portal(("127.0.0.1", 0), sessions, Auth(keys))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, keys
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def call(server, path, body=None, *, cookie="", csrf="", origin=None, host=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    headers = {"Host": host or server.host, "Cookie": cookie}
    data = None
    if body is not None:
        headers.update(
            {
                "Content-Type": "application/json",
                "Origin": origin or server.origin,
                "X-CSRF-Token": csrf,
            }
        )
        data = json.dumps(body)
    connection.request("GET" if body is None else "POST", path, data, headers)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def login(server, credential):
    status, headers, data = call(server, "/api/login", {"credential": credential})
    assert status == 200
    return headers["Set-Cookie"].split(";")[0], json.loads(data)["csrf"]


def test_unauthenticated_api_and_files_are_inaccessible(portal):
    server, _ = portal
    for path in ("/api/state", "/worker.stderr", "/cloud.jsonl", "/../credentials", "/api/start"):
        status, _, _ = call(server, path)
        assert status == 401
    status, headers, _ = call(server, "/")
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert call(server, "/api/state", host="attacker.example")[0] == 403


def test_roles_csrf_origin_logout_and_cookie_flags(portal):
    server, keys = portal
    assert call(server, "/api/login", {"credential": "bad"})[0] == 401
    assert (
        call(
            server, "/api/login", {"credential": keys["viewer"]}, origin="https://attacker.example"
        )[0]
        == 403
    )
    cookie, csrf = login(server, keys["viewer"])
    status, _, raw = call(server, "/api/state", cookie=cookie)
    assert status == 200 and not json.loads(raw)["operator"]
    assert "cloud_port" not in json.loads(raw)
    assert call(server, "/api/start", {}, cookie=cookie, csrf=csrf)[0] == 403
    assert call(server, "/api/logout", {}, cookie=cookie)[0] == 403
    assert call(server, "/api/logout", {}, cookie=cookie, csrf=csrf)[0] == 200
    assert call(server, "/api/state", cookie=cookie)[0] == 401
    status, headers, _ = call(server, "/api/login", {"credential": keys["operator"]})
    assert status == 200
    assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"]


def test_secure_cookie_and_auth_expiry():
    auth = Auth(credentials())
    key = secrets.token_urlsafe(32)
    auth = Auth({"viewer": key, "operator": secrets.token_urlsafe(32)})
    token, _, _ = auth.login(key)
    assert auth.get("care_session=" + token)[1][0] == "viewer"
    role, csrf, _ = auth.sessions[token]
    auth.sessions[token] = role, csrf, time.monotonic() - 1
    assert auth.get("care_session=" + token) == (None, None)
    assert token not in auth.sessions


def test_login_rate_limit_and_distinct_roles():
    keys = credentials()
    auth = Auth(keys)
    for _ in range(20):
        assert auth.login("bad") is None
    assert auth.login(keys["viewer"]) is None
    with pytest.raises(ValueError):
        Auth({"viewer": keys["viewer"], "operator": keys["viewer"]})


def emit(projection, event, ns, **fields):
    projection.accept({"event": event, "local_mono_ns": ns, **fields})


@pytest.mark.parametrize(
    "fault", ["none", "no_source", "no_ack", "rejected", "wrong_id", "invalid", "early"]
)
def test_causal_cycle_requires_source_applied_ack_and_later_valid_evidence(fault):
    p = Projection()
    emit(p, "observation", 1, observation_id="source", valid=True)
    emit(
        p,
        "agent_action",
        2,
        command_id="command",
        type="SET_TRANSMIT_FPS",
        caused_by_observation_id="unknown" if fault == "no_source" else "source",
    )
    if fault != "no_ack":
        emit(
            p,
            "ack_received",
            3,
            command_id="command",
            result="rejected" if fault == "rejected" else "applied",
        )
    assert p.snapshot()["causal_cycles"] == 0
    emit(
        p,
        "observation",
        2 if fault == "early" else 4,
        observation_id="new",
        valid=fault != "invalid",
        applied_command_id="other" if fault == "wrong_id" else "command",
    )
    assert p.snapshot()["causal_cycles"] == (1 if fault == "none" else 0)


def test_log_projection_does_not_expose_private_payloads_and_marks_stale(monkeypatch):
    p = Projection()
    secret = secrets.token_hex(20)
    emit(
        p,
        "server_ready",
        0,
        phase=1,
        cool_verified=True,
        opencv_version="4.0",
        runtime={"private_path": secret},
    )
    emit(p, "session_started", 1)
    emit(
        p,
        "observation",
        2,
        observation_id=secret,
        valid=True,
        image=secret,
        tracks=[secret],
        person_id=secret,
        processing_ms=float("nan"),
    )
    assert p.snapshot()["vision_live"]
    assert p.snapshot()["runtime"]["cool_verified"] is False
    assert secret not in json.dumps(p.snapshot())
    assert p.snapshot()["metrics"]["processing_ms"] is None
    p.last_observation -= 6
    assert not p.snapshot()["vision_live"]


def test_log_tail_partial_json_malformed_and_replacement(tmp_path):
    p = Projection()
    path = tmp_path / "cloud.jsonl"
    path.write_text('{"event":"frame_received"')
    tail = LogTail(path, p)
    tail.poll()
    assert p.counts["received"] == 0 and tail.offset == 0
    with path.open("a") as f:
        f.write("}\nnot-json\n")
    tail.poll()
    tail.poll()
    assert p.counts["received"] == 1 and p.malformed == 1
    path.unlink()
    path.write_text('{"event":"frame_dropped"}\n')
    tail.poll()
    assert p.counts["received"] == 0 and p.counts["dropped"] == 1 and p.counts["processed"] == 0


def test_portal_synthetic_session_executes_real_local_loop_and_closes_owned_children(portal):
    server, keys = portal
    cookie, csrf = login(server, keys["operator"])
    assert call(server, "/api/start", {}, cookie=cookie, csrf=csrf)[0] == 200
    assert call(server, "/api/start", {}, cookie=cookie, csrf=csrf)[0] == 409
    deadline = time.monotonic() + 15
    state = None
    while time.monotonic() < deadline:
        status, _, raw = call(server, "/api/state", cookie=cookie)
        state = json.loads(raw)
        if state["counts"]["processed"] >= 2:
            break
        time.sleep(0.1)
    assert state["counts"]["processed"] >= 2
    assert state["mode"] == "SYNTHETIC_LOCAL_NOT_COOL"
    assert state["runtime"]["cool_verified"] is False
    assert state["vision_live"]
    assert call(server, "/api/stop", {}, cookie=cookie, csrf=csrf)[0] == 200
    assert server.sessions.process.poll() is not None
    assert not server.sessions.snapshot()["vision_live"]
    assert server.sessions.snapshot()["session_state"] == "STOPPED"


def test_https_origin_sets_secure_session_cookie(portal):
    server, keys = portal
    server.origin = "https://review.example.org"
    server.host = "review.example.org"
    server.secure = True
    status, headers, _ = call(server, "/api/login", {"credential": keys["viewer"]})
    assert status == 200
    assert "; Secure" in headers["Set-Cookie"]
    assert "max-age=" in headers["Strict-Transport-Security"]


def test_cloud_worker_uses_explicit_interpreter_without_resolving_it(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    from unittest.mock import Mock

    from care_vision.judge import session as module

    python = tmp_path / "venv-python"
    python.symlink_to(sys.executable)
    monkeypatch.setenv("CARE_JUDGE_OPERATOR", secrets.token_urlsafe(32))
    fake = Mock()
    fake.poll.return_value = 0
    monkeypatch.setattr(module.subprocess, "Popen", fake)
    sessions = Sessions(
        tmp_path / "runs", cloud_python=str(python.absolute()), cloud_source=tmp_path
    )
    sessions.start()
    argv = fake.call_args.args[0]
    assert Path(argv[0]) == python
    assert "--phase" in argv and argv[argv.index("--phase") + 1] == "3"
    assert "CARE_JUDGE_OPERATOR" not in fake.call_args.kwargs["env"]
    sessions.stop()


def test_historical_evidence_cannot_make_live_vision_appear_current():
    p = Projection()
    emit(p, "session_started", 1)
    emit(p, "observation", 2, evidence_kind="historical", valid=True)
    assert not p.snapshot()["vision_live"]
    assert p.snapshot()["processed_fps"] is None
    assert p.counts["historical_processed"] == 1 and p.counts["live_processed"] == 0
    emit(p, "server_ready", 3, phase=3, cool_verified=True, opencv_version="5.0.0-dev")
    assert p.snapshot()["runtime"]["opencv"] == "5.0.0-dev"


@pytest.mark.parametrize(
    "address", ["0.0.0.0", "8.8.8.8", "127.0.0.1", "169.254.169.254", "::", "::1"]
)
def test_vpc_binding_rejects_public_wildcard_loopback_and_metadata_addresses(address):
    from care_vision.judge.__main__ import bind_address

    with pytest.raises(ValueError):
        bind_address(address, "https://review.example.org")


def test_vpc_binding_requires_https_and_keeps_default_loopback():
    from care_vision.judge.__main__ import bind_address

    assert bind_address(None, None) == "127.0.0.1"
    for origin in (None, "http://review.example.org"):
        with pytest.raises(ValueError):
            bind_address("10.1.2.3", origin)
    for address in ("10.1.2.3", "172.20.1.2", "192.168.1.8"):
        assert bind_address(address, "https://review.example.org") == address
