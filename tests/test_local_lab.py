import http.client
import json
import threading
import time

import pytest

from care_vision.judge.state import LogTail
from care_vision.lab.__main__ import LabServer
from care_vision.lab.session import LabSessions
from care_vision.lab.telemetry import EdgeTelemetry, VisionTelemetry


@pytest.fixture
def lab(tmp_path):
    sessions = LabSessions(tmp_path / 'runs')
    server = LabServer(sessions)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def call(server, path, *, method='GET', token=None, origin=None, host=None):
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
    headers = {'Host': host or server.host}
    if token:
        headers['X-CSRF-Token'] = token
    if origin:
        headers['Origin'] = origin
    connection.request(method, path, '{}' if method == 'POST' else None, headers)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_loopback_assets_host_checks_and_no_file_or_camera_access(lab):
    status, headers, body = call(lab, '/')
    assert status == 200 and b'lang="en"' in body
    assert headers['Cache-Control'] == 'no-store'
    assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
    for path in ('/app.js', '/style.css'):
        assert call(lab, path)[0] == 200
    for path in ('/demo.avi', '/cloud.jsonl', '/api/preview.jpg?path=/etc/passwd',
                 '/../pyproject.toml', '/api/camera', '/api/upload'):
        assert call(lab, path)[0] == 404
    assert call(lab, '/api/state', host='attacker.invalid')[0] == 403
    state = json.loads(call(lab, '/api/state')[2])
    assert state['mode'] == 'SYNTHETIC_LOCAL_NOT_COOL'
    assert not state['runtime']['cool_verified'] and not state['vision_live']
    assert state['commands'] == [] and state['edge']['series'] == []
    assert 'cloud_port' not in state


def test_mutations_require_same_origin_and_local_token(lab):
    token = json.loads(call(lab, '/api/state')[2])['csrf']
    assert call(lab, '/api/start', method='POST')[0] == 403
    assert call(lab, '/api/start', method='POST', token=token,
                origin='https://attacker.invalid')[0] == 403
    assert lab.sessions.process is None
    assert call(lab, '/api/unknown', method='POST', token=token,
                origin=lab.origin)[0] == 404


def test_edge_telemetry_preserves_actual_rate_and_expires(monkeypatch):
    projection = EdgeTelemetry()
    monkeypatch.setattr(time, 'monotonic_ns', lambda: 12_000_000_000)
    projection.accept({'event': 'edge_status', 'local_mono_ns': 10_000_000_000,
                       'capture_fps': 30, 'transmit_fps': 2, 'target_transmit_fps': 20,
                       'network_quality': 'GOOD', 'sent': 10, 'bytes_sent': 12345,
                       'arbitrary_secret': 'must not be projected'})
    s = projection.snapshot(True)
    assert s['metrics']['capture_fps'] == 30
    assert s['metrics']['transmit_fps'] == 2
    assert s['metrics']['target_transmit_fps'] == 20
    assert s['fresh']
    assert 'arbitrary_secret' not in json.dumps(s)
    assert not projection.snapshot(False)['fresh']
    monkeypatch.setattr(time, 'monotonic_ns', lambda: 14_000_000_000)
    assert not projection.snapshot(True)['fresh']


def test_command_inspection_requires_applied_ack_and_new_valid_evidence():
    p = VisionTelemetry()
    def emit(event, mono, **fields):
        p.accept({'event': event, 'local_mono_ns': mono, **fields})
    emit('observation', 1, observation_id='source', capture_seq=241,
         motion_score=.1, valid=True)
    emit('agent_action', 2, command_id='cmd', type='SET_TRANSMIT_FPS',
         transmit_fps=20, caused_by_observation_id='source',
         reason='sustained_motion_requires_more_evidence')
    emit('ack_received', 3, command_id='cmd', result='rejected')
    assert not p.lab_snapshot()['commands'][0]['ack_applied']
    emit('ack_received', 4, command_id='cmd', result='applied')
    assert not p.lab_snapshot()['commands'][0]['new_perception']
    emit('observation', 5, observation_id='later', applied_command_id='cmd', valid=False)
    assert not p.lab_snapshot()['commands'][0]['new_perception']
    emit('observation', 6, observation_id='valid', applied_command_id='cmd', valid=True)
    command = p.lab_snapshot()['commands'][0]
    assert command['source']['capture_seq'] == 241 and command['new_perception']
    assert command['reason'].startswith('Sustained motion')


def test_preview_is_only_a_frame_from_the_owned_generated_fixture(tmp_path):
    from care_vision.fixture import make_video
    sessions = LabSessions(tmp_path / 'runs')
    run = sessions.root / 'generated'
    run.mkdir()
    sessions.tail = LogTail(run / 'cloud.jsonl', sessions.projection)
    assert sessions.preview() is None
    make_video(run / 'demo.avi', seconds=.1)
    sessions.edge.last_capture_seq = 1
    frame = sessions.preview()
    assert frame.startswith(b'\xff\xd8\xff')
    assert sessions.preview() == frame
    sessions.edge.last_capture_seq = 2000
    assert sessions.preview() is None


@pytest.mark.integration
def test_local_lab_owns_a_complete_run_and_can_restart(lab):
    token = json.loads(call(lab, '/api/state')[2])['csrf']
    def control(action):
        return call(lab, '/api/' + action, method='POST', token=token, origin=lab.origin)[0]
    assert control('start') == 200
    assert control('start') == 409
    deadline = time.monotonic() + 80
    saw_live = saw_preview = False
    states = []
    while time.monotonic() < deadline:
        state = json.loads(call(lab, '/api/state')[2])
        states.append(state)
        if state['vision_live']:
            saw_live = True
        if state['edge']['preview_capture_seq'] and not saw_preview:
            status, _, image = call(lab, '/api/preview.jpg')
            saw_preview = status == 200 and image.startswith(b'\xff\xd8\xff')
        if state['session_state'] in {'COMPLETED', 'FAILED', 'TIME_LIMIT'}:
            break
        time.sleep(.5)
    assert state['session_state'] == 'COMPLETED', state
    assert state['report']['passed'] and state['report']['buffer_verified']
    assert saw_live and saw_preview
    assert state['counts']['processed'] == state['report']['observations']
    assert state['causal_cycles'] >= 2
    assert not state['vision_live'] and not state['edge']['fresh']
    assert any(s['edge']['metrics'].get('target_transmit_fps') == 20 for s in states)
    assert max(p['transmitted'] for p in state['edge']['series']) > 15
    sensing = [c for c in state['commands'] if c['action'] == 'SET_TRANSMIT_FPS']
    assert len(sensing) >= 2 and all(c['new_perception'] for c in sensing)
    history = [c for c in state['commands'] if c['action'] == 'REQUEST_RECENT_BUFFER']
    assert history and all(c['history_complete'] for c in history)
    assert control('start') == 200
    restarted = json.loads(call(lab, '/api/state')[2])
    assert restarted['commands'] == [] and restarted['report'] is None
    assert control('stop') == 200
    assert lab.sessions.process.poll() is not None
    stopped = json.loads(call(lab, '/api/state')[2])
    assert stopped['session_state'] == 'STOPPED' and not stopped['vision_live']
