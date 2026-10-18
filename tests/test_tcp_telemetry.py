import json
import subprocess

import pytest

from care_vision import tcp_telemetry


@pytest.mark.parametrize('blocked', [False, True])
def test_collector_cleanup_on_failed_trial(monkeypatch, tmp_path, blocked):
    calls = []

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            calls.append('terminate')
            if not blocked:
                self.returncode = -15

        def wait(self, timeout=None):
            if blocked and timeout is not None:
                raise subprocess.TimeoutExpired('nettop', timeout)
            return self.returncode

        def kill(self):
            calls.append('kill')
            self.returncode = -9

    def launch(command, stdout, stderr):
        assert command[command.index('-p') + 1] == '1234'
        assert command[command.index('-L') + 1] == '300'
        assert '-d' not in command  # Preserve cumulative native counters.
        stdout.write('time,,bytes_out,re-tx\n')
        return Process()

    monkeypatch.setattr(tcp_telemetry.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(tcp_telemetry.shutil, 'which', lambda name: '/usr/bin/nettop')
    monkeypatch.setattr(tcp_telemetry.subprocess, 'Popen', launch)
    with pytest.raises(RuntimeError, match='trial failed'):
        with tcp_telemetry.mac_tunnel_telemetry(1234, tmp_path):
            raise RuntimeError('trial failed')
    assert calls == (['terminate', 'kill'] if blocked else ['terminate'])
    metadata = json.loads((tmp_path / 'mac-tcp-metadata.json').read_text())
    assert metadata['returncode'] == (-9 if blocked else -15)
    assert metadata['finished_mono_ns'] >= metadata['started_mono_ns']
    assert (tmp_path / 'mac-tcp.csv').read_text().startswith('time,')


def test_missing_collector_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(tcp_telemetry.platform, 'system', lambda: 'Linux')
    with pytest.raises(RuntimeError, match='requires macOS'):
        with tcp_telemetry.mac_tunnel_telemetry(1234, tmp_path):
            pytest.fail('must not start trial with missing requested telemetry')


@pytest.mark.parametrize('pid', [0, -1, True])
def test_collector_rejects_broad_or_invalid_process_selection(tmp_path, pid):
    with pytest.raises(ValueError):
        with tcp_telemetry.mac_tunnel_telemetry(pid, tmp_path):
            pytest.fail('invalid PID')
