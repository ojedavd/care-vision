"""Timing instrumentation must preserve the original Net calls and return values."""

from pathlib import Path

import pytest


def test_timed_net_is_transparent(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'tools'))
    from diagnose_dnn_distribution import TimedNet

    class Net:
        def __init__(self):
            self.calls = []

        def setInput(self, *args):
            self.calls.append(('setInput', args))
            return 'input-result'

        def forward(self, *args):
            self.calls.append(('forward', args))
            return output

        def getUnconnectedOutLayersNames(self):
            return ('output',)

    net, timings, output = Net(), {}, object()
    proxy = TimedNet(net, timings)
    assert proxy.setInput('blob') == 'input-result'
    assert proxy.forward(proxy.getUnconnectedOutLayersNames()) is output
    assert net.calls == [('setInput', ('blob',)), ('forward', (('output',),))]
    assert set(timings) == {'set_input_ms', 'forward_ms'}
    assert all(v >= 0 for v in timings.values())


@pytest.mark.parametrize('fault', [None, 'confidence', 'missing', 'input'])
def test_diagnostic_output_comparison_rejects_changed_results(monkeypatch, fault):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'tools'))
    from audit_dnn_diagnostic import compare

    a = [{'input': 'one.jpg', 'detections': [{'bbox': [1, 1, 20, 40], 'confidence': .8}]}]
    b = [{'input': 'one.jpg', 'detections': [{'bbox': [1, 1, 20, 40], 'confidence': .8}]}]
    if fault == 'confidence':
        b[0]['detections'][0]['confidence'] = .7
    elif fault == 'missing':
        b[0]['detections'] = []
    elif fault == 'input':
        b[0]['input'] = 'two.jpg'
    assert compare(a, b)['passed'] == (fault is None)
