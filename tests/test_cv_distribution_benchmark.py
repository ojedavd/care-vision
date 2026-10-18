"""A fixed input schedule must use recorded time, not nominal video FPS."""

from copy import deepcopy
from pathlib import Path

import pytest


def test_grid_uses_nearest_original_time_with_deterministic_ties(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from benchmark_cv_distributions import sample_indices

    assert sample_indices([100, 110, 149, 151, 199, 210], 50) == [0, 2, 4]


def test_sparse_grid_does_not_repeat_captures(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from benchmark_cv_distributions import sample_indices

    assert sample_indices([100, 1000], 50) == [0, 1]


@pytest.mark.parametrize('fault', [None, 'box', 'confidence', 'track', 'missing'])
def test_output_audit_detects_changed_detections_and_identity(monkeypatch, fault):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'tools'))
    from audit_cv_distributions import compare

    person = {'bbox': [10, 10, 50, 100], 'tracked_bbox': [10, 10, 50, 100],
              'confidence': .8, 'track_id': 'track1', 'track_state': 'confirmed'}
    left = [{'input': 'a.jpg', 'observation': {'persons': [person], 'person_count': 1}}]
    right = deepcopy(left)
    changed = right[0]['observation']['persons'][0]
    if fault == 'box':
        changed['bbox'][0] += 20
    elif fault == 'confidence':
        changed['confidence'] = .7
    elif fault == 'track':
        changed['track_id'] = 'track2'
    elif fault == 'missing':
        right[0]['observation']['persons'] = []
        right[0]['observation']['person_count'] = 0
    assert compare(left, right)['passed'] == (fault is None)
