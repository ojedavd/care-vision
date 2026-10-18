import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "pose_score", Path(__file__).parents[1] / "tools/score_pose_cache.py"
)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def test_anchor_distance_explicitly_ignores_side_and_preserves_response():
    points = np.zeros((17, 3))
    points[13] = [100, 100, 0.2]
    points[14] = [200, 200, 0.9]
    row = scorer.anchor_errors(points, [{"xy": [103, 104], "acceptable_indices": [13, 14]}])[0]
    assert row["closest_family_error_pixels"] == 5
    assert row["selected_joint_index"] == 13
    assert row["selected_joint_response"] == 0.2


def test_missing_pose_stays_unscored_instead_of_zero_error():
    row = scorer.anchor_errors(None, [{"xy": [10, 20], "acceptable_indices": [5, 6]}])[0]
    assert row["closest_family_error_pixels"] is None
    assert row["selected_joint_index"] is None


def test_nonfinite_pose_rejected():
    with pytest.raises(ValueError):
        scorer.anchor_errors(np.full((17, 3), np.nan), [])
