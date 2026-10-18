import pytest

from care_vision.edge.sampler import Sampler


@pytest.mark.parametrize("rate,expected", [(2, 20), (20, 200), (30, 300)])
def test_ten_seconds_at_thirty_capture_fps(rate, expected):
    sampler = Sampler(rate)
    selected = [i for i in range(300) if sampler.select(i + 1, round(i * 1e9 / 30))]
    assert abs(len(selected) - expected) <= 1
    assert len(set(selected)) == len(selected)


def test_rate_change_preserves_capture_identity():
    sampler = Sampler(2)
    baseline = sum(sampler.select(i + 1, round(i * 1e9 / 30)) for i in range(150))
    sampler.set_rate(20, 5_000_000_000)
    high = sum(sampler.select(i + 1, round(i * 1e9 / 30)) for i in range(150, 300))
    assert baseline == 10
    assert high == 100


def test_no_duplicate_or_catchup_after_stall():
    sampler = Sampler(20)
    assert sampler.select(1, 0)
    assert not sampler.select(1, 1_000_000_000)
    assert sampler.select(40, 2_000_000_000)
    assert not sampler.select(41, 2_000_000_001)
    assert sampler.select(42, 2_050_000_000)


def test_old_capture_not_selected_after_command():
    sampler = Sampler()
    sampler.set_rate(20, 1_000_000_000)
    assert not sampler.select(1, 999_999_999)
    assert sampler.select(2, 1_010_000_000)
