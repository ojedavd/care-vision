import pytest

from care_vision.netem_probe import check_routes, validate_loss


@pytest.mark.parametrize("loss", [-1, 6, float("nan"), float("inf")])
def test_reject_unbounded_loss(loss):
    with pytest.raises(ValueError):
        validate_loss(loss)


@pytest.mark.parametrize("route", ["198.19.240.0/30", "198.19.0.0/16", "198.19.240.1"])
def test_refuse_overlapping_existing_routes(route):
    with pytest.raises(ValueError, match="overlaps"):
        check_routes([{"dst": route}])


def test_normal_routes_and_bounded_loss():
    check_routes([{"dst": "default"}, {"dst": "172.31.32.0/20"}, {"dst": "::1"}])
    for loss in (0, 1, 3, 5):
        validate_loss(loss)
