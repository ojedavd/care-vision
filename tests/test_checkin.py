from care_vision.checkin import distribution, replay_rows, resource_summary


def test_missing_metrics_are_unavailable_not_zero():
    assert distribution([])["median"] is None
    assert resource_summary([])["cpu_percent_one_core_mean"] is None
    assert distribution([1, 2, 3, 100])["p95"] == 100


def test_cpu_mean_is_weighted_by_actual_sample_interval():
    rows = [{"event": "process_resources", "interval_seconds": seconds,
             "cpu_percent_one_core": cpu, "rss_bytes": None, "peak_rss_bytes": 100}
            for seconds, cpu in ((1, 100), (3, 0))]
    assert resource_summary(rows)["cpu_percent_one_core_mean"] == 25
    assert resource_summary(rows)["rss_bytes"]["median"] is None


def test_replay_uses_availability_and_never_historical_capture_time():
    rows = [
        {"event": "frame_received", "local_mono_ns": 100_000_000_000},
        {"event": "observation", "local_mono_ns": 101_000_000_000,
         "captured_mono_ns": 9_000_000_000_000, "capture_seq": 300, "evidence_kind": "live"},
        {"event": "observation", "local_mono_ns": 102_000_000_000,
         "captured_mono_ns": 8_000_000_000_000, "capture_seq": 100,
         "evidence_kind": "historical"},
    ]
    result = replay_rows(rows)
    assert [r["t"] for r in result] == [1, 2]
    assert [r["capture_seq"] for r in result] == [300, 100]
