import os
import sqlite3
import stat
import time

import pytest

from care_vision.cloud.server import recovery_windows
from care_vision.edge.buffer import Capture
from care_vision.edge.source import VideoSource
from care_vision.edge.spool import EncryptedSpool
from care_vision.fixture import make_video


def test_encrypted_spool_is_bounded_and_survives_process_reopen(tmp_path):
    path = tmp_path / "private"
    key = bytes(range(32))
    start = time.monotonic_ns() - 10_000_000_000
    spool = EncryptedSpool(path, key, max_frames=3)
    boot_id = spool.boot_id
    for seq in range(1, 7):
        spool.append(
            Capture(seq, start + seq * 1_000_000_000, "2026-09-23T00:00:00Z", b"private-jpeg"),
            retain=True,
        )
    assert spool.status()["spool_frames"] == 3
    assert spool.status()["spool_evicted"] == 3
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert stat.S_IMODE((path / "offline.sqlite3").stat().st_mode) == 0o600
    assert b"private-jpeg" not in (path / "offline.sqlite3").read_bytes()
    spool.close()
    with pytest.raises(ValueError, match="key does not match"):
        EncryptedSpool(path, os.urandom(32))
    reopened = EncryptedSpool(path, key, max_frames=3)
    try:
        assert reopened.boot_id == boot_id
        assert reopened.last_seq == 6
        captures = reopened.select(start, start + 7_000_000_000, fps=2)
        assert [capture.seq for capture in captures] == [4, 5, 6]
        assert all(capture.jpeg == b"private-jpeg" for capture in captures)
        reopened.append(
            Capture(7, start + 7_000_000_000, "2026-09-23T00:00:00Z", b"online"),
            retain=False,
        )
        assert reopened.last_seq == 7
        assert reopened.status()["spool_frames"] == 3
    finally:
        reopened.close()

    with sqlite3.connect(path / "offline.sqlite3") as database:
        database.execute(
            "UPDATE meta SET value=? WHERE key='last_mono_ns'",
            (str(time.monotonic_ns() + 1_000_000_000),),
        )
    with pytest.raises(RuntimeError, match="monotonic clock reset"):
        EncryptedSpool(path, key)


def test_video_source_retains_offline_samples_across_restart(tmp_path):
    video = make_video(tmp_path / "static.avi", static=True, seconds=2)
    path = tmp_path / "private"
    key = bytes(range(32))
    source = VideoSource(str(video), spool=EncryptedSpool(path, key))
    source.start()
    source.thread.join(timeout=5)
    assert source.done and source.error is None and source.count == 60
    before = source.buffer_status()
    assert 3 <= before["spool_frames"] <= 5
    boot_id = source.boot_id
    source.close()

    resumed = VideoSource(str(video), spool=EncryptedSpool(path, key))
    assert resumed.boot_id == boot_id and resumed.count == 60
    retained = resumed.spool.select(
        before["spool_oldest_mono_ns"], before["spool_newest_mono_ns"] + 1, 30
    )
    assert len(retained) == before["spool_frames"]
    resumed.start()
    resumed.thread.join(timeout=5)
    assert resumed.done and resumed.error is None and resumed.count == 120
    assert resumed.buffer_status()["spool_frames"] > before["spool_frames"]
    resumed.close()


def test_spool_enforces_byte_and_age_budgets(tmp_path):
    spool = EncryptedSpool(
        tmp_path / "private",
        bytes(range(32)),
        max_seconds=1,
        max_bytes=60,
    )
    start = time.monotonic_ns() - 5_000_000_000
    try:
        for seq, offset in ((1, 0), (2, 500_000_000), (3, 2_000_000_000)):
            spool.append(
                Capture(seq, start + offset, "2026-09-23T00:00:00Z", b"x" * 40),
                retain=True,
            )
        status = spool.status()
        assert status["spool_frames"] == 1
        assert status["spool_encrypted_bytes"] <= 60
        assert status["spool_evicted"] == 2
    finally:
        spool.close()


def test_archived_samples_merge_with_ring_and_only_confirmed_samples_are_removed(tmp_path):
    spool = EncryptedSpool(tmp_path / "private", bytes(range(32)))
    source = VideoSource("unused.avi", spool=spool)
    start = time.monotonic_ns() - 12_000_000_000
    try:
        for seq in range(1, 13):
            capture = Capture(seq, start + seq * 500_000_000, "2026-09-23T00:00:00Z", b"jpeg")
            spool.append(capture, retain=True)
            if seq >= 9:
                source.buffer.append(capture)
        selected, status = source.recent(start, start + 7_000_000_000, 5)
        assert [capture.seq for capture in selected] == list(range(1, 13))
        assert status["archive_sparse"] is True
        assert source.acknowledge_captures([1, 2, 9]) == 3
        assert [capture.seq for capture in spool.select(start, start + 7_000_000_000, 5)] == [
            3,
            4,
            5,
            6,
            7,
            8,
            10,
            11,
            12,
        ]
    finally:
        spool.close()


def test_recovery_plan_splits_archived_prefix_and_preserves_ring_tail():
    status = {
        "spool_frames": 40,
        "spool_oldest_mono_ns": 1,
        "spool_newest_mono_ns": 24_000_000_000,
        "recovery_from_edge_mono_ns": 21_000_000_000,
        "recovery_to_edge_mono_ns": 30_000_000_000,
    }
    windows = recovery_windows(status)
    assert windows == [
        (1, 10_000_000_001, "archive"),
        (10_000_000_001, 20_000_000_001, "archive"),
        (20_000_000_001, 21_000_000_000, "archive"),
        (21_000_000_000, 30_000_000_000, "ring"),
    ]
