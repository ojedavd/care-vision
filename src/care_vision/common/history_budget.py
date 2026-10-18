"""Bounded transfer allowance compatible with the lowest supported wire budget."""

MIN_HISTORY_FPS = 2
MIN_HISTORY_BYTES_PER_SECOND = 64 * 1024
MAX_HISTORY_SECONDS = 180
HISTORY_IDLE_SECONDS = 30


def history_timeout_seconds(frames, payload_bytes, sample_fps):
    # Sum, rather than max, upper-bounds mixed large/small frame pacing. Reserve
    # 4 KiB per message for metadata and time for live priority and processing.
    paced_seconds = frames / min(sample_fps, MIN_HISTORY_FPS)
    paced_seconds += (payload_bytes + frames * 4096) / MIN_HISTORY_BYTES_PER_SECOND
    return min(MAX_HISTORY_SECONDS, max(30, 10 + 1.25 * paced_seconds))
