"""Experimental end-of-window posture evidence; never an incident decision."""

from collections import defaultdict, deque


class PostureEvidence:
    WINDOW_NS = 2_000_000_000
    RECENT_NS = 500_000_000

    def __init__(self):
        self.samples = deque()

    def observe(self, observation):
        now_ns = observation["captured_mono_ns"]
        self.samples.append(
            (
                now_ns,
                observation.get("primary_track_id")
                if observation.get("confirmed_person_detected")
                else None,
                observation.get("primary_orientation"),
            )
        )
        while self.samples and now_ns - self.samples[0][0] > self.WINDOW_NS:
            self.samples.popleft()

    def classify(self, now_ns):
        by_track = defaultdict(lambda: {"horizontal": [], "vertical": []})
        for timestamp, track, orientation in self.samples:
            if (
                track
                and orientation in {"horizontal", "vertical"}
                and now_ns - timestamp <= self.WINDOW_NS
            ):
                by_track[track][orientation].append(timestamp)
        for track, counts in by_track.items():
            horizontal = counts["horizontal"]
            if len(horizontal) >= 6 and now_ns - horizontal[-1] <= self.RECENT_NS:
                return {
                    "outcome": "horizontal_posture_observed",
                    "track_id": track,
                    "horizontal_frames": len(horizontal),
                    "vertical_frames": len(counts["vertical"]),
                    "window_seconds": 2,
                }
        for track, counts in by_track.items():
            vertical = counts["vertical"]
            if (
                len(vertical) >= 8
                and not counts["horizontal"]
                and now_ns - vertical[-1] <= self.RECENT_NS
            ):
                return {
                    "outcome": "upright_posture_observed",
                    "track_id": track,
                    "horizontal_frames": 0,
                    "vertical_frames": len(vertical),
                    "window_seconds": 2,
                }
        return {
            "outcome": "unresolved",
            "track_id": None,
            "horizontal_frames": max((len(x["horizontal"]) for x in by_track.values()), default=0),
            "vertical_frames": max((len(x["vertical"]) for x in by_track.values()), default=0),
            "window_seconds": 2,
        }
