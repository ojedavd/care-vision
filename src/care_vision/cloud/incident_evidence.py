"""Offline incident evidence ledger; never issues commands or confirms a fall.

Track IDs are scoped to one recording/view by the caller. Missing observations
mean unavailable evidence, not inactivity. A recovery proxy requires an earlier
candidate, a fresh upward cue and contiguous upright evidence for the SAME ID.
Identity continuity is a tracker assumption, not proof of physical identity.
"""

from copy import deepcopy
from dataclasses import dataclass

from care_vision.cloud.temporal_candidate import SequenceObservation, SequencePolicy


@dataclass
class _Context:
    last_visible: SequenceObservation
    visible: bool = True
    horizontal_since: float | None = None
    upright_since: float | None = None
    upward_at: float | None = None
    episode: dict | None = None


class IncidentEvidence:
    """One instance per isolated stream. Thresholds are experimental, not calibrated.

    update() takes all available person signals at one timestamp. A primary-only
    adapter must disclose that loss of selection also counts as unavailable.
    Context is retained for the whole finite offline recording, including IDs
    which disappear permanently. No spatial reassociation or timeout-to-safe.
    """

    def __init__(self, *, policy=SequencePolicy()):
        self.policy = policy
        self.contexts = {}
        self.episodes = []
        self.last_time = None
        self.last_frame = None
        self.finished = False

    def _lose(self, context, timestamp, frame, reason):
        context.visible = False
        context.horizontal_since = context.upright_since = context.upward_at = None
        episode = context.episode
        if episode is not None:
            if not episode["evidence_unavailable"]:
                episode["losses"].append({
                    "time_seconds": timestamp, "frame_index": frame, "reason": reason,
                })
            episode["evidence_unavailable"] = True

    def update(self, timestamp_seconds, frame_index, observations):
        # Validate the entire batch before mutating any state.
        SequenceObservation(timestamp_seconds, frame_index, None, None, "unknown", None)
        if self.finished:
            raise ValueError("recording already finished")
        if self.last_time is not None and (
            timestamp_seconds <= self.last_time or frame_index <= self.last_frame
        ):
            raise ValueError("frames and timestamps must strictly increase")
        observations = tuple(observations)
        seen = set()
        for o in observations:
            if (
                not isinstance(o, SequenceObservation)
                or o.timestamp_seconds != timestamp_seconds
                or o.frame_index != frame_index
                or o.track_id is None
                or o.track_id in seen
            ):
                raise ValueError("invalid or duplicate person observation in frame")
            seen.add(o.track_id)
        available = {o.track_id: o for o in observations if o.orientation is not None}
        events = []
        for track, context in self.contexts.items():
            gap = timestamp_seconds - context.last_visible.timestamp_seconds
            if track not in available or gap > self.policy.maximum_gap_seconds:
                self._lose(
                    context, timestamp_seconds, frame_index,
                    "person_evidence_unavailable" if track not in available else "observation_gap",
                )
        for track, o in available.items():
            context = self.contexts.setdefault(track, _Context(o))
            context.visible = True
            context.last_visible = o
            episode = context.episode
            if o.motion == "downward" and o.orientation != "horizontal":
                if episode is None:
                    episode = {
                        "episode_id": len(self.episodes) + 1, "track_id": track,
                        "descent_frame": frame_index, "descent_time": timestamp_seconds,
                        "last_descent_time": timestamp_seconds,
                        "candidate_frame": None, "candidate_time": None,
                        "recovery_frame": None, "recovery_time": None,
                        "losses": [], "evidence_unavailable": False,
                        "outcome": "UNRESOLVED", "end_reason": None,
                    }
                    self.episodes.append(episode)
                    context.episode = episode
                    events.append({"track_id": track, "event": "descent_observed"})
                episode["last_descent_time"] = timestamp_seconds
            if episode is None:
                continue
            episode["evidence_unavailable"] = False
            fresh_descent = (
                timestamp_seconds - episode["last_descent_time"]
                <= self.policy.descent_memory_seconds
            )
            if o.orientation == "horizontal" and fresh_descent:
                if context.horizontal_since is None:
                    context.horizontal_since = timestamp_seconds
                if (
                    episode["candidate_frame"] is None
                    and timestamp_seconds - context.horizontal_since
                    >= self.policy.horizontal_seconds
                ):
                    episode["candidate_frame"] = frame_index
                    episode["candidate_time"] = timestamp_seconds
                    events.append({"track_id": track, "event": "post_descent_candidate"})
            else:
                context.horizontal_since = None
            # Neither quietness nor an upright box alone resolves an episode.
            if episode["candidate_frame"] is not None:
                stale_upward = (
                    context.upward_at is not None
                    and timestamp_seconds - context.upward_at > self.policy.descent_memory_seconds
                )
                if stale_upward or o.motion == "downward" or o.orientation == "horizontal":
                    context.upward_at = context.upright_since = None
                if o.motion == "upward":
                    context.upward_at = timestamp_seconds
                if o.orientation == "vertical" and context.upward_at is not None:
                    if context.upright_since is None:
                        context.upright_since = timestamp_seconds
                    if (
                        timestamp_seconds - context.upward_at <= self.policy.descent_memory_seconds
                        and timestamp_seconds - context.upright_since >= self.policy.upright_seconds
                    ):
                        episode.update(
                            recovery_frame=frame_index, recovery_time=timestamp_seconds,
                            outcome="RECOVERY_PROXY_OBSERVED", end_reason="upward_then_upright",
                        )
                        events.append({"track_id": track, "event": "recovery_proxy_observed"})
                        context.episode = None
                        context.horizontal_since = None
                        context.upright_since = context.upward_at = None
                else:
                    context.upright_since = None
        self.last_time, self.last_frame = timestamp_seconds, frame_index
        states = []
        for track, context in sorted(self.contexts.items()):
            episode = context.episode
            if not context.visible:
                state = "INSUFFICIENT_EVIDENCE" if episode else "PERSON_UNOBSERVED"
            elif episode is None:
                state = "OBSERVING"
            elif episode["candidate_frame"] is not None:
                state = "POST_DESCENT_CANDIDATE"
            elif (
                timestamp_seconds - episode["last_descent_time"]
                <= self.policy.descent_memory_seconds
            ):
                state = "DESCENT_OBSERVED"
            else:
                state = "INSUFFICIENT_EVIDENCE"
            states.append({
                "track_id": track, "state": state, "visible": context.visible,
                "last_visible_time": context.last_visible.timestamp_seconds,
                "last_orientation": context.last_visible.orientation,
                "last_motion": context.last_visible.motion,
                "episode_id": episode["episode_id"] if episode else None,
                "suggested_observation": "REQUEST_CONTEXT" if episode else None,
            })
        return {
            "frame_index": frame_index, "source_time_seconds": timestamp_seconds,
            "tracks": states, "events": events,
            "unresolved_count": sum(c.episode is not None for c in self.contexts.values()),
            "fall_confirmed": False, "alert_enabled": False,
        }

    def finish(self):
        """Freeze unresolved evidence explicitly at EOF; never infer recovery."""
        for context in self.contexts.values():
            if context.episode is not None:
                context.episode["end_reason"] = "end_of_recording_unresolved"
        self.finished = True
        return deepcopy(self.episodes)
