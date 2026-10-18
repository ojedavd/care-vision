"""Replay cloud arrival order to compare rebuilt live-only and combined context."""

from collections import deque
from time import perf_counter

from care_vision.cloud.event_context import EventContext


def replay_event_contexts(records, *, association_policy="single_person"):
    contexts, snapshots = {}, []
    recent = deque(maxlen=512)
    last_clock = None
    live_session = None
    for record in records:
        now = record["local_mono_ns"]
        if last_clock is not None and now < last_clock:
            raise ValueError("cloud availability clock must not move backwards")
        last_clock = now
        changed = []
        if record["event"] == "observation":
            if record.get("evidence_kind", "live") == "live":
                if live_session != record["session_id"]:
                    recent.clear()
                    live_session = record["session_id"]
                if recent and record["captured_mono_ns"] <= recent[-1]["captured_mono_ns"]:
                    raise ValueError("live capture clock must increase")
                recent.append(record)
                while (
                    recent and record["captured_mono_ns"] - recent[0]["captured_mono_ns"] > 10**10
                ):
                    recent.popleft()
                for context in contexts.values():
                    if context.request["session_id"] == record["session_id"] and context.add(
                        record
                    ):
                        changed.append(context)
            else:
                context = contexts.get(record["buffer_request_id"])
                if context is None:
                    raise ValueError("historical observation precedes request")
                if context.complete:
                    raise ValueError("historical observation after transfer completion")
                if context.add(record):
                    changed.append(context)
        elif record["event"] == "agent_action" and record["type"] == "REQUEST_RECENT_BUFFER":
            if record["command_id"] in contexts or len(contexts) >= 16:
                raise ValueError("duplicate request or replay context capacity exceeded")
            trigger = next(
                (r for r in recent if r["observation_id"] == record["caused_by_observation_id"]),
                None,
            )
            if trigger is None:
                raise ValueError("request lacks already-available live anchor")
            context = EventContext(record, trigger, association_policy=association_policy)
            for row in recent:
                context.add(row)
            contexts[record["command_id"]] = context
            changed.append(context)
        elif record["event"] == "buffer_processed":
            context = contexts.get(record["buffer_request_id"])
            if context is None or context.complete:
                raise ValueError("unexpected or repeated history completion")
            historical = sum(r["evidence_kind"] == "historical" for r in context.frames.values())
            if not context.overflow and historical != record["frames_processed"]:
                raise ValueError("history completion does not match available observations")
            context.complete = True
            changed.append(context)
        for context in changed:
            for include_history in (False, True):
                started = perf_counter()
                snapshot = context.snapshot(now, include_history=include_history)
                snapshot["replay_processing_ms"] = (perf_counter() - started) * 1000
                snapshot["availability_event"] = record["event"]
                snapshots.append(snapshot)

    summaries = []
    for request_id, context in contexts.items():
        for variant in ("live_only_rebuilt", "live_plus_history"):
            revisions = [
                s for s in snapshots if s["request_id"] == request_id and s["variant"] == variant
            ]
            first = next((s for s in revisions if s["episodes"]), None)
            first_history_descent = next(
                (
                    s
                    for s in revisions
                    if s["historical_frames"] and s["signal_counts"].get("historical_downward", 0)
                ),
                None,
            )
            summaries.append(
                {
                    "request_id": request_id,
                    "variant": variant,
                    "revisions": len(revisions),
                    "history_transfer_complete": context.complete,
                    "first_candidate_available_after_request_ms": (
                        first["available_at_cloud_mono_ns"] - context.request["local_mono_ns"]
                    )
                    / 1e6
                    if first
                    else None,
                    "first_candidate_source_frame": first["episodes"][0]["candidate_frame"]
                    if first
                    else None,
                    "first_historical_descent_available_after_request_ms": (
                        first_history_descent["available_at_cloud_mono_ns"]
                        - context.request["local_mono_ns"]
                    )
                    / 1e6
                    if first_history_descent
                    else None,
                    "final": revisions[-1],
                }
            )
    return {
        "evaluation": "phase7h-causal-context-replay",
        "summaries": summaries,
        "snapshots": snapshots,
        "commands_enabled": False,
        "limits": {
            "frames_per_context": 512,
            "requests_per_replay": 16,
            "post_request_capture_seconds": EventContext.post_seconds,
        },
    }
