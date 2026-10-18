import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "analyze_phase4_transport",
    Path(__file__).parents[1] / "tools" / "analyze_phase4_transport.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
analyze = MODULE.analyze


def event(kind, frame_id, seq, local_ns, **fields):
    return {
        "event": kind,
        "frame_id": frame_id,
        "capture_seq": seq,
        "local_mono_ns": local_ns,
        "evidence_kind": "live",
        **fields,
    }


def test_analysis_separates_transport_bunching_from_cloud_queue_replacement():
    edge = []
    cloud = []
    for index, receive_ns in enumerate((10_000_000_000, 10_000_020_000, 10_000_040_000)):
        seq = index + 1
        frame_id = f"boot:{seq}"
        edge.append(
            event(
                "frame_sent",
                frame_id,
                seq,
                1_000_000_000 + index * 50_000_000,
                target_transmit_fps=20,
                captured_mono_ns=900_000_000 + index * 50_000_000,
                sampled_mono_ns=999_000_000 + index * 50_000_000,
                payload_bytes=100,
            )
        )
        cloud.append(event("frame_received", frame_id, seq, receive_ns))
    cloud.insert(2, event("frame_dropped", "boot:1", 1, 10_000_030_000))
    cloud.extend(
        [
            event("observation", "boot:2", 2, 10_001_000_000),
            event("observation", "boot:3", 3, 10_002_000_000),
        ]
    )

    rows, summary = analyze(edge, cloud)

    assert summary["transport_frame_loss"] == 0
    assert summary["cloud_queue_replacements"] == 1
    assert summary["dropped_frames"][0]["replaced_by_frame_id"] == "boot:2"
    assert rows[1]["edge_send_gap_ms"] == 50
    assert rows[1]["cloud_receive_gap_ms"] == 0.02
    assert rows[1]["relative_transport_drift_ms"] == -49.98


def test_baseline_analysis_does_not_count_missing_processing_as_observed():
    edge, cloud = [], []
    for i in range(3):
        edge.append(event(
            "frame_sent", f"F{i}", i, int((10 + i * .5) * 1e9),
            target_transmit_fps=2, captured_mono_ns=int((10 + i * .5) * 1e9),
            sampled_mono_ns=int((10 + i * .5) * 1e9), payload_bytes=100,
        ))
        cloud.append(event("frame_received", f"F{i}", i, int((100 + i * .001) * 1e9)))
    cloud.append(event("observation", "F2", 2, 101_000_000_000))
    rows, summary = analyze(edge, cloud, target_fps=2)
    assert summary["live_2fps_frames_sent"] == 3
    assert summary["live_2fps_frames_observed"] == 1
    assert summary["unprocessed_without_drop_record"] == 2
    assert rows[-1]["relative_transport_drift_ms"] == -998
