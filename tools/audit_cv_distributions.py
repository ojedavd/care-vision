"""Report paired distribution timings and output disagreement without claiming kernel speedup."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from benchmark_cv_distributions import ORDER, save


def iou(a, b):
    x = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    intersection = x * y
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union else 0


def compare(left, right):
    differences, box_ious, confidence_deltas, motion_deltas = [], [], [], []
    if [r['input'] for r in left] != [r['input'] for r in right]:
        return {'passed': False, 'reason': 'input_order_mismatch'}
    for a, b in zip(left, right, strict=True):
        x, y = a['observation'], b['observation']
        faults = []
        for key in ('valid', 'invalid_reason', 'person_count', 'confirmed_person_count',
                    'person_detection_status', 'person_tracking_status', 'primary_track_id'):
            if x.get(key) != y.get(key):
                faults.append(key)
        if x.get('motion_score') is not None and y.get('motion_score') is not None:
            motion_deltas.append(abs(x['motion_score'] - y['motion_score']))
        remaining = list(y['persons'])
        for person in x['persons']:
            if not remaining:
                faults.append('unmatched_person')
                continue
            match = max(remaining, key=lambda p: iou(person['bbox'], p['bbox']))
            remaining.remove(match)
            overlap = min(iou(person['bbox'], match['bbox']),
                          iou(person['tracked_bbox'], match['tracked_bbox']))
            delta = abs(person['confidence'] - match['confidence'])
            box_ious.append(overlap)
            confidence_deltas.append(delta)
            if overlap < .99:
                faults.append('bbox_iou_below_99pct')
            if delta > .001:
                faults.append('confidence_delta_above_001')
            for key in ('track_id', 'track_state', 'association_only'):
                if person.get(key) != match.get(key):
                    faults.append(key)
        if remaining:
            faults.append('unmatched_person')
        if faults:
            differences.append({'input': a['input'], 'fields': sorted(set(faults))})
    return {'passed': not differences, 'different_frames': differences,
            'minimum_bbox_iou': min(box_ious, default=None),
            'maximum_confidence_delta': max(confidence_deltas, default=None),
            'maximum_motion_score_delta': max(motion_deltas, default=None),
            'scope': 'Specified detector/tracker fields; not ground truth or all trajectory fields'}


def distribution(values):
    ordered = sorted(values)
    return {'median': statistics.median(ordered), 'p95': ordered[math.ceil(.95*len(ordered))-1],
            'min': ordered[0], 'max': ordered[-1]}


def audit(folder):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((folder / 'plan.json').read_text())
    runs = [json.loads((folder / f'{i+1}-{mode}.json').read_text()) for i, mode in enumerate(ORDER)]
    summaries = []
    for run in runs:
        summaries.append({
            'label': run['label'], 'mode': run['mode'], 'frames': len(run['observations']),
            'processing_ms': distribution([r['seconds']*1000 for r in run['observations']]),
            'person_detection_ms': distribution([
                r['observation']['person_detection_ms'] for r in run['observations']]),
            'person_detection_percent_of_wall': 100 * sum(
                r['observation']['person_detection_ms']/1000 for r in run['observations']
            ) / run['wall_seconds'],
            'wall_seconds': run['wall_seconds'], 'cpu_seconds': run['cpu_seconds'],
            'throughput_fps': len(run['observations']) / run['wall_seconds'],
            'mean_cpu_percent_one_core': 100 * run['cpu_seconds'] / run['wall_seconds'],
            'rss_bytes_end': run['rss_bytes_end'], 'load_before': run['load_before'],
            'load_after': run['load_after'],
        })
    comparisons = []
    for i in range(1, len(runs)):
        comparisons.append({'left': runs[0]['label'], 'right': runs[i]['label'],
                            **compare(runs[0]['observations'], runs[i]['observations'])})
    runtime = runs[0]['runtime']
    paths = ['numpy_native_sha256', 'python_sha256', 'numpy_path', 'python_executable',
             'numpy_version', 'opencv_threads', 'cpu_affinity']
    deployment = json.loads((folder / 'deployment.json').read_text())
    checks = {
        'same_python_numpy_threads_affinity': all(all(r['runtime'][k] == runtime[k] for k in paths)
                                                 for r in runs),
        'same_instance': all(r['runtime']['ec2'] == runtime['ec2'] for r in runs),
        'arm64': runtime['ec2']['architecture'] == 'arm64',
        'all_inputs_processed_in_order': all([o['input'] for o in r['observations']]
            == [f['path'] for f in plan['frames']] for r in runs),
        'same_decoded_pixels': all(r['decoded_sha256'] == runs[0]['decoded_sha256'] for r in runs),
        'runtime_separated': all((r['runtime'].get('cool', {}).get('verified') is True
            if r['mode'] == 'cool' else not any('/opt/cool/' in p
                for p in r['runtime']['mapped_libraries'])) for r in runs),
        'all_detector_tracker_ok': all(o['observation']['person_detection_status']
            == o['observation']['person_tracking_status'] == 'ok'
            for r in runs for o in r['observations']),
        'source_code_inputs_unchanged': all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h
            for key in ('code_sha256', 'input_sha256') for p, h in plan[key].items()),
        'cleanup_and_existing_state': all(deployment[k] for k in
            ('complete', 'existing_states_unchanged', 'workspace_removed')),
    }
    aggregates = {}
    for mode in ('standard', 'cool'):
        matching = [r for r in summaries if r['mode'] == mode]
        aggregates[mode] = {
            k: statistics.mean(r[k] for r in matching)
            for k in ('wall_seconds', 'cpu_seconds', 'throughput_fps', 'rss_bytes_end')
        }
    ratio = {k: aggregates['standard'][k]/aggregates['cool'][k]
             for k in ('wall_seconds', 'cpu_seconds')}
    reductions = {k: 100*(1-aggregates['cool'][k]/aggregates['standard'][k])
                  for k in ('wall_seconds', 'cpu_seconds', 'rss_bytes_end')}
    result = {'measurement_checks_passed': all(checks.values()), 'checks': checks,
              'output_tolerance_passed': all(c['passed'] for c in comparisons),
              'runs': summaries, 'output_comparisons': comparisons, 'means': aggregates,
              'standard_over_cool_ratio': ratio, 'cool_reduction_percent': reductions,
              'claim': 'Installed-distribution comparison on same Graviton; OpenCV version/build '
                       'confounded. No isolated COOL kernel gain, ARM gain or cost saving.'}
    save(folder / 'assessment.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder), indent=2))
