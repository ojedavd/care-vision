"""Allowlisted projections of the lab's own generated run, without raw logs."""

import time
from collections import deque

from care_vision.judge.state import STATES, Projection, number

REASONS = {
    'recover_pre_event_evidence':
        'Retrieve buffered frames from before the motion change for temporal context.',
    'motion_above_high_threshold': 'Motion exceeded the experimental high threshold.',
    'motion_not_sustained': 'The motion signal was not sustained.',
    'sustained_motion_requires_more_evidence':
        'Sustained motion requires denser visual evidence; request 20 FPS.',
    'new_evidence_is_quiet_return_to_baseline':
        'New evidence stayed quiet long enough; return to the 2 FPS baseline.',
    'quiet_recovery_confirmed': 'Quiet recovery was sustained; return to NORMAL.',
    'motion_resumed_during_recovery': 'Motion resumed during recovery.',
    'observation_gap': 'An observation gap interrupted temporal evidence.',
    'invalid_observation': 'The observation was not valid for a temporal decision.',
    'posture_change_requires_more_evidence':
        'A tracked posture change requires more evidence; request 5 FPS.',
    'unresolved_episode_requires_more_observation':
        'The episode is unresolved; request another bounded observation window.',
    's_only_posture_assessed_return_to_baseline':
        'The observation window ended; reduce transmission while preserving episode status.',
    'confirmed_person_missing_verify':
        'A confirmed person is missing; request denser evidence.',
    'unresolved_episode_requires_scene_context':
        'Request the full scene for an unresolved episode.',
    'roi_person_unconfirmed': 'The person is unconfirmed in the crop; request a full frame.',
    'verify_scene_before_return_to_baseline': 'Verify the full scene before reducing observation.',
    'pose_uncertainty_requests_roi': 'Pose uncertainty requires a more detailed person crop.',
    'return_to_full_frame': 'Restore the full-scene view.',
    'episode_resolved': 'Sustained recovery evidence resolved the experimental episode.',
    'reference_posture_recovered': 'The tracked person sustained their reference posture.',
    'recovery_observed': 'Sustained same-track reference posture supports experimental recovery.',
    'experimental_seated_closure': 'Sustained seated pose resolved the posture-change episode.',
}


class VisionTelemetry(Projection):
    def __init__(self):
        super().__init__()
        self.latest = {}
        self.evidence = {}
        self.details = {}
        self.transitions = deque(maxlen=30)
        self.pipeline_checks = {}
        self.pose = {}
        self.last_pose = None

    def accept(self, row):
        super().accept(row)
        event = row.get('event')
        if event == 'observation' and row.get('evidence_kind', 'live') == 'live':
            self.latest = {
                key: number(row.get(key)) for key in
                ('motion_score', 'processing_ms', 'capture_seq', 'target_transmit_fps',
                 'person_count', 'confirmed_person_count', 'primary_person_confidence',
                 'person_detection_ms', 'width', 'height')
            }
            self.latest.update(valid=row.get('valid') is True)
            self.latest['person_detection_status'] = (
                row.get('person_detection_status') if row.get('person_detection_status')
                in {'ok', 'error', 'invalid_image', 'not_configured'} else 'unknown')
            identity = row.get('observation_id')
            if isinstance(identity, str):
                self.evidence[identity] = dict(self.latest)
                if len(self.evidence) > 240:
                    del self.evidence[next(iter(self.evidence))]
        elif event == 'agent_action' and row.get('command_id') in self.commands:
            identity = row['command_id']
            self.details[identity] = {
                'reason': REASONS.get(
                    row.get('reason'), 'Additional evidence requested by policy.'),
                'requested_fps': number(row.get('transmit_fps')),
                'source': self.evidence.get(row.get('caused_by_observation_id'), {}),
            }
            if len(self.details) > 200:
                del self.details[next(iter(self.details))]
        elif event == 'buffer_processed':
            identity = row.get('buffer_request_id')
            command = self.commands.get(identity)
            detail = self.details.get(identity)
            if (command and detail and command['action'] == 'REQUEST_RECENT_BUFFER'
                    and command['ack'] and number(row.get('frames_processed')) is not None):
                detail['history_frames'] = number(row.get('frames_processed'))
                detail['history_complete'] = (
                    row.get('coverage') == 'full'
                    and row.get('frames_processed') == row.get('frames_sent')
                    and row['frames_processed'] > 0
                )
        elif event == 'agent_state_transition':
            self.transitions.append({
                'state': self.agent_state,
                'reason': REASONS.get(row.get('reason'), 'Experimental policy transition.'),
            })
        elif event in {'reference_pipeline_ready', 'cloud_preflight_verified'}:
            self.pipeline_checks = {
                'person_model_verified': row.get('person_model_verified') is True,
                'pose_model_verified': row.get('pose_model_verified') is True,
                'source_verified': row.get('source_verified') is True,
                'context_status': row.get('context_status') if row.get('context_status')
                in {'current', 'unknown', 'changed'} else 'unknown',
                'seated_closure_seconds': number(row.get('seated_closure_seconds')),
            }
        elif event == 'pose_shadow':
            self.last_pose = time.monotonic()
            source = self.evidence.get(row.get('observation_id'), {})
            self.pose = {
                'capture_seq': number(row.get('capture_seq')),
                'status': row.get('pose_status') if row.get('pose_status')
                in {'ok', 'not_requested', 'inference_error'} else 'unknown',
                'processing_ms': number(row.get('processing_ms')),
                'torso_angle_degrees': number(row.get('torso_angle_degrees')),
                'points': None, 'seat_support_verified': False,
                'context_status': row.get('context_status') if row.get('context_status')
                in {'current', 'unknown', 'changed'} else 'unknown',
            }
            width, height, points = source.get('width'), source.get('height'), row.get('points')
            if (width and height and isinstance(points, list) and len(points) == 17
                    and all(isinstance(p, list) and len(p) == 3
                            and all(number(v) is not None for v in p) for p in points)):
                self.pose['points'] = [[max(-.5, min(1.5, p[0] / width)),
                                        max(-.5, min(1.5, p[1] / height)), p[2]] for p in points]
            support = row.get('response_support') or {}
            self.pose['torso_supported'] = support.get('torso_supported') is True
            self.pose['legs_supported'] = support.get('legs_supported') is True
        elif event == 'pose_shadow_error':
            self.last_pose = time.monotonic()
            self.pose = {'status': 'inference_error', 'points': None}
        elif event == 'cloud_status_sent' and row.get('agent_state') in STATES:
            self.agent_state = row['agent_state']

    def lab_snapshot(self):
        return {
            'observation': dict(self.latest),
            'commands': [
                {
                    'index': command['index'],
                    'action': command['action'],
                    'source_seen': command['caused_by_seen_observation'],
                    'ack_applied': command['ack'],
                    'new_perception': command['perceived'],
                    **self.details.get(identity, {}),
                }
                for identity, command in self.commands.items()
            ],
            'transitions': list(self.transitions),
            'pipeline_checks': dict(self.pipeline_checks),
            'pose': {**self.pose, 'age_seconds': None if self.last_pose is None
                     else time.monotonic() - self.last_pose},
        }


class EdgeTelemetry:
    def __init__(self):
        self.malformed = 0
        self.metrics = {}
        self.series = deque(maxlen=300)
        self.started_ns = None
        self.last_status_ns = None
        self.last_capture_seq = None
        self.cpu = None

    def accept(self, row):
        if row.get('event') == 'frame_sent' and row.get('evidence_kind', 'live') == 'live':
            self.last_capture_seq = number(row.get('capture_seq'))
        elif row.get('event') == 'process_resources':
            self.cpu = number(row.get('cpu_percent_one_core'))
        elif row.get('event') == 'edge_status':
            mono = number(row.get('local_mono_ns'))
            if mono is None:
                return
            if self.started_ns is None:
                self.started_ns = mono
            self.last_status_ns = mono
            self.metrics = {
                key: number(row.get(key)) for key in (
                    'capture_fps', 'transmit_fps', 'target_transmit_fps',
                    'effective_target_transmit_fps', 'configured_capture_fps', 'captured',
                    'selected', 'sent', 'dropped', 'bytes_sent', 'historical_sent',
                    'buffer_frames', 'buffer_bytes', 'network_rtt_ms', 'window_seconds',
                )
            }
            self.metrics.update(
                network=row.get('network_quality') if row.get('network_quality') in
                {'GOOD', 'DEGRADED', 'OFFLINE'} else 'UNKNOWN',
                source_state=row.get('source_state') if row.get('source_state') in
                {'running', 'ended', 'error'} else 'unknown',
            )
            self.series.append({
                'seconds': (mono - self.started_ns) / 1e9,
                'capture': self.metrics['capture_fps'],
                'requested': self.metrics['target_transmit_fps'],
                'transmitted': self.metrics['transmit_fps'],
            })

    def snapshot(self, running):
        # All timestamps belong to owned processes on this same local host.
        age = None if self.last_status_ns is None else max(
            0, (time.monotonic_ns() - self.last_status_ns) / 1e9
        )
        return {
            'metrics': dict(self.metrics),
            'series': list(self.series),
            'cpu_percent_one_core': self.cpu,
            'status_age_seconds': age,
            'fresh': running and age is not None and age < 3,
            'preview_capture_seq': self.last_capture_seq,
        }
