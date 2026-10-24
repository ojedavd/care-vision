"""Explicit local reference of person/pose/agent stages, not competition COOL evidence.

The unchanged cloud-role service runs in a separate process on loopback. Standard
OpenCV executes the existing YOLOX, tracking and deterministic episode controller.
Portable pose is a passive diagnostic; scene-specific seat closure is not enabled.
"""

import argparse
import asyncio
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from care_vision.cloud.pose_evidence import assess_pose_support
from care_vision.cloud.pose_shadow import PoseShadowObserver
from care_vision.cloud.rtmpose import RTMPoseEstimator
from care_vision.cloud.server import CloudServer
from care_vision.cloud.yolox import YoloXPersonDetector
from care_vision.common.metrics import EventLog
from care_vision.common.resources import monitor_resources
from care_vision.lab.models import model_paths


class ReferencePose:
    def __init__(self, model):
        self.estimator = RTMPoseEstimator(model)
        self.evaluator = None  # Existing observer resets this on stale processing.

    def process(self, observation, jpeg, episode):
        result = {
            'mode': 'LOCAL_REFERENCE', 'observation_id': observation['observation_id'],
            'capture_seq': observation['capture_seq'], 'width': observation['width'],
            'height': observation['height'], 'pose_status': 'not_requested',
            'points': None, 'response_support': None, 'torso_angle_degrees': None,
            'seat_support_verified': False, 'would_close_episode': False,
            'suggested_observation': None, 'evidence_status': 'INSUFFICIENT_EVIDENCE',
            'reason': 'single_confirmed_person_required', 'seated_pose_candidate': False,
        }
        people = observation.get('persons', [])
        if (observation.get('valid') is not True or observation.get('evidence_kind') != 'live'
                or observation.get('view_kind') != 'full_frame' or len(people) != 1
                or people[0].get('track_state') != 'confirmed'
                or people[0].get('association_only', False)):
            return result
        started = time.monotonic()
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (observation['height'], observation['width']):
            raise ValueError('Pose/image geometry mismatch.')
        bbox = people[0]['tracked_bbox']
        points = self.estimator.estimate(image, bbox)
        support = assess_pose_support(points, bbox, image.shape)
        angle = None
        if support['torso_supported']:
            torso = points[[5, 6], :2].mean(axis=0) - points[[11, 12], :2].mean(axis=0)
            angle = float(np.degrees(np.arctan2(abs(torso[0]), abs(torso[1]))))
        return {**result, 'pose_status': 'ok', 'points': points.tolist(),
                'response_support': support, 'torso_angle_degrees': angle,
                'processing_ms': (time.monotonic() - started) * 1000,
                'reason': 'pose_diagnostic_only_no_scene_calibration'}


async def serve(models, log, seconds):
    person, pose = model_paths(models)
    cv2.setNumThreads(4)
    cv2.ocl.setUseOpenCL(False)
    detector = YoloXPersonDetector(person, input_size=640, score_threshold=.2, nms_threshold=.3)
    estimator = ReferencePose(pose)
    detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))
    estimator.estimator.estimate(np.zeros((360, 640, 3), dtype=np.uint8), [100, 10, 150, 340])
    log.emit('reference_pipeline_ready', person_model_verified=True, pose_model_verified=True,
             cool_verified=False, pose_mode='PASSIVE_DIAGNOSTIC', scene_calibrated=False)
    service = CloudServer(
        log, person_detector_factory=lambda: detector, experimental_s_only_5fps=True,
        experimental_weak_association=True, experimental_episode_continuity=True,
        pose_shadow_factory=partial(PoseShadowObserver, processor=estimator),
    )
    async with monitor_resources(log):
        try:
            await asyncio.wait_for(service.run('127.0.0.1', 0, phase=1), seconds)
        except TimeoutError:
            log.emit('server_stopped', reason='reference_time_limit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=90)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 180:
        parser.error('Reference service duration must be 1..180 seconds.')
    log = EventLog('cloud', args.log)
    try:
        asyncio.run(serve(args.models, log, args.seconds))
    finally:
        log.close()


if __name__ == '__main__':
    main()
