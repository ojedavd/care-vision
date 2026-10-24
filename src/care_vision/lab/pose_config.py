"""Create an explicit external scene config for an evaluator's own camera view."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from care_vision.cloud.seated_candidate import SeatContext
from care_vision.cloud.seated_pose_candidate import SeatPoseContext


def create(model, output, *, seat=None, hips=None, reviewed=False):
    if reviewed and (seat is None or hips is None):
        raise ValueError('A reviewed scene requires both independently reviewed regions.')
    if not reviewed and (seat is not None or hips is not None):
        raise ValueError('Regions require an explicit --reviewed-scene declaration.')
    model, output = Path(model).expanduser().absolute(), Path(output).expanduser().absolute()
    geometry = SeatContext('edge-1', 1280, 720, tuple(seat or [0, 0, 1, 1]),
                           'Evaluator supplied regions' if reviewed else 'Unreviewed placeholder')
    hip_context = SeatPoseContext(tuple(hips or [0, 0, 1, 1]), geometry.provenance)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    value = {
        'model_path': str(model), 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'context_status': 'current' if reviewed else 'unknown',
        'context_provenance': geometry.provenance,
        'geometry': vars(geometry), 'hips': vars(hip_context),
    }
    # Scene configs never overwrite an earlier reviewed calibration.
    with output.open('x') as stream:
        os.chmod(output, 0o600)
        stream.write(json.dumps(value, indent=2) + '\n')
    print('Scene config created. Context: ' + value['context_status'])
    print('SHA-256: ' + hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seat-region', type=float, nargs=4)
    parser.add_argument('--hip-region', type=float, nargs=4)
    parser.add_argument('--reviewed-scene', action='store_true')
    args = parser.parse_args()
    create(args.model, args.output, seat=args.seat_region, hips=args.hip_region,
           reviewed=args.reviewed_scene)
