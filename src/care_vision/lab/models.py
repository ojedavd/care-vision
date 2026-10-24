"""Explicit, hash-checked reference models. Weights remain outside the checkout."""

import argparse
import hashlib
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

PERSON_SHA = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
POSE_SHA = 'a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f'
POSE_ZIP_SHA = '937003a70832d9cc34ea16927f504792f3133e92dda1b9c626236bbbe9e805cb'
PERSON_URL = ('https://media.githubusercontent.com/media/opencv/opencv_zoo/'
              '48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/'
              'object_detection_yolox_2022nov.onnx')
POSE_URL = ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
            'rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip')


def default_directory():
    return Path.home() / '.cache' / 'care-vision' / 'models'


def checked(path, digest):
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Model missing or hash mismatch. Run care-lab --setup-models.')
    return path


def model_paths(directory):
    directory = Path(directory)
    return (checked(directory / 'person.onnx', PERSON_SHA),
            checked(directory / 'pose.onnx', POSE_SHA))


def available(directory):
    # UI readiness only. Both complete hashes are checked before starting input.
    return all((Path(directory) / name).is_file() for name in ('person.onnx', 'pose.onnx'))


def download(url, destination):
    subprocess.run(['curl', '--fail', '--location', '--show-error', '--proto', '=https',
                    '--proto-redir', '=https', '--connect-timeout', '15', '--max-time', '180',
                    '--max-filesize', '73400320', '--output', str(destination), url], check=True)


def prepare(directory):
    directory = Path(directory).expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=directory) as temporary:
        stage = Path(temporary)
        try:
            checked(directory / 'person.onnx', PERSON_SHA)
        except ValueError:
            download(PERSON_URL, stage / 'person.onnx')
            checked(stage / 'person.onnx', PERSON_SHA)
            os.replace(stage / 'person.onnx', directory / 'person.onnx')
        try:
            checked(directory / 'pose.onnx', POSE_SHA)
        except ValueError:
            archive = stage / 'pose.zip'
            download(POSE_URL, archive)
            checked(archive, POSE_ZIP_SHA)
            with zipfile.ZipFile(archive) as bundle:
                entries = [item for item in bundle.infolist()
                           if Path(item.filename).name == 'end2end.onnx']
                if len(entries) != 1 or entries[0].file_size > 32 * 1024 * 1024:
                    raise ValueError('Unexpected pose model archive.')
                (stage / 'pose.onnx').write_bytes(bundle.read(entries[0]))
            checked(stage / 'pose.onnx', POSE_SHA)
            os.replace(stage / 'pose.onnx', directory / 'pose.onnx')
    model_paths(directory)
    print(f'Reference models verified: {directory}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=default_directory())
    prepare(parser.parse_args().directory)
