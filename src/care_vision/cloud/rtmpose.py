"""OpenCV DNN adapter for OpenMMLab RTMPose-t Body7 (17 COCO keypoints).

Preprocessing follows the model's SDK pipeline.json: RGB, 1.25 bbox padding,
192x256 affine crop, channel normalization, SimCC split ratio 2. No runtime
dependency on MMPose, PyTorch or ONNX Runtime. See docs/phase7d-pose-comparison.md.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np


class RTMPoseEstimator:
    input_size = (192, 256)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __init__(self, model_path):
        path = Path(model_path)
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.backend = f"opencv-dnn-rtmpose-t-body7:{self.model_sha256[:12]}"
        self.net = cv2.dnn.readNet(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @classmethod
    def preprocess(cls, image, bbox):
        box = np.asarray(bbox, dtype=np.float32)
        if box.shape != (4,) or not np.all(np.isfinite(box)) or np.any(box[2:] <= 0):
            raise ValueError("pose requires a finite positive xywh bbox")
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("pose requires a BGR image")
        center = box[:2] + box[2:] / 2
        scale = box[2:] * 1.25
        aspect = cls.input_size[0] / cls.input_size[1]
        if scale[0] > scale[1] * aspect:
            scale[1] = scale[0] / aspect
        else:
            scale[0] = scale[1] * aspect
        # Zero-rotation top-down affine; expanded crop may extend outside image.
        factors = np.asarray(cls.input_size, np.float32) / scale
        offset = np.asarray(cls.input_size, np.float32) / 2 - center * factors
        affine = np.array([[factors[0], 0, offset[0]], [0, factors[1], offset[1]]], np.float32)
        crop = cv2.warpAffine(image, affine, cls.input_size, flags=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = np.ascontiguousarray(((rgb - cls.mean) / cls.std).transpose(2, 0, 1)[None])
        return blob, center, scale

    @classmethod
    def decode(cls, outputs, center, scale):
        x = next((out for out in outputs if out.shape == (1, 17, 384)), None)
        y = next((out for out in outputs if out.shape == (1, 17, 512)), None)
        if x is None or y is None or len(outputs) != 2:
            raise ValueError("unsupported RTMPose output shapes")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("non-finite RTMPose outputs")
        indices = np.stack((x[0].argmax(axis=1), y[0].argmax(axis=1)), axis=1)
        scores = np.minimum(x[0].max(axis=1), y[0].max(axis=1))
        points = indices / 2 / np.asarray(cls.input_size) * scale + center - scale / 2
        # Scores are SimCC responses, not calibrated probabilities. Preserve them.
        return np.column_stack((points, scores))

    def estimate(self, image, bbox):
        blob, center, scale = self.preprocess(image, bbox)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self.decode(outputs, center, scale)
