"""OpenCV DNN adapter for the OpenCV Zoo MediaPipe person detector."""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from care_vision.cloud.person import PersonDetection


class MediaPipePersonDetector:
    """Decode the OpenCV Zoo MediaPipe person detector into the common contract."""

    input_size = (224, 224)

    def __init__(self, model_path: str | Path, score_threshold=0.3, nms_threshold=0.3):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe person model not found: {model_path}")
        if not 0 < score_threshold <= 1 or not 0 < nms_threshold <= 1:
            raise ValueError("MediaPipe person thresholds must be in (0, 1]")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.backend = f"opencv-dnn-mediapipe-person-2023mar:{digest[:12]}"
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.net = cv2.dnn.readNet(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.anchors = self._make_anchors()

    @staticmethod
    def _make_anchors():
        anchors = []
        for side, repeats in ((28, 2), (14, 2), (7, 6)):
            for y in range(side):
                for x in range(side):
                    anchors.extend([((x + 0.5) / side, (y + 0.5) / side)] * repeats)
        return np.asarray(anchors, dtype=np.float32)

    def _preprocess(self, image):
        height, width = image.shape[:2]
        scale = min(self.input_size[0] / width, self.input_size[1] / height)
        resized_width = int(width * scale)
        resized_height = int(height * scale)
        normalized = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        normalized = (normalized - 0.5) * 2
        resized = cv2.resize(normalized, (resized_width, resized_height))
        left = (self.input_size[0] - resized_width) // 2
        top = (self.input_size[1] - resized_height) // 2
        padded = cv2.copyMakeBorder(
            resized,
            top,
            self.input_size[1] - resized_height - top,
            left,
            self.input_size[0] - resized_width - left,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        pad_x = int(left / scale)
        pad_y = int(top / scale)
        return np.transpose(padded, (2, 0, 1))[None], pad_x, pad_y

    def detect(self, image: np.ndarray) -> list[PersonDetection]:
        blob, pad_x, pad_y = self._preprocess(image)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        boxes_output = next((item for item in outputs if item.shape == (1, 2254, 12)), None)
        scores_output = next((item for item in outputs if item.shape == (1, 2254, 1)), None)
        if boxes_output is None or scores_output is None:
            shapes = [tuple(item.shape) for item in outputs]
            raise ValueError(f"unsupported MediaPipe person output shapes: {shapes}")

        logits = np.clip(scores_output[0, :, 0].astype(np.float64), -100, 100)
        scores = 1 / (1 + np.exp(-logits))
        box_delta = boxes_output[0, :, :4]
        center_delta = box_delta[:, :2] / self.input_size
        size_delta = box_delta[:, 2:] / self.input_size
        image_scale = max(image.shape[:2])
        xy1 = (center_delta - size_delta / 2 + self.anchors) * image_scale
        xy2 = (center_delta + size_delta / 2 + self.anchors) * image_scale
        boxes = np.column_stack(
            (
                xy1[:, 0] - pad_x,
                xy1[:, 1] - pad_y,
                xy2[:, 0] - pad_x,
                xy2[:, 1] - pad_y,
            )
        )
        kept = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.score_threshold,
            self.nms_threshold,
            top_k=5000,
        )
        height, width = image.shape[:2]
        detections = []
        for index in np.asarray(kept).reshape(-1):
            x1, y1, x2, y2 = boxes[index]
            x1 = int(np.clip(round(x1), 0, width - 1))
            y1 = int(np.clip(round(y1), 0, height - 1))
            x2 = int(np.clip(round(x2), x1 + 1, width))
            y2 = int(np.clip(round(y2), y1 + 1, height))
            detections.append(PersonDetection((x1, y1, x2 - x1, y2 - y1), scores[index]))
        return detections
