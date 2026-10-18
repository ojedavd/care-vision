"""OpenCV DNN NanoDet adapter for the normalized person contract."""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from care_vision.cloud.person import PersonDetection


class NanoDetPersonDetector:
    """Detect COCO class 0 using the OpenCV Zoo NanoDet model.

    Pre/postprocessing follows the Apache-2.0 OpenCV Zoo reference implementation:
    https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet
    """

    input_size = (416, 416)
    strides = (8, 16, 32, 64)
    reg_max = 7

    def __init__(self, model_path: str | Path, score_threshold=0.35, nms_threshold=0.3):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"NanoDet model not found: {model_path}")
        if not 0 < score_threshold <= 1 or not 0 < nms_threshold <= 1:
            raise ValueError("NanoDet thresholds must be in (0, 1]")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.backend = f"opencv-dnn-nanodet-2022nov:{digest[:12]}"
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.net = cv2.dnn.readNet(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.project = np.arange(self.reg_max + 1, dtype=np.float32)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.anchors = self._make_anchors()

    def _make_anchors(self):
        levels = []
        for stride in self.strides:
            size = self.input_size[0] // stride
            x, y = np.meshgrid(np.arange(size), np.arange(size))
            offset = 0.5 * (stride - 1)
            levels.append(
                np.column_stack(((x.ravel() * stride) + offset, (y.ravel() * stride) + offset))
            )
        return levels

    def _letterbox(self, image):
        height, width = image.shape[:2]
        scale = min(self.input_size[0] / width, self.input_size[1] / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        resized = cv2.resize(image, (resized_width, resized_height), cv2.INTER_AREA)
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
        return padded, scale, left, top

    def _decode(self, outputs):
        score_outputs = {
            output.shape[1]: np.squeeze(output, axis=0)
            for output in outputs
            if output.ndim == 3 and output.shape[2] == 80
        }
        box_outputs = {
            output.shape[1]: np.squeeze(output, axis=0)
            for output in outputs
            if output.ndim == 3 and output.shape[2] == 4 * (self.reg_max + 1)
        }
        if not score_outputs or score_outputs.keys() != box_outputs.keys():
            shapes = [tuple(output.shape) for output in outputs]
            raise ValueError(f"unsupported NanoDet output shapes: {shapes}")
        boxes_at_levels = []
        person_scores_at_levels = []
        for stride, anchors in zip(self.strides, self.anchors):
            count = len(anchors)
            if count not in score_outputs:
                continue
            class_scores = score_outputs[count]
            box_logits = box_outputs[count]
            logits = box_logits.reshape(-1, self.reg_max + 1)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            distances = (probabilities @ self.project).reshape(-1, 4) * stride
            boxes = np.column_stack(
                (
                    anchors[:, 0] - distances[:, 0],
                    anchors[:, 1] - distances[:, 1],
                    anchors[:, 0] + distances[:, 2],
                    anchors[:, 1] + distances[:, 3],
                )
            )
            boxes_at_levels.append(np.clip(boxes, 0, self.input_size[0]))
            person_scores_at_levels.append(class_scores[:, 0])
        if not boxes_at_levels:
            raise ValueError("NanoDet outputs do not match any configured feature level")
        return np.concatenate(boxes_at_levels), np.concatenate(person_scores_at_levels)

    def detect(self, image: np.ndarray) -> list[PersonDetection]:
        letterboxed, scale, left, top = self._letterbox(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        normalized = (letterboxed.astype(np.float32) - self.mean) / self.std
        self.net.setInput(cv2.dnn.blobFromImage(normalized))
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        boxes, scores = self._decode(outputs)
        xywh = boxes.copy()
        xywh[:, 2:] -= xywh[:, :2]
        kept = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores.tolist(),
            self.score_threshold,
            self.nms_threshold,
        )
        height, width = image.shape[:2]
        detections = []
        for index in np.asarray(kept).reshape(-1):
            x1, y1, x2, y2 = boxes[index]
            x1 = int(np.clip(round((x1 - left) / scale), 0, width - 1))
            y1 = int(np.clip(round((y1 - top) / scale), 0, height - 1))
            x2 = int(np.clip(round((x2 - left) / scale), x1 + 1, width))
            y2 = int(np.clip(round((y2 - top) / scale), y1 + 1, height))
            detections.append(PersonDetection((x1, y1, x2 - x1, y2 - y1), float(scores[index])))
        return detections
