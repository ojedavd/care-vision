"""OpenCV DNN YOLOX adapter for the normalized person contract."""

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

from care_vision.cloud.person import PersonDetection


def validate_dnn_engine(engine):
    """Explicit classic is an SDK capability, not a portable OpenCV version promise."""
    if engine not in {"auto", "classic"}:
        raise ValueError("YOLOX dnn_engine must be auto or classic")
    if engine == "classic":
        if "OPENCV_FORCE_DNN_ENGINE" in os.environ:
            raise ValueError("unset OPENCV_FORCE_DNN_ENGINE before requesting classic")
        if not hasattr(cv2.dnn, "ENGINE_CLASSIC"):
            raise ValueError("this OpenCV SDK does not expose ENGINE_CLASSIC; no fallback")


class YoloXPersonDetector:
    """Detect COCO person class 0 using an official YOLOX ONNX model."""

    strides = (8, 16, 32)

    def __init__(
        self,
        model_path: str | Path,
        score_threshold=0.35,
        nms_threshold=0.5,
        input_size=640,
        dnn_engine="auto",
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLOX model not found: {model_path}")
        if not 0 < score_threshold <= 1 or not 0 < nms_threshold <= 1:
            raise ValueError("YOLOX thresholds must be in (0, 1]")
        if type(input_size) is not int or input_size <= 0 or input_size % max(self.strides):
            raise ValueError("YOLOX input_size must be a positive multiple of 32")
        validate_dnn_engine(dnn_engine)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        model_label = "s-2022nov" if input_size == 640 else f"{input_size}x{input_size}"
        self.backend = f"opencv-dnn-yolox-{model_label}:{digest[:12]}"
        self.dnn_engine_requested = dnn_engine
        if dnn_engine == "classic":
            self.backend += ":engine-classic"
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (input_size, input_size)
        self.net = (
            cv2.dnn.readNet(str(model_path), engine=cv2.dnn.ENGINE_CLASSIC)
            if dnn_engine == "classic" else cv2.dnn.readNet(str(model_path))
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.grids, self.expanded_strides = self._make_grids()

    def _make_grids(self):
        grids = []
        expanded_strides = []
        for stride in self.strides:
            side = self.input_size[0] // stride
            x, y = np.meshgrid(np.arange(side), np.arange(side))
            grid = np.stack((x, y), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
        return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)

    def _preprocess(self, image):
        height, width = image.shape[:2]
        scale = min(self.input_size[0] / height, self.input_size[1] / width)
        resized = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded = np.full((*self.input_size, 3), 114, dtype=np.float32)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        return np.transpose(padded, (2, 0, 1))[None], scale

    def detect(self, image: np.ndarray) -> list[PersonDetection]:
        blob, scale = self._preprocess(image)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        expected_shape = (1, self.grids.shape[1], 85)
        output = next((item for item in outputs if item.shape == expected_shape), None)
        if output is None:
            shapes = [tuple(item.shape) for item in outputs]
            raise ValueError(f"unsupported YOLOX output shapes: {shapes}")

        predictions = output.copy()
        predictions[:, :, :2] = (predictions[:, :, :2] + self.grids) * self.expanded_strides
        predictions[:, :, 2:4] = np.exp(predictions[:, :, 2:4]) * self.expanded_strides
        values = predictions[0]
        boxes = np.column_stack(
            (
                values[:, 0] - values[:, 2] / 2,
                values[:, 1] - values[:, 3] / 2,
                values[:, 2],
                values[:, 3],
            )
        )
        class_scores = values[:, 4:5] * values[:, 5:]
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        kept = cv2.dnn.NMSBoxesBatched(
            boxes.tolist(),
            scores.tolist(),
            class_ids.tolist(),
            self.score_threshold,
            self.nms_threshold,
        )

        height, width = image.shape[:2]
        detections = []
        for index in np.asarray(kept).reshape(-1):
            if class_ids[index] != 0:
                continue
            x, y, box_width, box_height = boxes[index] / scale
            x1 = int(np.clip(round(x), 0, width - 1))
            y1 = int(np.clip(round(y), 0, height - 1))
            x2 = int(np.clip(round(x + box_width), x1 + 1, width))
            y2 = int(np.clip(round(y + box_height), y1 + 1, height))
            detections.append(PersonDetection((x1, y1, x2 - x1, y2 - y1), float(scores[index])))
        return detections
