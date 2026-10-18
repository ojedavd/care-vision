"""Offline-only YOLO11 pose ONNX decoder and common person-crop transforms.

No hook into the live controller. Scores are model responses, not visibility.
"""

import cv2
import numpy as np

from care_vision.cloud.pose_evidence import pose_crop_bbox
from care_vision.evaluation import intersection_over_union


def common_crop(image, bbox):
    x, y, width, height = pose_crop_bbox(bbox)
    out_width, out_height = round(width), round(height)
    if min(out_width, out_height) < 2:
        raise ValueError("crop too small")
    affine = np.array(
        [
            [out_width / width, 0, -x * out_width / width],
            [0, out_height / height, -y * out_height / height],
        ],
        np.float32,
    )
    crop = cv2.warpAffine(image, affine, (out_width, out_height), flags=cv2.INTER_LINEAR)
    return crop, [x, y, width, height]


def restore_points(points, crop_box, crop_shape):
    points = np.asarray(points, dtype=np.float64).copy()
    if points.shape != (17, 3) or not np.all(np.isfinite(points)):
        raise ValueError("expected 17 finite points")
    x, y, width, height = crop_box
    points[:, 0] = x + points[:, 0] * width / crop_shape[1]
    points[:, 1] = y + points[:, 1] * height / crop_shape[0]
    return points


def letterbox(crop):
    height, width = crop.shape[:2]
    scale = min(640 / width, 640 / height)
    rw, rh = round(width * scale), round(height * scale)
    left, top = (640 - rw) // 2, (640 - rh) // 2
    resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(
        resized,
        top,
        640 - rh - top,
        left,
        640 - rw - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (640, 640), swapRB=True, crop=False)
    return blob, [rw / width, rh / height, left, top]


def decode(output, transform, expected_bbox, *, confidence=0.25, nms_iou=0.7):
    values = np.asarray(output)
    if values.shape != (1, 56, 8400) or not np.all(np.isfinite(values)):
        raise ValueError("unsupported YOLO11 pose outputs; expected finite 1x56x8400")
    rows = values[0].T
    rows = rows[rows[:, 4] >= confidence]
    if not len(rows):
        return None, {"status": "no_pose_detection", "candidates": []}
    boxes = rows[:, :4].copy()
    boxes[:, :2] -= boxes[:, 2:] / 2
    if np.any(boxes[:, 2:] <= 0):
        raise ValueError("non-positive YOLO pose boxes")
    kept = np.asarray(
        cv2.dnn.NMSBoxes(boxes.tolist(), rows[:, 4].tolist(), confidence, nms_iou)
    ).reshape(-1)
    sx, sy, left, top = transform
    candidates = []
    for index in kept:
        box = boxes[index].copy()
        box[:2] = (box[:2] - [left, top]) / [sx, sy]
        box[2:] /= [sx, sy]
        candidates.append(
            {
                "index": int(index),
                "bbox": box.tolist(),
                "confidence": float(rows[index, 4]),
                "target_iou": float(intersection_over_union(box, expected_bbox)),
            }
        )
    candidates.sort(key=lambda c: c["target_iou"], reverse=True)
    metadata = {"status": "no_corresponding_pose", "candidates": candidates}
    if not candidates or candidates[0]["target_iou"] < 0.3:
        return None, metadata
    if len(candidates) > 1 and candidates[0]["target_iou"] - candidates[1]["target_iou"] < 0.1:
        metadata["status"] = "ambiguous_pose_selection"
        return None, metadata
    points = rows[candidates[0]["index"], 5:].reshape(17, 3).copy()
    points[:, :2] = (points[:, :2] - [left, top]) / [sx, sy]
    if np.any((points[:, 2] < 0) | (points[:, 2] > 1)):
        raise ValueError("invalid YOLO joint scores")
    metadata["status"] = "ok"
    return points, metadata


class YoloPoseCandidate:
    def __init__(self, model):
        self.net = cv2.dnn.readNet(str(model))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def estimate(self, crop, expected_bbox):
        blob, transform = letterbox(crop)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        if len(outputs) != 1:
            raise ValueError("expected one YOLO pose output")
        return decode(outputs[0], transform, expected_bbox)
