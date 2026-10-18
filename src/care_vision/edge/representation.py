"""Execute cloud-selected full-frame or ROI representations without inference."""

from dataclasses import dataclass

import cv2
import numpy as np

from care_vision.edge.buffer import fit_to_canvas


@dataclass
class VisualPolicy:
    source_width: int = 640
    source_height: int = 360
    output_width: int = 640
    output_height: int = 360
    view_kind: str = "full_frame"
    roi: tuple[int, int, int, int] | None = None
    command_id: str | None = None
    policy_seq: int = 0
    effective_from_capture_seq: int = 1

    context_fps: float = 0
    last_render_ns: int | None = None
    last_context_bucket: int | None = None

    def apply(self, command, effective_from_capture_seq=1):
        self.context_fps = command.get("context_fps", 0) if command["type"] == "REQUEST_ROI" else 0
        self.last_context_bucket = (
            int(self.last_render_ns * self.context_fps / 1e9)
            if self.context_fps and self.last_render_ns is not None
            else None
        )
        self.output_width = command["output_width"]
        self.output_height = command["output_height"]
        self.command_id = command["command_id"]
        self.policy_seq = command["command_seq"]
        self.effective_from_capture_seq = effective_from_capture_seq
        if command["type"] == "REQUEST_ROI":
            self.view_kind = "roi"
            self.roi = (
                command["roi_x"],
                command["roi_y"],
                command["roi_width"],
                command["roi_height"],
            )
        else:
            self.view_kind = "full_frame"
            self.roi = None

    def metadata(self):
        result = {
            "view_kind": self.view_kind,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "visual_policy_seq": self.policy_seq,
            "applied_visual_command_id": self.command_id,
        }
        if self.roi is not None:
            result.update(
                roi_x=self.roi[0],
                roi_y=self.roi[1],
                roi_width=self.roi[2],
                roi_height=self.roi[3],
            )
        return result

    def render(self, jpeg, captured_mono_ns=None):
        if self.context_fps and captured_mono_ns is None:
            raise ValueError("mixed ROI policy requires a capture timestamp")
        if captured_mono_ns is not None:
            if self.last_render_ns is not None and captured_mono_ns <= self.last_render_ns:
                raise ValueError("visual capture timestamps must strictly increase")
            self.last_render_ns = captured_mono_ns
        if self.view_kind == "roi" and self.context_fps:
            bucket = int(captured_mono_ns * self.context_fps / 1e9)
            if bucket != self.last_context_bucket:
                self.last_context_bucket = bucket
                context = VisualPolicy(
                    self.source_width,
                    self.source_height,
                    self.output_width,
                    self.output_height,
                    command_id=self.command_id,
                    policy_seq=self.policy_seq,
                )
                return context.render(jpeg)
        if (
            self.view_kind == "full_frame"
            and self.output_width == self.source_width
            and self.output_height == self.source_height
        ):
            return jpeg, self.metadata()
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (self.source_height, self.source_width):
            raise ValueError("capture does not match visual policy source dimensions")
        if self.roi is not None:
            x, y, width, height = self.roi
            image = image[y : y + height, x : x + width]
        canvas = fit_to_canvas(image, self.output_width, self.output_height)
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("visual representation JPEG encoding failed")
        return encoded.tobytes(), self.metadata()
