"""Pixel geometry shared by Edge rendering and cloud coordinate transforms."""

from dataclasses import dataclass


def letterbox_geometry(source_width, source_height, output_width, output_height):
    if any(type(v) is not int or v <= 0 for v in (
        source_width, source_height, output_width, output_height,
    )):
        raise ValueError("image dimensions must be positive integers")
    scale = min(output_width / source_width, output_height / source_height)
    width, height = max(1, round(source_width * scale)), max(1, round(source_height * scale))
    return width, height, (output_width - width) // 2, (output_height - height) // 2


@dataclass(frozen=True)
class ImageGeometry:
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    crop: tuple

    @classmethod
    def from_header(cls, header):
        sw = header.get("source_width", header["width"])
        sh = header.get("source_height", header["height"])
        kind = header.get("view_kind", "full_frame")
        if kind not in {"roi", "full_frame"}:
            raise ValueError("unsupported view geometry")
        crop = tuple(header[k] for k in ("roi_x", "roi_y", "roi_width", "roi_height")) \
            if kind == "roi" else (0, 0, sw, sh)
        result = cls(sw, sh, header["width"], header["height"], crop)
        x, y, w, h = crop
        letterbox_geometry(sw, sh, result.output_width, result.output_height)
        if any(type(v) is not int for v in crop) or (
            x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > sw or y + h > sh
        ):
            raise ValueError("crop outside source geometry")
        return result

    @property
    def render(self):
        return letterbox_geometry(*self.crop[2:], self.output_width, self.output_height)

    def to_source(self, bbox):
        x, y, w, h = bbox
        cx, cy, cw, ch = self.crop
        rw, rh, left, top = self.render
        x1, x2 = [max(cx, min(cx + cw, cx + (v - left) * cw / rw)) for v in (x, x + w)]
        y1, y2 = [max(cy, min(cy + ch, cy + (v - top) * ch / rh)) for v in (y, y + h)]
        return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]

    def to_image(self, bbox):
        # Keep the complete tracked extent, even outside the crop. Clipping it
        # would silently change its meaning. Raw detections remain image-bounded.
        x, y, w, h = bbox
        cx, cy, cw, ch = self.crop
        rw, rh, left, top = self.render
        return [(x - cx) * rw / cw + left, (y - cy) * rh / ch + top,
                w * rw / cw, h * rh / ch]
