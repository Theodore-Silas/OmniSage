"""
Set-of-Mark (SoM) annotation (v4.0).

Draws numbered bounding boxes over a screenshot for every interactable
element. The VLM then returns ``element_id`` (== the element's data-agent-index),
which maps back to a precise DOM node — solving the "VLM coordinates are
unreliable" grounding problem.

Coordinate system: viewport screenshot, so element bounding boxes (which are
viewport-relative) align 1:1.
"""

from __future__ import annotations

import io
from typing import List

from PIL import Image, ImageDraw

from src.browser.driver.base import InteractiveElement


def annotate(
    screenshot_bytes: bytes,
    elements: List[InteractiveElement],
    color: tuple = (214, 45, 45),
) -> Image.Image:
    """Overlay numbered boxes onto a screenshot image."""
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    for e in elements:
        bb = e.bounding_box
        if not bb:
            continue
        x = int(bb.get("x", 0))
        y = int(bb.get("y", 0))
        w = int(bb.get("width", 0))
        h = int(bb.get("height", 0))
        if w < 2 or h < 2:
            continue

        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        label_y = max(0, y - 16)
        draw.rectangle([x, label_y, x + 22, label_y + 14], fill=color)
        draw.text((x + 3, label_y + 1), str(e.index), fill=(255, 255, 255))

    return img


def to_jpeg_bytes(img: Image.Image, quality: int = 70) -> bytes:
    """Encode an annotated image to JPEG bytes for VLM upload."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def annotate_and_encode(
    screenshot_bytes: bytes,
    elements: List[InteractiveElement],
    quality: int = 70,
) -> bytes:
    """One-shot: annotate + encode to JPEG bytes."""
    img = annotate(screenshot_bytes, elements)
    return to_jpeg_bytes(img, quality)
