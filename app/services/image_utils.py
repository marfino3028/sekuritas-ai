"""Small image helpers shared by the optional model engines."""
from __future__ import annotations

import struct
from dataclasses import dataclass

from app.services.errors import InvalidImageError


@dataclass(frozen=True)
class ImageInfo:
    format: str
    width: int
    height: int


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    # JPEG dimensions live in one of the Start Of Frame markers.  Walking the
    # headers lets stub mode reject arbitrary bytes without requiring Pillow.
    position = 2
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while position + 4 <= len(data):
        if data[position] != 0xFF:
            position += 1
            continue
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:  # compressed image data begins
            break
        if position + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[position : position + 2])[0]
        if segment_length < 2 or position + segment_length > len(data):
            break
        if marker in sof_markers and segment_length >= 7:
            height, width = struct.unpack(">HH", data[position + 3 : position + 7])
            return width, height
        position += segment_length
    raise InvalidImageError("JPEG header does not contain valid dimensions")


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 30:
        raise InvalidImageError("WebP image is truncated")
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    if chunk == b"VP8L" and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    raise InvalidImageError("Unsupported or malformed WebP image")


def inspect_image(data: bytes) -> ImageInfo:
    """Validate a JPEG, PNG, or WebP header and return its dimensions."""
    if not data:
        raise InvalidImageError("Image is empty")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) < 24 or data[12:16] != b"IHDR":
            raise InvalidImageError("PNG image is truncated or malformed")
        width, height = struct.unpack(">II", data[16:24])
        image_format = "png"
    elif data.startswith(b"\xff\xd8"):
        width, height = _jpeg_dimensions(data)
        image_format = "jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        width, height = _webp_dimensions(data)
        image_format = "webp"
    else:
        raise InvalidImageError("Only valid JPEG, PNG, and WebP images are supported")

    if width <= 0 or height <= 0:
        raise InvalidImageError("Image dimensions must be positive")
    if width * height > 50_000_000:
        raise InvalidImageError("Image dimensions are too large")
    return ImageInfo(format=image_format, width=width, height=height)


def decode_image(data: bytes):
    """Decode image bytes with OpenCV, imported only for real engines."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - requires model extras
        from app.services.errors import EngineUnavailableError

        raise EngineUnavailableError(
            "OpenCV and NumPy are required by the configured model engine"
        ) from exc

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError("Image cannot be decoded as a three-channel image")
    return image


def quality_metrics(image) -> tuple[bool, bool]:
    """Return conservative blur and low-light flags for a decoded BGR image."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(gray.mean())
    return blur_variance < 80.0, mean_brightness < 45.0
