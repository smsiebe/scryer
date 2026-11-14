"""base.py - Base parser class and common utilities for Phase 1 deterministic extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import hashlib
import io
import math
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from scryer.contracts import (
    DocumentMetadata,
    ExtractedVisualElement,
    Phase1Output,
    ScryerExtractionOptions,
)
from scryer.protocols import DeterministicParserProtocol


_MIME_TO_PIL_FORMAT = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
    "image/bmp": "BMP",
    "image/tiff": "TIFF",
}

_PIL_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "MPO": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


def compute_sha256(data: bytes) -> str:
    """Compute hex SHA-256 hash of byte payload."""
    return hashlib.sha256(data).hexdigest()


def detect_mime(filename: str, fallback: str = "application/octet-stream") -> str:
    """Detect MIME type from filename extension."""
    ext = Path(filename).suffix.lower()
    mapping = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".eml": "message/rfc822",
        ".msg": "application/vnd.ms-outlook",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".csv": "text/csv",
    }
    return mapping.get(ext) or mimetypes.guess_type(filename)[0] or fallback


def downscale_pil_image(img: Any, max_pixels: Optional[int]) -> Tuple[Any, bool]:
    """Downscale a PIL Image preserving aspect ratio when pixel count exceeds ``max_pixels``."""
    if not max_pixels or max_pixels <= 0:
        return img, False
    width, height = img.size
    total = width * height
    if total <= max_pixels:
        return img, False
    scale = math.sqrt(max_pixels / float(total))
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    resample = None
    try:
        from PIL import Image
        resample = Image.Resampling.LANCZOS
    except Exception:
        pass
    resized = img.resize((new_w, new_h), resample=resample) if resample else img.resize((new_w, new_h))
    return resized, True


def _resolve_save_format(orig_format: str, mime_type: str) -> Tuple[str, str]:
    """Map PIL format (including MPO/TIFF/BMP) to a safe VLM-compatible PIL format and MIME type."""
    fmt = (orig_format or _MIME_TO_PIL_FORMAT.get(mime_type.lower(), "PNG")).upper()
    if fmt in {"JPEG", "JPG", "MPO"}:
        return "JPEG", "image/jpeg"
    if fmt == "WEBP":
        return "WEBP", "image/webp"
    if fmt == "GIF":
        return "GIF", "image/gif"
    return "PNG", "image/png"


def _normalize_mode_for_save(img: Any, save_fmt: str) -> Any:
    """Ensure image mode is compatible with the target PIL encoder."""
    if save_fmt != "JPEG":
        if save_fmt == "PNG" and img.mode in ("CMYK", "YCbCr", "LAB", "HSV"):
            return img.convert("RGB")
        return img

    if img.mode in ("RGB", "L"):
        return img

    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in getattr(img, "info", {})):
        from PIL import Image
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background

    return img.convert("RGB")


def prepare_visual_image_bytes(
    data: bytes,
    mime_type: str = "image/png",
    max_image_pixels: Optional[int] = None,
    strip_exif: bool = False,
) -> Tuple[bytes, str, Optional[Tuple[int, int]], bool, bool]:
    """Normalize format/mode, downscale oversized images, and optionally strip EXIF metadata.

    Returns ``(processed_bytes, output_mime, dimensions, downscaled, exif_stripped)``.
    """
    out_bytes = data
    out_mime = mime_type or "image/png"
    dims: Optional[Tuple[int, int]] = None
    downscaled = False
    exif_stripped = False

    if not data:
        return out_bytes, out_mime, dims, downscaled, exif_stripped

    try:
        from PIL import Image
    except ImportError:
        return out_bytes, out_mime, dims, downscaled, exif_stripped

    try:
        with Image.open(io.BytesIO(data)) as img:
            orig_format = (img.format or "").upper()
            if orig_format in _PIL_FORMAT_TO_MIME:
                out_mime = _PIL_FORMAT_TO_MIME[orig_format]

            work_img, downscaled = downscale_pil_image(img, max_image_pixels)
            dims = work_img.size
            needs_reencode = downscaled or orig_format in {"MPO"}

            if needs_reencode:
                save_fmt, out_mime = _resolve_save_format(orig_format, out_mime)
                prepared = _normalize_mode_for_save(work_img, save_fmt)
                buf = io.BytesIO()
                prepared.save(buf, format=save_fmt)
                out_bytes = buf.getvalue()
                dims = prepared.size
                if strip_exif:
                    # Re-encoding via PIL without forwarding exif bytes strips EXIF/XMP chunks
                    exif_stripped = True
    except Exception:
        return out_bytes, out_mime, dims, False, False

    if strip_exif and not exif_stripped:
        try:
            from aurarouter.media import strip_image_metadata

            stripped_bytes, did_strip = strip_image_metadata(out_bytes, mime_type=out_mime)
            out_bytes = stripped_bytes
            exif_stripped = bool(did_strip)
        except Exception:
            exif_stripped = False

        if not exif_stripped:
            try:
                with Image.open(io.BytesIO(out_bytes)) as img:
                    save_fmt, out_mime = _resolve_save_format((img.format or "").upper(), out_mime)
                    prepared = _normalize_mode_for_save(img, save_fmt)
                    buf = io.BytesIO()
                    prepared.save(buf, format=save_fmt)
                    out_bytes = buf.getvalue()
                    dims = prepared.size
                    exif_stripped = True
            except Exception:
                pass

    return out_bytes, out_mime, dims, downscaled, exif_stripped


class BaseParser(ABC, DeterministicParserProtocol):
    """Abstract base class for deterministic document and image parsers."""

    @abstractmethod
    def can_parse(self, filename_or_mime: str) -> bool:
        """Check if parser handles this file type."""
        pass

    @abstractmethod
    def parse(
        self,
        data: bytes,
        filename: str,
        options: Optional[ScryerExtractionOptions] = None,
    ) -> Phase1Output:
        """Parse raw file bytes deterministically."""
        pass
