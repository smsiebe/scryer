"""image.py - Deterministic image and EXIF metadata parser using Pillow."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional
from scryer.contracts import (
    DocumentMetadata,
    ExtractedVisualElement,
    Phase1Output,
    ScryerExtractionOptions,
)
from scryer.parsers.base import (
    BaseParser,
    detect_mime,
    prepare_visual_image_bytes,
)


class ImageParser(BaseParser):
    """Deterministic image parser extracting dimensions, format, EXIF tags, and base64 payloads."""

    EXTENSIONS = frozenset({
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
    })
    MIME_TYPES = frozenset({
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/bmp",
        "image/tiff",
    })

    def can_parse(self, filename_or_mime: str) -> bool:
        ext = Path(filename_or_mime).suffix.lower()
        if ext in self.EXTENSIONS:
            return True
        return filename_or_mime.lower() in self.MIME_TYPES

    def parse(
        self,
        data: bytes,
        filename: str,
        options: Optional[ScryerExtractionOptions] = None,
    ) -> Phase1Output:
        opts = options or ScryerExtractionOptions()
        mime_type = detect_mime(filename, fallback="image/png")

        try:
            from PIL import Image, ExifTags
        except ImportError:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type=mime_type,
                    custom_fields={"error": "Pillow is not installed"},
                ),
                raw_text="",
            )

        try:
            with Image.open(io.BytesIO(data)) as img:
                width, height = img.size
                img_format = (img.format or "PNG").upper()
                img_mode = img.mode

                # Extract EXIF tags
                exif_dict: Dict[str, Any] = {}
                try:
                    raw_exif = img.getexif()
                    if raw_exif:
                        for tag_id, val in raw_exif.items():
                            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                            # Keep only JSON-safe basic types
                            if isinstance(val, (int, float, str)):
                                exif_dict[str(tag_name)] = val
                            elif isinstance(val, bytes):
                                try:
                                    exif_dict[str(tag_name)] = val.decode("utf-8", errors="ignore").strip("\x00")
                                except Exception:
                                    pass
                except Exception:
                    pass

                # Downscale oversized images, normalize unsafe formats/modes, and strip EXIF if requested
                orig_dims = (width, height)
                clean_bytes, resolved_mime, final_dims, was_resized, exif_stripped = (
                    prepare_visual_image_bytes(
                        data,
                        mime_type=mime_type,
                        max_image_pixels=opts.max_image_pixels,
                        strip_exif=opts.strip_exif,
                    )
                )
                if final_dims is not None:
                    width, height = final_dims
                mime_type = resolved_mime

                b64_data = base64.b64encode(clean_bytes).decode("ascii")

                visual = ExtractedVisualElement(
                    element_id=f"{filename}_primary",
                    element_type="image",
                    mime_type=mime_type,
                    width=width,
                    height=height,
                    data_base64=b64_data,
                    caption=f"{img_format} image ({width}x{height}, mode {img_mode})",
                )

                metadata = DocumentMetadata(
                    file_size=len(data),
                    mime_type=mime_type,
                    page_count=1,
                    dimensions=(width, height),
                    exif_tags=exif_dict,
                    custom_fields={
                        "format": img_format,
                        "mode": img_mode,
                        "exif_tag_count": len(exif_dict),
                        "exif_stripped": exif_stripped,
                        "downscaled": was_resized,
                        "original_dimensions": orig_dims if was_resized else None,
                    },
                )

                return Phase1Output(
                    metadata=metadata,
                    raw_text="",  # Text for images will be produced by Phase 2 OCR / VLM
                    visual_elements=[visual],
                )
        except Exception as e:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type=mime_type,
                    custom_fields={"error": f"Failed to parse image: {e}"},
                ),
                raw_text="",
            )

