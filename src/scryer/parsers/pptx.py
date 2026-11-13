"""pptx.py - Deterministic PowerPoint presentation parser using python-pptx."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional
from scryer.contracts import (
    DocumentMetadata,
    ExtractedVisualElement,
    Phase1Output,
    ScryerExtractionOptions,
)
from scryer.parsers.base import BaseParser


class PPTXParser(BaseParser):
    """Deterministic PPTX parser extracting slide text, tables, notes, and slide pictures."""

    EXTENSIONS = frozenset({".pptx"})
    MIME_TYPES = frozenset({
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
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

        try:
            import pptx
            from pptx.enum.shapes import MSO_SHAPE_TYPE
        except ImportError:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    custom_fields={"error": "python-pptx is not installed"},
                ),
                raw_text="",
            )

        try:
            prs = pptx.Presentation(io.BytesIO(data))
        except Exception as e:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    custom_fields={"error": f"Failed to parse PPTX: {e}"},
                ),
                raw_text="",
            )

        slide_texts = []
        visual_elements = []
        img_counter = 0

        for slide_idx, slide in enumerate(prs.slides):
            parts = [f"## Slide {slide_idx + 1}"]
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            parts.append(text)
                elif shape.has_table:
                    table = shape.table
                    for row_idx, row in enumerate(table.rows):
                        cells = [c.text.strip() for c in row.cells]
                        parts.append("| " + " | ".join(cells) + " |")
                        if row_idx == 0:
                            parts.append("| " + " | ".join("---" for _ in cells) + " |")
                elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE and opts.extract_images:
                    try:
                        image = shape.image
                        img_bytes = image.blob
                        img_b64 = base64.b64encode(img_bytes).decode("ascii")
                        img_counter += 1
                        visual_elements.append(
                            ExtractedVisualElement(
                                element_id=f"{filename}_slide_{slide_idx+1}_img_{img_counter}",
                                element_type="image",
                                page_number=slide_idx + 1,
                                mime_type=image.content_type or "image/png",
                                data_base64=img_b64,
                                caption=f"Picture on Slide {slide_idx+1}",
                            )
                        )
                    except Exception:
                        pass

            # Slide notes
            try:
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    note = slide.notes_slide.notes_text_frame.text.strip()
                    if note:
                        parts.append(f"> Notes: {note}")
            except Exception:
                pass

            slide_texts.append("\n".join(parts))

        full_text = "\n\n".join(slide_texts)

        # Core properties
        author = None
        title = None
        created_ts = None
        modified_ts = None
        try:
            props = prs.core_properties
            author = props.author
            title = props.title
            created_ts = props.created
            modified_ts = props.modified
        except Exception:
            pass

        metadata = DocumentMetadata(
            file_size=len(data),
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            page_count=len(prs.slides),
            author=author,
            title=title,
            created_ts=created_ts,
            modified_ts=modified_ts,
            custom_fields={
                "slide_count": len(prs.slides),
                "extracted_images_count": len(visual_elements),
            },
        )

        return Phase1Output(
            metadata=metadata,
            raw_text=full_text,
            visual_elements=visual_elements,
            page_texts=slide_texts,
        )

