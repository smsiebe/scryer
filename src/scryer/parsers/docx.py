"""docx.py - Deterministic Word document parser using python-docx."""

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
from scryer.parsers.base import BaseParser, prepare_visual_image_bytes


class DOCXParser(BaseParser):
    """Deterministic DOCX parser extracting paragraphs, tables, core properties, and inline images."""

    EXTENSIONS = frozenset({".docx"})
    MIME_TYPES = frozenset({
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
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
            import docx
        except ImportError:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    custom_fields={"error": "python-docx is not installed"},
                ),
                raw_text="",
            )

        try:
            doc = docx.Document(io.BytesIO(data))
        except Exception as e:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    custom_fields={"error": f"Failed to parse DOCX: {e}"},
                ),
                raw_text="",
            )

        text_parts = []
        for p in doc.paragraphs:
            val = p.text.strip()
            if val:
                text_parts.append(val)

        # Extract tables as markdown
        for table_idx, table in enumerate(doc.tables):
            table_lines = [f"\n### Table {table_idx + 1}"]
            for row_idx, row in enumerate(table.rows):
                row_vals = [c.text.strip() for c in row.cells]
                table_lines.append("| " + " | ".join(row_vals) + " |")
                if row_idx == 0:
                    table_lines.append("| " + " | ".join("---" for _ in row_vals) + " |")
            if len(table_lines) > 1:
                text_parts.append("\n".join(table_lines))

        # Extract embedded images
        visual_elements = []
        if opts.extract_images:
            img_idx = 0
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        img_part = rel.target_part
                        raw_mime = img_part.content_type or "image/png"
                        clean_bytes, img_mime, dims, _, _ = prepare_visual_image_bytes(
                            img_part.blob,
                            mime_type=raw_mime,
                            max_image_pixels=opts.max_image_pixels,
                            strip_exif=opts.strip_exif,
                        )
                        img_b64 = base64.b64encode(clean_bytes).decode("ascii")
                        img_idx += 1
                        visual_elements.append(
                            ExtractedVisualElement(
                                element_id=f"{filename}_img_{img_idx}",
                                element_type="image",
                                mime_type=img_mime,
                                width=dims[0] if dims else None,
                                height=dims[1] if dims else None,
                                data_base64=img_b64,
                                caption=f"Embedded image {img_idx} from {filename}",
                            )
                        )
                    except Exception:
                        pass

        # Extract core properties
        author = None
        title = None
        created_ts = None
        modified_ts = None
        try:
            props = doc.core_properties
            author = props.author
            title = props.title
            created_ts = props.created
            modified_ts = props.modified
        except Exception:
            pass

        full_text = "\n\n".join(text_parts)

        metadata = DocumentMetadata(
            file_size=len(data),
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            page_count=1,
            author=author,
            title=title,
            created_ts=created_ts,
            modified_ts=modified_ts,
            custom_fields={
                "paragraph_count": len(doc.paragraphs),
                "table_count": len(doc.tables),
                "embedded_image_count": len(visual_elements),
            },
        )

        return Phase1Output(
            metadata=metadata,
            raw_text=full_text,
            visual_elements=visual_elements,
            page_texts=[full_text],
        )

