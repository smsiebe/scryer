"""pdf.py - Deterministic PDF parser with page rendering for low-text / scanned pages."""

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
from scryer.parsers.base import BaseParser, downscale_pil_image


class PDFParser(BaseParser):
    """Deterministic PDF parser using pypdfium2."""

    EXTENSIONS = frozenset({".pdf"})
    MIME_TYPES = frozenset({"application/pdf"})

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
        threshold = opts.min_text_threshold

        try:
            import pypdfium2 as pdfium
        except ImportError:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type="application/pdf",
                    page_count=0,
                    custom_fields={"error": "pypdfium2 is not installed"},
                ),
                raw_text="",
            )

        try:
            pdf = pdfium.PdfDocument(data)
        except Exception as e:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type="application/pdf",
                    page_count=0,
                    custom_fields={"error": f"Failed to parse PDF: {e}"},
                ),
                raw_text="",
            )

        try:
            page_count = len(pdf)
            page_texts = []
            low_text_pages = []
            visual_elements = []

            for i in range(page_count):
                try:
                    page = pdf[i]
                    textpage = page.get_textpage()
                    page_text = textpage.get_text_range() or ""
                except Exception:
                    page_text = ""

                page_texts.append(page_text)

                # Check text density: if low or image-only, render page to image for Phase 2 VLM
                if len(page_text.strip()) < threshold and opts.extract_images:
                    low_text_pages.append(i + 1)
                    if len(visual_elements) < opts.max_render_pages:
                        try:
                            pil_img = page.render(scale=opts.render_scale).to_pil()
                            pil_img, _ = downscale_pil_image(pil_img, opts.max_image_pixels)
                            buf = io.BytesIO()
                            pil_img.save(buf, format="PNG")
                            b64_data = base64.b64encode(buf.getvalue()).decode("ascii")
                            visual_elements.append(
                                ExtractedVisualElement(
                                    element_id=f"{filename}_page_{i+1}",
                                    element_type="rendered_page",
                                    page_number=i + 1,
                                    mime_type="image/png",
                                    width=pil_img.width,
                                    height=pil_img.height,
                                    data_base64=b64_data,
                                    caption=f"Rendered page {i+1} (low text density: {len(page_text.strip())} chars)",
                                )
                            )
                        except Exception:
                            pass

            # Extract PDF metadata if available
            meta_dict = {}
            try:
                meta_dict = pdf.get_metadata_dict() or {}
            except Exception:
                pass
        finally:
            try:
                pdf.close()
            except Exception:
                pass

        full_text = "\n\n".join(
            f"--- Page {idx+1} ---\n{t}" if t.strip() else f"--- Page {idx+1} (Visual/Scanned) ---"
            for idx, t in enumerate(page_texts)
        )

        metadata = DocumentMetadata(
            file_size=len(data),
            mime_type="application/pdf",
            page_count=page_count,
            author=meta_dict.get("Author"),
            title=meta_dict.get("Title"),
            subject=meta_dict.get("Subject"),
            custom_fields={
                "creator": meta_dict.get("Creator"),
                "producer": meta_dict.get("Producer"),
                "creation_date": meta_dict.get("CreationDate"),
                "mod_date": meta_dict.get("ModDate"),
                "low_text_pages_count": len(low_text_pages),
                "rendered_pages_truncated": len(low_text_pages) > opts.max_render_pages,
            },
        )

        return Phase1Output(
            metadata=metadata,
            raw_text=full_text,
            visual_elements=visual_elements,
            page_texts=page_texts,
            low_text_pages=low_text_pages,
        )

