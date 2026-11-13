"""registry.py - Unified registry of deterministic Phase 1 document and image parsers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from scryer.contracts import (
    DocumentMetadata,
    Phase1Output,
    ScryerExtractionOptions,
)
from scryer.parsers.base import BaseParser, detect_mime
from scryer.parsers.docx import DOCXParser
from scryer.parsers.email import EmailParser
from scryer.parsers.image import ImageParser
from scryer.parsers.pdf import PDFParser
from scryer.parsers.pptx import PPTXParser
from scryer.parsers.xlsx import XLSXParser


class TextParser(BaseParser):
    """Deterministic parser for plain text, Markdown, JSON, and CSV files."""

    EXTENSIONS = frozenset({".txt", ".md", ".json", ".csv", ".log", ".yaml", ".yml"})

    def can_parse(self, filename_or_mime: str) -> bool:
        ext = Path(filename_or_mime).suffix.lower()
        if ext in self.EXTENSIONS:
            return True
        return filename_or_mime.startswith("text/")

    def parse(
        self,
        data: bytes,
        filename: str,
        options: Optional[ScryerExtractionOptions] = None,
    ) -> Phase1Output:
        mime_type = detect_mime(filename, fallback="text/plain")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("latin-1")
            except Exception:
                text = str(data)

        metadata = DocumentMetadata(
            file_size=len(data),
            mime_type=mime_type,
            page_count=1,
            custom_fields={"line_count": len(text.splitlines())},
        )
        return Phase1Output(
            metadata=metadata,
            raw_text=text,
            page_texts=[text],
        )


class DeterministicParserRegistry:
    """Dispatches to registered parsers by filename extension and MIME type."""

    def __init__(self, parsers: Optional[List[BaseParser]] = None):
        self.parsers: List[BaseParser] = parsers or [
            PDFParser(),
            DOCXParser(),
            XLSXParser(),
            PPTXParser(),
            EmailParser(),
            ImageParser(),
            TextParser(),
        ]

    def get_parser(self, filename_or_mime: str) -> Optional[BaseParser]:
        """Find the first matching parser for the given filename or MIME type."""
        for p in self.parsers:
            if p.can_parse(filename_or_mime):
                return p
        return None

    def can_parse(self, filename_or_mime: str) -> bool:
        """Check if any registered parser can handle this input."""
        return self.get_parser(filename_or_mime) is not None

    def parse(
        self,
        data: bytes,
        filename: str,
        options: Optional[ScryerExtractionOptions] = None,
    ) -> Phase1Output:
        """Parse raw file bytes using the matching registered parser."""
        parser = self.get_parser(filename)
        if not parser:
            mime = detect_mime(filename)
            parser = self.get_parser(mime)

        if not parser:
            # Fallback: treat as plain text if possible
            return TextParser().parse(data, filename, options)

        return parser.parse(data, filename, options)
