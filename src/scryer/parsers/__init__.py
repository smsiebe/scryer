"""scryer.parsers - Phase 1 deterministic document and image parsers."""

from scryer.parsers.base import BaseParser, compute_sha256, detect_mime
from scryer.parsers.docx import DOCXParser
from scryer.parsers.email import EmailParser
from scryer.parsers.image import ImageParser
from scryer.parsers.pdf import PDFParser
from scryer.parsers.pptx import PPTXParser
from scryer.parsers.registry import DeterministicParserRegistry, TextParser
from scryer.parsers.xlsx import XLSXParser

__all__ = [
    "BaseParser",
    "compute_sha256",
    "detect_mime",
    "PDFParser",
    "DOCXParser",
    "XLSXParser",
    "PPTXParser",
    "EmailParser",
    "ImageParser",
    "TextParser",
    "DeterministicParserRegistry",
]

