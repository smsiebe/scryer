"""base.py - Base parser class and common utilities for Phase 1 deterministic extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import hashlib
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
