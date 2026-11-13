"""contracts.py - Canonical Pydantic DTOs and Schemas for Scryer.

Defines the lossless extraction result wrapper, extraction templates,
extraction options, document metadata, and serialization utilities.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
from pydantic import BaseModel, ConfigDict, Field


def json_default_serializer(obj: Any) -> Any:
    """Fallback serializer ensuring datetimes, paths, enums, and models serialize cleanly."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """Robust JSON serializer that guarantees standard JSON serialization without TypeError."""
    if "default" not in kwargs:
        kwargs["default"] = json_default_serializer
    return json.dumps(obj, **kwargs)


def safe_json_loads(s: Union[str, bytes], **kwargs: Any) -> Any:
    """Parse JSON strings or bytes safely."""
    return json.loads(s, **kwargs)


def compute_canonical_hash(payload: Any) -> str:
    """Compute a deterministic SHA-256 hex digest for JSON-compatible payloads."""
    serialized = safe_json_dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class TriMode(str, Enum):
    PYTHON = "python"
    IPC = "ipc"
    MAS = "mas"


class IntentMode(str, Enum):
    AUTO = "AUTO"
    VISION_OCR = "VISION_OCR"
    VISION_EXTRACTION = "VISION_EXTRACTION"


class DocumentMetadata(BaseModel):
    """Normalized metadata extracted from binary documents and images in Phase 1."""
    model_config = ConfigDict(frozen=True)

    file_size: int = 0
    mime_type: str = "application/octet-stream"
    page_count: int = 1
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    created_ts: Optional[datetime] = None
    modified_ts: Optional[datetime] = None
    dimensions: Optional[Tuple[int, int]] = None
    exif_tags: Dict[str, Any] = Field(default_factory=dict)
    sheet_names: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class ExtractedVisualElement(BaseModel):
    """Visual asset (embedded image, rendered page, diagram, table graphic) extracted from source."""
    model_config = ConfigDict(frozen=True)

    element_id: str
    element_type: str = "image"  # "image", "rendered_page", "table_crop", "diagram"
    page_number: Optional[int] = None
    mime_type: str = "image/png"
    width: Optional[int] = None
    height: Optional[int] = None
    data_base64: Optional[str] = None
    source_path: Optional[str] = None
    caption: Optional[str] = None
    ocr_text: Optional[str] = None


class ExtractionTemplate(BaseModel):
    """Specification of target fields, contextual prompt guidelines, and JSON Schema format."""
    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    schema_dict: Dict[str, Any] = Field(default_factory=dict)
    prompt_guidelines: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)

    def to_response_schema(self) -> Dict[str, Any]:
        """Return the JSON schema dict to pass to AuraRouter or VLMs."""
        if self.schema_dict:
            return self.schema_dict
        # Default fallback generic extraction schema
        return {
            "type": "object",
            "properties": {
                "extracted_fields": {"type": "object"},
                "summary": {"type": "string"},
            },
        }


class ScryerExtractionOptions(BaseModel):
    """Controls for the extraction pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    template: Optional[Any] = None  # ExtractionTemplate, BaseModel subclass, or JSON schema dict
    enable_phase2: bool = True
    force_phase2: bool = False  # Invoke Phase 2 even for plain-text untemplated docs
    tri_mode: TriMode = TriMode.PYTHON
    route_override: Optional[str] = None
    execution_mode: Optional[str] = None  # "standard", "speculative", "monologue"
    prefer_local: Optional[bool] = None
    enable_privacy: bool = True
    enable_savings: bool = True
    extract_images: bool = True
    min_text_threshold: int = 30  # Render PDF page to image if extracted characters < threshold
    render_scale: float = 2.0
    max_render_pages: int = 50
    max_image_pixels: Optional[int] = 4_000_000
    max_images_per_call: int = 4
    max_context_chars: int = 12_000
    max_xlsx_rows_per_sheet: int = 5_000
    strip_exif: bool = False
    custom_prompt: Optional[str] = None
    max_tokens: int = 2048
    timeout_seconds: float = 60.0
    max_retries: int = 1
    cache_dir: Optional[str] = None
    aurarouter_config_path: Optional[str] = None
    ipc_pipe_name: Optional[str] = None
    mcp_command: Optional[List[str]] = None
    mcp_url: Optional[str] = None


class Phase1Output(BaseModel):
    """Intermediate deterministic extraction output from Phase 1 parsers."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    raw_text: str = ""
    visual_elements: List[ExtractedVisualElement] = Field(default_factory=list)
    page_texts: List[str] = Field(default_factory=list)
    low_text_pages: List[int] = Field(default_factory=list)


class ScryerResult(BaseModel):
    """Universal lossless extraction wrapper.

    Ensures that raw text, structured schema-validated data, visual elements,
    and supplementary model annotations are preserved without loss.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_id: str
    source_path: Optional[str] = None
    mime_type: str = "application/octet-stream"
    sha256: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    raw_text: str = ""
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    visual_elements: List[ExtractedVisualElement] = Field(default_factory=list)
    unstructured_annotations: List[str] = Field(default_factory=list)
    route_telemetry: Dict[str, Any] = Field(default_factory=dict)
    extracted_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dictionary representation."""
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "mime_type": self.mime_type,
            "sha256": self.sha256,
            "metadata": self.metadata.model_dump(mode="json"),
            "raw_text": self.raw_text,
            "structured_data": self.structured_data,
            "visual_elements": [v.model_dump(mode="json") for v in self.visual_elements],
            "unstructured_annotations": self.unstructured_annotations,
            "route_telemetry": self.route_telemetry,
            "extracted_at_utc": self.extracted_at_utc.isoformat(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """JSON string representation."""
        return safe_json_dumps(self.to_dict(), indent=indent)

