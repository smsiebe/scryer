"""Unit tests for Phase 0 Scryer contracts, protocols, and serialization."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import pytest

from scryer.contracts import (
    DocumentMetadata,
    ExtractedVisualElement,
    ExtractionTemplate,
    IntentMode,
    Phase1Output,
    ScryerExtractionOptions,
    ScryerResult,
    TriMode,
    safe_json_dumps,
    safe_json_loads,
)
from scryer.protocols import (
    DeterministicParserProtocol,
    RouterBridgeProtocol,
    ScryerExtractorProtocol,
    TemplateEngineProtocol,
)


class TestScryerContracts:
    def test_scryer_result_roundtrip_lossless(self):
        now = datetime.now(timezone.utc)
        meta = DocumentMetadata(
            file_size=1024,
            mime_type="application/pdf",
            page_count=3,
            author="Test Author",
            title="Sample Doc",
            created_ts=now,
            dimensions=(800, 600),
            sheet_names=["Summary", "Data"],
            custom_fields={"department": "Engineering", "confidential": True},
        )
        visual = ExtractedVisualElement(
            element_id="elem_1",
            element_type="rendered_page",
            page_number=1,
            mime_type="image/png",
            width=800,
            height=600,
            data_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            caption="Page 1 scan",
        )
        result = ScryerResult(
            doc_id="doc_abc_123",
            source_path="/path/to/sample.pdf",
            mime_type="application/pdf",
            sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            metadata=meta,
            raw_text="Deterministic extracted text from page 1 and page 2.",
            structured_data={"invoice_id": "INV-2026-001", "total": 450.50},
            visual_elements=[visual],
            unstructured_annotations=["Detected blue logo on header", "Handwritten signature present"],
            route_telemetry={"model": "qwen2-vl-7b", "latency_ms": 350, "mode": "python"},
            extracted_at_utc=now,
        )

        # 1. JSON dump via method
        json_str = result.to_json()
        assert "INV-2026-001" in json_str
        assert "doc_abc_123" in json_str

        # 2. Deserialization via safe_json_loads
        raw_dict = safe_json_loads(json_str)
        assert raw_dict["doc_id"] == "doc_abc_123"
        assert raw_dict["structured_data"]["invoice_id"] == "INV-2026-001"
        assert raw_dict["raw_text"] == "Deterministic extracted text from page 1 and page 2."
        assert len(raw_dict["visual_elements"]) == 1
        assert raw_dict["visual_elements"][0]["element_id"] == "elem_1"
        assert len(raw_dict["unstructured_annotations"]) == 2

        # 3. Reconstruct ScryerResult from dict
        reconstructed = ScryerResult.model_validate(raw_dict)
        assert reconstructed.doc_id == result.doc_id
        assert reconstructed.sha256 == result.sha256
        assert reconstructed.metadata.author == "Test Author"
        assert reconstructed.structured_data == result.structured_data
        assert reconstructed.raw_text == result.raw_text
        assert len(reconstructed.visual_elements) == 1
        assert reconstructed.unstructured_annotations == result.unstructured_annotations

    def test_extraction_template_response_schema(self):
        custom_schema = {
            "type": "object",
            "properties": {
                "organization": {"type": "string"},
                "location": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["organization"],
        }
        tmpl = ExtractionTemplate(
            name="EntityTemplate",
            description="Extract organization and location",
            schema_dict=custom_schema,
            prompt_guidelines=["Look for corporate names", "Extract city/state"],
            categories=["entities", "geography"],
        )
        schema = tmpl.to_response_schema()
        assert schema == custom_schema
        assert "organization" in schema["properties"]

    def test_enums_and_options(self):
        assert TriMode.PYTHON.value == "python"
        assert TriMode.IPC.value == "ipc"
        assert TriMode.MAS.value == "mas"

        options = ScryerExtractionOptions(
            enable_phase2=True,
            tri_mode=TriMode.IPC,
            min_text_threshold=50,
            strip_exif=True,
        )
        assert options.tri_mode == TriMode.IPC
        assert options.min_text_threshold == 50
        assert options.strip_exif is True

    def test_protocol_runtime_checkable(self):
        class DummyParser:
            def can_parse(self, filename_or_mime: str) -> bool:
                return True

            def parse(self, data: bytes, filename: str, options=None) -> Phase1Output:
                return Phase1Output()

        parser = DummyParser()
        assert isinstance(parser, DeterministicParserProtocol)

