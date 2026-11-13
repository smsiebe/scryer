"""adversarial test suite for Scryer extraction engine (corrupted files, malformed schemas, router outages)."""

from __future__ import annotations

from pathlib import Path
import pytest
from scryer.bridge import BaseRouterBridge, BridgeResponse
from scryer.contracts import ScryerExtractionOptions
from scryer.extractor import ScryerExtractor
from scryer.templates import TemplateCompiler


class FailingRouterBridge(BaseRouterBridge):
    """Real failure simulator without in-memory mock transports."""

    def is_available(self) -> bool:
        return True

    def execute_multimodal(self, *args, **kwargs) -> BridgeResponse:
        return BridgeResponse(
            raw_output="",
            success=False,
            error_message="Connection refused to LLM endpoint: 503 Service Unavailable",
            annotations=["AuraRouter was not reachable. Extracted data relies solely on Phase 1."],
        )

    def execute_with_heuristics(self, *args, **kwargs) -> BridgeResponse:
        return self.execute_multimodal(*args, **kwargs)


@pytest.mark.adversarial
class TestAdversarialExtraction:
    def test_nonexistent_file_raises_filenotfound(self):
        extractor = ScryerExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract("C:/nonexistent/path/to/missing_file.pdf")

    def test_zero_byte_file_resilience(self, tmp_path: Path):
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")

        extractor = ScryerExtractor()
        result = extractor.extract(empty_file)

        assert result.metadata.file_size == 0
        assert result.raw_text == ""
        assert result.doc_id == "empty.pdf"

    def test_corrupted_pdf_stream(self, tmp_path: Path):
        corrupt_pdf = tmp_path / "corrupt.pdf"
        corrupt_pdf.write_bytes(b"%PDF-1.7\r\n\xff\xfe\xfd\x00CorruptedTrailerNoEOF")

        extractor = ScryerExtractor()
        result = extractor.extract(corrupt_pdf)

        assert result.metadata.file_size > 0
        assert "error" in result.metadata.custom_fields
        # Raw text should be empty without crashing
        assert result.raw_text == ""

    def test_corrupted_docx_stream(self, tmp_path: Path):
        corrupt_docx = tmp_path / "corrupt.docx"
        corrupt_docx.write_bytes(b"PK\x03\x04\x14\x00\x00\x00CorruptedZipStreamTruncated")

        extractor = ScryerExtractor()
        result = extractor.extract(corrupt_docx)

        assert result.metadata.file_size > 0
        assert "error" in result.metadata.custom_fields
        assert result.raw_text == ""

    def test_corrupted_image_stream(self, tmp_path: Path):
        corrupt_img = tmp_path / "corrupt.png"
        corrupt_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorruptedtruncated")

        extractor = ScryerExtractor()
        result = extractor.extract(corrupt_img)

        assert result.metadata.file_size > 0
        assert "error" in result.metadata.custom_fields

    def test_malformed_yaml_template(self, tmp_path: Path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("invalid: yaml: [unbalanced: bracket\n  missing", encoding="utf-8")

        compiler = TemplateCompiler()
        with pytest.raises(Exception):
            compiler.compile(bad_yaml)

    def test_router_total_outage_does_not_lose_phase1_text(self, tmp_path: Path):
        valid_doc = tmp_path / "doc.txt"
        valid_doc.write_text("Critical report data that must never be lost under any circumstances.", encoding="utf-8")

        # Real router bridge implementation simulating service outage
        outage_bridge = FailingRouterBridge()
        extractor = ScryerExtractor(bridge=outage_bridge)
        options = ScryerExtractionOptions(enable_phase2=True, custom_prompt="Extract all entities")

        result = extractor.extract(valid_doc, options=options)

        # Invariant: Phase 1 text MUST NOT be discarded when router fails
        assert "Critical report data" in result.raw_text
        assert result.route_telemetry["success"] is False
        assert "Connection refused" in result.route_telemetry["error_message"]
        assert len(result.unstructured_annotations) > 0
