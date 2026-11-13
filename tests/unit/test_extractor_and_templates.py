"""unit tests for Extraction Template Engine, ScryerExtractor, and Lossless Result Assembler."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image
from pydantic import BaseModel, Field
import pytest

from scryer.bridge import BridgeResponse, TriModeBridge
from scryer.cli import main as cli_main
from scryer.contracts import ScryerExtractionOptions
from scryer.extractor import ScryerExtractor
from scryer.templates import (
    INVOICE_TEMPLATE,
    InvoiceModel,
    TemplateCompiler,
    VisualSceneModel,
)


class CustomVehicleTemplate(BaseModel):
    """Extraction model for vehicles identified in field images."""
    make: str = Field(description="Vehicle manufacturer")
    model: str = Field(description="Vehicle model name")
    color: str = Field(description="Exterior color")
    license_plate: str = Field(description="Readable license plate number")


class TestExtractorAndTemplates:
    def test_pydantic_template_compilation(self):
        compiler = TemplateCompiler()
        schema, guidelines = compiler.compile(CustomVehicleTemplate)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "license_plate" in schema["properties"]
        assert any("Vehicle manufacturer" in g for g in guidelines)
        assert any("license_plate" in g for g in guidelines)

    def test_declarative_dict_template_compilation(self):
        compiler = TemplateCompiler()
        raw_tmpl = {
            "name": "BadgeInspection",
            "description": "Inspect ID badges on site",
            "fields": {
                "badge_number": {"type": "string", "description": "Employee badge ID", "required": True},
                "security_clearance": {"type": "string", "enum": ["Confidential", "Secret", "TopSecret"]},
            },
            "guidelines": ["Check for tamper-evident hologram"],
        }
        schema, guidelines = compiler.compile(raw_tmpl)
        assert schema["type"] == "object"
        assert "badge_number" in schema["properties"]
        assert "security_clearance" in schema["properties"]
        assert "required" in schema
        assert "badge_number" in schema["required"]
        assert any("Inspect ID badges" in g for g in guidelines)

    def test_lossless_extraction_assembly(self, tmp_path: Path):
        # 1. Prepare sample text document
        sample_file = tmp_path / "invoice.txt"
        sample_file.write_text(
            "ACME Corp Invoice\nInvoice Number: INV-9901\nDate: 2026-11-12\nTotal Due: $540.00\nThank you for your business.",
            encoding="utf-8",
        )

        # 2. Mock router bridge returning structured data + supplementary annotation
        mock_bridge = MagicMock(spec=TriModeBridge)
        mock_bridge.execute_with_heuristics.return_value = BridgeResponse(
            raw_output="Raw model output notes",
            structured_output={
                "vendor_name": "ACME Corp",
                "invoice_number": "INV-9901",
                "total_amount": 540.0,
            },
            intent="VISION_EXTRACTION",
            complexity=3,
            annotations=["Detected high-confidence vendor header"],
        )

        extractor = ScryerExtractor(bridge=mock_bridge)
        options = ScryerExtractionOptions(
            template=INVOICE_TEMPLATE,
            enable_phase2=True,
        )

        result = extractor.extract(sample_file, options=options)

        # 3. Verify that NOTHING was lost:
        # Phase 1 text preserved
        assert "ACME Corp Invoice" in result.raw_text
        assert "Total Due: $540.00" in result.raw_text
        # Metadata populated
        assert result.metadata.file_size > 0
        assert result.sha256 != ""
        # Structured data populated from Phase 2
        assert result.structured_data["vendor_name"] == "ACME Corp"
        assert result.structured_data["invoice_number"] == "INV-9901"
        assert result.structured_data["total_amount"] == 540.0
        # Annotations captured
        assert "Detected high-confidence vendor header" in result.unstructured_annotations
        # Telemetry captured
        assert result.route_telemetry["intent"] == "VISION_EXTRACTION"

    def test_cli_execution(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        sample_file = tmp_path / "notes.txt"
        sample_file.write_text("Meeting Notes: Phase 0 and 1 are complete.", encoding="utf-8")

        # Test CLI extract with --no-phase2 and --json
        code = cli_main(["extract", str(sample_file), "--no-phase2", "--json"])
        assert code == 0

        captured = capsys.readouterr()
        assert "notes.txt" in captured.out
        assert "Meeting Notes" in captured.out
