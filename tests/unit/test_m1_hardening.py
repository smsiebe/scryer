"""Unit tests covering Milestone 1 core hardening features."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
from pydantic import BaseModel, Field

from scryer.bridge import BridgeResponse, TriModeBridge
from scryer.cache import DiskExtractionCache
from scryer.cli import main as cli_main
from scryer.contracts import ScryerExtractionOptions
from scryer.extractor import AsyncScryerExtractor, ScryerExtractor
from scryer.parsers.image import ImageParser
from scryer.parsers.xlsx import XLSXParser
from scryer.templates import (
    INVOICE_TEMPLATE,
    InvoiceModel,
    TemplateCompiler,
    inline_json_schema_refs,
    validate_structured_output,
)


class _NestedItem(BaseModel):
    sku: str = Field(description="SKU code")
    qty: int = Field(description="Quantity")


class _NestedOrder(BaseModel):
    order_id: str = Field(description="Order identifier")
    items: list[_NestedItem] = Field(default_factory=list)


class TestMilestone1Hardening:
    def test_schema_defs_and_refs_are_inlined(self):
        compiler = TemplateCompiler()
        schema, _, model_cls = compiler.resolve_template(_NestedOrder)
        assert model_cls is _NestedOrder
        dumped = json.dumps(schema)
        assert "$defs" not in dumped
        assert "$ref" not in dumped
        assert schema["properties"]["items"]["items"]["properties"]["sku"]["type"] == "string"

        # INVOICE_TEMPLATE preset is also free of $defs/$ref
        inv_schema, _ = compiler.compile(INVOICE_TEMPLATE)
        assert "$defs" not in json.dumps(inv_schema)
        assert "$ref" not in json.dumps(inv_schema)

    def test_validate_structured_output_preserves_extra_and_partial_data(self):
        data = {
            "order_id": "ORD-42",
            "items": [{"sku": "A1", "qty": 3}],
            "surprise_note": "Fragile",
        }
        normalized, notes = validate_structured_output(data, model_cls=_NestedOrder)
        assert normalized["order_id"] == "ORD-42"
        assert any("surprise_note" in n for n in notes)

        # Partial conformance still preserves data & adds notice instead of raising
        bad_data = {"items": "not-a-list"}
        partial_norm, bad_notes = validate_structured_output(bad_data, model_cls=_NestedOrder)
        assert partial_norm == bad_data
        assert len(bad_notes) >= 1

    def test_image_parser_delegates_exif_strip_and_downscales(self):
        # Create 200x100 PNG with pixel cap 2000 (should scale down maintaining aspect ratio)
        img = Image.new("RGB", (200, 100), color=(10, 120, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()

        parser = ImageParser()
        out = parser.parse(
            raw,
            "big.png",
            options=ScryerExtractionOptions(strip_exif=True, max_image_pixels=2000),
        )
        assert out.metadata.dimensions is not None
        assert out.metadata.dimensions[0] * out.metadata.dimensions[1] <= 2100
        assert out.metadata.custom_fields["downscaled"] is True

    def test_xlsx_row_cap(self):
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Data"
        for i in range(15):
            ws.append([f"row_{i}", i])
        buf = io.BytesIO()
        wb.save(buf)
        wb.close()

        parser = XLSXParser()
        out = parser.parse(
            buf.getvalue(),
            "many_rows.xlsx",
            options=ScryerExtractionOptions(max_xlsx_rows_per_sheet=5),
        )
        assert out.metadata.custom_fields["total_rows_extracted"] == 5
        assert "Data" in out.metadata.custom_fields["truncated_sheets"]

    def test_long_text_chunking_batching_and_cache_invalidation(self, tmp_path: Path):
        doc_path = tmp_path / "long_doc.txt"
        long_text = ("Segment marker A. " * 40) + ("Segment marker B. " * 40)
        doc_path.write_text(long_text, encoding="utf-8")

        chunk_calls = 0

        def _fake_batch(**kwargs):
            nonlocal chunk_calls
            chunk_calls += 1
            return BridgeResponse(
                raw_output=f"Chunk {chunk_calls}",
                structured_output={
                    "summary": f"Part {chunk_calls}",
                    "topics": [f"T{chunk_calls}"],
                    "organizations": ["OrgB"],
                    "action_items": [],
                },
                intent="GENERAL",
                model_id="test-vlm",
            )

        mock_bridge = MagicMock(spec=TriModeBridge)
        mock_bridge.execute_with_heuristics.side_effect = _fake_batch

        cache = DiskExtractionCache(tmp_path / "cache")
        extractor = ScryerExtractor(bridge=mock_bridge, cache=cache)
        opts = ScryerExtractionOptions(
            force_phase2=True,
            max_context_chars=600,
        )
        res1 = extractor.extract(doc_path, options=opts)
        assert mock_bridge.execute_with_heuristics.call_count >= 2
        assert "T1" in res1.structured_data["topics"]
        assert "T2" in res1.structured_data["topics"]
        assert res1.route_telemetry.get("batch_count", 1) >= 2

        # Second identical extraction hits cache without calling bridge again
        call_count_before = mock_bridge.execute_with_heuristics.call_count
        res2 = extractor.extract(doc_path, options=opts)
        assert mock_bridge.execute_with_heuristics.call_count == call_count_before
        assert res2.route_telemetry.get("cache_hit") is True

        # Changing template busts cache cleanly
        mock_bridge.execute_with_heuristics.side_effect = lambda **kw: BridgeResponse(
            raw_output="Invoice chunk",
            structured_output={"vendor_name": "Acme", "total_amount": 10.0},
        )
        res3 = extractor.extract(
            doc_path,
            options=ScryerExtractionOptions(
                template=InvoiceModel,
                max_context_chars=600,
            ),
        )
        assert mock_bridge.execute_with_heuristics.call_count > call_count_before
        assert res3.structured_data["vendor_name"] == "Acme"

    def test_ocr_text_woven_into_visual_elements_and_raw_text(self, tmp_path: Path):
        img = Image.new("RGB", (64, 64), color=(255, 255, 255))
        img_path = tmp_path / "scan.png"
        img.save(img_path, format="PNG")

        mock_bridge = MagicMock(spec=TriModeBridge)
        mock_bridge.execute_with_heuristics.return_value = BridgeResponse(
            raw_output="Scanned text body",
            structured_output={"summary": "Scanned sheet", "extracted_text": "Line 1: Confidential"},
            intent="VISION_OCR",
        )
        extractor = ScryerExtractor(bridge=mock_bridge)
        res = extractor.extract(img_path)
        assert res.visual_elements[0].ocr_text == "Line 1: Confidential"
        assert "Line 1: Confidential" in res.raw_text

    def test_async_batch_and_cli_doctor_templates(self, tmp_path: Path, capsys):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("Alpha content", encoding="utf-8")
        f2.write_text("Beta content", encoding="utf-8")

        extractor = ScryerExtractor()
        async_ext = AsyncScryerExtractor(extractor=extractor, max_concurrency=2)
        results = asyncio.run(
            async_ext.extract_batch_async(
                [f1, f2],
                options=ScryerExtractionOptions(enable_phase2=False),
            )
        )
        assert [r.raw_text for r in results] == ["Alpha content", "Beta content"]

        assert cli_main(["templates", "list"]) == 0
        out = capsys.readouterr().out
        assert "InvoiceExtraction" in out

        assert cli_main(["doctor"]) == 0
        doc_out = json.loads(capsys.readouterr().out)
        assert doc_out["packages"]["aurarouter"] is True

        out_dir = tmp_path / "out_batch"
        assert (
            cli_main(["batch", str(tmp_path), "--glob", "*.txt", "--no-phase2", "--out-dir", str(out_dir)])
            == 0
        )
        assert (out_dir / "a.txt.scryer.json").exists()
        assert (out_dir / "b.txt.scryer.json").exists()
