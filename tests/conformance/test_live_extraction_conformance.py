"""conformance test suite: unmocked end-to-end document and image extraction conformance."""

from __future__ import annotations

import io
from email.message import EmailMessage
from pathlib import Path
from PIL import Image
import pytest
import pypdfium2 as pdfium
import docx
import openpyxl
import pptx

from scryer.contracts import ScryerExtractionOptions, ScryerResult, safe_json_loads
from scryer.extractor import ScryerExtractor
from scryer.templates import INVOICE_TEMPLATE, InvoiceModel


@pytest.mark.conformance
class TestLiveExtractionConformance:
    def test_live_pdf_conformance(self, tmp_path: Path):
        pdf_path = tmp_path / "live_sample.pdf"
        pdf = pdfium.PdfDocument.new()
        # Add a page
        pdf.new_page(300, 300)
        buf = io.BytesIO()
        pdf.save(buf)
        pdf_path.write_bytes(buf.getvalue())

        extractor = ScryerExtractor()
        options = ScryerExtractionOptions(min_text_threshold=10, extract_images=True, enable_phase2=False)
        result = extractor.extract(pdf_path, options=options)

        assert isinstance(result, ScryerResult)
        assert result.mime_type == "application/pdf"
        assert result.metadata.page_count == 1
        assert len(result.sha256) == 64
        # Since it had 0 chars, low-text rendering kicked in
        assert len(result.visual_elements) == 1
        assert result.visual_elements[0].mime_type == "image/png"

        # Verify JSON roundtrip
        json_repr = result.to_json()
        deserialized = safe_json_loads(json_repr)
        assert deserialized["doc_id"] == "live_sample.pdf"

    def test_live_docx_conformance(self, tmp_path: Path):
        doc_path = tmp_path / "live_spec.docx"
        doc = docx.Document()
        doc.core_properties.author = "System Architect"
        doc.add_heading("Operational Directives", level=1)
        doc.add_paragraph("Deterministic Phase 1 extractors operate without LLM latency.")

        tbl = doc.add_table(rows=2, cols=2)
        tbl.cell(0, 0).text = "Metric"
        tbl.cell(0, 1).text = "Target"
        tbl.cell(1, 0).text = "Coverage"
        tbl.cell(1, 1).text = "100%"
        doc.save(str(doc_path))

        extractor = ScryerExtractor()
        result = extractor.extract(doc_path, options=ScryerExtractionOptions(enable_phase2=False))

        assert result.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert result.metadata.author == "System Architect"
        assert "Operational Directives" in result.raw_text
        assert "| Metric | Target |" in result.raw_text
        assert "| Coverage | 100% |" in result.raw_text

    def test_live_xlsx_conformance(self, tmp_path: Path):
        xlsx_path = tmp_path / "live_budget.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Q4_Expenses"
        ws.append(["Department", "Cost"])
        ws.append(["Platform", 50000])
        wb.save(str(xlsx_path))

        extractor = ScryerExtractor()
        result = extractor.extract(xlsx_path, options=ScryerExtractionOptions(enable_phase2=False))

        assert result.mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert "Q4_Expenses" in result.metadata.sheet_names
        assert "| Department | Cost |" in result.raw_text
        assert "| Platform | 50000 |" in result.raw_text

    def test_live_pptx_conformance(self, tmp_path: Path):
        pptx_path = tmp_path / "live_deck.pptx"
        prs = pptx.Presentation()
        prs.core_properties.author = "Lead Presenter"
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = "Scryer Architecture"
        slide.placeholders[1].text = "Lossless extraction wrapper for documents and images."
        prs.save(str(pptx_path))

        extractor = ScryerExtractor()
        result = extractor.extract(pptx_path, options=ScryerExtractionOptions(enable_phase2=False))

        assert result.mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        assert result.metadata.author == "Lead Presenter"
        assert "Scryer Architecture" in result.raw_text

    def test_live_image_conformance(self, tmp_path: Path):
        img_path = tmp_path / "live_diagram.png"
        img = Image.new("RGBA", (256, 128), color=(20, 40, 80, 255))
        img.save(str(img_path))

        extractor = ScryerExtractor()
        result = extractor.extract(img_path, options=ScryerExtractionOptions(enable_phase2=False))

        assert result.mime_type == "image/png"
        assert result.metadata.dimensions == (256, 128)
        assert len(result.visual_elements) == 1
        assert result.visual_elements[0].width == 256
        assert result.visual_elements[0].height == 128

    def test_live_email_conformance(self, tmp_path: Path):
        eml_path = tmp_path / "live_message.eml"
        msg = EmailMessage()
        msg["Subject"] = "Extraction Conformance Verified"
        msg["From"] = "architect@auracoredynamics.com"
        msg["To"] = "engineering@auracoredynamics.com"
        msg.set_content("Live email parsing executes deterministically.")
        eml_path.write_bytes(msg.as_bytes())

        extractor = ScryerExtractor()
        result = extractor.extract(eml_path, options=ScryerExtractionOptions(enable_phase2=False))

        assert result.mime_type == "message/rfc822"
        assert result.metadata.author == "architect@auracoredynamics.com"
        assert "Extraction Conformance Verified" in result.raw_text
        assert "Live email parsing executes deterministically." in result.raw_text

