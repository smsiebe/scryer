"""unit tests for Phase 1 deterministic parsers (PDF, DOCX, XLSX, PPTX, Email, Image, Text)."""

from __future__ import annotations

import io
from email.message import EmailMessage
from pathlib import Path
import pytest
from PIL import Image

from scryer.contracts import ScryerExtractionOptions
from scryer.parsers import (
    DOCXParser,
    DeterministicParserRegistry,
    EmailParser,
    ImageParser,
    PDFParser,
    PPTXParser,
    TextParser,
    XLSXParser,
)


class TestDeterministicParsers:
    def test_image_parser(self):
        # Create minimal 100x50 red PNG
        img = Image.new("RGB", (100, 50), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        parser = ImageParser()
        assert parser.can_parse("photo.png")
        assert parser.can_parse("image/png")

        out = parser.parse(raw_bytes, "photo.png")
        assert out.metadata.dimensions == (100, 50)
        assert out.metadata.mime_type == "image/png"
        assert len(out.visual_elements) == 1
        assert out.visual_elements[0].width == 100
        assert out.visual_elements[0].height == 50
        assert out.visual_elements[0].data_base64 is not None

    def test_docx_parser(self):
        import docx

        doc = docx.Document()
        doc.core_properties.author = "Dr. Tester"
        doc.core_properties.title = "Architecture Spec"
        doc.add_heading("System Overview", level=1)
        doc.add_paragraph("This is the main body paragraph.")

        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Header A"
        table.cell(0, 1).text = "Header B"
        table.cell(1, 0).text = "Value 1"
        table.cell(1, 1).text = "Value 2"

        buf = io.BytesIO()
        doc.save(buf)
        raw_bytes = buf.getvalue()

        parser = DOCXParser()
        assert parser.can_parse("spec.docx")
        out = parser.parse(raw_bytes, "spec.docx")

        assert out.metadata.author == "Dr. Tester"
        assert out.metadata.title == "Architecture Spec"
        assert "This is the main body paragraph." in out.raw_text
        assert "| Header A | Header B |" in out.raw_text
        assert "| Value 1 | Value 2 |" in out.raw_text

    def test_xlsx_parser(self):
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Financials"
        ws.append(["Category", "Amount"])
        ws.append(["Hosting", 1200])
        ws.append(["Compute", 3400])

        buf = io.BytesIO()
        wb.save(buf)
        raw_bytes = buf.getvalue()

        parser = XLSXParser()
        assert parser.can_parse("fin.xlsx")
        out = parser.parse(raw_bytes, "fin.xlsx")

        assert "Financials" in out.metadata.sheet_names
        assert "| Category | Amount |" in out.raw_text
        assert "| Hosting | 1200 |" in out.raw_text
        assert "| Compute | 3400 |" in out.raw_text

    def test_pptx_parser(self):
        import pptx

        prs = pptx.Presentation()
        prs.core_properties.author = "Presenter"
        prs.core_properties.title = "Pitch Deck"

        slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]
        title.text = "Hello Scryer"
        subtitle.text = "Multimodal extraction pipeline"

        buf = io.BytesIO()
        prs.save(buf)
        raw_bytes = buf.getvalue()

        parser = PPTXParser()
        assert parser.can_parse("deck.pptx")
        out = parser.parse(raw_bytes, "deck.pptx")

        assert out.metadata.author == "Presenter"
        assert out.metadata.title == "Pitch Deck"
        assert "Hello Scryer" in out.raw_text
        assert "Multimodal extraction pipeline" in out.raw_text

    def test_email_parser_eml(self):
        msg = EmailMessage()
        msg["Subject"] = "Project Update"
        msg["From"] = "lead@example.com"
        msg["To"] = "team@example.com"
        msg["Date"] = "Wed, 13 Nov 2026 15:30:00 -0500"
        msg.set_content("We have completed Phase 1 extraction.\nAll parsers are green.")

        raw_bytes = msg.as_bytes()

        parser = EmailParser()
        assert parser.can_parse("mail.eml")
        out = parser.parse(raw_bytes, "mail.eml")

        assert out.metadata.author == "lead@example.com"
        assert out.metadata.title == "Project Update"
        assert "From: lead@example.com" in out.raw_text
        assert "We have completed Phase 1 extraction." in out.raw_text

    def test_pdf_parser_with_page_rendering(self):
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument.new()
        # Add an empty / low-text page
        pdf.new_page(200, 200)

        buf = io.BytesIO()
        pdf.save(buf)
        raw_bytes = buf.getvalue()

        parser = PDFParser()
        assert parser.can_parse("document.pdf")

        options = ScryerExtractionOptions(min_text_threshold=30, extract_images=True)
        out = parser.parse(raw_bytes, "document.pdf", options=options)

        assert out.metadata.page_count == 1
        assert 1 in out.low_text_pages
        # Low text page triggers rendered page visual element
        assert len(out.visual_elements) == 1
        assert out.visual_elements[0].element_type == "rendered_page"
        assert out.visual_elements[0].page_number == 1
        assert out.visual_elements[0].data_base64 is not None

    def test_registry_dispatch(self):
        reg = DeterministicParserRegistry()
        assert reg.can_parse("test.pdf")
        assert reg.can_parse("test.docx")
        assert reg.can_parse("test.xlsx")
        assert reg.can_parse("test.pptx")
        assert reg.can_parse("test.eml")
        assert reg.can_parse("test.png")
        assert reg.can_parse("test.txt")

        # Test text fallback
        txt_bytes = b"Plain text content here."
        out = reg.parse(txt_bytes, "notes.txt")
        assert out.raw_text == "Plain text content here."
        assert out.metadata.mime_type == "text/plain"
