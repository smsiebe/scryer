"""End-to-end conformance tests for real-world image formats, presets, and routing context hardening."""

from __future__ import annotations

import base64
from email.message import EmailMessage
import io
from typing import Any, Dict, Optional

import docx
from PIL import Image
import pptx
from pptx.util import Inches
import pytest

from scryer.bridge.base import BaseRouterBridge, BridgeResponse, parse_structured_json
from scryer.bridge.tri_mode import TriModeBridge
from scryer.contracts import ScryerExtractionOptions
from scryer.extractor import ScryerExtractor
from scryer.parsers.base import prepare_visual_image_bytes
from scryer.parsers.docx import DOCXParser
from scryer.parsers.email import EmailParser
from scryer.parsers.image import ImageParser
from scryer.parsers.pptx import PPTXParser
from scryer.templates.compiler import TemplateCompiler


class _VisualSummaryStubBridge(BaseRouterBridge):
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload

    def is_available(self) -> bool:
        return True

    def execute_multimodal(
        self,
        task: str,
        images: Any,
        response_schema: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        context: Optional[str] = None,
        execution_mode: Optional[str] = None,
        route_override: Optional[str] = None,
        prefer_local: Optional[bool] = None,
    ) -> BridgeResponse:
        return BridgeResponse(
            raw_output="",
            structured_output=dict(self.payload),
            intent=intent or "VISION_EXTRACTION",
            model_id="stub-vlm",
            provider="local",
            success=True,
        )


def _make_png_bytes(w: int, h: int, mode: str = "RGB") -> bytes:
    color = (20, 120, 220, 128) if mode == "RGBA" else (20, 120, 220)
    img = Image.new(mode, (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.conformance
class TestE2ERealFormatsAndPresets:
    def test_parse_structured_json_strips_aura_routing_context_and_prose(self):
        schema = {"type": "object", "properties": {"vendor": {"type": "string"}}}

        raw_with_comment = (
            '{"vendor": "Northwind Corp"}\n'
            '<!-- _aura_routing_context: {"complexity": 4, "selected_model": "qwen-vl"} -->'
        )
        parsed, notes = parse_structured_json(raw_with_comment, schema)
        assert parsed == {"vendor": "Northwind Corp"}
        assert notes == []

        raw_fenced_with_prose = (
            "Here is the extracted result:\n"
            "```json\n"
            '{"vendor": "Contoso Ltd"}\n'
            "```\n"
            '<!-- _aura_routing_context: {"latency_ms": 120} -->'
        )
        parsed2, notes2 = parse_structured_json(raw_fenced_with_prose, schema)
        assert parsed2 == {"vendor": "Contoso Ltd"}
        assert notes2 == []

        raw_brace_fallback = (
            'Analysis complete: {"vendor": "Fabrikam"} -- end of response.'
        )
        parsed3, notes3 = parse_structured_json(raw_brace_fallback, schema)
        assert parsed3 == {"vendor": "Fabrikam"}
        assert notes3 == []

    def test_prepare_visual_image_handles_mpo_cmyk_rgba_and_exif(self):
        # 1. RGBA oversized JPEG normalization & EXIF stripping
        rgba_png = _make_png_bytes(300, 200, mode="RGBA")
        # Force a simulated MPO format image via Pillow or monkeypatched Image.open format
        img_rgba = Image.open(io.BytesIO(rgba_png))
        img_rgba.format = "MPO"
        buf_jpeg = io.BytesIO()
        img_rgba.convert("RGB").save(buf_jpeg, format="JPEG")
        jpeg_with_exif_img = Image.open(io.BytesIO(buf_jpeg.getvalue()))
        exif = jpeg_with_exif_img.getexif()
        exif[271] = "Apple Camera"
        exif_buf = io.BytesIO()
        jpeg_with_exif_img.save(exif_buf, format="JPEG", exif=exif.tobytes())
        exif_bytes = exif_buf.getvalue()

        # Verify EXIF is present before stripping
        parsed_before = ImageParser().parse(
            exif_bytes,
            "iphone_photo.jpg",
            options=ScryerExtractionOptions(strip_exif=False),
        )
        assert parsed_before.metadata.exif_tags.get("Make") == "Apple Camera"

        # Now downscale + strip EXIF on CMYK JPEG
        cmyk_img = Image.new("CMYK", (240, 160), color=(0, 100, 100, 0))
        cmyk_buf = io.BytesIO()
        cmyk_img.save(cmyk_buf, format="JPEG")
        out_bytes, out_mime, dims, downscaled, stripped = prepare_visual_image_bytes(
            cmyk_buf.getvalue(),
            mime_type="image/jpeg",
            max_image_pixels=4000,
            strip_exif=True,
        )
        assert out_mime == "image/jpeg"
        assert downscaled is True
        assert stripped is True
        assert dims is not None and (dims[0] * dims[1]) <= 4200
        with Image.open(io.BytesIO(out_bytes)) as check_img:
            assert check_img.mode == "RGB"
            assert check_img.format == "JPEG"

        # 2. Verify ImageParser handles RGBA -> PNG downscale cleanly
        rgba_out = ImageParser().parse(
            rgba_png,
            "diagram.png",
            options=ScryerExtractionOptions(max_image_pixels=3000, strip_exif=True),
        )
        assert rgba_out.metadata.custom_fields["downscaled"] is True
        assert rgba_out.metadata.custom_fields["exif_stripped"] is True
        decoded = base64.b64decode(rgba_out.visual_elements[0].data_base64 or "")
        with Image.open(io.BytesIO(decoded)) as check_png:
            assert (check_png.width * check_png.height) <= 3200

    def test_embedded_images_respect_max_pixels_in_docx_pptx_and_email(self):
        large_png = _make_png_bytes(320, 200, mode="RGBA")  # 64,000 px
        opts = ScryerExtractionOptions(max_image_pixels=4000, strip_exif=True)

        # 1. DOCX with embedded picture
        doc = docx.Document()
        doc.add_paragraph("Diagram section")
        doc.add_picture(io.BytesIO(large_png), width=docx.shared.Inches(2.0))
        docx_buf = io.BytesIO()
        doc.save(docx_buf)
        docx_out = DOCXParser().parse(docx_buf.getvalue(), "embedded.docx", options=opts)
        assert len(docx_out.visual_elements) == 1
        assert docx_out.visual_elements[0].width is not None
        assert (docx_out.visual_elements[0].width * docx_out.visual_elements[0].height) <= 4200

        # 2. PPTX with embedded picture
        prs = pptx.Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.add_picture(io.BytesIO(large_png), Inches(1), Inches(1))
        pptx_buf = io.BytesIO()
        prs.save(pptx_buf)
        pptx_out = PPTXParser().parse(pptx_buf.getvalue(), "deck.pptx", options=opts)
        assert len(pptx_out.visual_elements) == 1
        assert (pptx_out.visual_elements[0].width * pptx_out.visual_elements[0].height) <= 4200

        # 3. EML with embedded image attachment
        msg = EmailMessage()
        msg["Subject"] = "Chart"
        msg["From"] = "alice@example.com"
        msg["To"] = "bob@example.com"
        msg.set_content("See chart attached.")
        msg.add_attachment(large_png, maintype="image", subtype="png", filename="chart.png")
        eml_out = EmailParser().parse(msg.as_bytes(), "mail.eml", options=opts)
        assert len(eml_out.visual_elements) == 1
        assert (eml_out.visual_elements[0].width * eml_out.visual_elements[0].height) <= 4200

    def test_template_compiler_resolves_preset_names_and_rejects_missing_paths(self):
        compiler = TemplateCompiler()

        inv_schema, _, _ = compiler.resolve_template("invoice")
        assert "vendor_name" in inv_schema["properties"]

        scene_schema, _, _ = compiler.resolve_template("Scene-Analysis")
        assert "detected_objects" in scene_schema["properties"]

        summary_schema, _, _ = compiler.resolve_template("summary")
        assert "topics" in summary_schema["properties"]

        with pytest.raises(FileNotFoundError):
            compiler.resolve_template("does_not_exist_preset_or_path.yaml")

    def test_extractor_propagates_summary_to_ocr_and_raw_text_for_visual_only_docs(self):
        img_bytes = _make_png_bytes(100, 80)
        bridge = TriModeBridge(
            in_process=_VisualSummaryStubBridge(
                {
                    "summary": "Network rack photograph showing Patch Panel 3B and Switch SW-09.",
                    "topics": ["networking", "hardware"],
                }
            )
        )
        extractor = ScryerExtractor(bridge=bridge)
        res = extractor.extract(
            img_bytes,
            filename="rack.png",
            options=ScryerExtractionOptions(template="summary"),
        )

        assert res.structured_data["topics"] == ["networking", "hardware"]
        assert "Patch Panel 3B" in (res.visual_elements[0].ocr_text or "")
        assert "Switch SW-09" in res.raw_text

