"""unit tests for Phase 2 Tri-Mode AuraRouter Bridge and Route-Assistance Heuristics."""

from __future__ import annotations

import json
from unittest.mock import MagicMock
import pytest

from scryer.bridge import (
    BridgeResponse,
    InProcessRouterBridge,
    IpcRouterBridge,
    MasRouterBridge,
    RouteAssistanceHeuristics,
    TriModeBridge,
)
from scryer.contracts import (
    DocumentMetadata,
    ExtractedVisualElement,
    Phase1Output,
    ScryerExtractionOptions,
    TriMode,
)


class TestRouterBridgeAndHeuristics:
    def test_route_assistance_heuristics(self):
        # 1. Scanned / Low text document
        phase1_scan = Phase1Output(
            metadata=DocumentMetadata(page_count=2),
            raw_text="A few words.",
            low_text_pages=[1, 2],
            visual_elements=[
                ExtractedVisualElement(element_id="p1", element_type="rendered_page")
            ],
        )
        hints = RouteAssistanceHeuristics.derive_routing_hints(phase1_scan)
        assert hints["intent"] == "VISION_OCR"
        assert hints["prefer_local"] is True

        # 2. Document with multiple embedded photos/diagrams
        phase1_visual = Phase1Output(
            metadata=DocumentMetadata(page_count=5),
            raw_text="Extensive report with many pages of text.",
            visual_elements=[
                ExtractedVisualElement(element_id="img1", element_type="image"),
                ExtractedVisualElement(element_id="img2", element_type="image"),
            ],
        )
        hints_vis = RouteAssistanceHeuristics.derive_routing_hints(phase1_visual)
        assert hints_vis["intent"] == "VISION_EXTRACTION"
        assert hints_vis["complexity"] >= 5

    def test_in_process_bridge_with_mock_api(self):
        mock_api = MagicMock()
        mock_task_res = MagicMock()
        mock_task_res.output = json.dumps({"summary": "A corporate invoice", "total": 1250.0})
        mock_task_res.intent = "VISION_EXTRACTION"
        mock_task_res.complexity = 4
        mock_task_res.total_elapsed = 0.45
        mock_api.execute_task.return_value = mock_task_res

        bridge = InProcessRouterBridge(api=mock_api)
        assert bridge.is_available() is True

        schema = {"type": "object", "properties": {"total": {"type": "number"}}}
        visual = ExtractedVisualElement(
            element_id="img1",
            data_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            mime_type="image/png",
        )

        resp = bridge.execute_multimodal(
            task="Extract invoice total",
            images=[visual],
            response_schema=schema,
            intent="VISION_EXTRACTION",
        )

        assert resp.success is True
        assert resp.structured_output == {"summary": "A corporate invoice", "total": 1250.0}
        assert resp.intent == "VISION_EXTRACTION"
        assert resp.complexity == 4

        # Verify call args passed to AuraRouterAPI
        mock_api.execute_task.assert_called_once()
        call_kwargs = mock_api.execute_task.call_args[1]
        assert call_kwargs["response_schema"] == schema
        assert call_kwargs["intent"] == "VISION_EXTRACTION"

    def test_in_process_bridge_markdown_json_parsing(self):
        mock_api = MagicMock()
        mock_task_res = MagicMock()
        mock_task_res.output = "```json\n{\"category\": \"Receipt\", \"vendor\": \"Acme Corp\"}\n```"
        mock_task_res.intent = "VISION_OCR"
        mock_task_res.complexity = 3
        mock_api.execute_task.return_value = mock_task_res

        bridge = InProcessRouterBridge(api=mock_api)
        resp = bridge.execute_multimodal(
            task="Extract vendor",
            images=[],
            response_schema={"type": "object"},
        )
        assert resp.success is True
        assert resp.structured_output == {"category": "Receipt", "vendor": "Acme Corp"}

    def test_tri_mode_bridge_fallback(self):
        # Create TriModeBridge with mocked backend
        mock_in_process = MagicMock()
        mock_in_process.is_available.return_value = True
        mock_in_process.execute_multimodal.return_value = BridgeResponse(
            raw_output="Fallback executed successfully",
            success=True,
        )

        mock_ipc = MagicMock()
        mock_ipc.is_available.return_value = False

        tri = TriModeBridge(in_process=mock_in_process, ipc=mock_ipc)

        # Request IPC mode, but IPC is unavailable -> auto fallback to in-process
        resp = tri.execute_multimodal(
            task="Test fallback",
            images=[],
            tri_mode=TriMode.IPC,
        )
        assert resp.success is True
        assert resp.raw_output == "Fallback executed successfully"
        mock_in_process.execute_multimodal.assert_called_once()

