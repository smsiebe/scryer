"""in_process.py - In-process Python SDK bridge to AuraRouter (AuraRouterAPI)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence
from scryer.bridge.base import BaseRouterBridge, BridgeResponse
from scryer.contracts import ExtractedVisualElement, safe_json_loads

logger = logging.getLogger("Scryer.Bridge.InProcess")

try:
    from aurarouter.api import APIConfig, AuraRouterAPI
    from aurarouter.media import ImageInput
    _AURAROUTER_AVAILABLE = True
except ImportError:
    _AURAROUTER_AVAILABLE = False


def _to_aurarouter_image(img: Any) -> Any:
    """Normalize ExtractedVisualElement, dict, or data URI into an object accepted by AuraRouter."""
    if isinstance(img, ExtractedVisualElement):
        if img.data_base64:
            return f"data:{img.mime_type};base64,{img.data_base64}"
        if img.source_path:
            return img.source_path
    if isinstance(img, dict):
        b64 = img.get("data_base64") or img.get("data")
        mime = img.get("mime_type", "image/png")
        if b64:
            return f"data:{mime};base64,{b64}"
    return img


class InProcessRouterBridge(BaseRouterBridge):
    """Direct in-process execution using the installed aurarouter Python package."""

    def __init__(self, api: Optional[Any] = None, config_path: Optional[str] = None):
        self._api = api
        self._config_path = config_path

    def is_available(self) -> bool:
        return _AURAROUTER_AVAILABLE

    def _get_api(self) -> Any:
        if self._api is not None:
            return self._api
        if not _AURAROUTER_AVAILABLE:
            raise RuntimeError("aurarouter package is not installed in the active environment.")
        self._api = AuraRouterAPI(
            APIConfig(
                config_path=self._config_path,
                enable_savings=False,
                enable_privacy=False,
            )
        )
        return self._api

    def execute_multimodal(
        self,
        task: str,
        images: Sequence[Any],
        response_schema: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        context: Optional[str] = None,
        execution_mode: Optional[str] = None,
        route_override: Optional[str] = None,
    ) -> BridgeResponse:
        t0 = time.monotonic()
        try:
            api = self._get_api()
        except Exception as e:
            return BridgeResponse(
                raw_output="",
                success=False,
                error_message=f"InProcessRouterBridge initialization error: {e}",
                elapsed_seconds=time.monotonic() - t0,
            )

        formatted_images = [_to_aurarouter_image(im) for im in images]

        try:
            task_result = api.execute_task(
                task=task,
                context=context,
                images=formatted_images if formatted_images else None,
                response_schema=response_schema,
                intent=intent,
                execution_mode=execution_mode,
            )

            raw_text = task_result.output or ""
            structured = None
            annotations: List[str] = []

            # Try to parse structured JSON if response_schema was requested
            if response_schema and raw_text:
                try:
                    structured = safe_json_loads(raw_text)
                except Exception:
                    # Sometimes LLM outputs markdown fences ```json ... ```
                    clean = raw_text.strip()
                    if clean.startswith("```"):
                        lines = clean.splitlines()
                        if len(lines) >= 3 and lines[0].startswith("```"):
                            clean = "\n".join(lines[1:-1])
                            try:
                                structured = safe_json_loads(clean)
                            except Exception:
                                pass
                    if structured is None:
                        annotations.append(f"Model output did not strictly parse to JSON: {raw_text[:200]}")

            elapsed = getattr(task_result, "total_elapsed", 0.0) or (time.monotonic() - t0)

            return BridgeResponse(
                raw_output=raw_text,
                structured_output=structured,
                intent=getattr(task_result, "intent", intent or "VISION_EXTRACTION"),
                complexity=getattr(task_result, "complexity", 5),
                elapsed_seconds=elapsed,
                success=True,
                annotations=annotations,
            )
        except Exception as e:
            logger.warning(f"InProcessRouterBridge execution failure: {e}")
            return BridgeResponse(
                raw_output="",
                success=False,
                error_message=str(e),
                elapsed_seconds=time.monotonic() - t0,
            )
