"""in_process.py - In-process Python SDK bridge to AuraRouter (AuraRouterAPI)."""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Dict, Optional, Sequence
from aurarouter.api import APIConfig, AuraRouterAPI
from scryer.bridge.base import BaseRouterBridge, BridgeResponse, parse_structured_json
from scryer.contracts import ExtractedVisualElement

logger = logging.getLogger("Scryer.Bridge.InProcess")
_AURAROUTER_AVAILABLE = True


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

    def __init__(
        self,
        api: Optional[Any] = None,
        config_path: Optional[str] = None,
        enable_privacy: bool = True,
        enable_savings: bool = True,
    ):
        self._api = api
        self._config_path = config_path
        self._enable_privacy = enable_privacy
        self._enable_savings = enable_savings

    def is_available(self) -> bool:
        return _AURAROUTER_AVAILABLE

    def _get_api(self) -> Any:
        if self._api is not None:
            return self._api
        self._api = AuraRouterAPI(
            APIConfig(
                config_path=self._config_path,
                enable_savings=self._enable_savings,
                enable_privacy=self._enable_privacy,
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
        prefer_local: Optional[bool] = None,
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
        last_tried_model = ""

        def _track_tried(role: str, model_id: str, success: bool, elapsed: float) -> None:
            nonlocal last_tried_model
            if success and model_id:
                last_tried_model = model_id

        exec_kwargs: Dict[str, Any] = {
            "task": task,
            "context": context or "",
            "images": formatted_images if formatted_images else None,
            "response_schema": response_schema,
            "intent": intent,
            "execution_mode": execution_mode,
        }
        try:
            sig = inspect.signature(api.execute_task)
            has_varkw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if (has_varkw or "on_model_tried" in sig.parameters):
                exec_kwargs["on_model_tried"] = _track_tried
            if prefer_local is not None and (has_varkw or "prefer_local" in sig.parameters):
                exec_kwargs["prefer_local"] = bool(prefer_local)
            if route_override and (has_varkw or "role_override" in sig.parameters):
                exec_kwargs["role_override"] = route_override
        except (TypeError, ValueError):
            pass

        try:
            task_result = api.execute_task(**exec_kwargs)

            raw_text = task_result.output or ""
            structured, annotations = parse_structured_json(raw_text, response_schema)

            elapsed = getattr(task_result, "total_elapsed", 0.0) or (time.monotonic() - t0)
            model_id = getattr(task_result, "model_id", "") or last_tried_model
            provider = getattr(task_result, "provider", "") or ""
            in_tok = int(getattr(task_result, "input_tokens", 0) or 0)
            out_tok = int(getattr(task_result, "output_tokens", 0) or 0)
            sov_verdict = str(getattr(task_result, "sovereignty_verdict", "") or "")

            return BridgeResponse(
                raw_output=raw_text,
                structured_output=structured,
                intent=getattr(task_result, "intent", intent or "VISION_EXTRACTION"),
                complexity=getattr(task_result, "complexity", 5),
                elapsed_seconds=elapsed,
                model_id=model_id if isinstance(model_id, str) else "",
                provider=provider if isinstance(provider, str) else "",
                input_tokens=in_tok,
                output_tokens=out_tok,
                sovereignty_verdict=sov_verdict,
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

