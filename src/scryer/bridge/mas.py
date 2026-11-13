"""mas.py - AuraGrid Multi-Agent System (MAS) pub/sub bridge for federated cell execution."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence
from scryer.bridge.base import BaseRouterBridge, BridgeResponse
from scryer.bridge.in_process import _to_aurarouter_image

logger = logging.getLogger("Scryer.Bridge.MAS")

try:
    import auragrid
    _AURAGRID_AVAILABLE = True
except ImportError:
    _AURAGRID_AVAILABLE = False


class MasRouterBridge(BaseRouterBridge):
    """Dispatches inference requests across an AuraGrid Cell using MAS eventing."""

    def __init__(self, grid_client: Optional[Any] = None):
        self.grid_client = grid_client

    def is_available(self) -> bool:
        return _AURAGRID_AVAILABLE and self.grid_client is not None

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
        if not self.is_available():
            return BridgeResponse(
                raw_output="",
                success=False,
                error_message="AuraGrid MAS environment or client is not connected.",
                elapsed_seconds=time.monotonic() - t0,
            )

        req_id = f"inf_{uuid.uuid4().hex[:12]}"
        topic = "aurarouter.inference_requests"
        payload = {
            "request_id": req_id,
            "task": task,
            "images": [_to_aurarouter_image(im) for im in images],
            "response_schema": response_schema,
            "intent": intent,
            "context": context,
            "execution_mode": execution_mode,
        }

        try:
            # Publish event across AuraGrid
            self.grid_client.publish(topic, payload)
            # Synchronous wait for response topic or timeout
            resp_event = self.grid_client.wait_for_event(
                f"aurarouter.inference_responses.{req_id}",
                timeout=30.0,
            )
            if resp_event:
                data = resp_event.get("data", {})
                return BridgeResponse(
                    raw_output=data.get("output", ""),
                    structured_output=data.get("structured_data"),
                    intent=data.get("intent", intent or "VISION_EXTRACTION"),
                    complexity=data.get("complexity", 5),
                    elapsed_seconds=time.monotonic() - t0,
                    model_id=data.get("model_id", "grid-routed"),
                    success=True,
                )
            return BridgeResponse(
                raw_output="",
                success=False,
                error_message="AuraGrid MAS inference request timed out after 30 seconds.",
                elapsed_seconds=time.monotonic() - t0,
            )
        except Exception as e:
            return BridgeResponse(
                raw_output="",
                success=False,
                error_message=f"AuraGrid MAS eventing error: {e}",
                elapsed_seconds=time.monotonic() - t0,
            )

