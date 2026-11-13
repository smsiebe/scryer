"""base.py - Base router bridge and bridge response contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from scryer.protocols import RouterBridgeProtocol


@dataclass
class BridgeResponse:
    """Unified response from an AuraRouter bridge invocation."""

    raw_output: str
    structured_output: Optional[Dict[str, Any]] = None
    intent: str = ""
    complexity: int = 5
    elapsed_seconds: float = 0.0
    model_id: str = ""
    provider: str = ""
    success: bool = True
    error_message: Optional[str] = None
    annotations: List[str] = field(default_factory=list)

    def to_telemetry(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "complexity": self.complexity,
            "elapsed_seconds": self.elapsed_seconds,
            "model_id": self.model_id,
            "provider": self.provider,
            "success": self.success,
            "error_message": self.error_message,
        }


class BaseRouterBridge(RouterBridgeProtocol):
    """Abstract base class for Tri-Mode AuraRouter bridges."""

    def is_available(self) -> bool:
        raise NotImplementedError

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
        raise NotImplementedError
