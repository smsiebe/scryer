"""base.py - Base router bridge and bridge response contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from scryer.contracts import safe_json_loads
from scryer.protocols import RouterBridgeProtocol


def parse_structured_json(
    raw_text: str,
    response_schema: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Parse structured JSON from raw model text, stripping optional fences and routing context."""
    structured = None
    annotations: List[str] = []
    if not (response_schema and raw_text):
        return structured, annotations

    clean = re.sub(
        r"\s*<!--\s*_aura_routing_context:.*?-->\s*",
        "",
        raw_text,
        flags=re.DOTALL,
    ).strip()

    try:
        structured = safe_json_loads(clean)
    except Exception:
        fence_match = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            clean,
            flags=re.IGNORECASE,
        )
        if fence_match:
            try:
                structured = safe_json_loads(fence_match.group(1).strip())
            except Exception:
                pass
        if structured is None:
            first_brace = clean.find("{")
            last_brace = clean.rfind("}")
            if first_brace != -1 and last_brace > first_brace:
                try:
                    structured = safe_json_loads(clean[first_brace : last_brace + 1])
                except Exception:
                    pass
        if structured is None:
            annotations.append(
                f"Model output did not strictly parse to JSON: {raw_text[:200]}"
            )
    return structured, annotations


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
    input_tokens: int = 0
    output_tokens: int = 0
    sovereignty_verdict: str = ""
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
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "sovereignty_verdict": self.sovereignty_verdict,
            "success": self.success,
            "error_message": self.error_message,
        }


class BaseRouterBridge(ABC, RouterBridgeProtocol):
    """Abstract base class for Tri-Mode AuraRouter bridges."""

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
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
        pass
