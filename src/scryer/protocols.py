"""protocols.py - Static Protocol Interfaces for Scryer components.

Defines the pluggable contracts for deterministic parsers, router bridges,
template compilers, and the core extractor orchestrator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
from scryer.contracts import (
    ExtractionTemplate,
    Phase1Output,
    ScryerExtractionOptions,
    ScryerResult,
)


@runtime_checkable
class DeterministicParserProtocol(Protocol):
    """Protocol implemented by deterministic Phase 1 document and image parsers."""

    def can_parse(self, filename_or_mime: str) -> bool:
        """Return True if the parser can handle the provided filename extension or MIME type."""
        ...

    def parse(self, data: bytes, filename: str, options: Optional[ScryerExtractionOptions] = None) -> Phase1Output:
        """Parse raw file bytes deterministically into Phase1Output."""
        ...


@runtime_checkable
class RouterBridgeProtocol(Protocol):
    """Protocol for dispatching multimodal inference requests to AuraRouter (Python, IPC, or MAS)."""

    def execute_multimodal(
        self,
        task: str,
        images: Sequence[Any],
        response_schema: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        context: Optional[str] = None,
        execution_mode: Optional[str] = None,
        route_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute multimodal task with images and schema, returning output text, telemetry, and structured data."""
        ...

    def is_available(self) -> bool:
        """Return True if the router backend is reachable and configured."""
        ...


@runtime_checkable
class TemplateEngineProtocol(Protocol):
    """Protocol for compiling programmatic or declarative templates into JSON Schemas and prompt hints."""

    def compile(self, template: Any) -> Tuple[Dict[str, Any], List[str]]:
        """Compile a template specification into (response_schema_dict, prompt_guidelines)."""
        ...


@runtime_checkable
class ScryerExtractorProtocol(Protocol):
    """Core public extractor interface."""

    def extract(
        self,
        source: Union[str, Path, bytes],
        options: Optional[ScryerExtractionOptions] = None,
        filename: Optional[str] = None,
    ) -> ScryerResult:
        """Extract lossless metadata, raw text, structured schema payload, and visual elements."""
        ...
