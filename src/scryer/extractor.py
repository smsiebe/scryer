"""extractor.py - ScryerExtractor: Two-Phase Lossless Context Extraction Orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from scryer.bridge import BridgeResponse, TriModeBridge
from scryer.contracts import (
    DocumentMetadata,
    Phase1Output,
    ScryerExtractionOptions,
    ScryerResult,
)
from scryer.parsers import DeterministicParserRegistry, compute_sha256, detect_mime
from scryer.protocols import ScryerExtractorProtocol
from scryer.templates import TemplateCompiler

logger = logging.getLogger("Scryer.Extractor")


class ScryerExtractor(ScryerExtractorProtocol):
    """Orchestrates deterministic Phase 1 extraction and AuraRouter Phase 2 multimodal enrichment."""

    def __init__(
        self,
        parser_registry: Optional[DeterministicParserRegistry] = None,
        bridge: Optional[TriModeBridge] = None,
        template_compiler: Optional[TemplateCompiler] = None,
    ):
        self.parser_registry = parser_registry or DeterministicParserRegistry()
        self.bridge = bridge or TriModeBridge()
        self.template_compiler = template_compiler or TemplateCompiler()

    def extract(
        self,
        source: Union[str, Path, bytes],
        options: Optional[ScryerExtractionOptions] = None,
        filename: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> ScryerResult:
        """Extract metadata, text, visual elements, and structured context from document or image."""
        opts = options or ScryerExtractionOptions()

        # 1. Resolve source data, path, and filename
        source_path_str: Optional[str] = None
        if isinstance(source, (str, Path)):
            path_obj = Path(source)
            if not path_obj.exists():
                raise FileNotFoundError(f"Source file not found: {path_obj}")
            raw_bytes = path_obj.read_bytes()
            effective_filename = filename or path_obj.name
            source_path_str = str(path_obj.resolve())
        elif isinstance(source, bytes):
            raw_bytes = source
            effective_filename = filename or "document.bin"
        else:
            raise TypeError(f"Unsupported source type: {type(source).__name__}")

        sha256_hash = compute_sha256(raw_bytes)
        effective_doc_id = doc_id or effective_filename
        mime_type = detect_mime(effective_filename)

        # 2. Phase 1: Deterministic Extraction
        phase1: Phase1Output = self.parser_registry.parse(
            raw_bytes,
            effective_filename,
            options=opts,
        )

        # Ensure metadata has file size and mime type
        if phase1.metadata.file_size == 0:
            metadata_dict = phase1.metadata.model_dump()
            metadata_dict["file_size"] = len(raw_bytes)
            metadata_dict["mime_type"] = mime_type
            phase1_metadata = DocumentMetadata(**metadata_dict)
        else:
            phase1_metadata = phase1.metadata

        # 3. Phase 2: Multimodal Enrichment via AuraRouter
        structured_data: Dict[str, Any] = {}
        unstructured_annotations: List[str] = []
        route_telemetry: Dict[str, Any] = {}

        should_invoke_phase2 = opts.enable_phase2 and (
            bool(phase1.visual_elements)
            or bool(phase1.low_text_pages)
            or opts.template is not None
            or opts.custom_prompt is not None
        )

        if should_invoke_phase2:
            # Compile template schema and guidelines
            response_schema, guidelines = self.template_compiler.compile(opts.template)

            bridge_resp: BridgeResponse = self.bridge.execute_with_heuristics(
                phase1=phase1,
                options=opts,
                task_prompt=opts.custom_prompt,
                response_schema=response_schema,
                prompt_guidelines=guidelines,
            )

            if bridge_resp.structured_output:
                structured_data = bridge_resp.structured_output
            elif bridge_resp.raw_output.strip():
                unstructured_annotations.append(bridge_resp.raw_output.strip())

            if bridge_resp.annotations:
                unstructured_annotations.extend(bridge_resp.annotations)

            route_telemetry = bridge_resp.to_telemetry()
        else:
            route_telemetry = {"phase2_executed": False, "reason": "No visual elements or template requested"}

        # 4. Assemble Universal Lossless ScryerResult
        return ScryerResult(
            doc_id=effective_doc_id,
            source_path=source_path_str,
            mime_type=mime_type,
            sha256=sha256_hash,
            metadata=phase1_metadata,
            raw_text=phase1.raw_text,
            structured_data=structured_data,
            visual_elements=phase1.visual_elements,
            unstructured_annotations=unstructured_annotations,
            route_telemetry=route_telemetry,
        )
