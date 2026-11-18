"""extractor.py - ScryerExtractor & AsyncScryerExtractor: Two-Phase Lossless Context Extraction."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from scryer.bridge import BridgeResponse, TriModeBridge
from scryer.cache import DiskExtractionCache, build_cache_key
from scryer.contracts import (
    DocumentMetadata,
    ExtractedVisualElement,
    Phase1Output,
    ScryerExtractionOptions,
    ScryerResult,
)
from scryer.parsers import DeterministicParserRegistry, compute_sha256, detect_mime
from scryer.protocols import ExtractionCacheProtocol, ScryerExtractorProtocol
from scryer.templates import TemplateCompiler, validate_structured_output
from scryer.normalizers import canonicalize_extracted_entities

logger = logging.getLogger("Scryer.Extractor")


def _chunk_text(text: str, max_chars: int, overlap: int = 400) -> List[str]:
    """Split long Phase 1 text into overlapping slices within ``max_chars``."""
    if not text or len(text) <= max_chars:
        return [text] if text else [""]
    step = max(1, max_chars - min(overlap, max_chars // 4))
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks


def _merge_structured_dicts(
    accumulated: Dict[str, Any],
    incoming: Dict[str, Any],
    annotations: List[str],
) -> Dict[str, Any]:
    """Merge batch/chunk structured extraction dicts losslessly."""
    if not incoming:
        return accumulated
    merged = dict(accumulated)
    for k, val in incoming.items():
        if val is None or val == "" or val == []:
            continue
        if k not in merged or merged[k] is None or merged[k] == "" or merged[k] == []:
            merged[k] = val
        elif isinstance(merged[k], list) and isinstance(val, list):
            seen = set()
            combined: List[Any] = []
            for item in merged[k] + val:
                marker = repr(item)
                if marker not in seen:
                    seen.add(marker)
                    combined.append(item)
            merged[k] = combined
        elif isinstance(merged[k], dict) and isinstance(val, dict):
            merged[k] = _merge_structured_dicts(merged[k], val, annotations)
        elif merged[k] != val:
            annotations.append(f"Additional batch value for '{k}': {val}")
    return merged


class ScryerExtractor(ScryerExtractorProtocol):
    """Orchestrates deterministic Phase 1 extraction and AuraRouter Phase 2 multimodal enrichment."""

    def __init__(
        self,
        parser_registry: Optional[DeterministicParserRegistry] = None,
        bridge: Optional[TriModeBridge] = None,
        template_compiler: Optional[TemplateCompiler] = None,
        cache: Optional[ExtractionCacheProtocol] = None,
    ):
        self.parser_registry = parser_registry or DeterministicParserRegistry()
        self.bridge = bridge or TriModeBridge()
        self.template_compiler = template_compiler or TemplateCompiler()
        self.cache = cache

    def _resolve_cache(self, opts: ScryerExtractionOptions) -> Optional[ExtractionCacheProtocol]:
        if self.cache is not None:
            return self.cache
        if opts.cache_dir:
            return DiskExtractionCache(opts.cache_dir)
        return None

    def _execute_phase2_batches(
        self,
        phase1: Phase1Output,
        opts: ScryerExtractionOptions,
        response_schema: Dict[str, Any],
        guidelines: List[str],
    ) -> Tuple[Dict[str, Any], List[str], List[ExtractedVisualElement], Dict[str, Any]]:
        """Run Phase 2 across bounded text chunks and visual batches with retries."""
        text_chunks = _chunk_text(phase1.raw_text, max(500, opts.max_context_chars))
        batch_size = max(1, opts.max_images_per_call)
        visuals = list(phase1.visual_elements)
        visual_batches: List[List[ExtractedVisualElement]] = (
            [visuals[i : i + batch_size] for i in range(0, len(visuals), batch_size)]
            if visuals
            else [[]]
        )

        # Single fast path when no batching/chunking is required
        single_pass = len(text_chunks) == 1 and len(visual_batches) == 1
        structured_acc: Dict[str, Any] = {}
        annotations: List[str] = []
        updated_visuals: List[ExtractedVisualElement] = []
        telemetry_runs: List[Dict[str, Any]] = []
        last_telemetry: Dict[str, Any] = {}

        work_units: List[Tuple[str, List[ExtractedVisualElement]]] = []
        if single_pass:
            work_units.append((text_chunks[0], visual_batches[0]))
        else:
            # Pair primary text chunk with each visual batch, then remaining text chunks
            for idx, v_batch in enumerate(visual_batches):
                t_chunk = text_chunks[idx] if idx < len(text_chunks) else ""
                work_units.append((t_chunk, v_batch))
            if len(text_chunks) > len(visual_batches):
                for t_chunk in text_chunks[len(visual_batches) :]:
                    work_units.append((t_chunk, []))

        for unit_idx, (chunk_txt, v_batch) in enumerate(work_units):
            slice_phase1 = (
                phase1
                if single_pass
                else phase1.model_copy(
                    update={
                        "raw_text": chunk_txt,
                        "visual_elements": v_batch,
                    }
                )
            )

            bridge_resp = self._invoke_with_retries(
                phase1=slice_phase1,
                opts=opts,
                response_schema=response_schema,
                guidelines=guidelines,
            )

            telemetry = bridge_resp.to_telemetry()
            telemetry["batch_index"] = unit_idx
            telemetry_runs.append(telemetry)
            last_telemetry = telemetry

            extracted_text_for_batch = ""
            if bridge_resp.structured_output:
                structured_acc = _merge_structured_dicts(
                    structured_acc,
                    bridge_resp.structured_output,
                    annotations,
                )
                ext_val = (
                    bridge_resp.structured_output.get("extracted_text")
                    or bridge_resp.structured_output.get("text_content")
                    or bridge_resp.structured_output.get("summary")
                )
                if isinstance(ext_val, str) and ext_val.strip():
                    extracted_text_for_batch = ext_val.strip()
            elif bridge_resp.raw_output.strip():
                raw_out = bridge_resp.raw_output.strip()
                annotations.append(raw_out)
                extracted_text_for_batch = raw_out

            if bridge_resp.annotations:
                annotations.extend(bridge_resp.annotations)

            # Propagate OCR/vision text back onto visual elements in this batch
            for ve in v_batch:
                if extracted_text_for_batch and not ve.ocr_text:
                    updated_visuals.append(ve.model_copy(update={"ocr_text": extracted_text_for_batch}))
                else:
                    updated_visuals.append(ve)

        if len(work_units) > 1:
            last_telemetry = {
                **last_telemetry,
                "batch_count": len(work_units),
                "batches": telemetry_runs,
            }

        return structured_acc, annotations, (updated_visuals if visuals else []), last_telemetry

    def _invoke_with_retries(
        self,
        phase1: Phase1Output,
        opts: ScryerExtractionOptions,
        response_schema: Dict[str, Any],
        guidelines: List[str],
    ) -> BridgeResponse:
        attempts = max(1, 1 + opts.max_retries)
        last_resp: Optional[BridgeResponse] = None
        for _ in range(attempts):
            resp = self.bridge.execute_with_heuristics(
                phase1=phase1,
                options=opts,
                task_prompt=opts.custom_prompt,
                response_schema=response_schema,
                prompt_guidelines=guidelines,
            )
            last_resp = resp
            if resp.success:
                return resp
        return last_resp or BridgeResponse(raw_output="", success=False, error_message="Phase 2 invocation failed")

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

        # Compile template early for schema resolution + cache keying
        response_schema, guidelines, pydantic_cls = self.template_compiler.resolve_template(
            opts.template
        )

        # Check content-addressed cache if configured
        cache = self._resolve_cache(opts)
        cache_key: Optional[str] = None
        if cache is not None:
            cache_key = build_cache_key(
                doc_sha256=sha256_hash,
                schema=response_schema if opts.enable_phase2 else None,
                options_payload={
                    "enable_phase2": opts.enable_phase2,
                    "force_phase2": opts.force_phase2,
                    "custom_prompt": opts.custom_prompt,
                    "strip_exif": opts.strip_exif,
                    "min_text_threshold": opts.min_text_threshold,
                    "execution_mode": opts.execution_mode,
                    "route_override": opts.route_override,
                },
            )
            cached = cache.get(cache_key)
            if cached is not None:
                return cached.model_copy(
                    update={
                        "doc_id": effective_doc_id,
                        "source_path": source_path_str,
                        "route_telemetry": {**cached.route_telemetry, "cache_hit": True},
                    }
                )

        # 2. Phase 1: Deterministic Extraction
        phase1: Phase1Output = self.parser_registry.parse(
            raw_bytes,
            effective_filename,
            options=opts,
        )

        if phase1.metadata.file_size == 0:
            metadata_dict = phase1.metadata.model_dump()
            metadata_dict["file_size"] = len(raw_bytes)
            metadata_dict["mime_type"] = mime_type
            phase1_metadata = DocumentMetadata(**metadata_dict)
        else:
            phase1_metadata = phase1.metadata

        # 3. Phase 2: Multimodal / Semi-Structured Enrichment via AuraRouter
        structured_data: Dict[str, Any] = {}
        unstructured_annotations: List[str] = []
        route_telemetry: Dict[str, Any] = {}
        final_visuals = list(phase1.visual_elements)
        final_raw_text = phase1.raw_text

        should_invoke_phase2 = opts.enable_phase2 and (
            bool(phase1.visual_elements)
            or bool(phase1.low_text_pages)
            or opts.template is not None
            or opts.custom_prompt is not None
            or opts.force_phase2
        )

        if should_invoke_phase2:
            raw_structured, batch_annotations, updated_visuals, route_telemetry = (
                self._execute_phase2_batches(
                    phase1=phase1,
                    opts=opts,
                    response_schema=response_schema,
                    guidelines=guidelines,
                )
            )
            unstructured_annotations.extend(batch_annotations)
            if updated_visuals:
                final_visuals = updated_visuals

            if raw_structured:
                validated_data, val_notes = validate_structured_output(
                    raw_structured,
                    schema=response_schema,
                    model_cls=pydantic_cls,
                )
                structured_data = canonicalize_extracted_entities(validated_data)
                unstructured_annotations.extend(val_notes)

            # Weave page OCR text back into raw_text when Phase 1 text was empty/scanned
            ocr_snippets = [
                (
                    f"--- Page {ve.page_number} (OCR) ---\n{ve.ocr_text}"
                    if ve.page_number is not None
                    else ve.ocr_text
                )
                for ve in final_visuals
                if ve.ocr_text
            ]
            if ocr_snippets and (not final_raw_text.strip() or phase1.low_text_pages):
                joined_ocr = "\n\n".join(ocr_snippets)
                final_raw_text = (
                    f"{final_raw_text}\n\n{joined_ocr}".strip()
                    if final_raw_text.strip()
                    else joined_ocr
                )
        else:
            route_telemetry = {
                "phase2_executed": False,
                "reason": "No visual elements, template, or force_phase2 requested",
            }

        # 4. Assemble Universal Lossless ScryerResult
        result = ScryerResult(
            doc_id=effective_doc_id,
            source_path=source_path_str,
            mime_type=mime_type,
            sha256=sha256_hash,
            metadata=phase1_metadata,
            raw_text=final_raw_text,
            structured_data=structured_data,
            visual_elements=final_visuals,
            unstructured_annotations=unstructured_annotations,
            route_telemetry=route_telemetry,
        )

        if cache is not None and cache_key is not None:
            cache.put(cache_key, result)

        return result


class AsyncScryerExtractor:
    """Async wrapper providing bounded-concurrency document and batch extraction."""

    def __init__(
        self,
        extractor: Optional[ScryerExtractor] = None,
        max_concurrency: int = 4,
    ) -> None:
        self.extractor = extractor or ScryerExtractor()
        self.max_concurrency = max(1, max_concurrency)

    async def extract_async(
        self,
        source: Union[str, Path, bytes],
        options: Optional[ScryerExtractionOptions] = None,
        filename: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> ScryerResult:
        """Execute extraction off the event loop cleanly."""
        return await asyncio.to_thread(
            self.extractor.extract,
            source,
            options=options,
            filename=filename,
            doc_id=doc_id,
        )

    async def extract_batch_async(
        self,
        sources: Sequence[Union[str, Path, bytes]],
        options: Optional[ScryerExtractionOptions] = None,
    ) -> List[ScryerResult]:
        """Extract multiple sources concurrently with bounded semaphore concurrency."""
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _worker(src: Union[str, Path, bytes]) -> ScryerResult:
            async with sem:
                return await self.extract_async(src, options=options)

        return await asyncio.gather(*(_worker(s) for s in sources))
