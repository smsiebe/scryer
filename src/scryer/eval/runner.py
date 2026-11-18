"""runner.py - Orchestrates benchmark dataset execution and aggregates metrics."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from scryer.contracts import ScryerExtractionOptions, TriMode
from scryer.eval.adapters.base import BaseDatasetAdapter
from scryer.eval.adapters.cord import CORDAdapter
from scryer.eval.adapters.office_ocb import OfficeOCBAdapter
from scryer.eval.adapters.sroie import SROIEAdapter
from scryer.eval.cache import BenchmarkCacheManager
from scryer.eval.contracts import (
    DatasetType,
    EvaluationReport,
    MetricScore,
    SampleEvaluationResult,
)
from scryer.eval.metrics import (
    compute_character_error_rate,
    evaluate_extracted_entities,
)
from scryer.extractor import ScryerExtractor

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Executes extraction on benchmark samples and compiles evaluation reports."""

    def __init__(
        self,
        extractor: Optional[ScryerExtractor] = None,
        cache_mgr: Optional[BenchmarkCacheManager] = None,
    ):
        self.extractor = extractor or ScryerExtractor()
        self.cache_mgr = cache_mgr or BenchmarkCacheManager()

    def get_adapter(self, dataset: DatasetType | str) -> BaseDatasetAdapter:
        ds_type = DatasetType(dataset) if isinstance(dataset, str) else dataset
        if ds_type == DatasetType.SROIE:
            return SROIEAdapter(self.cache_mgr)
        elif ds_type == DatasetType.CORD:
            return CORDAdapter(self.cache_mgr)
        elif ds_type == DatasetType.OFFICE_OCB:
            return OfficeOCBAdapter(self.cache_mgr)
        else:
            raise ValueError(f"Unsupported benchmark dataset: {dataset}")

    def run_benchmark(
        self,
        dataset: DatasetType | str,
        limit: Optional[int] = 10,
        options: Optional[ScryerExtractionOptions] = None,
    ) -> EvaluationReport:
        adapter = self.get_adapter(dataset)
        ds_type = adapter.dataset_type

        # Default to invoice preset for receipts if not explicitly provided
        if options is None:
            template_name = "invoice" if ds_type in (DatasetType.SROIE, DatasetType.CORD) else None
            options = ScryerExtractionOptions(
                template=template_name,
                tri_mode=TriMode.PYTHON,
                enable_phase2=False,  # default to deterministic/OCR unless caller requests Phase 2
            )

        sample_results: List[SampleEvaluationResult] = []
        total_latency = 0.0

        for sample in adapter.iter_samples(limit=limit):
            t0 = time.perf_counter()
            try:
                res = self.extractor.extract(sample.file_path, options=options)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                total_latency += latency_ms

                # Compare structured_data or entities with ground truth
                extracted_data: Dict[str, Any] = dict(res.structured_data or {})
                extracted_data["filename"] = sample.sample_id
                if res.metadata:
                    if getattr(res.metadata, "sheet_names", None):
                        extracted_data.setdefault("sheet_names", res.metadata.sheet_names)
                    if getattr(res.metadata, "page_count", None):
                        extracted_data.setdefault("page_count", res.metadata.page_count)
                    if getattr(res.metadata, "custom_fields", None):
                        for k, v in res.metadata.custom_fields.items():
                            extracted_data.setdefault(k, v)
                if getattr(res, "tables", None):
                    extracted_data["table_count"] = len(res.tables)

                # Also include metadata / text annotations
                for an in res.unstructured_annotations:
                    extracted_data.setdefault("notes", []).append(an)

                em, f1, breakdown = evaluate_extracted_entities(
                    extracted_data, sample.ground_truth_entities
                )

                field_scores_map: Dict[str, Any] = {}
                for f_name, f_vals in breakdown.items():
                    field_scores_map[f_name] = MetricScore(
                        metric_name=f_name,
                        score=f_vals.get("f1", 0.0),
                        details=f_vals,
                    )

                cer = None
                if sample.ground_truth_text:
                    cer = compute_character_error_rate(res.raw_text, sample.ground_truth_text)

                sample_results.append(
                    SampleEvaluationResult(
                        sample_id=sample.sample_id,
                        dataset=ds_type,
                        latency_ms=latency_ms,
                        field_exact_match=em,
                        field_f1_score=f1,
                        char_error_rate=cer,
                        extracted_data=extracted_data,
                        ground_truth_data=sample.ground_truth_entities,
                        field_scores=field_scores_map,
                    )
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                sample_results.append(
                    SampleEvaluationResult(
                        sample_id=sample.sample_id,
                        dataset=ds_type,
                        latency_ms=latency_ms,
                        error=str(exc),
                    )
                )

        successful = [s for s in sample_results if s.error is None]
        mean_em = (
            sum(s.field_exact_match for s in successful) / len(successful) if successful else 0.0
        )
        mean_f1 = sum(s.field_f1_score for s in successful) / len(successful) if successful else 0.0
        mean_lat = total_latency / max(1, len(sample_results))

        cer_scores = [s.char_error_rate for s in successful if s.char_error_rate is not None]
        mean_cer = sum(cer_scores) / len(cer_scores) if cer_scores else None

        return EvaluationReport(
            dataset=ds_type,
            total_samples=len(sample_results),
            successful_samples=len(successful),
            failed_samples=len(sample_results) - len(successful),
            mean_latency_ms=mean_lat,
            mean_field_exact_match=mean_em,
            mean_field_f1=mean_f1,
            mean_char_error_rate=mean_cer,
            samples=sample_results,
        )
