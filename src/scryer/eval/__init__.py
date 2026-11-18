"""scryer.eval package entrypoints."""

from __future__ import annotations

from scryer.eval.cache import BenchmarkCacheManager
from scryer.eval.contracts import (
    BenchmarkSample,
    DatasetType,
    EvaluationReport,
    MetricScore,
    SampleEvaluationResult,
)
from scryer.eval.metrics import (
    compute_character_error_rate,
    compute_exact_match,
    compute_levenshtein_distance,
    compute_token_f1,
    evaluate_extracted_entities,
)
from scryer.eval.local_loop import LocalPhiLoopOrchestrator
from scryer.eval.runner import BenchmarkRunner

__all__ = [
    "BenchmarkCacheManager",
    "BenchmarkRunner",
    "BenchmarkSample",
    "DatasetType",
    "EvaluationReport",
    "LocalPhiLoopOrchestrator",
    "MetricScore",
    "SampleEvaluationResult",
    "compute_character_error_rate",
    "compute_exact_match",
    "compute_levenshtein_distance",
    "compute_token_f1",
    "evaluate_extracted_entities",
]

