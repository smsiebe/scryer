"""contracts.py - Data transfer objects and contracts for the Scryer evaluation harness."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetType(str, Enum):
    SROIE = "sroie"
    CORD = "cord"
    OFFICE_OCB = "office_ocb"
    CUSTOM = "custom"


class BenchmarkSample(BaseModel):
    """Represents a single evaluated document or image with ground-truth labels."""

    sample_id: str = Field(description="Unique sample identifier or filename")
    dataset: DatasetType = Field(description="Origin dataset")
    file_path: Path = Field(description="Local path to the downloaded document/image")
    mime_type: Optional[str] = Field(default=None, description="Expected MIME type")
    ground_truth_entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value ground truth entities (e.g. company, date, total)",
    )
    ground_truth_text: Optional[str] = Field(
        default=None, description="Ground truth full text if available"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional dataset-specific metadata"
    )


class MetricScore(BaseModel):
    """Quantitative scoring metric for extraction evaluation."""

    metric_name: str
    score: float = Field(ge=0.0, le=1.0, description="Normalized score between 0.0 and 1.0")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Breakdown of per-field or per-entity performance"
    )


class SampleEvaluationResult(BaseModel):
    """Evaluation result for an individual benchmark sample."""

    sample_id: str
    dataset: DatasetType
    latency_ms: float
    field_exact_match: float = 0.0
    field_f1_score: float = 0.0
    char_error_rate: Optional[float] = None
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    ground_truth_data: Dict[str, Any] = Field(default_factory=dict)
    field_scores: Dict[str, MetricScore] = Field(default_factory=dict)
    error: Optional[str] = None


class EvaluationReport(BaseModel):
    """Aggregate benchmark report produced by BenchmarkRunner."""

    dataset: DatasetType
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_samples: int
    successful_samples: int
    failed_samples: int
    mean_latency_ms: float
    mean_field_exact_match: float
    mean_field_f1: float
    mean_char_error_rate: Optional[float] = None
    per_field_accuracy: Dict[str, float] = Field(default_factory=dict)
    samples: List[SampleEvaluationResult] = Field(default_factory=list)
