"""test_eval_harness.py - Unit tests for evaluation harness, metrics, caching, and runner."""

from __future__ import annotations

from pathlib import Path
from scryer.contracts import ScryerResult
from scryer.eval.cache import BenchmarkCacheManager
from scryer.eval.contracts import BenchmarkSample, DatasetType
from scryer.eval.metrics import (
    compute_character_error_rate,
    compute_exact_match,
    compute_token_f1,
    evaluate_extracted_entities,
    normalize_field_value,
)
from scryer.eval.runner import BenchmarkRunner


def test_metric_field_normalization():
    assert normalize_field_value("  Acme  Corp., Ltd.  ") == "acme corp ltd"
    assert normalize_field_value(120.00) == "120"
    assert normalize_field_value(120.50) == "120.5"
    assert normalize_field_value(None) == ""


def test_metric_exact_match_and_f1():
    assert compute_exact_match("Acme Corp", "ACME CORP") == 1.0
    assert compute_exact_match("Acme Corp", "Beta Corp") == 0.0

    # Token F1
    f1 = compute_token_f1("Acme Global Solutions", "Acme Solutions Ltd")
    assert 0.6 < f1 < 0.9
    assert compute_token_f1("Total: 50.00", "Total: 50.00") == 1.0
    assert compute_token_f1("", "") == 1.0
    assert compute_token_f1("Acme", "") == 0.0


def test_metric_character_error_rate():
    assert compute_character_error_rate("hello", "hello") == 0.0
    assert compute_character_error_rate("helo", "hello") == 0.2
    assert compute_character_error_rate("", "") == 0.0
    assert compute_character_error_rate("abc", "") == 1.0


def test_evaluate_extracted_entities():
    extracted = {"company": "Acme Corp", "date": "2024-01-01", "total": "100.00"}
    ground_truth = {"company": "ACME CORP.", "date": "2024-01-01", "total": 100}

    em, f1, breakdown = evaluate_extracted_entities(extracted, ground_truth)
    assert em == 1.0
    assert f1 == 1.0
    assert "company" in breakdown
    assert breakdown["company"]["exact_match"] == 1.0


def test_benchmark_cache_manager(tmp_path: Path):
    cache = BenchmarkCacheManager(cache_dir=tmp_path)
    stored = cache.store_file("test_ds", "sample.txt", b"hello world")
    assert stored.exists()
    assert stored.read_text() == "hello world"

    retrieved = cache.get_cached_file("test_ds", "sample.txt")
    assert retrieved is not None
    assert retrieved == stored


def test_benchmark_runner_mock_adapter(tmp_path: Path):
    # Create a dummy sample
    sample_file = tmp_path / "dummy.txt"
    sample_file.write_text("Invoice from Stark Industries. Total: $500", encoding="utf-8")

    class _MockAdapter:
        dataset_type = DatasetType.CUSTOM

        def iter_samples(self, limit=None):
            yield BenchmarkSample(
                sample_id="sample_001",
                dataset=DatasetType.CUSTOM,
                file_path=sample_file,
                mime_type="text/plain",
                ground_truth_entities={"vendor": "Stark Industries", "total": "500"},
            )

    class _MockExtractor:
        def extract(self, path, options=None):
            return ScryerResult(
                doc_id="dummy.txt",
                mime_type="text/plain",
                raw_text="Invoice from Stark Industries. Total: $500",
                structured_data={"vendor": "Stark Industries", "total": "500"},
            )

    runner = BenchmarkRunner(extractor=_MockExtractor(), cache_mgr=BenchmarkCacheManager(tmp_path))
    # Override get_adapter for custom
    runner.get_adapter = lambda _: _MockAdapter()

    report = runner.run_benchmark(dataset=DatasetType.CUSTOM, limit=1)
    assert report.total_samples == 1
    assert report.successful_samples == 1
    assert report.failed_samples == 0
    assert report.mean_field_exact_match == 1.0
    assert report.mean_field_f1 == 1.0
