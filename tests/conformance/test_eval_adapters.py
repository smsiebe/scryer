"""test_eval_adapters.py - Conformance tests for downloading and iterating dataset adapters."""

from __future__ import annotations

from pathlib import Path
import pytest

from scryer.eval.adapters.office_ocb import OfficeOCBAdapter
from scryer.eval.adapters.sroie import SROIEAdapter
from scryer.eval.cache import BenchmarkCacheManager


@pytest.mark.conformance
def test_sroie_adapter_download_and_cache(tmp_path: Path):
    cache = BenchmarkCacheManager(cache_dir=tmp_path)
    adapter = SROIEAdapter(cache_mgr=cache)

    samples = adapter.download_and_prepare(limit=2)
    assert len(samples) == 2
    for s in samples:
        assert s.file_path.exists()
        assert s.file_path.stat().st_size > 0
        assert s.mime_type == "image/jpeg"
        assert len(s.ground_truth_entities) > 0


@pytest.mark.conformance
def test_office_ocb_adapter_download_and_cache(tmp_path: Path):
    cache = BenchmarkCacheManager(cache_dir=tmp_path)
    adapter = OfficeOCBAdapter(cache_mgr=cache)

    samples = adapter.download_and_prepare(limit=2)
    assert len(samples) >= 1
    for s in samples:
        assert s.file_path.exists()
        assert s.file_path.stat().st_size > 0
