"""base.py - Base contract for benchmark dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, List, Optional
from scryer.eval.cache import BenchmarkCacheManager
from scryer.eval.contracts import BenchmarkSample, DatasetType


class BaseDatasetAdapter(ABC):
    """Abstract base class for benchmark dataset fetchers and iterators."""

    def __init__(self, cache_mgr: Optional[BenchmarkCacheManager] = None):
        self.cache_mgr = cache_mgr or BenchmarkCacheManager()

    @property
    @abstractmethod
    def dataset_type(self) -> DatasetType:
        """The dataset enumeration type."""
        ...

    @abstractmethod
    def download_and_prepare(self, limit: Optional[int] = None) -> List[BenchmarkSample]:
        """Download or fetch up to limit samples and return parsed BenchmarkSample list.

        If limit is None or <= 0, all available samples should be prepared.
        """
        ...

    @abstractmethod
    def iter_samples(self, limit: Optional[int] = None) -> Generator[BenchmarkSample, None, None]:
        """Iterate lazily through downloaded benchmark samples."""
        ...

