"""cache.py - Local disk cache manager for downloaded benchmark datasets."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_CACHE_DIR = Path.home() / ".cache" / "scryer" / "benchmarks"


class BenchmarkCacheManager:
    """Manages downloading, storing, and hashing benchmark files locally."""

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            env_dir = os.environ.get("SCRYER_BENCHMARK_CACHE")
            self.cache_dir = Path(env_dir) if env_dir else DEFAULT_BENCHMARK_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_dataset_dir(self, dataset_name: str) -> Path:
        """Get or create the root directory for a specific dataset."""
        d = self.cache_dir / dataset_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_cached_file(self, dataset_name: str, relative_path: str) -> Optional[Path]:
        """Check if a file exists in the cache."""
        target = self.get_dataset_dir(dataset_name) / relative_path
        return target if target.exists() and target.stat().st_size > 0 else None

    def store_file(
        self, dataset_name: str, relative_path: str, data: bytes, expected_sha256: Optional[str] = None
    ) -> Path:
        """Store bytes to the cache and optionally verify SHA-256."""
        if expected_sha256:
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha.lower() != expected_sha256.lower():
                raise ValueError(
                    f"Integrity check failed for {relative_path}: expected {expected_sha256}, got {actual_sha}"
                )

        target = self.get_dataset_dir(dataset_name) / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target
