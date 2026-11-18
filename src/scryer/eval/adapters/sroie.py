"""sroie.py - SROIE (ICDAR 2019 Scanned Receipts OCR & Information Extraction) Adapter."""

from __future__ import annotations

import json
import logging
from typing import Generator, List, Optional
import httpx

from scryer.eval.adapters.base import BaseDatasetAdapter
from scryer.eval.contracts import BenchmarkSample, DatasetType

logger = logging.getLogger(__name__)

SROIE_REPO_RAW_BASE = "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data"
SROIE_REPO_API = "https://api.github.com/repos/zzzDavid/ICDAR-2019-SROIE/contents/data/key"


class SROIEAdapter(BaseDatasetAdapter):
    """Adapter for downloading and loading ICDAR 2019 SROIE receipt samples and ground truth."""

    @property
    def dataset_type(self) -> DatasetType:
        return DatasetType.SROIE

    def download_and_prepare(self, limit: Optional[int] = 10) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []
        for sample in self.iter_samples(limit=limit):
            samples.append(sample)
        return samples

    def iter_samples(self, limit: Optional[int] = 10) -> Generator[BenchmarkSample, None, None]:
        # 1. Discover sample IDs
        sample_ids: List[str] = []
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                resp = client.get(SROIE_REPO_API, headers={"User-Agent": "scryer-eval-harness"})
                if resp.status_code == 200:
                    entries = resp.json()
                    sample_ids = [
                        e["name"].replace(".json", "") for e in entries if e["name"].endswith(".json")
                    ]
        except Exception as exc:
            logger.warning("Failed to query GitHub API for SROIE samples: %s. Using numeric fallback.", exc)

        if not sample_ids:
            # Fallback to standard 3-digit identifiers (000, 001, ...)
            sample_ids = [f"{i:03d}" for i in range(100)]

        count = 0
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            for s_id in sample_ids:
                if limit and limit > 0 and count >= limit:
                    break

                img_rel = f"{s_id}.jpg"
                json_rel = f"{s_id}.json"

                cached_img = self.cache_mgr.get_cached_file(self.dataset_type.value, img_rel)
                cached_json = self.cache_mgr.get_cached_file(self.dataset_type.value, json_rel)

                if not cached_img:
                    img_url = f"{SROIE_REPO_RAW_BASE}/img/{s_id}.jpg"
                    resp = client.get(img_url)
                    if resp.status_code != 200:
                        continue
                    cached_img = self.cache_mgr.store_file(
                        self.dataset_type.value, img_rel, resp.content
                    )

                if not cached_json:
                    json_url = f"{SROIE_REPO_RAW_BASE}/key/{s_id}.json"
                    resp = client.get(json_url)
                    if resp.status_code != 200:
                        continue
                    cached_json = self.cache_mgr.store_file(
                        self.dataset_type.value, json_rel, resp.content
                    )

                try:
                    gt_data = json.loads(cached_json.read_text(encoding="utf-8"))
                except Exception:
                    continue

                yield BenchmarkSample(
                    sample_id=s_id,
                    dataset=self.dataset_type,
                    file_path=cached_img,
                    mime_type="image/jpeg",
                    ground_truth_entities=gt_data,
                    metadata={"source_url": f"{SROIE_REPO_RAW_BASE}/img/{s_id}.jpg"},
                )
                count += 1
