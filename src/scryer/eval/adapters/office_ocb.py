"""office_ocb.py - Microsoft Office Comprehension Benchmark & Table Extraction Adapter."""

from __future__ import annotations

import logging
from typing import Generator, List, Optional
import httpx

from scryer.eval.adapters.base import BaseDatasetAdapter
from scryer.eval.contracts import BenchmarkSample, DatasetType

logger = logging.getLogger(__name__)

HF_OCB_API = "https://huggingface.co/api/datasets/microsoft/OfficeComprehensionBenchmark"
HF_OCB_RESOLVE_BASE = "https://huggingface.co/datasets/microsoft/OfficeComprehensionBenchmark/resolve/main"


class OfficeOCBAdapter(BaseDatasetAdapter):
    """Adapter for downloading and evaluating native Office (.xlsx, .docx, .pptx) files."""

    @property
    def dataset_type(self) -> DatasetType:
        return DatasetType.OFFICE_OCB

    def download_and_prepare(self, limit: Optional[int] = 10) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []
        for sample in self.iter_samples(limit=limit):
            samples.append(sample)
        return samples

    def iter_samples(self, limit: Optional[int] = 10) -> Generator[BenchmarkSample, None, None]:
        # 1. Discover reference files from HF repository
        files: List[str] = []
        headers = {"User-Agent": "scryer-eval-harness/0.2.0"}

        try:
            with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
                resp = client.get(HF_OCB_API)
                if resp.status_code == 200:
                    siblings = resp.json().get("siblings", [])
                    files = [
                        s["rfilename"]
                        for s in siblings
                        if s["rfilename"].startswith("reference_files/")
                        and any(s["rfilename"].endswith(ext) for ext in (".xlsx", ".docx", ".pptx", ".csv"))
                    ]
        except Exception as exc:
            logger.warning("Failed to query HF API for OCB files: %s. Using default file list.", exc)

        if not files:
            files = [
                "reference_files/15.xlsx",
                "reference_files/121-small.xlsx",
                "reference_files/232.xlsx",
                "reference_files/368.xlsx",
            ]

        count = 0
        with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
            for rfile in files:
                if limit and limit > 0 and count >= limit:
                    break

                filename = rfile.split("/")[-1]
                cached_file = self.cache_mgr.get_cached_file(self.dataset_type.value, filename)

                if not cached_file:
                    url = f"{HF_OCB_RESOLVE_BASE}/{rfile}"
                    try:
                        resp = client.get(url)
                        if resp.status_code != 200:
                            continue
                        cached_file = self.cache_mgr.store_file(
                            self.dataset_type.value, filename, resp.content
                        )
                    except Exception as exc:
                        logger.warning("Failed to download %s: %s", rfile, exc)
                        continue

                # Determine expected MIME type
                mime = "application/octet-stream"
                if filename.endswith(".xlsx"):
                    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif filename.endswith(".docx"):
                    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif filename.endswith(".pptx"):
                    mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                elif filename.endswith(".csv"):
                    mime = "text/csv"

                yield BenchmarkSample(
                    sample_id=filename,
                    dataset=self.dataset_type,
                    file_path=cached_file,
                    mime_type=mime,
                    ground_truth_entities={"filename": filename},
                    metadata={"original_reference": rfile},
                )
                count += 1

