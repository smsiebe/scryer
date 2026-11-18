"""cord.py - CORD (Consolidated Receipt Dataset for Post-OCR Parsing) Adapter."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Generator, List, Optional
import httpx

from scryer.eval.adapters.base import BaseDatasetAdapter
from scryer.eval.contracts import BenchmarkSample, DatasetType

logger = logging.getLogger(__name__)

HF_CORD_ROWS_API = (
    "https://datasets-server.huggingface.co/rows?dataset=naver-clova-ix%2Fcord-v2&config=default&split=test"
)


class CORDAdapter(BaseDatasetAdapter):
    """Adapter for downloading and loading CORD receipt test samples and ground truth parse trees."""

    @property
    def dataset_type(self) -> DatasetType:
        return DatasetType.CORD

    def download_and_prepare(self, limit: Optional[int] = 10) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []
        for sample in self.iter_samples(limit=limit):
            samples.append(sample)
        return samples

    def iter_samples(self, limit: Optional[int] = 10) -> Generator[BenchmarkSample, None, None]:
        # 1. First check if we already have cached samples locally
        dataset_dir = self.cache_mgr.get_dataset_dir(self.dataset_type.value)
        cached_imgs = sorted(dataset_dir.glob("cord_test_*.png"))
        if cached_imgs:
            count = 0
            for img_path in cached_imgs:
                if limit and limit > 0 and count >= limit:
                    break
                s_id = img_path.stem
                json_path = img_path.with_suffix(".json")
                norm_entities: Dict[str, Any] = {}
                if json_path.exists():
                    try:
                        gt_parsed = json.loads(json_path.read_text(encoding="utf-8"))
                        gt_tree = gt_parsed.get("gt_parse", {})
                        if "total" in gt_tree and isinstance(gt_tree["total"], dict):
                            norm_entities["total"] = gt_tree["total"].get("total_price")
                        if "sub_total" in gt_tree and isinstance(gt_tree["sub_total"], dict):
                            norm_entities["subtotal"] = gt_tree["sub_total"].get("subtotal_price")
                            norm_entities["tax"] = gt_tree["sub_total"].get("tax_price")
                        if "menu" in gt_tree:
                            norm_entities["items"] = gt_tree["menu"]
                    except Exception:
                        pass

                yield BenchmarkSample(
                    sample_id=s_id,
                    dataset=self.dataset_type,
                    file_path=img_path,
                    mime_type="image/png",
                    ground_truth_entities=norm_entities,
                )
                count += 1
            if count > 0:
                return

        # 2. Otherwise download from Hugging Face datasets-server
        batch_size = 20
        offset = 0
        count = 0
        headers = {"User-Agent": "scryer-eval-harness/0.2.0"}

        with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
            while True:
                if limit and limit > 0 and count >= limit:
                    break

                req_limit = min(batch_size, limit - count) if limit and limit > 0 else batch_size
                url = f"{HF_CORD_ROWS_API}&offset={offset}&limit={req_limit}"

                try:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        logger.warning("CORD HF API returned status %s: %s", resp.status_code, resp.text[:200])
                        break
                    data = resp.json()
                    rows = data.get("rows", [])
                    if not rows:
                        break
                except Exception as exc:
                    logger.error("Failed to fetch CORD rows from Hugging Face: %s", exc)
                    break

                for item in rows:
                    if limit and limit > 0 and count >= limit:
                        break

                    row_idx = item.get("row_idx", offset)
                    row = item.get("row", {})
                    img_info = row.get("image", {})
                    gt_raw = row.get("ground_truth", "{}")

                    s_id = f"cord_test_{row_idx:04d}"
                    img_rel = f"{s_id}.png"
                    json_rel = f"{s_id}.json"

                    cached_img = self.cache_mgr.get_cached_file(self.dataset_type.value, img_rel)
                    cached_json = self.cache_mgr.get_cached_file(self.dataset_type.value, json_rel)

                    if not cached_img:
                        img_src = img_info.get("src")
                        if not img_src:
                            continue
                        img_resp = client.get(img_src)
                        if img_resp.status_code != 200:
                            continue
                        cached_img = self.cache_mgr.store_file(
                            self.dataset_type.value, img_rel, img_resp.content
                        )

                    if not cached_json:
                        cached_json = self.cache_mgr.store_file(
                            self.dataset_type.value, json_rel, gt_raw.encode("utf-8")
                        )

                    try:
                        gt_parsed = json.loads(cached_json.read_text(encoding="utf-8"))
                        gt_tree = gt_parsed.get("gt_parse", {})
                        # Normalize key receipt entities from CORD tree
                        norm_entities: Dict[str, Any] = {}
                        if "total" in gt_tree and isinstance(gt_tree["total"], dict):
                            norm_entities["total"] = gt_tree["total"].get("total_price")
                        if "sub_total" in gt_tree and isinstance(gt_tree["sub_total"], dict):
                            norm_entities["subtotal"] = gt_tree["sub_total"].get("subtotal_price")
                            norm_entities["tax"] = gt_tree["sub_total"].get("tax_price")
                        if "menu" in gt_tree:
                            norm_entities["items"] = gt_tree["menu"]
                    except Exception:
                        norm_entities = {}

                    yield BenchmarkSample(
                        sample_id=s_id,
                        dataset=self.dataset_type,
                        file_path=cached_img,
                        mime_type="image/png",
                        ground_truth_entities=norm_entities,
                        metadata={"raw_ground_truth": gt_raw},
                    )
                    count += 1

                offset += len(rows)

