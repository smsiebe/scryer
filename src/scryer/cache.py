"""cache.py - Content-addressed disk/SQLite cache for Scryer extraction results."""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scryer.contracts import ScryerResult, compute_canonical_hash, safe_json_loads
from scryer.protocols import ExtractionCacheProtocol

logger = logging.getLogger("Scryer.Cache")


def build_cache_key(
    doc_sha256: str,
    schema: Optional[dict],
    options_payload: dict,
) -> str:
    """Build deterministic composite cache key from content hash, schema, and options."""
    return compute_canonical_hash(
        {
            "doc_sha256": doc_sha256,
            "schema_hash": compute_canonical_hash(schema or {}),
            "options_hash": compute_canonical_hash(options_payload),
        }
    )


class DiskExtractionCache(ExtractionCacheProtocol):
    """Thread-safe SQLite-backed content-addressed cache for ScryerResult instances."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "scryer_cache.db"
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS extraction_cache (
                    cache_key TEXT PRIMARY KEY,
                    doc_sha256 TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_sha ON extraction_cache(doc_sha256)"
            )
            conn.commit()

    def get(self, cache_key: str) -> Optional[ScryerResult]:
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT result_json FROM extraction_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
            if not row:
                return None
            payload = safe_json_loads(row[0])
            return ScryerResult.model_validate(payload)
        except Exception as e:
            logger.debug("Cache read miss/error for %s: %s", cache_key, e)
            return None

    def put(self, cache_key: str, result: ScryerResult) -> None:
        try:
            now_utc = datetime.now(timezone.utc).isoformat()
            result_json = result.to_json()
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO extraction_cache (cache_key, doc_sha256, result_json, created_at_utc)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        result_json = excluded.result_json,
                        created_at_utc = excluded.created_at_utc
                    """,
                    (cache_key, result.sha256, result_json, now_utc),
                )
                conn.commit()
        except Exception as e:
            logger.warning("Failed to persist cache entry %s: %s", cache_key, e)
