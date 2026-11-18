"""adapters package initialization."""

from __future__ import annotations

from scryer.eval.adapters.base import BaseDatasetAdapter
from scryer.eval.adapters.cord import CORDAdapter
from scryer.eval.adapters.office_ocb import OfficeOCBAdapter
from scryer.eval.adapters.sroie import SROIEAdapter

__all__ = [
    "BaseDatasetAdapter",
    "CORDAdapter",
    "OfficeOCBAdapter",
    "SROIEAdapter",
]

