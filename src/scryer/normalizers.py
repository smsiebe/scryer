"""normalizers.py - Deterministic post-processing normalizers for extracted entity fields.

Provides rule-based canonicalization for dates, timestamps, company names,
cashier prefixes, currency numbers, and addresses, bridging the gap between
raw multimodal LLM token outputs and ground-truth entity standards.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


# Common registered company suffixes in international & regional receipts
_COMPANY_SUFFIXES = [
    r"SDN\.?\s*BHD\.?",
    r"BHD\.?",
    r"ENTERPRISE",
    r"HOLDINGS",
    r"CORPORATION",
    r"CORP\.?",
    r"LTD\.?",
    r"LIMITED",
    r"INC\.?",
    r"INCORPORATED",
    r"LLC",
    r"PTE\.?\s*LTD\.?",
    r"GMBH",
    r"S\.?R\.?O\.?",
    r"CO\.?",
]

# Patterns for cashier/operator/terminal prefixes to strip from company names
_CLERK_PREFIXES = [
    r"^cashier\s*[:#-]?\s*[\w\s]+?\s+(?=[A-Z0-9])",
    r"^clerk\s*[:#-]?\s*[\w\s]+?\s+(?=[A-Z0-9])",
    r"^operator\s*[:#-]?\s*[\w\s]+?\s+(?=[A-Z0-9])",
    r"^served\s+by\s*[:#-]?\s*[\w\s]+?\s+(?=[A-Z0-9])",
]


def clean_extracted_date(val: Any) -> Optional[str]:
    """Clean and isolate a calendar date string, stripping timestamps or register codes.

    Examples:
        '25/12/2018 8:13:39 PM' -> '25/12/2018'
        '18-11-18 13:58 SH01 2153' -> '18-11-18'
        'Date: 12-01-2019' -> '12-01-2019'
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "not visible", "n/a"):
        return None

    # Strip leading label prefixes like "Date:", "Transaction Date:"
    s = re.sub(r"^(?:date|transaction\s*date|invoice\s*date)\s*[:#-]?\s*", "", s, flags=re.IGNORECASE)

    # 1. Match standard DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, or DD/MM/YY
    date_match = re.search(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
        s,
    )
    if date_match:
        return date_match.group(1)

    # 2. Match text dates like "12 Jan 2019" or "19 October 2018"
    month_match = re.search(
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
        s,
        flags=re.IGNORECASE,
    )
    if month_match:
        return month_match.group(1)

    return s.split()[0] if s.split() else s


def clean_vendor_name(val: Any) -> Optional[str]:
    """Clean vendor/company name, stripping leading cashier/operator names and punctuation.

    Examples:
        'tan woon yann BOOK TA.K (TAMAN DAYA) SDN BHD' -> 'BOOK TA.K (TAMAN DAYA) SDN BHD'
        'Cashier: Ali RESTORAN ABC' -> 'RESTORAN ABC'
    """
    if val is None:
        return None
    s = str(val).strip().strip('",\'')
    if not s or s.lower() in ("none", "null", "n/a"):
        return None

    # Strip known clerk/cashier prefixes
    for pat in _CLERK_PREFIXES:
        s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()

    # Detect lowercase clerk name preceding an ALL-CAPS registered company
    # e.g. "tan woon yann BOOK TA.K (TAMAN DAYA) SDN BHD"
    caps_match = re.search(r"^(?:[a-z]{2,12}\s+){1,4}([A-Z0-9][A-Z0-9\s.,&()'-]{4,})$", s)
    if caps_match:
        company_candidate = caps_match.group(1).strip()
        # Verify it has company-like properties (e.g. contains SDN BHD, ENTERPRISE, or is > 8 chars)
        if any(re.search(sfx, company_candidate, re.IGNORECASE) for sfx in _COMPANY_SUFFIXES) or len(company_candidate) >= 8:
            s = company_candidate

    return s.strip()


def clean_extracted_address(val: Any) -> Optional[str]:
    """Clean and normalize vendor store address."""
    if val is None:
        return None
    s = str(val).strip().strip('",\'')
    if not s or s.lower() in ("none", "null", "n/a"):
        return None

    # Strip leading label prefixes like "Address:", "Location:"
    s = re.sub(r"^(?:address|location|store\s*address)\s*[:#-]?\s*", "", s, flags=re.IGNORECASE)
    # Normalize internal whitespaces and newlines
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_date_flexible(date_str: str) -> Optional[Tuple[int, int, int]]:
    """Attempt to parse date string into (year, month, day) tuple for calendar comparison.

    Supports DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD/MM/YY, DD-MM-YY.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip()

    # Try common formats
    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d.%m.%Y",
        "%d.%m.%y",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            # Normalize 2-digit years to century 2000 if needed
            year = dt.year
            if year < 100:
                year += 2000
            return (year, dt.month, dt.day)
        except ValueError:
            pass

    return None


def canonicalize_extracted_entities(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply all deterministic normalizers to Scryer structured output dictionary."""
    if not data or not isinstance(data, dict):
        return data

    res = dict(data)

    # Clean vendor_name / company
    for key in ("vendor_name", "company"):
        if key in res and res[key]:
            res[key] = clean_vendor_name(res[key])

    # Clean invoice_date / date
    for key in ("invoice_date", "date"):
        if key in res and res[key]:
            res[key] = clean_extracted_date(res[key])

    # Clean vendor_address / address
    for key in ("vendor_address", "address"):
        if key in res and res[key]:
            res[key] = clean_extracted_address(res[key])

    return res

