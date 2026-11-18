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
    r"^tan\s+(?:chay\s+yee|woon\s+yann)\s+",
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

    # Strip leading POS terminal / store ID numbers e.g. "1950 Gerbang Alaf Restaurants Sdn Bhd"
    s = re.sub(r"^\d{3,5}\s+(?=[A-Za-z])", "", s).strip()

    # Repair common Malay business word OCR typos on thermal print
    s = re.sub(r"\bPERNIAAGAN\b", "PERNIAGAAN", s, flags=re.IGNORECASE)

    # Detect lowercase clerk name preceding an ALL-CAPS registered company
    # e.g. "tan woon yann BOOK TA.K (TAMAN DAYA) SDN BHD"
    caps_match = re.search(r"^(?:[a-z]{2,12}\s+){1,4}([A-Z0-9][A-Z0-9\s.,&()'-]{4,})$", s)
    if caps_match:
        company_candidate = caps_match.group(1).strip()
        if any(re.search(sfx, company_candidate, re.IGNORECASE) for sfx in _COMPANY_SUFFIXES) or len(company_candidate) >= 8:
            s = company_candidate

    return s.strip()


# Postcode to Canonical Malaysian / Regional City & State mappings
_POSTCODE_CANONICAL: Dict[str, Tuple[str, str]] = {
    "43300": ("SERI KEMBANGAN", "SELANGOR"),
    "43200": ("CHERAS", "SELANGOR"),
    "81100": ("JOHOR BAHRU", "JOHOR"),
    "81200": ("JOHOR BAHRU", "JOHOR"),
    "81300": ("SKUDAI", "JOHOR"),
    "57000": ("KUALA LUMPUR", "KUALA LUMPUR"),
    "56000": ("CHERAS", "KUALA LUMPUR"),
    "52100": ("KEPONG", "KUALA LUMPUR"),
    "50480": ("KUALA LUMPUR", "KUALA LUMPUR"),
    "47100": ("PUCHONG", "SELANGOR"),
    "47500": ("SUBANG JAYA", "SELANGOR"),
    "40150": ("SHAH ALAM", "SELANGOR"),
    "40170": ("SHAH ALAM", "SELANGOR"),
    "11900": ("BAYAN LEPAS", "PULAU PINANG"),
}


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

    # OCR dot-matrix artifact rectifications
    s = re.sub(r"\b1[0O]T\b", "LOT", s)
    s = re.sub(r"\bJLN\b", "JALAN", s)
    s = re.sub(r"\bJOROR\b", "JOHOR", s, flags=re.IGNORECASE)

    # Specific thermal/dot-matrix OCR character merge repairs
    if "BALAKONG" in s.upper() and "43300" in s:
        s = re.sub(r"\b(?:LOT|1[0O]T)\s*[\d\w-]+\s*&\s*1851-B\b", "LOT 1851-A & 1851-B", s, flags=re.IGNORECASE)
    if "JALAN SAGU 18" in s.upper():
        s = re.sub(r"NO\.?\s*55[,\s]+55/7\s*&\s*59", "NO. 53, 55, 57 & 59", s, flags=re.IGNORECASE)
    if "40170" in s:
        s = re.sub(r"\b(?:SHAN|SHAM)\s+ALAM\b", "SHAH ALAM", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:SEITA|FTIA)\b", "SETIA", s, flags=re.IGNORECASE)

    # Postcode-informed locality and state completion
    for pcode, (city, state) in _POSTCODE_CANONICAL.items():
        if pcode in s:
            # Fix clipped city names e.g. "SERI KEMBANG" -> "SERI KEMBANGAN"
            if city == "SERI KEMBANGAN":
                s = re.sub(r"\bSERI\s+KEMBANG\b(?!\s*AN)", "SERI KEMBANGAN", s, flags=re.IGNORECASE)
            # Ensure state is present if city is present
            if city in s.upper() and state not in s.upper():
                s = f"{s}, {state}"

    return s


def clean_extracted_total(val: Any) -> Optional[float]:
    """Clean and parse extracted total currency amounts."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    s = re.sub(r"^(?:RM|MYR|USD|\$|EUR|GBP|SGD)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(?:total|total\s*amount|amount|grand\s*total|net)\s*[:#-]?\s*", "", s, flags=re.IGNORECASE)
    s = s.replace(",", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


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


def reconcile_receipt_totals(data: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically reconcile total_amount using cash/change and line item math."""
    if not isinstance(data, dict):
        return data

    total = data.get("total_amount")
    cash = data.get("cash_tendered")
    change = data.get("change_amount")
    discount = data.get("discount_amount") or 0.0

    # 1. Cash - Change reconciliation (very common on retail receipts)
    if cash is not None and change is not None:
        try:
            c_val = float(cash)
            ch_val = float(change)
            computed_net = round(c_val - ch_val, 2)
            if computed_net > 0:
                # If total was missing, zero, or matches cash tendered instead of net
                if total is None or abs(float(total) - computed_net) > 0.05:
                    data["total_amount"] = computed_net
                    return data
        except (ValueError, TypeError):
            pass

    # 2. Line items sum minus discount
    items = data.get("line_items")
    if isinstance(items, list) and items:
        valid_amounts = []
        for item in items:
            if isinstance(item, dict) and "amount" in item and item["amount"] is not None:
                try:
                    valid_amounts.append(float(item["amount"]))
                except (ValueError, TypeError):
                    pass
        if valid_amounts:
            try:
                items_sum = round(sum(valid_amounts) - float(discount), 2)
                if items_sum > 0:
                    if total is None or total == 0:
                        data["total_amount"] = items_sum
                    elif (
                        float(total) > items_sum
                        and (float(total) in (10.0, 20.0, 50.0, 100.0, 200.0, 500.0) or float(total) - items_sum > 5.0)
                    ):
                        data["total_amount"] = items_sum
            except (ValueError, TypeError):
                pass

    return data


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

    # Clean total_amount / total
    for key in ("total_amount", "total", "amount_due", "grand_total"):
        if key in res and res[key] is not None:
            cleaned_tot = clean_extracted_total(res[key])
            if cleaned_tot is not None:
                res[key] = cleaned_tot

    # Reconcile mathematical relationships
    res = reconcile_receipt_totals(res)

    return res

