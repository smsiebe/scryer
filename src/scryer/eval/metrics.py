"""metrics.py - Evaluation metric algorithms for text, entity, and table extraction."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


def normalize_field_value(val: Any) -> str:
    """Normalize a field value for resilient entity comparisons."""
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        return str(val)
    text = str(val).strip().lower()
    # If it's a numeric string with trailing decimal zeroes (e.g. '100.00' or '9.0'), strip them
    num_match = re.match(r"^(\d+)\.0+$", text)
    if num_match:
        return num_match.group(1)
    # Normalize internal whitespaces and punctuation
    text = re.sub(r"[\s\-_,.]+", " ", text).strip()
    return text


def compute_exact_match(predicted: Any, target: Any) -> float:
    """Binary exact match score (1.0 or 0.0) after value normalization."""
    p_norm = normalize_field_value(predicted)
    t_norm = normalize_field_value(target)
    if not p_norm and not t_norm:
        return 1.0
    if p_norm == t_norm:
        return 1.0
    # Date prefix match: if target date is contained at start of predicted timestamp
    if t_norm and p_norm.startswith(t_norm):
        return 1.0
    # Numerical equivalence: e.g. 60.31 vs 60.30 within 0.02
    try:
        if abs(float(p_norm) - float(t_norm)) <= 0.02:
            return 1.0
    except (ValueError, TypeError):
        pass
    return 0.0


def compute_token_f1(predicted: Any, target: Any) -> float:
    """Compute token-level precision, recall, and F1 score with token multiset counts."""
    p_norm = normalize_field_value(predicted)
    t_norm = normalize_field_value(target)

    if not p_norm and not t_norm:
        return 1.0
    if not p_norm or not t_norm:
        return 0.0
    if p_norm == t_norm:
        return 1.0

    from collections import Counter

    p_counts = Counter(p_norm.split())
    t_counts = Counter(t_norm.split())

    intersection = sum((p_counts & t_counts).values())
    if intersection == 0:
        return 0.0

    precision = intersection / sum(p_counts.values())
    recall = intersection / sum(t_counts.values())
    return (2 * precision * recall) / (precision + recall)


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance."""
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_character_error_rate(predicted: str, target: str) -> float:
    """Normalized character error rate (CER) between 0.0 and 1.0+."""
    if not target and not predicted:
        return 0.0
    if not target:
        return 1.0
    dist = compute_levenshtein_distance(predicted.strip(), target.strip())
    return min(1.0, dist / max(1, len(target.strip())))


_KEY_ALIASES: Dict[str, Set[str]] = {
    "company": {"vendor", "vendor_name", "merchant", "store", "company", "issuer"},
    "vendor": {"vendor", "vendor_name", "company", "merchant", "store"},
    "total": {"total", "total_amount", "amount_due", "grand_total", "balance_due"},
    "subtotal": {"subtotal", "sub_total", "net_amount"},
    "tax": {"tax", "tax_amount", "vat", "gst"},
    "date": {"date", "invoice_date", "receipt_date", "transaction_date"},
    "address": {"address", "vendor_address", "store_address", "location"},
}


def evaluate_extracted_entities(
    extracted: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """Evaluate predicted dictionary against ground truth dictionary.

    Returns:
        (mean_exact_match, mean_f1, per_field_scores)
    """
    if not ground_truth:
        return 1.0, 1.0, {}

    em_scores: List[float] = []
    f1_scores: List[float] = []
    field_breakdown: Dict[str, Dict[str, float]] = {}

    for key, gt_val in ground_truth.items():
        # Match case-insensitively or with common key aliases
        pred_val = None
        key_lower = key.lower().replace("_", "")
        aliases = _KEY_ALIASES.get(key.lower(), {key.lower()})

        for k, v in extracted.items():
            k_lower = k.lower()
            if k_lower in aliases or k_lower.replace("_", "") == key_lower:
                pred_val = v
                break

        em = compute_exact_match(pred_val, gt_val)
        f1 = compute_token_f1(pred_val, gt_val)
        em_scores.append(em)
        f1_scores.append(f1)
        field_breakdown[key] = {"exact_match": em, "f1": f1}

    mean_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return mean_em, mean_f1, field_breakdown
