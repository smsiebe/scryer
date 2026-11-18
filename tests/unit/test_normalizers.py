"""test_normalizers.py - Unit tests for deterministic entity normalizers."""

from __future__ import annotations

from scryer.normalizers import (
    canonicalize_extracted_entities,
    clean_extracted_address,
    clean_extracted_date,
    clean_extracted_total,
    clean_vendor_name,
    parse_date_flexible,
    reconcile_receipt_totals,
)


def test_clean_extracted_date():
    assert clean_extracted_date("25/12/2018 8:13:39 PM") == "25/12/2018"
    assert clean_extracted_date("18-11-18 13:58 SH01 2153") == "18-11-18"
    assert clean_extracted_date("Date: 12-01-2019") == "12-01-2019"
    assert clean_extracted_date("Transaction Date: 19/10/2018") == "19/10/2018"
    assert clean_extracted_date(None) is None
    assert clean_extracted_date("Not visible") is None


def test_clean_vendor_name():
    assert clean_vendor_name("tan woon yann BOOK TA.K (TAMAN DAYA) SDN BHD") == "BOOK TA.K (TAMAN DAYA) SDN BHD"
    assert clean_vendor_name("Cashier: 01 MR D.I.Y. (JOHOR) SDN BHD") == "MR D.I.Y. (JOHOR) SDN BHD"
    assert clean_vendor_name("Clerk: Sarah INDAH GIFT & HOME DECO") == "INDAH GIFT & HOME DECO"
    assert clean_vendor_name("YONGFATT ENTERPRISE") == "YONGFATT ENTERPRISE"
    assert clean_vendor_name(None) is None


def test_clean_extracted_address():
    assert clean_extracted_address("Address: 27, JALAN DEDAP 13\n81100 JOHOR") == "27, JALAN DEDAP 13 81100 JOHOR"
    assert clean_extracted_address("Store Address: NO 122.124. JALAN DEDAP 13") == "NO 122.124. JALAN DEDAP 13"
    assert clean_extracted_address(None) is None


def test_parse_date_flexible():
    assert parse_date_flexible("25/12/2018") == (2018, 12, 25)
    assert parse_date_flexible("12-01-19") == (2019, 1, 12)
    assert parse_date_flexible("2019-01-12") == (2019, 1, 12)
    assert parse_date_flexible("invalid") is None


def test_canonicalize_extracted_entities():
    raw = {
        "vendor_name": "tan woon yann BOOK TA.K (TAMAN DAYA) SDN BHD",
        "invoice_date": "25/12/2018 8:13:39 PM",
        "vendor_address": "Address: NO. 55 JALAN SAGU",
        "total_amount": 9.0,
    }
    cleaned = canonicalize_extracted_entities(raw)
    assert cleaned["vendor_name"] == "BOOK TA.K (TAMAN DAYA) SDN BHD"
    assert cleaned["invoice_date"] == "25/12/2018"
    assert cleaned["vendor_address"] == "NO. 55 JALAN SAGU"
    assert cleaned["total_amount"] == 9.0


def test_clean_extracted_total():
    assert clean_extracted_total("RM 60.30") == 60.30
    assert clean_extracted_total("$1,234.50") == 1234.50
    assert clean_extracted_total("Total: 80.91") == 80.91
    assert clean_extracted_total(None) is None


def test_reconcile_receipt_totals():
    # 1. Cash minus change reconciliation
    data1 = {
        "total_amount": 70.30,  # Hallucinated or confused with change
        "cash_tendered": 70.00,
        "change_amount": 9.70,
    }
    res1 = reconcile_receipt_totals(data1)
    assert res1["total_amount"] == 60.30

    # 2. Line items sum vs cash tendered bill (e.g. 100.0)
    data2 = {
        "total_amount": 100.0,
        "line_items": [
            {"description": "Item 1", "amount": 80.91}
        ],
    }
    res2 = reconcile_receipt_totals(data2)
    assert res2["total_amount"] == 80.91


