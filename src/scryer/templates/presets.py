"""presets.py - Ready-to-use built-in extraction templates and Pydantic schema models."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field
from scryer.contracts import ExtractionTemplate


class DocumentSummaryModel(BaseModel):
    """General document summary and key topic extraction model."""
    summary: str = Field(description="Comprehensive summary of the document")
    topics: List[str] = Field(default_factory=list, description="Key topics discussed")
    organizations: List[str] = Field(default_factory=list, description="Organizations mentioned")
    action_items: List[str] = Field(default_factory=list, description="Explicit action items or tasks")


class InvoiceLineItem(BaseModel):
    description: str = Field(description="Description of the product or service")
    quantity: Optional[float] = Field(default=1.0, description="Quantity billed")
    unit_price: Optional[float] = Field(default=None, description="Price per unit")
    amount: Optional[float] = Field(default=None, description="Total line amount")


class InvoiceModel(BaseModel):
    """Standard financial invoice extraction model."""
    vendor_name: str = Field(description="Name of the vendor or issuing business")
    vendor_address: Optional[str] = Field(default=None, description="Vendor postal address")
    customer_name: Optional[str] = Field(default=None, description="Customer or billed entity name")
    invoice_number: Optional[str] = Field(default=None, description="Invoice reference identifier")
    invoice_date: Optional[str] = Field(default=None, description="Date the invoice was issued (YYYY-MM-DD)")
    due_date: Optional[str] = Field(default=None, description="Date payment is due")
    subtotal: Optional[float] = Field(default=None, description="Subtotal before taxes")
    tax_amount: Optional[float] = Field(default=None, description="Tax or VAT amount")
    total_amount: float = Field(description="Final total balance due")
    currency: Optional[str] = Field(default="USD", description="Currency symbol or code (e.g. USD, EUR)")
    line_items: List[InvoiceLineItem] = Field(default_factory=list, description="Itemized billing rows")


class VisualSceneModel(BaseModel):
    """Detailed visual context, object, and activity recognition model."""
    setting: str = Field(description="Physical environment or setting (e.g. office, warehouse, outdoor)")
    dominant_colors: List[str] = Field(default_factory=list, description="Dominant colors present in the scene")
    detected_objects: List[str] = Field(default_factory=list, description="Key objects, equipment, or machinery identified")
    detected_activities: List[str] = Field(default_factory=list, description="Observed actions or ongoing events")
    text_content: Optional[str] = Field(default=None, description="Any text, labels, signs, or badges visible")


# Pre-built ExtractionTemplate contract objects
INVOICE_TEMPLATE = ExtractionTemplate(
    name="InvoiceExtraction",
    description="Extract financial metadata, totals, vendor postal address, receipt date, and line items from invoices or receipts",
    schema_dict=InvoiceModel.model_json_schema(),
    prompt_guidelines=[
        "Extract vendor_name, vendor_address, customer_name, and invoice_date exactly as printed",
        "Ensure vendor_address includes the full printed address or street details",
        "Ensure invoice_date captures the date visible on the document or receipt",
        "Capture itemized rows with quantities and unit prices",
        "Verify subtotal, tax_amount, and total_amount accurately",
    ],
    categories=["finance", "accounting", "invoicing", "receipts"],
)

SCENE_TEMPLATE = ExtractionTemplate(
    name="VisualSceneExtraction",
    description="Extract visual scene context, objects, shapes, colors, and activities",
    schema_dict=VisualSceneModel.model_json_schema(),
    prompt_guidelines=[
        "Identify the primary environment and setting",
        "Catalog notable objects and their visual properties (colors, shapes)",
        "Describe human actions or physical activities underway",
    ],
    categories=["vision", "scene_understanding"],
)

SUMMARY_TEMPLATE = ExtractionTemplate(
    name="DocumentSummary",
    description="Extract executive summary, topics, and organizations",
    schema_dict=DocumentSummaryModel.model_json_schema(),
    prompt_guidelines=[
        "Draft a clear executive summary",
        "List main topics and referenced entities",
    ],
    categories=["general", "summarization"],
)

