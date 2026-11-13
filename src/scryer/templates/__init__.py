"""scryer.templates - Programmatic and declarative extraction template engine."""

from scryer.templates.compiler import DEFAULT_EXTRACTION_SCHEMA, TemplateCompiler
from scryer.templates.presets import (
    INVOICE_TEMPLATE,
    SCENE_TEMPLATE,
    SUMMARY_TEMPLATE,
    DocumentSummaryModel,
    InvoiceLineItem,
    InvoiceModel,
    VisualSceneModel,
)

__all__ = [
    "TemplateCompiler",
    "DEFAULT_EXTRACTION_SCHEMA",
    "DocumentSummaryModel",
    "InvoiceModel",
    "InvoiceLineItem",
    "VisualSceneModel",
    "INVOICE_TEMPLATE",
    "SCENE_TEMPLATE",
    "SUMMARY_TEMPLATE",
]
