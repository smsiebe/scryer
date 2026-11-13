"""scryer.templates - Programmatic and declarative extraction template engine."""

from scryer.templates.compiler import (
    DEFAULT_EXTRACTION_SCHEMA,
    TemplateCompiler,
    inline_json_schema_refs,
    validate_structured_output,
)
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
    "inline_json_schema_refs",
    "validate_structured_output",
    "DocumentSummaryModel",
    "InvoiceModel",
    "InvoiceLineItem",
    "VisualSceneModel",
    "INVOICE_TEMPLATE",
    "SCENE_TEMPLATE",
    "SUMMARY_TEMPLATE",
]

