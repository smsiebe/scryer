"""Scryer - Lossless Two-Phase Document & Image Context Extractor Library."""

from scryer.contracts import (
    DocumentMetadata,
    ExtractedVisualElement,
    ExtractionTemplate,
    IntentMode,
    Phase1Output,
    ScryerExtractionOptions,
    ScryerResult,
    TriMode,
    safe_json_dumps,
    safe_json_loads,
)
from scryer.cache import DiskExtractionCache
from scryer.extractor import AsyncScryerExtractor, ScryerExtractor
from scryer.protocols import (
    DeterministicParserProtocol,
    ExtractionCacheProtocol,
    RouterBridgeProtocol,
    ScryerExtractorProtocol,
    TemplateEngineProtocol,
)
from scryer.templates import (
    INVOICE_TEMPLATE,
    SCENE_TEMPLATE,
    SUMMARY_TEMPLATE,
    DocumentSummaryModel,
    InvoiceModel,
    TemplateCompiler,
    VisualSceneModel,
)

__version__ = "0.3.0"

__all__ = [
    "AsyncScryerExtractor",
    "DiskExtractionCache",
    "ExtractionCacheProtocol",
    "ScryerExtractor",
    "TemplateCompiler",
    "DocumentMetadata",
    "ExtractedVisualElement",
    "ExtractionTemplate",
    "IntentMode",
    "Phase1Output",
    "ScryerExtractionOptions",
    "ScryerResult",
    "TriMode",
    "safe_json_dumps",
    "safe_json_loads",
    "DeterministicParserProtocol",
    "RouterBridgeProtocol",
    "TemplateEngineProtocol",
    "ScryerExtractorProtocol",
    "DocumentSummaryModel",
    "InvoiceModel",
    "VisualSceneModel",
    "INVOICE_TEMPLATE",
    "SCENE_TEMPLATE",
    "SUMMARY_TEMPLATE",
    "__version__",
]
