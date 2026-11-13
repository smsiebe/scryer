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
from scryer.protocols import (
    DeterministicParserProtocol,
    RouterBridgeProtocol,
    ScryerExtractorProtocol,
    TemplateEngineProtocol,
)

__version__ = "0.1.0"

__all__ = [
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
    "__version__",
]
