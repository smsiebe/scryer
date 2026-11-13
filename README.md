# Scryer

Two-phase lossless semi-structured context extraction library for documents and images, powered by **AuraRouter**.

## Overview

Scryer bridges deterministic traditional file parsers (Phase 1) with ML/VLM visual and multimodal inferencing models (Phase 2):
1. **Phase 1 (Deterministic Extraction)**: Extracts raw text, layout tables, document properties, author, timestamps, dimensions, and visual elements from PDFs, Word (`.docx`), Excel (`.xlsx`), PowerPoint (`.pptx`), emails (`.eml`, `.msg`), and image formats (`.png`, `.jpg`, etc.).
2. **Phase 2 (Multimodal Inference via AuraRouter)**: Uses AuraRouter's Tri-Mode bridge (in-process Python, Named Pipes/MCP, or AuraGrid MAS) to route visual elements and low-text rendered pages to VLMs with optional structured schemas and prompt guidelines.
3. **Lossless Wrapper**: Every extraction returns a `ScryerResult` containing metadata, unified plain text, validated structured JSON, visual elements, and unstructured model commentary. Nothing is lost.

## Quick Start

```python
from scryer.extractor import ScryerExtractor
from scryer.contracts import ScryerExtractionOptions

extractor = ScryerExtractor()
result = extractor.extract("sample.pdf")

print("Raw text:", result.raw_text)
print("Metadata:", result.metadata)
print("Structured Data:", result.structured_data)
```
