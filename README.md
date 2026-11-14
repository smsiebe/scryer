# Scryer

Two-phase lossless semi-structured context extraction library and CLI for documents and images, powered by **AuraRouter**.

## Overview

Scryer combines deterministic file parsing (**Phase 1**) with multi-modal VLM/LLM routing via **AuraRouter** (**Phase 2**) to extract structured JSON alongside raw text and metadata in a single lossless `ScryerResult` envelope:

1. **Phase 1 (Deterministic Extraction)**
   - **PDF (`.pdf`)**: Text density analysis via `pypdfium2`; low-text/scanned pages are automatically rendered to PNG for Phase 2 VLM OCR/extraction.
   - **Office (`.docx`, `.xlsx`, `.pptx`)**: Paragraphs, Markdown tables, slide notes, workbook sheets (bounded by `max_xlsx_rows_per_sheet`), core document properties, and embedded images.
   - **Email (`.eml`, `.msg`)**: Headers, plain/HTML bodies, and image attachments.
   - **Images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tiff`, `.tif`, `MPO`)**: Dimensions, EXIF metadata, safe pixel downscaling (`max_image_pixels`), format/alpha normalization, and optional lossless metadata stripping (`strip_exif`) via `aurarouter.media.strip_image_metadata`.
   - **Plain Text (`.txt`, `.md`, `.json`, `.csv`, `.log`, `.yaml`, `.yml`)**: Line/encoding normalization and metadata.
2. **Phase 2 (AuraRouter Multi-Modal Enrichment)**
   - Executes across **Tri-Mode** backends (`TriMode.PYTHON` in-process `AuraRouterAPI`, `TriMode.IPC` over `LFR1`-framed named pipes / persistent stdio or HTTP MCP, and `TriMode.MAS` over AuraGrid distributed discovery/auction).
   - Supports optional extraction templates (`None` default schema, named presets, `ExtractionTemplate`, Pydantic `BaseModel` subclasses with `$defs`/`$ref` inlining, `dict` specifications, or YAML/JSON template files).
   - Enforces schema validation (`validate_structured_output`), strips routing diagnostic comments (`<!-- _aura_routing_context: ... -->`), propagates visual/summary OCR text back onto elements and `raw_text`, and records execution telemetry (`model_id`, `provider`, token counts, sovereignty verdict, latency).
3. **Lossless Output & Caching**
   - Every `ScryerResult` preserves `metadata`, `raw_text`, `structured_data`, `visual_elements`, `unstructured_annotations`, and `route_telemetry`. Nothing is discarded if Phase 2 is disabled, partial, or unparsed.
   - Optional content-addressed SQLite cache (`DiskExtractionCache`) keyed by document SHA-256, canonical schema hash, and extraction options.

---

## Installation

Requires **Python 3.12+** and **AuraRouter `>=0.5.8`**.

```bash
pip install -e .
```

Optional extras:
- `[msg]` (`extract-msg`) for legacy Outlook `.msg` files
- `[auragrid]` (`auragrid`) for `TriMode.MAS` distributed grid routing
- `[dev]` (`pytest`, `pytest-asyncio`, `build`) for development and packaging

---

## Python Quick Start

### Basic & Templated Extraction

```python
from pydantic import BaseModel, Field
from scryer.contracts import ScryerExtractionOptions, TriMode
from scryer.extractor import ScryerExtractor

class ReceiptSchema(BaseModel):
    vendor: str = Field(description="Merchant name")
    total: float = Field(description="Total amount paid")
    currency: str = Field(default="USD", description="ISO currency code")

extractor = ScryerExtractor()

# Using a Pydantic model, preset string ("invoice", "scene", "summary"), or None
result = extractor.extract(
    "receipt.png",
    options=ScryerExtractionOptions(
        template=ReceiptSchema,
        tri_mode=TriMode.PYTHON,
        prefer_local=True,
        strip_exif=True,
        cache_dir="./.scryer_cache",
    ),
)

print(result.structured_data)
print(result.raw_text)
print(result.route_telemetry)
```

### Async & Batch Extraction

```python
import asyncio
from scryer.contracts import ScryerExtractionOptions
from scryer.extractor import AsyncScryerExtractor

async def run():
    async_extractor = AsyncScryerExtractor(max_concurrency=4)
    results = await async_extractor.extract_batch_async(
        ["doc1.pdf", "doc2.docx", "scan.png"],
        options=ScryerExtractionOptions(template="summary"),
    )
    for res in results:
        print(res.doc_id, len(res.raw_text), res.structured_data)

asyncio.run(run())
```

### Tri-Mode Configuration (`ScryerExtractionOptions`)

| Option | Default | Purpose |
| :--- | :--- | :--- |
| `template` | `None` | Preset name (`"invoice"`, `"scene"`, `"summary"`), `ExtractionTemplate`, Pydantic class, dict, or `.yaml`/`.json` path |
| `enable_phase2` | `True` | Enable AuraRouter Phase 2 VLM/LLM enrichment |
| `force_phase2` | `False` | Force Phase 2 even for plain text without a custom template |
| `tri_mode` | `TriMode.PYTHON` | `python` (in-process), `ipc` (named pipe / MCP), or `mas` (AuraGrid) |
| `prefer_local` | `None` | Forward sovereign local-model preference to AuraRouter |
| `aurarouter_config_path` | `None` | Explicit `auraconfig.yaml` path |
| `ipc_pipe_name` / `mcp_url` | `None` | Override named-pipe/socket name or StreamableHTTP MCP URL |
| `max_image_pixels` | `4_000_000` | Downscale oversized images before visual inference |
| `strip_exif` | `False` | Strip image EXIF/metadata before Phase 2 inference |
| `cache_dir` | `None` | Enable SQLite `DiskExtractionCache` at this directory |

---

## CLI Reference

```bash
# Inspect installed parsers and available AuraRouter bridges
scryer doctor --json

# List built-in preset templates ("invoice", "scene", "summary")
scryer templates list

# Extract a single document or image
scryer extract invoice.pdf --template invoice --json -o invoice.scryer.json

# Extract via IPC / MCP bridge preferring local models
scryer extract diagram.png --template scene --tri-mode ipc --prefer-local --json

# Concurrently extract a folder of documents
scryer batch ./documents --recursive --concurrency 4 --out-dir ./extracted
```
