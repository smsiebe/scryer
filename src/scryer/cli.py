"""cli.py - Command-line interface for Scryer extraction engine."""

from __future__ import annotations

import asyncio
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from scryer.bridge import TriModeBridge
from scryer.contracts import ScryerExtractionOptions, TriMode, safe_json_dumps
from scryer.extractor import AsyncScryerExtractor, ScryerExtractor
from scryer.templates.presets import INVOICE_TEMPLATE, SCENE_TEMPLATE, SUMMARY_TEMPLATE

_PRESET_MAP = {
    "invoice": INVOICE_TEMPLATE,
    "scene": SCENE_TEMPLATE,
    "summary": SUMMARY_TEMPLATE,
}


def _resolve_template_arg(template_name: Optional[str], template_file: Optional[str]) -> Any:
    if template_name and template_name in _PRESET_MAP:
        return _PRESET_MAP[template_name]
    if template_file:
        return template_file
    return None


def _run_doctor(pipe_name: Optional[str] = None, mcp_url: Optional[str] = None) -> Dict[str, Any]:
    """Probe local parsers, hard/optional packages, and tri-mode AuraRouter readiness."""
    packages = {}
    for mod in ("aurarouter", "pydantic", "jsonschema", "mcp", "PIL", "pypdfium2", "docx", "openpyxl", "pptx", "extract_msg", "yaml", "auragrid"):
        packages[mod] = importlib.util.find_spec(mod) is not None

    bridge = TriModeBridge()
    if pipe_name:
        bridge.ipc.pipe_name = pipe_name
    if mcp_url:
        bridge.ipc.mcp_url = mcp_url

    return {
        "python_version": sys.version.split()[0],
        "packages": packages,
        "bridges": {
            "python_in_process": bridge.in_process.is_available(),
            "ipc_or_mcp": bridge.ipc.is_available(),
            "mas_auragrid": bridge.mas.is_available(),
        },
    }


def _cmd_extract(args: argparse.Namespace) -> int:
    target_path = Path(args.path)
    if not target_path.exists():
        sys.stderr.write(f"Error: file not found '{target_path}'\n")
        return 1

    selected_template = _resolve_template_arg(args.template, args.template_file)
    options = ScryerExtractionOptions(
        template=selected_template,
        enable_phase2=not args.no_phase2,
        force_phase2=args.force_phase2,
        tri_mode=TriMode(args.tri_mode),
        prefer_local=args.prefer_local,
        cache_dir=args.cache_dir,
    )

    extractor = ScryerExtractor()
    try:
        result = extractor.extract(target_path, options=options)
    except Exception as e:
        sys.stderr.write(f"Extraction failed: {e}\n")
        return 1

    output_str = (
        result.to_json(indent=2)
        if args.json
        else (
            f"=== Scryer Extraction: {result.doc_id} ===\n"
            f"MIME: {result.mime_type} | Size: {result.metadata.file_size} bytes\n\n"
            f"--- Phase 1 Raw Text ({len(result.raw_text)} chars) ---\n"
            f"{result.raw_text[:1000]}...\n\n"
            f"--- Phase 2 Structured Data ---\n"
            f"{result.structured_data}\n\n"
            f"--- Annotations ({len(result.unstructured_annotations)}) ---\n"
            f"{result.unstructured_annotations}\n"
        )
    )

    if args.out:
        Path(args.out).write_text(output_str, encoding="utf-8")
    else:
        print(output_str)
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    root = Path(args.directory)
    if not root.exists() or not root.is_dir():
        sys.stderr.write(f"Error: directory not found '{root}'\n")
        return 1

    iterator = root.rglob(args.glob) if args.recursive else root.glob(args.glob)
    extractor = ScryerExtractor()
    files = sorted(
        p for p in iterator if p.is_file() and extractor.parser_registry.can_parse(p.name)
    )
    if not files:
        print(safe_json_dumps({"processed": 0, "results": []}, indent=2))
        return 0

    selected_template = _resolve_template_arg(args.template, args.template_file)
    options = ScryerExtractionOptions(
        template=selected_template,
        enable_phase2=not args.no_phase2,
        force_phase2=args.force_phase2,
        tri_mode=TriMode(args.tri_mode),
        prefer_local=args.prefer_local,
        cache_dir=args.cache_dir,
    )
    async_extractor = AsyncScryerExtractor(
        extractor=extractor,
        max_concurrency=max(1, args.concurrency),
    )
    results = asyncio.run(async_extractor.extract_batch_async(files, options=options))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for res in results:
            target = out_dir / f"{Path(res.doc_id).name}.scryer.json"
            target.write_text(res.to_json(indent=2), encoding="utf-8")

    summary = {
        "processed": len(results),
        "files": [r.doc_id for r in results],
    }
    print(safe_json_dumps(summary if not args.json else [r.to_dict() for r in results], indent=2))
    return 0


def _cmd_templates(args: argparse.Namespace) -> int:
    payload = [
        {
            "key": k,
            "name": tmpl.name,
            "description": tmpl.description,
            "categories": tmpl.categories,
        }
        for k, tmpl in _PRESET_MAP.items()
    ]
    print(safe_json_dumps(payload, indent=2))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    report = _run_doctor(pipe_name=args.pipe_name, mcp_url=args.mcp_url)
    print(safe_json_dumps(report, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from scryer.eval import BenchmarkRunner, DatasetType
    from scryer.eval.cache import BenchmarkCacheManager

    cache_mgr = BenchmarkCacheManager(cache_dir=Path(args.cache_dir) if args.cache_dir else None)
    extractor = ScryerExtractor()
    runner = BenchmarkRunner(extractor=extractor, cache_mgr=cache_mgr)

    options = ScryerExtractionOptions(
        template=args.template,
        tri_mode=TriMode(args.tri_mode),
        enable_phase2=not args.no_phase2,
        prefer_local=args.prefer_local,
    )

    report = runner.run_benchmark(
        dataset=DatasetType(args.dataset),
        limit=args.limit,
        options=options,
    )

    out_json = report.model_dump_json(indent=2)
    if args.out:
        Path(args.out).write_text(out_json, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(out_json)
    return 0


def _add_extraction_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--template", choices=["invoice", "scene", "summary"], help="Named preset template")
    p.add_argument("--template-file", help="Path to custom YAML or JSON template file")
    p.add_argument("--tri-mode", choices=["python", "ipc", "mas"], default="python", help="AuraRouter mode")
    p.add_argument("--no-phase2", action="store_true", help="Disable Phase 2 VLM inferencing")
    p.add_argument("--force-phase2", action="store_true", help="Invoke Phase 2 even for plain text without template")
    p.add_argument("--prefer-local", action="store_true", default=None, help="Prefer local AuraRouter models")
    p.add_argument("--cache-dir", help="Directory for content-addressed SQLite extraction cache")
    p.add_argument("--json", action="store_true", help="Output JSON")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scryer",
        description="Scryer: Two-Phase Lossless Context Extraction Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    extract_parser = subparsers.add_parser("extract", help="Extract context from a document or image file")
    extract_parser.add_argument("path", help="Path to the document or image file")
    _add_extraction_flags(extract_parser)
    extract_parser.add_argument("--out", "-o", help="Write output to specified file path")

    batch_parser = subparsers.add_parser("batch", help="Extract all supported files in a directory concurrently")
    batch_parser.add_argument("directory", help="Directory to scan")
    batch_parser.add_argument("--glob", default="*", help="Glob pattern (default: *)")
    batch_parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    batch_parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent extractions")
    batch_parser.add_argument("--out-dir", help="Write per-file .scryer.json results to directory")
    _add_extraction_flags(batch_parser)

    templates_parser = subparsers.add_parser("templates", help="List built-in extraction preset templates")
    templates_parser.add_argument("action", nargs="?", default="list", choices=["list"])

    doctor_parser = subparsers.add_parser("doctor", help="Diagnose installed parsers and AuraRouter bridges")
    doctor_parser.add_argument("--json", action="store_true", default=True, help="JSON output")
    doctor_parser.add_argument("--pipe-name", help="Override IPC pipe/socket path to probe")
    doctor_parser.add_argument("--mcp-url", help="MCP StreamableHTTP/SSE URL to probe")

    eval_parser = subparsers.add_parser("eval", help="Evaluate Scryer accuracy across benchmark datasets (sroie, cord, office_ocb)")
    eval_parser.add_argument("--dataset", "-d", required=True, choices=["sroie", "cord", "office_ocb"], help="Target benchmark dataset")
    eval_parser.add_argument("--limit", "-n", type=int, default=10, help="Sample limit (default: 10, 0 for all)")
    eval_parser.add_argument("--template", help="Extraction template preset or model to use")
    eval_parser.add_argument("--tri-mode", choices=["python", "ipc", "mas"], default="python", help="AuraRouter execution backend")
    eval_parser.add_argument("--no-phase2", action="store_true", help="Disable Phase 2 enrichment to benchmark pure Phase 1 deterministic parsers")
    eval_parser.add_argument("--prefer-local", action="store_true", default=None, help="Forward local model preference to AuraRouter")
    eval_parser.add_argument("--cache-dir", help="Directory for dataset cache")
    eval_parser.add_argument("--out", "-o", help="Write evaluation JSON report to path")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "extract":
        return _cmd_extract(args)
    if args.command == "batch":
        return _cmd_batch(args)
    if args.command == "templates":
        return _cmd_templates(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "eval":
        return _cmd_eval(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
