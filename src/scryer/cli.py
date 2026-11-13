"""cli.py - Command-line interface for Scryer extraction engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from scryer.contracts import ScryerExtractionOptions, TriMode
from scryer.extractor import ScryerExtractor
from scryer.templates.presets import INVOICE_TEMPLATE, SCENE_TEMPLATE, SUMMARY_TEMPLATE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scryer",
        description="Scryer: Two-Phase Lossless Context Extraction Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # extract command
    extract_parser = subparsers.add_parser("extract", help="Extract context from a document or image file")
    extract_parser.add_argument("path", help="Path to the document or image file")
    extract_parser.add_argument(
        "--template",
        choices=["invoice", "scene", "summary"],
        help="Named preset template (invoice, scene, summary)",
    )
    extract_parser.add_argument("--template-file", help="Path to custom YAML or JSON template file")
    extract_parser.add_argument(
        "--tri-mode",
        choices=["python", "ipc", "mas"],
        default="python",
        help="AuraRouter execution mode (default: python)",
    )
    extract_parser.add_argument("--no-phase2", action="store_true", help="Disable Phase 2 VLM inferencing")
    extract_parser.add_argument("--json", action="store_true", help="Output full ScryerResult as JSON")
    extract_parser.add_argument("--out", "-o", help="Write output to specified file path")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "extract":
        target_path = Path(args.path)
        if not target_path.exists():
            sys.stderr.write(f"Error: file not found '{target_path}'\n")
            return 1

        # Resolve template
        selected_template = None
        if args.template == "invoice":
            selected_template = INVOICE_TEMPLATE
        elif args.template == "scene":
            selected_template = SCENE_TEMPLATE
        elif args.template == "summary":
            selected_template = SUMMARY_TEMPLATE
        elif args.template_file:
            selected_template = args.template_file

        options = ScryerExtractionOptions(
            template=selected_template,
            enable_phase2=not args.no_phase2,
            tri_mode=TriMode(args.tri_mode),
        )

        extractor = ScryerExtractor()
        try:
            result = extractor.extract(target_path, options=options)
        except Exception as e:
            sys.stderr.write(f"Extraction failed: {e}\n")
            return 1

        output_str = result.to_json(indent=2) if args.json else (
            f"=== Scryer Extraction: {result.doc_id} ===\n"
            f"MIME: {result.mime_type} | Size: {result.metadata.file_size} bytes\n\n"
            f"--- Phase 1 Raw Text ({len(result.raw_text)} chars) ---\n"
            f"{result.raw_text[:1000]}...\n\n"
            f"--- Phase 2 Structured Data ---\n"
            f"{result.structured_data}\n\n"
            f"--- Annotations ({len(result.unstructured_annotations)}) ---\n"
            f"{result.unstructured_annotations}\n"
        )

        if args.out:
            Path(args.out).write_text(output_str, encoding="utf-8")
        else:
            print(output_str)

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
