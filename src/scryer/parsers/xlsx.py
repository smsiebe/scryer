"""xlsx.py - Deterministic Excel workbook parser using openpyxl."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional
from scryer.contracts import (
    DocumentMetadata,
    Phase1Output,
    ScryerExtractionOptions,
)
from scryer.parsers.base import BaseParser


class XLSXParser(BaseParser):
    """Deterministic XLSX parser extracting sheets, rows, and table cells into Markdown."""

    EXTENSIONS = frozenset({".xlsx", ".xlsm"})
    MIME_TYPES = frozenset({
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    })

    def can_parse(self, filename_or_mime: str) -> bool:
        ext = Path(filename_or_mime).suffix.lower()
        if ext in self.EXTENSIONS:
            return True
        return filename_or_mime.lower() in self.MIME_TYPES

    def parse(
        self,
        data: bytes,
        filename: str,
        options: Optional[ScryerExtractionOptions] = None,
    ) -> Phase1Output:
        try:
            import openpyxl
        except ImportError:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    custom_fields={"error": "openpyxl is not installed"},
                ),
                raw_text="",
            )

        try:
            wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
        except Exception as e:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    custom_fields={"error": f"Failed to parse XLSX: {e}"},
                ),
                raw_text="",
            )

        sheet_texts = []
        sheet_names = list(wb.sheetnames)
        total_rows = 0

        for sname in sheet_names:
            sheet = wb[sname]
            lines = [f"## Sheet: {sname}"]
            sheet_rows = 0
            first_row = True
            for row in sheet.iter_rows(values_only=True):
                non_empty = [str(v).strip() if v is not None else "" for v in row]
                if any(non_empty):
                    sheet_rows += 1
                    lines.append("| " + " | ".join(non_empty) + " |")
                    if first_row:
                        lines.append("| " + " | ".join("---" for _ in non_empty) + " |")
                        first_row = False
            total_rows += sheet_rows
            if len(lines) > 1:
                sheet_texts.append("\n".join(lines))

        full_text = "\n\n".join(sheet_texts)

        # Core properties
        author = None
        title = None
        created_ts = None
        modified_ts = None
        try:
            props = wb.properties
            author = props.creator
            title = props.title
            created_ts = props.created
            modified_ts = props.modified
        except Exception:
            pass

        metadata = DocumentMetadata(
            file_size=len(data),
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            page_count=len(sheet_names),
            sheet_names=sheet_names,
            author=author,
            title=title,
            created_ts=created_ts,
            modified_ts=modified_ts,
            custom_fields={
                "sheet_count": len(sheet_names),
                "total_rows_extracted": total_rows,
            },
        )

        return Phase1Output(
            metadata=metadata,
            raw_text=full_text,
            page_texts=sheet_texts,
        )
