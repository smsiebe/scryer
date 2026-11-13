"""email.py - Deterministic email message parser for .eml and .msg files."""

from __future__ import annotations

import base64
import email
from email import policy
import io
from pathlib import Path
import re
from typing import Optional
from scryer.contracts import (
    DocumentMetadata,
    ExtractedVisualElement,
    Phase1Output,
    ScryerExtractionOptions,
)
from scryer.parsers.base import BaseParser


def _strip_html_tags(html: str) -> str:
    """Deterministic basic HTML tag stripper."""
    clean = re.sub(r"<[^>]+>", " ", html)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


class EmailParser(BaseParser):
    """Deterministic parser for RFC-822 (.eml) and Outlook Compound Binary (.msg) emails."""

    EXTENSIONS = frozenset({".eml", ".msg"})
    MIME_TYPES = frozenset({"message/rfc822", "application/vnd.ms-outlook"})

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
        ext = Path(filename).suffix.lower()
        if ext == ".msg":
            return self._parse_msg(data, filename, options)
        return self._parse_eml(data, filename, options)

    def _parse_eml(
        self,
        data: bytes,
        filename: str,
        options: Optional[ScryerExtractionOptions] = None,
    ) -> Phase1Output:
        opts = options or ScryerExtractionOptions()
        msg = email.message_from_bytes(data, policy=policy.default)

        subject = msg.get("subject", "")
        sender = msg.get("from", "")
        to = msg.get("to", "")
        date_str = msg.get("date", "")

        body_text_parts = []
        body_html_parts = []
        visual_elements = []
        att_counter = 0

        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in content_disposition:
                att_filename = part.get_filename() or f"attachment_{att_counter+1}"
                att_bytes = part.get_payload(decode=True) or b""
                att_counter += 1
                if content_type.startswith("image/") and opts.extract_images:
                    visual_elements.append(
                        ExtractedVisualElement(
                            element_id=f"{filename}_{att_filename}",
                            element_type="image",
                            mime_type=content_type,
                            data_base64=base64.b64encode(att_bytes).decode("ascii"),
                            caption=f"Email attachment {att_filename}",
                        )
                    )
            elif content_type == "text/plain":
                payload = part.get_content()
                if isinstance(payload, str):
                    body_text_parts.append(payload.strip())
            elif content_type == "text/html":
                payload = part.get_content()
                if isinstance(payload, str):
                    body_html_parts.append(payload.strip())

        full_body = "\n\n".join(body_text_parts)
        if not full_body.strip() and body_html_parts:
            full_body = _strip_html_tags("\n\n".join(body_html_parts))

        header_block = (
            f"From: {sender}\n"
            f"To: {to}\n"
            f"Subject: {subject}\n"
            f"Date: {date_str}\n\n"
        )
        complete_text = header_block + full_body

        metadata = DocumentMetadata(
            file_size=len(data),
            mime_type="message/rfc822",
            page_count=1,
            author=sender,
            title=subject,
            custom_fields={
                "to": to,
                "date_header": date_str,
                "attachment_count": att_counter,
            },
        )

        return Phase1Output(
            metadata=metadata,
            raw_text=complete_text,
            visual_elements=visual_elements,
            page_texts=[complete_text],
        )

    def _parse_msg(
        self,
        data: bytes,
        filename: str,
        options: Optional[ScryerExtractionOptions] = None,
    ) -> Phase1Output:
        opts = options or ScryerExtractionOptions()
        try:
            import extract_msg
        except ImportError:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type="application/vnd.ms-outlook",
                    custom_fields={"error": "extract_msg is not installed"},
                ),
                raw_text="",
            )

        try:
            msg = extract_msg.Message(io.BytesIO(data))
            subject = msg.subject or ""
            sender = msg.sender or ""
            to = msg.to or ""
            date_str = str(msg.date or "")
            body = msg.body or ""
            if not body and msg.htmlBody:
                body = _strip_html_tags(str(msg.htmlBody))

            visual_elements = []
            att_count = 0
            if opts.extract_images and hasattr(msg, "attachments"):
                for att in msg.attachments:
                    att_count += 1
                    att_name = getattr(att, "longFilename", None) or getattr(att, "shortFilename", f"att_{att_count}")
                    att_data = getattr(att, "data", b"")
                    # Check if attachment is an image
                    if isinstance(att_data, bytes) and att_data.startswith((b"\x89PNG", b"\xff\xd8", b"GIF8")):
                        visual_elements.append(
                            ExtractedVisualElement(
                                element_id=f"{filename}_{att_name}",
                                element_type="image",
                                mime_type="image/png",
                                data_base64=base64.b64encode(att_data).decode("ascii"),
                                caption=f"MSG attachment {att_name}",
                            )
                        )

            header_block = (
                f"From: {sender}\n"
                f"To: {to}\n"
                f"Subject: {subject}\n"
                f"Date: {date_str}\n\n"
            )
            complete_text = header_block + body

            metadata = DocumentMetadata(
                file_size=len(data),
                mime_type="application/vnd.ms-outlook",
                page_count=1,
                author=sender,
                title=subject,
                custom_fields={
                    "to": to,
                    "date_header": date_str,
                    "attachment_count": att_count,
                },
            )

            return Phase1Output(
                metadata=metadata,
                raw_text=complete_text,
                visual_elements=visual_elements,
                page_texts=[complete_text],
            )
        except Exception as e:
            return Phase1Output(
                metadata=DocumentMetadata(
                    file_size=len(data),
                    mime_type="application/vnd.ms-outlook",
                    custom_fields={"error": f"Failed to parse MSG: {e}"},
                ),
                raw_text="",
            )

