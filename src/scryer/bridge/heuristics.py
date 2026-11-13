"""heuristics.py - Route-assistance heuristics based on Phase 1 extraction telemetry."""

from __future__ import annotations

from typing import Any, Dict, Optional
from scryer.contracts import Phase1Output


class RouteAssistanceHeuristics:
    """Analyzes Phase 1 deterministic outputs to derive optimal AuraRouter routing intents and complexity."""

    @staticmethod
    def derive_routing_hints(
        phase1: Phase1Output,
        user_intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Derive intent, estimated complexity, and execution hints from Phase 1 output."""
        if user_intent and user_intent != "AUTO":
            intent = user_intent
        elif phase1.low_text_pages and len(phase1.low_text_pages) > 0:
            # Low text density on pages indicates scanned documents or image-based forms -> OCR priority
            intent = "VISION_OCR"
        elif phase1.visual_elements:
            # Embedded images, diagrams, photos -> extraction priority
            intent = "VISION_EXTRACTION"
        else:
            intent = "GENERAL"

        # Complexity scoring heuristic (1 to 10)
        complexity = 3
        if phase1.visual_elements:
            num_visuals = len(phase1.visual_elements)
            if num_visuals == 1:
                complexity = 5
            elif num_visuals <= 4:
                complexity = 7
            else:
                complexity = 9

        if phase1.metadata.sheet_names and len(phase1.metadata.sheet_names) > 3:
            complexity = max(complexity, 6)

        # Detect tabular density
        table_count = phase1.metadata.custom_fields.get("table_count", 0)
        if table_count > 0:
            complexity = min(10, complexity + 1)

        # Local sovereignty hint: keep OCR and simple scans on local models if available
        prefer_local = intent == "VISION_OCR" or complexity <= 6

        return {
            "intent": intent,
            "complexity": complexity,
            "prefer_local": prefer_local,
            "has_visual_elements": bool(phase1.visual_elements),
            "visual_count": len(phase1.visual_elements),
            "low_text_pages": phase1.low_text_pages,
        }
