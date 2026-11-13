"""tri_mode.py - Tri-Mode dispatcher coordinating in-process, IPC, and MAS AuraRouter backends."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
from scryer.bridge.base import BaseRouterBridge, BridgeResponse
from scryer.bridge.heuristics import RouteAssistanceHeuristics
from scryer.bridge.in_process import InProcessRouterBridge
from scryer.bridge.ipc import IpcRouterBridge
from scryer.bridge.mas import MasRouterBridge
from scryer.contracts import Phase1Output, ScryerExtractionOptions, TriMode

logger = logging.getLogger("Scryer.Bridge.TriMode")


class TriModeBridge(BaseRouterBridge):
    """Coordinates execution across Python in-process, Win32/MCP IPC, and AuraGrid MAS backends."""

    def __init__(
        self,
        in_process: Optional[InProcessRouterBridge] = None,
        ipc: Optional[IpcRouterBridge] = None,
        mas: Optional[MasRouterBridge] = None,
        default_mode: TriMode = TriMode.PYTHON,
    ):
        self.in_process = in_process or InProcessRouterBridge()
        self.ipc = ipc or IpcRouterBridge()
        self.mas = mas or MasRouterBridge()
        self.default_mode = default_mode

    def is_available(self) -> bool:
        return self.in_process.is_available() or self.ipc.is_available() or self.mas.is_available()

    def get_backend(self, mode: TriMode) -> BaseRouterBridge:
        if mode == TriMode.MAS:
            return self.mas
        if mode == TriMode.IPC:
            return self.ipc
        return self.in_process

    def execute_with_heuristics(
        self,
        phase1: Phase1Output,
        options: Optional[ScryerExtractionOptions] = None,
        task_prompt: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        prompt_guidelines: Optional[List[str]] = None,
    ) -> BridgeResponse:
        """Derive routing hints from Phase 1 telemetry and dispatch multimodal extraction."""
        opts = options or ScryerExtractionOptions()
        hints = RouteAssistanceHeuristics.derive_routing_hints(phase1)

        # Build task instruction
        guidelines_str = "\n".join(f"- {g}" for g in prompt_guidelines) if prompt_guidelines else ""
        base_task = task_prompt or (
            "Extract all text, tabular structures, and contextual entities from this visual content."
            if hints["intent"] == "VISION_OCR"
            else "Analyze this visual scene, document, and activities, extracting structured information."
        )
        if guidelines_str:
            base_task += f"\n\nExtraction Guidelines:\n{guidelines_str}"

        # Context from Phase 1 text up to configured context budget
        context = None
        if phase1.raw_text.strip():
            limit = max(500, opts.max_context_chars)
            snippet = phase1.raw_text[:limit]
            context = f"First-phase deterministic extracted text:\n{snippet}"

        effective_prefer_local = (
            opts.prefer_local if opts.prefer_local is not None else hints.get("prefer_local")
        )

        # Propagate runtime options to in-process & IPC bridges
        if opts.aurarouter_config_path and self.in_process._config_path is None:
            self.in_process._config_path = opts.aurarouter_config_path
        self.in_process._enable_privacy = opts.enable_privacy
        self.in_process._enable_savings = opts.enable_savings
        if opts.ipc_pipe_name:
            self.ipc.pipe_name = opts.ipc_pipe_name
        if opts.mcp_command:
            self.ipc.mcp_command = opts.mcp_command
        if opts.mcp_url:
            self.ipc.mcp_url = opts.mcp_url
        self.ipc.timeout_seconds = opts.timeout_seconds
        self.mas.timeout_seconds = opts.timeout_seconds

        return self.execute_multimodal(
            task=base_task,
            images=phase1.visual_elements,
            response_schema=response_schema,
            intent=hints["intent"],
            context=context,
            execution_mode=opts.execution_mode,
            route_override=opts.route_override,
            prefer_local=effective_prefer_local,
            tri_mode=opts.tri_mode,
        )

    def execute_multimodal(
        self,
        task: str,
        images: Sequence[Any],
        response_schema: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        context: Optional[str] = None,
        execution_mode: Optional[str] = None,
        route_override: Optional[str] = None,
        prefer_local: Optional[bool] = None,
        tri_mode: Optional[TriMode] = None,
    ) -> BridgeResponse:
        target_mode = tri_mode or self.default_mode
        backend = self.get_backend(target_mode)

        if not backend.is_available():
            # Graceful fallback: if Python is unavailable, try IPC, or vice versa
            if target_mode == TriMode.PYTHON and self.ipc.is_available():
                logger.info("InProcess bridge unavailable; falling back to IPC bridge.")
                backend = self.ipc
            elif target_mode == TriMode.IPC and self.in_process.is_available():
                logger.info("IPC bridge unavailable; falling back to InProcess bridge.")
                backend = self.in_process

        if not backend.is_available():
            return BridgeResponse(
                raw_output="",
                success=False,
                error_message=f"No available AuraRouter backend for mode '{target_mode}'.",
                annotations=["AuraRouter was not reachable. Extracted data relies solely on Phase 1."],
            )

        resp = backend.execute_multimodal(
            task=task,
            images=images,
            response_schema=response_schema,
            intent=intent,
            context=context,
            execution_mode=execution_mode,
            route_override=route_override,
            prefer_local=prefer_local,
        )

        # If primary failed and was not in-process, try in-process as fallback
        if not resp.success and target_mode != TriMode.PYTHON and self.in_process.is_available():
            logger.info("IPC/MAS execution failed; attempting fallback to in-process AuraRouter.")
            resp = self.in_process.execute_multimodal(
                task=task,
                images=images,
                response_schema=response_schema,
                intent=intent,
                context=context,
                execution_mode=execution_mode,
                route_override=route_override,
                prefer_local=prefer_local,
            )

        return resp

