"""ipc.py - Interprocess (Win32 Named Pipe / FastMCP) bridge to an external AuraRouter daemon."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence
from scryer.bridge.base import BaseRouterBridge, BridgeResponse
from scryer.bridge.in_process import _to_aurarouter_image
from scryer.contracts import safe_json_dumps, safe_json_loads

logger = logging.getLogger("Scryer.Bridge.IPC")


class IpcRouterBridge(BaseRouterBridge):
    """Interprocess bridge connecting to an out-of-process AuraRouter server via Named Pipes or MCP."""

    def __init__(
        self,
        pipe_name: Optional[str] = None,
        mcp_command: Optional[List[str]] = None,
    ):
        self.pipe_name = pipe_name or os.environ.get("AURAROUTER_IPC_PIPE", r"\\.\pipe\aurarouter_ipc")
        self.mcp_command = mcp_command

    def is_available(self) -> bool:
        # Check if Win32 pipe exists or MCP command is configured
        if sys.platform == "win32" and self.pipe_name:
            try:
                import ctypes
                # Check pipe existence without full open
                return os.path.exists(self.pipe_name)
            except Exception:
                return False
        return bool(self.mcp_command)

    def execute_multimodal(
        self,
        task: str,
        images: Sequence[Any],
        response_schema: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        context: Optional[str] = None,
        execution_mode: Optional[str] = None,
        route_override: Optional[str] = None,
    ) -> BridgeResponse:
        t0 = time.monotonic()
        payload = {
            "action": "route_task",
            "task": task,
            "images": [_to_aurarouter_image(im) for im in images],
            "response_schema": response_schema,
            "intent": intent,
            "context": context,
            "execution_mode": execution_mode,
        }

        # If running on Windows and Named Pipe is present
        if sys.platform == "win32" and os.path.exists(self.pipe_name):
            try:
                with open(self.pipe_name, "r+b", buffering=0) as pipe:
                    req_bytes = safe_json_dumps(payload).encode("utf-8") + b"\n"
                    pipe.write(req_bytes)
                    pipe.flush()
                    resp_bytes = pipe.readline()
                    data = safe_json_loads(resp_bytes.decode("utf-8"))
                    return BridgeResponse(
                        raw_output=data.get("output", ""),
                        structured_output=data.get("structured_data"),
                        intent=data.get("intent", intent or "VISION_EXTRACTION"),
                        complexity=data.get("complexity", 5),
                        elapsed_seconds=time.monotonic() - t0,
                        success=True,
                    )
            except Exception as e:
                logger.warning(f"Win32 Named Pipe execution failed: {e}")

        # If MCP command is specified, execute via subprocess stdio JSON-RPC
        if self.mcp_command:
            try:
                import subprocess
                req_line = safe_json_dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "route_task", "arguments": payload},
                }) + "\n"
                proc = subprocess.Popen(
                    self.mcp_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, _ = proc.communicate(input=req_line, timeout=30)
                resp = safe_json_loads(stdout)
                res_content = resp.get("result", {}).get("content", [{}])[0].get("text", "")
                return BridgeResponse(
                    raw_output=res_content,
                    elapsed_seconds=time.monotonic() - t0,
                    success=True,
                )
            except Exception as e:
                return BridgeResponse(
                    raw_output="",
                    success=False,
                    error_message=f"MCP process execution error: {e}",
                    elapsed_seconds=time.monotonic() - t0,
                )

        return BridgeResponse(
            raw_output="",
            success=False,
            error_message=f"AuraRouter IPC pipe '{self.pipe_name}' is not reachable.",
            elapsed_seconds=time.monotonic() - t0,
        )
