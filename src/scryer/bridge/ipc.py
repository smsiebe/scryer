"""ipc.py - Interprocess (Win32 Named Pipe / UDS IPCClient + Persistent FastMCP ClientSession) bridge."""

from __future__ import annotations

import asyncio
import atexit
from contextlib import AsyncExitStack
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from aurarouter.ipc import IPC_ADDRESS, IPCClient
from scryer.bridge.base import BaseRouterBridge, BridgeResponse, parse_structured_json
from scryer.bridge.in_process import _to_aurarouter_image

logger = logging.getLogger("Scryer.Bridge.IPC")


class _PersistentMcpWorker:
    """Maintains a persistent background event loop and mcp.ClientSession."""

    def __init__(
        self,
        mcp_command: Optional[List[str]] = None,
        mcp_url: Optional[str] = None,
    ) -> None:
        self.mcp_command = list(mcp_command) if mcp_command else None
        self.mcp_url = mcp_url
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[Any] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        ready = threading.Event()
        loop_holder: List[asyncio.AbstractEventLoop] = []

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_holder.append(loop)
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_runner, name="Scryer-MCP-Client", daemon=True)
        self._thread.start()
        ready.wait(timeout=5.0)
        self._loop = loop_holder[0]
        return self._loop

    async def _connect_async(self) -> Any:
        if self._session is not None:
            return self._session
        from mcp import ClientSession

        stack = AsyncExitStack()
        if self.mcp_url:
            if "/sse" in self.mcp_url.lower():
                from mcp.client.sse import sse_client

                read_s, write_s = await stack.enter_async_context(sse_client(self.mcp_url))
            else:
                from mcp.client.streamable_http import streamablehttp_client

                read_s, write_s, _ = await stack.enter_async_context(
                    streamablehttp_client(self.mcp_url)
                )
        elif self.mcp_command:
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=self.mcp_command[0],
                args=self.mcp_command[1:],
            )
            read_s, write_s = await stack.enter_async_context(stdio_client(params))
        else:
            await stack.aclose()
            raise RuntimeError("No MCP command or URL configured.")

        session = await stack.enter_async_context(ClientSession(read_s, write_s))
        await session.initialize()
        self._exit_stack = stack
        self._session = session
        return session

    async def _call_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        session = await self._connect_async()
        result = await session.call_tool(tool_name, arguments=arguments)
        texts: List[str] = []
        for item in getattr(result, "content", []) or []:
            txt = getattr(item, "text", None)
            if txt:
                texts.append(txt)
        return "\n".join(texts)

    async def _close_async(self) -> None:
        self._session = None
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            self._exit_stack = None

    def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: float = 60.0) -> str:
        with self._lock:
            loop = self._ensure_loop()
            fut = asyncio.run_coroutine_threadsafe(
                self._call_tool_async(tool_name, arguments),
                loop,
            )
            try:
                return fut.result(timeout=timeout)
            except Exception:
                # Drop potentially stale session so next attempt reconnects cleanly
                try:
                    asyncio.run_coroutine_threadsafe(self._close_async(), loop).result(timeout=5.0)
                except Exception:
                    pass
                raise

    def close(self) -> None:
        with self._lock:
            if self._loop is not None and not self._loop.is_closed():
                try:
                    asyncio.run_coroutine_threadsafe(self._close_async(), self._loop).result(timeout=3.0)
                except Exception:
                    pass
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None


class IpcRouterBridge(BaseRouterBridge):
    """Interprocess bridge connecting to an out-of-process AuraRouter via IPCClient or MCP ClientSession."""

    def __init__(
        self,
        pipe_name: Optional[str] = None,
        mcp_command: Optional[List[str]] = None,
        mcp_url: Optional[str] = None,
        ipc_port: Optional[int] = None,
        auth_token: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ):
        self.pipe_name = pipe_name or os.environ.get("AURAROUTER_IPC_PIPE") or IPC_ADDRESS
        env_port = os.environ.get("AURAROUTER_IPC_PORT")
        self.ipc_port = ipc_port if ipc_port is not None else (int(env_port) if env_port else None)
        self.auth_token = auth_token or os.environ.get("AURAROUTER_IPC_TOKEN")
        self.mcp_command = mcp_command
        self.mcp_url = mcp_url or os.environ.get("AURAROUTER_MCP_URL")
        self.timeout_seconds = timeout_seconds
        self._mcp_worker: Optional[_PersistentMcpWorker] = None

    def _build_ipc_client(self) -> IPCClient:
        return IPCClient(
            address=self.pipe_name,
            port=self.ipc_port,
            auth_token=self.auth_token,
        )

    def _get_mcp_worker(self) -> _PersistentMcpWorker:
        if (
            self._mcp_worker is None
            or self._mcp_worker.mcp_command != self.mcp_command
            or self._mcp_worker.mcp_url != self.mcp_url
        ):
            if self._mcp_worker is not None:
                self._mcp_worker.close()
            self._mcp_worker = _PersistentMcpWorker(
                mcp_command=self.mcp_command,
                mcp_url=self.mcp_url,
            )
        return self._mcp_worker

    def close(self) -> None:
        if self._mcp_worker is not None:
            self._mcp_worker.close()

    def is_available(self) -> bool:
        try:
            if self._build_ipc_client().ping(timeout=0.5):
                return True
        except Exception:
            pass
        return bool(self.mcp_command or self.mcp_url)

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
    ) -> BridgeResponse:
        t0 = time.monotonic()
        formatted_images = [_to_aurarouter_image(im) for im in images]

        # 1. Primary IPC channel: AuraRouter IPCClient JSON-RPC ("route_task")
        ipc_params: Dict[str, Any] = {
            "task": task,
            "context": context or "",
            "images": formatted_images if formatted_images else None,
            "response_schema": response_schema,
            "intent": intent,
            "execution_mode": execution_mode,
            "prefer_local": bool(prefer_local) if prefer_local is not None else False,
            "role_override": route_override,
            "output_format": "json" if response_schema else "text",
        }
        try:
            client = self._build_ipc_client()
            if client.ping(timeout=0.5):
                data = client.call(
                    "route_task",
                    params=ipc_params,
                    timeout=self.timeout_seconds,
                    framed=bool(formatted_images),
                )
                if isinstance(data, dict):
                    raw_text = str(data.get("output", "") or "")
                    structured, annotations = parse_structured_json(raw_text, response_schema)
                    return BridgeResponse(
                        raw_output=raw_text,
                        structured_output=structured,
                        intent=str(data.get("intent") or intent or "VISION_EXTRACTION"),
                        complexity=int(data.get("complexity") or 5),
                        elapsed_seconds=float(data.get("total_elapsed") or (time.monotonic() - t0)),
                        model_id=str(data.get("model_id") or ""),
                        provider=str(data.get("provider") or ""),
                        input_tokens=int(data.get("input_tokens") or 0),
                        output_tokens=int(data.get("output_tokens") or 0),
                        sovereignty_verdict=str(data.get("sovereignty_verdict") or ""),
                        success=True,
                        annotations=annotations,
                    )
        except Exception as e:
            logger.warning("AuraRouter IPCClient execution failed: %s", e)

        # 2. Persistent MCP ClientSession (stdio subprocess reuse or StreamableHTTP/SSE URL)
        if self.mcp_command or self.mcp_url:
            mcp_args: Dict[str, Any] = {
                "task": task,
                "context": context or "",
                "format": "json" if response_schema else "text",
                "execution_mode": execution_mode or "",
                "intent": intent or "",
            }
            if formatted_images:
                mcp_args["images"] = formatted_images
            if response_schema is not None:
                mcp_args["response_schema"] = response_schema
            try:
                raw_text = self._get_mcp_worker().call_tool(
                    "route_task",
                    mcp_args,
                    timeout=self.timeout_seconds,
                )
                structured, annotations = parse_structured_json(raw_text, response_schema)
                return BridgeResponse(
                    raw_output=raw_text,
                    structured_output=structured,
                    intent=intent or "VISION_EXTRACTION",
                    elapsed_seconds=time.monotonic() - t0,
                    model_id="mcp-routed",
                    success=True,
                    annotations=annotations,
                )
            except Exception as e:
                return BridgeResponse(
                    raw_output="",
                    success=False,
                    error_message=f"MCP ClientSession execution error: {e}",
                    elapsed_seconds=time.monotonic() - t0,
                )

        return BridgeResponse(
            raw_output="",
            success=False,
            error_message=f"AuraRouter IPC endpoint '{self.pipe_name}' is not reachable and no MCP endpoint is configured.",
            elapsed_seconds=time.monotonic() - t0,
        )
