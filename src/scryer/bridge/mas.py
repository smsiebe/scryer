"""mas.py - AuraGrid Multi-Agent System (MAS) bridge using AsyncGridContext, Service Registry, and CAS Blob Storage."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from scryer.bridge.base import BaseRouterBridge, BridgeResponse, parse_structured_json
from scryer.bridge.in_process import _to_aurarouter_image
from scryer.contracts import ExtractedVisualElement

logger = logging.getLogger("Scryer.Bridge.MAS")

try:
    from auragrid import AsyncGridContext
    from auragrid.registry_client import ServiceEndpoint

    _AURAGRID_AVAILABLE = True
except ImportError:
    AsyncGridContext = None  # type: ignore[assignment,misc]
    ServiceEndpoint = Any  # type: ignore[assignment,misc]
    _AURAGRID_AVAILABLE = False


class MasRouterBridge(BaseRouterBridge):
    """Dispatches extraction requests to distributed AuraRouter MAS nodes via AuraGrid SDK."""

    def __init__(
        self,
        grid_context: Optional[Any] = None,
        *,
        service_name: str = "RouterService",
        endpoints: Optional[List[Any]] = None,
        ipc_token: Optional[str] = None,
        timeout_seconds: float = 60.0,
        blob_offload_threshold_bytes: int = 65_536,
        use_auction: bool = False,
        auction_window_seconds: float = 0.25,
    ):
        self.grid_context = grid_context
        self.service_name = service_name
        self.endpoints = list(endpoints) if endpoints else []
        self.ipc_token = ipc_token or os.environ.get("AURAGRID_IPC_TOKEN", "")
        self.timeout_seconds = timeout_seconds
        self.blob_offload_threshold_bytes = blob_offload_threshold_bytes
        self.use_auction = use_auction
        self.auction_window_seconds = auction_window_seconds

    def is_available(self) -> bool:
        if not _AURAGRID_AVAILABLE:
            return bool(self.endpoints)
        if self.grid_context is not None or self.endpoints:
            return True
        return bool(
            os.environ.get("AURAGRID_IPC_PORT")
            or os.environ.get("AURAGRID_MAS_ID")
            or os.environ.get("AURAGRID_EMULATION_MODE") in ("1", "true", "yes")
        )

    @staticmethod
    def _decode_raw_bytes(img: Any) -> Tuple[Optional[bytes], str, Any]:
        if isinstance(img, ExtractedVisualElement) and img.data_base64:
            return base64.b64decode(img.data_base64), img.mime_type, f"data:{img.mime_type};base64,{img.data_base64}"
        if isinstance(img, bytes):
            return img, "image/png", f"data:image/png;base64,{base64.b64encode(img).decode('ascii')}"
        formatted = _to_aurarouter_image(img)
        if isinstance(formatted, str) and formatted.startswith("data:") and ";base64," in formatted:
            header, b64 = formatted.split(";base64,", 1)
            mime = header[5:] or "image/png"
            return base64.b64decode(b64), mime, formatted
        return None, "image/png", formatted

    async def _prepare_images(
        self,
        ctx: Any,
        req_id: str,
        images: Sequence[Any],
    ) -> Tuple[List[Any], List[str]]:
        """Offload large binary payloads to AuraGrid storage and return lightweight references."""
        prepared: List[Any] = []
        uploaded_uris: List[str] = []
        storage = getattr(ctx, "storage", None) if ctx is not None else None

        for idx, img in enumerate(images):
            raw_bytes, mime, inline_ref = self._decode_raw_bytes(img)
            if (
                raw_bytes is not None
                and storage is not None
                and len(raw_bytes) >= self.blob_offload_threshold_bytes
            ):
                sha = hashlib.sha256(raw_bytes).hexdigest()
                blob_uri = f"scryer/blobs/{req_id}/{idx}_{sha[:12]}.bin"
                await storage.write(blob_uri, raw_bytes)
                uploaded_uris.append(blob_uri)
                prepared.append(
                    {
                        "storage_uri": blob_uri,
                        "sha256": sha,
                        "mime_type": mime,
                    }
                )
            else:
                prepared.append(inline_ref)
        return prepared, uploaded_uris

    async def _cleanup_blobs(self, ctx: Any, uris: List[str]) -> None:
        storage = getattr(ctx, "storage", None) if ctx is not None else None
        if storage is None or not uris:
            return
        for uri in uris:
            try:
                await storage.delete(uri)
            except Exception as e:
                logger.debug("Failed to delete transient grid blob %s: %s", uri, e)

    async def _rank_endpoints_with_auction(
        self,
        ctx: Any,
        req_id: str,
        endpoints: List[Any],
        prefer_local: Optional[bool],
    ) -> List[Any]:
        if len(endpoints) <= 1 or not self.use_auction:
            return endpoints
        events = getattr(ctx, "events", None)
        if events is None:
            return endpoints

        bid_scores: Dict[str, float] = {}
        try:
            req_payload = {
                "requestId": req_id,
                "modelId": "vision",
                "promptTokenEstimate": 1024,
                "maxTokens": 2048,
                "originatorNodeId": getattr(ctx, "node_id", "scryer-client"),
            }
            await events.publish(
                "aurarouter.inference_requests",
                json.dumps(req_payload).encode("utf-8"),
                event_type="aurarouter.inference_request",
            )

            async def _collect_bids() -> None:
                async for ev in events.subscribe(f"aurarouter.inference_bids.{req_id}"):
                    raw = ev.payload if hasattr(ev, "payload") else ev
                    data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
                    if isinstance(data, dict) and data.get("nodeId"):
                        score = float(data.get("score", 0.0))
                        if data.get("isWarm"):
                            score += 0.25
                        bid_scores[str(data["nodeId"])] = score

            try:
                await asyncio.wait_for(_collect_bids(), timeout=self.auction_window_seconds)
            except asyncio.TimeoutError:
                pass
        except Exception as e:
            logger.debug("Auction bidding skipped (%s); using discovery order", e)

        if not bid_scores:
            return endpoints

        def _ep_key(ep: Any) -> float:
            nid = getattr(ep, "node_id", "") or (ep.get("nodeId", "") if isinstance(ep, dict) else "")
            return bid_scores.get(nid, 0.0)

        return sorted(endpoints, key=_ep_key, reverse=True)

    async def execute_multimodal_async(
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
        if not self.is_available():
            return BridgeResponse(
                raw_output="",
                success=False,
                error_message="AuraGrid MAS environment or client is not connected.",
                elapsed_seconds=time.monotonic() - t0,
            )

        req_id = f"inf_{uuid.uuid4().hex[:12]}"
        owns_ctx = False
        ctx = self.grid_context
        uploaded_uris: List[str] = []

        try:
            if ctx is None and AsyncGridContext is not None and not self.endpoints:
                ctx = AsyncGridContext()
                await ctx.__aenter__()
                owns_ctx = True

            # 1. Discover endpoints (configured overrides take precedence or augment discovery)
            candidates = list(self.endpoints)
            if ctx is not None and hasattr(ctx, "registry") and not candidates:
                candidates = await ctx.registry.discover(self.service_name)
            if not candidates:
                return BridgeResponse(
                    raw_output="",
                    success=False,
                    error_message=f"No AuraGrid endpoints found for service '{self.service_name}'.",
                    elapsed_seconds=time.monotonic() - t0,
                )

            ranked = await self._rank_endpoints_with_auction(ctx, req_id, candidates, prefer_local)
            prepared_images, uploaded_uris = await self._prepare_images(ctx, req_id, images)

            rpc_body = {
                "task": task,
                "context": context or "",
                "images": prepared_images if prepared_images else None,
                "response_schema": response_schema,
                "intent": intent,
                "execution_mode": execution_mode,
                "prefer_local": bool(prefer_local) if prefer_local is not None else False,
                "role_override": route_override,
            }

            headers = {"Content-Type": "application/json"}
            if self.ipc_token:
                headers["X-AuraGrid-IPC-Token"] = self.ipc_token

            errors: List[str] = []
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                for ep in ranked:
                    ep_url = (
                        getattr(ep, "endpoint_url", None)
                        or (ep.get("endpointUrl") if isinstance(ep, dict) else None)
                        or str(ep)
                    )
                    if not ep_url:
                        continue
                    target_url = f"{ep_url.rstrip('/')}/invoke/execute_multimodal"
                    try:
                        resp = await client.post(target_url, json=rpc_body, headers=headers)
                        resp.raise_for_status()
                        data = resp.json()
                        raw_text = str(data.get("output", "") or "")
                        structured, annotations = parse_structured_json(raw_text, response_schema)
                        return BridgeResponse(
                            raw_output=raw_text,
                            structured_output=structured,
                            intent=str(data.get("intent") or intent or "VISION_EXTRACTION"),
                            elapsed_seconds=time.monotonic() - t0,
                            model_id=str(data.get("model_id") or "grid-routed"),
                            provider=str(data.get("provider") or "auragrid"),
                            input_tokens=int(data.get("input_tokens") or 0),
                            output_tokens=int(data.get("output_tokens") or 0),
                            success=bool(data.get("success", True)),
                            annotations=annotations,
                        )
                    except Exception as ep_err:
                        errors.append(f"{ep_url}: {ep_err}")

            return BridgeResponse(
                raw_output="",
                success=False,
                error_message=f"All AuraGrid endpoints failed: {'; '.join(errors)}",
                elapsed_seconds=time.monotonic() - t0,
            )
        except Exception as e:
            return BridgeResponse(
                raw_output="",
                success=False,
                error_message=f"AuraGrid MAS dispatch error: {e}",
                elapsed_seconds=time.monotonic() - t0,
            )
        finally:
            await self._cleanup_blobs(ctx, uploaded_uris)
            if owns_ctx and ctx is not None:
                try:
                    await ctx.__aexit__(None, None, None)
                except Exception:
                    pass

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
        coro = self.execute_multimodal_async(
            task=task,
            images=images,
            response_schema=response_schema,
            intent=intent,
            context=context,
            execution_mode=execution_mode,
            route_override=route_override,
            prefer_local=prefer_local,
        )
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)
