"""Unmocked M4 conformance tests using real AuraGrid AsyncGridContext (standalone), ServiceServer, and RouterService."""

from __future__ import annotations

import io
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from PIL import Image

from auragrid import AsyncGridContext
from auragrid.registry_client import ServiceEndpoint
from auragrid.service_server import ServiceServer
from aurarouter.auragrid.services import RouterService
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from scryer.bridge import MasRouterBridge, TriModeBridge
from scryer.contracts import ScryerExtractionOptions, TriMode
from scryer.extractor import ScryerExtractor
from scryer.templates import SCENE_TEMPLATE


def _start_ollama_server(response_json: Dict[str, Any]):
    recorded: List[Dict[str, Any]] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            recorded.append(body)
            payload = {
                "model": body.get("model", "vision-mas"),
                "response": json.dumps(response_json),
                "done": True,
                "prompt_eval_count": 50,
                "eval_count": 25,
            }
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, recorded


@pytest.mark.conformance
class TestM4MasConformance:
    @pytest.mark.asyncio
    async def test_mas_router_bridge_service_server_and_blob_offload(self, tmp_path: Path):
        expected = {
            "setting": "grid-datacenter",
            "dominant_colors": ["black", "green"],
            "detected_objects": ["server rack"],
            "detected_activities": ["monitoring"],
            "text_content": "RACK 12",
        }
        ollama_srv, ollama_thr, reqs = _start_ollama_server(expected)
        ollama_port = ollama_srv.server_address[1]

        cfg_path = tmp_path / "auraconfig.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "system": {"log_level": "WARNING", "default_timeout": 10.0},
                    "models": {
                        "grid_vision": {
                            "provider": "ollama",
                            "endpoint": f"http://127.0.0.1:{ollama_port}/api/generate",
                            "model_name": "vision-mas",
                            "hosting_tier": "local",
                            "capabilities": ["vision"],
                        }
                    },
                    "roles": {"vision": ["grid_vision"], "coding": ["grid_vision"]},
                }
            ),
            encoding="utf-8",
        )

        fabric = ComputeFabric(ConfigLoader(config_path=str(cfg_path)))
        ipc_token = f"grid_tok_{uuid.uuid4().hex}"

        async with AsyncGridContext(standalone=True) as ctx:
            router_svc = RouterService(fabric, storage_client=ctx.storage)
            svc_server = ServiceServer(router_svc, ipc_token=ipc_token, port=0)
            svc_server.start()
            svc_port = svc_server.port

            try:
                dead_ep = ServiceEndpoint(
                    service_id="dead-1",
                    service_name="RouterService",
                    node_id="node-dead",
                    endpoint_url="http://127.0.0.1:1",
                    capabilities=["vision"],
                    metadata={},
                )
                live_ep = ServiceEndpoint(
                    service_id="live-1",
                    service_name="RouterService",
                    node_id="node-live",
                    endpoint_url=f"http://127.0.0.1:{svc_port}",
                    capabilities=["vision"],
                    metadata={},
                )

                mas_bridge = MasRouterBridge(
                    grid_context=ctx,
                    endpoints=[dead_ep, live_ep],  # also exercises endpoint failover
                    ipc_token=ipc_token,
                    blob_offload_threshold_bytes=512,
                    timeout_seconds=15.0,
                )
                assert mas_bridge.is_available() is True

                # Build PNG > 512 bytes so MasRouterBridge offloads it to ctx.storage
                img = Image.effect_noise((96, 96), 30.0).convert("RGB")
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                assert len(img_buf.getvalue()) > 512

                deleted_uris: List[str] = []
                orig_delete = ctx.storage.delete

                async def _tracked_delete(path: str, recursive: bool = False) -> bool:
                    deleted_uris.append(path)
                    return await orig_delete(path, recursive=recursive)

                ctx.storage.delete = _tracked_delete  # type: ignore[method-assign]

                extractor = ScryerExtractor(bridge=TriModeBridge(mas=mas_bridge))
                res = await extractor_extract_via_async_bridge(extractor, img_buf.getvalue())

                assert res.structured_data["setting"] == "grid-datacenter"
                assert res.route_telemetry["model_id"] == "grid_vision"
                assert res.route_telemetry["provider"] == "ollama"
                assert len(reqs) == 1
                # Transient blob was uploaded to ctx.storage and cleaned up via delete()
                assert len(getattr(ctx.ipc, "_storage", {})) == 1
                uploaded_keys = list(getattr(ctx.ipc, "_storage", {}).keys())
                assert deleted_uris == uploaded_keys
            finally:
                svc_server.stop()
                ollama_srv.shutdown()
                ollama_srv.server_close()
                ollama_thr.join(timeout=2)


async def extractor_extract_via_async_bridge(extractor: ScryerExtractor, img_bytes: bytes):
    import asyncio

    return await asyncio.to_thread(
        extractor.extract,
        img_bytes,
        options=ScryerExtractionOptions(template=SCENE_TEMPLATE, tri_mode=TriMode.MAS),
        filename="rack12.png",
    )
