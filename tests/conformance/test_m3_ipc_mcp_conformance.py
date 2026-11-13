"""Unmocked M3 conformance tests for Named Pipe / UDS IPCServer + Persistent MCP ClientSession."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from PIL import Image

from aurarouter.api import APIConfig, AuraRouterAPI
from aurarouter.ipc import IPCServer, register_router_handlers
from scryer.bridge import IpcRouterBridge, TriModeBridge
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
                "model": body.get("model", "vision-m3"),
                "response": json.dumps(response_json),
                "done": True,
                "prompt_eval_count": 30,
                "eval_count": 15,
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
class TestM3IpcAndMcpConformance:
    def test_real_ipc_server_named_pipe_or_uds_with_auth_and_large_frame(self, tmp_path: Path):
        expected = {
            "setting": "lab",
            "dominant_colors": ["white", "silver"],
            "detected_objects": ["oscilloscope"],
            "detected_activities": ["testing"],
            "text_content": "PASS",
        }
        srv, thr, reqs = _start_ollama_server(expected)
        port = srv.server_address[1]
        cfg_path = tmp_path / "auraconfig.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "system": {"log_level": "WARNING", "default_timeout": 10.0},
                    "models": {
                        "local_vision": {
                            "provider": "ollama",
                            "endpoint": f"http://127.0.0.1:{port}/api/generate",
                            "model_name": "vision-m3",
                            "hosting_tier": "local",
                            "capabilities": ["vision"],
                        }
                    },
                    "roles": {"vision": ["local_vision"], "coding": ["local_vision"]},
                }
            ),
            encoding="utf-8",
        )

        api = AuraRouterAPI(APIConfig(config_path=str(cfg_path)))
        pipe_addr = (
            rf"\\.\pipe\ScryerM3Test_{uuid.uuid4().hex[:10]}"
            if sys.platform == "win32"
            else str(tmp_path / "aurarouter_m3.sock")
        )
        secret = f"tok_{uuid.uuid4().hex}"
        ipc_srv = IPCServer(address=pipe_addr, request_timeout=20.0, auth_token=secret)
        register_router_handlers(ipc_srv, api)
        ipc_srv.start(wait_ready=True, timeout=5.0)

        try:
            # Verify unauthorized client is rejected
            bad_bridge = IpcRouterBridge(pipe_name=pipe_addr, auth_token="wrong-token")
            assert bad_bridge.is_available() is False

            # Build large noisy RGB PNG (>40KB) to exercise LFR1 length-prefixed framing
            large_img = Image.effect_noise((280, 280), 40.0).convert("RGB")
            img_buf = io.BytesIO()
            large_img.save(img_buf, format="PNG")
            assert len(img_buf.getvalue()) > 32768

            good_bridge = IpcRouterBridge(pipe_name=pipe_addr, auth_token=secret, timeout_seconds=20.0)
            assert good_bridge.is_available() is True

            extractor = ScryerExtractor(bridge=TriModeBridge(ipc=good_bridge))
            res = extractor.extract(
                img_buf.getvalue(),
                filename="noise_scope.png",
                options=ScryerExtractionOptions(template=SCENE_TEMPLATE, tri_mode=TriMode.IPC),
            )
            assert res.structured_data["setting"] == "lab"
            assert res.route_telemetry["model_id"] == "local_vision"
            assert res.route_telemetry["provider"] == "ollama"
            assert len(reqs) == 1
        finally:
            ipc_srv.stop()
            srv.shutdown()
            srv.server_close()
            thr.join(timeout=2)

    def test_persistent_mcp_stdio_session_reuse(self, tmp_path: Path):
        expected = {
            "setting": "cleanroom",
            "dominant_colors": ["white"],
            "detected_objects": ["sensor"],
            "detected_activities": ["calibrating"],
            "text_content": "OK",
        }
        srv, thr, reqs = _start_ollama_server(expected)
        port = srv.server_address[1]
        cfg_path = tmp_path / "auraconfig_mcp.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "system": {"log_level": "WARNING", "default_timeout": 10.0},
                    "models": {
                        "local_vision": {
                            "provider": "ollama",
                            "endpoint": f"http://127.0.0.1:{port}/api/generate",
                            "model_name": "vision-mcp",
                            "hosting_tier": "local",
                            "capabilities": ["vision"],
                        }
                    },
                    "roles": {"vision": ["local_vision"], "coding": ["local_vision"]},
                }
            ),
            encoding="utf-8",
        )

        # Ensure Scryer connects strictly via MCP stdio, not a pre-existing IPC pipe
        mcp_cmd = [
            sys.executable,
            "-m",
            "aurarouter",
            "--config",
            str(cfg_path),
        ]
        unused_pipe = (
            rf"\\.\pipe\NonExistent_{uuid.uuid4().hex[:8]}"
            if sys.platform == "win32"
            else str(tmp_path / "missing.sock")
        )
        ipc_bridge = IpcRouterBridge(
            pipe_name=unused_pipe,
            mcp_command=mcp_cmd,
            timeout_seconds=30.0,
        )
        extractor = ScryerExtractor(bridge=TriModeBridge(ipc=ipc_bridge))

        img = Image.new("RGB", (48, 48), color=(240, 240, 240))
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")

        try:
            r1 = extractor.extract(
                img_buf.getvalue(),
                filename="mcp1.png",
                options=ScryerExtractionOptions(template=SCENE_TEMPLATE, tri_mode=TriMode.IPC),
            )
            assert r1.structured_data["setting"] == "cleanroom"
            assert r1.route_telemetry["model_id"] == "mcp-routed"
            worker_thread_1 = ipc_bridge._mcp_worker._thread

            r2 = extractor.extract(
                img_buf.getvalue(),
                filename="mcp2.png",
                options=ScryerExtractionOptions(template=SCENE_TEMPLATE, tri_mode=TriMode.IPC),
            )
            assert r2.structured_data["setting"] == "cleanroom"
            # Verify same persistent MCP worker/session reused across both calls
            assert ipc_bridge._mcp_worker._thread is worker_thread_1
            assert len(reqs) >= 2
        finally:
            ipc_bridge.close()
            srv.shutdown()
            srv.server_close()
            thr.join(timeout=2)
