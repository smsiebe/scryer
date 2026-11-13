"""Unmocked M2 conformance tests wiring Scryer -> real AuraRouterAPI -> real loopback HTTP model server."""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from PIL import Image

from scryer.bridge import InProcessRouterBridge, TriModeBridge
from scryer.contracts import ScryerExtractionOptions
from scryer.extractor import ScryerExtractor
from scryer.templates import InvoiceModel, SCENE_TEMPLATE


def _start_ollama_server(response_json: Dict[str, Any]):
    recorded: List[Dict[str, Any]] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            recorded.append({"path": self.path, "body": body})
            payload = {
                "model": body.get("model", "qwen-vl"),
                "response": json.dumps(response_json),
                "done": True,
                "prompt_eval_count": 42,
                "eval_count": 21,
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
class TestM2InProcessConformance:
    def test_real_aurarouter_inprocess_direct_extraction_and_telemetry(self, tmp_path: Path):
        expected_invoice = {
            "vendor_name": "Northwind Supply",
            "total_amount": 320.5,
            "invoice_number": "NW-2026-11",
            "line_items": [{"description": "Widgets", "quantity": 2.0, "unit_price": 160.25, "amount": 320.5}],
        }
        expected_scene = {
            "setting": "warehouse",
            "dominant_colors": ["blue", "gray"],
            "detected_objects": ["forklift"],
            "detected_activities": ["loading"],
            "text_content": "BAY 4",
        }

        local_srv, local_thr, local_reqs = _start_ollama_server(expected_invoice)
        vision_srv, vision_thr, vision_reqs = _start_ollama_server(expected_scene)
        try:
            local_port = local_srv.server_address[1]
            vision_port = vision_srv.server_address[1]

            cfg_path = tmp_path / "auraconfig.yaml"
            cfg_data = {
                "system": {
                    "log_level": "WARNING",
                    "default_timeout": 10.0,
                    "sovereignty_enforcement": True,
                    "sovereignty_patterns": [
                        {"name": "confidential_tag", "pattern": r"\bCONFIDENTIAL\b", "severity": "high"}
                    ],
                },
                "models": {
                    "cloud_mock": {
                        "provider": "ollama",
                        "endpoint": f"http://127.0.0.1:{vision_port}/api/generate",
                        "model_name": "cloud-model",
                        "hosting_tier": "cloud",
                    },
                    "local_text": {
                        "provider": "ollama",
                        "endpoint": f"http://127.0.0.1:{local_port}/api/generate",
                        "model_name": "local-text",
                        "hosting_tier": "local",
                    },
                    "local_vision": {
                        "provider": "ollama",
                        "endpoint": f"http://127.0.0.1:{vision_port}/api/generate",
                        "model_name": "local-vlm",
                        "hosting_tier": "local",
                        "capabilities": ["vision"],
                    },
                },
                "roles": {
                    "coding": ["cloud_mock", "local_text"],
                    "extraction": ["local_text"],
                    "vision": ["local_vision"],
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

            in_proc = InProcessRouterBridge(config_path=str(cfg_path))
            extractor = ScryerExtractor(bridge=TriModeBridge(in_process=in_proc))

            # 1. Text invoice with CONFIDENTIAL marker, prefer_local=True, and explicit route_override
            txt_file = tmp_path / "inv.txt"
            txt_file.write_text("CONFIDENTIAL Northwind Supply Invoice NW-2026-11 Total $320.50", encoding="utf-8")
            res_txt = extractor.extract(
                txt_file,
                options=ScryerExtractionOptions(
                    template=InvoiceModel,
                    prefer_local=True,
                    route_override="coding",
                    aurarouter_config_path=str(cfg_path),
                ),
            )

            assert res_txt.structured_data["vendor_name"] == "Northwind Supply"
            assert res_txt.structured_data["total_amount"] == 320.5
            # prefer_local=True routed to local_text FIRST even though cloud_mock was first in chain
            assert res_txt.route_telemetry["model_id"] == "local_text"
            assert res_txt.route_telemetry["provider"] == "ollama"
            assert res_txt.route_telemetry["sovereignty_verdict"] == "sovereign"
            assert len(local_reqs) == 1  # single-step direct path (no plan/review loop)
            # Privacy auditor & sovereignty gate active (NOT disabled)
            assert in_proc._get_api()._privacy_auditor is not None

            # 2. Image extraction routes cleanly to vision role
            img = Image.new("RGB", (64, 64), color=(0, 80, 180))
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            res_img = extractor.extract(
                img_buf.getvalue(),
                filename="bay4.png",
                options=ScryerExtractionOptions(template=SCENE_TEMPLATE),
            )
            assert res_img.structured_data["setting"] == "warehouse"
            assert res_img.route_telemetry["model_id"] == "local_vision"
            assert len(vision_reqs) == 1
        finally:
            local_srv.shutdown()
            vision_srv.shutdown()
            local_srv.server_close()
            vision_srv.server_close()
            local_thr.join(timeout=2)
            vision_thr.join(timeout=2)
