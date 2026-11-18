"""test_local_loop.py - Unit tests for autonomous local loop orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from scryer.eval.local_loop import LocalPhiLoopOrchestrator


def test_orchestrator_initialization(tmp_path: Path):
    orch = LocalPhiLoopOrchestrator(
        repo_root=tmp_path,
        cache_dir=str(tmp_path / "cache"),
        significance_threshold=0.01,
    )
    assert orch.repo_root == tmp_path
    assert orch.significance_threshold == 0.01
    assert orch.model_name == "auracore-assist-phi:latest"


def test_apply_and_revert_patch(tmp_path: Path):
    orch = LocalPhiLoopOrchestrator(repo_root=tmp_path)
    test_file = tmp_path / "sample.py"
    test_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    # Apply valid patch
    applied = orch._apply_patch(
        "sample.py",
        "return 'world'",
        "return 'scryer'",
    )
    assert applied is True
    assert "return 'scryer'" in test_file.read_text(encoding="utf-8")

    # Apply invalid patch (search block not present)
    invalid_applied = orch._apply_patch(
        "sample.py",
        "non_existent_text",
        "something_else",
    )
    assert invalid_applied is False


@patch("httpx.post")
def test_call_phi_chat_success(mock_post, tmp_path: Path):
    orch = LocalPhiLoopOrchestrator(repo_root=tmp_path)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "message": {
            "content": json.dumps({
                "analysis": "Test failure",
                "hypothesis": "Test hypothesis",
                "target_file": "sample.py",
                "search_block": "return 'world'",
                "replace_block": "return 'scryer'",
                "commit_msg": "fix: update return value",
            })
        }
    }
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    res = orch._call_phi_chat("Test prompt")
    assert res["hypothesis"] == "Test hypothesis"
    assert res["commit_msg"] == "fix: update return value"


def test_generate_markdown_report(tmp_path: Path):
    orch = LocalPhiLoopOrchestrator(repo_root=tmp_path)
    orch.ledger.append({
        "iteration": 1,
        "target_file": "presets.py",
        "hypothesis": "Fix vendor regex",
        "accepted": True,
        "delta_f1": 0.05,
        "delta_em": 0.05,
        "commit_hash": "abc1234",
    })
    orch.ledger.append({
        "iteration": 2,
        "target_file": "presets.py",
        "hypothesis": "Insignificant change",
        "accepted": False,
        "delta_f1": 0.001,
        "delta_em": 0.0,
        "commit_hash": None,
    })

    report = orch._generate_markdown_report(
        dataset_name="sroie",
        initial_em=0.50,
        initial_f1=0.52,
        final_em=0.55,
        final_f1=0.57,
        total_iterations=2,
    )
    assert "# Autonomous On-Prem Phi Optimization Report: SROIE" in report
    assert "✅ Accepted" in report
    assert "❌ Reverted" in report
    assert "+0.0500" in report
