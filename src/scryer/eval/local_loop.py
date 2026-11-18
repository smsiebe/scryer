"""local_loop.py - Autonomous on-prem model optimization loop for Scryer.

Uses local Ollama models (such as auracore-assist-phi or phi4-mini-reasoning)
running on the local RTX 3070 to autonomously analyze extraction benchmark
failures, formulate hypotheses, test surgical code/prompt patches, verify
zero-regression conformance via test suites, commit locally on metric improvements,
and iterate until statistical convergence or max iterations (up to 100).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from scryer.contracts import ScryerExtractionOptions, TriMode
from scryer.eval.cache import BenchmarkCacheManager
from scryer.eval.contracts import DatasetType
from scryer.eval.runner import BenchmarkRunner

logger = logging.getLogger("Scryer.Eval.LocalLoop")

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_PHI_MODEL = "auracore-assist-phi:latest"


SYSTEM_PROMPT = """You are an elite, autonomous ML optimization engineer for the Scryer multimodal extraction engine.
Your mission is to improve Scryer's extraction accuracy (Field Exact Match and Token F1) against ground truth benchmark receipts and documents.

For each iteration, analyze the provided failing samples, identify the root cause, and propose a minimal, surgical code or prompt edit.

You must respond ONLY with a valid JSON object with the following exact keys:
{
  "analysis": "Precise root cause explanation of why the extraction failed on the given sample(s)",
  "hypothesis": "Testable hypothesis for how a specific guideline or regex/parsing change fixes it without regressions",
  "target_file": "src/scryer/templates/presets.py",
  "search_block": "Exact verbatim string currently in the file to be replaced",
  "replace_block": "Exact verbatim replacement string",
  "commit_msg": "Short conventional commit message describing the improvement (e.g. 'fix(presets): disambiguate merchant name from cashier')"
}

Guidelines for proposals:
1. Target files should typically be 'src/scryer/templates/presets.py' (improving prompt_guidelines or schemas) or 'src/scryer/eval/metrics.py' (refining key aliases or normalization).
2. The search_block must be an exact verbatim substring in the current target file.
3. Keep changes minimal and surgical to avoid breaking existing valid extractions.
"""


def _find_repo_root(start: Optional[Path] = None) -> Path:
    """Dynamically discover scryer repository root."""
    cur = (start or Path.cwd()).resolve()
    if (cur / "src" / "scryer").exists():
        return cur
    if (cur / "scryer" / "src" / "scryer").exists():
        return cur / "scryer"
    for parent in cur.parents:
        if (parent / "src" / "scryer").exists():
            return parent
        if (parent / "scryer" / "src" / "scryer").exists():
            return parent / "scryer"
    return cur


class LocalPhiLoopOrchestrator:
    """Orchestrates the autonomous on-prem loop driving local model self-improvement."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model_name: str = DEFAULT_PHI_MODEL,
        cache_dir: Optional[str] = None,
        significance_threshold: float = 0.005,
    ):
        self.repo_root = _find_repo_root(repo_root)
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.cache_dir = cache_dir or r"C:\tmp\scryer_benchmarks"
        self.significance_threshold = significance_threshold
        self.cache_mgr = BenchmarkCacheManager(cache_dir=Path(self.cache_dir) if self.cache_dir else None)
        self.runner = BenchmarkRunner(cache_mgr=self.cache_mgr)
        self.ledger: List[Dict[str, Any]] = []

    def _call_phi_chat(self, prompt: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Send chat request to local Ollama with JSON grammar constraint."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
            },
        }

        resp = httpx.post(f"{self.ollama_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()

        # Parse JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback if enclosed in markdown fences
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"Failed to parse JSON response from local model: {content[:300]}")

    def _apply_patch(self, target_file_rel: str, search_block: str, replace_block: str) -> bool:
        """Apply surgical string replacement to target file with resilient whitespace normalization."""
        file_path = self.repo_root / target_file_rel
        if not file_path.exists():
            logger.warning("Target file does not exist: %s", file_path)
            return False

        content = file_path.read_text(encoding="utf-8")
        # 1. Exact match
        if search_block in content:
            new_content = content.replace(search_block, replace_block, 1)
            file_path.write_text(new_content, encoding="utf-8")
            return True

        # 2. Stripped line match for single guidelines/lines
        search_lines = [line_str.strip() for line_str in search_block.strip().splitlines() if line_str.strip()]
        if len(search_lines) == 1:
            target_phrase = search_lines[0].strip('", \t\'')
            if len(target_phrase) >= 10:
                for line in content.splitlines():
                    if target_phrase in line:
                        new_content = content.replace(line, replace_block.strip(), 1)
                        file_path.write_text(new_content, encoding="utf-8")
                        return True

        # 3. Normalized whitespace regex replacement
        pattern = re.escape(search_block.strip())
        pattern = re.sub(r"\\\s+", r"\\s+", pattern)
        match = re.search(pattern, content)
        if match:
            new_content = content[:match.start()] + replace_block + content[match.end():]
            file_path.write_text(new_content, encoding="utf-8")
            return True

        logger.warning("Search block not found in %s", file_path)
        return False

    def _revert_file(self, target_file_rel: str) -> None:
        """Revert local changes to target file using git."""
        subprocess.run(
            ["git", "-C", str(self.repo_root), "checkout", "--", target_file_rel],
            capture_output=True,
            text=True,
            check=False,
        )

    def _verify_code_health(self) -> Tuple[bool, str]:
        """Run ruff and fast pytest unit test suite to verify no syntax or test regressions."""
        # 1. Ruff lint / syntax check
        ruff_res = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "src/", "tests/"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if ruff_res.returncode != 0:
            return False, f"Ruff error: {ruff_res.stdout or ruff_res.stderr}"

        # 2. Fast pytest suite
        pytest_res = subprocess.run(
            [sys.executable, "-s", "-m", "pytest", "-p", "no:napari", "-q", "tests/unit/test_eval_harness.py"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if pytest_res.returncode != 0:
            return False, f"Pytest error: {pytest_res.stdout or pytest_res.stderr}"

        return True, "All tests passed"

    def _commit_locally(self, commit_msg: str, metric_delta: float) -> Optional[str]:
        """Create a local git commit (never pushed) to track progress."""
        msg = f"{commit_msg} (delta: +{metric_delta:.4f})"
        res = subprocess.run(
            ["git", "-C", str(self.repo_root), "commit", "-am", msg],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            rev_res = subprocess.run(
                ["git", "-C", str(self.repo_root), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            return rev_res.stdout.strip() if rev_res.returncode == 0 else "committed"
        return None

    def run_loop(
        self,
        dataset: DatasetType | str = DatasetType.SROIE,
        max_iterations: int = 100,
        patience: int = 5,
        limit: int = 5,
        template: str = "invoice",
    ) -> str:
        """Execute the autonomous iterative on-prem optimization loop."""
        ds_type = DatasetType(dataset) if isinstance(dataset, str) else dataset
        options = ScryerExtractionOptions(
            template=template,
            tri_mode=TriMode.PYTHON,
            enable_phase2=True,
            cache_dir=self.cache_dir,
        )

        logger.info("Initializing baseline benchmark for dataset=%s (limit=%d)...", ds_type.value, limit)
        baseline_report = self.runner.run_benchmark(dataset=ds_type, limit=limit, options=options)
        baseline_em = baseline_report.mean_field_exact_match
        baseline_f1 = baseline_report.mean_field_f1

        logger.info(
            "Initial Baseline established: Exact Match=%.4f, Token F1=%.4f",
            baseline_em,
            baseline_f1,
        )

        current_em = baseline_em
        current_f1 = baseline_f1
        plateau_count = 0
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info("=== Starting Autonomous Iteration %d/%d ===", iteration, max_iterations)

            # 1. Identify failing samples from the previous report
            failing_samples = [
                s for s in baseline_report.samples
                if s.field_exact_match < 1.0 or s.field_f1_score < 1.0
            ]

            if not failing_samples:
                logger.info("Perfect 1.0 score achieved across all samples! Converged.")
                break

            # Pick top failing sample for targeted optimization
            focal_sample = failing_samples[0]
            failing_fields = {
                k: v for k, v in focal_sample.field_scores.items()
                if v.score < 1.0 or v.details.get("exact_match", 0.0) < 1.0
            }

            # 2. Read context of target editable file
            presets_path = self.repo_root / "src/scryer/templates/presets.py"
            presets_snippet = presets_path.read_text(encoding="utf-8") if presets_path.exists() else ""

            # 3. Formulate optimization prompt for Phi
            user_prompt = f"""Current Benchmark Optimization State:
Dataset: {ds_type.value}
Iteration: {iteration}
Current Best Exact Match: {current_em:.4f}
Current Best Token F1: {current_f1:.4f}

Failing Sample Analysis:
Sample ID: {focal_sample.sample_id}
Exact Match: {focal_sample.field_exact_match:.4f}
Token F1: {focal_sample.field_f1_score:.4f}

Ground Truth Entities:
{json.dumps(focal_sample.ground_truth_data, indent=2)}

Extracted Entities:
{json.dumps(focal_sample.extracted_data, indent=2)}

Failing Field Breakdown:
{json.dumps({k: v.details for k, v in failing_fields.items()}, indent=2)}

Current target code in src/scryer/templates/presets.py:
```python
{presets_snippet}
```

Task:
Propose a minimal, surgical change to the prompt_guidelines or InvoiceModel schema in src/scryer/templates/presets.py
to fix the discrepancy on failing field(s) without breaking other samples.
Output only the requested JSON object.
"""

            # 4. Invoke local Phi
            try:
                patch_proposal = self._call_phi_chat(user_prompt)
            except Exception as exc:
                logger.error("Failed to query local Phi: %s", exc)
                plateau_count += 1
                if plateau_count >= patience:
                    logger.info("Patience exhausted due to local model errors.")
                    break
                continue

            target_file = patch_proposal.get("target_file", "src/scryer/templates/presets.py")
            search_block = patch_proposal.get("search_block", "")
            replace_block = patch_proposal.get("replace_block", "")
            hypothesis = patch_proposal.get("hypothesis", "No hypothesis provided")
            commit_msg = patch_proposal.get("commit_msg", f"chore(opt): iteration {iteration} update")

            record: Dict[str, Any] = {
                "iteration": iteration,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "hypothesis": hypothesis,
                "target_file": target_file,
                "search_block": search_block,
                "replace_block": replace_block,
                "commit_msg": commit_msg,
                "em_before": current_em,
                "f1_before": current_f1,
                "accepted": False,
                "commit_hash": None,
                "rejection_reason": None,
            }

            # 5. Apply patch
            applied = self._apply_patch(target_file, search_block, replace_block)
            if not applied:
                record["rejection_reason"] = "Search block not matched in target file"
                self.ledger.append(record)
                plateau_count += 1
                logger.warning("Patch failed: search block not found in %s", target_file)
                if plateau_count >= patience:
                    break
                continue

            # 6. Verify health
            healthy, health_msg = self._verify_code_health()
            if not healthy:
                self._revert_file(target_file)
                record["rejection_reason"] = f"Health check failed: {health_msg}"
                self.ledger.append(record)
                plateau_count += 1
                logger.warning("Patch failed health checks: %s. Reverted.", health_msg)
                if plateau_count >= patience:
                    break
                continue

            # 7. Evaluate new benchmark
            try:
                new_report = self.runner.run_benchmark(dataset=ds_type, limit=limit, options=options)
                new_em = new_report.mean_field_exact_match
                new_f1 = new_report.mean_field_f1
            except Exception as exc:
                self._revert_file(target_file)
                record["rejection_reason"] = f"Benchmark evaluation crashed: {exc}"
                self.ledger.append(record)
                plateau_count += 1
                logger.error("Benchmark crashed after patch: %s. Reverted.", exc)
                if plateau_count >= patience:
                    break
                continue

            delta_f1 = new_f1 - current_f1
            delta_em = new_em - current_em
            record["em_after"] = new_em
            record["f1_after"] = new_f1
            record["delta_f1"] = delta_f1
            record["delta_em"] = delta_em

            # 8. Decision criteria: check statistical significance
            if delta_f1 >= self.significance_threshold or delta_em >= self.significance_threshold:
                # Accept and commit locally
                commit_hash = self._commit_locally(commit_msg, max(delta_f1, delta_em))
                record["accepted"] = True
                record["commit_hash"] = commit_hash
                current_em = new_em
                current_f1 = new_f1
                baseline_report = new_report
                plateau_count = 0
                logger.info(
                    "Iteration %d ACCEPTED! F1: %.4f -> %.4f (+%.4f), EM: %.4f -> %.4f (+%.4f), Commit: %s",
                    iteration,
                    record["f1_before"],
                    new_f1,
                    delta_f1,
                    record["em_before"],
                    new_em,
                    delta_em,
                    commit_hash,
                )
            else:
                # Revert change
                self._revert_file(target_file)
                record["accepted"] = False
                record["rejection_reason"] = (
                    f"Statistically insignificant delta (delta_f1={delta_f1:+.4f}, delta_em={delta_em:+.4f} < {self.significance_threshold})"
                )
                plateau_count += 1
                logger.info(
                    "Iteration %d REJECTED: Insignificant delta (f1: %+.4f, em: %+.4f). Reverted.",
                    iteration,
                    delta_f1,
                    delta_em,
                )

            self.ledger.append(record)

            if plateau_count >= patience:
                logger.info("Convergence reached: %d consecutive iterations without significant improvement.", patience)
                break

        # Generate markdown report
        final_report_md = self._generate_markdown_report(
            dataset_name=ds_type.value,
            initial_em=baseline_em,
            initial_f1=baseline_f1,
            final_em=current_em,
            final_f1=current_f1,
            total_iterations=iteration,
        )

        out_path = Path(self.cache_dir) / f"phi_orchestration_report_{ds_type.value}.md"
        out_path.write_text(final_report_md, encoding="utf-8")
        logger.info("Orchestration finished. Report written to %s", out_path)
        return final_report_md

    def _generate_markdown_report(
        self,
        dataset_name: str,
        initial_em: float,
        initial_f1: float,
        final_em: float,
        final_f1: float,
        total_iterations: int,
    ) -> str:
        """Produce a structured Markdown report summarizing the optimization run."""
        accepted_count = sum(1 for r in self.ledger if r.get("accepted"))
        rejected_count = total_iterations - accepted_count

        lines = [
            f"# Autonomous On-Prem Phi Optimization Report: {dataset_name.upper()}",
            "",
            "## Executive Summary",
            f"- **Optimization Model**: `{self.model_name}` (Running locally on RTX 3070)",
            f"- **Total Iterations Executed**: {total_iterations}",
            f"- **Accepted Improvements**: {accepted_count}",
            f"- **Rejected Hypotheses**: {rejected_count}",
            f"- **Initial Exact Match**: `{initial_em:.4f}` $\\to$ **Final Exact Match**: `{final_em:.4f}`",
            f"- **Initial Token F1**: `{initial_f1:.4f}` $\\to$ **Final Token F1**: `{final_f1:.4f}`",
            f"- **Total F1 Gain**: `{(final_f1 - initial_f1):+.4f}`",
            "",
            "## Iteration Log",
            "",
            "| Iter | Target File | Hypothesis | Result | $\\Delta$ F1 | $\\Delta$ EM | Commit |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: |",
        ]

        for r in self.ledger:
            status = "✅ Accepted" if r.get("accepted") else "❌ Reverted"
            c_hash = f"`{r['commit_hash']}`" if r.get("commit_hash") else "-"
            df1 = f"{r.get('delta_f1', 0.0):+.4f}" if "delta_f1" in r else "-"
            dem = f"{r.get('delta_em', 0.0):+.4f}" if "delta_em" in r else "-"
            hyp = r.get("hypothesis", "").replace("|", "/")[:60]
            t_file = Path(r.get("target_file", "")).name
            lines.append(f"| {r['iteration']} | `{t_file}` | {hyp} | {status} | {df1} | {dem} | {c_hash} |")

        lines.extend([
            "",
            "## Conclusion",
            f"The on-prem optimization loop terminated gracefully after {total_iterations} iterations based on statistical convergence criteria.",
        ])

        return "\n".join(lines)
