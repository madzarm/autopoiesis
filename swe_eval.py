"""SWE-bench evaluation — local + sb-cli cloud evaluation.

Handles:
1. Formatting predictions in SWE-bench format
2. Running evaluation via sb-cli (free cloud eval)
3. Local evaluation via Docker (if available)
4. Quick proxy evaluation (patch-applies + basic checks)
"""

import json
import os
import subprocess
import time
import tempfile
from typing import Optional


def format_predictions(results: list, output_path: str) -> str:
    """Format agent results into SWE-bench predictions JSONL."""
    predictions = []
    for r in results:
        pred = {
            "instance_id": r["instance_id"],
            "model_name_or_path": r.get("model_name_or_path", "adas_agent"),
            "model_patch": r.get("model_patch", ""),
        }
        predictions.append(pred)

    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Wrote {len(predictions)} predictions to {output_path}")
    non_empty = sum(1 for p in predictions if p["model_patch"].strip())
    print(f"  Non-empty patches: {non_empty}/{len(predictions)}")
    return output_path


def evaluate_sb_cli(predictions_path: str, split: str = "verified",
                    run_id: str = None) -> dict:
    """Evaluate using sb-cli (free cloud evaluation).

    Requires: SWEBENCH_API_KEY env var or prior sb-cli setup.
    Returns dict with results.
    """
    if run_id is None:
        run_id = f"adas_{int(time.time())}"

    dataset_map = {
        "verified": "swe-bench_verified",
        "lite": "swe-bench_lite",
        "test": "swe-bench",
    }
    dataset = dataset_map.get(split, split)

    # Submit
    cmd = [
        "sb-cli", "submit", dataset, "test",
        "--predictions_path", predictions_path,
        "--run_id", run_id,
    ]

    print(f"Submitting to sb-cli: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(f"Submit output: {result.stdout}")
    if result.returncode != 0:
        print(f"Submit error: {result.stderr}")
        return {"error": result.stderr, "run_id": run_id}

    # Poll for results
    for i in range(60):  # Wait up to 30 minutes
        time.sleep(30)
        report_cmd = ["sb-cli", "get-report", dataset, "test", run_id]
        report = subprocess.run(report_cmd, capture_output=True, text=True, timeout=60)

        if report.returncode == 0 and "pending" not in report.stdout.lower():
            print(f"Results ready after {(i+1)*30}s")
            return parse_sb_cli_report(report.stdout, run_id)

        if (i + 1) % 4 == 0:
            print(f"  Waiting for results... ({(i+1)*30}s elapsed)")

    return {"error": "Timeout waiting for results", "run_id": run_id}


def parse_sb_cli_report(report_text: str, run_id: str) -> dict:
    """Parse sb-cli report output."""
    result = {"run_id": run_id, "raw": report_text}

    # Try to extract resolved count
    import re
    resolved_match = re.search(r'resolved[:\s]+(\d+)', report_text, re.IGNORECASE)
    total_match = re.search(r'total[:\s]+(\d+)', report_text, re.IGNORECASE)

    if resolved_match:
        result["resolved"] = int(resolved_match.group(1))
    if total_match:
        result["total"] = int(total_match.group(1))
    if "resolved" in result and "total" in result and result["total"] > 0:
        result["resolve_rate"] = round(result["resolved"] / result["total"] * 100, 1)

    return result


def evaluate_local_docker(predictions_path: str,
                          dataset_name: str = "princeton-nlp/SWE-bench_Verified",
                          max_workers: int = 4,
                          run_id: str = None) -> dict:
    """Evaluate using local Docker (swebench harness)."""
    if run_id is None:
        run_id = f"adas_{int(time.time())}"

    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", predictions_path,
        "--max_workers", str(max_workers),
        "--run_id", run_id,
        "--cache_level", "env",
    ]

    print(f"Running local eval: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    return {
        "run_id": run_id,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def quick_proxy_eval(results: list, instances: list) -> dict:
    """Quick proxy evaluation — checks if patches are syntactically valid diffs.

    This is NOT a real eval — just a fast sanity check.
    Returns stats on patch quality.
    """
    stats = {
        "total": len(results),
        "has_patch": 0,
        "valid_diff": 0,
        "has_hunks": 0,
        "has_error": 0,
        "empty": 0,
    }

    for r in results:
        patch = r.get("model_patch", "")
        if r.get("error"):
            stats["has_error"] += 1
            continue
        if not patch.strip():
            stats["empty"] += 1
            continue

        stats["has_patch"] += 1

        # Check if it looks like a valid diff
        if "diff --git" in patch or ("---" in patch and "+++" in patch):
            stats["valid_diff"] += 1
        if "@@" in patch:
            stats["has_hunks"] += 1

    stats["patch_rate"] = round(stats["has_patch"] / max(stats["total"], 1) * 100, 1)
    stats["valid_diff_rate"] = round(stats["valid_diff"] / max(stats["total"], 1) * 100, 1)

    return stats


def save_results(results: list, path: str):
    """Save full results with metadata."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {len(results)} results to {path}")


def load_results(path: str) -> list:
    """Load results from file."""
    with open(path) as f:
        return json.load(f)
