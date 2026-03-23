"""Archive of discovered agent designs and their scores.

Maintains a JSON archive of all evaluated agent configs with their benchmark scores.
Supports querying for best configs, Pareto-optimal configs, etc.
"""

import json
import os
from typing import Optional
from agents import AgentConfig

ARCHIVE_PATH = "archive.json"


def load_archive() -> list[dict]:
    """Load the archive from disk."""
    if not os.path.exists(ARCHIVE_PATH):
        return []
    with open(ARCHIVE_PATH, "r") as f:
        return json.load(f)


def save_archive(archive: list[dict]):
    """Save the archive to disk."""
    with open(ARCHIVE_PATH, "w") as f:
        json.dump(archive, f, indent=2)


def add_to_archive(
    config: AgentConfig,
    benchmark: str,
    score: float,
    cost_usd: float,
    notes: str = "",
):
    """Add a result to the archive."""
    archive = load_archive()
    entry = {
        "config": config.model_dump(),
        "benchmark": benchmark,
        "score": score,
        "cost_usd": cost_usd,
        "notes": notes,
    }
    archive.append(entry)
    save_archive(archive)
    return entry


def get_best(benchmark: str, top_k: int = 5) -> list[dict]:
    """Get top-K configs for a given benchmark by score."""
    archive = load_archive()
    relevant = [e for e in archive if e["benchmark"] == benchmark]
    relevant.sort(key=lambda x: x["score"], reverse=True)
    return relevant[:top_k]


def get_pareto_front(benchmark: str) -> list[dict]:
    """Get Pareto-optimal configs (score vs cost)."""
    archive = load_archive()
    relevant = [e for e in archive if e["benchmark"] == benchmark]

    # Sort by score descending
    relevant.sort(key=lambda x: x["score"], reverse=True)

    pareto = []
    min_cost = float("inf")
    for entry in relevant:
        if entry["cost_usd"] < min_cost:
            pareto.append(entry)
            min_cost = entry["cost_usd"]
    return pareto


def get_archive_summary() -> str:
    """Get a text summary of the archive."""
    archive = load_archive()
    if not archive:
        return "Archive is empty."

    benchmarks = set(e["benchmark"] for e in archive)
    lines = [f"Archive: {len(archive)} entries across {len(benchmarks)} benchmarks\n"]

    for bench in sorted(benchmarks):
        best = get_best(bench, top_k=3)
        lines.append(f"\n{bench}:")
        for i, entry in enumerate(best):
            name = entry["config"].get("name", "unnamed")
            lines.append(f"  #{i+1}: {name} — score={entry['score']:.2f}, cost=${entry['cost_usd']:.4f}")

    return "\n".join(lines)
