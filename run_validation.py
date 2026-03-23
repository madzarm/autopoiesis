#!/usr/bin/env python3
"""Run full validation of best configs on larger samples across all benchmarks."""

import json
import time
from agents_v2 import AgentV2Config, run_agent_v2
from evaluate import (
    load_gsm8k, load_arc, load_drop, load_mmlu,
    evaluate_math_accuracy, evaluate_arc_accuracy, evaluate_drop_f1, evaluate_mmlu_accuracy,
)
from llm import get_session_cost, reset_cost_tracking


def validate(config: AgentV2Config, n: int = 100, seed: int = 42):
    """Full validation across all benchmarks."""
    print(f"=== Validating: {config.name} ({config.architecture}) ===")
    print(f"Samples per benchmark: {n}")
    print()

    benchmarks = [
        ("gsm8k", "numeric", load_gsm8k, evaluate_math_accuracy),
        ("arc", "mc", load_arc, evaluate_arc_accuracy),
        ("drop", "text", load_drop, evaluate_drop_f1),
    ]

    total_cost = 0.0
    results = {}

    for bench_name, fmt, loader, evaluator in benchmarks:
        reset_cost_tracking()
        start = time.time()
        print(f"--- {bench_name} ---")

        samples = loader(n=n, seed=seed)

        if bench_name == "gsm8k":
            agent_fn = lambda q: run_agent_v2(config, q, answer_format="numeric")
            result = evaluator(agent_fn, samples, "gsm8k")
        elif bench_name == "arc":
            agent_fn = lambda q, c: run_agent_v2(
                config,
                f"{q}\n\nChoices:\n{c}\n\nSelect the correct answer letter.",
                answer_format="mc",
            )
            result = evaluator(agent_fn, samples)
        elif bench_name == "drop":
            agent_fn = lambda p, q: run_agent_v2(config, q, passage=p, answer_format="text")
            result = evaluator(agent_fn, samples)

        elapsed = time.time() - start
        cost = get_session_cost()
        total_cost += cost

        print(f"  score: {result['score']}")
        print(f"  cost: ${cost:.4f}")
        print(f"  time: {elapsed:.1f}s")
        print(f"  correct: {result.get('correct', 'N/A')}/{result['total']}")

        results[bench_name] = {
            "score": result["score"],
            "cost": cost,
            "time": elapsed,
        }

    # Summary
    scores = [r["score"] for r in results.values()]
    avg = sum(scores) / len(scores)
    print(f"\n=== SUMMARY: {config.name} ===")
    for k, v in results.items():
        print(f"  {k}: {v['score']}%")
    print(f"  Average: {avg:.2f}%")
    print(f"  Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        data = json.load(f)
    config = AgentV2Config(**data)
    validate(config, n=args.n)
