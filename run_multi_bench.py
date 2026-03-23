#!/usr/bin/env python3
"""Run V2 agents across multiple benchmarks in parallel."""

import json
import sys
import time
from datetime import datetime, timezone

from agents_v2 import AgentV2Config, run_agent_v2, V2_CONFIGS
from evaluate import (
    load_gsm8k, load_drop, load_arc, load_mmlu,
    evaluate_math_accuracy, evaluate_drop_f1,
    evaluate_arc_accuracy, evaluate_mmlu_accuracy,
)
from llm import get_session_cost, reset_cost_tracking, CHEAP


def run_v2_on_benchmark(config: AgentV2Config, benchmark: str, n: int = 50, seed: int = 42) -> dict:
    """Run a V2 agent on a single benchmark."""
    reset_cost_tracking()
    start = time.time()

    if benchmark == "gsm8k":
        samples = load_gsm8k(n=n, seed=seed)
        agent_fn = lambda q: run_agent_v2(config, q, answer_format="numeric")
        result = evaluate_math_accuracy(agent_fn, samples, "gsm8k")
    elif benchmark == "drop":
        samples = load_drop(n=n, seed=seed)
        agent_fn = lambda p, q: run_agent_v2(config, q, passage=p, answer_format="text")
        result = evaluate_drop_f1(agent_fn, samples)
    elif benchmark == "arc":
        samples = load_arc(n=n, seed=seed)
        agent_fn = lambda q, c: run_agent_v2(config, f"{q}\n\nChoices:\n{c}\n\nSelect the correct answer letter (A, B, C, or D).", answer_format="mc")
        result = evaluate_arc_accuracy(agent_fn, samples)
    elif benchmark == "mmlu":
        samples = load_mmlu(n=n, seed=seed)
        agent_fn = lambda q, c: run_agent_v2(config, f"{q}\n\nChoices:\n{c}\n\nSelect the correct answer letter (A, B, C, or D).", answer_format="mc")
        result = evaluate_mmlu_accuracy(agent_fn, samples)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    elapsed = time.time() - start
    result["time_s"] = round(elapsed, 1)
    result["cost_usd"] = round(get_session_cost(), 4)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True, help="V2 architecture name")
    parser.add_argument("--benchmarks", type=str, default="gsm8k,arc,drop,mmlu", help="Comma-separated benchmarks")
    parser.add_argument("--n", type=int, default=50, help="Samples per benchmark")
    args = parser.parse_args()

    if args.arch in V2_CONFIGS:
        config = V2_CONFIGS[args.arch]
    else:
        print(f"Unknown arch: {args.arch}. Available: {list(V2_CONFIGS.keys())}")
        sys.exit(1)

    benchmarks = args.benchmarks.split(",")

    print(f"agent: {config.name}")
    print(f"architecture: {config.architecture}")
    print(f"benchmarks: {benchmarks}")
    print(f"n_samples: {args.n}")
    print("---")

    results = {}
    for bench in benchmarks:
        try:
            result = run_v2_on_benchmark(config, bench, args.n)
            results[bench] = result
            print(f"benchmark: {bench}")
            print(f"score: {result['score']}")
            print(f"cost_usd: {result['cost_usd']}")
            print(f"time_s: {result['time_s']}")
            print(f"correct: {result.get('correct', 'N/A')}")
            print(f"total: {result['total']}")
            print("---")
        except Exception as e:
            print(f"benchmark: {bench}")
            print(f"error: {e}")
            print("---")

    # Summary
    print("\n=== SUMMARY ===")
    for bench, result in results.items():
        print(f"{bench}: {result['score']}% (${result['cost_usd']}, {result['time_s']}s)")


if __name__ == "__main__":
    main()
