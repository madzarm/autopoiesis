#!/usr/bin/env python3
"""Main experiment runner.

Usage:
    python run_experiment.py --agent direct --benchmark gsm8k --n 50
    python run_experiment.py --agent cot --benchmark mgsm --n 50
    python run_experiment.py --config config.json --benchmark gsm8k --n 50
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from agents import AgentConfig, BASELINE_CONFIGS, run_agent
from evaluate import (
    load_gsm8k, load_mgsm, load_drop,
    evaluate_math_accuracy, evaluate_drop_f1,
)
from archive import add_to_archive
from llm import get_session_cost, reset_cost_tracking


def run(config: AgentConfig, benchmark: str, n: int, seed: int = 42) -> dict:
    """Run a single experiment and return results."""
    reset_cost_tracking()
    start_time = time.time()

    print(f"benchmark: {benchmark}")
    print(f"agent: {config.name}")
    print(f"model: {config.model}")
    print(f"n_samples: {n}")
    print(f"reasoning: {config.reasoning}")
    print(f"reflection: {config.reflection}")
    print(f"ensemble: {config.ensemble} (n={config.ensemble_n})")
    print("---")

    if benchmark == "gsm8k":
        samples = load_gsm8k(split="test", n=n, seed=seed)
        agent_fn = lambda q: run_agent(config, q)
        result = evaluate_math_accuracy(agent_fn, samples, benchmark_name="gsm8k")

    elif benchmark == "mgsm":
        samples = load_mgsm(lang="en", n=n, seed=seed)
        agent_fn = lambda q: run_agent(config, q)
        result = evaluate_math_accuracy(agent_fn, samples, benchmark_name="mgsm")

    elif benchmark == "drop":
        samples = load_drop(split="validation", n=n, seed=seed)
        agent_fn = lambda p, q: run_agent(config, q, passage=p)
        result = evaluate_drop_f1(agent_fn, samples)

    else:
        print(f"error: unknown benchmark '{benchmark}'")
        sys.exit(1)

    elapsed = time.time() - start_time
    total_cost = get_session_cost()

    # Print results in parseable format
    print(f"score: {result['score']}")
    print(f"cost_usd: {total_cost:.4f}")
    print(f"time_s: {elapsed:.1f}")
    print(f"correct: {result.get('correct', 'N/A')}")
    print(f"total: {result['total']}")

    # Add to archive
    add_to_archive(config, benchmark, result["score"], total_cost)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run ADAS experiment")
    parser.add_argument("--agent", type=str, default=None, help="Baseline agent name")
    parser.add_argument("--config", type=str, default=None, help="Path to agent config JSON")
    parser.add_argument("--benchmark", type=str, default="gsm8k", help="Benchmark name")
    parser.add_argument("--n", type=int, default=50, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config_data = json.load(f)
        config = AgentConfig(**config_data)
    elif args.agent:
        if args.agent not in BASELINE_CONFIGS:
            print(f"error: unknown agent '{args.agent}'. Available: {list(BASELINE_CONFIGS.keys())}")
            sys.exit(1)
        config = BASELINE_CONFIGS[args.agent]
    else:
        print("error: must specify --agent or --config")
        sys.exit(1)

    run(config, args.benchmark, args.n, args.seed)


if __name__ == "__main__":
    main()
