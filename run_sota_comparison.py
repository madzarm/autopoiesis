#!/usr/bin/env python3
"""Run SOTA comparison — test our best methods on gpt-4o-mini against published results.

Published SOTA (gpt-4o-mini):
- MaAS: GSM8K 92.30%, MATH 51.82%, HumanEval 92.85%
- AFlow: GSM8K 91.2%, MATH 51.3%, HumanEval 90.9%
"""

import json
import time
from agents_v2 import AgentV2Config, run_agent_v2
from evaluate import load_gsm8k, load_arc, evaluate_math_accuracy, evaluate_arc_accuracy
from llm import get_session_cost, reset_cost_tracking

# Use gpt-4o-mini to match published results
MODEL = "gpt-4o-mini"


def run_method(config: AgentV2Config, benchmark: str, n: int, seed: int = 42):
    """Run a method on a benchmark."""
    reset_cost_tracking()
    config.model = MODEL
    start = time.time()

    if benchmark == "gsm8k":
        samples = load_gsm8k(n=n, seed=seed)
        agent_fn = lambda q: run_agent_v2(config, q, answer_format="numeric")
        result = evaluate_math_accuracy(agent_fn, samples, "gsm8k")
    elif benchmark == "arc":
        samples = load_arc(n=n, seed=seed)
        agent_fn = lambda q, c: run_agent_v2(
            config, f"{q}\n\nChoices:\n{c}\n\nSelect the correct answer letter.",
            answer_format="mc"
        )
        result = evaluate_arc_accuracy(agent_fn, samples)

    elapsed = time.time() - start
    cost = get_session_cost()

    print(f"  {config.name} on {benchmark}: {result['score']}% "
          f"(${cost:.4f}, {elapsed:.0f}s, {result.get('correct', '?')}/{result['total']})")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Sample size")
    args = parser.parse_args()

    print(f"=== SOTA Comparison (gpt-4o-mini, n={args.n}) ===")
    print(f"Published SOTA: MaAS GSM8K=92.30%, AFlow GSM8K=91.2%")
    print()

    # Define our methods
    methods = [
        # Baseline
        AgentV2Config(
            name="cot_baseline", architecture="cot", model=MODEL, temperature=0.0,
        ),
        # Our best evolved ensemble
        AgentV2Config(
            name="aide_ensemble",
            architecture="ensemble_diverse",
            model=MODEL,
            temperature=0.0,
            ensemble_architectures=["code_solve", "plan_solve_verify", "progressive_refine", "cot"],
            ensemble_n=3,
            refine_rounds=2,
            persona="You are an expert problem solver. Think step by step with precision.",
            custom_instructions="1. For numerical tasks, verify calculations. 2. For knowledge tasks, reason carefully. 3. Double-check your answer. Put final answer after ####.",
        ),
        # Progressive refine
        AgentV2Config(
            name="aide_progressive",
            architecture="progressive_refine",
            model=MODEL,
            temperature=0.0,
            refine_rounds=2,
            persona="You are a meticulous problem solver who always double-checks work.",
            custom_instructions="Think step by step. After solving, review each step for errors. Put final answer after ####.",
        ),
        # Code solve
        AgentV2Config(
            name="aide_code",
            architecture="code_solve",
            model=MODEL,
            temperature=0.0,
            code_max_attempts=2,
        ),
    ]

    print("--- GSM8K ---")
    for config in methods:
        run_method(config, "gsm8k", args.n)

    print("\n--- ARC ---")
    for config in methods:
        run_method(config, "arc", args.n)


if __name__ == "__main__":
    main()
