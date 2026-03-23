#!/usr/bin/env python3
"""Full SOTA comparison — test AIDE methods on same benchmarks/models as published papers.

Published SOTA (gpt-4o-mini):
| Method | GSM8K | MATH | HumanEval |
|--------|-------|------|-----------|
| MaAS   | 92.30 | 51.82| 92.85    |
| AFlow  | 91.2  | 51.3 | 90.9     |
| ADAS   | 86.1  | 43.2 | 84.2     |
"""

import json
import time
from agents_v2 import AgentV2Config, run_agent_v2
from evaluate import (
    load_gsm8k, load_math, load_humaneval,
    evaluate_math_accuracy, evaluate_math_bench, evaluate_humaneval,
)
from llm import get_session_cost, reset_cost_tracking

MODEL = "gpt-4o-mini"


def test_method(config: AgentV2Config, bench_name: str, n: int, seed: int = 42):
    reset_cost_tracking()
    config.model = MODEL
    start = time.time()

    if bench_name == "gsm8k":
        samples = load_gsm8k(n=n, seed=seed)
        agent_fn = lambda q: run_agent_v2(config, q, answer_format="numeric")
        result = evaluate_math_accuracy(agent_fn, samples, "gsm8k")
    elif bench_name == "math":
        samples = load_math(n=n, seed=seed)
        agent_fn = lambda q: run_agent_v2(
            config, q, answer_format="numeric"
        )
        result = evaluate_math_bench(agent_fn, samples)
    elif bench_name == "humaneval":
        samples = load_humaneval(n=n, seed=seed)
        agent_fn = lambda q: run_agent_v2(
            config,
            f"Complete the following Python function. Return ONLY the function body (no markdown, no explanation):\n\n{q}",
            answer_format="text",
        )
        result = evaluate_humaneval(agent_fn, samples)

    elapsed = time.time() - start
    cost = get_session_cost()
    print(f"  {config.name:25s} | {bench_name:10s} | {result['score']:6.2f}% | ${cost:.4f} | {elapsed:.0f}s | {result.get('correct','?')}/{result['total']}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gsm", type=int, default=200)
    parser.add_argument("--n-math", type=int, default=100)
    parser.add_argument("--n-he", type=int, default=164)  # Full HumanEval
    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"AIDE vs Published SOTA (gpt-4o-mini)")
    print(f"{'='*80}")
    print(f"{'Method':25s} | {'Benchmark':10s} | {'Score':6s} | {'Cost':7s} | {'Time':5s} | Correct")
    print(f"{'-'*80}")

    # Published baselines (from papers)
    print(f"\n--- Published SOTA (from papers) ---")
    print(f"  {'MaAS':25s} | {'gsm8k':10s} |  92.30% |         |       |")
    print(f"  {'MaAS':25s} | {'math':10s} |  51.82% |         |       |")
    print(f"  {'MaAS':25s} | {'humaneval':10s} |  92.85% |         |       |")
    print(f"  {'AFlow':25s} | {'gsm8k':10s} |  91.20% |         |       |")
    print(f"  {'AFlow':25s} | {'math':10s} |  51.30% |         |       |")
    print(f"  {'AFlow':25s} | {'humaneval':10s} |  90.90% |         |       |")

    methods = {
        "cot": AgentV2Config(
            name="cot", architecture="cot", model=MODEL, temperature=0.0,
            persona="You are a precise problem solver.",
            custom_instructions="Think step by step. Show your reasoning. Put final answer after #### or in \\boxed{}.",
        ),
        "progressive_refine": AgentV2Config(
            name="progressive_refine", architecture="progressive_refine",
            model=MODEL, temperature=0.0, refine_rounds=2,
            persona="You are an expert who double-checks everything.",
            custom_instructions="Solve carefully, then review for errors. Put answer after #### or in \\boxed{}.",
        ),
        "code_solve": AgentV2Config(
            name="code_solve", architecture="code_solve",
            model=MODEL, temperature=0.0, code_max_attempts=2,
        ),
        "aide_ensemble": AgentV2Config(
            name="aide_ensemble", architecture="ensemble_diverse",
            model=MODEL, temperature=0.0,
            ensemble_architectures=["cot", "code_solve", "progressive_refine"],
            ensemble_n=3, refine_rounds=2,
            persona="You are an expert problem solver with precision.",
            custom_instructions="Think carefully. Verify your answer. Use #### or \\boxed{} for the final answer.",
        ),
    }

    print(f"\n--- Our Methods (AIDE) ---")

    # GSM8K
    for name, config in methods.items():
        test_method(config, "gsm8k", args.n_gsm)

    # MATH
    for name, config in methods.items():
        test_method(config, "math", args.n_math)

    # HumanEval (only code-related methods)
    for name in ["cot", "code_solve"]:
        test_method(methods[name], "humaneval", args.n_he)


if __name__ == "__main__":
    main()
