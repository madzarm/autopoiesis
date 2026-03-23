#!/usr/bin/env python3
"""Run improved SOTA comparison — focused on MATH and HumanEval improvements.

Runs all benchmarks in parallel for speed.
"""

import json
import time
import concurrent.futures
from agents_v2 import AgentV2Config, run_agent_v2
from evaluate import (
    load_gsm8k, load_math, load_humaneval,
    evaluate_math_accuracy, evaluate_math_bench, evaluate_humaneval,
)
from llm import call_llm, get_session_cost, reset_cost_tracking

MODEL = "gpt-4o-mini"


def run_bench(config, bench_name, samples):
    """Run a single benchmark evaluation."""
    reset_cost_tracking()
    start = time.time()

    if bench_name == "gsm8k":
        agent_fn = lambda q: run_agent_v2(config, q, answer_format="numeric")
        result = evaluate_math_accuracy(agent_fn, samples, "gsm8k")
    elif bench_name == "math":
        def math_agent(problem):
            return run_agent_v2(config, problem, answer_format="numeric")
        result = evaluate_math_bench(math_agent, samples)
    elif bench_name == "humaneval":
        def he_agent(prompt):
            return run_agent_v2(
                config,
                f"Complete the following Python function. Return ONLY the function body code (no markdown fences, no explanation, no function signature):\n\n{prompt}",
                answer_format="text",
            )
        result = evaluate_humaneval(he_agent, samples)

    elapsed = time.time() - start
    cost = get_session_cost()
    return {
        "benchmark": bench_name,
        "score": result["score"],
        "correct": result.get("correct", "?"),
        "total": result["total"],
        "cost": cost,
        "time": elapsed,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gsm", type=int, default=200)
    parser.add_argument("--n-math", type=int, default=200)
    parser.add_argument("--n-he", type=int, default=164)
    args = parser.parse_args()

    print(f"=== Improved AIDE vs SOTA (gpt-4o-mini) ===")

    # Load all benchmarks
    gsm_samples = load_gsm8k(n=args.n_gsm, seed=42)
    math_samples = load_math(n=args.n_math, seed=42)
    he_samples = load_humaneval(n=args.n_he, seed=42)

    # Our best method: tuned CoT with strong math prompting
    aide_math = AgentV2Config(
        name="aide_math_expert",
        architecture="cot",
        model=MODEL,
        temperature=0.0,
        persona="You are a mathematics professor with expertise in competition math. You solve problems with rigorous proofs and precise calculations.",
        custom_instructions=(
            "Solve the problem step by step with complete rigor. "
            "For the final answer, ALWAYS use \\boxed{answer} format. "
            "If the answer is a fraction, use \\frac{a}{b}. "
            "If the answer is a number, give the exact value."
        ),
    )

    # Progressive refine for MATH
    aide_math_refine = AgentV2Config(
        name="aide_math_refine",
        architecture="progressive_refine",
        model=MODEL,
        temperature=0.0,
        refine_rounds=2,
        persona="You are a meticulous mathematics professor who always verifies solutions.",
        custom_instructions=(
            "Solve step by step. After your solution, carefully verify each step. "
            "Use \\boxed{answer} for the final answer."
        ),
    )

    # Better HumanEval approach
    aide_code = AgentV2Config(
        name="aide_code_expert",
        architecture="cot",
        model=MODEL,
        temperature=0.0,
        persona="You are an expert Python programmer who writes clean, correct code.",
        custom_instructions=(
            "Write ONLY the function body. Do NOT include the function signature, "
            "docstring, markdown fences, or any explanation. "
            "Your output should be directly insertable into the function."
        ),
    )

    methods = [
        ("aide_math_expert", aide_math),
        ("aide_math_refine", aide_math_refine),
        ("aide_code_expert", aide_code),
    ]

    print(f"\n{'Method':25s} | {'Benchmark':10s} | {'Score':8s} | {'Cost':8s} | Correct")
    print("-" * 75)

    # Run GSM8K tests
    for name, config in methods[:2]:
        r = run_bench(config, "gsm8k", gsm_samples)
        print(f"  {name:25s} | {r['benchmark']:10s} | {r['score']:6.2f}% | ${r['cost']:.4f} | {r['correct']}/{r['total']}")

    # Run MATH tests
    for name, config in methods[:2]:
        r = run_bench(config, "math", math_samples)
        print(f"  {name:25s} | {r['benchmark']:10s} | {r['score']:6.2f}% | ${r['cost']:.4f} | {r['correct']}/{r['total']}")

    # Run HumanEval
    r = run_bench(aide_code, "humaneval", he_samples)
    print(f"  {aide_code.name:25s} | {r['benchmark']:10s} | {r['score']:6.2f}% | ${r['cost']:.4f} | {r['correct']}/{r['total']}")


if __name__ == "__main__":
    main()
