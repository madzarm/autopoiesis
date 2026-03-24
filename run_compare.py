#!/usr/bin/env python3
"""Quick cross-benchmark comparison of top approaches.

Runs best genomes from each approach on GSM8K/50 and HumanEval/20.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from genesis import Genome, Stage, fast_eval
from dag_evolve import DAGGenome, DAGNode, fast_eval_dag
from llm import CHEAP, get_session_cost, reset_cost_tracking
from evaluate import load_gsm8k, load_humaneval


def run_comparison():
    # Load benchmarks
    gsm_samples = load_gsm8k(n=50, seed=42)
    he_samples = load_humaneval(n=20, seed=42)

    # Define best genomes from each approach
    approaches = {}

    # 1. Genesis best — with format-neutral prompts for cross-benchmark fairness
    approaches["genesis"] = Genome(name="genesis_best", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Solve this problem carefully and provide a clear answer."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="vote", condition="always"),
    ])

    # 2. DAG-Evolve best — with format-neutral prompts
    approaches["dag_evolve_as_genome"] = Genome(name="dag_best", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Solve carefully and provide a clear answer."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure"),
        Stage(action="vote", condition="always"),
    ])

    # 3. MCTS-Morph best (2-stage: expert generate + code)
    approaches["mcts_morph"] = Genome(name="mcts_best", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="You are a world-class mathematician. Solve with rigor. Provide a clear final answer."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
    ])

    # 4. Immune-QD best (full pipeline)
    approaches["immune_qd"] = Genome(name="immune_best", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Provide a clear final answer."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure", temperature=0.1),
        Stage(action="vote", condition="always"),
    ])

    # 5. LLM-Architect best (2-stage with conditional)
    approaches["llm_architect"] = Genome(name="architect_best", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="You solve GSM8K arithmetic word problems carefully and cheaply. Think step by step. Provide a clear final answer."),
        Stage(action="generate", temperature=0.4, condition="low_confidence",
              system_prompt="Solve the GSM8K problem independently from scratch. Use a different approach. Provide a clear final answer."),
    ])

    # 6. Bayesian best (6-stage)
    approaches["bayesian"] = Genome(name="bayesian_best", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Provide a clear final answer."),
        Stage(action="verify", condition="always"),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure"),
        Stage(action="vote", condition="always"),
    ])

    # Simple baselines
    approaches["baseline_cot"] = Genome(name="baseline_cot", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Provide a clear final answer."),
    ])

    approaches["baseline_code"] = Genome(name="baseline_code", model=CHEAP, stages=[
        Stage(action="generate_code", temperature=0.0, condition="always"),
    ])

    print(f"═══ Cross-Benchmark Comparison ═══")
    print(f"GSM8K: {len(gsm_samples)} samples | HumanEval: {len(he_samples)} samples")
    print(f"Approaches: {len(approaches)}\n")

    results = {}

    for name, genome in approaches.items():
        print(f"Evaluating: {name}...", end=" ", flush=True)
        reset_cost_tracking()

        # Eval GSM8K
        gsm_result = fast_eval(genome, gsm_samples, "gsm8k")
        gsm_score = gsm_result["score"]

        # Eval HumanEval
        he_result = fast_eval(genome, he_samples, "humaneval")
        he_score = he_result["score"]

        cost = get_session_cost()
        avg = (gsm_score + he_score) / 2

        results[name] = {
            "gsm8k": gsm_score, "humaneval": he_score,
            "avg": avg, "cost": cost, "n_stages": len(genome.stages)
        }
        print(f"GSM8K={gsm_score:.1f}% HE={he_score:.1f}% avg={avg:.1f}% cost=${cost:.4f}")

    # Print comparison table
    print(f"\n{'═'*80}")
    print(f"{'Approach':25s} | {'GSM8K/50':>8s} | {'HE/20':>8s} | {'Avg':>8s} | {'Stages':>6s} | {'Cost':>8s}")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")

    for name in sorted(results, key=lambda n: results[n]["avg"], reverse=True):
        r = results[name]
        print(f"{name:25s} | {r['gsm8k']:7.1f}% | {r['humaneval']:7.1f}% | "
              f"{r['avg']:7.1f}% | {r['n_stages']:6d} | ${r['cost']:.4f}")

    print(f"{'═'*80}")


if __name__ == "__main__":
    run_comparison()
