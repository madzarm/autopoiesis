#!/usr/bin/env python3
"""Benchmark-specific evaluation of ALL approaches on 20 samples each.

Each approach gets benchmark-specific prompts:
- GSM8K: "Answer after ####" format
- HumanEval: "Write clean Python code" format

This matches published paper methodology (AFlow uses different optimized workflows per benchmark).
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from genesis import Genome, Stage, _eval_single_genesis
from llm import get_session_cost, reset_cost_tracking
from evaluate import load_gsm8k, load_humaneval

MODEL = "gpt-4o-mini"

# ═══════════════════════════════════════════════════════════════
# GSM8K-SPECIFIC WORKFLOWS (with #### format)
# ═══════════════════════════════════════════════════════════════

GSM8K_WORKFLOWS = {
    "genesis": Genome(name="genesis_gsm", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Put your final numerical answer after ####."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="vote", condition="always"),
    ]),
    "dag_evolve": Genome(name="dag_gsm", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Put your final numerical answer after ####."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure"),
        Stage(action="vote", condition="always"),
    ]),
    "mcts_morph": Genome(name="mcts_gsm", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="You are an expert mathematician. Think step by step. Put answer after ####."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
    ]),
    "immune_qd": Genome(name="immune_gsm", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Put your final answer after ####."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure", temperature=0.1),
        Stage(action="vote", condition="always"),
    ]),
    "bayesian": Genome(name="bayesian_gsm", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Solve step by step. Put answer after ####."),
        Stage(action="verify", condition="always"),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure"),
        Stage(action="vote", condition="always"),
    ]),
    "llm_architect": Genome(name="arch_gsm", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Put your final numerical answer after ####."),
        Stage(action="generate", temperature=0.4, condition="low_confidence",
              system_prompt="Solve from scratch. Different approach. Answer after ####."),
    ]),
    "hybrid": Genome(name="hybrid_gsm", model=MODEL, stages=[
        Stage(action="generate_code", temperature=0.5, condition="always"),
    ]),
    "adaptive_universal": Genome(name="adaptive_gsm", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Put answer after ####."),
        Stage(action="generate_code", temperature=0.5, condition="low_confidence"),
    ]),
    "cot_baseline": Genome(name="cot_gsm", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Put your final numerical answer after ####."),
    ]),
}

# ═══════════════════════════════════════════════════════════════
# HUMANEVAL-SPECIFIC WORKFLOWS (code prompts)
# ═══════════════════════════════════════════════════════════════

HE_WORKFLOWS = {
    "genesis": Genome(name="genesis_he", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Complete the function. Write clean, correct Python code."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="vote", condition="always"),
    ]),
    "dag_evolve": Genome(name="dag_he", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Complete the function. Write clean, correct Python code."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure"),
        Stage(action="vote", condition="always"),
    ]),
    "mcts_morph": Genome(name="mcts_he", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Complete this Python function correctly. Be precise."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
    ]),
    "immune_qd": Genome(name="immune_he", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Write clean, correct Python code to complete this function."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure", temperature=0.1),
        Stage(action="vote", condition="always"),
    ]),
    "bayesian": Genome(name="bayesian_he", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Complete this Python function. Write correct code."),
        Stage(action="verify", condition="always"),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure"),
        Stage(action="vote", condition="always"),
    ]),
    "llm_architect": Genome(name="arch_he", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Complete this Python function. Write clean, correct code."),
        Stage(action="generate", temperature=0.4, condition="low_confidence",
              system_prompt="Try a different approach. Write the function from scratch."),
    ]),
    "hybrid": Genome(name="hybrid_he", model=MODEL, stages=[
        Stage(action="generate_code", temperature=0.5, condition="always"),
    ]),
    "adaptive_universal": Genome(name="adaptive_he", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Complete this Python function correctly."),
        Stage(action="generate_code", temperature=0.5, condition="low_confidence"),
    ]),
    "cot_baseline": Genome(name="cot_he", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Complete this Python function. Write clean, correct code."),
    ]),
}


def eval_quick(genome, samples, benchmark, max_workers=20):
    genome_dict = genome.to_dict()
    total = len(samples)
    correct = 0
    lock = threading.Lock()

    def _eval(args):
        nonlocal correct
        idx, sample = args
        result = _eval_single_genesis(genome_dict, sample, idx, benchmark)
        with lock:
            if result.get("correct"):
                correct += 1
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_eval, enumerate(samples)))

    return round(correct / total * 100, 1) if total > 0 else 0


def main():
    gsm = load_gsm8k(n=200, seed=42)
    he = load_humaneval(n=None)  # Full HumanEval (164 samples)

    print(f"═══ Benchmark-Specific Eval (model={MODEL}, GSM8K={len(gsm)}, HE={len(he)}) ═══\n", flush=True)

    # Run ALL approaches in parallel (each approach evals both benchmarks)
    results = {}
    lock = threading.Lock()

    def _eval_approach(name):
        gsm_score = eval_quick(GSM8K_WORKFLOWS[name], gsm, "gsm8k")
        he_score = eval_quick(HE_WORKFLOWS[name], he, "humaneval")
        avg = (gsm_score + he_score) / 2
        with lock:
            results[name] = {"gsm8k": gsm_score, "humaneval": he_score, "avg": avg}
            print(f"{name:25s} | {gsm_score:7.1f}% | {he_score:7.1f}% | {avg:7.1f}%", flush=True)

    print(f"{'Approach':25s} | {'GSM8K/20':>8s} | {'HE/20':>8s} | {'Avg':>8s}", flush=True)
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}", flush=True)

    with ThreadPoolExecutor(max_workers=9) as ex:
        list(ex.map(_eval_approach, GSM8K_WORKFLOWS.keys()))

    # Sort by avg
    print(f"\n{'═'*60}", flush=True)
    print(f"{'Approach':25s} | {'GSM8K':>8s} | {'HE':>8s} | {'Avg':>8s}", flush=True)
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}", flush=True)
    for name in sorted(results, key=lambda n: results[n]["avg"], reverse=True):
        r = results[name]
        print(f"{name:25s} | {r['gsm8k']:7.1f}% | {r['humaneval']:7.1f}% | {r['avg']:7.1f}%", flush=True)

    print(f"\nPublished SOTA (gpt-4o-mini): AFlow 93.5/94.7, MaAS 92.3/92.9", flush=True)


if __name__ == "__main__":
    main()
