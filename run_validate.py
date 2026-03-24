#!/usr/bin/env python3
"""Validate the Adaptive-Universal winner on larger samples + MATH."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from genesis import Genome, Stage, fast_eval
from llm import CHEAP, get_session_cost, reset_cost_tracking
from evaluate import load_gsm8k, load_humaneval


def run_validation():
    # The Adaptive-Universal winner
    winner = Genome(name="adaptive_universal_best", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Provide a clear final answer."),
        Stage(action="generate_code", temperature=0.5, condition="low_confidence"),
    ])

    # Also test the best single-benchmark designs for comparison
    baselines = {
        "cot_only": Genome(name="cot", model=CHEAP, stages=[
            Stage(action="generate", temperature=0.0, condition="always",
                  system_prompt="Think step by step. Answer after ####."),
        ]),
        "code_only": Genome(name="code", model=CHEAP, stages=[
            Stage(action="generate_code", temperature=0.0, condition="always"),
        ]),
        "gen_code": Genome(name="gen_code", model=CHEAP, stages=[
            Stage(action="generate", temperature=0.0, condition="always",
                  system_prompt="Think step by step. Answer after ####."),
            Stage(action="generate_code", temperature=0.0, condition="always"),
        ]),
    }

    # Load larger samples
    gsm100 = load_gsm8k(n=100, seed=42)
    he40 = load_humaneval(n=40, seed=42)

    print(f"═══ Validation on Larger Samples ═══")
    print(f"GSM8K: {len(gsm100)}, HumanEval: {len(he40)}")
    print()

    all_designs = {"winner": winner, **baselines}

    for name, genome in all_designs.items():
        reset_cost_tracking()
        gsm_result = fast_eval(genome, gsm100, "gsm8k")
        he_result = fast_eval(genome, he40, "humaneval")
        cost = get_session_cost()
        avg = (gsm_result["score"] + he_result["score"]) / 2
        print(f"  {name:25s} | GSM8K/100={gsm_result['score']:5.1f}% | "
              f"HE/40={he_result['score']:5.1f}% | avg={avg:.1f}% | cost=${cost:.4f}")

    print(f"\n{'═'*60}")
    print("VALIDATION COMPLETE")


if __name__ == "__main__":
    run_validation()
