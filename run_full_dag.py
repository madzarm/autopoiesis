#!/usr/bin/env python3
"""Full test set validation of DAG-Evolve's best architecture."""

import time
from genesis import Genome, Stage, fast_eval
from llm import CHEAP, get_session_cost, reset_cost_tracking
from evaluate import load_gsm8k, load_humaneval


def main():
    # DAG-Evolve's best architecture (converted to Genome for eval)
    dag_best = Genome(name="dag_evolve_best", model="gpt-4o-mini", stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Solve carefully and provide a clear answer."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure"),
        Stage(action="vote", condition="always"),
    ])

    # Also test the Adaptive-Universal winner for comparison
    adaptive_best = Genome(name="adaptive_universal_best", model="gpt-4o-mini", stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Provide a clear final answer."),
        Stage(action="generate_code", temperature=0.5, condition="low_confidence"),
    ])

    designs = {"dag_evolve": dag_best, "adaptive_universal": adaptive_best}

    print("═══ Full Test Set Validation — Top Approaches ═══\n")

    for name, genome in designs.items():
        print(f"--- {name} ---")

        # GSM8K full
        gsm = load_gsm8k(n=None)
        print(f"GSM8K: {len(gsm)} samples")
        reset_cost_tracking()
        t0 = time.time()
        gsm_result = fast_eval(genome, gsm, "gsm8k")
        print(f"  Score: {gsm_result['score']:.2f}% ({gsm_result['correct']}/{gsm_result['total']})")
        print(f"  Time: {time.time()-t0:.1f}s, Cost: ${get_session_cost():.4f}")

        # HumanEval full
        he = load_humaneval(n=None)
        print(f"HumanEval: {len(he)} samples")
        reset_cost_tracking()
        t0 = time.time()
        he_result = fast_eval(genome, he, "humaneval")
        print(f"  Score: {he_result['score']:.2f}% ({he_result['correct']}/{he_result['total']})")
        print(f"  Time: {time.time()-t0:.1f}s, Cost: ${get_session_cost():.4f}")

        avg = (gsm_result['score'] + he_result['score']) / 2
        print(f"  Average: {avg:.2f}%\n")


if __name__ == "__main__":
    main()
