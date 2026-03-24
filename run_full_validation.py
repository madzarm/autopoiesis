#!/usr/bin/env python3
"""Full test set validation of the Adaptive-Universal winner.

Runs on FULL GSM8K (1319) and FULL HumanEval (164) for direct comparison
with published SOTA numbers.
"""

import time
from genesis import Genome, Stage, fast_eval
from llm import CHEAP, get_session_cost, reset_cost_tracking
from evaluate import load_gsm8k, load_humaneval


def main():
    winner = Genome(name="adaptive_universal_best", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Provide a clear final answer."),
        Stage(action="generate_code", temperature=0.5, condition="low_confidence"),
    ])

    # Full GSM8K
    print("═══ Full Test Set Validation ═══\n")
    print("Loading GSM8K full test set...")
    gsm_full = load_gsm8k(n=None)
    print(f"GSM8K: {len(gsm_full)} samples")

    reset_cost_tracking()
    t0 = time.time()
    gsm_result = fast_eval(winner, gsm_full, "gsm8k")
    gsm_time = time.time() - t0
    gsm_cost = get_session_cost()
    print(f"GSM8K: {gsm_result['score']:.2f}% ({gsm_result['correct']}/{gsm_result['total']})")
    print(f"  Time: {gsm_time:.1f}s, Cost: ${gsm_cost:.4f}")

    # Full HumanEval
    print("\nLoading HumanEval full set...")
    he_full = load_humaneval(n=None)
    print(f"HumanEval: {len(he_full)} samples")

    reset_cost_tracking()
    t0 = time.time()
    he_result = fast_eval(winner, he_full, "humaneval")
    he_time = time.time() - t0
    he_cost = get_session_cost()
    print(f"HumanEval: {he_result['score']:.2f}% ({he_result['correct']}/{he_result['total']})")
    print(f"  Time: {he_time:.1f}s, Cost: ${he_cost:.4f}")

    avg = (gsm_result['score'] + he_result['score']) / 2
    print(f"\n{'═'*60}")
    print(f"FULL VALIDATION COMPLETE")
    print(f"  GSM8K (full): {gsm_result['score']:.2f}%")
    print(f"  HumanEval (full): {he_result['score']:.2f}%")
    print(f"  Average: {avg:.2f}%")
    print(f"  Total cost: ${gsm_cost + he_cost:.4f}")
    print(f"{'═'*60}")

    # Compare with SOTA
    print("\nComparison with published SOTA (gpt-4o-mini backbone):")
    print(f"  AutoMaAS: GSM8K=95.4% HE=97.2%")
    print(f"  AFlow:    GSM8K=93.5% HE=94.7%")
    print(f"  MaAS:     GSM8K=92.3% HE=92.9%")
    print(f"  Ours:     GSM8K={gsm_result['score']:.1f}% HE={he_result['score']:.1f}%")
    print(f"\nNote: We use gpt-5.4-nano, they use gpt-4o-mini. Models may differ.")


if __name__ == "__main__":
    main()
