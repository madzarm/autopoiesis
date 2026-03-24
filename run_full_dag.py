#!/usr/bin/env python3
"""Full test set validation of DAG-Evolve's best architecture with gpt-4o-mini."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from genesis import Genome, Stage, _eval_single_genesis
from llm import get_session_cost, reset_cost_tracking
from evaluate import load_gsm8k, load_humaneval

MODEL = "gpt-4o-mini"


def eval_with_progress(genome, samples, benchmark, max_workers=16):
    """Evaluate with real-time progress printing."""
    genome_dict = genome.to_dict()
    total = len(samples)
    correct = 0
    done = 0
    lock = threading.Lock()
    errors = []

    def _eval_and_count(args):
        nonlocal correct, done
        idx, sample = args
        result = _eval_single_genesis(genome_dict, sample, idx, benchmark)
        with lock:
            done += 1
            if result.get("correct"):
                correct += 1
            else:
                errors.append(result)
            if done % 50 == 0 or done == total:
                pct = correct / done * 100 if done > 0 else 0
                print(f"  Progress: {done}/{total} ({pct:.1f}% so far)", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_eval_and_count, (i, s)) for i, s in enumerate(samples)]
        for f in as_completed(futures):
            f.result()  # Raise any exceptions

    score = round(correct / total * 100, 2) if total > 0 else 0
    return {"score": score, "correct": correct, "total": total, "errors": errors[:5]}


def main():
    dag_best = Genome(name="dag_evolve_best", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Solve carefully and provide a clear answer."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure"),
        Stage(action="vote", condition="always"),
    ])

    # Also the simple CoT baseline (previously scored 94.16% on full GSM8K)
    cot_best = Genome(name="cot_baseline", model=MODEL, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Answer after ####."),
    ])

    designs = {"dag_evolve": dag_best, "cot_baseline": cot_best}

    print(f"═══ Full Test Set Validation (model={MODEL}) ═══\n", flush=True)

    for name, genome in designs.items():
        print(f"--- {name} ({len(genome.stages)} stages) ---", flush=True)

        # GSM8K full
        gsm = load_gsm8k(n=None)
        print(f"GSM8K: {len(gsm)} samples", flush=True)
        reset_cost_tracking()
        t0 = time.time()
        gsm_result = eval_with_progress(genome, gsm, "gsm8k")
        elapsed = time.time() - t0
        print(f"  FINAL: {gsm_result['score']:.2f}% ({gsm_result['correct']}/{gsm_result['total']})")
        print(f"  Time: {elapsed:.1f}s, Cost: ${get_session_cost():.4f}", flush=True)

        # HumanEval full
        he = load_humaneval(n=None)
        print(f"HumanEval: {len(he)} samples", flush=True)
        reset_cost_tracking()
        t0 = time.time()
        he_result = eval_with_progress(genome, he, "humaneval")
        elapsed = time.time() - t0
        print(f"  FINAL: {he_result['score']:.2f}% ({he_result['correct']}/{he_result['total']})")
        print(f"  Time: {elapsed:.1f}s, Cost: ${get_session_cost():.4f}", flush=True)

        avg = (gsm_result['score'] + he_result['score']) / 2
        print(f"  Average: {avg:.2f}%\n", flush=True)

    print("VALIDATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
