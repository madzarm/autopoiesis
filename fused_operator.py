#!/usr/bin/env python3
"""Fused-Operator — Single-Call Agent with Compound Reasoning.

Approach 12: Instead of a multi-stage pipeline (generate → verify → repair → vote),
use a SINGLE compound LLM call that performs all stages internally.

Inspired by AutoMaAS's operator fusion: CoT+Self-Refine in one call yields
92% success rate and +4.2% improvement. The key insight is that maintaining
context across reasoning stages within a single call is more effective than
passing compressed outputs between separate calls.

The search space: different compound prompts that instruct the LLM to reason,
check, and correct within one response. We evolve these compound prompts.
"""

import json
import re
import random
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking
from evaluate import extract_number, load_gsm8k, load_humaneval
from genesis import Genome, Stage, fast_eval, _sanitize_code


# ═══════════════════════════════════════════════════════════════
# FUSED OPERATOR TEMPLATES
# ═══════════════════════════════════════════════════════════════

FUSED_PROMPTS = [
    # 1. Basic fused: reason + check
    """Solve this problem step by step.
After your solution, VERIFY your answer by checking the key steps.
If you find an error, correct it.
Put your final answer after ####.""",

    # 2. Code-augmented fused
    """Solve this problem two ways:
1. REASONING: Think step by step and get an answer.
2. CODE: Write Python code to solve it numerically.
3. COMPARE: If both methods agree, that's your answer.
   If they disagree, figure out which is correct.
Put your final answer after ####.""",

    # 3. Multi-perspective fused
    """Solve this problem using three different approaches:
Approach 1: Direct calculation step by step.
Approach 2: Work backwards from what the answer should look like.
Approach 3: Estimate the answer first, then verify.
RECONCILE: Pick the answer that at least 2 approaches agree on.
Put your final answer after ####.""",

    # 4. Error-anticipating fused
    """Solve this problem carefully.
Common mistakes for problems like this include:
- Arithmetic errors in multi-step calculations
- Misreading the question
- Off-by-one errors
- Forgetting to convert units
Be extra careful about these. Show your work.
Then double-check your final answer.
Put your final answer after ####.""",

    # 5. Universal fused (works for math AND code)
    """Solve this problem completely.
Think step by step. Show all your reasoning.
If this involves code, write clean, correct code.
If this involves math, verify your calculations.
Check your answer before finalizing.
Provide your final answer clearly.""",
]


def eval_fused(prompt_template: str, samples: list, benchmark: str, model: str = CHEAP) -> dict:
    """Evaluate a fused prompt on a benchmark."""
    correct = 0
    errors = []

    def _eval_single(idx_sample):
        idx, sample = idx_sample
        if benchmark == "gsm8k":
            problem = sample["question"]
            full_prompt = f"{prompt_template}\n\nProblem: {problem}"
            result = call_llm(prompt=full_prompt, system="Expert problem solver.",
                            model=model, temperature=0.0, max_tokens=3000)
            response = result["content"]
            predicted = extract_number(response)
            gold = sample["gold_answer"]
            is_correct = predicted is not None and abs(predicted - gold) < 1e-6
            return is_correct, {"problem": problem[:100], "gold": str(gold), "predicted": str(predicted)} if not is_correct else {}

        elif benchmark == "humaneval":
            prompt = sample["prompt"]
            full_prompt = f"Complete this Python function body:\n\n{prompt}\n\n{prompt_template}"
            result = call_llm(prompt=full_prompt, system="Expert Python programmer.",
                            model=model, temperature=0.0, max_tokens=2000)
            response = result["content"]
            body = _sanitize_code(response)
            full = sample["prompt"] + body + "\n" + sample["test"] + f"\ncheck({sample['entry_point']})"
            try:
                import threading
                ok = [False]
                def _run():
                    try:
                        exec(full, {"__builtins__": __builtins__})
                        ok[0] = True
                    except:
                        pass
                t = threading.Thread(target=_run)
                t.start()
                t.join(timeout=15)
                return ok[0], {}
            except:
                return False, {}
        return False, {}

    with ThreadPoolExecutor(max_workers=min(16, len(samples))) as ex:
        results = list(ex.map(_eval_single, enumerate(samples)))

    correct = sum(1 for r, _ in results if r)
    errors = [e for _, e in results if e]
    score = round(correct / len(samples) * 100, 2) if samples else 0
    return {"score": score, "correct": correct, "total": len(samples), "errors": errors[:5]}


def run_fused_operator(n_gsm=30, n_he=15, n_iterations=10, seed=42):
    """Search over fused operator prompts."""
    gsm_samples = load_gsm8k(n=n_gsm, seed=seed)
    he_samples = load_humaneval(n=n_he, seed=seed)

    print(f"═══ Fused-Operator ═══")
    print(f"GSM8K: {len(gsm_samples)}, HumanEval: {len(he_samples)}")
    print(f"Iterations: {n_iterations}\n")

    best_avg = 0.0
    best_prompt = ""

    # Eval seed prompts
    print("── Seeds ──")
    scored = []
    for i, prompt in enumerate(FUSED_PROMPTS):
        reset_cost_tracking()
        gsm_r = eval_fused(prompt, gsm_samples, "gsm8k")
        he_r = eval_fused(prompt, he_samples, "humaneval")
        avg = (gsm_r["score"] + he_r["score"]) / 2
        scored.append((prompt, avg, gsm_r, he_r))
        print(f"  seed_{i}: GSM={gsm_r['score']:5.1f}% HE={he_r['score']:5.1f}% avg={avg:.1f}%")
        if avg > best_avg:
            best_avg = avg
            best_prompt = prompt

    print(f"Best seed: {best_avg:.1f}%\n")

    # Evolve prompts
    print("── Search ──")
    for i in range(n_iterations):
        scored.sort(key=lambda x: x[1], reverse=True)
        parent_prompt = scored[0][0]
        parent_avg = scored[0][1]
        parent_gsm = scored[0][2]

        # LLM-guided prompt mutation
        error_str = "\n".join(f"- {e.get('problem','')[:80]}: gold={e.get('gold','')}, got={e.get('predicted','')}"
                              for e in parent_gsm.get("errors", [])[:3])

        mutation_prompt = f"""Improve this compound reasoning prompt (currently {parent_avg:.1f}% avg):

"{parent_prompt[:300]}"

Errors: {error_str or "None"}

The prompt instructs an LLM to solve problems in a SINGLE call — no multi-step pipeline.
It must work for BOTH math word problems AND code completion.
Key: the LLM should reason, check its work, and correct errors within ONE response.

Return ONLY the improved prompt text (no explanation).
"""
        result = call_llm(prompt=mutation_prompt, system="Prompt optimizer.",
                         model=STRONG, temperature=0.8, max_tokens=500)
        child_prompt = result["content"].strip().strip('"')

        # Evaluate child
        reset_cost_tracking()
        gsm_r = eval_fused(child_prompt, gsm_samples, "gsm8k")
        he_r = eval_fused(child_prompt, he_samples, "humaneval")
        avg = (gsm_r["score"] + he_r["score"]) / 2

        scored.append((child_prompt, avg, gsm_r, he_r))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:5]

        marker = ""
        if avg > best_avg:
            best_avg = avg
            best_prompt = child_prompt
            marker = " *** NEW BEST ***"

        if (i + 1) % 2 == 0 or marker:
            print(f"  [{i+1:3d}/{n_iterations}] GSM={gsm_r['score']:5.1f}% HE={he_r['score']:5.1f}% "
                  f"avg={avg:.1f}%{marker}")

    print(f"\n{'═'*50}")
    print(f"FUSED-OPERATOR COMPLETE")
    print(f"{'═'*50}")
    print(f"Best avg: {best_avg:.1f}%")
    print(f"Best prompt:\n{best_prompt[:300]}")

    return {"avg": best_avg, "prompt": best_prompt}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gsm", type=int, default=30)
    p.add_argument("--he", type=int, default=15)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()
    run_fused_operator(args.gsm, args.he, args.iters)
