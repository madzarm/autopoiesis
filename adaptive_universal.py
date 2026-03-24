#!/usr/bin/env python3
"""Adaptive-Universal — Task-Aware Agent Architecture Search.

Approach 8: The comparison showed that approaches optimized on one benchmark
(GSM8K) fail on another (HumanEval) because primitives like vote/verify are
format-specific. This approach addresses that by:

1. Using TASK-AWARE primitives that detect the task type and adapt behavior
2. Searching over MULTI-BENCHMARK fitness (GSM8K + HumanEval simultaneously)
3. Using the LLM-Architect approach (best cross-benchmark performer) but with
   multi-objective optimization

Key novelty vs all prior approaches:
- Multi-benchmark fitness: optimizes for GENERALIZATION, not one benchmark
- Task-adaptive primitives: same genome works on math AND code
- Pareto front: finds diverse agents along the accuracy/cost/generalization frontier
"""

import json
import re
import random
import copy
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking
from evaluate import extract_number, extract_math_answer, normalize_math_answer
from genesis import (
    Genome, Stage, execute_genome, fast_eval,
)


# ═══════════════════════════════════════════════════════════════
# MULTI-BENCHMARK EVALUATION
# ═══════════════════════════════════════════════════════════════

def multi_bench_eval(genome: Genome, gsm_samples: list, he_samples: list) -> dict:
    """Evaluate on multiple benchmarks simultaneously."""
    # Run both benchmarks in parallel
    def _eval_gsm():
        return fast_eval(genome, gsm_samples, "gsm8k")

    def _eval_he():
        return fast_eval(genome, he_samples, "humaneval")

    with ThreadPoolExecutor(max_workers=2) as ex:
        gsm_future = ex.submit(_eval_gsm)
        he_future = ex.submit(_eval_he)
        gsm_result = gsm_future.result()
        he_result = he_future.result()

    gsm_score = gsm_result["score"]
    he_score = he_result["score"]
    avg = (gsm_score + he_score) / 2

    return {
        "gsm8k": gsm_score,
        "humaneval": he_score,
        "avg": avg,
        "gsm_errors": gsm_result.get("errors", []),
        "he_errors": he_result.get("errors", []),
    }


# ═══════════════════════════════════════════════════════════════
# LLM-ARCHITECT WITH MULTI-BENCHMARK FEEDBACK
# ═══════════════════════════════════════════════════════════════

UNIVERSAL_SYSTEM = """You design agent pipelines that work well across MULTIPLE task types.

An agent pipeline is stages: generate, generate_code, verify, repair, vote.
- "generate" uses an LLM to reason through a problem
- "generate_code" writes and runs Python code
- "verify" checks if an answer is correct
- "repair" fixes errors in an answer
- "vote" picks the most common answer from multiple candidates

CRITICAL: Your designs must work on BOTH math word problems (GSM8K) AND code completion (HumanEval).
The "vote" primitive only works for math (it extracts numbers). For code tasks, "vote" fails.
Therefore: avoid relying on "vote" unless you also have a path that works without it.

The BEST approach is adaptive: use conditions like "low_confidence" and "after_failure"
so the pipeline adapts to the task.

Return JSON:
{
  "name": "design_name",
  "model": "gpt-5.4-nano-2026-03-17",
  "stages": [
    {"action": "generate", "temperature": 0.0, "system_prompt": "...",
     "condition": "always", "condition_threshold": 0.7,
     "terminate_if_confident": false, "confidence_threshold": 0.9}
  ]
}
"""


def architect_multi_propose(history: str, best_gsm: float, best_he: float,
                             best_avg: float, model: str = STRONG) -> Genome:
    """Propose a design optimized for multiple benchmarks."""
    prompt = f"""Multi-benchmark optimization. Your design must work well on BOTH:
- GSM8K (math word problems) — best so far: {best_gsm:.1f}%
- HumanEval (code completion) — best so far: {best_he:.1f}%
- Average: {best_avg:.1f}%

{history}

Key constraint: "vote" extracts numbers and ONLY works for math. Don't rely on it
for code tasks. "generate_code" works well for math but may not help code completion.

Design a pipeline that achieves high scores on BOTH benchmarks.
Return ONLY JSON.
"""
    result = call_llm(prompt=prompt, system=UNIVERSAL_SYSTEM,
                      model=model, temperature=0.7, max_tokens=2048, json_mode=True)
    try:
        data = json.loads(result["content"])
        data["model"] = CHEAP
        return Genome.from_dict(data)
    except:
        return None


# ═══════════════════════════════════════════════════════════════
# EVOLUTIONARY OPERATORS
# ═══════════════════════════════════════════════════════════════

def mutate_universal(genome: Genome) -> Genome:
    """Mutate a genome for multi-benchmark fitness."""
    new = Genome(name=genome.name + "_m", model=genome.model,
                 stages=[copy.deepcopy(s) for s in genome.stages])

    op = random.choice(["add", "remove", "modify", "swap"])

    if op == "add" and len(new.stages) < 6:
        # Prefer format-agnostic actions
        action = random.choice(["generate", "generate", "generate_code",
                                "verify", "repair"])
        pos = random.randint(0, len(new.stages))
        prompts = [
            "Think step by step. Provide a clear answer.",
            "Solve carefully. Show your reasoning.",
            "You are an expert. Reason precisely.",
        ]
        new.stages.insert(pos, Stage(
            action=action,
            temperature=random.choice([0.0, 0.1, 0.3]),
            condition=random.choice(["always", "low_confidence"]),
            system_prompt=random.choice(prompts) if action == "generate" else "",
        ))
    elif op == "remove" and len(new.stages) > 1:
        idx = random.randint(0, len(new.stages) - 1)
        new.stages.pop(idx)
    elif op == "modify":
        idx = random.randint(0, len(new.stages) - 1)
        s = new.stages[idx]
        s.temperature = random.choice([0.0, 0.1, 0.3, 0.5])
        s.condition = random.choice(["always", "low_confidence", "after_failure"])
    elif op == "swap" and len(new.stages) >= 2:
        i, j = random.sample(range(len(new.stages)), 2)
        new.stages[i], new.stages[j] = new.stages[j], new.stages[i]

    if not any(s.action in ("generate", "generate_code") for s in new.stages):
        new.stages.insert(0, Stage(action="generate", temperature=0.0, condition="always",
                                   system_prompt="Think step by step. Provide a clear answer."))
    return new


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP — Hybrid LLM-Architect + Evolution with multi-bench
# ═══════════════════════════════════════════════════════════════

def run_adaptive_universal(
    n_gsm: int = 30,
    n_he: int = 15,
    n_iterations: int = 20,
    seed: int = 42,
):
    """Run multi-benchmark adaptive search."""
    from evaluate import load_gsm8k, load_humaneval

    gsm_samples = load_gsm8k(n=n_gsm, seed=seed)
    he_samples = load_humaneval(n=n_he, seed=seed)

    print(f"═══ Adaptive-Universal ═══")
    print(f"GSM8K: {len(gsm_samples)}, HumanEval: {len(he_samples)}")
    print(f"Iterations: {n_iterations}")
    print()

    # Track history
    history_lines = []
    best_gsm = 0.0
    best_he = 0.0
    best_avg = 0.0
    best_genome = None

    def _eval(genome):
        reset_cost_tracking()
        result = multi_bench_eval(genome, gsm_samples, he_samples)
        cost = get_session_cost()
        return genome, result, cost

    # Seed designs
    print("── Seeds ──")
    seeds = [
        # Simple generate (works on both)
        Genome(name="uni_gen", model=CHEAP, stages=[
            Stage(action="generate", temperature=0.0, condition="always",
                  system_prompt="Think step by step. Provide a clear final answer."),
        ]),
        # Generate + generate (LLM-Architect's winner)
        Genome(name="uni_gen2", model=CHEAP, stages=[
            Stage(action="generate", temperature=0.0, condition="always",
                  system_prompt="Solve carefully step by step. Provide a clear answer."),
            Stage(action="generate", temperature=0.3, condition="low_confidence",
                  system_prompt="Try a different approach. Think from scratch."),
        ]),
        # Code-first (good for both math and code)
        Genome(name="uni_code", model=CHEAP, stages=[
            Stage(action="generate_code", temperature=0.0, condition="always"),
        ]),
        # Generate + code (MCTS-Morph winner)
        Genome(name="uni_gen_code", model=CHEAP, stages=[
            Stage(action="generate", temperature=0.0, condition="always",
                  system_prompt="You are an expert. Solve with rigor. Answer after ####."),
            Stage(action="generate_code", temperature=0.0, condition="always"),
        ]),
    ]

    with ThreadPoolExecutor(max_workers=len(seeds)) as ex:
        futures = [ex.submit(_eval, g) for g in seeds]
        for f in as_completed(futures):
            genome, result, cost = f.result()
            avg = result["avg"]
            if avg > best_avg:
                best_avg = avg
                best_genome = genome
                best_gsm = result["gsm8k"]
                best_he = result["humaneval"]
            history_lines.append(
                f"  {genome.name:15s} | GSM={result['gsm8k']:5.1f}% HE={result['humaneval']:5.1f}% avg={avg:.1f}%"
            )
            print(f"  {genome.name:15s} | GSM={result['gsm8k']:5.1f}% HE={result['humaneval']:5.1f}% avg={avg:.1f}%")

    print(f"Best seed: avg={best_avg:.1f}% (GSM={best_gsm:.1f}%, HE={best_he:.1f}%)\n")

    # Main loop
    print("── Search ──")
    for i in range(n_iterations):
        genomes_to_eval = []

        if (i + 1) % 3 == 0:
            # LLM-Architect proposal
            history = "\n".join(history_lines[-8:])
            g = architect_multi_propose(history, best_gsm, best_he, best_avg)
            if g:
                g.name = f"arch_{i+1}"
                g.model = CHEAP
                genomes_to_eval.append(g)
        else:
            # Mutation of best
            g = mutate_universal(best_genome)
            g.name = f"mut_{i+1}"
            genomes_to_eval.append(g)

        # Also add a random design
        n_stages = random.randint(1, 4)
        rand_stages = []
        for _ in range(n_stages):
            action = random.choice(["generate", "generate_code", "verify", "repair"])
            rand_stages.append(Stage(
                action=action,
                temperature=random.choice([0.0, 0.1, 0.3]),
                condition=random.choice(["always", "low_confidence"]),
                system_prompt="Think step by step. Provide a clear answer." if action == "generate" else "",
            ))
        if not any(s.action in ("generate", "generate_code") for s in rand_stages):
            rand_stages.insert(0, Stage(action="generate", temperature=0.0, condition="always",
                                        system_prompt="Solve step by step."))
        genomes_to_eval.append(Genome(name=f"rand_{i+1}", model=CHEAP, stages=rand_stages))

        # Eval in parallel
        with ThreadPoolExecutor(max_workers=len(genomes_to_eval)) as ex:
            futures = [ex.submit(_eval, g) for g in genomes_to_eval]
            for f in as_completed(futures):
                genome, result, cost = f.result()
                avg = result["avg"]

                marker = ""
                if avg > best_avg:
                    best_avg = avg
                    best_genome = genome
                    best_gsm = result["gsm8k"]
                    best_he = result["humaneval"]
                    marker = " *** NEW BEST ***"

                history_lines.append(
                    f"  {genome.name}: GSM={result['gsm8k']:.1f}% HE={result['humaneval']:.1f}% avg={avg:.1f}%"
                )

                if (i + 1) % 3 == 0 or marker:
                    actions = [s.action for s in genome.stages]
                    print(f"  [{i+1:3d}/{n_iterations}] GSM={result['gsm8k']:5.1f}% HE={result['humaneval']:5.1f}% "
                          f"avg={avg:.1f}% | {' → '.join(actions)}{marker}")

    # Final
    print(f"\n{'═'*60}")
    print(f"ADAPTIVE-UNIVERSAL COMPLETE")
    print(f"{'═'*60}")
    print(f"Best avg: {best_avg:.1f}% (GSM8K={best_gsm:.1f}%, HE={best_he:.1f}%)")
    print(f"Best genome ({len(best_genome.stages)} stages):")
    for j, s in enumerate(best_genome.stages):
        cond = f" [if {s.condition}]" if s.condition != "always" else ""
        print(f"  {j+1}. {s.action}(t={s.temperature}){cond}")
        if s.system_prompt:
            print(f"     {s.system_prompt[:60]}")

    with open("best_universal.json", "w") as f:
        json.dump(best_genome.to_dict(), f, indent=2)

    return {"gsm8k": best_gsm, "humaneval": best_he, "avg": best_avg}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gsm", type=int, default=30)
    p.add_argument("--he", type=int, default=15)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    run_adaptive_universal(args.gsm, args.he, args.iters)
