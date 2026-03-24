#!/usr/bin/env python3
"""Code-Architect — LLM-as-Search for Code Workflow Discovery.

Different search algorithm from code_adas.py (evolutionary).
Here the LLM directly proposes, critiques, and refines workflows
based on error analysis. Uses a "design journal" to accumulate insights.

This is approach #2 for code-task ADAS.
"""

import json
import re
import ast
import time
import random
import threading
import io
import contextlib
from concurrent.futures import ThreadPoolExecutor
from code_adas import (
    CodeGenome, CodeStage, execute_code_genome, eval_genomes_parallel,
    sanitize_code, prim_test, load_humaneval_cached, HARDCODED_SOLUTIONS,
    CODE_ACTIONS, CODE_CONDITIONS, CODE_PROMPTS, SEED_GENOMES,
)
from llm import call_llm, get_session_cost, reset_cost_tracking

MODEL = "gpt-4o-mini"


def propose_workflows(journal: str, n: int = 3) -> list[CodeGenome]:
    """LLM proposes n diverse workflows based on the design journal."""
    prompt = f"""You are designing agent workflows for Python code generation (HumanEval benchmark).

## Design Journal (what we've learned so far)
{journal}

## Available Actions
- "generate": LLM generates code completion (params: system_prompt, temperature)
- "test": Execute code against test cases → pass/fail + error message
- "repair": LLM fixes code using test error message (params: temperature)
- "reflect": LLM reviews code for bugs before testing
- "select_passing": Test all candidates, succeed if any passes
- "restart": Generate fresh code with different strategy (params: temperature)

## Available Conditions
- "always": always runs
- "after_failure": only after a test failed
- "not_yet_passed": only if no candidate passed yet
- "has_candidates": only if candidates exist

## Task
Propose {n} DIVERSE workflows. Each should try a fundamentally different strategy.
Think about: how many candidates? when to test? when to repair? what prompts?

Return a JSON array of {n} workflow objects, each with format:
{{"name": "...", "model": "gpt-4o-mini", "max_candidates": 7, "stages": [...]}}

Keep workflows to 3-6 stages. More stages = slower evaluation."""

    result = call_llm(prompt=prompt, system="Expert agent architect. Return valid JSON array.",
                      model=MODEL, temperature=0.8, max_tokens=4096, json_mode=True)
    try:
        data = json.loads(result["content"])
        if isinstance(data, dict) and "workflows" in data:
            data = data["workflows"]
        if not isinstance(data, list):
            data = [data]
        genomes = []
        for d in data[:n]:
            d["model"] = MODEL
            g = CodeGenome.from_dict(d)
            if not any(s.action == "generate" for s in g.stages):
                g.stages.insert(0, CodeStage(action="generate", condition="always"))
            genomes.append(g)
        return genomes
    except Exception as e:
        print(f"  [propose failed: {e}]", flush=True)
        return []


def analyze_failures(genome: CodeGenome, failures: list[str],
                     samples: list[dict]) -> str:
    """Analyze why a workflow failed on specific problems."""
    # Get details of up to 3 failures
    fail_details = []
    for s in samples:
        if s["entry_point"] in failures[:3]:
            fail_details.append(f"- {s['entry_point']}: {s['prompt'][:150]}...")

    if not fail_details:
        return "No specific failure details available."

    prompt = (
        f"Workflow '{genome.name}' ({len(genome.stages)} stages) failed on:\n"
        + "\n".join(fail_details) +
        f"\n\nWorkflow: {json.dumps(genome.to_dict(), indent=1)}\n\n"
        f"Why might this workflow fail on these problems? Be brief (2-3 sentences)."
    )
    result = call_llm(prompt=prompt, system="Expert code analyst.",
                      model=MODEL, temperature=0.0, max_tokens=256)
    return result["content"]


def run_code_architect(
    n_samples: int = 20,
    iterations: int = 10,
    proposals_per_iter: int = 3,
    max_workers: int = 50,
):
    """LLM-as-search: propose → evaluate → analyze → refine."""
    samples = load_humaneval_cached(n=n_samples)
    print(f"\n{'═' * 70}", flush=True)
    print(f"  Code-Architect: LLM-as-Search for HumanEval Workflows", flush=True)
    print(f"  samples={n_samples}, iters={iterations}, proposals/iter={proposals_per_iter}", flush=True)
    print(f"{'═' * 70}\n", flush=True)

    journal_entries = [
        "Initial observations:",
        "- Simple generate + test + repair works well for most problems",
        "- More diverse candidates (different temps/prompts) catch more edge cases",
        "- Repair with actual error messages is very effective",
        "- The hardest problems involve tricky edge cases the model misunderstands",
    ]

    best_ever = None
    best_ever_score = 0.0
    all_results = {}  # name → {score, failures}

    # Start with seed genomes
    seeds = list(SEED_GENOMES[:5])
    print("Evaluating seeds...", flush=True)
    seed_results = eval_genomes_parallel(seeds, samples, max_workers=max_workers)
    for name, r in seed_results.items():
        all_results[name] = r
        print(f"  {name}: {r['score']}%", flush=True)
        if r["score"] > best_ever_score:
            best_ever_score = r["score"]
            best_ever = next(g for g in seeds if g.name == name)

    journal_entries.append(f"\nSeed results: " + ", ".join(
        f"{n}={r['score']}%" for n, r in sorted(seed_results.items(),
        key=lambda x: x[1]["score"], reverse=True)
    ))

    # Main loop: propose → evaluate → analyze → update journal
    for it in range(iterations):
        t0 = time.time()
        journal = "\n".join(journal_entries[-15:])  # Keep journal concise

        # Propose new workflows
        proposals = propose_workflows(journal, n=proposals_per_iter)
        if not proposals:
            continue
        for i, g in enumerate(proposals):
            g.name = f"arch_i{it}_{i}"

        # Evaluate all proposals at once
        prop_results = eval_genomes_parallel(proposals, samples, max_workers=max_workers)

        # Process results
        iter_best_name = None
        iter_best_score = 0
        for name, r in prop_results.items():
            all_results[name] = r
            if r["score"] > iter_best_score:
                iter_best_score = r["score"]
                iter_best_name = name
            if r["score"] > best_ever_score:
                best_ever_score = r["score"]
                best_ever = next(g for g in proposals if g.name == name)

        elapsed = time.time() - t0
        print(f"Iter {it:2d} ({elapsed:.0f}s): best={iter_best_name} {iter_best_score:.1f}% | "
              f"overall_best={best_ever_score:.1f}%", flush=True)

        # Analyze failures of best proposal
        if iter_best_name and prop_results[iter_best_name]["failures"]:
            best_proposal = next(g for g in proposals if g.name == iter_best_name)
            analysis = analyze_failures(
                best_proposal, prop_results[iter_best_name]["failures"], samples
            )
            journal_entries.append(
                f"\nIter {it}: {iter_best_name}={iter_best_score:.1f}%, "
                f"workflow={[s.action for s in best_proposal.stages]}, "
                f"analysis: {analysis[:200]}"
            )
        else:
            journal_entries.append(f"\nIter {it}: {iter_best_name}={iter_best_score:.1f}%")

    # Final summary
    print(f"\n{'═' * 70}", flush=True)
    print(f"  BEST DISCOVERED WORKFLOW (score={best_ever_score:.1f}%)", flush=True)
    print(f"{'═' * 70}", flush=True)
    if best_ever:
        print(best_ever.describe(), flush=True)
        with open("best_code_architect.json", "w") as f:
            json.dump(best_ever.to_dict(), f, indent=2)
        print(f"Saved to best_code_architect.json", flush=True)

    return best_ever, best_ever_score


if __name__ == "__main__":
    import sys
    reset_cost_tracking()
    t0 = time.time()

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    best, score = run_code_architect(n_samples=n, iterations=iters, max_workers=50)

    # Validate on full set
    if best:
        print(f"\n{'═' * 70}", flush=True)
        print(f"  FULL VALIDATION", flush=True)
        print(f"{'═' * 70}", flush=True)
        from code_adas import validate_genome
        validate_genome(best, n=None, max_workers=50)

    print(f"\nTotal: {time.time()-t0:.0f}s, Cost: ${get_session_cost():.2f}", flush=True)
