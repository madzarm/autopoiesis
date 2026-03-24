#!/usr/bin/env python3
"""LLM-Architect — Direct LLM-as-Search-Algorithm over Agent Design Space.

Approach 6: The meta-LLM directly proposes agent architectures based on:
1. A description of the task/benchmark
2. A history of what's been tried and how it scored
3. Error analysis from failed attempts

No population, no GP, no tree, no archive — the strong LLM IS the search algorithm.
This is closest to the original ADAS paper's Meta Agent Search, but with key differences:
- We search over structured configs (not raw code)
- We provide explicit error diagnosis
- We use a chain-of-reasoning: analyze errors → hypothesize cause → propose fix
- We track a "design journal" of insights accumulated across iterations

This tests whether pure LLM reasoning can match or beat structured search algorithms.
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
    prim_generate, prim_generate_code, prim_verify, prim_repair, prim_vote,
)


# ═══════════════════════════════════════════════════════════════
# DESIGN JOURNAL — accumulated insights across iterations
# ═══════════════════════════════════════════════════════════════

@dataclass
class DesignJournal:
    """Running log of design attempts and insights."""
    entries: list = field(default_factory=list)
    insights: list = field(default_factory=list)

    def add_attempt(self, genome: Genome, score: float, errors: list):
        self.entries.append({
            "genome": genome.to_dict(),
            "score": score,
            "n_stages": len(genome.stages),
            "actions": [s.action for s in genome.stages],
            "error_summary": [
                f"gold={e.get('gold','')}, got={e.get('predicted','')}"
                for e in errors[:3]
            ],
        })

    def add_insight(self, insight: str):
        self.insights.append(insight)

    def get_summary(self, last_n: int = 8) -> str:
        """Get a compact summary of recent history."""
        lines = []
        recent = self.entries[-last_n:]
        for i, e in enumerate(recent):
            actions = " → ".join(e["actions"])
            lines.append(f"  {i+1}. [{e['score']:5.1f}%] {actions}")
            if e["error_summary"]:
                lines.append(f"     Errors: {'; '.join(e['error_summary'][:2])}")

        insight_str = "\n".join(f"  - {ins}" for ins in self.insights[-5:])
        return f"Recent attempts:\n" + "\n".join(lines) + (
            f"\n\nAccumulated insights:\n{insight_str}" if self.insights else ""
        )


# ═══════════════════════════════════════════════════════════════
# LLM-AS-SEARCH: the meta-architect
# ═══════════════════════════════════════════════════════════════

ARCHITECT_SYSTEM = """You are an expert AI agent architect. Your job is to design agent pipelines
that solve problems effectively.

An agent pipeline is a sequence of STAGES. Each stage has:
- action: "generate" (LLM reasoning), "generate_code" (write+run Python), "verify" (check answer), "repair" (fix errors), "vote" (majority vote across candidates)
- temperature: 0.0-0.7 (for generate/generate_code)
- condition: "always" (always run), "low_confidence" (run if uncertain), "after_failure" (run if verify failed), "disagreement" (run if candidates disagree)
- system_prompt: instructions for generate stages (should end with 'Answer after ####.')

Design principles:
- generate_code is VERY powerful for math — it runs Python and captures output
- Multiple generates at different temperatures + vote = robust ensemble
- verify + repair is a self-correction loop
- More stages = higher cost but potentially better accuracy
- Conditions let you skip stages for easy problems (cost savings)
- The final answer comes from the last executed generate/vote/repair stage

Return JSON format:
{
  "name": "design_name",
  "model": "gpt-5.4-nano-2026-03-17",
  "stages": [
    {"action": "generate", "temperature": 0.0, "system_prompt": "...", "condition": "always",
     "condition_threshold": 0.7, "terminate_if_confident": false, "confidence_threshold": 0.9}
  ]
}
"""


def architect_propose(journal: DesignJournal, benchmark: str, best_score: float,
                      model: str = STRONG) -> Genome:
    """Use a strong LLM to propose a new agent design."""
    history = journal.get_summary()

    prompt = f"""BENCHMARK: {benchmark} (math word problems)
BEST SCORE SO FAR: {best_score:.1f}%
TARGET: Beat the best score.

{history}

Based on this history, analyze what patterns work and what doesn't.
Then propose a NEW agent design that you think will score higher.

Think step by step:
1. What error patterns do you see?
2. What structural changes might fix them?
3. Design the new pipeline.

Return ONLY the JSON genome (no explanation text outside JSON).
"""
    result = call_llm(prompt=prompt, system=ARCHITECT_SYSTEM,
                      model=model, temperature=0.7, max_tokens=2048, json_mode=True)
    try:
        data = json.loads(result["content"])
        data["model"] = CHEAP
        genome = Genome.from_dict(data)
        genome.name = data.get("name", f"arch_{len(journal.entries)}")
        return genome
    except Exception:
        return None


def architect_analyze(journal: DesignJournal, model: str = STRONG) -> str:
    """Use LLM to analyze patterns and generate insights."""
    history = journal.get_summary()
    prompt = f"""Analyze these agent design experiments:

{history}

What patterns do you see? What works, what doesn't? What should we try next?
Be specific and actionable. One paragraph, max 3 key insights.
"""
    result = call_llm(prompt=prompt, system="You analyze experiments and extract insights.",
                      model=model, temperature=0.3, max_tokens=512)
    return result["content"]


def architect_diverse_batch(journal: DesignJournal, benchmark: str,
                            best_score: float, n: int = 3,
                            model: str = STRONG) -> list:
    """Generate N diverse designs in parallel."""
    history = journal.get_summary()

    prompts = [
        f"Design a SIMPLE, CHEAP agent (1-2 stages max) for {benchmark}. "
        f"History:\n{history}\nBest: {best_score:.1f}%. Beat it with minimal cost. JSON only.",

        f"Design a COMPLEX, THOROUGH agent (4-6 stages) for {benchmark}. "
        f"Use ensembles, verification, code, repair. "
        f"History:\n{history}\nBest: {best_score:.1f}%. JSON only.",

        f"Design a CREATIVE, UNCONVENTIONAL agent for {benchmark}. "
        f"Try something DIFFERENT from all previous designs. "
        f"History:\n{history}\nBest: {best_score:.1f}%. JSON only.",
    ]

    genomes = []
    def _propose(prompt_text):
        result = call_llm(prompt=prompt_text, system=ARCHITECT_SYSTEM,
                          model=model, temperature=0.8, max_tokens=2048, json_mode=True)
        try:
            data = json.loads(result["content"])
            data["model"] = CHEAP
            return Genome.from_dict(data)
        except:
            return None

    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(_propose, p) for p in prompts[:n]]
        for f in as_completed(futures):
            g = f.result()
            if g:
                genomes.append(g)

    return genomes


# ═══════════════════════════════════════════════════════════════
# MAIN SEARCH LOOP
# ═══════════════════════════════════════════════════════════════

def run_llm_architect(
    benchmark: str = "gsm8k",
    n_samples: int = 30,
    n_iterations: int = 20,
    seed: int = 42,
):
    """Run LLM-as-architect search."""
    from evaluate import load_gsm8k, load_humaneval

    if benchmark == "gsm8k":
        samples = load_gsm8k(n=n_samples, seed=seed)
    elif benchmark == "humaneval":
        samples = load_humaneval(n=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown: {benchmark}")

    print(f"═══ LLM-Architect ═══")
    print(f"Benchmark: {benchmark}, Samples: {len(samples)}, Iterations: {n_iterations}")
    print(f"Meta-agent: {STRONG}, Inner agent: {CHEAP}")
    print()

    journal = DesignJournal()
    best_score = 0.0
    best_genome = None

    def _eval_genome(genome):
        reset_cost_tracking()
        result = fast_eval(genome, samples, benchmark)
        cost = get_session_cost()
        return genome, result, cost

    # Seed with basic designs
    print("── Seed Designs ──")
    seeds = [
        Genome(name="seed_cot", model=CHEAP, stages=[
            Stage(action="generate", temperature=0.0, condition="always",
                  system_prompt="Think step by step. Answer after ####."),
        ]),
        Genome(name="seed_code", model=CHEAP, stages=[
            Stage(action="generate_code", temperature=0.0, condition="always"),
        ]),
        Genome(name="seed_ensemble", model=CHEAP, stages=[
            Stage(action="generate", temperature=0.0, condition="always",
                  system_prompt="Think step by step. Answer after ####."),
            Stage(action="generate_code", temperature=0.0, condition="always"),
            Stage(action="vote", condition="always"),
        ]),
    ]

    with ThreadPoolExecutor(max_workers=len(seeds)) as ex:
        futures = [ex.submit(_eval_genome, g) for g in seeds]
        for f in as_completed(futures):
            genome, result, cost = f.result()
            journal.add_attempt(genome, result["score"], result.get("errors", []))
            if result["score"] > best_score:
                best_score = result["score"]
                best_genome = genome
            print(f"  {genome.name:20s} | {result['score']:5.1f}%")

    print(f"Best seed: {best_score:.1f}%\n")

    # Main loop: LLM proposes, we evaluate
    print("── LLM-Guided Search ──")
    for i in range(n_iterations):
        if (i + 1) % 5 == 0:
            # Every 5th iteration: analyze and generate insight
            insight = architect_analyze(journal)
            journal.add_insight(insight[:200])
            print(f"  [insight] {insight[:100]}...")

        if (i + 1) % 3 == 0:
            # Every 3rd: generate diverse batch
            genomes = architect_diverse_batch(journal, benchmark, best_score)
        else:
            # Normal: single proposal
            g = architect_propose(journal, benchmark, best_score)
            genomes = [g] if g else []

        # Evaluate all proposals in parallel
        if not genomes:
            continue

        for g in genomes:
            g.model = CHEAP
            if not g.name or g.name == "unnamed":
                g.name = f"arch_{i+1}"

        with ThreadPoolExecutor(max_workers=len(genomes)) as ex:
            futures = [ex.submit(_eval_genome, g) for g in genomes]
            for f in as_completed(futures):
                genome, result, cost = f.result()
                journal.add_attempt(genome, result["score"], result.get("errors", []))

                marker = ""
                if result["score"] > best_score:
                    best_score = result["score"]
                    best_genome = genome
                    marker = " *** NEW BEST ***"

                n_stages = len(genome.stages)
                actions = [s.action for s in genome.stages]
                print(f"  [{i+1:3d}/{n_iterations}] {result['score']:5.1f}% | "
                      f"{n_stages} stages: {' → '.join(actions)}{marker}")

    # Final
    print(f"\n{'═'*50}")
    print(f"LLM-ARCHITECT COMPLETE")
    print(f"{'═'*50}")
    print(f"Best score: {best_score}%")
    print(f"Best genome ({len(best_genome.stages)} stages):")
    for i, s in enumerate(best_genome.stages):
        cond = f" [if {s.condition}]" if s.condition != "always" else ""
        print(f"  {i+1}. {s.action}(t={s.temperature}){cond}")
        if s.system_prompt:
            print(f"     {s.system_prompt[:60]}")

    print(f"\nTotal designs tried: {len(journal.entries)}")
    print(f"Insights: {len(journal.insights)}")

    with open("best_architect.json", "w") as f:
        json.dump(best_genome.to_dict(), f, indent=2)

    return {"score": best_score, "genome": best_genome, "n_designs": len(journal.entries)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    run_llm_architect(args.benchmark, args.n, args.iters)
