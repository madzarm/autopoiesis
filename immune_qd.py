#!/usr/bin/env python3
"""Immune-QD — Quality-Diversity Repertoire of Specialist Agents.

Approach 4: Inspired by the adaptive immune system. Instead of finding ONE best
agent, maintain a REPERTOIRE of diverse specialists indexed by behavioral
descriptors. At inference time, route each problem to the best-matching specialist.

Key novelty vs Genesis/DAG/MCTS:
- MAP-Elites style archive: maintain diversity along behavioral axes
- Problem-adaptive routing: different agents for different problem types
- Clonal selection: expand niches that perform well, mutate with error diagnosis
- Affinity maturation: iteratively improve specialists via targeted mutations

Biological analogy:
- Archive = immune repertoire (diverse antibodies)
- Behavioral descriptors = antigen epitopes
- Routing = antigen-antibody matching
- Mutation = somatic hypermutation (targeted, not random)
- Selection = clonal selection (amplify what works)
"""

import json
import re
import random
import copy
import time
import math
from dataclasses import dataclass, field
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking
from evaluate import extract_number, extract_math_answer, normalize_math_answer
from genesis import (
    Genome, Stage, execute_genome, fast_eval,
    prim_generate, prim_generate_code, prim_verify, prim_repair, prim_vote,
)


# ═══════════════════════════════════════════════════════════════
# BEHAVIORAL DESCRIPTORS — axes of the MAP-Elites archive
# ═══════════════════════════════════════════════════════════════

# We use 2 behavioral dimensions:
# 1. COST (number of LLM calls) — binned into: low (1), medium (2-3), high (4+)
# 2. STRATEGY TYPE — classified by what actions the genome uses:
#    "direct" (generate only), "verify" (generate + verify), "code" (uses code),
#    "ensemble" (uses vote/multiple generates), "full" (verify + repair + code)

COST_BINS = ["low", "medium", "high"]
STRATEGY_TYPES = ["direct", "verify", "code", "ensemble", "full"]


def classify_cost(genome: Genome) -> str:
    """Classify a genome's cost bin based on stage count."""
    n = len(genome.stages)
    if n <= 1:
        return "low"
    elif n <= 3:
        return "medium"
    else:
        return "high"


def classify_strategy(genome: Genome) -> str:
    """Classify a genome's strategy type based on its actions."""
    actions = set(s.action for s in genome.stages)
    has_code = "generate_code" in actions
    has_verify = "verify" in actions
    has_repair = "repair" in actions
    has_vote = "vote" in actions
    n_generates = sum(1 for s in genome.stages if s.action in ("generate", "generate_code"))

    if has_code and has_verify:
        return "full"
    elif has_vote or n_generates >= 3:
        return "ensemble"
    elif has_code:
        return "code"
    elif has_verify or has_repair:
        return "verify"
    else:
        return "direct"


def get_niche(genome: Genome) -> tuple:
    """Get the (cost_bin, strategy_type) niche for a genome."""
    return (classify_cost(genome), classify_strategy(genome))


# ═══════════════════════════════════════════════════════════════
# MAP-ELITES ARCHIVE
# ═══════════════════════════════════════════════════════════════

@dataclass
class ArchiveEntry:
    genome: Genome
    score: float
    niche: tuple
    errors: list = field(default_factory=list)
    eval_count: int = 0


class QDArchive:
    """MAP-Elites archive indexed by (cost_bin, strategy_type)."""

    def __init__(self):
        self.grid = {}  # (cost_bin, strategy_type) -> ArchiveEntry

    def try_add(self, genome: Genome, score: float, errors: list = None) -> bool:
        """Add genome if it's better than current occupant of its niche."""
        niche = get_niche(genome)
        errors = errors or []

        if niche not in self.grid or score > self.grid[niche].score:
            self.grid[niche] = ArchiveEntry(
                genome=genome, score=score, niche=niche,
                errors=errors, eval_count=1
            )
            return True
        return False

    def get_all(self) -> list:
        """Get all archive entries."""
        return list(self.grid.values())

    def get_best(self, n: int = 3) -> list:
        """Get top-N entries by score."""
        entries = sorted(self.grid.values(), key=lambda e: e.score, reverse=True)
        return entries[:n]

    def get_random_parent(self) -> ArchiveEntry:
        """Select a random entry, biased toward higher scores."""
        entries = list(self.grid.values())
        if not entries:
            return None
        # Fitness-proportionate selection
        scores = [max(e.score, 1.0) for e in entries]
        total = sum(scores)
        r = random.random() * total
        cumulative = 0
        for entry, s in zip(entries, scores):
            cumulative += s
            if cumulative >= r:
                return entry
        return entries[-1]

    def get_weakest_niche(self) -> tuple:
        """Get the niche with lowest score (or an empty niche)."""
        # Check empty niches first
        for cost in COST_BINS:
            for strat in STRATEGY_TYPES:
                if (cost, strat) not in self.grid:
                    return (cost, strat)
        # All occupied — return weakest
        return min(self.grid.items(), key=lambda x: x[1].score)[0]

    def coverage(self) -> float:
        """Fraction of niches filled."""
        total = len(COST_BINS) * len(STRATEGY_TYPES)
        return len(self.grid) / total

    def summary(self) -> str:
        lines = []
        for cost in COST_BINS:
            for strat in STRATEGY_TYPES:
                key = (cost, strat)
                if key in self.grid:
                    e = self.grid[key]
                    lines.append(f"  [{cost:6s}|{strat:8s}] {e.score:5.1f}% — {e.genome.name}")
                else:
                    lines.append(f"  [{cost:6s}|{strat:8s}] empty")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# GENOME GENERATORS — create diverse initial genomes
# ═══════════════════════════════════════════════════════════════

def _make_direct() -> Genome:
    """Single generate stage."""
    prompts = [
        "Think step by step. Answer after ####.",
        "Solve carefully. Answer after ####.",
        "You are an expert mathematician. Reason precisely. Answer after ####.",
    ]
    return Genome(name="direct", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt=random.choice(prompts)),
    ])


def _make_verify() -> Genome:
    """Generate + verify + repair."""
    return Genome(name="verify", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Answer after ####."),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure", temperature=0.1),
    ])


def _make_code() -> Genome:
    """Code-based solver."""
    return Genome(name="code", model=CHEAP, stages=[
        Stage(action="generate_code", temperature=0.0, condition="always"),
    ])


def _make_ensemble() -> Genome:
    """Multiple diverse generates + vote."""
    return Genome(name="ensemble", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Answer after ####."),
        Stage(action="generate", temperature=0.3, condition="always",
              system_prompt="Solve carefully. Answer after ####."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="vote", condition="always"),
    ])


def _make_full() -> Genome:
    """Generate + code + verify + repair + vote."""
    return Genome(name="full", model=CHEAP, stages=[
        Stage(action="generate", temperature=0.0, condition="always",
              system_prompt="Think step by step. Answer after ####."),
        Stage(action="generate_code", temperature=0.0, condition="always"),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure", temperature=0.1),
        Stage(action="vote", condition="always"),
    ])


SEED_GENERATORS = [_make_direct, _make_verify, _make_code, _make_ensemble, _make_full]


def _make_random() -> Genome:
    """Random genome."""
    n_stages = random.randint(1, 5)
    stages = []
    for _ in range(n_stages):
        action = random.choice(["generate", "generate_code", "verify", "repair", "vote"])
        temp = random.choice([0.0, 0.1, 0.3, 0.5])
        cond = random.choice(["always", "low_confidence", "after_failure"])
        prompts = [
            "Think step by step. Answer after ####.",
            "Solve carefully. Answer after ####.",
            "Be precise. Answer after ####.",
            "Break the problem into parts. Answer after ####.",
        ]
        stages.append(Stage(action=action, temperature=temp, condition=cond,
                           system_prompt=random.choice(prompts) if action == "generate" else ""))
    # Ensure at least one generate
    if not any(s.action in ("generate", "generate_code") for s in stages):
        stages.insert(0, Stage(action="generate", temperature=0.0, condition="always",
                               system_prompt="Think step by step. Answer after ####."))
    return Genome(name=f"rand_{random.randint(0,999)}", model=CHEAP, stages=stages)


# ═══════════════════════════════════════════════════════════════
# SOMATIC HYPERMUTATION — error-diagnosis-driven mutation
# ═══════════════════════════════════════════════════════════════

def mutate_genome(genome: Genome) -> Genome:
    """Random structural mutation of a genome."""
    new = Genome(name=genome.name + "_mut", model=genome.model,
                 stages=[copy.deepcopy(s) for s in genome.stages])

    op = random.choice(["add_stage", "remove_stage", "swap_stage", "modify_temp",
                        "modify_condition", "modify_prompt"])

    if op == "add_stage" and len(new.stages) < 7:
        action = random.choice(["generate", "generate_code", "verify", "repair", "vote"])
        pos = random.randint(0, len(new.stages))
        new.stages.insert(pos, Stage(
            action=action,
            temperature=random.choice([0.0, 0.1, 0.3]),
            condition=random.choice(["always", "low_confidence", "after_failure"]),
            system_prompt="Think step by step. Answer after ####." if action == "generate" else ""
        ))

    elif op == "remove_stage" and len(new.stages) > 1:
        idx = random.randint(0, len(new.stages) - 1)
        new.stages.pop(idx)

    elif op == "swap_stage" and len(new.stages) >= 2:
        i, j = random.sample(range(len(new.stages)), 2)
        new.stages[i], new.stages[j] = new.stages[j], new.stages[i]

    elif op == "modify_temp":
        idx = random.randint(0, len(new.stages) - 1)
        new.stages[idx].temperature = random.choice([0.0, 0.1, 0.3, 0.5, 0.7])

    elif op == "modify_condition":
        idx = random.randint(0, len(new.stages) - 1)
        new.stages[idx].condition = random.choice(["always", "low_confidence",
                                                    "after_failure", "disagreement"])

    elif op == "modify_prompt":
        gen_stages = [i for i, s in enumerate(new.stages) if s.action == "generate"]
        if gen_stages:
            idx = random.choice(gen_stages)
            prompts = [
                "Think step by step. Answer after ####.",
                "Solve carefully. Answer after ####.",
                "Be precise and methodical. Answer after ####.",
                "Break the problem into smaller parts. Answer after ####.",
                "You are an expert. Reason precisely. Answer after ####.",
            ]
            new.stages[idx].system_prompt = random.choice(prompts)

    # Ensure at least one generate
    if not any(s.action in ("generate", "generate_code") for s in new.stages):
        new.stages.insert(0, Stage(action="generate", temperature=0.0, condition="always",
                                   system_prompt="Think step by step. Answer after ####."))

    return new


def targeted_mutate(genome: Genome, errors: list, model: str = MID) -> Genome:
    """LLM-guided mutation based on error analysis (somatic hypermutation).

    Unlike random mutation, this analyzes WHY the agent failed and proposes
    targeted fixes — like how somatic hypermutation targets the antigen-binding region.
    """
    genome_json = json.dumps(genome.to_dict(), indent=2)
    error_str = "\n".join(
        f"- Problem: {e.get('problem', '')[:100]}\n  Gold: {e.get('gold', '')}\n  Got: {e.get('predicted', '')}"
        for e in errors[:3]
    )

    prompt = f"""This agent genome scored poorly on these problems:

GENOME:
{genome_json}

ERRORS:
{error_str}

Analyze the errors and propose a MODIFIED genome that fixes the failure patterns.

Rules:
- stages is a list of {{action, temperature, system_prompt, condition, condition_threshold, terminate_if_confident, confidence_threshold}}
- action: "generate", "generate_code", "verify", "repair", "vote"
- condition: "always", "low_confidence", "after_failure", "disagreement"
- Keep it to 1-6 stages max
- Must have at least one "generate" or "generate_code" stage
- Focus on STRUCTURAL changes (add/remove stages, change conditions), not just prompts

Return ONLY valid JSON of the genome (same format as above).
"""
    result = call_llm(prompt=prompt, system="Agent architect. Diagnose errors and evolve.",
                      model=model, temperature=0.7, max_tokens=2048, json_mode=True)
    try:
        data = json.loads(result["content"])
        data["model"] = CHEAP
        new = Genome.from_dict(data)
        new.name = genome.name + "_evolved"
        return new
    except Exception:
        return mutate_genome(genome)


# ═══════════════════════════════════════════════════════════════
# PROBLEM ROUTING — classify problems and route to best specialist
# ═══════════════════════════════════════════════════════════════

def classify_problem(problem: str) -> dict:
    """Quick heuristic classification of a problem's features."""
    text_lower = problem.lower()
    features = {
        "has_numbers": bool(re.search(r'\d+', problem)),
        "is_long": len(problem) > 300,
        "has_code": any(kw in text_lower for kw in ["code", "python", "function", "program"]),
        "has_steps": any(kw in text_lower for kw in ["step", "process", "each"]),
        "is_word_problem": any(kw in text_lower for kw in ["how many", "how much", "what is", "find"]),
        "complexity": "simple" if len(problem) < 150 else "medium" if len(problem) < 400 else "complex",
    }
    return features


def route_to_specialist(problem: str, archive: QDArchive) -> Genome:
    """Route a problem to the best specialist in the archive.

    Simple heuristic routing (can be evolved later):
    - Code problems → code specialist
    - Simple problems → direct specialist (save cost)
    - Complex problems → full or ensemble specialist
    """
    features = classify_problem(problem)
    entries = archive.get_all()
    if not entries:
        return _make_direct()

    # Score each specialist for this problem
    scored = []
    for entry in entries:
        niche_cost, niche_strat = entry.niche
        fit = entry.score  # Base fitness

        # Bonus for matching strategy to problem
        if features["has_code"] and niche_strat == "code":
            fit += 5
        if features["complexity"] == "simple" and niche_cost == "low":
            fit += 3
        if features["complexity"] == "complex" and niche_strat in ("full", "ensemble"):
            fit += 5
        if features["is_word_problem"] and niche_strat in ("verify", "ensemble"):
            fit += 3

        scored.append((entry, fit))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0].genome


# ═══════════════════════════════════════════════════════════════
# IMMUNE-QD SEARCH LOOP
# ═══════════════════════════════════════════════════════════════

def run_immune_qd(
    benchmark: str = "gsm8k",
    n_samples: int = 30,
    n_iterations: int = 40,
    seed: int = 42,
):
    """Run the Immune-QD quality-diversity search."""
    from evaluate import load_gsm8k, load_humaneval

    if benchmark == "gsm8k":
        samples = load_gsm8k(n=n_samples, seed=seed)
    elif benchmark == "humaneval":
        samples = load_humaneval(n=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown: {benchmark}")

    print(f"═══ Immune-QD ═══")
    print(f"Benchmark: {benchmark}, Samples: {len(samples)}, Iterations: {n_iterations}")
    print()

    archive = QDArchive()

    # Phase 1: INNATE — seed the archive with diverse specialists
    print("── Phase 1: Innate (seeding archive) ──")
    seed_genomes = [gen() for gen in SEED_GENERATORS]
    # Add a few random ones too
    for _ in range(3):
        seed_genomes.append(_make_random())

    def _eval_genome(genome):
        reset_cost_tracking()
        result = fast_eval(genome, samples, benchmark)
        cost = get_session_cost()
        return genome, result, cost

    # Eval all seeds in parallel
    with ThreadPoolExecutor(max_workers=len(seed_genomes)) as ex:
        futures = [ex.submit(_eval_genome, g) for g in seed_genomes]
        for f in as_completed(futures):
            genome, result, cost = f.result()
            added = archive.try_add(genome, result["score"], result.get("errors", []))
            niche = get_niche(genome)
            marker = " [ADDED]" if added else " [beaten]"
            print(f"  {genome.name:15s} | {result['score']:5.1f}% | "
                  f"niche={niche[0]:6s},{niche[1]:8s}{marker}")

    print(f"\nArchive coverage: {archive.coverage():.0%} ({len(archive.grid)}/{len(COST_BINS)*len(STRATEGY_TYPES)} niches)")
    print(f"Best: {archive.get_best(1)[0].score:.1f}%\n")

    # Phase 2: ADAPTIVE — targeted improvement with error-diagnosis
    print("── Phase 2: Adaptive (clonal selection + somatic hypermutation) ──")

    for i in range(n_iterations):
        # Decide what to do this iteration
        roll = random.random()

        if roll < 0.30:
            # CLONAL EXPANSION: mutate a strong agent
            parent_entry = archive.get_random_parent()
            if parent_entry and parent_entry.errors:
                child = targeted_mutate(parent_entry.genome, parent_entry.errors)
                method = "targeted"
            elif parent_entry:
                child = mutate_genome(parent_entry.genome)
                method = "mutate"
            else:
                child = _make_random()
                method = "random"

        elif roll < 0.55:
            # RANDOM EXPLORATION: try something new
            child = _make_random()
            method = "random"

        elif roll < 0.75:
            # NICHE TARGETING: create an agent for the weakest niche
            target_niche = archive.get_weakest_niche()
            child = _make_for_niche(target_niche)
            method = f"niche({target_niche[0]},{target_niche[1]})"

        else:
            # CROSSOVER: combine two archive entries
            entries = archive.get_all()
            if len(entries) >= 2:
                e1, e2 = random.sample(entries, 2)
                child = crossover_genomes(e1.genome, e2.genome)
                method = "crossover"
            else:
                child = _make_random()
                method = "random"

        child.name = f"iqd_{i+1}_{method[:5]}"
        child.model = CHEAP

        # Evaluate
        reset_cost_tracking()
        result = fast_eval(child, samples, benchmark)
        cost = get_session_cost()

        added = archive.try_add(child, result["score"], result.get("errors", []))
        niche = get_niche(child)

        marker = " *** NEW" if added else ""
        if (i + 1) % 5 == 0 or added:
            print(f"  [{i+1:3d}/{n_iterations}] [{method:15s}] {result['score']:5.1f}% | "
                  f"niche={niche[0]:6s},{niche[1]:8s} | "
                  f"archive={len(archive.grid)} niches{marker}")

    # Phase 3: EVALUATION — test routing across full archive
    print(f"\n── Final Archive ──")
    print(archive.summary())

    best = archive.get_best(1)[0]
    print(f"\nBest single agent: {best.genome.name} = {best.score:.1f}%")

    # Test routing performance
    print(f"\n── Routing Test ──")
    reset_cost_tracking()
    correct = 0
    total = len(samples)

    def _eval_routed(sample_idx_pair):
        idx, sample = sample_idx_pair
        if benchmark == "gsm8k":
            problem = sample.get("question", "")
            specialist = route_to_specialist(problem, archive)
            response = execute_genome(specialist, problem)
            predicted = extract_number(response)
            gold = sample["gold_answer"]
            is_correct = predicted is not None and abs(predicted - gold) < 1e-6
            return is_correct
        return False

    with ThreadPoolExecutor(max_workers=min(16, total)) as ex:
        results = list(ex.map(_eval_routed, list(enumerate(samples))))

    correct = sum(results)
    routed_score = round(correct / total * 100, 2)
    print(f"Routed ensemble: {routed_score}% ({correct}/{total})")
    print(f"Best single: {best.score:.1f}%")
    print(f"Improvement from routing: {routed_score - best.score:+.1f}%")

    # Save results
    print(f"\n{'═'*50}")
    print(f"IMMUNE-QD COMPLETE")
    print(f"{'═'*50}")
    print(f"Archive: {len(archive.grid)} niches filled ({archive.coverage():.0%} coverage)")
    print(f"Best single: {best.score:.1f}%")
    print(f"Routed: {routed_score}%")
    print(f"Cost: ${get_session_cost():.4f}")

    # Save best genome and archive
    with open("best_immune.json", "w") as f:
        json.dump(best.genome.to_dict(), f, indent=2)

    archive_data = {}
    for key, entry in archive.grid.items():
        archive_data[f"{key[0]}_{key[1]}"] = {
            "score": entry.score,
            "genome": entry.genome.to_dict(),
            "niche": list(key),
        }
    with open("immune_archive.json", "w") as f:
        json.dump(archive_data, f, indent=2)

    return {"best_score": best.score, "routed_score": routed_score,
            "coverage": archive.coverage(), "archive_size": len(archive.grid)}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _make_for_niche(target_niche: tuple) -> Genome:
    """Create a genome targeted at a specific niche."""
    cost_bin, strat = target_niche

    if strat == "direct":
        genome = _make_direct()
    elif strat == "verify":
        genome = _make_verify()
    elif strat == "code":
        genome = _make_code()
    elif strat == "ensemble":
        genome = _make_ensemble()
    elif strat == "full":
        genome = _make_full()
    else:
        genome = _make_random()

    # Adjust for cost target
    if cost_bin == "low" and len(genome.stages) > 1:
        genome.stages = genome.stages[:1]
    elif cost_bin == "high" and len(genome.stages) < 4:
        # Add more stages
        genome.stages.append(Stage(action="verify", condition="always"))
        genome.stages.append(Stage(action="repair", condition="after_failure"))

    return genome


def crossover_genomes(p1: Genome, p2: Genome) -> Genome:
    """Crossover two genomes by mixing their stages."""
    cut1 = random.randint(0, len(p1.stages))
    cut2 = random.randint(0, len(p2.stages))

    child_stages = [copy.deepcopy(s) for s in p1.stages[:cut1]]
    child_stages.extend([copy.deepcopy(s) for s in p2.stages[cut2:]])

    if not child_stages:
        child_stages = [copy.deepcopy(random.choice(p1.stages + p2.stages))]

    # Ensure at least one generate
    if not any(s.action in ("generate", "generate_code") for s in child_stages):
        child_stages.insert(0, Stage(action="generate", temperature=0.0, condition="always",
                                     system_prompt="Think step by step. Answer after ####."))

    return Genome(name=f"cross_{p1.name}_{p2.name}", model=p1.model,
                  stages=child_stages[:6])  # Cap at 6 stages


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--iters", type=int, default=40)
    args = p.parse_args()
    run_immune_qd(args.benchmark, args.n, args.iters)
