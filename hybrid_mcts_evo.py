#!/usr/bin/env python3
"""Hybrid-MCTS-Evo — Tree Search for Structure + Evolution for Parameters.

Approach 7: Cross-pollination of MCTS-Morph and Genesis.

Key insight: MCTS is good at exploring STRUCTURAL decisions (what stages to include)
but bad at optimizing PARAMETERS within a structure (temperatures, prompts).
Evolution is good at parameter optimization but wastes time exploring bad structures.

Solution: Two-level search:
1. OUTER: MCTS over structural decisions (which actions to include)
2. INNER: Mini-evolution to optimize parameters within the chosen structure

This addresses the weaknesses of both:
- MCTS-Morph got stuck at 93.3% because rollouts with random parameters were noisy
- Genesis got to 96.7% but took 15 generations over a flat population

The hybrid should converge faster and explore more effectively.
"""

import json
import math
import random
import copy
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from genesis import (
    Genome, Stage, execute_genome, fast_eval,
    prim_generate, prim_generate_code, prim_verify, prim_repair, prim_vote,
    CHEAP,
)
from llm import call_llm, MID, get_session_cost, reset_cost_tracking


# ═══════════════════════════════════════════════════════════════
# STRUCTURAL DECISIONS (MCTS level)
# ═══════════════════════════════════════════════════════════════

# Each structural decision adds an action type to the pipeline
STRUCTURAL_ACTIONS = [
    "generate",       # Add a generate stage
    "generate_code",  # Add a code generation stage
    "verify",         # Add a verification stage
    "repair",         # Add a repair stage
    "vote",           # Add a vote/merge stage
    "DONE",           # Mark design as complete
]


@dataclass
class HybridNode:
    """MCTS node representing a structural decision."""
    actions: list = field(default_factory=list)  # list of action type strings
    visits: int = 0
    total_score: float = 0.0
    best_score: float = 0.0
    best_genome: object = None  # Best evolved genome for this structure
    children: dict = field(default_factory=dict)  # action_name -> HybridNode
    parent: object = None
    decision: str = ""
    is_terminal: bool = False

    @property
    def avg_score(self):
        return self.total_score / self.visits if self.visits > 0 else 0.0

    @property
    def depth(self):
        d = 0
        node = self.parent
        while node:
            d += 1
            node = node.parent
        return d


# ═══════════════════════════════════════════════════════════════
# INNER EVOLUTION — optimize parameters for a given structure
# ═══════════════════════════════════════════════════════════════

PROMPTS = [
    "Think step by step. Answer after ####.",
    "Solve carefully and check your work. Answer after ####.",
    "You are a world-class mathematician. Solve with rigor. Answer after ####.",
    "Break the problem into parts. Solve each part. Answer after ####.",
    "Be precise. Show your reasoning. Answer after ####.",
]


def structure_to_genome(actions: list, name: str = "hybrid") -> Genome:
    """Convert a list of action types into a Genome with random parameters."""
    stages = []
    for action in actions:
        if action == "generate":
            stages.append(Stage(
                action="generate",
                temperature=random.choice([0.0, 0.1, 0.3]),
                condition=random.choice(["always", "low_confidence"]),
                system_prompt=random.choice(PROMPTS),
            ))
        elif action == "generate_code":
            stages.append(Stage(
                action="generate_code",
                temperature=random.choice([0.0, 0.1, 0.3]),
                condition="always",
            ))
        elif action == "verify":
            stages.append(Stage(
                action="verify",
                condition=random.choice(["always", "low_confidence"]),
            ))
        elif action == "repair":
            stages.append(Stage(
                action="repair",
                temperature=random.choice([0.0, 0.1]),
                condition=random.choice(["after_failure", "always"]),
            ))
        elif action == "vote":
            stages.append(Stage(action="vote", condition="always"))
    return Genome(name=name, model=CHEAP, stages=stages)


def mini_evolve(actions: list, samples: list, benchmark: str,
                pop_size: int = 5, gens: int = 3) -> tuple:
    """Mini evolution to optimize parameters for a given structure.

    Returns (best_score, best_genome)
    """
    # Generate initial population
    population = []
    for i in range(pop_size):
        genome = structure_to_genome(actions, f"evo_{i}")
        result = fast_eval(genome, samples, benchmark)
        population.append((genome, result["score"]))

    population.sort(key=lambda x: x[1], reverse=True)
    best_genome, best_score = population[0]

    for gen in range(gens):
        # Keep top 2
        parents = population[:2]
        new_pop = list(parents)

        # Generate children by mutating parameters
        for _ in range(pop_size - 2):
            parent = random.choice(parents)[0]
            child = _mutate_params(parent)
            child.name = f"evo_g{gen}_{random.randint(0,99)}"
            result = fast_eval(child, samples, benchmark)
            new_pop.append((child, result["score"]))

        new_pop.sort(key=lambda x: x[1], reverse=True)
        population = new_pop[:pop_size]

        if population[0][1] > best_score:
            best_genome, best_score = population[0]

    return best_score, best_genome


def _mutate_params(genome: Genome) -> Genome:
    """Mutate parameters of a genome (without changing structure)."""
    new = Genome(name=genome.name + "_m", model=genome.model,
                 stages=[copy.deepcopy(s) for s in genome.stages])

    idx = random.randint(0, len(new.stages) - 1)
    stage = new.stages[idx]

    roll = random.random()
    if roll < 0.4 and stage.action == "generate":
        stage.system_prompt = random.choice(PROMPTS)
    elif roll < 0.7:
        stage.temperature = random.choice([0.0, 0.1, 0.3, 0.5])
    else:
        stage.condition = random.choice(["always", "low_confidence", "after_failure"])

    return new


# ═══════════════════════════════════════════════════════════════
# OUTER MCTS — explores structural decisions
# ═══════════════════════════════════════════════════════════════

def ucb1(node: HybridNode, parent_visits: int, C: float = 1.414) -> float:
    if node.visits == 0:
        return float('inf')
    exploitation = node.avg_score / 100.0
    exploration = C * math.sqrt(math.log(parent_visits) / node.visits)
    return exploitation + exploration


def select(root: HybridNode) -> HybridNode:
    node = root
    while node.children and not node.is_terminal:
        # If unexpanded actions exist, return for expansion
        possible = [a for a in STRUCTURAL_ACTIONS if a not in node.children]
        # Don't allow structures longer than 6
        if node.depth >= 5:
            possible = ["DONE"] if "DONE" not in node.children else []
        if possible:
            return node
        node = max(node.children.values(), key=lambda c: ucb1(c, node.visits))
    return node


def expand(node: HybridNode) -> HybridNode:
    possible = [a for a in STRUCTURAL_ACTIONS if a not in node.children]
    if node.depth >= 5:
        possible = ["DONE"] if "DONE" not in node.children else []
    if not possible:
        return node

    action = random.choice(possible)
    child_actions = node.actions + ([action] if action != "DONE" else [])

    child = HybridNode(
        actions=child_actions,
        parent=node,
        decision=action,
        is_terminal=(action == "DONE" or node.depth >= 4),
    )
    node.children[action] = child
    return child


def backpropagate(node: HybridNode, score: float, genome=None):
    while node is not None:
        node.visits += 1
        node.total_score += score
        if score > node.best_score:
            node.best_score = score
            if genome:
                node.best_genome = genome
        node = node.parent


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def run_hybrid(
    benchmark: str = "gsm8k",
    n_samples: int = 30,
    n_outer_iters: int = 25,
    inner_pop: int = 4,
    inner_gens: int = 2,
    seed: int = 42,
):
    """Run Hybrid MCTS-Evo search."""
    from evaluate import load_gsm8k, load_humaneval

    if benchmark == "gsm8k":
        samples = load_gsm8k(n=n_samples, seed=seed)
    elif benchmark == "humaneval":
        samples = load_humaneval(n=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown: {benchmark}")

    print(f"═══ Hybrid-MCTS-Evo ═══")
    print(f"Benchmark: {benchmark}, Samples: {len(samples)}")
    print(f"Outer: {n_outer_iters} MCTS iterations")
    print(f"Inner: pop={inner_pop}, gens={inner_gens} per structure")
    print()

    root = HybridNode()
    best_score = 0.0
    best_genome = None

    for i in range(n_outer_iters):
        # 1. SELECT
        node = select(root)

        # 2. EXPAND
        child = expand(node)

        actions = child.actions
        if not actions:
            # Root node — add a generate as minimum
            actions = ["generate"]

        # Ensure at least one generate
        if not any(a in ("generate", "generate_code") for a in actions):
            actions = ["generate"] + actions

        # 3. INNER EVOLUTION — optimize parameters for this structure
        reset_cost_tracking()
        score, genome = mini_evolve(actions, samples, benchmark, inner_pop, inner_gens)

        # 4. BACKPROPAGATE
        backpropagate(child, score, genome)

        marker = ""
        if score > best_score:
            best_score = score
            best_genome = genome
            marker = " *** NEW BEST ***"

        if (i + 1) % 3 == 0 or marker:
            struct = " → ".join(actions) if actions else "(empty)"
            print(f"  [{i+1:3d}/{n_outer_iters}] score={score:5.1f}% | "
                  f"structure: {struct} | best={best_score:.1f}%{marker}")

    # Final
    print(f"\n{'═'*50}")
    print(f"HYBRID-MCTS-EVO COMPLETE")
    print(f"{'═'*50}")
    print(f"Best score: {best_score}%")
    print(f"Best genome ({len(best_genome.stages)} stages):")
    for j, s in enumerate(best_genome.stages):
        cond = f" [if {s.condition}]" if s.condition != "always" else ""
        print(f"  {j+1}. {s.action}(t={s.temperature}){cond}")
        if s.system_prompt:
            print(f"     {s.system_prompt[:60]}")

    # Save
    with open("best_hybrid.json", "w") as f:
        json.dump(best_genome.to_dict(), f, indent=2)

    return {"score": best_score, "genome": best_genome}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--iters", type=int, default=25)
    args = p.parse_args()
    run_hybrid(args.benchmark, args.n, args.iters)
