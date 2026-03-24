#!/usr/bin/env python3
"""MCTS-Morph — Monte Carlo Tree Search over Morphological Space.

Approach 3: Instead of evolving a population (Genesis) or graph (DAG-Evolve),
MCTS-Morph builds a TREE of design decisions. Each path from root to leaf
is a sequence of choices: "add CoT" → "add verification" → "add code" → ...

Key novelty vs AFlow: AFlow MCTS nodes are complete workflows. Our nodes are
PARTIAL designs — the tree represents the DECISION space, not the design space.

Key novelty vs Genesis: Genesis evolves flat populations. MCTS naturally balances
explore/exploit via UCB1 and remembers what was tried in each branch.
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
    SYSTEM_PROMPTS, ACTIONS, CHEAP,
)
from llm import call_llm, MID, get_session_cost, reset_cost_tracking


# ═══════════════════════════════════════════════════════════════
# MCTS TREE NODES — each represents a design decision
# ═══════════════════════════════════════════════════════════════

@dataclass
class MCTSNode:
    """A node in the MCTS tree. Represents a partial agent design."""
    # The design so far (stages accumulated along the path from root)
    stages: list = field(default_factory=list)  # list of Stage objects
    # MCTS statistics
    visits: int = 0
    total_score: float = 0.0
    # Tree structure
    children: list = field(default_factory=list)  # list of MCTSNode
    parent: object = None  # MCTSNode or None
    # What decision was made to reach this node
    decision: str = ""  # e.g., "add_generate_t0", "add_verify", "add_code"
    # Is this a terminal node (complete design)?
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
# DESIGN DECISIONS — the action space for expanding MCTS nodes
# ═══════════════════════════════════════════════════════════════

DESIGN_DECISIONS = [
    # Generate variants
    ("add_generate_t0", Stage(action="generate", temperature=0.0, condition="always",
                              system_prompt="Think step by step. Answer after ####.")),
    ("add_generate_t03", Stage(action="generate", temperature=0.3, condition="always",
                               system_prompt="Solve carefully. Answer after ####.")),
    ("add_generate_t06", Stage(action="generate", temperature=0.6, condition="always",
                               system_prompt="Be creative. Answer after ####.")),
    ("add_generate_expert", Stage(action="generate", temperature=0.0, condition="always",
                                   system_prompt="You are a world-class mathematician. Solve with rigor. Answer after ####.")),
    # Code generation
    ("add_code", Stage(action="generate_code", temperature=0.0, condition="always")),
    ("add_code_t03", Stage(action="generate_code", temperature=0.3, condition="always")),
    # Verification
    ("add_verify", Stage(action="verify", condition="always")),
    ("add_verify_low_conf", Stage(action="verify", condition="low_confidence", condition_threshold=0.7)),
    # Repair
    ("add_repair", Stage(action="repair", condition="after_failure", temperature=0.1)),
    ("add_repair_disagree", Stage(action="repair", condition="disagreement", temperature=0.0)),
    # Vote/merge
    ("add_vote", Stage(action="vote", condition="always")),
    # Early termination
    ("add_stop_confident", Stage(action="generate", temperature=0.0, condition="always",
                                  system_prompt="Think step by step. Answer after ####.",
                                  terminate_if_confident=True, confidence_threshold=0.95)),
    # Terminal — mark design as complete
    ("DONE", None),
]


# ═══════════════════════════════════════════════════════════════
# MCTS ALGORITHM
# ═══════════════════════════════════════════════════════════════

def ucb1(node: MCTSNode, parent_visits: int, C: float = 1.414) -> float:
    """Upper Confidence Bound for Trees."""
    if node.visits == 0:
        return float('inf')
    exploitation = node.avg_score / 100.0  # Normalize to [0,1]
    exploration = C * math.sqrt(math.log(parent_visits) / node.visits)
    return exploitation + exploration


def select(root: MCTSNode) -> MCTSNode:
    """Select a leaf node using UCB1."""
    node = root
    while node.children and not node.is_terminal:
        # If there are unexpanded decisions, return this node for expansion
        if len(node.children) < len(DESIGN_DECISIONS):
            return node
        # Otherwise, select best child by UCB1
        node = max(node.children, key=lambda c: ucb1(c, node.visits))
    return node


def expand(node: MCTSNode) -> MCTSNode:
    """Expand a node by adding a new child (unexplored decision)."""
    # Find which decisions haven't been tried yet
    tried = {c.decision for c in node.children}
    untried = [(name, stage) for name, stage in DESIGN_DECISIONS if name not in tried]

    if not untried:
        return node  # Fully expanded

    # Don't allow designs longer than 7 stages
    if len(node.stages) >= 7:
        # Only allow DONE
        untried = [(n, s) for n, s in untried if n == "DONE"]
        if not untried:
            return node

    name, stage = random.choice(untried)

    child = MCTSNode(
        stages=node.stages + ([copy.deepcopy(stage)] if stage else []),
        parent=node,
        decision=name,
        is_terminal=(name == "DONE" or len(node.stages) >= 6),
    )
    node.children.append(child)
    return child


def rollout(node: MCTSNode, samples: list, benchmark: str) -> float:
    """Evaluate a (possibly partial) design by completing it randomly and testing."""
    stages = list(node.stages)

    # If not terminal, add random stages to complete it
    if not node.is_terminal:
        n_more = random.randint(0, 3)
        for _ in range(n_more):
            name, stage = random.choice(DESIGN_DECISIONS[:-1])  # Exclude DONE
            if stage:
                stages.append(copy.deepcopy(stage))

    # Ensure at least one generate stage
    if not any(s.action in ("generate", "generate_code") for s in stages):
        stages.insert(0, Stage(action="generate", condition="always",
                               system_prompt="Think step by step. Answer after ####."))

    genome = Genome(name="rollout", model=CHEAP, stages=stages)
    result = fast_eval(genome, samples, benchmark)
    return result["score"]


def backpropagate(node: MCTSNode, score: float):
    """Propagate score up the tree."""
    while node is not None:
        node.visits += 1
        node.total_score += score
        node = node.parent


def run_mcts_morph(
    benchmark: str = "gsm8k",
    n_samples: int = 20,
    n_iterations: int = 60,
    seed: int = 42,
):
    """Run MCTS over design decision space."""
    from evaluate import load_gsm8k, load_humaneval

    if benchmark == "gsm8k":
        samples = load_gsm8k(n=n_samples, seed=seed)
    elif benchmark == "humaneval":
        samples = load_humaneval(n=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown: {benchmark}")

    print(f"═══ MCTS-Morph ═══")
    print(f"Benchmark: {benchmark}, Samples: {len(samples)}, Iterations: {n_iterations}")
    print()

    root = MCTSNode()
    best_score = 0.0
    best_stages = []

    for i in range(n_iterations):
        # 1. SELECT
        node = select(root)

        # 2. EXPAND
        child = expand(node)

        # 3. ROLLOUT (evaluate)
        reset_cost_tracking()
        score = rollout(child, samples, benchmark)

        # 4. BACKPROPAGATE
        backpropagate(child, score)

        # Track best
        if score > best_score:
            best_score = score
            best_stages = list(child.stages)
            marker = " *** NEW BEST ***"
        else:
            marker = ""

        if (i + 1) % 5 == 0 or marker:
            path = []
            n = child
            while n.parent:
                path.append(n.decision)
                n = n.parent
            path.reverse()
            print(f"  [{i+1:3d}/{n_iterations}] score={score:5.1f}% | "
                  f"depth={child.depth} | tree_size={_tree_size(root)} | "
                  f"best={best_score:.1f}%{marker}")
            if marker:
                print(f"         path: {' → '.join(path)}")

    # Final
    print(f"\n{'═'*50}")
    print(f"MCTS-MORPH COMPLETE")
    print(f"{'═'*50}")
    print(f"Best score: {best_score}%")
    print(f"Best design ({len(best_stages)} stages):")
    for i, s in enumerate(best_stages):
        cond = f" [if {s.condition}]" if s.condition != "always" else ""
        print(f"  {i+1}. {s.action}(t={s.temperature}){cond}")
        if s.system_prompt:
            print(f"     {s.system_prompt[:60]}...")

    # Print tree statistics
    print(f"\nTree: {_tree_size(root)} nodes, max depth {_tree_depth(root)}")
    print(f"Root visits: {root.visits}")

    # Save best
    genome = Genome(name="mcts_best", model=CHEAP, stages=best_stages)
    with open("best_mcts.json", "w") as f:
        json.dump(genome.to_dict(), f, indent=2)

    return {"score": best_score, "stages": best_stages}


def _tree_size(node):
    return 1 + sum(_tree_size(c) for c in node.children)

def _tree_depth(node):
    if not node.children:
        return 0
    return 1 + max(_tree_depth(c) for c in node.children)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--iters", type=int, default=60)
    args = p.parse_args()
    run_mcts_morph(args.benchmark, args.n, args.iters)
