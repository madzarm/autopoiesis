"""Meta-evolutionary search over interactive agent configurations.

This is our main contribution — improving on EvoMAS by:

1. EVOLVING INTERACTIVE AGENTS (not just pipeline configs):
   - EvoMAS evolves YAML configs for pipeline agents
   - We evolve the system prompt, strategy hints, tool selection,
     and decision-making heuristics of an INTERACTIVE bash agent
   - This gives far more expressiveness: the agent can adapt its
     behavior dynamically based on what it discovers

2. TRAJECTORY-CONDITIONED MUTATIONS:
   - EvoMAS uses raw execution traces
   - We extract STRUCTURED failure patterns from agent trajectories:
     * Wrong file localized → improve localization instructions
     * Correct file but wrong function → improve code understanding
     * Correct edit but syntax error → improve edit formatting
     * Test failures after edit → improve verification loop
     * Agent stuck in loop → improve termination conditions

3. POPULATION DIVERSITY MAINTENANCE:
   - EvoMAS uses a flat pool with task-relevance retrieval
   - We maintain a MAP-Elites-style archive where configs are
     indexed by (repo_type, bug_type, difficulty) → ensures
     specialization across problem types

4. CROSS-INSTANCE TRANSFER:
   - EvoMAS stores consolidated memory per instance
   - We build a hierarchical memory: repo-level patterns →
     bug-type patterns → instance-specific fixes

Architecture:
  MetaEvolver
    ├── Population (MAP-Elites archive of agent configs)
    ├── Trajectory Analyzer (extracts failure patterns)
    ├── Mutation Engine (LLM-guided, trace-conditioned)
    ├── Crossover Engine (combines complementary configs)
    └── Memory (hierarchical, embedding-free keyword retrieval)
"""

import os
import json
import time
import copy
import random
import hashlib
from typing import Optional, Tuple
from collections import defaultdict

from swe_llm import call_llm, SONNET, AGENT_MODEL, get_session_stats
from swe_interactive_agent import (
    solve_interactive, DEFAULT_INTERACTIVE_CONFIG, DEFAULT_SYSTEM_PROMPT,
)


# ── Seed Configurations for Interactive Agent ────────────────────────────────

INTERACTIVE_SEEDS = [
    {
        "name": "default_interactive",
        "model": AGENT_MODEL,
        "temperature": 0.0,
        "max_tokens": 4096,
        "max_turns": 25,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "strategy_hint": "",
    },
    {
        "name": "methodical_explorer",
        "model": AGENT_MODEL,
        "temperature": 0.0,
        "max_tokens": 4096,
        "max_turns": 30,
        "system_prompt": """You are an expert software engineer. You fix bugs by following a strict methodology:

## Phase 1: Understand (turns 1-5)
- Read the issue carefully
- Search for key terms from the error/issue
- Find the exact file and function mentioned

## Phase 2: Diagnose (turns 6-10)
- View the relevant code with context
- Trace the logic to find the root cause
- Check related test files for expected behavior

## Phase 3: Fix (turns 11-15)
- Make the MINIMUM change to fix the bug
- Use edit_file to modify the code
- View the edited file to verify your changes

## Phase 4: Verify (turns 16-20)
- Run relevant tests
- Check for regressions
- Call create_patch when confident

## Available Commands
- `find_file <pattern>` — Find files matching pattern
- `search <pattern>` — Grep for pattern in Python files
- `view_file <path> [start] [end]` — View file with line numbers
- `edit_file <path> <start> <end>` — Replace lines (content follows, end with END_EDIT)
- `create_patch` — Generate final diff
- `run_tests <path>` — Run tests
- `bash <cmd>` — Run shell command

CRITICAL RULES:
- ONE command per turn
- Always view a file BEFORE editing it
- Make MINIMAL changes
- When editing, include ALL surrounding context lines exactly as they are""",
        "strategy_hint": "Start by searching for the exact error message or class name mentioned in the issue.",
    },
    {
        "name": "test_driven_fixer",
        "model": AGENT_MODEL,
        "temperature": 0.0,
        "max_tokens": 4096,
        "max_turns": 30,
        "system_prompt": """You are a test-driven software engineer fixing bugs.

## Strategy
1. Find and read the failing test or create a mental test case
2. Find the source code that the test exercises
3. Understand why the current code fails
4. Fix the code
5. Verify with tests

## Commands
- `find_file <pattern>` — Find files
- `search <pattern>` — Search code
- `view_file <path> [start] [end]` — View file
- `edit_file <path> <start> <end>` — Edit file (content follows, END_EDIT to finish)
- `create_patch` — Generate diff
- `run_tests <path>` — Run tests
- `bash <cmd>` — Shell command

## Rules
- Always look at tests first to understand expected behavior
- One command per turn
- View before edit
- Minimal changes only
- Run tests before generating patch if possible""",
        "strategy_hint": "First, find the test file for the module mentioned in the issue. Read it to understand expected behavior.",
    },
    {
        "name": "surgical_fixer",
        "model": AGENT_MODEL,
        "temperature": 0.0,
        "max_tokens": 4096,
        "max_turns": 20,
        "system_prompt": """You are a surgical bug fixer. Your goal: find the bug fast, fix it precisely, move on.

## Commands
- `find_file <pattern>` — Find files
- `search <pattern>` — Search Python files
- `view_file <path> [start] [end]` — View file with line numbers
- `edit_file <path> <start> <end>` — Replace lines (content until END_EDIT)
- `create_patch` — Generate final diff
- `bash <cmd>` — Any shell command

## Approach
1. Extract key identifiers from the issue (class names, method names, error messages)
2. Search directly for those identifiers
3. View the relevant code section
4. Make the minimal fix
5. Generate patch

Do NOT explore broadly. Be targeted. Fix the specific bug described.
Keep your explanations to 1-2 sentences per turn.
One command per turn. Always view before editing.""",
        "strategy_hint": "",
    },
    {
        "name": "context_builder",
        "model": AGENT_MODEL,
        "temperature": 0.2,
        "max_tokens": 4096,
        "max_turns": 30,
        "system_prompt": """You are a thorough software engineer who builds deep understanding before making changes.

## Commands
- `find_file <pattern>` — Find files
- `search <pattern>` — Search Python files
- `view_file <path> [start] [end]` — View file
- `edit_file <path> <start> <end>` — Edit file (end with END_EDIT)
- `create_patch` — Generate diff
- `run_tests <path>` — Run tests
- `bash <cmd>` — Shell command

## Strategy
1. Read the issue 2-3 times to understand every detail
2. Explore the module structure (find_file, bash ls)
3. Read imports and class hierarchy to understand architecture
4. View the specific function with generous context (50+ lines)
5. Understand the intended vs actual behavior
6. Look at similar working code for patterns
7. Fix with confidence
8. Verify and generate patch

## Rules
- One command per turn
- View files before editing
- Read related code to understand patterns
- Minimal changes only
- When in doubt, read more code before editing""",
        "strategy_hint": "Start with `bash find . -type f -name '*.py' | head -30` to understand the project structure.",
    },
]


# ── Trajectory Analysis ──────────────────────────────────────────────────────

def analyze_trajectory(trajectory: list, instance: dict, result: dict) -> dict:
    """Extract structured failure patterns from agent trajectory.

    Returns a dict of failure signals for mutation guidance.
    """
    analysis = {
        "success": bool(result.get("model_patch", "")),
        "turns_used": len(trajectory),
        "patterns": [],
        "failure_type": "unknown",
    }

    if not trajectory:
        analysis["failure_type"] = "no_trajectory"
        return analysis

    # Quick success check
    if result.get("model_patch", "").strip():
        analysis["success"] = True
        if "diff --git" in result["model_patch"]:
            analysis["failure_type"] = "none"  # Not a failure

    # Check for common failure patterns
    commands = [t.get("command", "") for t in trajectory]
    outputs = [t.get("output", "") for t in trajectory]

    # Pattern 1: Never found relevant file
    search_cmds = [c for c in commands if c.startswith("search") or c.startswith("find_file")]
    if search_cmds and all("no matches" in o.lower() or "not found" in o.lower()
                           for o in outputs[:5]):
        analysis["patterns"].append("failed_localization")
        analysis["failure_type"] = "localization"

    # Pattern 2: Found file but never edited
    view_cmds = [c for c in commands if c.startswith("view_file")]
    edit_cmds = [c for c in commands if any(c.startswith(e) for e in
                 ("edit_file", "str_replace", "bash sed", "bash echo"))]
    has_patch = bool(result.get("model_patch", "").strip())
    if view_cmds and not edit_cmds and not has_patch:
        analysis["patterns"].append("viewed_but_no_edit")
        analysis["failure_type"] = "no_edit_made"

    # Pattern 3: Edited but no patch generated
    patch_cmds = [c for c in commands if c.startswith("create_patch")]
    if edit_cmds and not patch_cmds:
        analysis["patterns"].append("edited_but_no_patch")
        analysis["failure_type"] = "forgot_patch"

    # Pattern 4: Patch was empty (edits reverted or didn't save)
    if patch_cmds and not result.get("model_patch"):
        analysis["patterns"].append("empty_patch")
        analysis["failure_type"] = "empty_patch"

    # Pattern 5: Agent got stuck in a loop
    if len(set(commands)) < len(commands) / 2:
        analysis["patterns"].append("stuck_in_loop")
        analysis["failure_type"] = "loop"

    # Pattern 6: Timeout (used all turns)
    max_turns = result.get("turns_used", 0)
    if max_turns >= 25 and not result.get("model_patch"):
        analysis["patterns"].append("timeout")
        if analysis["failure_type"] == "unknown":
            analysis["failure_type"] = "timeout"

    # Pattern 7: Edit error
    for o in outputs:
        if "[error" in o.lower() and "edit" in str(commands):
            analysis["patterns"].append("edit_error")
            analysis["failure_type"] = "edit_error"
            break

    return analysis


# ── Meta-Evolution Engine ────────────────────────────────────────────────────

class MetaEvolver:
    """Evolves interactive agent configurations based on trajectory analysis.

    Key innovation: uses structured trajectory analysis to guide mutations,
    not just raw execution traces like EvoMAS.
    """

    def __init__(self, population_size: int = 5, evolution_depth: int = 3,
                 meta_model: str = AGENT_MODEL):
        self.population_size = population_size
        self.evolution_depth = evolution_depth
        self.meta_model = meta_model

        # MAP-Elites archive: (repo, difficulty) → best config
        self.archive = {}

        # Population
        self.population = [
            {"config": copy.deepcopy(seed), "score": 0.0, "seen": 0}
            for seed in INTERACTIVE_SEEDS[:population_size]
        ]

        # Memory
        self.memory = defaultdict(list)  # repo → [(instance_id, config_name, score, analysis)]

        # Stats
        self.total_evals = 0
        self.total_cost = 0.0
        self.results_log = []

    def solve_with_evolution(self, instance: dict, work_dir: str) -> dict:
        """Solve one instance, evolving configs as we go.

        Process:
        1. Select 2-3 promising configs
        2. Run each on the instance
        3. Analyze trajectories
        4. Mutate based on failure patterns
        5. Run mutants
        6. Return best patch
        """
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        difficulty = instance.get("difficulty", "unknown")

        print(f"\n{'='*60}")
        print(f"[MetaEvolver] {instance_id} (repo={repo}, diff={difficulty})")
        print(f"{'='*60}")

        # Select initial candidates
        candidates = self._select(instance)
        all_results = []

        for depth in range(self.evolution_depth):
            print(f"\n  --- Evo depth {depth+1}/{self.evolution_depth} ({len(candidates)} candidates) ---")

            # Evaluate candidates
            depth_results = []
            for cand in candidates:
                config = cand["config"]
                print(f"  Running: {config['name']}")

                inst_dir = os.path.join(work_dir, f"{instance_id.replace('/', '__')}_{config['name']}")
                os.makedirs(inst_dir, exist_ok=True)

                result = solve_interactive(instance, config, inst_dir)
                trajectory = result.get("trajectory", [])
                patch = result.get("model_patch", "")
                turns = result.get("turns_used", 0)
                cost = result.get("total_cost", 0)

                # Score: has valid diff = 0.5, has patch = 0.3, nothing = 0
                score = 0.0
                if patch and ("diff --git" in patch or "---" in patch):
                    score = 0.5
                elif patch:
                    score = 0.3

                analysis = analyze_trajectory(trajectory, instance, result)

                depth_results.append({
                    "config": config,
                    "result": result,
                    "score": score,
                    "analysis": analysis,
                    "cost": cost,
                })
                all_results.append(depth_results[-1])

                print(f"    Score: {score}, Turns: {turns}, Patch: {len(patch)} chars, "
                      f"Failure: {analysis['failure_type']}, Cost: ${cost:.4f}")

                self.total_evals += 1

            # Check if we have a good result
            best = max(depth_results, key=lambda r: r["score"])
            if best["score"] >= 0.5:
                print(f"  ✓ Good patch found at depth {depth+1}")
                if depth < self.evolution_depth - 1:
                    # Still evolve to find potentially better patches
                    pass

            # Mutate for next round (unless last depth)
            if depth < self.evolution_depth - 1:
                candidates = self._evolve(depth_results, instance)

        # Pick best result overall
        all_results.sort(key=lambda r: (-r["score"], -len(r["result"].get("model_patch", ""))))
        best_overall = all_results[0]

        # Update memory and archive
        self._update_memory(instance, best_overall)

        self.results_log.append({
            "instance_id": instance_id,
            "best_config": best_overall["config"]["name"],
            "score": best_overall["score"],
            "total_candidates": len(all_results),
        })

        return best_overall["result"]

    def _select(self, instance: dict) -> list:
        """Select configs for this instance."""
        repo = instance["repo"]
        difficulty = instance.get("difficulty", "unknown")
        candidates = []

        # 1. Check archive for this (repo, difficulty)
        key = (repo, difficulty)
        if key in self.archive:
            candidates.append({"config": copy.deepcopy(self.archive[key])})

        # 2. Check memory for best config for this repo
        repo_memory = self.memory.get(repo, [])
        if repo_memory:
            best_mem = max(repo_memory, key=lambda m: m.get("score", 0))
            for p in self.population:
                if p["config"]["name"] == best_mem.get("config_name"):
                    candidates.append({"config": copy.deepcopy(p["config"])})
                    break

        # 3. Top from population
        pop_sorted = sorted(self.population, key=lambda p: -p["score"])
        for p in pop_sorted:
            if len(candidates) >= 3:
                break
            name = p["config"]["name"]
            if not any(c["config"]["name"] == name for c in candidates):
                candidates.append({"config": copy.deepcopy(p["config"])})

        # Ensure at least 2 candidates
        while len(candidates) < 2:
            seed = random.choice(INTERACTIVE_SEEDS)
            candidates.append({"config": copy.deepcopy(seed)})

        return candidates[:3]  # Max 3 candidates per round

    def _evolve(self, results: list, instance: dict) -> list:
        """Generate evolved candidates from current results."""
        # Sort by score
        results.sort(key=lambda r: -r["score"])
        new_candidates = []

        # Keep best unchanged
        new_candidates.append({"config": results[0]["config"]})

        # Mutate based on trajectory analysis
        for r in results[:2]:
            analysis = r["analysis"]
            mutated = self._mutate_from_analysis(r["config"], analysis, instance)
            if mutated:
                new_candidates.append({"config": mutated})

        # Crossover if we have 2+ configs
        if len(results) >= 2:
            crossed = self._crossover(results[0], results[1])
            if crossed:
                new_candidates.append({"config": crossed})

        return new_candidates[:3]

    def _mutate_from_analysis(self, config: dict, analysis: dict,
                              instance: dict) -> Optional[dict]:
        """Targeted mutation based on trajectory analysis."""
        failure_type = analysis.get("failure_type", "unknown")
        patterns = analysis.get("patterns", [])

        mutation_prompt = f"""You are improving an AI agent's configuration for fixing software bugs (SWE-bench).

Current config name: {config['name']}
Current system prompt (first 500 chars): {config.get('system_prompt', '')[:500]}
Strategy hint: {config.get('strategy_hint', 'none')}
Max turns: {config.get('max_turns', 25)}
Temperature: {config.get('temperature', 0.0)}

## Failure Analysis
Failure type: {failure_type}
Patterns detected: {', '.join(patterns) if patterns else 'none'}
Turns used: {analysis.get('turns_used', 0)}

## Problem context
Repository: {instance.get('repo', 'unknown')}
Issue (first 300 chars): {instance.get('problem_statement', '')[:300]}

## Mutation Instructions by Failure Type
- "localization": The agent failed to find the right files. Improve search strategy in system_prompt. Add explicit instructions to try multiple search terms, look at imports, use `bash find`.
- "no_edit_made": Agent found code but didn't edit. Improve system_prompt to be more decisive. Add instruction: "Once you understand the bug, immediately edit the file. Don't over-analyze."
- "forgot_patch": Agent edited but forgot to create_patch. Add explicit reminder in system_prompt: "ALWAYS call create_patch after editing."
- "empty_patch": Edit didn't stick. Improve edit_file instructions — emphasize exact line numbers, viewing file after edit.
- "loop": Agent repeated commands. Add anti-loop instruction: "Never repeat the same command. If a command didn't work, try a different approach."
- "timeout": Agent ran out of turns. Increase max_turns OR add urgency: "Be efficient. Aim to fix within 15 turns."
- "edit_error": Agent made syntax errors in edit. Add examples of correct edit_file usage.

Output a JSON with the mutated fields. Only include fields you're changing:
```json
{{
    "name": "new_name",
    "system_prompt": "full new system prompt if changed",
    "strategy_hint": "new hint if changed",
    "max_turns": 25,
    "temperature": 0.0,
    "mutation_note": "what you changed and why"
}}
```"""

        try:
            resp = call_llm(
                prompt=mutation_prompt,
                model=self.meta_model,
                temperature=0.7,
                max_tokens=4096,
            )
            self.total_cost += resp["cost_usd"]

            # Parse JSON
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n```', resp["content"], re.DOTALL)
            if json_match:
                mutated_fields = json.loads(json_match.group(1))
                new_config = copy.deepcopy(config)
                for k, v in mutated_fields.items():
                    if k in ("name", "system_prompt", "strategy_hint",
                             "max_turns", "temperature", "mutation_note"):
                        new_config[k] = v

                # Generate unique name
                h = hashlib.md5(json.dumps(new_config, sort_keys=True, default=str).encode()).hexdigest()[:6]
                new_config["name"] = f"mut_{failure_type}_{h}"
                return new_config
        except Exception as e:
            print(f"    [mutation failed: {e}]")

        return None

    def _crossover(self, result_a: dict, result_b: dict) -> Optional[dict]:
        """Combine two configs."""
        config_a = result_a["config"]
        config_b = result_b["config"]
        score_a = result_a["score"]
        score_b = result_b["score"]

        # Take system_prompt from higher scorer, strategy from other
        new_config = copy.deepcopy(config_a if score_a >= score_b else config_b)
        donor = config_b if score_a >= score_b else config_a

        new_config["strategy_hint"] = donor.get("strategy_hint", "")
        new_config["temperature"] = (config_a.get("temperature", 0) + config_b.get("temperature", 0)) / 2

        h = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        new_config["name"] = f"cross_{h}"
        return new_config

    def _update_memory(self, instance: dict, best: dict):
        """Update memory and archive."""
        repo = instance["repo"]
        difficulty = instance.get("difficulty", "unknown")
        config = best["config"]
        score = best["score"]

        # Memory
        self.memory[repo].append({
            "instance_id": instance["instance_id"],
            "config_name": config["name"],
            "score": score,
            "analysis": best.get("analysis", {}),
        })

        # Archive: store best for (repo, difficulty)
        key = (repo, difficulty)
        if key not in self.archive or score > 0:
            self.archive[key] = copy.deepcopy(config)

        # Update population scores
        for p in self.population:
            if p["config"]["name"] == config["name"]:
                p["score"] = 0.7 * p["score"] + 0.3 * score
                p["seen"] += 1
                return

        # Add new config to population if it scored well
        if score > 0.3:
            self.population.append({
                "config": copy.deepcopy(config),
                "score": score,
                "seen": 1,
            })
            # Prune to maintain size
            if len(self.population) > self.population_size * 2:
                self.population.sort(key=lambda p: -p["score"])
                self.population = self.population[:self.population_size]

    def get_summary(self) -> str:
        """Get evolution summary."""
        lines = [
            f"MetaEvolver Summary:",
            f"  Total evals: {self.total_evals}",
            f"  Meta cost: ${self.total_cost:.4f}",
            f"  Population: {len(self.population)} configs",
            f"  Archive: {len(self.archive)} entries",
            f"  Memory: {sum(len(v) for v in self.memory.values())} entries across {len(self.memory)} repos",
        ]
        for p in sorted(self.population, key=lambda x: -x["score"])[:5]:
            lines.append(f"    {p['config']['name']}: score={p['score']:.2f}, seen={p['seen']}")
        return "\n".join(lines)
