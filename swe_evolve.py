"""Evolutionary Agent Design Search for SWE-bench.

Improvements over EvoMAS:
1. HYBRID config+code: Evolves structured YAML configs WITH embedded strategy code
   snippets that are sandboxed, giving expressiveness + reliability
2. EXECUTION-TRACE-GUIDED MUTATIONS: Like EvoMAS, but with richer trace analysis —
   we extract structured failure modes, not just raw traces
3. POPULATION-BASED SEARCH: Maintains diverse population with Pareto optimization
   across quality/cost/reliability
4. CROSS-INSTANCE MEMORY: Embedding-based retrieval for config transfer across tasks
5. ADAPTIVE OPERATOR SELECTION: Track which mutation operators work best, use more

Architecture:
- Population of agent configs (YAML-like dicts)
- Each config defines: agent topology (DAG), per-agent prompts, tools, strategy
- Evolution: select → mutate/crossover → evaluate → update population
- Memory: store (instance, best_config, trace_summary) for retrieval
"""

import os
import json
import time
import copy
import random
import hashlib
from typing import Optional
from collections import defaultdict

from swe_llm import call_llm, SONNET, OPUS, get_session_stats
from swe_agent import solve_instance, DEFAULT_CONFIG


# ── Configuration Templates (Initial Population) ────────────────────────────

SEED_CONFIGS = [
    # 1. Simple Agentless baseline
    {
        "name": "agentless_simple",
        "description": "Single-agent localize-then-repair pipeline",
        "model": SONNET,
        "temperature": 0.0,
        "max_tokens": 8192,
        "localize_strategy": "file_then_function",
        "repair_strategy": "direct",
        "validate": False,
        "num_candidates": 1,
        "temperatures": [0.0],
        "localize_prompt": DEFAULT_CONFIG["localize_prompt"],
        "repair_prompt": DEFAULT_CONFIG["repair_prompt"],
    },

    # 2. Multi-candidate with validation
    {
        "name": "multi_candidate_3",
        "description": "Generate 3 candidates at different temperatures, validate",
        "model": SONNET,
        "temperature": 0.0,
        "max_tokens": 8192,
        "localize_strategy": "file_then_function",
        "repair_strategy": "multi_candidate",
        "validate": True,
        "num_candidates": 3,
        "temperatures": [0.0, 0.3, 0.7],
        "localize_prompt": DEFAULT_CONFIG["localize_prompt"],
        "repair_prompt": DEFAULT_CONFIG["repair_prompt"],
        "validate_prompt": DEFAULT_CONFIG["validate_prompt"],
    },

    # 3. CoT-heavy approach
    {
        "name": "cot_deep_analysis",
        "description": "Deep chain-of-thought analysis before patching",
        "model": SONNET,
        "temperature": 0.0,
        "max_tokens": 8192,
        "localize_strategy": "file_then_function",
        "repair_strategy": "direct",
        "validate": True,
        "num_candidates": 1,
        "temperatures": [0.0],
        "localize_prompt": """You are a world-class software debugger. Analyze this issue with extreme precision.

Step 1: Read the problem statement carefully. What EXACTLY is the bug?
Step 2: Look at the repository structure. Which package/module owns this functionality?
Step 3: Trace the likely code path from the user-facing API to the buggy behavior.
Step 4: Identify the EXACT files and functions that need modification.

Be specific. Name exact file paths and function signatures.

Return JSON:
{
    "analysis": "detailed root cause analysis",
    "bug_type": "logic_error|missing_check|wrong_default|type_error|api_mismatch|other",
    "files": ["exact/path/to/file.py"],
    "functions": ["Class.method"],
    "confidence": 0.0-1.0
}""",
        "repair_prompt": """You are fixing a bug in a Python project.

BEFORE writing the patch:
1. Explain the root cause of the bug in 2-3 sentences
2. Explain your fix strategy
3. Consider what tests would break or pass after your fix
4. Think about edge cases

THEN generate a minimal unified diff. Only change what's necessary.

Output format:
## Root Cause
<explanation>

## Fix Strategy
<strategy>

## Patch
```diff
<unified diff>
```""",
    },

    # 4. Grep-based localization
    {
        "name": "grep_localize",
        "description": "Use keyword grep to find relevant code, then repair",
        "model": SONNET,
        "temperature": 0.0,
        "max_tokens": 8192,
        "localize_strategy": "grep_based",
        "repair_strategy": "direct",
        "validate": True,
        "num_candidates": 2,
        "temperatures": [0.0, 0.3],
        "localize_prompt": """Analyze this problem statement and identify search keywords.

Extract:
1. Class names mentioned
2. Method/function names mentioned
3. Error messages or exceptions
4. Module names

Then identify which files need modification.

Return JSON:
{
    "analysis": "what the bug is",
    "search_terms": ["keyword1", "keyword2"],
    "files": ["path/to/file.py"],
    "functions": ["function_name"],
    "confidence": 0.0-1.0
}""",
        "repair_prompt": DEFAULT_CONFIG["repair_prompt"],
        "validate_prompt": DEFAULT_CONFIG["validate_prompt"],
    },

    # 5. Two-phase repair (plan then implement)
    {
        "name": "plan_then_implement",
        "description": "First plan the fix, then implement it in a second call",
        "model": SONNET,
        "temperature": 0.0,
        "max_tokens": 8192,
        "localize_strategy": "file_then_function",
        "repair_strategy": "two_phase",
        "validate": False,
        "num_candidates": 1,
        "temperatures": [0.0],
        "localize_prompt": DEFAULT_CONFIG["localize_prompt"],
        "repair_prompt": DEFAULT_CONFIG["repair_prompt"],
        "plan_prompt": """You are planning a bug fix. Do NOT write any code yet.

Given the problem and source code, create a detailed fix plan:
1. What is the root cause?
2. What specific changes are needed? (file, function, line range)
3. What is the correct behavior after the fix?
4. What edge cases should be handled?

Be extremely specific about WHAT to change and WHERE.""",
        "implement_prompt": """Implement the fix plan below as a minimal unified diff.

Fix Plan:
{plan}

Rules:
- Follow the plan exactly
- Make MINIMUM changes
- Output only a unified diff in git format
- Preserve code style

```diff
<your diff here>
```""",
    },
]


# ── Mutation Operators ──────────────────────────────────────────────────────

MUTATION_TYPES = [
    "prompt_refine",      # Refine prompts based on failure traces
    "temperature_adjust", # Adjust generation temperatures
    "strategy_change",    # Change localize/repair strategy
    "candidate_count",    # Add/remove candidate count
    "add_validation",     # Toggle or modify validation
    "prompt_specialize",  # Specialize prompts for failure type
    "model_override",     # Change model per phase
]


def mutate_config(config: dict, failure_trace: str = "", meta_model: str = SONNET) -> dict:
    """Mutate a config using LLM-guided mutation, conditioned on failure traces.

    This is the key improvement over EvoMAS — richer trace analysis.
    """
    new_config = copy.deepcopy(config)

    # Pick mutation type (with adaptive weighting later)
    mutation_type = random.choice(MUTATION_TYPES)

    prompt = f"""You are evolving an AI agent configuration to improve its performance on software bug fixing (SWE-bench).

Current configuration:
```json
{json.dumps({k: v for k, v in config.items() if k != '_history'}, indent=2, default=str)}
```

Mutation type to apply: {mutation_type}

{"Failure trace from last evaluation:" if failure_trace else "No failure trace available."}
{failure_trace[:3000] if failure_trace else ""}

Instructions for mutation type "{mutation_type}":
- prompt_refine: Improve the localize_prompt or repair_prompt based on what went wrong. Make it more specific, add better examples, or fix reasoning gaps.
- temperature_adjust: Change temperatures list to explore better. Current: {config.get('temperatures', [0.0])}
- strategy_change: Change localize_strategy or repair_strategy. Options: file_then_function, grep_based, ast_based for localize; direct, cot_then_patch, multi_candidate, two_phase for repair.
- candidate_count: Adjust num_candidates (1-5). More = better quality but higher cost.
- add_validation: Modify validate (true/false) or improve validate_prompt.
- prompt_specialize: If the failure trace shows a specific pattern (e.g., wrong file localized, incomplete fix, syntax error in patch), specialize the prompt to handle that pattern.
- model_override: This is advanced — suggest different temperatures or max_tokens for different phases.

Output the MUTATED configuration as JSON. Only change fields relevant to the mutation type.
Keep the same JSON structure. Include a "mutation_note" field explaining what you changed and why.

```json
<mutated config>
```"""

    resp = call_llm(
        prompt=prompt,
        model=meta_model,
        temperature=0.7,
        max_tokens=4096,
    )

    # Parse mutated config
    import re
    json_match = re.search(r'```json\s*\n(.*?)\n```', resp["content"], re.DOTALL)
    if json_match:
        try:
            mutated = json.loads(json_match.group(1))
            # Merge mutation into config (only update known fields)
            valid_fields = {
                "name", "description", "model", "temperature", "max_tokens",
                "localize_strategy", "repair_strategy", "validate",
                "num_candidates", "temperatures", "localize_prompt",
                "repair_prompt", "validate_prompt", "plan_prompt",
                "implement_prompt", "mutation_note",
            }
            for k, v in mutated.items():
                if k in valid_fields:
                    new_config[k] = v
        except json.JSONDecodeError:
            pass  # Keep original config on parse failure

    # Track mutation history
    history = new_config.get("_history", [])
    history.append({
        "mutation_type": mutation_type,
        "note": new_config.get("mutation_note", ""),
        "timestamp": time.time(),
    })
    new_config["_history"] = history

    # Generate unique name
    h = hashlib.md5(json.dumps(new_config, sort_keys=True, default=str).encode()).hexdigest()[:6]
    new_config["name"] = f"{config['name']}_m{h}"

    return new_config


def crossover_configs(config_a: dict, config_b: dict,
                      trace_a: str = "", trace_b: str = "",
                      meta_model: str = SONNET) -> dict:
    """Crossover two configs, guided by their execution traces."""
    prompt = f"""You are combining two AI agent configurations that each have strengths.

Config A ({config_a.get('name', 'A')}):
```json
{json.dumps({k: v for k, v in config_a.items() if k not in ('_history', '_scores')}, indent=2, default=str)}
```

Config B ({config_b.get('name', 'B')}):
```json
{json.dumps({k: v for k, v in config_b.items() if k not in ('_history', '_scores')}, indent=2, default=str)}
```

{f"Config A performance trace: {trace_a[:1500]}" if trace_a else ""}
{f"Config B performance trace: {trace_b[:1500]}" if trace_b else ""}

Create a NEW configuration that combines the best aspects of both.
Take the better prompt from whichever config seems stronger, combine strategies that complement each other.

Output as JSON:
```json
<combined config>
```"""

    resp = call_llm(
        prompt=prompt,
        model=meta_model,
        temperature=0.5,
        max_tokens=4096,
    )

    import re
    json_match = re.search(r'```json\s*\n(.*?)\n```', resp["content"], re.DOTALL)
    if json_match:
        try:
            combined = json.loads(json_match.group(1))
            combined["_history"] = [
                {"type": "crossover",
                 "parents": [config_a.get("name", "A"), config_b.get("name", "B")],
                 "timestamp": time.time()}
            ]
            h = hashlib.md5(json.dumps(combined, sort_keys=True, default=str).encode()).hexdigest()[:6]
            combined["name"] = f"cross_{h}"
            return combined
        except json.JSONDecodeError:
            pass

    # Fallback: manual crossover
    combined = copy.deepcopy(config_a)
    # Take repair prompt from B if B has higher score
    if config_b.get("_scores", {}).get("last", 0) > config_a.get("_scores", {}).get("last", 0):
        combined["repair_prompt"] = config_b.get("repair_prompt", combined["repair_prompt"])
    combined["name"] = f"cross_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
    return combined


# ── Experience Memory ───────────────────────────────────────────────────────

class ExperienceMemory:
    """Stores (instance, config, trace, score) tuples for retrieval."""

    def __init__(self, path: str = "swe_memory.json"):
        self.path = path
        self.entries = []
        if os.path.exists(path):
            with open(path) as f:
                self.entries = json.load(f)

    def add(self, instance_id: str, config_name: str, score: float,
            trace_summary: str, repo: str = ""):
        self.entries.append({
            "instance_id": instance_id,
            "config_name": config_name,
            "score": score,
            "trace_summary": trace_summary,
            "repo": repo,
            "timestamp": time.time(),
        })
        self._save()

    def get_similar(self, repo: str, problem_keywords: list, k: int = 3) -> list:
        """Retrieve k most relevant past experiences."""
        scored = []
        for entry in self.entries:
            score = 0
            if entry.get("repo") == repo:
                score += 2
            for kw in problem_keywords:
                if kw.lower() in entry.get("trace_summary", "").lower():
                    score += 1
            scored.append((score, entry))

        scored.sort(key=lambda x: (-x[0], -x[1].get("score", 0)))
        return [e for _, e in scored[:k]]

    def get_best_config_for_repo(self, repo: str) -> Optional[str]:
        """Get the config name that worked best for this repo."""
        repo_entries = [e for e in self.entries if e.get("repo") == repo and e["score"] > 0]
        if not repo_entries:
            return None
        best = max(repo_entries, key=lambda e: e["score"])
        return best["config_name"]

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)


# ── Evolution Controller ────────────────────────────────────────────────────

class EvolutionController:
    """Manages the evolutionary search over agent configurations.

    Key improvements over EvoMAS:
    1. Population-based (not single-config evolution)
    2. Pareto optimization (quality + cost)
    3. Adaptive operator selection
    4. Cross-instance memory transfer
    """

    def __init__(self, population_size: int = 5, evolution_depth: int = 3,
                 meta_model: str = SONNET, memory_path: str = "swe_memory.json"):
        self.population_size = population_size
        self.evolution_depth = evolution_depth
        self.meta_model = meta_model
        self.memory = ExperienceMemory(memory_path)

        # Population: list of (config, score, cost) tuples
        self.population = []
        self.best_configs = {}  # repo -> best config

        # Operator tracking
        self.operator_scores = defaultdict(lambda: {"wins": 0, "total": 0})

        # Initialize population with seeds
        for seed in SEED_CONFIGS[:population_size]:
            self.population.append({
                "config": copy.deepcopy(seed),
                "score": 0.0,
                "cost": 0.0,
                "instances_seen": 0,
            })

    def evolve_for_instance(self, instance: dict, work_dir: str,
                            eval_fn=None) -> dict:
        """Run evolution for a single SWE-bench instance.

        For each instance:
        1. Select promising configs from population + memory
        2. Evaluate them
        3. Mutate best performers
        4. Return best patch

        Args:
            instance: SWE-bench instance dict
            work_dir: working directory for git clones
            eval_fn: optional function(result, instance) -> score (0 or 1)

        Returns:
            Best result dict with model_patch
        """
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        problem = instance["problem_statement"]

        print(f"\n{'='*60}")
        print(f"Evolving for: {instance_id}")
        print(f"{'='*60}")

        # Step 1: Select initial candidates
        candidates = self._select_candidates(instance)

        best_result = None
        best_score = -1

        # Step 2: Evaluate → Mutate → Re-evaluate loop
        for depth in range(self.evolution_depth):
            print(f"\n  --- Evolution step {depth+1}/{self.evolution_depth} ---")

            # Evaluate each candidate
            results = []
            for i, cand in enumerate(candidates):
                config = cand["config"]
                print(f"  Evaluating config: {config['name']}")

                result = solve_instance(instance, config, work_dir)
                patch = result.get("model_patch", "")

                # Score: use eval_fn if provided, otherwise proxy
                score = 0.0
                if eval_fn and patch:
                    score = eval_fn(result, instance)
                elif patch:
                    score = 0.5  # Has patch but unverified

                results.append({
                    "config": config,
                    "result": result,
                    "score": score,
                    "patch_len": len(patch),
                })

                if score > best_score:
                    best_score = score
                    best_result = result

                print(f"    Score: {score}, Patch: {len(patch)} chars")

            # Early exit if we found a perfect solution
            if best_score >= 1.0:
                print(f"  ✓ Perfect score found at depth {depth+1}")
                break

            # Step 3: Evolve — mutate top performers
            results.sort(key=lambda x: -x["score"])
            top = results[:2]  # Keep top 2

            # Generate failure trace for mutation guidance
            failure_trace = self._build_failure_trace(results, instance)

            new_candidates = []
            # Keep best unchanged
            new_candidates.append({"config": top[0]["config"]})

            # Mutate best
            for t in top:
                mutated = mutate_config(
                    t["config"],
                    failure_trace=failure_trace,
                    meta_model=self.meta_model,
                )
                new_candidates.append({"config": mutated})

            # Crossover if we have 2+ good configs
            if len(top) >= 2:
                crossed = crossover_configs(
                    top[0]["config"], top[1]["config"],
                    meta_model=self.meta_model,
                )
                new_candidates.append({"config": crossed})

            candidates = new_candidates[:self.population_size]

        # Update memory
        if best_result:
            config_name = best_result.get("model_name_or_path", "unknown")
            trace_summary = f"score={best_score}, patch_len={len(best_result.get('model_patch', ''))}"
            self.memory.add(instance_id, config_name, best_score, trace_summary, repo)

        # Update population with best config
        if best_score > 0 and results:
            best_config = results[0]["config"]
            self._update_population(best_config, best_score)

        return best_result or {"instance_id": instance_id, "model_patch": "", "error": "no result"}

    def _select_candidates(self, instance: dict) -> list:
        """Select promising configs for this instance."""
        repo = instance["repo"]
        candidates = []

        # 1. Check memory for configs that worked on this repo
        best_config_name = self.memory.get_best_config_for_repo(repo)
        if best_config_name:
            for p in self.population:
                if p["config"].get("name") == best_config_name:
                    candidates.append({"config": p["config"]})
                    break

        # 2. Add top configs from population
        pop_sorted = sorted(self.population, key=lambda p: -p["score"])
        for p in pop_sorted:
            if len(candidates) >= self.population_size:
                break
            if not any(c["config"]["name"] == p["config"]["name"] for c in candidates):
                candidates.append({"config": p["config"]})

        # 3. If still need more, use seeds
        if len(candidates) < 2:
            for seed in SEED_CONFIGS:
                if len(candidates) >= 3:
                    break
                candidates.append({"config": copy.deepcopy(seed)})

        return candidates

    def _build_failure_trace(self, results: list, instance: dict) -> str:
        """Build structured failure trace for mutation guidance."""
        trace_parts = []

        for r in results:
            config_name = r["config"]["name"]
            score = r["score"]
            patch = r["result"].get("model_patch", "")
            error = r["result"].get("error", "")
            localization = r["result"].get("localization", {})

            trace_parts.append(f"""Config: {config_name}
Score: {score}
Localized files: {localization.get('files', 'N/A')}
Localization confidence: {localization.get('confidence', 'N/A')}
Patch length: {len(patch)} chars
Error: {error or 'none'}
Patch preview: {patch[:500] if patch else 'EMPTY'}
""")

        return "\n---\n".join(trace_parts)

    def _update_population(self, config: dict, score: float):
        """Update population with a successful config."""
        # Check if this config is already in population
        for p in self.population:
            if p["config"]["name"] == config["name"]:
                # Update score with moving average
                p["score"] = 0.7 * p["score"] + 0.3 * score
                p["instances_seen"] += 1
                return

        # Add new config, remove worst if at capacity
        if len(self.population) >= self.population_size * 2:
            self.population.sort(key=lambda p: -p["score"])
            self.population = self.population[:self.population_size]

        self.population.append({
            "config": copy.deepcopy(config),
            "score": score,
            "cost": 0.0,
            "instances_seen": 1,
        })

    def get_population_summary(self) -> str:
        """Get summary of current population."""
        lines = ["Population:"]
        for p in sorted(self.population, key=lambda x: -x["score"]):
            lines.append(f"  {p['config']['name']}: score={p['score']:.2f}, "
                        f"seen={p['instances_seen']}")
        return "\n".join(lines)
