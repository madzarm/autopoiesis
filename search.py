"""Search algorithm for discovering novel agent designs.

Implements a hybrid search combining:
- LLM-guided proposal (meta-agent generates new configs)
- Execution-trace-conditioned mutation
- Archive-based exploration (quality-diversity)
- Cost-aware multi-objective optimization
"""

import json
import random
import copy
from typing import Optional
from agents import AgentConfig, BASELINE_CONFIGS, run_agent
from archive import load_archive, get_best, get_archive_summary
from llm import call_llm, STRONG, MID, CHEAP


# All mutable fields and their valid values
SEARCH_SPACE = {
    "reasoning": ["direct", "cot", "cot_sc", "decompose", "analogy", "abstract"],
    "planning": ["none", "step_by_step", "recursive", "divide_conquer"],
    "reflection": ["none", "self_check", "self_refine", "critic"],
    "ensemble": ["none", "majority_vote", "best_of_n", "debate"],
    "ensemble_n": [1, 3, 5, 7],
    "output_format": ["free", "structured", "step_numbered"],
    "temperature": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    "reflection_rounds": [1, 2, 3],
    "decompose_strategy": ["sequential", "parallel", "tree"],
}


def random_config(name: str = "random") -> AgentConfig:
    """Generate a random agent config from the search space."""
    return AgentConfig(
        name=name,
        reasoning=random.choice(SEARCH_SPACE["reasoning"]),
        planning=random.choice(SEARCH_SPACE["planning"]),
        reflection=random.choice(SEARCH_SPACE["reflection"]),
        ensemble=random.choice(SEARCH_SPACE["ensemble"]),
        ensemble_n=random.choice(SEARCH_SPACE["ensemble_n"]),
        output_format=random.choice(SEARCH_SPACE["output_format"]),
        temperature=random.choice(SEARCH_SPACE["temperature"]),
        model=CHEAP,
    )


def mutate_config(config: AgentConfig, n_mutations: int = 1) -> AgentConfig:
    """Apply random mutations to a config."""
    new = config.model_copy()
    fields = list(SEARCH_SPACE.keys())

    for _ in range(n_mutations):
        field = random.choice(fields)
        new_val = random.choice(SEARCH_SPACE[field])
        setattr(new, field, new_val)

    # Fix ensemble_n consistency
    if new.ensemble == "none":
        new.ensemble_n = 1

    return new


def crossover_configs(parent1: AgentConfig, parent2: AgentConfig) -> AgentConfig:
    """Create a child config by crossing over two parents."""
    child_data = {}
    for field in SEARCH_SPACE:
        if random.random() < 0.5:
            child_data[field] = getattr(parent1, field)
        else:
            child_data[field] = getattr(parent2, field)

    child = AgentConfig(
        name=f"cross_{parent1.name}_{parent2.name}",
        model=CHEAP,
        **child_data,
    )

    # Fix consistency
    if child.ensemble == "none":
        child.ensemble_n = 1

    return child


def llm_propose_config(
    archive_summary: str,
    benchmark: str,
    error_examples: list[dict] = None,
    model: str = STRONG,
) -> AgentConfig:
    """Use the meta-agent LLM to propose a new agent configuration.

    This is the core of the search — the LLM analyzes what has worked,
    what has failed, and proposes a novel configuration.
    """
    system = (
        "You are a meta-agent that designs AI agent configurations. "
        "You analyze past results and propose new configurations that might perform better. "
        "You must return a valid JSON config."
    )

    search_space_desc = json.dumps(SEARCH_SPACE, indent=2)

    prompt = f"""You are designing agents for the '{benchmark}' benchmark.

## Search Space
The following fields can be set to these values:
{search_space_desc}

Additionally, you can set:
- persona: A custom system prompt prefix (string, be creative but relevant)
- custom_instructions: Additional instructions to append (string)
- model: Always use "{CHEAP}"

## Archive Summary (past results)
{archive_summary}

## Error Analysis
"""

    if error_examples:
        prompt += "Here are examples where recent agents got the WRONG answer:\n\n"
        for ex in error_examples[:5]:
            prompt += f"- Question: {ex.get('question', 'N/A')[:200]}\n"
            prompt += f"  Gold: {ex.get('gold', 'N/A')}, Predicted: {ex.get('predicted', 'N/A')}\n\n"
    else:
        prompt += "No error examples available yet.\n"

    prompt += """
## Task
Propose a NEW agent configuration that is likely to score higher than anything in the archive.
Think about:
1. What patterns of configs have worked well?
2. What errors keep occurring and how might a different config avoid them?
3. What combinations haven't been tried yet?
4. Can you invent creative personas or custom instructions that might help?

Return ONLY a JSON object with the config fields. Example:
{
    "name": "creative_name",
    "reasoning": "cot",
    "planning": "step_by_step",
    "reflection": "self_refine",
    "ensemble": "none",
    "ensemble_n": 1,
    "output_format": "step_numbered",
    "temperature": 0.0,
    "persona": "You are a world-class mathematician...",
    "custom_instructions": "Double-check all arithmetic...",
    "reflection_rounds": 1
}
"""

    result = call_llm(
        prompt=prompt,
        system=system,
        model=model,
        temperature=0.8,
        max_tokens=2048,
        json_mode=True,
    )

    try:
        config_data = json.loads(result["content"])
        config_data["model"] = CHEAP  # Force cheap model for inner agent
        config = AgentConfig(**config_data)
        return config
    except (json.JSONDecodeError, Exception) as e:
        # Fallback: return a random mutation of the best config
        best = get_best(benchmark, top_k=1)
        if best:
            base = AgentConfig(**best[0]["config"])
            return mutate_config(base, n_mutations=2)
        return random_config(name="fallback_random")


def llm_mutate_config(
    config: AgentConfig,
    score: float,
    error_examples: list[dict] = None,
    model: str = MID,
) -> AgentConfig:
    """Use LLM to intelligently mutate a config based on its performance."""
    system = (
        "You are a meta-agent that improves AI agent configurations. "
        "Given a config and its score, suggest targeted modifications to improve it. "
        "Return ONLY a valid JSON config."
    )

    config_json = json.dumps(config.model_dump(), indent=2)
    search_space_desc = json.dumps(SEARCH_SPACE, indent=2)

    prompt = f"""This agent config scored {score:.1f}% on the benchmark:

{config_json}

## Valid search space:
{search_space_desc}

## Errors made by this agent:
"""
    if error_examples:
        for ex in error_examples[:3]:
            prompt += f"- Q: {ex.get('question', '')[:150]} | Gold: {ex.get('gold', '')} | Predicted: {ex.get('predicted', '')}\n"

    prompt += f"""
## Task
Modify this config to fix the errors and improve the score.
You may change any field. You may also modify persona and custom_instructions.
Make targeted changes — don't change everything at once.
Return ONLY the complete modified JSON config (all fields).
Set model to "{CHEAP}".
"""

    result = call_llm(
        prompt=prompt,
        system=system,
        model=model,
        temperature=0.7,
        max_tokens=2048,
        json_mode=True,
    )

    try:
        config_data = json.loads(result["content"])
        config_data["model"] = CHEAP
        return AgentConfig(**config_data)
    except Exception:
        return mutate_config(config, n_mutations=2)


def get_error_examples(details: list[dict], samples: list[dict], max_examples: int = 5) -> list[dict]:
    """Extract error examples from evaluation details."""
    errors = []
    for d in details:
        if not d.get("correct", True) and not d.get("error"):
            idx = d.get("idx", 0)
            if idx < len(samples):
                errors.append({
                    "question": samples[idx].get("question", ""),
                    "gold": d.get("gold", ""),
                    "predicted": d.get("predicted", ""),
                })
    return errors[:max_examples]
