#!/usr/bin/env python3
"""AIDE V2 Evolutionary Search — evolve agent architectures across benchmarks.

Key innovation: error-trace-conditioned mutation on ARCHITECTURE choices,
not just prompt parameters. The meta-agent analyzes failures and decides
whether to change the architecture (e.g., switch from CoT to code_solve)
or tune parameters within the current architecture.

Multi-benchmark fitness: agents are scored on a weighted combination
of benchmark scores to find generally capable designs.
"""

import json
import random
import time
import re
from datetime import datetime, timezone
from collections import Counter

from agents_v2 import AgentV2Config, run_agent_v2, V2_CONFIGS
from evaluate import (
    load_gsm8k, load_arc, load_drop,
    evaluate_math_accuracy, evaluate_arc_accuracy, evaluate_drop_f1,
    extract_text_answer,
)
from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking


V2_SEARCH_SPACE = {
    "architecture": ["cot", "code_solve", "plan_solve_verify", "classify_route",
                     "ensemble_diverse", "progressive_refine"],
    "temperature": [0.0, 0.1, 0.3, 0.5, 0.7],
    "code_max_attempts": [1, 2, 3],
    "verify_strategy": ["recompute", "substitute", "alternative_method"],
    "refine_rounds": [1, 2, 3],
    "ensemble_n": [2, 3, 5],
}


def evaluate_multi_benchmark(config: AgentV2Config, benchmarks: dict[str, list]) -> dict:
    """Evaluate a config across multiple benchmarks. Returns per-benchmark scores and weighted avg."""
    results = {}
    total_cost = 0.0

    for bench_name, samples in benchmarks.items():
        reset_cost_tracking()
        try:
            if bench_name == "gsm8k":
                agent_fn = lambda q: run_agent_v2(config, q, answer_format="numeric")
                result = evaluate_math_accuracy(agent_fn, samples, "gsm8k")
            elif bench_name == "arc":
                agent_fn = lambda q, c: run_agent_v2(
                    config,
                    f"{q}\n\nChoices:\n{c}\n\nSelect the correct answer letter.",
                    answer_format="mc"
                )
                result = evaluate_arc_accuracy(agent_fn, samples)
            elif bench_name == "drop":
                agent_fn = lambda p, q: run_agent_v2(config, q, passage=p, answer_format="text")
                result = evaluate_drop_f1(agent_fn, samples)
            else:
                continue

            results[bench_name] = {
                "score": result["score"],
                "cost": result.get("cost_usd", 0),
                "details": result.get("details", []),
            }
            total_cost += result.get("cost_usd", 0)
        except Exception as e:
            results[bench_name] = {"score": 0.0, "cost": 0, "details": [], "error": str(e)}

    # Weighted average (equal weights for now)
    scores = [r["score"] for r in results.values() if "error" not in r]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "benchmarks": results,
        "avg_score": round(avg_score, 2),
        "total_cost": round(total_cost, 4),
    }


def v2_error_trace_mutation(
    config: AgentV2Config,
    eval_result: dict,
    model: str = MID,
) -> AgentV2Config:
    """Analyze errors across benchmarks and propose architecture-level changes."""
    system = (
        "You are an AI architect who designs agent systems. "
        "Analyze failures and propose architectural improvements. "
        "Return ONLY a valid JSON config."
    )

    # Build error summary
    error_summary = ""
    for bench, data in eval_result.get("benchmarks", {}).items():
        score = data.get("score", 0)
        error_summary += f"\n{bench}: {score}%"
        if score < 80:
            details = data.get("details", [])
            errors = [d for d in details if not d.get("correct", True) and d.get("f1", 1.0) < 0.5][:3]
            if errors:
                error_summary += " — example errors: " + str(errors[:2])[:200]

    config_json = json.dumps(config.model_dump(), indent=2)
    space_json = json.dumps(V2_SEARCH_SPACE, indent=2)

    prompt = f"""## Current Agent
{config_json}

## Performance
{error_summary}
Average: {eval_result.get('avg_score', 0)}%

## Architecture Options
{space_json}

Additional string fields: persona, custom_instructions
Ensemble architectures (list): e.g., ["cot", "code_solve", "plan_solve_verify"]
Model must be "{CHEAP}"

## Task
Propose an improved config. Think about:
- Which benchmarks are weak? Would a different architecture help?
- "code_solve" excels at math but needs code execution
- "plan_solve_verify" is thorough but slow
- "progressive_refine" helps catch errors on knowledge tasks
- "ensemble_diverse" combines strategies but is expensive
- A good persona and custom_instructions can add 2-5% on any architecture

Return ONLY the complete JSON config.
"""

    result = call_llm(
        prompt=prompt, system=system, model=model,
        temperature=0.7, max_tokens=2048, json_mode=True,
    )

    try:
        data = json.loads(result["content"])
        data["model"] = CHEAP
        # Ensure ensemble_architectures is a list
        if "ensemble_architectures" in data and isinstance(data["ensemble_architectures"], str):
            data["ensemble_architectures"] = [data["ensemble_architectures"]]
        return AgentV2Config(**data)
    except Exception:
        # Fallback: random V2 config
        return AgentV2Config(
            name="fallback",
            architecture=random.choice(V2_SEARCH_SPACE["architecture"]),
            model=CHEAP,
        )


def run_v2_evolution(
    n_samples: int = 30,
    population_size: int = 5,
    generations: int = 8,
    elite_size: int = 2,
    seed: int = 42,
):
    """Run V2 evolutionary search across multiple benchmarks."""
    # Load benchmarks
    print("Loading benchmarks...")
    benchmarks = {
        "gsm8k": load_gsm8k(n=n_samples, seed=seed),
        "arc": load_arc(n=n_samples, seed=seed),
    }
    print(f"Loaded: {', '.join(f'{k}={len(v)}' for k, v in benchmarks.items())}")

    # Initialize population with diverse architectures
    init_configs = [
        AgentV2Config(name="cot_base", architecture="cot", model=CHEAP),
        AgentV2Config(name="code_base", architecture="code_solve", model=CHEAP),
        AgentV2Config(name="psv_base", architecture="plan_solve_verify", model=CHEAP),
        AgentV2Config(name="pr_base", architecture="progressive_refine", model=CHEAP, refine_rounds=2),
        AgentV2Config(name="ens_base", architecture="ensemble_diverse", model=CHEAP,
                      ensemble_architectures=["cot", "code_solve"], ensemble_n=2),
    ][:population_size]

    # Evaluate initial population
    print("\n--- Initial Population ---")
    population = []
    for config in init_configs:
        result = evaluate_multi_benchmark(config, benchmarks)
        population.append({"config": config, "eval": result})
        scores = {k: v["score"] for k, v in result["benchmarks"].items()}
        print(f"  {config.name} ({config.architecture}): avg={result['avg_score']}% | {scores}")

    best_ever = max(population, key=lambda x: x["eval"]["avg_score"])
    print(f"\nBest initial: {best_ever['config'].name} = {best_ever['eval']['avg_score']}%")

    # Evolution loop
    for gen in range(generations):
        print(f"\n{'='*50}")
        print(f"Generation {gen+1}/{generations}")
        print(f"{'='*50}")

        population.sort(key=lambda x: x["eval"]["avg_score"], reverse=True)
        new_population = population[:elite_size]

        while len(new_population) < population_size:
            # Select parent
            parent = random.choice(population[:3])  # Top 3

            # Error-trace mutation (the AIDE innovation)
            child = v2_error_trace_mutation(
                parent["config"],
                parent["eval"],
            )
            child.name = f"evo_g{gen+1}_{len(new_population)}"
            child.model = CHEAP

            # Evaluate
            try:
                result = evaluate_multi_benchmark(child, benchmarks)
                new_population.append({"config": child, "eval": result})
                scores = {k: v["score"] for k, v in result["benchmarks"].items()}
                print(f"  {child.name} ({child.architecture}): avg={result['avg_score']}% | {scores}")

                if result["avg_score"] > best_ever["eval"]["avg_score"]:
                    best_ever = {"config": child, "eval": result}
                    print(f"    *** NEW BEST: {result['avg_score']}% ***")
            except Exception as e:
                print(f"  {child.name}: CRASHED - {e}")

        population = new_population
        scores_list = [p["eval"]["avg_score"] for p in population]
        print(f"\nGen {gen+1}: best={max(scores_list):.1f}%, avg={sum(scores_list)/len(scores_list):.1f}%")

    # Final
    print(f"\n{'='*50}")
    print(f"EVOLUTION COMPLETE")
    print(f"{'='*50}")
    print(f"Best: {best_ever['config'].name} ({best_ever['config'].architecture})")
    print(f"Score: {best_ever['eval']['avg_score']}%")
    for k, v in best_ever["eval"]["benchmarks"].items():
        print(f"  {k}: {v['score']}%")
    print(f"\nConfig:")
    print(json.dumps(best_ever["config"].model_dump(), indent=2))

    with open("best_v2_config.json", "w") as f:
        json.dump(best_ever["config"].model_dump(), f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--pop", type=int, default=5)
    parser.add_argument("--gens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_v2_evolution(n_samples=args.n, population_size=args.pop,
                     generations=args.gens, seed=args.seed)
