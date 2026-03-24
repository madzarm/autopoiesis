#!/usr/bin/env python3
"""Genesis Multi-Benchmark — evolve genomes across GSM8K + HumanEval simultaneously.

Forces evolution to discover universal adaptive strategies rather than
benchmark-specific tricks. The phenotypic plasticity should emerge naturally
when the same genome must handle both math and code tasks.
"""

import json
import random
import time
import copy

from genesis import (
    Genome, Stage, execute_genome, fast_eval,
    random_genome, mutate_genome, crossover_genomes, llm_evolve_genome,
    SEED_GENOMES, CHEAP,
)
from evaluate import load_gsm8k, load_humaneval
from llm import get_session_cost, reset_cost_tracking, STRONG


def multi_benchmark_eval(genome: Genome, benchmarks: dict) -> dict:
    """Evaluate genome across multiple benchmarks. Returns per-bench scores + weighted avg."""
    results = {}
    total_cost = 0.0

    for bench_name, samples in benchmarks.items():
        reset_cost_tracking()
        result = fast_eval(genome, samples, bench_name)
        results[bench_name] = result["score"]
        total_cost += get_session_cost()

    scores = list(results.values())
    avg = sum(scores) / len(scores) if scores else 0.0

    return {
        "scores": results,
        "avg_score": round(avg, 2),
        "cost": round(total_cost, 4),
    }


def run_genesis_multi(
    n_gsm: int = 20,
    n_he: int = 15,
    population_size: int = 8,
    generations: int = 15,
    elite_size: int = 2,
    seed: int = 42,
):
    """Evolve genomes across multiple benchmarks simultaneously."""
    benchmarks = {
        "gsm8k": load_gsm8k(n=n_gsm, seed=seed),
        "humaneval": load_humaneval(n=n_he, seed=seed),
    }
    print(f"═══ Genesis Multi-Benchmark ═══")
    print(f"Benchmarks: {', '.join(f'{k}={len(v)}' for k, v in benchmarks.items())}")
    print(f"Population: {population_size}, Generations: {generations}")
    print()

    # Initialize
    population = []
    init_genomes = SEED_GENOMES[:population_size]
    while len(init_genomes) < population_size:
        init_genomes.append(random_genome(f"random_{len(init_genomes)}"))

    print("── Initial Population ──")
    for genome in init_genomes:
        result = multi_benchmark_eval(genome, benchmarks)
        population.append({
            "genome": genome,
            "avg_score": result["avg_score"],
            "scores": result["scores"],
            "cost": result["cost"],
            "errors": [],
        })
        print(f"  {genome.name:25s} | avg={result['avg_score']:5.1f}% | {result['scores']}")

    best_ever = max(population, key=lambda x: x["avg_score"])
    print(f"\nBest: {best_ever['genome'].name} = {best_ever['avg_score']}%")

    for gen in range(generations):
        print(f"\n── Generation {gen+1}/{generations} ──")
        population.sort(key=lambda x: x["avg_score"], reverse=True)
        new_pop = population[:elite_size]

        while len(new_pop) < population_size:
            roll = random.random()
            if roll < 0.35:
                parent = random.choice(population[:4])
                child = llm_evolve_genome(parent["genome"], parent["avg_score"],
                                          parent.get("errors", []))
                child.name = f"llm_g{gen+1}_{len(new_pop)}"
                method = "llm_evolve"
            elif roll < 0.55:
                parent = random.choice(population[:4])
                child = mutate_genome(parent["genome"])
                child.name = f"mut_g{gen+1}_{len(new_pop)}"
                method = "mutation"
            elif roll < 0.75:
                p1, p2 = random.sample(population[:5], 2)
                child = crossover_genomes(p1["genome"], p2["genome"])
                child.name = f"cross_g{gen+1}_{len(new_pop)}"
                method = "crossover"
            else:
                child = random_genome(f"rand_g{gen+1}_{len(new_pop)}")
                method = "random"

            child.model = CHEAP

            try:
                result = multi_benchmark_eval(child, benchmarks)
                entry = {
                    "genome": child,
                    "avg_score": result["avg_score"],
                    "scores": result["scores"],
                    "cost": result["cost"],
                    "errors": [],
                }
                new_pop.append(entry)

                marker = ""
                if result["avg_score"] > best_ever["avg_score"]:
                    best_ever = entry
                    marker = " *** NEW BEST ***"

                print(f"  [{method:10s}] {child.name:22s} | avg={result['avg_score']:5.1f}% | "
                      f"{result['scores']}{marker}")
            except Exception as e:
                print(f"  [{method:10s}] CRASHED: {str(e)[:60]}")
                new_pop.append({"genome": child, "avg_score": 0, "scores": {},
                                "cost": 0, "errors": []})

        population = new_pop
        avgs = [p["avg_score"] for p in population]
        print(f"  Gen {gen+1}: best={max(avgs):.1f}%, avg={sum(avgs)/len(avgs):.1f}%, "
              f"best_ever={best_ever['avg_score']:.1f}%")

    print(f"\n{'═'*60}")
    print(f"GENESIS MULTI-BENCHMARK COMPLETE")
    print(f"{'═'*60}")
    print(f"Best: {best_ever['genome'].name} = {best_ever['avg_score']}%")
    print(f"Scores: {best_ever['scores']}")
    print(f"\nGenome:")
    for i, s in enumerate(best_ever["genome"].stages):
        cond = f" [if {s.condition}]" if s.condition != "always" else ""
        term = " → STOP" if s.terminate_if_confident else ""
        print(f"  {i+1}. {s.action}(t={s.temperature}){cond}{term}")
        if s.system_prompt:
            print(f"     {s.system_prompt[:70]}...")

    with open("best_genesis_multi.json", "w") as f:
        json.dump(best_ever["genome"].to_dict(), f, indent=2)

    return best_ever


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gsm", type=int, default=20)
    parser.add_argument("--n-he", type=int, default=15)
    parser.add_argument("--pop", type=int, default=8)
    parser.add_argument("--gens", type=int, default=15)
    args = parser.parse_args()

    run_genesis_multi(n_gsm=args.n_gsm, n_he=args.n_he,
                      population_size=args.pop, generations=args.gens)
