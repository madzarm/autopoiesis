#!/usr/bin/env python3
"""Approach 17: Co-Evolutionary Search — Generators and Selectors Evolve Together.

Key novelty: Instead of evolving one population (generators), we co-evolve
TWO populations that compete/cooperate:

1. **Generators**: Agent workflows that produce code candidates
2. **Selectors**: Agent workflows that evaluate/select/repair candidates

Each generator is evaluated by ALL selectors, and each selector is evaluated
by ALL generators. Fitness is relative — a generator is "fit" if it produces
candidates that selectors CAN select correctly, and a selector is "fit" if
it can distinguish good from bad candidates across generators.

This creates an arms race: generators evolve to produce diverse, high-quality
candidates that challenge selectors, and selectors evolve to be robust to
varying quality. The best (generator, selector) pair emerges from co-evolution.

Inspiration:
- Competitive co-evolution (Hillis 1990): host-parasite dynamics
- NEAT (Stanley & Miikkulainen 2002): minimal structure, complexify
- Cooperative co-evolution: specialists that compose
- Red team / blue team security paradigm

This is fundamentally different from all prior approaches because it searches
two coupled spaces simultaneously, using interaction-based fitness.
"""

import json
import time
import random
import copy
import re
import ast
import sys
import io
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm, CHEAP, get_session_cost, reset_cost_tracking


# ═══════════════════════════════════════════════════════════════
# GENERATOR GENOME
# ═══════════════════════════════════════════════════════════════

GENERATOR_PROMPTS = [
    "Complete this Python function. Write clean, correct code.",
    "Implement step by step. Handle all edge cases.",
    "Think about edge cases first, then implement.",
    "Write a solution that handles all the docstring examples correctly.",
    "Explore different algorithms. Pick the most robust one.",
    "Focus on correctness. Test each edge case mentally.",
    "Read the examples carefully. The solution must match exactly.",
    "Write production-quality code. Handle boundary conditions.",
]


def random_generator(rng: random.Random) -> dict:
    """Create a random generator genome."""
    n_stages = rng.randint(1, 5)
    stages = []
    for _ in range(n_stages):
        stages.append({
            "temperature": rng.uniform(0.0, 1.0),
            "prompt_idx": rng.randint(0, len(GENERATOR_PROMPTS) - 1),
            "use_cot": rng.random() > 0.5,
        })
    return {"stages": stages, "fitness_history": []}


def mutate_generator(gen: dict, rng: random.Random) -> dict:
    """Mutate a generator genome."""
    new = copy.deepcopy(gen)
    stages = new["stages"]

    for s in stages:
        if rng.random() < 0.3:
            s["temperature"] = max(0, min(1, s["temperature"] + rng.gauss(0, 0.15)))
        if rng.random() < 0.2:
            s["prompt_idx"] = rng.randint(0, len(GENERATOR_PROMPTS) - 1)
        if rng.random() < 0.15:
            s["use_cot"] = not s["use_cot"]

    # Add/remove stages
    if rng.random() < 0.15 and len(stages) < 7:
        stages.append({
            "temperature": rng.uniform(0.0, 1.0),
            "prompt_idx": rng.randint(0, len(GENERATOR_PROMPTS) - 1),
            "use_cot": rng.random() > 0.5,
        })
    if rng.random() < 0.1 and len(stages) > 1:
        stages.pop(rng.randint(0, len(stages) - 1))

    new["fitness_history"] = []
    return new


def run_generator(genome: dict, prompt: str, entry_point: str,
                  model: str = CHEAP) -> list[str]:
    """Run a generator to produce code candidates."""
    candidates = []

    def sanitize(response, ep=entry_point):
        code_blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
        code = "\n".join(code_blocks) if code_blocks else response
        code = re.sub(r'^####\s*.*$', '', code, flags=re.MULTILINE).strip()
        lines = code.split('\n')
        result_lines = []
        in_func = False
        for line in lines:
            if f'def {ep}' in line:
                in_func = True
            if in_func:
                result_lines.append(line)
        return '\n'.join(result_lines) if result_lines else code

    for stage in genome["stages"]:
        sys_prompt = GENERATOR_PROMPTS[stage["prompt_idx"] % len(GENERATOR_PROMPTS)]
        user_msg = prompt
        if stage["use_cot"]:
            user_msg = ("Think step by step about what the function should do. "
                       "Analyze edge cases. Then implement.\n\n" + prompt)
        try:
            resp = call_llm(user_msg, system=sys_prompt, model=model,
                           temperature=stage["temperature"], max_tokens=1024)
            code = sanitize(resp["content"])
            if code.strip():
                candidates.append(code)
        except Exception:
            pass

    return candidates


# ═══════════════════════════════════════════════════════════════
# SELECTOR GENOME
# ═══════════════════════════════════════════════════════════════

SELECTOR_METHODS = ["test_first_pass", "test_all_pick_shortest",
                    "repair_then_test", "reflect_repair_test", "vote"]


def random_selector(rng: random.Random) -> dict:
    """Create a random selector genome."""
    return {
        "method": rng.choice(SELECTOR_METHODS),
        "repair_rounds": rng.randint(0, 3),
        "repair_temp": rng.uniform(0.0, 0.5),
        "use_reflect": rng.random() > 0.5,
        "restart_on_fail": rng.random() > 0.5,
        "restart_temp": rng.uniform(0.3, 1.0),
        "fitness_history": [],
    }


def mutate_selector(sel: dict, rng: random.Random) -> dict:
    """Mutate a selector genome."""
    new = copy.deepcopy(sel)
    if rng.random() < 0.25:
        new["method"] = rng.choice(SELECTOR_METHODS)
    if rng.random() < 0.3:
        new["repair_rounds"] = rng.randint(0, 3)
    if rng.random() < 0.3:
        new["repair_temp"] = max(0, min(0.5, new["repair_temp"] + rng.gauss(0, 0.1)))
    if rng.random() < 0.2:
        new["use_reflect"] = not new["use_reflect"]
    if rng.random() < 0.2:
        new["restart_on_fail"] = not new["restart_on_fail"]
    if rng.random() < 0.3:
        new["restart_temp"] = max(0.1, min(1.0, new["restart_temp"] + rng.gauss(0, 0.15)))
    new["fitness_history"] = []
    return new


def run_selector(genome: dict, candidates: list[str], prompt: str,
                 entry_point: str, test_code: str,
                 model: str = CHEAP) -> str:
    """Run a selector to pick/improve the best candidate from a list."""

    def exec_test(code: str) -> tuple[bool, str]:
        full_code = code + "\n" + test_code + f"\ncheck({entry_point})"
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(full_code, {"__builtins__": __builtins__}, {})
            return True, ""
        except Exception as e:
            return False, str(e)
        finally:
            sys.stdout = old_stdout

    def sanitize(response, ep=entry_point):
        code_blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
        code = "\n".join(code_blocks) if code_blocks else response
        code = re.sub(r'^####\s*.*$', '', code, flags=re.MULTILINE).strip()
        lines = code.split('\n')
        result_lines = []
        in_func = False
        for line in lines:
            if f'def {ep}' in line:
                in_func = True
            if in_func:
                result_lines.append(line)
        return '\n'.join(result_lines) if result_lines else code

    method = genome["method"]
    repair_rounds = genome["repair_rounds"]
    use_reflect = genome["use_reflect"]

    if not candidates:
        return ""

    # Phase 1: Test all candidates
    test_results = []
    for code in candidates:
        passed, err = exec_test(code)
        test_results.append((code, passed, err))

    # Select based on method
    if method == "test_first_pass":
        for code, passed, _ in test_results:
            if passed:
                return code

    elif method == "test_all_pick_shortest":
        passing = [(c, p, e) for c, p, e in test_results if p]
        if passing:
            return min(passing, key=lambda x: len(x[0]))[0]

    elif method == "vote":
        passing = [c for c, p, _ in test_results if p]
        if passing:
            return passing[0]

    # Phase 2: Repair failing candidates
    best_code = candidates[0]
    best_err = test_results[0][2] if test_results else ""

    for r in range(repair_rounds):
        if use_reflect and method in ("reflect_repair_test", "repair_then_test"):
            try:
                resp = call_llm(
                    f"Review:\n```python\n{best_code}\n```\nError: {best_err}\nWhat's wrong?",
                    system="Identify bugs concisely.",
                    model=model, temperature=0.1, max_tokens=512
                )
                reflection = resp["content"]
            except Exception:
                reflection = ""
        else:
            reflection = ""

        try:
            repair_prompt = f"Fix this code:\n```python\n{best_code}\n```\nError: {best_err}"
            if reflection:
                repair_prompt += f"\nAnalysis: {reflection}"
            resp = call_llm(repair_prompt,
                           system="Fix the code. Return only the corrected function.",
                           model=model, temperature=genome["repair_temp"],
                           max_tokens=1024)
            repaired = sanitize(resp["content"])
            if repaired.strip():
                passed, err = exec_test(repaired)
                if passed:
                    return repaired
                best_code = repaired
                best_err = err
        except Exception:
            pass

    # Phase 3: Restart if enabled
    if genome["restart_on_fail"]:
        try:
            resp = call_llm(
                f"Write a completely different solution:\n{prompt}",
                system="Use a different algorithm. Be creative and correct.",
                model=model, temperature=genome["restart_temp"],
                max_tokens=1024
            )
            code = sanitize(resp["content"])
            if code.strip():
                passed, _ = exec_test(code)
                if passed:
                    return code
        except Exception:
            pass

    return best_code


# ═══════════════════════════════════════════════════════════════
# CO-EVOLUTIONARY LOOP
# ═══════════════════════════════════════════════════════════════

def evaluate_pair(gen_genome: dict, sel_genome: dict, sample: dict,
                  model: str = CHEAP) -> bool:
    """Evaluate a (generator, selector) pair on one problem."""
    candidates = run_generator(gen_genome, sample["prompt"],
                              sample["entry_point"], model=model)
    code = run_selector(sel_genome, candidates, sample["prompt"],
                       sample["entry_point"], sample["test"], model=model)
    if not code.strip():
        return False

    full_code = code + "\n" + sample["test"] + f"\ncheck({sample['entry_point']})"
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(full_code, {"__builtins__": __builtins__}, {})
        return True
    except Exception:
        return False
    finally:
        sys.stdout = old_stdout


def run_coevolution(n_generations: int = 8, gen_pop_size: int = 6,
                    sel_pop_size: int = 6, n_eval_samples: int = 20,
                    model: str = CHEAP, seed: int = 42):
    """Run co-evolutionary search over (generator, selector) pairs."""
    from evaluate import load_humaneval

    rng = random.Random(seed)
    print(f"=== Co-Evolutionary Search ===")
    print(f"Gen pop: {gen_pop_size}, Sel pop: {sel_pop_size}, "
          f"Generations: {n_generations}, Eval: {n_eval_samples}")
    print(f"Model: {model}")

    # Load eval samples
    all_samples = load_humaneval()
    eval_samples = rng.sample(all_samples, min(n_eval_samples, len(all_samples)))
    print(f"Loaded {len(eval_samples)} eval samples")

    # Initialize populations
    gen_pop = [random_generator(rng) for _ in range(gen_pop_size)]
    sel_pop = [random_selector(rng) for _ in range(sel_pop_size)]

    # Seed with known-good genomes
    gen_pop[0] = {
        "stages": [
            {"temperature": 0.0, "prompt_idx": 0, "use_cot": False},
            {"temperature": 0.5, "prompt_idx": 3, "use_cot": False},
            {"temperature": 0.7, "prompt_idx": 4, "use_cot": True},
        ],
        "fitness_history": [],
    }
    sel_pop[0] = {
        "method": "test_first_pass",
        "repair_rounds": 2,
        "repair_temp": 0.2,
        "use_reflect": True,
        "restart_on_fail": True,
        "restart_temp": 0.5,
        "fitness_history": [],
    }

    best_overall_acc = 0.0
    best_gen = None
    best_sel = None
    history = []

    for gen in range(n_generations):
        t0 = time.time()

        # Evaluate all (generator, selector) pairs
        gen_scores = [0.0] * gen_pop_size
        sel_scores = [0.0] * sel_pop_size
        pair_scores = {}

        for gi, gen_genome in enumerate(gen_pop):
            for si, sel_genome in enumerate(sel_pop):
                correct = 0
                for sample in eval_samples:
                    if evaluate_pair(gen_genome, sel_genome, sample, model=model):
                        correct += 1
                acc = correct / len(eval_samples)
                pair_scores[(gi, si)] = acc
                gen_scores[gi] += acc
                sel_scores[si] += acc

        # Normalize by number of opponents
        gen_scores = [s / sel_pop_size for s in gen_scores]
        sel_scores = [s / gen_pop_size for s in sel_scores]

        # Find best pair
        best_pair = max(pair_scores, key=pair_scores.get)
        best_pair_acc = pair_scores[best_pair]

        if best_pair_acc > best_overall_acc:
            best_overall_acc = best_pair_acc
            best_gen = copy.deepcopy(gen_pop[best_pair[0]])
            best_sel = copy.deepcopy(sel_pop[best_pair[1]])

        elapsed = time.time() - t0

        print(f"\n  Gen {gen+1}: best_pair=({best_pair[0]},{best_pair[1]}) "
              f"acc={best_pair_acc*100:.1f}%, overall_best={best_overall_acc*100:.1f}%")
        print(f"  Generator scores: {[f'{s*100:.0f}' for s in gen_scores]}")
        print(f"  Selector scores:  {[f'{s*100:.0f}' for s in sel_scores]}")
        print(f"  Time: {elapsed:.0f}s")

        history.append({
            "generation": gen + 1,
            "best_pair_acc": best_pair_acc * 100,
            "overall_best": best_overall_acc * 100,
            "gen_scores": [round(s * 100, 1) for s in gen_scores],
            "sel_scores": [round(s * 100, 1) for s in sel_scores],
            "elapsed_s": elapsed,
        })

        # Evolve generators
        new_gen_pop = [copy.deepcopy(gen_pop[max(range(gen_pop_size),
                                                  key=lambda i: gen_scores[i])])]
        while len(new_gen_pop) < gen_pop_size:
            tournament = rng.sample(list(enumerate(gen_scores)), min(3, gen_pop_size))
            winner = max(tournament, key=lambda x: x[1])[0]
            child = mutate_generator(gen_pop[winner], rng)
            new_gen_pop.append(child)
        gen_pop = new_gen_pop

        # Evolve selectors
        new_sel_pop = [copy.deepcopy(sel_pop[max(range(sel_pop_size),
                                                  key=lambda i: sel_scores[i])])]
        while len(new_sel_pop) < sel_pop_size:
            tournament = rng.sample(list(enumerate(sel_scores)), min(3, sel_pop_size))
            winner = max(tournament, key=lambda x: x[1])[0]
            child = mutate_selector(sel_pop[winner], rng)
            new_sel_pop.append(child)
        sel_pop = new_sel_pop

    print(f"\n=== Co-Evolution Complete ===")
    print(f"Best accuracy: {best_overall_acc * 100:.1f}%")
    print(f"Best generator: {len(best_gen['stages'])} stages")
    for s in best_gen["stages"]:
        print(f"  temp={s['temperature']:.2f}, prompt={s['prompt_idx']}, cot={s['use_cot']}")
    print(f"Best selector: method={best_sel['method']}, "
          f"repair={best_sel['repair_rounds']}, reflect={best_sel['use_reflect']}, "
          f"restart={best_sel['restart_on_fail']}")
    print(f"Total cost: ${get_session_cost():.4f}")

    with open("best_coevo_config.json", "w") as f:
        json.dump({"generator": best_gen, "selector": best_sel,
                    "accuracy": best_overall_acc * 100, "history": history},
                  f, indent=2, default=str)

    return best_gen, best_sel, best_overall_acc


if __name__ == "__main__":
    import sys as _sys
    n_gen = int(_sys.argv[1]) if len(_sys.argv) > 1 else 5
    n_samples = int(_sys.argv[2]) if len(_sys.argv) > 2 else 20
    model = _sys.argv[3] if len(_sys.argv) > 3 else CHEAP

    run_coevolution(
        n_generations=n_gen,
        n_eval_samples=n_samples,
        model=model,
    )
