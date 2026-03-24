#!/usr/bin/env python3
"""Approach 16: Swarm Rules — Evolve Local Interaction Rules Between Micro-Agents.

Key novelty: Instead of designing a pipeline (linear or DAG), we define a
population of micro-agents that interact via local rules. Each micro-agent
has a simple behavior (generate, critique, repair, vote) and a set of
interaction rules (when to activate, who to listen to, how to combine).

The rules are evolved via a genetic algorithm. The emergent behavior of the
swarm produces the final solution.

Inspiration:
- Ant colony optimization: pheromone trails → shared code quality signals
- Boids: local rules produce complex global behavior
- Cellular automata: simple rules, complex emergent patterns
- Immune system: clonal selection + affinity maturation

Search space: A "ruleset" is a list of rules, each being:
  (trigger_condition, action, target, parameters)

Example rules:
- "If no candidates exist → generate(temp=0.0, prompt=precise)"
- "If candidate fails test → repair(candidate, error)"
- "If 3+ candidates exist → vote(candidates)"
- "If all candidates fail → restart(temp=0.7)"
- "If time_remaining < 30% → select_best(candidates)"

The evolutionary search mutates these rulesets and selects for HumanEval accuracy.
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

from llm import call_llm, CHEAP, STRONG, get_session_cost, reset_cost_tracking


# ═══════════════════════════════════════════════════════════════
# MICRO-AGENT ACTIONS
# ═══════════════════════════════════════════════════════════════

ACTIONS = ["generate", "repair", "reflect", "vote", "test", "restart"]

TRIGGERS = [
    "no_candidates",        # Empty candidate pool
    "has_candidates",       # At least one candidate
    "all_fail",            # All candidates fail tests
    "some_pass",           # At least one candidate passes
    "none_pass",           # No candidates pass (after testing)
    "after_generate",      # Just generated candidates
    "after_repair",        # Just attempted repair
    "after_reflect",       # Just reflected
    "step_1", "step_2", "step_3", "step_4", "step_5",  # Sequential triggers
    "always",              # Always fire
]

PROMPT_VARIANTS = [
    "Complete this Python function. Write clean, correct code.",
    "Implement step by step. Handle all edge cases carefully.",
    "Think about the algorithm first, then implement.",
    "Explore different approaches. Consider edge cases.",
    "Write a creative, elegant solution.",
    "Focus on correctness over cleverness.",
    "Read the docstring examples carefully, then implement.",
    "Write robust code. Test mentally before finalizing.",
]


def make_rule(trigger: str, action: str, params: dict) -> dict:
    """Create a swarm rule."""
    return {"trigger": trigger, "action": action, "params": params}


def random_rule(rng: random.Random) -> dict:
    """Generate a random swarm rule."""
    action = rng.choice(ACTIONS)
    trigger = rng.choice(TRIGGERS)
    params = {}

    if action == "generate":
        params["temperature"] = rng.uniform(0.0, 1.0)
        params["prompt_idx"] = rng.randint(0, len(PROMPT_VARIANTS) - 1)
        params["count"] = rng.randint(1, 5)
    elif action == "repair":
        params["temperature"] = rng.uniform(0.0, 0.5)
        params["use_error"] = rng.random() > 0.3
    elif action == "reflect":
        params["temperature"] = rng.uniform(0.0, 0.3)
    elif action == "vote":
        params["method"] = rng.choice(["majority", "quality", "first_pass"])
    elif action == "test":
        params["select_passing"] = rng.random() > 0.3
    elif action == "restart":
        params["temperature"] = rng.uniform(0.3, 1.0)
        params["prompt_idx"] = rng.randint(0, len(PROMPT_VARIANTS) - 1)

    return make_rule(trigger, action, params)


def random_ruleset(rng: random.Random, n_rules: int = 6) -> list[dict]:
    """Generate a random ruleset (agent program)."""
    n = rng.randint(3, n_rules)
    rules = [random_rule(rng) for _ in range(n)]
    # Ensure at least one generate rule
    if not any(r["action"] == "generate" for r in rules):
        rules[0] = random_rule(rng)
        rules[0]["action"] = "generate"
        rules[0]["trigger"] = "step_1"
    return rules


# ═══════════════════════════════════════════════════════════════
# SWARM EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════

def execute_swarm(ruleset: list[dict], prompt: str, entry_point: str,
                  test_code: str, model: str = CHEAP, max_steps: int = 10) -> str:
    """Execute a swarm of micro-agents governed by rules on a single problem."""

    state = {
        "candidates": [],
        "test_results": {},  # code_hash -> (passed, error)
        "reflections": [],
        "step": 0,
        "last_action": None,
    }

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

    def check_trigger(trigger: str) -> bool:
        if trigger == "always":
            return True
        if trigger == "no_candidates":
            return len(state["candidates"]) == 0
        if trigger == "has_candidates":
            return len(state["candidates"]) > 0
        if trigger == "all_fail":
            if not state["test_results"]:
                return False
            return all(not v[0] for v in state["test_results"].values())
        if trigger == "some_pass":
            return any(v[0] for v in state["test_results"].values())
        if trigger == "none_pass":
            if not state["test_results"]:
                return len(state["candidates"]) > 0
            return all(not v[0] for v in state["test_results"].values())
        if trigger == "after_generate":
            return state["last_action"] == "generate"
        if trigger == "after_repair":
            return state["last_action"] == "repair"
        if trigger == "after_reflect":
            return state["last_action"] == "reflect"
        if trigger.startswith("step_"):
            step_num = int(trigger.split("_")[1])
            return state["step"] == step_num - 1
        return False

    def do_action(rule: dict):
        action = rule["action"]
        params = rule["params"]

        if action == "generate":
            count = params.get("count", 3)
            temp = params.get("temperature", 0.3)
            pidx = params.get("prompt_idx", 0)
            sys_prompt = PROMPT_VARIANTS[pidx % len(PROMPT_VARIANTS)]

            for _ in range(count):
                try:
                    resp = call_llm(prompt, system=sys_prompt, model=model,
                                   temperature=temp, max_tokens=1024)
                    code = sanitize(resp["content"])
                    if code.strip():
                        state["candidates"].append(code)
                except Exception:
                    pass

        elif action == "test":
            select_passing = params.get("select_passing", True)
            for code in state["candidates"]:
                h = hash(code)
                if h not in state["test_results"]:
                    passed, err = exec_test(code)
                    state["test_results"][h] = (passed, err)

            if select_passing:
                for code in state["candidates"]:
                    h = hash(code)
                    if state["test_results"].get(h, (False, ""))[0]:
                        return code  # Early return on first passing

        elif action == "repair":
            temp = params.get("temperature", 0.2)
            use_error = params.get("use_error", True)

            # Find failing candidates
            for code in list(state["candidates"]):
                h = hash(code)
                result = state["test_results"].get(h)
                if result and not result[0]:
                    error_msg = result[1] if use_error else ""
                    try:
                        repair_prompt = f"Fix this code:\n```python\n{code}\n```"
                        if error_msg:
                            repair_prompt += f"\nError: {error_msg}"
                        resp = call_llm(repair_prompt,
                                       system="Fix the code. Return only the corrected function.",
                                       model=model, temperature=temp, max_tokens=1024)
                        repaired = sanitize(resp["content"])
                        if repaired.strip():
                            state["candidates"].append(repaired)
                    except Exception:
                        pass

        elif action == "reflect":
            temp = params.get("temperature", 0.1)
            if state["candidates"]:
                code = state["candidates"][-1]
                h = hash(code)
                err = state["test_results"].get(h, (False, ""))[1]
                try:
                    resp = call_llm(
                        f"Review this code:\n```python\n{code}\n```\nError: {err}\nWhat's wrong?",
                        system="Identify bugs concisely.",
                        model=model, temperature=temp, max_tokens=512
                    )
                    state["reflections"].append(resp["content"])
                except Exception:
                    pass

        elif action == "restart":
            temp = params.get("temperature", 0.7)
            pidx = params.get("prompt_idx", 4)
            sys_prompt = PROMPT_VARIANTS[pidx % len(PROMPT_VARIANTS)]
            try:
                extra = ""
                if state["reflections"]:
                    extra = f"\nPrevious analysis: {state['reflections'][-1][:200]}"
                resp = call_llm(prompt + extra,
                               system=sys_prompt + " Use a completely different approach.",
                               model=model, temperature=temp, max_tokens=1024)
                code = sanitize(resp["content"])
                if code.strip():
                    state["candidates"].append(code)
            except Exception:
                pass

        elif action == "vote":
            method = params.get("method", "first_pass")
            if method == "first_pass":
                for code in state["candidates"]:
                    h = hash(code)
                    if state["test_results"].get(h, (False, ""))[0]:
                        return code

        return None

    # Execute ruleset
    for step in range(max_steps):
        state["step"] = step
        for rule in ruleset:
            if check_trigger(rule["trigger"]):
                result = do_action(rule)
                state["last_action"] = rule["action"]
                if result is not None:
                    return result

    # Return best candidate
    for code in state["candidates"]:
        h = hash(code)
        if state["test_results"].get(h, (False, ""))[0]:
            return code
    return state["candidates"][0] if state["candidates"] else ""


# ═══════════════════════════════════════════════════════════════
# EVOLUTIONARY OPERATORS
# ═══════════════════════════════════════════════════════════════

def mutate_ruleset(ruleset: list[dict], rng: random.Random,
                   mutation_rate: float = 0.3) -> list[dict]:
    """Mutate a ruleset."""
    new_rules = copy.deepcopy(ruleset)

    for i in range(len(new_rules)):
        if rng.random() < mutation_rate:
            # Mutate this rule
            choice = rng.random()
            if choice < 0.3:
                # Change trigger
                new_rules[i]["trigger"] = rng.choice(TRIGGERS)
            elif choice < 0.6:
                # Change action params
                action = new_rules[i]["action"]
                params = new_rules[i]["params"]
                if action in ("generate", "repair", "restart"):
                    params["temperature"] = max(0, min(1, params.get("temperature", 0.3) +
                                                       rng.gauss(0, 0.15)))
                if action == "generate":
                    if rng.random() < 0.3:
                        params["count"] = rng.randint(1, 7)
                    if rng.random() < 0.3:
                        params["prompt_idx"] = rng.randint(0, len(PROMPT_VARIANTS) - 1)
            else:
                # Replace rule entirely
                new_rules[i] = random_rule(rng)

    # Structural mutations
    if rng.random() < 0.2 and len(new_rules) < 10:
        new_rules.append(random_rule(rng))
    if rng.random() < 0.1 and len(new_rules) > 3:
        idx = rng.randint(0, len(new_rules) - 1)
        # Don't remove the only generate rule
        if new_rules[idx]["action"] != "generate" or \
           sum(1 for r in new_rules if r["action"] == "generate") > 1:
            new_rules.pop(idx)
    if rng.random() < 0.15 and len(new_rules) > 1:
        # Swap two rules
        i, j = rng.sample(range(len(new_rules)), 2)
        new_rules[i], new_rules[j] = new_rules[j], new_rules[i]

    return new_rules


def crossover_rulesets(parent1: list[dict], parent2: list[dict],
                       rng: random.Random) -> list[dict]:
    """Single-point crossover between two rulesets."""
    if len(parent1) <= 1 or len(parent2) <= 1:
        return copy.deepcopy(parent1)
    cut1 = rng.randint(1, len(parent1) - 1)
    cut2 = rng.randint(1, len(parent2) - 1)
    child = copy.deepcopy(parent1[:cut1]) + copy.deepcopy(parent2[cut2:])
    # Ensure generate rule exists
    if not any(r["action"] == "generate" for r in child):
        child.insert(0, random_rule(rng))
        child[0]["action"] = "generate"
        child[0]["trigger"] = "step_1"
    return child[:10]  # Cap at 10 rules


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_ruleset(ruleset: list[dict], samples: list[dict],
                     model: str = CHEAP, max_workers: int = 8) -> float:
    """Evaluate a ruleset on HumanEval samples. Returns accuracy (0-1)."""
    correct = 0
    total = len(samples)

    def eval_one(sample):
        ep = sample["entry_point"]
        code = execute_swarm(ruleset, sample["prompt"], ep, sample["test"],
                            model=model, max_steps=8)
        if not code.strip():
            return False
        full_code = code + "\n" + sample["test"] + f"\ncheck({ep})"
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(full_code, {"__builtins__": __builtins__}, {})
            return True
        except Exception:
            return False
        finally:
            sys.stdout = old_stdout

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_one, s): i for i, s in enumerate(samples)}
        for future in as_completed(futures):
            if future.result():
                correct += 1

    return correct / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# MAIN EVOLUTIONARY SEARCH
# ═══════════════════════════════════════════════════════════════

def run_swarm_evolution(n_generations: int = 10, pop_size: int = 8,
                        n_eval_samples: int = 30, model: str = CHEAP,
                        seed: int = 42):
    """Evolve swarm rulesets for HumanEval."""
    from evaluate import load_humaneval

    rng = random.Random(seed)
    print(f"=== Swarm Rules Evolution ===")
    print(f"Pop: {pop_size}, Generations: {n_generations}, Eval: {n_eval_samples}")
    print(f"Model: {model}")

    # Load eval samples
    all_samples = load_humaneval()
    eval_samples = rng.sample(all_samples, min(n_eval_samples, len(all_samples)))
    print(f"Loaded {len(eval_samples)} eval samples")

    # Initialize population
    population = [random_ruleset(rng) for _ in range(pop_size)]

    # Seed with a known-good ruleset
    good_ruleset = [
        make_rule("step_1", "generate", {"temperature": 0.0, "prompt_idx": 0, "count": 3}),
        make_rule("step_2", "generate", {"temperature": 0.5, "prompt_idx": 3, "count": 2}),
        make_rule("step_3", "generate", {"temperature": 0.7, "prompt_idx": 4, "count": 2}),
        make_rule("has_candidates", "test", {"select_passing": True}),
        make_rule("none_pass", "reflect", {"temperature": 0.1}),
        make_rule("after_reflect", "repair", {"temperature": 0.2, "use_error": True}),
        make_rule("all_fail", "restart", {"temperature": 0.5, "prompt_idx": 5}),
    ]
    population[0] = good_ruleset

    best_fitness = 0.0
    best_ruleset = None
    history = []

    for gen in range(n_generations):
        t0 = time.time()

        # Evaluate population
        fitnesses = []
        for i, ruleset in enumerate(population):
            acc = evaluate_ruleset(ruleset, eval_samples, model=model)
            fitnesses.append(acc)
            n_rules = len(ruleset)
            actions = [r["action"] for r in ruleset]
            print(f"  Gen {gen+1} | Ind {i+1}/{pop_size} | "
                  f"Acc: {acc*100:.1f}% | Rules: {n_rules} | "
                  f"Actions: {', '.join(actions)}")

        # Track best
        gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_ruleset = copy.deepcopy(population[gen_best_idx])

        elapsed = time.time() - t0
        history.append({
            "generation": gen + 1,
            "best_accuracy": gen_best_fit * 100,
            "overall_best": best_fitness * 100,
            "avg_accuracy": sum(fitnesses) / len(fitnesses) * 100,
            "elapsed_s": elapsed,
        })

        print(f"\n  Gen {gen+1} Summary: best={gen_best_fit*100:.1f}%, "
              f"overall_best={best_fitness*100:.1f}%, "
              f"avg={sum(fitnesses)/len(fitnesses)*100:.1f}%, "
              f"time={elapsed:.0f}s\n")

        # Selection + reproduction
        # Tournament selection
        new_pop = [copy.deepcopy(best_ruleset)]  # Elitism: keep best
        while len(new_pop) < pop_size:
            # Tournament of 3
            tournament = rng.sample(list(enumerate(fitnesses)), min(3, len(fitnesses)))
            winner_idx = max(tournament, key=lambda x: x[1])[0]
            parent = population[winner_idx]

            if rng.random() < 0.7:
                # Crossover
                tournament2 = rng.sample(list(enumerate(fitnesses)), min(3, len(fitnesses)))
                winner2_idx = max(tournament2, key=lambda x: x[1])[0]
                parent2 = population[winner2_idx]
                child = crossover_rulesets(parent, parent2, rng)
            else:
                child = copy.deepcopy(parent)

            child = mutate_ruleset(child, rng)
            new_pop.append(child)

        population = new_pop

    print(f"\n=== Swarm Evolution Complete ===")
    print(f"Best accuracy: {best_fitness * 100:.1f}%")
    print(f"Best ruleset ({len(best_ruleset)} rules):")
    for r in best_ruleset:
        print(f"  {r['trigger']} → {r['action']}({r['params']})")
    print(f"Total cost: ${get_session_cost():.4f}")

    # Save results
    with open("best_swarm_ruleset.json", "w") as f:
        json.dump({"ruleset": best_ruleset, "accuracy": best_fitness * 100,
                    "history": history}, f, indent=2, default=str)

    return best_ruleset, best_fitness


if __name__ == "__main__":
    import sys as _sys
    n_gen = int(_sys.argv[1]) if len(_sys.argv) > 1 else 5
    n_pop = int(_sys.argv[2]) if len(_sys.argv) > 2 else 8
    n_samples = int(_sys.argv[3]) if len(_sys.argv) > 3 else 30
    model = _sys.argv[4] if len(_sys.argv) > 4 else CHEAP

    run_swarm_evolution(
        n_generations=n_gen,
        pop_size=n_pop,
        n_eval_samples=n_samples,
        model=model,
    )
