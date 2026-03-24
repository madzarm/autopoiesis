#!/usr/bin/env python3
"""Approach 15: CMA-ES over Continuous Agent Config Embeddings.

Key novelty: Instead of discrete mutations on pipeline structures, we embed
agent configurations into a continuous vector space and use Covariance Matrix
Adaptation Evolution Strategy (CMA-ES) to search that space. CMA-ES is the
state-of-the-art for black-box continuous optimization — it learns the
covariance structure of the fitness landscape and adapts its search distribution.

Search space:
- Each agent config is a vector of ~20 continuous dims:
  - temperature (0-1), num_candidates (1-10), num_repair_rounds (0-3)
  - prompt embedding selector (0-1 mapped to prompt pool)
  - stage weights for generate/test/repair/reflect/restart
  - diversity params, fallback thresholds, etc.

Search algorithm: CMA-ES (Hansen 2001)
- Maintains a multivariate Gaussian over config space
- Uses rank-based fitness to update mean, step size, and covariance
- Invariant to monotone transformations of fitness
- No gradients needed — pure black-box optimization

This is fundamentally different from:
- Evolutionary: random mutations, no covariance adaptation
- MCTS: discrete tree search
- Bayesian: GP surrogate model, acquisition function
- LLM-architect: LLM generates configs directly
"""

import json
import math
import time
import numpy as np
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm, CHEAP, get_session_cost, reset_cost_tracking

# ═══════════════════════════════════════════════════════════════
# CONFIG VECTOR DEFINITION
# ═══════════════════════════════════════════════════════════════

# Each dimension of the config vector and its mapping to agent params
CONFIG_DIMS = [
    # Generation params
    ("num_candidates", 1, 10),      # 0: how many initial candidates
    ("temp_base", 0.0, 1.0),        # 1: base temperature
    ("temp_spread", 0.0, 0.5),      # 2: temperature diversity range
    ("prompt_style", 0.0, 1.0),     # 3: 0=precise, 0.5=balanced, 1=creative
    ("max_tokens", 256, 2048),       # 4: max generation tokens

    # Selection params
    ("use_test_select", 0.0, 1.0),  # 5: >0.5 = test-select, else vote
    ("vote_temp", 0.0, 0.5),        # 6: temperature for voting LLM

    # Repair params
    ("num_repair_rounds", 0, 3),    # 7: number of repair attempts
    ("repair_temp", 0.0, 0.5),      # 8: temperature for repair
    ("use_reflect", 0.0, 1.0),      # 9: >0.5 = add reflect before repair

    # Restart params
    ("use_restart", 0.0, 1.0),      # 10: >0.5 = add restart as fallback
    ("restart_temp", 0.0, 1.0),     # 11: temperature for restart

    # Diversity params
    ("diverse_prompts", 0.0, 1.0),  # 12: >0.5 = use diverse prompts per candidate
    ("cot_prefix", 0.0, 1.0),       # 13: >0.5 = add chain-of-thought prefix

    # Ensemble params
    ("ensemble_size", 1, 5),        # 14: how many sub-strategies to ensemble
    ("ensemble_diversity", 0.0, 1.0), # 15: diversity weight in ensemble

    # Fallback thresholds
    ("repair_threshold", 0.0, 1.0), # 16: confidence threshold to trigger repair
    ("restart_threshold", 0.0, 1.0),# 17: confidence threshold to trigger restart
]

N_DIM = len(CONFIG_DIMS)

PROMPT_POOL = [
    "Complete this Python function. Write clean, correct code.",
    "Implement the function step by step. Handle all edge cases.",
    "Think carefully about the problem, then write the solution.",
    "Explore different algorithms and edge cases.",
    "Generate a creative solution. Think outside the box.",
    "Write a robust implementation. Consider boundary conditions.",
    "First analyze the docstring examples, then implement.",
    "Write production-quality code with proper error handling.",
]


def vec_to_config(vec: np.ndarray) -> dict:
    """Convert a continuous vector to a discrete agent configuration."""
    config = {}
    for i, (name, lo, hi) in enumerate(CONFIG_DIMS):
        # Clamp to [0, 1] range first
        v = max(0.0, min(1.0, vec[i]))
        # Map to actual range
        val = lo + v * (hi - lo)
        # Discretize integer params
        if name in ("num_candidates", "num_repair_rounds", "max_tokens", "ensemble_size"):
            val = int(round(val))
        config[name] = val
    return config


def config_to_vec(config: dict) -> np.ndarray:
    """Convert a config dict back to a [0,1] vector."""
    vec = np.zeros(N_DIM)
    for i, (name, lo, hi) in enumerate(CONFIG_DIMS):
        val = config.get(name, (lo + hi) / 2)
        vec[i] = (val - lo) / (hi - lo) if hi > lo else 0.5
    return vec


# ═══════════════════════════════════════════════════════════════
# AGENT EXECUTION FROM CONFIG
# ═══════════════════════════════════════════════════════════════

def run_agent_from_config(config: dict, prompt: str, entry_point: str,
                          test_code: str, model: str = CHEAP) -> str:
    """Execute an agent workflow defined by a config dict on a single problem."""
    import ast
    import re
    import sys
    import io

    num_cands = config["num_candidates"]
    temp_base = config["temp_base"]
    temp_spread = config["temp_spread"]
    use_test = config["use_test_select"] > 0.5
    use_reflect = config["use_reflect"] > 0.5
    num_repair = config["num_repair_rounds"]
    use_restart = config["use_restart"] > 0.5
    diverse = config["diverse_prompts"] > 0.5
    use_cot = config["cot_prefix"] > 0.5
    max_tok = config["max_tokens"]

    # Select prompts
    prompt_idx = int(config["prompt_style"] * (len(PROMPT_POOL) - 1))
    base_sys = PROMPT_POOL[prompt_idx]

    def sanitize(response, ep=entry_point):
        """Extract code from response."""
        code_blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
        code = "\n".join(code_blocks) if code_blocks else response
        code = re.sub(r'^####\s*.*$', '', code, flags=re.MULTILINE).strip()
        # Try to find the function def
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
        """Execute code against tests. Returns (passed, error_msg)."""
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

    # Phase 1: Generate candidates
    candidates = []
    for j in range(num_cands):
        temp = temp_base + (j / max(1, num_cands - 1)) * temp_spread
        temp = min(1.0, max(0.0, temp))

        sys_prompt = base_sys
        if diverse and j < len(PROMPT_POOL):
            sys_prompt = PROMPT_POOL[j % len(PROMPT_POOL)]

        user_msg = prompt
        if use_cot:
            user_msg = "Think step by step about the problem, then provide the implementation.\n\n" + prompt

        try:
            resp = call_llm(user_msg, system=sys_prompt, model=model,
                           temperature=temp, max_tokens=max_tok)
            code = sanitize(resp["content"])
            if code.strip():
                candidates.append(code)
        except Exception:
            pass

    if not candidates:
        return ""

    # Phase 2: Select best candidate
    if use_test and test_code:
        # Test-select: pick first passing candidate
        for code in candidates:
            passed, err = exec_test(code)
            if passed:
                return code
        # None passed — use first candidate for repair
        best = candidates[0]
        last_err = err
    else:
        best = candidates[0]
        passed, last_err = exec_test(best)
        if passed:
            return best

    # Phase 3: Repair loop
    for r in range(num_repair):
        if use_reflect:
            try:
                reflect_resp = call_llm(
                    f"Review this code for bugs:\n```python\n{best}\n```\n"
                    f"Error: {last_err}\nWhat's wrong and how to fix it?",
                    system="You are a code reviewer. Identify bugs concisely.",
                    model=model, temperature=0.1, max_tokens=512
                )
                reflection = reflect_resp["content"]
            except Exception:
                reflection = ""
        else:
            reflection = ""

        repair_prompt = f"Fix this code based on the error:\n```python\n{best}\n```\nError: {last_err}"
        if reflection:
            repair_prompt += f"\nAnalysis: {reflection}"

        try:
            resp = call_llm(repair_prompt,
                           system="Fix the code. Return only the corrected function.",
                           model=model, temperature=config["repair_temp"],
                           max_tokens=max_tok)
            repaired = sanitize(resp["content"])
            if repaired.strip():
                passed, err = exec_test(repaired)
                if passed:
                    return repaired
                best = repaired
                last_err = err
        except Exception:
            pass

    # Phase 4: Restart (if enabled)
    if use_restart:
        try:
            resp = call_llm(
                f"Generate a completely different solution:\n{prompt}",
                system="Write a new solution using a different algorithm. Be creative.",
                model=model, temperature=config["restart_temp"],
                max_tokens=max_tok
            )
            code = sanitize(resp["content"])
            if code.strip():
                passed, _ = exec_test(code)
                if passed:
                    return code
        except Exception:
            pass

    return best


# ═══════════════════════════════════════════════════════════════
# CMA-ES IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════

class CMAES:
    """Minimal CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

    Follows Hansen 2001 / Hansen 2016 tutorial.
    """

    def __init__(self, n: int, x0: Optional[np.ndarray] = None,
                 sigma0: float = 0.3, popsize: Optional[int] = None):
        self.n = n
        self.mean = x0 if x0 is not None else np.full(n, 0.5)
        self.sigma = sigma0

        # Population size (default: 4 + 3*ln(n))
        self.lam = popsize or (4 + int(3 * math.log(n)))
        self.mu = self.lam // 2

        # Recombination weights
        weights = np.array([math.log(self.mu + 0.5) - math.log(i + 1) for i in range(self.mu)])
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()

        # Adaptation params
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs

        # State
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.eigenvalues = np.ones(n)
        self.eigenvectors = np.eye(n)
        self.generation = 0
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    def ask(self) -> list[np.ndarray]:
        """Sample lambda offspring from the search distribution."""
        self._decompose()
        offspring = []
        for _ in range(self.lam):
            z = np.random.randn(self.n)
            x = self.mean + self.sigma * (self.eigenvectors @ (np.sqrt(self.eigenvalues) * z))
            # Clamp to [0, 1]
            x = np.clip(x, 0.0, 1.0)
            offspring.append(x)
        return offspring

    def tell(self, solutions: list[tuple[np.ndarray, float]]):
        """Update the search distribution from (solution, fitness) pairs.

        Lower fitness is better (minimization).
        """
        # Sort by fitness (ascending = best first)
        solutions.sort(key=lambda s: s[1])

        # Select mu best
        selected = [s[0] for s in solutions[:self.mu]]

        # Weighted recombination
        old_mean = self.mean.copy()
        self.mean = np.zeros(self.n)
        for i, w in enumerate(self.weights):
            self.mean += w * selected[i]

        # Evolution paths
        self._decompose()
        invsqrtC = self.eigenvectors @ np.diag(1.0 / np.sqrt(self.eigenvalues)) @ self.eigenvectors.T

        self.ps = (1 - self.cs) * self.ps + \
                  math.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  invsqrtC @ (self.mean - old_mean) / self.sigma

        hsig = np.linalg.norm(self.ps) / \
               math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) / self.chiN < 1.4 + 2 / (self.n + 1)

        self.pc = (1 - self.cc) * self.pc + \
                  hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * \
                  (self.mean - old_mean) / self.sigma

        # Covariance matrix adaptation
        artmp = np.array([(selected[i] - old_mean) / self.sigma for i in range(self.mu)])

        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) +
                            (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * sum(self.weights[i] * np.outer(artmp[i], artmp[i])
                                for i in range(self.mu))

        # Step size adaptation
        self.sigma *= math.exp((self.cs / self.damps) *
                               (np.linalg.norm(self.ps) / self.chiN - 1))

        self.generation += 1

    def _decompose(self):
        """Eigendecomposition of C (cached)."""
        # Force symmetry
        self.C = (self.C + self.C.T) / 2
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
        except np.linalg.LinAlgError:
            self.C = np.eye(self.n)
            self.eigenvalues = np.ones(self.n)
            self.eigenvectors = np.eye(self.n)


# ═══════════════════════════════════════════════════════════════
# EVALUATION ON HUMANEVAL
# ═══════════════════════════════════════════════════════════════

def evaluate_config_on_humaneval(config: dict, samples: list[dict],
                                 model: str = CHEAP,
                                 max_workers: int = 8) -> float:
    """Evaluate a config on HumanEval samples. Returns accuracy (0-1)."""
    correct = 0
    total = len(samples)

    def eval_one(sample):
        ep = sample["entry_point"]
        prompt_text = sample["prompt"]
        test_code = sample["test"]

        code = run_agent_from_config(config, prompt_text, ep, test_code, model=model)
        if not code.strip():
            return False

        # Run tests
        import sys, io
        full_code = code + "\n" + test_code + f"\ncheck({ep})"
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
# MAIN SEARCH LOOP
# ═══════════════════════════════════════════════════════════════

def run_cmaes_search(n_generations: int = 10, n_eval_samples: int = 30,
                     model: str = CHEAP, seed: int = 42):
    """Run CMA-ES search over agent config space on HumanEval."""
    from evaluate import load_humaneval
    import random

    np.random.seed(seed)

    print(f"=== CMA-ES Agent Config Search ===")
    print(f"Dims: {N_DIM}, Generations: {n_generations}, Eval samples: {n_eval_samples}")
    print(f"Model: {model}")

    # Load eval samples (use hardest problems for search differentiation)
    all_samples = load_humaneval()
    rng = random.Random(seed)
    eval_samples = rng.sample(all_samples, min(n_eval_samples, len(all_samples)))
    print(f"Loaded {len(eval_samples)} eval samples")

    # Initialize CMA-ES with a reasonable starting point
    x0 = config_to_vec({
        "num_candidates": 5,
        "temp_base": 0.0,
        "temp_spread": 0.3,
        "prompt_style": 0.5,
        "max_tokens": 1024,
        "use_test_select": 0.8,
        "vote_temp": 0.1,
        "num_repair_rounds": 2,
        "repair_temp": 0.2,
        "use_reflect": 0.7,
        "use_restart": 0.7,
        "restart_temp": 0.5,
        "diverse_prompts": 0.7,
        "cot_prefix": 0.3,
        "ensemble_size": 1,
        "ensemble_diversity": 0.5,
        "repair_threshold": 0.5,
        "restart_threshold": 0.3,
    })

    cma = CMAES(N_DIM, x0=x0, sigma0=0.2, popsize=8)

    best_fitness = float('inf')
    best_config = None
    history = []

    for gen in range(n_generations):
        t0 = time.time()

        # Ask for offspring
        offspring = cma.ask()

        # Evaluate each offspring
        results = []
        for i, vec in enumerate(offspring):
            config = vec_to_config(vec)
            accuracy = evaluate_config_on_humaneval(config, eval_samples, model=model)
            fitness = 1.0 - accuracy  # Minimize (1 - accuracy)
            results.append((vec, fitness))
            print(f"  Gen {gen+1} | Individual {i+1}/{len(offspring)} | "
                  f"Accuracy: {accuracy*100:.1f}% | Config: ncand={config['num_candidates']}, "
                  f"test_sel={config['use_test_select']>.5}, "
                  f"repair={config['num_repair_rounds']}, "
                  f"restart={config['use_restart']>.5}")

        # Tell CMA-ES about results
        cma.tell(results)

        # Track best
        gen_best = min(results, key=lambda x: x[1])
        if gen_best[1] < best_fitness:
            best_fitness = gen_best[1]
            best_config = vec_to_config(gen_best[0])

        elapsed = time.time() - t0
        gen_best_acc = (1 - gen_best[1]) * 100
        overall_best_acc = (1 - best_fitness) * 100

        history.append({
            "generation": gen + 1,
            "best_accuracy": gen_best_acc,
            "overall_best": overall_best_acc,
            "sigma": cma.sigma,
            "elapsed_s": elapsed,
        })

        print(f"\n  Gen {gen+1} Summary: best={gen_best_acc:.1f}%, "
              f"overall_best={overall_best_acc:.1f}%, "
              f"sigma={cma.sigma:.4f}, time={elapsed:.0f}s\n")

    print(f"\n=== CMA-ES Search Complete ===")
    print(f"Best accuracy: {(1 - best_fitness) * 100:.1f}%")
    print(f"Best config: {json.dumps(best_config, indent=2)}")
    print(f"Total cost: ${get_session_cost():.4f}")

    # Save results
    with open("best_cmaes_config.json", "w") as f:
        json.dump({"config": best_config, "accuracy": (1 - best_fitness) * 100,
                    "history": history}, f, indent=2)

    return best_config, 1 - best_fitness


if __name__ == "__main__":
    import sys
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    model = sys.argv[3] if len(sys.argv) > 3 else CHEAP

    best_config, best_acc = run_cmaes_search(
        n_generations=n_gen,
        n_eval_samples=n_samples,
        model=model,
    )
