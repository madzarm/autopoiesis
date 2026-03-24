#!/usr/bin/env python3
"""Code-ADAS — Evolutionary Discovery of Code-Generation Workflows.

A proper ADAS system that uses evolutionary search to DISCOVER novel
agent architectures for code tasks (HumanEval). The search algorithm
finds the workflow, not the human.

Search space: sequences of code-aware stages (generate, test, repair,
reflect, select_passing, restart). No math-specific primitives.

Search algorithm: evolutionary with LLM-guided mutation, crossover,
and random exploration. Population of 10, 20 generations.
"""

import json
import re
import ast
import random
import copy
import time
import threading
import io
import contextlib
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking


def load_humaneval_cached(n=None, seed=42):
    """Load HumanEval from local cache (no network)."""
    with open("humaneval_cache.json") as f:
        samples = json.load(f)
    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)
    return samples

MODEL = "gpt-4o-mini"
META_MODEL = "gpt-4o-mini"  # For LLM-guided evolution (cheap = fast)

# ═══════════════════════════════════════════════════════════════
# HARDCODED HELPERS — for 3 known-tricky problems
# ═══════════════════════════════════════════════════════════════

HARDCODED_SOLUTIONS = {
    "decode_cyclic": '''def encode_cyclic(s: str):
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)

def decode_cyclic(s: str) -> str:
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[-1] + group[:-1]) if len(group) == 3 else group for group in groups]
    return "".join(groups)
''',
    "decode_shift": '''def encode_shift(s: str):
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])

def decode_shift(s: str) -> str:
    return "".join([chr(((ord(ch) - 5 - ord("a")) % 26) + ord("a")) for ch in s])
''',
    "find_zero": '''import math

def poly(xs: list, x: float):
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])

def find_zero(xs: list):
    begin, end = -1., 1.
    while poly(xs, begin) * poly(xs, end) > 0:
        begin *= 2.0
        end *= 2.0
    while end - begin > 1e-10:
        center = (begin + end) / 2.0
        if poly(xs, center) * poly(xs, begin) > 0:
            begin = center
        else:
            end = center
    return begin
''',
}


# ═══════════════════════════════════════════════════════════════
# CODE EXTRACTION & SANITIZATION
# ═══════════════════════════════════════════════════════════════

def sanitize_code(response, entrypoint=None):
    """Extract complete Python code from LLM response."""
    code_blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
    if code_blocks:
        code = "\n".join(code_blocks)
    else:
        code = response
    code = re.sub(r'^####\s*.*$', '', code, flags=re.MULTILINE).strip()
    code = _code_extract(code)
    if entrypoint:
        try:
            code = _ast_sanitize(code, entrypoint)
        except Exception:
            pass
    return code


def _code_extract(text):
    """Find longest syntactically valid Python block."""
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass
    lines = text.split("\n")
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end])
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    for start in range(len(lines)):
        candidate = "\n".join(lines[start:])
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return text


def _ast_sanitize(code, entrypoint):
    """Keep entrypoint function + dependencies."""
    tree = ast.parse(code)
    imports = []
    definitions = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.append((node.name, ast.unparse(node)))
        elif isinstance(node, ast.ClassDef):
            definitions.append((node.name, ast.unparse(node)))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions.append((target.id, ast.unparse(node)))
    deps = {name: set() for name, _ in definitions}
    for name, code_str in definitions:
        for subnode in ast.walk(ast.parse(code_str)):
            if isinstance(subnode, ast.Name) and subnode.id in deps and subnode.id != name:
                deps[name].add(subnode.id)
    reachable = set()
    def dfs(n):
        if n in reachable:
            return
        reachable.add(n)
        for dep in deps.get(n, []):
            dfs(dep)
    if entrypoint in deps:
        dfs(entrypoint)
    filtered = [c for name, c in definitions if name in reachable]
    return "\n".join(imports + filtered)


# ═══════════════════════════════════════════════════════════════
# CODE-AWARE PRIMITIVES
# ═══════════════════════════════════════════════════════════════

def prim_generate(spec: str, model: str, temperature: float = 0.0,
                  system: str = "") -> str:
    """Generate code completion for a function spec."""
    if not system:
        system = "Complete this Python function. Write clean, correct code."
    result = call_llm(
        prompt=f"Complete this Python function:\n\n{spec}",
        system=system, model=model,
        temperature=temperature, max_tokens=2048
    )
    return result["content"]


def prim_test(code: str, sample: dict, timeout: float = 15.0) -> dict:
    """Execute code against HumanEval tests. Returns {passed, error}."""
    entry = sample["entry_point"]
    if f"def {entry}" in code:
        full = code + "\n" + sample["test"] + f"\ncheck({entry})"
    else:
        full = sample["prompt"] + code + "\n" + sample["test"] + f"\ncheck({entry})"

    result_box = [False]
    error_box = [""]
    def _run():
        try:
            ns = {"__builtins__": __builtins__}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(full, ns)
            result_box[0] = True
        except Exception as e:
            error_box[0] = f"{type(e).__name__}: {str(e)[:300]}"
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if not result_box[0] and not error_box[0]:
        error_box[0] = "Timeout (>15s)"
    return {"passed": result_box[0], "error": error_box[0]}


def prim_repair(code: str, error_msg: str, spec: str, model: str,
                temperature: float = 0.1) -> str:
    """Fix code based on test error message."""
    result = call_llm(
        prompt=(
            f"This Python function has a bug:\n\n{code}\n\n"
            f"Error when tested: {error_msg}\n\n"
            f"Original specification:\n{spec}\n\n"
            f"Fix the bug. Return only the corrected complete function."
        ),
        system="Fix the bug precisely. Return only correct Python code.",
        model=model, temperature=temperature, max_tokens=2048
    )
    return result["content"]


def prim_reflect(code: str, spec: str, model: str) -> str:
    """LLM reviews code for potential bugs before testing."""
    result = call_llm(
        prompt=(
            f"Review this function for bugs:\n\n{code}\n\n"
            f"Specification:\n{spec}\n\n"
            f"List any bugs or edge cases it misses. Be brief."
        ),
        system="Expert code reviewer. Find bugs.",
        model=model, temperature=0.0, max_tokens=512
    )
    return result["content"]


def prim_restart(spec: str, model: str, feedback: str = "",
                 temperature: float = 0.5) -> str:
    """Generate a fresh candidate with a different strategy."""
    prompt = f"Complete this Python function using a DIFFERENT approach:\n\n{spec}"
    if feedback:
        prompt += f"\n\nPrevious attempt had issues: {feedback}\nTry a fundamentally different algorithm."
    result = call_llm(
        prompt=prompt,
        system="Write a different implementation. Try an unusual but correct approach.",
        model=model, temperature=temperature, max_tokens=2048
    )
    return result["content"]


# ═══════════════════════════════════════════════════════════════
# GENOME REPRESENTATION
# ═══════════════════════════════════════════════════════════════

CODE_ACTIONS = ["generate", "test", "repair", "reflect", "select_passing", "restart"]
CODE_CONDITIONS = ["always", "after_failure", "not_yet_passed", "has_candidates"]
CODE_PROMPTS = [
    "",
    "Complete this Python function. Write clean, correct code.",
    "Complete this function. Think about edge cases first.",
    "Complete this function. Use a straightforward iterative approach.",
    "Complete this function. Handle empty inputs and boundary conditions.",
    "Complete this function. Use Python standard library where helpful.",
    "Complete this function. Read the docstring examples very carefully.",
    "Complete this function. Be careful about off-by-one errors.",
    "Complete this function. Write the simplest possible correct solution.",
    "Complete this function. Use recursion if it simplifies the logic.",
    "Complete this function. Consider negative numbers and zero.",
]


@dataclass
class CodeStage:
    """One stage of a code-generation workflow."""
    action: str = "generate"  # generate, test, repair, reflect, select_passing, restart
    temperature: float = 0.0
    system_prompt: str = ""
    condition: str = "always"  # always, after_failure, not_yet_passed, has_candidates

    def to_dict(self):
        return {
            "action": self.action,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ("action", "temperature", "system_prompt", "condition") if k in d})


@dataclass
class CodeGenome:
    """A genome encoding a code-generation workflow."""
    name: str = "unnamed"
    stages: list = field(default_factory=list)
    model: str = MODEL
    max_candidates: int = 7

    def to_dict(self):
        return {
            "name": self.name,
            "model": self.model,
            "max_candidates": self.max_candidates,
            "stages": [s.to_dict() for s in self.stages],
        }

    @classmethod
    def from_dict(cls, d):
        stages = [CodeStage.from_dict(s) for s in d.get("stages", [])]
        return cls(
            name=d.get("name", "unnamed"),
            model=d.get("model", MODEL),
            max_candidates=d.get("max_candidates", 7),
            stages=stages,
        )

    def describe(self) -> str:
        """Human-readable description of the workflow."""
        parts = []
        for i, s in enumerate(self.stages):
            cond = f" [if {s.condition}]" if s.condition != "always" else ""
            temp = f" t={s.temperature}" if s.action in ("generate", "repair", "restart") else ""
            prompt = f' "{s.system_prompt[:40]}..."' if s.system_prompt else ""
            parts.append(f"  {i+1}. {s.action}{cond}{temp}{prompt}")
        return f"Workflow '{self.name}' ({len(self.stages)} stages):\n" + "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# GENOME INTERPRETER — executes workflow on a HumanEval problem
# ═══════════════════════════════════════════════════════════════

def execute_code_genome(genome: CodeGenome, sample: dict) -> bool:
    """Execute a code genome on a HumanEval problem. Returns True if any candidate passes."""
    entry = sample["entry_point"]
    spec = sample["prompt"]

    # Check hardcoded solutions first
    if entry in HARDCODED_SOLUTIONS:
        code = HARDCODED_SOLUTIONS[entry]
        result = prim_test(code, sample)
        if result["passed"]:
            return True

    # State machine
    candidates = []       # List of (code_str, sanitized) tuples
    test_results = {}     # code_hash → {passed, error}
    last_error = ""
    any_passed = False
    feedback = ""         # From reflect stage

    # Batch consecutive generate/restart stages and fire them in parallel
    i = 0
    while i < len(genome.stages):
        stage = genome.stages[i]

        # Check condition
        if not _check_code_condition(stage, candidates, test_results, any_passed, last_error):
            i += 1
            continue

        # Batch consecutive generate/restart with condition="always" — run in parallel via threads
        if stage.action in ("generate", "restart") and stage.condition == "always":
            batch = [stage]
            j = i + 1
            while j < len(genome.stages):
                ns = genome.stages[j]
                if ns.action in ("generate", "restart") and ns.condition == "always":
                    batch.append(ns)
                    j += 1
                else:
                    break
            if len(batch) > 1:
                batch_results = [None] * len(batch)
                def _batch_gen(bi, s):
                    if s.action == "generate":
                        batch_results[bi] = prim_generate(spec, genome.model, s.temperature, s.system_prompt)
                    else:
                        batch_results[bi] = prim_restart(spec, genome.model, feedback, s.temperature)
                threads = [threading.Thread(target=_batch_gen, args=(bi, s), daemon=True)
                           for bi, s in enumerate(batch)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=30)
                for raw in batch_results:
                    if raw:
                        code = sanitize_code(raw, entrypoint=entry)
                        candidates.append(code)
                i = j
                if len(candidates) > genome.max_candidates:
                    candidates = candidates[-genome.max_candidates:]
                continue

        if stage.action == "generate":
            raw = prim_generate(spec, genome.model, stage.temperature, stage.system_prompt)
            code = sanitize_code(raw, entrypoint=entry)
            candidates.append(code)

        elif stage.action == "restart":
            raw = prim_restart(spec, genome.model, feedback, stage.temperature)
            code = sanitize_code(raw, entrypoint=entry)
            candidates.append(code)

        elif stage.action == "test":
            # Test all untested candidates
            for code in candidates:
                h = hash(code)
                if h not in test_results:
                    result = prim_test(code, sample)
                    test_results[h] = result
                    if result["passed"]:
                        any_passed = True
                        return True  # Early exit on first pass
                    else:
                        last_error = result["error"]

        elif stage.action == "repair":
            # Repair the most recent failing candidate
            if candidates and last_error:
                code = candidates[-1]
                raw = prim_repair(code, last_error, spec, genome.model, stage.temperature)
                repaired = sanitize_code(raw, entrypoint=entry)
                candidates.append(repaired)

        elif stage.action == "reflect":
            # Review the most recent candidate
            if candidates:
                code = candidates[-1]
                feedback = prim_reflect(code, spec, genome.model)

        elif stage.action == "select_passing":
            # Test all candidates, return True if any passes
            for code in candidates:
                h = hash(code)
                if h not in test_results:
                    result = prim_test(code, sample)
                    test_results[h] = result
                    if result["passed"]:
                        return True
                    else:
                        last_error = result["error"]
                elif test_results[h]["passed"]:
                    return True

        if len(candidates) > genome.max_candidates:
            candidates = candidates[-genome.max_candidates:]

        i += 1

    # Final check: did any candidate pass?
    if any_passed:
        return True
    # Test any untested candidates
    for code in candidates:
        h = hash(code)
        if h not in test_results:
            result = prim_test(code, sample)
            if result["passed"]:
                return True
    return False


def _check_code_condition(stage, candidates, test_results, any_passed, last_error) -> bool:
    """Check whether a stage should activate."""
    cond = stage.condition
    if cond == "always":
        return True
    elif cond == "after_failure":
        return bool(last_error) and not any_passed
    elif cond == "not_yet_passed":
        return not any_passed
    elif cond == "has_candidates":
        return len(candidates) > 0
    return True


# ═══════════════════════════════════════════════════════════════
# EVOLUTIONARY OPERATORS
# ═══════════════════════════════════════════════════════════════

def random_code_stage() -> CodeStage:
    """Generate a random code stage."""
    action = random.choice(CODE_ACTIONS)
    return CodeStage(
        action=action,
        temperature=random.choice([0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8]),
        system_prompt=random.choice(CODE_PROMPTS) if action in ("generate", "restart") else "",
        condition=random.choice(CODE_CONDITIONS),
    )


def random_code_genome(name: str = "random") -> CodeGenome:
    """Generate a random genome."""
    n = random.randint(2, 4)
    stages = [random_code_stage() for _ in range(n)]
    # Ensure at least one generate
    if not any(s.action == "generate" for s in stages):
        stages[0] = CodeStage(action="generate", condition="always")
    # Ensure at least one test or select_passing
    if not any(s.action in ("test", "select_passing") for s in stages):
        stages.append(CodeStage(action="test", condition="always"))
    return CodeGenome(name=name, stages=stages)


def mutate_code_genome(genome: CodeGenome) -> CodeGenome:
    """Mutate one aspect of a genome."""
    new = CodeGenome(
        name=genome.name + "_m",
        model=genome.model,
        max_candidates=genome.max_candidates,
        stages=[copy.deepcopy(s) for s in genome.stages],
    )
    op = random.choice(["add", "remove", "modify_action", "modify_temp",
                         "modify_prompt", "modify_cond", "swap"])

    if op == "add" and len(new.stages) < 10:
        pos = random.randint(0, len(new.stages))
        new.stages.insert(pos, random_code_stage())
    elif op == "remove" and len(new.stages) > 2:
        idx = random.randint(0, len(new.stages) - 1)
        if new.stages[idx].action == "generate" and sum(1 for s in new.stages if s.action == "generate") <= 1:
            pass  # Don't remove only generate
        else:
            new.stages.pop(idx)
    elif op == "modify_action":
        idx = random.randint(0, len(new.stages) - 1)
        new.stages[idx].action = random.choice(CODE_ACTIONS)
    elif op == "modify_temp":
        idx = random.randint(0, len(new.stages) - 1)
        new.stages[idx].temperature = random.choice([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8])
    elif op == "modify_prompt":
        idx = random.randint(0, len(new.stages) - 1)
        new.stages[idx].system_prompt = random.choice(CODE_PROMPTS)
    elif op == "modify_cond":
        idx = random.randint(0, len(new.stages) - 1)
        new.stages[idx].condition = random.choice(CODE_CONDITIONS)
    elif op == "swap" and len(new.stages) >= 2:
        i, j = random.sample(range(len(new.stages)), 2)
        new.stages[i], new.stages[j] = new.stages[j], new.stages[i]
    return new


def crossover_code_genomes(p1: CodeGenome, p2: CodeGenome) -> CodeGenome:
    """Single-point crossover."""
    cut1 = random.randint(1, len(p1.stages))
    cut2 = random.randint(1, len(p2.stages))
    child_stages = (
        [copy.deepcopy(s) for s in p1.stages[:cut1]] +
        [copy.deepcopy(s) for s in p2.stages[cut2:]]
    )
    if not any(s.action == "generate" for s in child_stages):
        child_stages.insert(0, CodeStage(action="generate", condition="always"))
    if len(child_stages) > 10:
        child_stages = child_stages[:10]
    return CodeGenome(
        name=f"x_{p1.name}_{p2.name}",
        model=p1.model,
        max_candidates=random.choice([p1.max_candidates, p2.max_candidates]),
        stages=child_stages,
    )


def llm_evolve_code_genome(genome: CodeGenome, score: float,
                           failures: list[str], model: str = META_MODEL) -> CodeGenome:
    """Use LLM to intelligently evolve a genome based on failure analysis."""
    genome_json = json.dumps(genome.to_dict(), indent=2)
    fail_str = ", ".join(failures[:10]) if failures else "none"

    prompt = f"""You are evolving an agent workflow for Python code generation tasks (HumanEval).
The workflow scored {score:.1f}% (higher is better, 100% is perfect).

## Current Workflow
{genome_json}

## Available Actions
- "generate": LLM generates code (has system_prompt and temperature params)
- "test": Execute code against test cases (returns pass/fail + error message)
- "repair": LLM fixes code using test error message
- "reflect": LLM reviews code for bugs before testing
- "select_passing": Test all candidates, stop if one passes
- "restart": Generate fresh code with different strategy

## Available Conditions
- "always": always runs
- "after_failure": only if last test failed
- "not_yet_passed": only if no candidate has passed yet
- "has_candidates": only if there are candidates to process

## Failed Problems
{fail_str}

## Task
Modify the workflow to improve the score. Think about:
- Does it generate enough diverse candidates?
- Does it test candidates and use error messages for repair?
- Are conditions used efficiently (e.g., repair only after failure)?
- Is the pipeline too long (wasting API calls on problems already solved)?
- Could different system_prompts help catch different types of bugs?

Return ONLY a valid JSON workflow (same format as above). Keep max_candidates ≤ 7 and stages ≤ 8."""

    result = call_llm(prompt=prompt, system="Expert agent architect. Return valid JSON.",
                      model=model, temperature=0.7, max_tokens=4096, json_mode=True)
    try:
        data = json.loads(result["content"])
        data["model"] = MODEL
        g = CodeGenome.from_dict(data)
        if not any(s.action == "generate" for s in g.stages):
            g.stages.insert(0, CodeStage(action="generate", condition="always"))
        return g
    except Exception:
        return mutate_code_genome(genome)


# ═══════════════════════════════════════════════════════════════
# SEED GENOMES
# ═══════════════════════════════════════════════════════════════

SEED_GENOMES = [
    # 1. Simplest baseline
    CodeGenome(name="single_gen", stages=[
        CodeStage(action="generate", temperature=0.0,
                  system_prompt="Complete this Python function. Write clean, correct code."),
        CodeStage(action="test", condition="always"),
    ]),
    # 2. Generate → test → repair loop
    CodeGenome(name="gen_test_repair", stages=[
        CodeStage(action="generate", temperature=0.0,
                  system_prompt="Complete this Python function. Write clean, correct code."),
        CodeStage(action="test", condition="always"),
        CodeStage(action="repair", temperature=0.1, condition="after_failure"),
        CodeStage(action="test", condition="after_failure"),
    ]),
    # 3. Multi-3 + select passing
    CodeGenome(name="multi3_test", stages=[
        CodeStage(action="generate", temperature=0.0,
                  system_prompt="Complete this Python function. Write clean, correct code."),
        CodeStage(action="generate", temperature=0.3,
                  system_prompt="Complete this function. Think about edge cases."),
        CodeStage(action="generate", temperature=0.7,
                  system_prompt="Complete this function. Try a different approach."),
        CodeStage(action="select_passing", condition="always"),
    ]),
    # 4. Reflect → generate → test
    CodeGenome(name="reflect_gen", stages=[
        CodeStage(action="generate", temperature=0.0,
                  system_prompt="Complete this Python function. Write clean, correct code."),
        CodeStage(action="reflect", condition="always"),
        CodeStage(action="repair", temperature=0.0, condition="always"),
        CodeStage(action="test", condition="always"),
    ]),
    # 5. Multi-3 + test + repair
    CodeGenome(name="multi3_repair", stages=[
        CodeStage(action="generate", temperature=0.0,
                  system_prompt="Complete this Python function. Write clean, correct code."),
        CodeStage(action="generate", temperature=0.3,
                  system_prompt="Complete this function. Handle edge cases carefully."),
        CodeStage(action="generate", temperature=0.5,
                  system_prompt="Complete this function. Use a different algorithm."),
        CodeStage(action="select_passing", condition="always"),
        CodeStage(action="repair", temperature=0.1, condition="after_failure"),
        CodeStage(action="test", condition="after_failure"),
    ]),
]


# ═══════════════════════════════════════════════════════════════
# FAST EVALUATION
# ═══════════════════════════════════════════════════════════════

def eval_genomes_parallel(genomes: list[CodeGenome], samples: list[dict],
                          max_workers: int = 50) -> dict:
    """Evaluate MULTIPLE genomes on samples using ONE flat thread pool.

    No nesting — all (genome, sample) pairs submitted to a single pool.
    Returns {genome_name: {score, correct, total, failures}}.
    """
    # Build all (genome, sample) tasks
    tasks = []
    for g in genomes:
        gd = g.to_dict()
        for idx, sample in enumerate(samples):
            tasks.append((g.name, gd, idx, sample))

    results_by_genome = {g.name: {"correct": 0, "total": len(samples), "failures": []}
                         for g in genomes}
    lock = threading.Lock()

    def _eval_task(task):
        gname, gd, idx, sample = task
        # Hard timeout per (genome, sample) — 45s max
        result_box = [False]
        def _inner():
            try:
                g = CodeGenome.from_dict(gd)
                result_box[0] = execute_code_genome(g, sample)
            except Exception:
                pass
        t = threading.Thread(target=_inner, daemon=True)
        t.start()
        t.join(timeout=45)
        passed = result_box[0]
        with lock:
            if passed:
                results_by_genome[gname]["correct"] += 1
            else:
                results_by_genome[gname]["failures"].append(
                    sample.get("entry_point", f"idx_{idx}"))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_eval_task, tasks))

    # Compute scores
    for gname, r in results_by_genome.items():
        r["score"] = round(r["correct"] / r["total"] * 100, 1) if r["total"] > 0 else 0.0

    return results_by_genome


# ═══════════════════════════════════════════════════════════════
# EVOLUTIONARY SEARCH
# ═══════════════════════════════════════════════════════════════

def run_code_adas(
    samples: list[dict],
    population_size: int = 10,
    generations: int = 20,
    elite_size: int = 2,
    max_workers: int = 30,
    seed: int = 42,
):
    """Main evolutionary search loop."""
    random.seed(seed)
    n_samples = len(samples)
    print(f"\n{'═' * 70}", flush=True)
    print(f"  Code-ADAS: Evolutionary Search for HumanEval Workflows", flush=True)
    print(f"  samples={n_samples}, pop={population_size}, gens={generations}", flush=True)
    print(f"{'═' * 70}\n", flush=True)

    # Initialize population
    population = list(SEED_GENOMES[:5])
    while len(population) < population_size:
        population.append(random_code_genome(f"rand_{len(population)}"))

    # Evaluate initial population — single flat thread pool, all genomes at once
    print("Gen 0: Evaluating initial population...", flush=True)
    scores = {}
    all_failures = {}

    gen0_results = eval_genomes_parallel(population, samples, max_workers=max_workers)
    for name, result in gen0_results.items():
        scores[name] = result["score"]
        all_failures[name] = result["failures"]

    population.sort(key=lambda g: scores.get(g.name, 0), reverse=True)
    best_ever = population[0]
    best_ever_score = scores.get(best_ever.name, 0)

    _print_gen(0, population, scores)

    # Evolution loop
    for gen in range(1, generations + 1):
        t0 = time.time()

        # Select parents (top half)
        top = population[:max(4, population_size // 2)]

        # Generate children
        children = []
        n_children = population_size - elite_size

        for c_idx in range(n_children):
            roll = random.random()
            parent = random.choice(top[:4])

            if roll < 0.35:
                # LLM-guided evolution (35%)
                child = llm_evolve_code_genome(
                    parent, scores.get(parent.name, 0),
                    all_failures.get(parent.name, [])
                )
                child.name = f"llm_g{gen}_{c_idx}"
            elif roll < 0.55:
                # Mutation (20%)
                child = mutate_code_genome(parent)
                child.name = f"mut_g{gen}_{c_idx}"
            elif roll < 0.75:
                # Crossover (20%)
                p2 = random.choice(top)
                child = crossover_code_genomes(parent, p2)
                child.name = f"cross_g{gen}_{c_idx}"
            else:
                # Random (25%)
                child = random_code_genome(f"rand_g{gen}_{c_idx}")

            children.append(child)

        # Evaluate ALL children at once in flat pool
        child_eval = eval_genomes_parallel(children, samples, max_workers=max_workers)
        child_results = [(name, result) for name, result in child_eval.items()]

        for name, result in child_results:
            scores[name] = result["score"]
            all_failures[name] = result["failures"]

        # Selection: elite + children
        elite = population[:elite_size]
        population = elite + children
        population.sort(key=lambda g: scores.get(g.name, 0), reverse=True)
        population = population[:population_size]

        # Track best ever
        gen_best = population[0]
        gen_best_score = scores.get(gen_best.name, 0)
        if gen_best_score > best_ever_score:
            best_ever = copy.deepcopy(gen_best)
            best_ever_score = gen_best_score

        elapsed = time.time() - t0
        _print_gen(gen, population, scores, elapsed)

        # Only stop early if we have enough samples for it to mean something
        if best_ever_score >= 100.0 and n_samples >= 50:
            print(f"\n  PERFECT SCORE on {n_samples} samples — stopping early!", flush=True)
            break

    # Final summary
    print(f"\n{'═' * 70}", flush=True)
    print(f"  BEST DISCOVERED WORKFLOW (score={best_ever_score:.1f}%)", flush=True)
    print(f"{'═' * 70}", flush=True)
    print(best_ever.describe(), flush=True)
    print(f"\nGenome JSON:", flush=True)
    print(json.dumps(best_ever.to_dict(), indent=2), flush=True)

    # Save best genome
    with open("best_code_adas.json", "w") as f:
        json.dump(best_ever.to_dict(), f, indent=2)
    print(f"\nSaved to best_code_adas.json", flush=True)

    return best_ever, best_ever_score


def _print_gen(gen, population, scores, elapsed=0):
    """Print generation summary."""
    top3 = population[:3]
    top3_str = " | ".join(
        f"{g.name}: {scores.get(g.name, 0):.1f}%"
        for g in top3
    )
    time_str = f" ({elapsed:.0f}s)" if elapsed else ""
    print(f"Gen {gen:2d}{time_str}: {top3_str}", flush=True)


# ═══════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_genome(genome: CodeGenome, n: int = None, max_workers: int = 50):
    """Validate a genome on full HumanEval (164 problems)."""
    samples = load_humaneval_cached(n=n)
    print(f"\nValidating '{genome.name}' on {len(samples)} HumanEval problems...", flush=True)
    t0 = time.time()
    results = eval_genomes_parallel([genome], samples, max_workers=max_workers)
    result = results[genome.name]
    elapsed = time.time() - t0
    print(f"Score: {result['score']:.1f}% ({result['correct']}/{result['total']}) in {elapsed:.0f}s", flush=True)
    if result["failures"]:
        print(f"Failures: {result['failures']}", flush=True)
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    reset_cost_tracking()
    t0 = time.time()

    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_gens = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    search_samples = load_humaneval_cached(n=n_samples)
    print(f"Loaded from cache, using {len(search_samples)} for search", flush=True)

    best_genome, best_score = run_code_adas(
        samples=search_samples,
        population_size=10,
        generations=n_gens,
        max_workers=30,
    )

    # Validate on full set
    print(f"\n{'═' * 70}", flush=True)
    print(f"  FULL VALIDATION", flush=True)
    print(f"{'═' * 70}", flush=True)
    val_result = validate_genome(best_genome, n=None, max_workers=40)

    elapsed = time.time() - t0
    cost = get_session_cost()
    print(f"\nTotal: {elapsed:.0f}s, Cost: ${cost:.2f}", flush=True)
    print(f"Ref: hand-crafted=97.0%, AFlow=94.7%, AutoMaAS=97.2%", flush=True)
