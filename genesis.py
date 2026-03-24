#!/usr/bin/env python3
"""Genesis — Genotype-Expression Network for Evolving Scalable Inference Strategies.

A genuinely novel ADAS approach: instead of searching for fixed agent architectures,
Genesis evolves "genomes" — conditional programs that construct different computation
graphs depending on the problem being solved.

KEY NOVELTY:
1. Phenotypic plasticity — one genome, many behaviors depending on the problem
2. Adaptive depth — easy problems exit early, hard problems trigger more compute
3. Confidence routing — intermediate results determine next steps
4. The search evolves STRATEGIES (programs that build programs), not architectures

This is to agent architecture search what evo-devo is to traditional evolution:
we search over developmental programs, not organisms directly.
"""

import json
import re
import random
import time
import copy
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking
from evaluate import extract_number, extract_math_answer, normalize_math_answer


# ═══════════════════════════════════════════════════════════════
# PRIMITIVES — the atomic operations a genome can invoke
# ═══════════════════════════════════════════════════════════════

def prim_generate(problem: str, model: str, temperature: float = 0.0,
                  system: str = "", max_tokens: int = 2048) -> dict:
    """Generate a response. Returns {text, confidence}."""
    if not system:
        system = "You are a precise problem solver. Think step by step."
    result = call_llm(prompt=problem, system=system, model=model,
                      temperature=temperature, max_tokens=max_tokens)
    text = result["content"]
    # Estimate confidence: does it contain a clear answer marker?
    has_answer = "####" in text or "\\boxed" in text or "```" in text
    confidence = 0.8 if has_answer else 0.4
    return {"text": text, "confidence": confidence}


def prim_generate_code(problem: str, model: str, temperature: float = 0.0) -> dict:
    """Generate Python code to solve the problem."""
    result = call_llm(
        prompt=f"Write Python code to solve this. Print ONLY the final answer.\n\n{problem}",
        system="Expert Python programmer. Write clean code.",
        model=model, temperature=temperature, max_tokens=1024)
    text = result["content"]
    # Extract and execute code
    code_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    code = code_match.group(1) if code_match else text
    try:
        import io, contextlib
        stdout = io.StringIO()
        safe_globals = {"__builtins__": {
            'print': print, 'range': range, 'len': len, 'int': int, 'float': float,
            'str': str, 'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
            'sorted': sorted, 'enumerate': enumerate, 'zip': zip, 'map': map,
            'True': True, 'False': False, 'None': None, 'pow': pow, 'divmod': divmod,
        }}
        import math
        safe_globals['math'] = math
        try:
            import sympy
            safe_globals['sympy'] = sympy
            for n in ['sqrt','Rational','simplify','solve','symbols','Symbol','factor',
                       'expand','pi','gcd','lcm','binomial','factorial']:
                if hasattr(sympy, n):
                    safe_globals[n] = getattr(sympy, n)
        except ImportError:
            pass
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals)
        output = stdout.getvalue().strip()
        if output:
            return {"text": f"#### {output.split(chr(10))[-1]}", "confidence": 0.95,
                    "code_output": output}
    except Exception as e:
        pass
    return {"text": text, "confidence": 0.3}


def prim_verify(problem: str, answer_text: str, model: str) -> dict:
    """Verify an answer. Returns {correct_prob, feedback}."""
    result = call_llm(
        prompt=(f"Problem: {problem}\n\nProposed answer:\n{answer_text}\n\n"
                "Is this answer correct? Check the reasoning and calculations. "
                "Reply with CORRECT or INCORRECT and explain why."),
        system="You are a careful verifier.",
        model=model, temperature=0.0, max_tokens=512)
    text = result["content"].upper()
    correct_prob = 0.8 if "CORRECT" in text and "INCORRECT" not in text else 0.2
    return {"correct_prob": correct_prob, "feedback": result["content"]}


def prim_repair(problem: str, answer_text: str, feedback: str, model: str,
                temperature: float = 0.1) -> dict:
    """Repair an answer based on feedback."""
    result = call_llm(
        prompt=(f"Problem: {problem}\n\nPrevious answer:\n{answer_text}\n\n"
                f"Feedback: {feedback}\n\n"
                "Fix the errors. Put your corrected answer after #### or in \\boxed{}."),
        system="Fix the identified errors precisely.",
        model=model, temperature=temperature, max_tokens=2048)
    return {"text": result["content"], "confidence": 0.7}


def prim_vote(candidates: list[str]) -> dict:
    """Majority vote across candidate answers.

    Task-aware: detects whether candidates contain code or math answers.
    For math: extracts numbers and picks majority.
    For code: picks the candidate that looks most like code.
    For mixed: prefers code candidates when detected.
    """
    if not candidates:
        return {"text": "", "confidence": 0.3}

    # Classify each candidate
    code_candidates = []
    math_candidates = []
    for c in candidates:
        # Check if candidate looks like code (has Python syntax markers)
        code_markers = ["def ", "return ", "print(", "import ", "for ", "while ", "if ",
                        "class ", "    ", "```"]
        code_score = sum(1 for m in code_markers if m in c)
        has_number = extract_number(c) is not None

        if code_score >= 2:
            code_candidates.append(c)
        elif has_number or extract_math_answer(c):
            math_candidates.append(c)
        else:
            # Ambiguous — add to both
            code_candidates.append(c)
            math_candidates.append(c)

    # If we have code candidates, use best code candidate
    # (prefer candidates with actual code structure)
    if code_candidates and not math_candidates:
        best = max(code_candidates, key=len)
        return {"text": best, "confidence": 0.6}

    # If purely math, do majority vote on extracted numbers
    if math_candidates and not code_candidates:
        answers = []
        for c in math_candidates:
            num = extract_number(c)
            if num is not None:
                answers.append(str(num))
            else:
                ans = extract_math_answer(c)
                if ans:
                    answers.append(normalize_math_answer(ans))
        if answers:
            counter = Counter(answers)
            best, count = counter.most_common(1)[0]
            confidence = count / len(answers)
            return {"text": f"#### {best}", "confidence": confidence}

    # Mixed case: we have both code and math candidates
    # Try math majority vote first (since it's most reliable for math tasks)
    answers = []
    for c in candidates:
        num = extract_number(c)
        if num is not None:
            answers.append((str(num), c))
    if len(answers) >= 2:
        nums = [a[0] for a in answers]
        counter = Counter(nums)
        best_num, count = counter.most_common(1)[0]
        if count >= 2:
            # Strong math consensus
            return {"text": f"#### {best_num}", "confidence": count / len(candidates)}

    # No consensus — return candidate that has the most structured content
    # Prefer code if it produced executable output (has #### from code execution)
    for c in candidates:
        if "#### " in c and ("code" in c.lower() or "output" in c.lower() or "print" in c.lower()):
            return {"text": c, "confidence": 0.7}

    # Default: return the last candidate (most recent, likely most refined)
    return {"text": candidates[-1], "confidence": 0.4}


# ═══════════════════════════════════════════════════════════════
# GENOME — the evolved program that produces adaptive behavior
# ═══════════════════════════════════════════════════════════════

@dataclass
class Stage:
    """One stage of the adaptive pipeline."""
    action: str  # "generate", "generate_code", "verify", "repair", "vote"
    temperature: float = 0.0
    system_prompt: str = ""
    # Activation condition
    condition: str = "always"  # "always", "low_confidence", "disagreement", "after_failure"
    condition_threshold: float = 0.7
    # Termination check
    terminate_if_confident: bool = False
    confidence_threshold: float = 0.9


@dataclass
class Genome:
    """A genome that encodes an adaptive inference strategy.

    The genome is a sequence of conditional stages. At inference time,
    stages are executed in order, but stages with conditions may be skipped.
    This produces different computation graphs for different problems.
    """
    name: str = "unnamed"
    stages: list = field(default_factory=list)
    model: str = CHEAP
    max_candidates: int = 5  # Max candidates to keep in pool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "max_candidates": self.max_candidates,
            "stages": [
                {
                    "action": s.action,
                    "temperature": s.temperature,
                    "system_prompt": s.system_prompt,
                    "condition": s.condition,
                    "condition_threshold": s.condition_threshold,
                    "terminate_if_confident": s.terminate_if_confident,
                    "confidence_threshold": s.confidence_threshold,
                }
                for s in self.stages
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        stages = [Stage(**s) for s in d.get("stages", [])]
        return cls(
            name=d.get("name", "unnamed"),
            model=d.get("model", CHEAP),
            max_candidates=d.get("max_candidates", 5),
            stages=stages,
        )


# ═══════════════════════════════════════════════════════════════
# INTERPRETER — executes a genome on a problem
# ═══════════════════════════════════════════════════════════════

def execute_genome(genome: Genome, problem: str) -> str:
    """Execute a genome on a problem, producing an adaptive computation graph.

    Batches independent generate/generate_code stages and fires them concurrently.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    candidates = []
    best_confidence = 0.0
    best_answer = ""
    verification_feedback = ""
    had_failure = False

    # Group stages into batches: consecutive generate/generate_code with "always"
    # condition can run in parallel. Everything else runs sequentially.
    i = 0
    while i < len(genome.stages):
        stage = genome.stages[i]

        # Check activation condition
        if not _check_condition(stage, best_confidence, candidates, had_failure):
            i += 1
            continue

        # Check if we can batch consecutive independent generate stages
        if stage.action in ("generate", "generate_code") and stage.condition == "always":
            # Collect all consecutive independent generate stages
            batch = [stage]
            j = i + 1
            while j < len(genome.stages):
                next_s = genome.stages[j]
                if next_s.action in ("generate", "generate_code") and next_s.condition == "always":
                    batch.append(next_s)
                    j += 1
                else:
                    break

            if len(batch) > 1:
                # Fire all generates in parallel
                def _run_gen(s):
                    if s.action == "generate":
                        return prim_generate(problem, genome.model, s.temperature, s.system_prompt)
                    else:
                        return prim_generate_code(problem, genome.model, s.temperature)

                with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                    results = list(ex.map(_run_gen, batch))

                for result in results:
                    candidates.append(result["text"])
                    if result["confidence"] > best_confidence:
                        best_confidence = result["confidence"]
                        best_answer = result["text"]

                i = j  # Skip past the batch
                # Check termination after batch
                if any(s.terminate_if_confident for s in batch) and best_confidence >= 0.9:
                    break
                continue

        # Single stage execution (sequential)
        if stage.action == "generate":
            result = prim_generate(problem, genome.model, stage.temperature, stage.system_prompt)
            candidates.append(result["text"])
            if result["confidence"] > best_confidence:
                best_confidence = result["confidence"]
                best_answer = result["text"]

        elif stage.action == "generate_code":
            result = prim_generate_code(problem, genome.model, stage.temperature)
            candidates.append(result["text"])
            if result["confidence"] > best_confidence:
                best_confidence = result["confidence"]
                best_answer = result["text"]

        elif stage.action == "verify":
            if best_answer:
                result = prim_verify(problem, best_answer, genome.model)
                if result["correct_prob"] < 0.5:
                    had_failure = True
                    verification_feedback = result["feedback"]
                    best_confidence = min(best_confidence, 0.5)
                else:
                    best_confidence = max(best_confidence, result["correct_prob"])

        elif stage.action == "repair":
            if best_answer and (had_failure or verification_feedback):
                result = prim_repair(problem, best_answer, verification_feedback,
                                     genome.model, stage.temperature)
                candidates.append(result["text"])
                if result["confidence"] > best_confidence:
                    best_confidence = result["confidence"]
                    best_answer = result["text"]
                had_failure = False

        elif stage.action == "vote":
            if len(candidates) >= 2:
                result = prim_vote(candidates)
                best_answer = result["text"]
                best_confidence = result["confidence"]

        if len(candidates) > genome.max_candidates:
            candidates = candidates[-genome.max_candidates:]

        if stage.terminate_if_confident and best_confidence >= stage.confidence_threshold:
            break

        i += 1

    return best_answer


def _check_condition(stage: Stage, confidence: float, candidates: list, had_failure: bool) -> bool:
    """Check whether a stage should activate."""
    if stage.condition == "always":
        return True
    elif stage.condition == "low_confidence":
        return confidence < stage.condition_threshold
    elif stage.condition == "disagreement":
        if len(candidates) < 2:
            return False
        # Check if last two candidates agree
        nums = []
        for c in candidates[-2:]:
            n = extract_number(c)
            if n is not None:
                nums.append(n)
        if len(nums) == 2:
            return abs(nums[0] - nums[1]) > 1e-6
        return True  # Can't compare, assume disagreement
    elif stage.condition == "after_failure":
        return had_failure
    return True


# ═══════════════════════════════════════════════════════════════
# EVOLUTIONARY SEARCH — evolves genomes
# ═══════════════════════════════════════════════════════════════

# The actions and conditions available for mutation
ACTIONS = ["generate", "generate_code", "verify", "repair", "vote"]
CONDITIONS = ["always", "low_confidence", "disagreement", "after_failure"]
SYSTEM_PROMPTS = [
    "",
    "You are a precise mathematician. Think step by step. Answer after ####.",
    "You are an expert problem solver. Be rigorous and check your work.",
    "You are a careful reasoner. Break the problem into parts.",
    "You are a creative thinker. Try an unusual approach.",
    "You are a coding expert. Solve with clear logic.",
    "Solve step by step. Double-check arithmetic. Answer after ####.",
    "Think about edge cases. Be thorough. Answer after ####.",
]


def random_stage() -> Stage:
    """Generate a random stage."""
    action = random.choice(ACTIONS)
    return Stage(
        action=action,
        temperature=random.choice([0.0, 0.1, 0.3, 0.5, 0.7]),
        system_prompt=random.choice(SYSTEM_PROMPTS),
        condition=random.choice(CONDITIONS),
        condition_threshold=random.choice([0.5, 0.6, 0.7, 0.8, 0.9]),
        terminate_if_confident=random.random() < 0.3,
        confidence_threshold=random.choice([0.8, 0.85, 0.9, 0.95]),
    )


def random_genome(name: str = "random", n_stages: int = None) -> Genome:
    """Generate a random genome."""
    if n_stages is None:
        n_stages = random.randint(2, 6)
    stages = [random_stage() for _ in range(n_stages)]
    # Ensure at least one generate stage
    if not any(s.action == "generate" for s in stages):
        stages[0] = Stage(action="generate", condition="always")
    return Genome(name=name, stages=stages, model=CHEAP)


def mutate_genome(genome: Genome) -> Genome:
    """Mutate a genome — change one aspect."""
    new = Genome(
        name=genome.name + "_mut",
        model=genome.model,
        max_candidates=genome.max_candidates,
        stages=[copy.deepcopy(s) for s in genome.stages],
    )

    mutation_type = random.choice([
        "add_stage", "remove_stage", "modify_stage",
        "swap_stages", "modify_condition", "modify_prompt",
    ])

    if mutation_type == "add_stage" and len(new.stages) < 8:
        pos = random.randint(0, len(new.stages))
        new.stages.insert(pos, random_stage())

    elif mutation_type == "remove_stage" and len(new.stages) > 2:
        idx = random.randint(0, len(new.stages) - 1)
        # Don't remove the only generate stage
        if new.stages[idx].action == "generate" and \
           sum(1 for s in new.stages if s.action == "generate") <= 1:
            pass
        else:
            new.stages.pop(idx)

    elif mutation_type == "modify_stage":
        idx = random.randint(0, len(new.stages) - 1)
        field = random.choice(["action", "temperature", "condition"])
        if field == "action":
            new.stages[idx].action = random.choice(ACTIONS)
        elif field == "temperature":
            new.stages[idx].temperature = random.choice([0.0, 0.1, 0.3, 0.5, 0.7])
        elif field == "condition":
            new.stages[idx].condition = random.choice(CONDITIONS)

    elif mutation_type == "swap_stages" and len(new.stages) >= 2:
        i, j = random.sample(range(len(new.stages)), 2)
        new.stages[i], new.stages[j] = new.stages[j], new.stages[i]

    elif mutation_type == "modify_condition":
        idx = random.randint(0, len(new.stages) - 1)
        new.stages[idx].condition_threshold = random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
        new.stages[idx].terminate_if_confident = random.random() < 0.3

    elif mutation_type == "modify_prompt":
        idx = random.randint(0, len(new.stages) - 1)
        new.stages[idx].system_prompt = random.choice(SYSTEM_PROMPTS)

    return new


def crossover_genomes(parent1: Genome, parent2: Genome) -> Genome:
    """Crossover two genomes — take stages from both parents."""
    # Single-point crossover
    cut1 = random.randint(1, len(parent1.stages))
    cut2 = random.randint(1, len(parent2.stages))

    child_stages = (
        [copy.deepcopy(s) for s in parent1.stages[:cut1]] +
        [copy.deepcopy(s) for s in parent2.stages[cut2:]]
    )

    # Ensure at least one generate and not too long
    if not any(s.action == "generate" for s in child_stages):
        child_stages.insert(0, Stage(action="generate", condition="always"))
    if len(child_stages) > 8:
        child_stages = child_stages[:8]

    return Genome(
        name=f"cross_{parent1.name}_{parent2.name}",
        model=parent1.model,
        max_candidates=random.choice([parent1.max_candidates, parent2.max_candidates]),
        stages=child_stages,
    )


def llm_evolve_genome(
    genome: Genome,
    score: float,
    error_examples: list[dict],
    model: str = MID,
) -> Genome:
    """Use LLM to intelligently evolve a genome based on error analysis.

    This is the META-EVOLUTION: the LLM acts as a "developmental biologist"
    that understands the genome→phenotype mapping and proposes targeted changes.
    """
    genome_json = json.dumps(genome.to_dict(), indent=2)

    error_str = ""
    for ex in error_examples[:3]:
        error_str += f"- Problem: {ex.get('problem', '')[:150]}\n"
        error_str += f"  Expected: {ex.get('gold', '')}, Got: {ex.get('predicted', '')}\n"

    prompt = f"""You are evolving an agent genome that scored {score:.1f}%.

## Current Genome
{genome_json}

## How Genomes Work
Each genome has stages executed in order. Each stage has:
- action: "generate" (LLM response), "generate_code" (Python), "verify", "repair", "vote"
- condition: "always", "low_confidence", "disagreement", "after_failure"
- temperature: 0.0-0.7
- system_prompt: instructions for the LLM
- terminate_if_confident: stop early if confident

The genome expresses different behavior for different problems (phenotypic plasticity).

## Errors Made
{error_str if error_str else "No specific errors available."}

## Task
Modify the genome to fix these errors. Think about:
- Are we generating enough diverse candidates?
- Should we add verification for certain conditions?
- Is the pipeline too long (wasting compute on easy problems)?
- Should we add code-based solving for math problems?
- Are the system prompts effective?

Return ONLY a valid JSON genome (same format as above).
"""

    result = call_llm(prompt=prompt, system="Expert agent architect. Return valid JSON.",
                      model=model, temperature=0.7, max_tokens=4096, json_mode=True)

    try:
        data = json.loads(result["content"])
        data["model"] = CHEAP
        return Genome.from_dict(data)
    except Exception:
        return mutate_genome(genome)


# ═══════════════════════════════════════════════════════════════
# FAST EVALUATOR — quick eval for the search loop
# ═══════════════════════════════════════════════════════════════

def _sanitize_code(response: str) -> str:
    """Extract function body from LLM response, matching AFlow/MaAS methodology.

    Tries AST-based extraction first (finds longest valid Python), then falls
    back to regex-based extraction.
    """
    import ast

    # Strip code fences and math format
    body = re.sub(r'```python\s*', '', response)
    body = re.sub(r'```\s*', '', body)
    body = re.sub(r'^####\s*.*$', '', body, flags=re.MULTILINE).strip()

    # Try to find the function body by removing the def line and docstring
    lines = body.split('\n')
    if lines and lines[0].strip().startswith('def '):
        i_s = 1
        # Skip docstring
        if i_s < len(lines) and ('"""' in lines[i_s] or "'''" in lines[i_s]):
            i_s += 1
            while i_s < len(lines) and '"""' not in lines[i_s] and "'''" not in lines[i_s]:
                i_s += 1
            i_s += 1
        body = '\n'.join(lines[i_s:])

    # Try AST-based validation: find longest substring that parses as valid Python
    # (matching AFlow's sanitize approach)
    if body.strip():
        try:
            # Quick check if the whole body is valid
            ast.parse(body)
            return body
        except SyntaxError:
            pass

        # Try progressively shorter substrings from the start
        lines = body.split('\n')
        for end in range(len(lines), 0, -1):
            candidate = '\n'.join(lines[:end])
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError:
                continue

    return body


def _eval_single_genesis(genome_dict: dict, sample: dict, idx: int, benchmark: str) -> dict:
    """Evaluate a single sample with a genome. Thread-safe."""
    genome = Genome.from_dict(genome_dict)
    try:
        if benchmark == "gsm8k":
            problem = sample.get("question", "")
            response = execute_genome(genome, problem)
            predicted = extract_number(response)
            gold = sample["gold_answer"]
            is_correct = predicted is not None and abs(predicted - gold) < 1e-6
            if not is_correct:
                return {"idx": idx, "correct": False,
                        "problem": problem[:200], "gold": str(gold), "predicted": str(predicted)}
            return {"idx": idx, "correct": True}
        elif benchmark == "math":
            problem = sample.get("problem", "")
            response = execute_genome(genome, problem)
            pred_ans = extract_math_answer(response)
            gold_ans = extract_math_answer(sample.get("solution", ""))
            if pred_ans and gold_ans:
                is_correct = normalize_math_answer(pred_ans) == normalize_math_answer(gold_ans)
            else:
                pred_num = extract_number(response)
                gold_num_match = extract_math_answer(sample.get("solution", ""))
                is_correct = False
                if pred_num is not None and gold_num_match:
                    try:
                        is_correct = abs(pred_num - float(normalize_math_answer(gold_num_match))) < 1e-6
                    except (ValueError, TypeError):
                        pass
            if not is_correct:
                return {"idx": idx, "correct": False,
                        "problem": problem[:200], "gold": str(gold_ans), "predicted": str(pred_ans)}
            return {"idx": idx, "correct": True}
        elif benchmark == "humaneval":
            prompt = sample["prompt"]
            response = execute_genome(genome, f"Complete this Python function body:\n\n{prompt}")
            body = _sanitize_code(response)
            full = sample["prompt"] + body + "\n" + sample["test"] + f"\ncheck({sample['entry_point']})"
            try:
                # Execute with timeout (15s, matching AFlow/MaAS)
                import threading
                result_box = [False]
                def _run():
                    try:
                        exec(full, {"__builtins__": __builtins__})
                        result_box[0] = True
                    except Exception:
                        pass
                t = threading.Thread(target=_run)
                t.start()
                t.join(timeout=15)
                if result_box[0]:
                    return {"idx": idx, "correct": True}
                return {"idx": idx, "correct": False, "problem": prompt[:100]}
            except Exception:
                return {"idx": idx, "correct": False, "problem": prompt[:100]}
    except Exception as e:
        return {"idx": idx, "correct": False, "problem": str(e)[:100]}
    return {"idx": idx, "correct": False}


def fast_eval(genome: Genome, samples: list[dict], benchmark: str,
              early_stop_threshold: float = 0.3, early_stop_after: float = 0.4) -> dict:
    """Fast PARALLEL evaluation with early stopping for hopeless candidates.

    If after evaluating early_stop_after fraction of samples the score is below
    early_stop_threshold, abort and return the partial score.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    total = len(samples)
    genome_dict = genome.to_dict()
    details = []
    correct_count = 0
    completed_count = 0
    _lock = threading.Lock()
    _abort = threading.Event()

    def _eval_and_track(sample, idx):
        if _abort.is_set():
            return {"idx": idx, "correct": False, "aborted": True}
        result = _eval_single_genesis(genome_dict, sample, idx, benchmark)
        nonlocal correct_count, completed_count
        with _lock:
            completed_count += 1
            if result.get("correct", False):
                correct_count += 1
            # Early stopping check
            check_point = int(total * early_stop_after)
            if completed_count == check_point and check_point > 0:
                current_score = correct_count / completed_count
                if current_score < early_stop_threshold:
                    _abort.set()
        return result

    with ThreadPoolExecutor(max_workers=min(16, total)) as executor:
        futures = {
            executor.submit(_eval_and_track, s, i): i
            for i, s in enumerate(samples)
        }
        for future in as_completed(futures):
            details.append(future.result())

    correct = sum(1 for d in details if d.get("correct", False))
    evaluated = sum(1 for d in details if not d.get("aborted", False))
    errors = [d for d in details if not d.get("correct", False) and "problem" in d]

    # If aborted, extrapolate score from evaluated portion
    if _abort.is_set() and evaluated > 0:
        score = round(correct / evaluated * 100, 2)
    else:
        score = round(correct / total * 100, 2) if total > 0 else 0.0

    return {"score": score, "correct": correct, "total": total,
            "evaluated": evaluated, "errors": errors[:5],
            "aborted": _abort.is_set()}


# ═══════════════════════════════════════════════════════════════
# SEED GENOMES — starting population with diverse strategies
# ═══════════════════════════════════════════════════════════════

SEED_GENOMES = [
    # Simple CoT
    Genome(name="simple_cot", model=CHEAP, stages=[
        Stage(action="generate", condition="always", temperature=0.0,
              system_prompt="Think step by step. Answer after ####.",
              terminate_if_confident=True, confidence_threshold=0.9),
    ]),
    # CoT + Verify + Repair
    Genome(name="cot_verify_repair", model=CHEAP, stages=[
        Stage(action="generate", condition="always", temperature=0.0,
              system_prompt="Solve step by step. Answer after ####."),
        Stage(action="verify", condition="always"),
        Stage(action="repair", condition="after_failure", temperature=0.1,
              terminate_if_confident=True, confidence_threshold=0.85),
    ]),
    # Multi-candidate vote
    Genome(name="multi_vote", model=CHEAP, stages=[
        Stage(action="generate", condition="always", temperature=0.0,
              system_prompt="Think step by step. Answer after ####."),
        Stage(action="generate", condition="always", temperature=0.3,
              system_prompt="Solve carefully. Answer after ####."),
        Stage(action="generate", condition="always", temperature=0.6,
              system_prompt="Be creative in your approach. Answer after ####."),
        Stage(action="vote", condition="always"),
    ]),
    # Code-first with fallback
    Genome(name="code_first", model=CHEAP, stages=[
        Stage(action="generate_code", condition="always", temperature=0.0,
              terminate_if_confident=True, confidence_threshold=0.9),
        Stage(action="generate", condition="low_confidence", condition_threshold=0.7,
              temperature=0.0, system_prompt="Solve step by step. Answer after ####."),
        Stage(action="vote", condition="always"),
    ]),
    # Adaptive depth
    Genome(name="adaptive_depth", model=CHEAP, stages=[
        Stage(action="generate", condition="always", temperature=0.0,
              system_prompt="Solve precisely. Answer after ####.",
              terminate_if_confident=True, confidence_threshold=0.95),
        Stage(action="generate", condition="low_confidence", condition_threshold=0.8,
              temperature=0.3, system_prompt="Try a different approach. Answer after ####."),
        Stage(action="generate_code", condition="disagreement", temperature=0.0),
        Stage(action="vote", condition="always"),
        Stage(action="verify", condition="low_confidence", condition_threshold=0.7),
        Stage(action="repair", condition="after_failure"),
    ]),
]


# ═══════════════════════════════════════════════════════════════
# MAIN EVOLUTION LOOP
# ═══════════════════════════════════════════════════════════════

def run_genesis(
    benchmark: str = "gsm8k",
    n_samples: int = 20,
    population_size: int = 8,
    generations: int = 15,
    elite_size: int = 2,
    seed: int = 42,
):
    """Run the Genesis evolutionary search."""
    from evaluate import load_gsm8k, load_humaneval

    # Load small eval set for fast iteration
    if benchmark == "gsm8k":
        all_samples = load_gsm8k(n=n_samples, seed=seed)
    elif benchmark == "humaneval":
        all_samples = load_humaneval(n=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    print(f"═══ Genesis Evolution ═══")
    print(f"Benchmark: {benchmark}, Samples: {len(all_samples)}")
    print(f"Population: {population_size}, Generations: {generations}")
    print()

    # Initialize population
    population = []
    init_genomes = SEED_GENOMES[:population_size]
    while len(init_genomes) < population_size:
        init_genomes.append(random_genome(f"random_{len(init_genomes)}"))

    # Evaluate initial population IN PARALLEL
    print("── Initial Population ──")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval_genome(genome, samples, bench):
        reset_cost_tracking()
        result = fast_eval(genome, samples, bench)
        return genome, result, get_session_cost()

    with ThreadPoolExecutor(max_workers=len(init_genomes)) as ex:
        futures = [ex.submit(_eval_genome, g, all_samples, benchmark) for g in init_genomes]
        for future in as_completed(futures):
            genome, result, cost = future.result()
            population.append({
                "genome": genome, "score": result["score"], "cost": cost,
                "errors": result.get("errors", []), "n_stages": len(genome.stages),
            })
            stages_str = "→".join(s.action for s in genome.stages)
            print(f"  {genome.name:25s} | {result['score']:5.1f}% | ${cost:.4f} | "
                  f"stages={len(genome.stages)} | {stages_str}")

    best_ever = max(population, key=lambda x: x["score"])
    print(f"\nBest: {best_ever['genome'].name} = {best_ever['score']}%")

    # Evolution loop
    for gen in range(generations):
        print(f"\n── Generation {gen+1}/{generations} ──")
        population.sort(key=lambda x: x["score"], reverse=True)
        new_pop = population[:elite_size]

        # STEP 1: Generate ALL children first (including LLM-evolved ones)
        n_children = population_size - elite_size
        children = []
        methods = []

        def _generate_child(idx, pop, gen_num):
            """Generate a single child genome. Can run in parallel."""
            roll = random.random()
            if roll < 0.35:
                parent = random.choice(pop[:4])
                child = llm_evolve_genome(parent["genome"], parent["score"],
                                          parent.get("errors", []))
                child.name = f"llm_g{gen_num}_{idx}"
                return child, "llm_evolve"
            elif roll < 0.55:
                parent = random.choice(pop[:4])
                child = mutate_genome(parent["genome"])
                child.name = f"mut_g{gen_num}_{idx}"
                return child, "mutation"
            elif roll < 0.75:
                p1, p2 = random.sample(pop[:5], 2)
                child = crossover_genomes(p1["genome"], p2["genome"])
                child.name = f"cross_g{gen_num}_{idx}"
                return child, "crossover"
            else:
                child = random_genome(f"rand_g{gen_num}_{idx}")
                return child, "random"

        # Generate children in parallel (LLM calls for llm_evolve run concurrently)
        with ThreadPoolExecutor(max_workers=n_children) as ex:
            gen_futures = [ex.submit(_generate_child, i, population, gen+1)
                          for i in range(elite_size, population_size)]
            for future in as_completed(gen_futures):
                child, method = future.result()
                child.model = CHEAP
                children.append(child)
                methods.append(method)

        # STEP 2: Evaluate ALL children in parallel
        with ThreadPoolExecutor(max_workers=len(children)) as ex:
            eval_futures = {
                ex.submit(_eval_genome, child, all_samples, benchmark): (child, method)
                for child, method in zip(children, methods)
            }
            for future in as_completed(eval_futures):
                child, method = eval_futures[future]
                try:
                    genome, result, cost = future.result()
                    entry = {
                        "genome": genome, "score": result["score"], "cost": cost,
                        "errors": result.get("errors", []), "n_stages": len(genome.stages),
                    }
                    new_pop.append(entry)

                    marker = ""
                    if result["score"] > best_ever["score"]:
                        best_ever = entry
                        marker = " *** NEW BEST ***"

                    stages_str = "→".join(s.action for s in genome.stages)
                    print(f"  [{method:10s}] {genome.name:25s} | {result['score']:5.1f}% | "
                          f"${cost:.4f} | stages={len(genome.stages)}{marker}")
                except Exception as e:
                    print(f"  [{method:10s}] CRASHED: {str(e)[:80]}")
                    new_pop.append({
                        "genome": child, "score": 0.0, "cost": 0,
                        "errors": [], "n_stages": len(child.stages),
                    })

        population = new_pop

        scores = [p["score"] for p in population]
        print(f"  Gen {gen+1}: best={max(scores):.1f}%, avg={sum(scores)/len(scores):.1f}%, "
              f"best_ever={best_ever['score']:.1f}%")

    # Final
    print(f"\n{'═'*60}")
    print(f"GENESIS EVOLUTION COMPLETE")
    print(f"{'═'*60}")
    print(f"Best genome: {best_ever['genome'].name}")
    print(f"Score: {best_ever['score']}%")
    print(f"Stages: {len(best_ever['genome'].stages)}")
    print(f"\nGenome structure:")
    for i, s in enumerate(best_ever["genome"].stages):
        cond = f" [if {s.condition}" + (f"<{s.condition_threshold}" if s.condition != "always" else "") + "]"
        term = " → STOP if confident" if s.terminate_if_confident else ""
        print(f"  {i+1}. {s.action}(t={s.temperature}){cond}{term}")
        if s.system_prompt:
            print(f"     prompt: {s.system_prompt[:60]}...")

    # Save best genome
    with open("best_genesis_genome.json", "w") as f:
        json.dump(best_ever["genome"].to_dict(), f, indent=2)

    return best_ever


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="gsm8k")
    parser.add_argument("--n", type=int, default=20, help="Samples for fast eval")
    parser.add_argument("--pop", type=int, default=8)
    parser.add_argument("--gens", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_genesis(
        benchmark=args.benchmark,
        n_samples=args.n,
        population_size=args.pop,
        generations=args.gens,
        seed=args.seed,
    )
