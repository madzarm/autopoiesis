"""V2 Agent System — Code-generating agents with tool use and multi-step workflows.

This goes beyond prompt engineering to discover novel agent ARCHITECTURES:
- Code generation: agent writes Python to solve problems
- Multi-step workflows: chain of agents (plan → solve → verify)
- Tool use: calculator, code interpreter
- Dynamic routing: classify problem type, route to specialist
"""

import json
import re
import traceback
from typing import Optional
from pydantic import BaseModel
from llm import call_llm, CHEAP, MID


class AgentV2Config(BaseModel):
    """V2 agent config with richer architecture options."""
    name: str = "unnamed"

    # Architecture type — the key dimension
    architecture: str = "cot"
    # Options:
    #   "direct" - single prompt, direct answer
    #   "cot" - chain of thought
    #   "code_solve" - generate Python code, extract answer from code output
    #   "plan_solve_verify" - 3-step: plan approach, solve, verify answer
    #   "classify_route" - classify problem type, use different strategy per type
    #   "ensemble_diverse" - run multiple diverse architectures, vote
    #   "progressive_refine" - solve, analyze errors, refine (multiple rounds)

    # For code_solve
    code_max_attempts: int = 2

    # For plan_solve_verify
    verify_strategy: str = "recompute"  # recompute, substitute, alternative_method

    # For ensemble_diverse
    ensemble_architectures: list[str] = ["cot", "code_solve", "plan_solve_verify"]
    ensemble_n: int = 3

    # For progressive_refine
    refine_rounds: int = 2

    # Shared settings
    persona: str = ""
    custom_instructions: str = ""
    model: str = CHEAP
    temperature: float = 0.0
    max_tokens: int = 2048


def run_agent_v2(config: AgentV2Config, question: str, passage: str = "", answer_format: str = "numeric") -> str:
    """Run a V2 agent on a question.

    answer_format: "numeric" for math, "text" for reading comprehension, "mc" for multiple choice
    """
    arch = config.architecture

    if arch == "direct":
        return _direct(config, question, passage, answer_format)
    elif arch == "cot":
        return _cot(config, question, passage, answer_format)
    elif arch == "code_solve":
        return _code_solve(config, question, passage)
    elif arch == "plan_solve_verify":
        return _plan_solve_verify(config, question, passage, answer_format)
    elif arch == "classify_route":
        return _classify_route(config, question, passage, answer_format)
    elif arch == "ensemble_diverse":
        return _ensemble_diverse(config, question, passage, answer_format)
    elif arch == "progressive_refine":
        return _progressive_refine(config, question, passage, answer_format)
    else:
        return _cot(config, question, passage, answer_format)


def _build_base_prompt(config: AgentV2Config, question: str, passage: str = "", answer_format: str = "numeric") -> tuple[str, str]:
    """Build base system + user prompts.

    answer_format: "numeric" for math, "text" for reading comprehension, "mc" for multiple choice
    """
    system = config.persona or "You are a helpful AI assistant that solves problems accurately."
    if config.custom_instructions:
        system += "\n\n" + config.custom_instructions

    if answer_format == "numeric":
        system += "\n\nAlways put your final numeric answer after #### on its own line."
    elif answer_format == "text":
        system += "\n\nProvide a concise, direct answer. State just the answer without explanation in your final line after ####."
    elif answer_format == "mc":
        system += "\n\nSelect the correct answer letter. State your answer letter after #### on its own line."

    user = f"Question: {question}" if not passage else f"Passage: {passage}\n\nQuestion: {question}"
    return system, user


def _direct(config: AgentV2Config, question: str, passage: str = "", answer_format: str = "numeric") -> str:
    system, user = _build_base_prompt(config, question, passage, answer_format)
    result = call_llm(prompt=user, system=system, model=config.model,
                      temperature=config.temperature, max_tokens=config.max_tokens)
    return result["content"]


def _cot(config: AgentV2Config, question: str, passage: str = "", answer_format: str = "numeric") -> str:
    system, user = _build_base_prompt(config, question, passage, answer_format)
    system += "\n\nThink step by step. Number each step of your reasoning."
    result = call_llm(prompt=user, system=system, model=config.model,
                      temperature=config.temperature, max_tokens=config.max_tokens)
    return result["content"]


def _code_solve(config: AgentV2Config, question: str, passage: str = "") -> str:
    """Generate Python code to solve the problem, then execute it."""
    system = (
        "You are a Python programmer who solves math problems by writing code. "
        "Write a Python program that computes the answer. "
        "The program MUST print ONLY the final numeric answer as the last line of output. "
        "Use only basic Python (no imports needed for math). "
        "Wrap your code in ```python ... ``` blocks."
    )

    user = f"Write Python code to solve this problem:\n\n{question}"

    for attempt in range(config.code_max_attempts):
        result = call_llm(prompt=user, system=system, model=config.model,
                          temperature=config.temperature, max_tokens=config.max_tokens)
        content = result["content"]

        # Extract Python code
        code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
        if not code_match:
            # Try to execute the whole response as code
            code = content
        else:
            code = code_match.group(1)

        # Execute safely
        try:
            output = _safe_exec(code)
            if output.strip():
                # Return the output as the answer
                return f"#### {output.strip().split(chr(10))[-1]}"
        except Exception as e:
            if attempt < config.code_max_attempts - 1:
                user = (
                    f"Your code had an error: {str(e)[:200]}\n\n"
                    f"Original problem: {question}\n\n"
                    f"Fix the code. Print ONLY the numeric answer."
                )
            continue

    # Fallback to CoT
    return _cot(config, question, passage)


def _safe_exec(code: str, timeout: int = 5) -> str:
    """Execute Python code safely and return stdout."""
    import io
    import contextlib

    # Restricted builtins
    safe_builtins = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sum': sum, 'len': len, 'range': range, 'int': int,
        'float': float, 'str': str, 'list': list, 'dict': dict,
        'tuple': tuple, 'set': set, 'sorted': sorted, 'enumerate': enumerate,
        'zip': zip, 'map': map, 'filter': filter, 'any': any, 'all': all,
        'print': print, 'True': True, 'False': False, 'None': None,
        'pow': pow, 'divmod': divmod,
    }

    stdout = io.StringIO()
    local_vars = {}

    # Allow math import
    import math
    local_vars['math'] = math

    # Allow itertools and collections
    import itertools
    import collections
    local_vars['itertools'] = itertools
    local_vars['collections'] = collections

    with contextlib.redirect_stdout(stdout):
        exec(code, {"__builtins__": safe_builtins, **local_vars}, local_vars)

    return stdout.getvalue()


def _plan_solve_verify(config: AgentV2Config, question: str, passage: str = "", answer_format: str = "numeric") -> str:
    """Three-step pipeline: Plan → Solve → Verify."""
    context = f"Passage: {passage}\n\n" if passage else ""

    # Step 1: Plan
    plan_result = call_llm(
        prompt=f"{context}Question: {question}\n\nCreate a step-by-step plan to solve this. List the key steps needed.",
        system="You are a strategic problem solver. Create clear, actionable plans.",
        model=config.model, temperature=0.0, max_tokens=512,
    )
    plan = plan_result["content"]

    # Step 2: Solve
    solve_result = call_llm(
        prompt=(
            f"{context}Question: {question}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Now execute this plan step by step. Show all calculations. "
            f"Put your final answer after ####."
        ),
        system="You are a precise executor. Follow the plan exactly and show all work.",
        model=config.model, temperature=0.0, max_tokens=config.max_tokens,
    )
    solution = solve_result["content"]

    # Step 3: Verify
    if config.verify_strategy == "recompute":
        verify_prompt = (
            f"Question: {question}\n\n"
            f"Solution:\n{solution}\n\n"
            f"Verify this solution by re-doing the key calculations from scratch. "
            f"If you find errors, provide the correct answer. Put final answer after ####."
        )
    elif config.verify_strategy == "substitute":
        verify_prompt = (
            f"Question: {question}\n\n"
            f"Solution:\n{solution}\n\n"
            f"Verify by substituting the answer back into the problem to check if it makes sense. "
            f"If incorrect, solve again. Put final answer after ####."
        )
    else:  # alternative_method
        verify_prompt = (
            f"Question: {question}\n\n"
            f"Solution:\n{solution}\n\n"
            f"Verify by solving the problem using a completely different approach. "
            f"If the answers differ, determine which is correct. Put final answer after ####."
        )

    verify_result = call_llm(
        prompt=verify_prompt,
        system="You are a careful verifier. Check every step.",
        model=config.model, temperature=0.0, max_tokens=config.max_tokens,
    )
    return verify_result["content"]


def _classify_route(config: AgentV2Config, question: str, passage: str = "", answer_format: str = "numeric") -> str:
    """Classify problem type and route to the best strategy."""
    # Classify
    classify_result = call_llm(
        prompt=(
            f"Classify this math problem into ONE category:\n\n"
            f"Question: {question}\n\n"
            f"Categories:\n"
            f"- ARITHMETIC: basic calculations, percentages, ratios\n"
            f"- ALGEBRA: equations, unknowns, systems\n"
            f"- WORD_PROBLEM: multi-step real-world scenarios\n"
            f"- GEOMETRY: shapes, distances, areas\n"
            f"- LOGIC: deduction, patterns, sequences\n\n"
            f"Reply with ONLY the category name."
        ),
        system="Classify the problem type. Reply with one word.",
        model=config.model, temperature=0.0, max_tokens=50,
    )

    category = classify_result["content"].strip().upper()

    # Route to specialist
    if "ARITH" in category:
        # Simple arithmetic — code is best
        code_config = AgentV2Config(
            architecture="code_solve", model=config.model,
            temperature=0.0, code_max_attempts=2,
        )
        return _code_solve(code_config, question, passage)
    elif "ALGEBRA" in category:
        # Algebra — code with sympy-style approach
        code_config = AgentV2Config(
            architecture="code_solve", model=config.model,
            temperature=0.0, code_max_attempts=2,
        )
        return _code_solve(code_config, question, passage)
    elif "WORD" in category:
        # Word problems — plan-solve-verify
        psv_config = AgentV2Config(
            architecture="plan_solve_verify", model=config.model,
            temperature=0.0, verify_strategy="recompute",
        )
        return _plan_solve_verify(psv_config, question, passage)
    else:
        # Default — CoT
        return _cot(config, question, passage)


def _ensemble_diverse(config: AgentV2Config, question: str, passage: str = "", answer_format: str = "numeric") -> str:
    """Run multiple diverse architectures and vote on the answer."""
    from evaluate import extract_number
    from collections import Counter

    answers = []
    responses = []

    for arch in config.ensemble_architectures[:config.ensemble_n]:
        sub_config = AgentV2Config(
            architecture=arch, model=config.model,
            temperature=0.2 if arch != "code_solve" else 0.0,
            max_tokens=config.max_tokens,
            persona=config.persona,
            custom_instructions=config.custom_instructions,
        )
        try:
            resp = run_agent_v2(sub_config, question, passage)
            responses.append(resp)
            num = extract_number(resp)
            if num is not None:
                answers.append(str(num))
        except Exception:
            continue

    if not answers:
        return responses[-1] if responses else _cot(config, question, passage)

    # Majority vote
    counter = Counter(answers)
    best_answer = counter.most_common(1)[0][0]
    return f"#### {best_answer}"


def _progressive_refine(config: AgentV2Config, question: str, passage: str = "", answer_format: str = "numeric") -> str:
    """Progressively refine the answer through multiple rounds of self-critique."""
    # Initial solve
    system, user = _build_base_prompt(config, question, passage)
    system += "\n\nThink step by step. Be very careful with arithmetic."
    result = call_llm(prompt=user, system=system, model=config.model,
                      temperature=config.temperature, max_tokens=config.max_tokens)
    current_answer = result["content"]

    for round_num in range(config.refine_rounds):
        # Critique
        critique_result = call_llm(
            prompt=(
                f"Question: {question}\n\n"
                f"Current solution:\n{current_answer}\n\n"
                f"Carefully analyze this solution for errors:\n"
                f"1. Are all arithmetic calculations correct?\n"
                f"2. Is the reasoning logically sound?\n"
                f"3. Does the answer make sense given the problem?\n"
                f"4. Were any conditions or constraints missed?\n\n"
                f"List specific errors found, or say 'NO ERRORS FOUND'."
            ),
            system="You are a meticulous error-checker. Find every mistake.",
            model=config.model, temperature=0.0, max_tokens=1024,
        )
        critique = critique_result["content"]

        if "NO ERRORS FOUND" in critique.upper():
            break

        # Refine based on critique
        refine_result = call_llm(
            prompt=(
                f"Question: {question}\n\n"
                f"Previous solution:\n{current_answer}\n\n"
                f"Errors found:\n{critique}\n\n"
                f"Fix ALL identified errors and provide the corrected solution. "
                f"Put your final answer after ####."
            ),
            system="Fix the errors precisely. Show corrected work.",
            model=config.model, temperature=0.0, max_tokens=config.max_tokens,
        )
        current_answer = refine_result["content"]

    return current_answer


# Predefined V2 configs
V2_CONFIGS = {
    "cot_v2": AgentV2Config(name="cot_v2", architecture="cot"),
    "code_solve": AgentV2Config(name="code_solve", architecture="code_solve"),
    "plan_solve_verify": AgentV2Config(
        name="plan_solve_verify", architecture="plan_solve_verify",
        verify_strategy="recompute",
    ),
    "classify_route": AgentV2Config(name="classify_route", architecture="classify_route"),
    "ensemble_diverse": AgentV2Config(
        name="ensemble_diverse", architecture="ensemble_diverse",
        ensemble_architectures=["cot", "code_solve", "plan_solve_verify"],
    ),
    "progressive_refine": AgentV2Config(
        name="progressive_refine", architecture="progressive_refine",
        refine_rounds=2,
    ),
}
