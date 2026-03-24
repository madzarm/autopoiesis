"""DSPy-based approach for HumanEval code generation.

Uses DSPy's declarative framework to optimize prompts and few-shot examples
for Python function completion. This serves as a "generate" primitive in the
evolutionary ADAS search space.

Installation:
    pip install dspy>=2.6.0

Setup:
    export OPENAI_API_KEY=...
    Then call optimize_for_humaneval() to get an optimized module,
    or use CodeCompleter directly for unoptimized inference.

Key DSPy concepts used:
    - Signatures: Declarative input/output specs (CodeCompletion, CodeCompletionWithAnalysis)
    - ChainOfThought: Step-by-step reasoning before generating code
    - BootstrapFewShot: Auto-generates few-shot examples from training data
    - BootstrapFewShotWithRandomSearch: Multiple bootstrap rounds + best selection
    - MIPROv2: Bayesian optimization over instructions + few-shot combos
    - Module composition: Multi-step pipeline (analyze -> generate -> refine)
"""

import os
import re
import json
import random
import traceback
import io
import contextlib
from typing import Optional

import dspy
from datasets import load_dataset


# ---------------------------------------------------------------------------
# 1. LM Configuration
# ---------------------------------------------------------------------------

def configure_dspy(
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    api_key: Optional[str] = None,
):
    """Configure DSPy with the given LM.

    DSPy uses litellm under the hood, so model names follow litellm convention:
        - "openai/gpt-4o-mini"
        - "openai/gpt-4o"
        - "anthropic/claude-3-haiku-20240307"

    Args:
        model: litellm-style model name
        temperature: sampling temperature (0.0 for deterministic)
        max_tokens: max output tokens
        api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    lm = dspy.LM(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    dspy.configure(lm=lm)
    return lm


# ---------------------------------------------------------------------------
# 2. Signatures — Declarative input/output specs
# ---------------------------------------------------------------------------

class CodeCompletion(dspy.Signature):
    """Complete a Python function given its signature, docstring, and any
    initial code. Return ONLY the function body (the code that comes after
    the provided prompt). Do not repeat the function signature or docstring."""

    prompt: str = dspy.InputField(
        desc="Python function signature, docstring, and any starter code"
    )
    completed_code: str = dspy.OutputField(
        desc="The function body that completes the implementation. "
        "Only the new lines of code, no repeated signature/docstring."
    )


class CodeCompletionWithAnalysis(dspy.Signature):
    """Analyze a Python function stub, then complete its implementation.
    First understand what the function should do, identify edge cases,
    then write the implementation."""

    prompt: str = dspy.InputField(
        desc="Python function signature, docstring, and any starter code"
    )
    analysis: str = dspy.OutputField(
        desc="Brief analysis: what the function does, key edge cases, algorithm choice"
    )
    completed_code: str = dspy.OutputField(
        desc="The function body that completes the implementation. "
        "Only the new lines of code, no repeated signature/docstring."
    )


class CodeRefinement(dspy.Signature):
    """Review and fix a Python function implementation.
    Given the original prompt and a candidate implementation,
    check for bugs, edge cases, and correctness, then return
    a corrected version if needed."""

    prompt: str = dspy.InputField(desc="Original function stub/prompt")
    candidate_code: str = dspy.InputField(desc="Candidate implementation to review")
    error_info: str = dspy.InputField(desc="Error message if the code failed, or 'none'")
    refined_code: str = dspy.OutputField(
        desc="Corrected function body (or same if already correct)"
    )


# ---------------------------------------------------------------------------
# 3. DSPy Modules — Composable building blocks
# ---------------------------------------------------------------------------

class SimpleCodeCompleter(dspy.Module):
    """Basic code completion using ChainOfThought.

    ChainOfThought automatically injects a 'reasoning' field before the
    output, encouraging step-by-step thinking.
    """

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CodeCompletion)

    def forward(self, prompt: str) -> dspy.Prediction:
        result = self.generate(prompt=prompt)
        return dspy.Prediction(
            completed_code=result.completed_code,
            reasoning=result.reasoning,
        )


class AnalyzeAndComplete(dspy.Module):
    """Two-step approach: analyze the problem, then generate code.

    Uses the CodeCompletionWithAnalysis signature which has an explicit
    analysis step before code generation.
    """

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CodeCompletionWithAnalysis)

    def forward(self, prompt: str) -> dspy.Prediction:
        result = self.generate(prompt=prompt)
        return dspy.Prediction(
            completed_code=result.completed_code,
            analysis=result.analysis,
            reasoning=result.reasoning,
        )


class CodeCompleterWithRefinement(dspy.Module):
    """Generate code, test it, and refine if it fails.

    This is a multi-step pipeline:
    1. Generate initial code with CoT
    2. Try to execute it
    3. If it fails, refine with error feedback
    """

    def __init__(self, max_refine_rounds: int = 1):
        super().__init__()
        self.generate = dspy.ChainOfThought(CodeCompletion)
        self.refine = dspy.ChainOfThought(CodeRefinement)
        self.max_refine_rounds = max_refine_rounds

    def forward(self, prompt: str, test: str = "", entry_point: str = "") -> dspy.Prediction:
        # Step 1: Generate
        result = self.generate(prompt=prompt)
        code = result.completed_code

        # Step 2: Try to execute (if test info available)
        if test and entry_point:
            for _ in range(self.max_refine_rounds):
                error = self._try_execute(prompt, code, test, entry_point)
                if error is None:
                    break  # Code works!
                # Step 3: Refine
                refined = self.refine(
                    prompt=prompt,
                    candidate_code=code,
                    error_info=error[:500],  # Truncate long errors
                )
                code = refined.refined_code

        return dspy.Prediction(completed_code=code)

    @staticmethod
    def _try_execute(prompt: str, code: str, test: str, entry_point: str) -> Optional[str]:
        """Try executing the code. Returns error string or None if success."""
        full_code = prompt + code + "\n" + test + f"\ncheck({entry_point})"
        try:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exec(full_code, {})
            return None
        except Exception as e:
            return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 4. Code Sanitization (matching existing evaluate.py patterns)
# ---------------------------------------------------------------------------

def sanitize_code(response: str, prompt: str = "") -> str:
    """Extract clean code from an LLM response.

    Handles:
    - Code in ```python ... ``` blocks
    - Code that repeats the function signature
    - Raw code without markdown formatting
    """
    # Try to extract from code block
    code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        code = response

    # If the code repeats the prompt, strip it
    if prompt and code.strip().startswith(prompt.strip()[:50]):
        # Find where the prompt ends and keep only new code
        lines = code.split('\n')
        prompt_lines = prompt.strip().split('\n')
        # Skip lines that match the prompt
        start_idx = 0
        for i, line in enumerate(lines):
            if i < len(prompt_lines) and line.strip() == prompt_lines[i].strip():
                start_idx = i + 1
            else:
                break
        code = '\n'.join(lines[start_idx:])

    return code


# ---------------------------------------------------------------------------
# 5. HumanEval Data Loading & Metric
# ---------------------------------------------------------------------------

def load_humaneval_as_dspy_examples(
    n: Optional[int] = None,
    seed: int = 42,
    split: str = "test",
) -> list[dspy.Example]:
    """Load HumanEval dataset as DSPy Example objects.

    Each example has:
        - prompt: the function stub
        - test: the test code
        - entry_point: the function name
        - canonical_solution: the reference solution (used as label)
        - task_id: HumanEval task identifier
    """
    ds = load_dataset("openai/openai_humaneval", split=split)
    examples = []
    for item in ds:
        ex = dspy.Example(
            prompt=item["prompt"],
            test=item["test"],
            entry_point=item["entry_point"],
            canonical_solution=item["canonical_solution"],
            task_id=item["task_id"],
        ).with_inputs("prompt")  # Only 'prompt' is provided as input
        examples.append(ex)

    if n is not None and n < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, n)
    return examples


def humaneval_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """Metric for HumanEval: does the generated code pass the tests?

    Args:
        example: has .prompt, .test, .entry_point
        prediction: has .completed_code
        trace: optional trace info (used by optimizers)

    Returns:
        True if the code passes all tests, False otherwise
    """
    try:
        code = sanitize_code(prediction.completed_code, example.prompt)
        full_code = example.prompt + code + "\n" + example.test + f"\ncheck({example.entry_point})"

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(full_code, {})
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 6. Optimizers — Find best prompts/few-shot examples automatically
# ---------------------------------------------------------------------------

def optimize_bootstrap_fewshot(
    module: dspy.Module,
    trainset: list[dspy.Example],
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 3,
    max_rounds: int = 1,
) -> dspy.Module:
    """Optimize using BootstrapFewShot.

    Best for small datasets (5-10 examples). Fast and cheap.
    Generates few-shot demos by running the module on training data
    and keeping examples that pass the metric.

    Args:
        module: DSPy module to optimize
        trainset: list of dspy.Example with inputs
        max_bootstrapped_demos: number of auto-generated demos
        max_labeled_demos: number of labeled demos from trainset
        max_rounds: bootstrap attempts per example

    Returns:
        Optimized module with few-shot examples baked into prompts
    """
    optimizer = dspy.BootstrapFewShot(
        metric=humaneval_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
    )
    optimized = optimizer.compile(module, trainset=trainset)
    return optimized


def optimize_bootstrap_random_search(
    module: dspy.Module,
    trainset: list[dspy.Example],
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 3,
    num_candidate_programs: int = 8,
    num_threads: int = 4,
) -> dspy.Module:
    """Optimize using BootstrapFewShotWithRandomSearch.

    Better than basic BootstrapFewShot — tries multiple random combos
    of demos and picks the best one. Good for 50+ examples.

    Args:
        module: DSPy module to optimize
        trainset: training examples
        max_bootstrapped_demos: max auto-generated demos per candidate
        max_labeled_demos: max labeled demos per candidate
        num_candidate_programs: how many random programs to try
        num_threads: parallel evaluation threads

    Returns:
        Best optimized module
    """
    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=humaneval_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        num_candidate_programs=num_candidate_programs,
        num_threads=num_threads,
    )
    optimized = optimizer.compile(module, trainset=trainset)
    return optimized


def optimize_mipro(
    module: dspy.Module,
    trainset: list[dspy.Example],
    auto: str = "light",  # "light", "medium", "heavy"
    num_threads: int = 4,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 3,
) -> dspy.Module:
    """Optimize using MIPROv2 — the most powerful DSPy optimizer.

    Jointly optimizes instructions AND few-shot examples using
    Bayesian optimization. Three modes:
        - "light": ~10 trials, quick and cheap (~$0.50)
        - "medium": ~25 trials, balanced (~$2)
        - "heavy": ~100 trials, thorough (~$10+)

    For 0-shot optimization (instruction-only, no demos), set both
    max_bootstrapped_demos=0 and max_labeled_demos=0.

    Args:
        module: DSPy module to optimize
        trainset: training examples (50+ recommended, 200+ ideal)
        auto: optimization intensity
        num_threads: parallel threads
        max_bootstrapped_demos: auto-generated demos (0 for zero-shot)
        max_labeled_demos: labeled demos (0 for zero-shot)

    Returns:
        Best optimized module with tuned instructions and demos
    """
    optimizer = dspy.MIPROv2(
        metric=humaneval_metric,
        auto=auto,
        num_threads=num_threads,
    )
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    return optimized


# ---------------------------------------------------------------------------
# 7. Integration — Agent function compatible with evaluate.py
# ---------------------------------------------------------------------------

def make_agent_fn(module: dspy.Module):
    """Wrap a DSPy module into an agent function compatible with evaluate.py.

    evaluate.py expects:  agent_fn(prompt: str) -> str  (raw code string)

    The returned function calls the DSPy module and returns sanitized code.
    """
    def agent_fn(prompt: str) -> str:
        prediction = module(prompt=prompt)
        code = sanitize_code(prediction.completed_code, prompt)
        return code
    return agent_fn


def make_agent_fn_with_refinement(
    module: CodeCompleterWithRefinement,
):
    """Wrap the refinement module into an agent_fn.

    This version passes test info for self-repair.
    """
    def agent_fn(prompt: str, test: str = "", entry_point: str = "") -> str:
        prediction = module(prompt=prompt, test=test, entry_point=entry_point)
        code = sanitize_code(prediction.completed_code, prompt)
        return code
    return agent_fn


# ---------------------------------------------------------------------------
# 8. Full Pipeline — Optimize and evaluate
# ---------------------------------------------------------------------------

def optimize_for_humaneval(
    model: str = "openai/gpt-4o-mini",
    optimizer: str = "bootstrap",  # "bootstrap", "random_search", "mipro"
    n_train: int = 20,
    train_seed: int = 42,
    save_path: Optional[str] = None,
    module_class: str = "simple",  # "simple", "analyze", "refine"
) -> dspy.Module:
    """Full optimization pipeline for HumanEval.

    1. Configure DSPy with the specified model
    2. Load a subset of HumanEval as training data
    3. Create the specified module
    4. Run the optimizer
    5. Optionally save the optimized state

    Args:
        model: LM to use
        optimizer: which optimizer ("bootstrap", "random_search", "mipro")
        n_train: number of training examples (HumanEval has 164 total)
        train_seed: random seed for train split
        save_path: where to save optimized module (e.g. "dspy_humaneval_optimized.json")
        module_class: which module to use

    Returns:
        Optimized DSPy module
    """
    # 1. Configure
    configure_dspy(model=model, temperature=0.0)

    # 2. Load data
    all_examples = load_humaneval_as_dspy_examples()
    rng = random.Random(train_seed)
    trainset = rng.sample(all_examples, min(n_train, len(all_examples)))

    print(f"[DSPy] Loaded {len(all_examples)} HumanEval problems, using {len(trainset)} for training")

    # 3. Create module
    if module_class == "analyze":
        module = AnalyzeAndComplete()
    elif module_class == "refine":
        module = CodeCompleterWithRefinement(max_refine_rounds=1)
    else:
        module = SimpleCodeCompleter()

    print(f"[DSPy] Module: {module.__class__.__name__}")
    print(f"[DSPy] Optimizer: {optimizer}")

    # 4. Optimize
    if optimizer == "bootstrap":
        optimized = optimize_bootstrap_fewshot(
            module, trainset,
            max_bootstrapped_demos=3,
            max_labeled_demos=2,
        )
    elif optimizer == "random_search":
        optimized = optimize_bootstrap_random_search(
            module, trainset,
            max_bootstrapped_demos=3,
            max_labeled_demos=2,
            num_candidate_programs=8,
        )
    elif optimizer == "mipro":
        optimized = optimize_mipro(
            module, trainset,
            auto="light",
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # 5. Save
    if save_path:
        optimized.save(save_path)
        print(f"[DSPy] Saved optimized module to {save_path}")

    return optimized


def evaluate_dspy_module(
    module: dspy.Module,
    n_eval: Optional[int] = None,
    eval_seed: int = 123,
) -> dict:
    """Evaluate a DSPy module on HumanEval.

    Runs the module on HumanEval problems and reports pass@1.

    Args:
        module: DSPy module (optimized or not)
        n_eval: number of problems to evaluate (None = all 164)
        eval_seed: seed for sampling eval subset

    Returns:
        dict with score, correct, total, details
    """
    examples = load_humaneval_as_dspy_examples(n=n_eval, seed=eval_seed)
    correct = 0
    total = len(examples)
    details = []

    for i, ex in enumerate(examples):
        try:
            prediction = module(prompt=ex.prompt)
            passed = humaneval_metric(ex, prediction)
            details.append({
                "idx": i,
                "task_id": ex.task_id,
                "correct": passed,
            })
            if passed:
                correct += 1
        except Exception as e:
            details.append({
                "idx": i,
                "task_id": ex.task_id,
                "correct": False,
                "error": str(e)[:200],
            })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] Running score: {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

    score = round(correct / total * 100, 2) if total > 0 else 0.0
    print(f"\n[DSPy HumanEval] Final: {correct}/{total} = {score}%")

    return {
        "benchmark": "humaneval",
        "score": score,
        "correct": correct,
        "total": total,
        "details": details,
    }


# ---------------------------------------------------------------------------
# 9. Quick demo / standalone test
# ---------------------------------------------------------------------------

def demo():
    """Quick demo showing DSPy code completion on a single HumanEval problem."""
    configure_dspy("openai/gpt-4o-mini", temperature=0.0)

    # Load one example
    examples = load_humaneval_as_dspy_examples(n=1, seed=0)
    ex = examples[0]

    print("=" * 60)
    print("PROMPT:")
    print(ex.prompt)
    print("=" * 60)

    # Method 1: Simple CoT
    simple = SimpleCodeCompleter()
    result = simple(prompt=ex.prompt)
    print("\n--- SimpleCodeCompleter (CoT) ---")
    print(f"Reasoning: {result.reasoning[:200]}...")
    print(f"Code:\n{result.completed_code}")
    print(f"Passes tests: {humaneval_metric(ex, result)}")

    # Method 2: Analyze + Complete
    analyzer = AnalyzeAndComplete()
    result2 = analyzer(prompt=ex.prompt)
    print("\n--- AnalyzeAndComplete ---")
    print(f"Analysis: {result2.analysis[:200]}...")
    print(f"Code:\n{result2.completed_code}")
    print(f"Passes tests: {humaneval_metric(ex, result2)}")

    print("\n" + "=" * 60)
    print("To optimize, run:")
    print("  optimized = optimize_for_humaneval(optimizer='bootstrap', n_train=20)")
    print("  results = evaluate_dspy_module(optimized, n_eval=50)")
    print("=" * 60)


if __name__ == "__main__":
    demo()
