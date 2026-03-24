#!/usr/bin/env python3
"""AutoFlow-Converge — Self-Directing Multi-Turn Agent with Convergence Detection.

Approach 11: Instead of a FIXED pipeline that the search algorithm discovers,
the agent ITSELF decides what to do next at each step. The agent gets:
1. The problem
2. Its previous reasoning steps
3. A meta-prompt asking: "What should you do next?"

Options: continue_reasoning, try_code, verify, finalize_answer

The agent converges when it's confident in its answer. This is fundamentally
different from all other approaches because there's NO FIXED ARCHITECTURE —
the architecture is EMERGENT from the agent's own reasoning about what to do next.

Inspired by: OneFlow (2601.12307) which shows single-agent multi-turn matches
multi-agent workflows, and ReAct which interleaves reasoning with actions.

What we SEARCH OVER: The meta-prompt instructions that guide the agent's
self-direction. This is a PROMPT PROGRAM that controls the agent's behavior,
not a fixed pipeline.
"""

import json
import re
import random
import copy
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm, STRONG, MID, CHEAP, get_session_cost, reset_cost_tracking
from evaluate import extract_number, extract_math_answer, load_gsm8k, load_humaneval


# ═══════════════════════════════════════════════════════════════
# SELF-DIRECTING AGENT
# ═══════════════════════════════════════════════════════════════

@dataclass
class MetaPrompt:
    """The meta-instructions that guide the self-directing agent."""
    name: str = "default"
    initial_instruction: str = "Solve this problem step by step."
    decision_prompt: str = "What should you do next? Choose: REASON_MORE, TRY_CODE, VERIFY, FINALIZE"
    max_turns: int = 3
    model: str = CHEAP


def execute_autoflow(meta: MetaPrompt, problem: str) -> str:
    """Execute a self-directing agent on a problem."""
    conversation = []
    final_answer = ""

    # Initial reasoning turn
    result = call_llm(
        prompt=f"{meta.initial_instruction}\n\nProblem: {problem}",
        system="You are a precise problem solver.",
        model=meta.model, temperature=0.0, max_tokens=2048
    )
    conversation.append({"role": "reasoning", "content": result["content"]})
    final_answer = result["content"]

    for turn in range(meta.max_turns - 1):
        # Ask the agent what to do next
        history = "\n\n".join(f"[{c['role']}]: {c['content'][:500]}" for c in conversation)

        decision_result = call_llm(
            prompt=f"""Problem: {problem}

Your work so far:
{history}

{meta.decision_prompt}
Reply with EXACTLY one of: REASON_MORE, TRY_CODE, VERIFY, FINALIZE""",
            system="Decision maker. Reply with EXACTLY one action word.",
            model=meta.model, temperature=0.0, max_tokens=50
        )
        decision = decision_result["content"].strip().upper()

        if "FINALIZE" in decision or "FINAL" in decision:
            break

        elif "CODE" in decision:
            code_result = call_llm(
                prompt=f"Write Python code to solve this. Print ONLY the final answer.\n\n{problem}",
                system="Expert Python programmer. Write clean code.",
                model=meta.model, temperature=0.0, max_tokens=1024
            )
            code_text = code_result["content"]
            # Try to execute
            code_match = re.search(r'```python\s*(.*?)\s*```', code_text, re.DOTALL)
            code = code_match.group(1) if code_match else code_text
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
                with contextlib.redirect_stdout(stdout):
                    exec(code, safe_globals)
                output = stdout.getvalue().strip()
                if output:
                    final_answer = f"#### {output.split(chr(10))[-1]}"
                    conversation.append({"role": "code", "content": f"Code output: {output}"})
            except:
                conversation.append({"role": "code", "content": f"Code failed. {code_text[:200]}"})

        elif "VERIFY" in decision:
            verify_result = call_llm(
                prompt=f"Problem: {problem}\n\nProposed answer:\n{final_answer}\n\nIs this correct? Check carefully.",
                system="Careful verifier.",
                model=meta.model, temperature=0.0, max_tokens=512
            )
            conversation.append({"role": "verify", "content": verify_result["content"]})
            if "INCORRECT" in verify_result["content"].upper():
                # Try to fix
                fix_result = call_llm(
                    prompt=f"Problem: {problem}\n\nPrevious answer:\n{final_answer}\n\n"
                           f"Feedback: {verify_result['content']}\n\nFix the errors.",
                    system="Fix errors precisely.",
                    model=meta.model, temperature=0.1, max_tokens=2048
                )
                final_answer = fix_result["content"]
                conversation.append({"role": "repair", "content": fix_result["content"]})

        elif "REASON" in decision:
            reason_result = call_llm(
                prompt=f"Problem: {problem}\n\nYour previous attempt:\n{final_answer}\n\n"
                       f"Think about this problem from a different angle. Double-check your work.",
                system="Precise problem solver. Think step by step.",
                model=meta.model, temperature=0.1, max_tokens=2048
            )
            final_answer = reason_result["content"]
            conversation.append({"role": "reasoning", "content": reason_result["content"]})

        else:
            break  # Unknown decision, finalize

    return final_answer


# ═══════════════════════════════════════════════════════════════
# SEARCH OVER META-PROMPTS
# ═══════════════════════════════════════════════════════════════

SEED_METAS = [
    MetaPrompt(
        name="basic",
        initial_instruction="Solve this problem step by step. Show your work. Put your final answer after ####.",
        decision_prompt="What should you do next? Choose: REASON_MORE, TRY_CODE, VERIFY, FINALIZE",
        max_turns=3,
    ),
    MetaPrompt(
        name="code_first",
        initial_instruction="Write Python code to solve this problem. Print the final answer.",
        decision_prompt="The code approach was tried. Choose: VERIFY, REASON_MORE, FINALIZE",
        max_turns=2,
    ),
    MetaPrompt(
        name="verify_heavy",
        initial_instruction="Solve carefully step by step. Answer after ####.",
        decision_prompt="Always VERIFY first, then FINALIZE if correct, or REASON_MORE if not.",
        max_turns=4,
    ),
    MetaPrompt(
        name="adaptive",
        initial_instruction="Analyze this problem. If it involves calculation, try TRY_CODE. If it's reasoning, think step by step. Answer clearly.",
        decision_prompt="Based on your confidence: high confidence → FINALIZE, medium → VERIFY, low → TRY_CODE or REASON_MORE",
        max_turns=3,
    ),
]


def mutate_meta(meta: MetaPrompt, model: str = STRONG) -> MetaPrompt:
    """LLM-guided mutation of meta-prompt."""
    prompt = f"""Improve this agent self-direction prompt:

Name: {meta.name}
Initial instruction: {meta.initial_instruction}
Decision prompt: {meta.decision_prompt}
Max turns: {meta.max_turns}

The agent uses these to decide how to solve problems. It can REASON_MORE, TRY_CODE, VERIFY, or FINALIZE.
Make the instructions more effective. Return JSON:
{{"name": "...", "initial_instruction": "...", "decision_prompt": "...", "max_turns": N}}
"""
    result = call_llm(prompt=prompt, system="Prompt optimizer. Return JSON.",
                      model=model, temperature=0.7, max_tokens=512, json_mode=True)
    try:
        data = json.loads(result["content"])
        return MetaPrompt(
            name=data.get("name", meta.name + "_m"),
            initial_instruction=data.get("initial_instruction", meta.initial_instruction),
            decision_prompt=data.get("decision_prompt", meta.decision_prompt),
            max_turns=min(data.get("max_turns", meta.max_turns), 5),
        )
    except:
        return meta


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def eval_meta(meta: MetaPrompt, gsm_samples: list, he_samples: list) -> dict:
    """Evaluate a meta-prompt on both benchmarks."""
    gsm_correct = 0
    he_correct = 0

    for sample in gsm_samples:
        response = execute_autoflow(meta, sample["question"])
        predicted = extract_number(response)
        gold = sample["gold_answer"]
        if predicted is not None and abs(predicted - gold) < 1e-6:
            gsm_correct += 1

    for sample in he_samples:
        response = execute_autoflow(meta, f"Complete this Python function:\n\n{sample['prompt']}")
        body = re.sub(r'```python\s*', '', response)
        body = re.sub(r'```\s*', '', body)
        lines = body.split('\n')
        if lines and lines[0].strip().startswith('def '):
            i = 1
            if i < len(lines) and ('"""' in lines[i] or "'''" in lines[i]):
                i += 1
                while i < len(lines) and '"""' not in lines[i] and "'''" not in lines[i]:
                    i += 1
                i += 1
            body = '\n'.join(lines[i:])
        full = sample["prompt"] + body + "\n" + sample["test"] + f"\ncheck({sample['entry_point']})"
        try:
            exec(full, {})
            he_correct += 1
        except:
            pass

    gsm_score = round(gsm_correct / max(len(gsm_samples), 1) * 100, 1)
    he_score = round(he_correct / max(len(he_samples), 1) * 100, 1)
    return {"gsm8k": gsm_score, "humaneval": he_score, "avg": (gsm_score + he_score) / 2}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_autoflow(n_gsm=20, n_he=10, n_iterations=10, seed=42):
    gsm_samples = load_gsm8k(n=n_gsm, seed=seed)
    he_samples = load_humaneval(n=n_he, seed=seed)

    print(f"═══ AutoFlow-Converge ═══")
    print(f"GSM8K: {len(gsm_samples)}, HumanEval: {len(he_samples)}")
    print(f"Iterations: {n_iterations}\n")

    best_score = 0.0
    best_meta = None
    population = list(SEED_METAS)

    print("── Seeds ──")
    scored = []
    for meta in population:
        reset_cost_tracking()
        result = eval_meta(meta, gsm_samples, he_samples)
        scored.append((meta, result))
        print(f"  {meta.name:15s} | GSM={result['gsm8k']:5.1f}% HE={result['humaneval']:5.1f}% avg={result['avg']:.1f}%")
        if result["avg"] > best_score:
            best_score = result["avg"]
            best_meta = meta

    print(f"Best seed: {best_score:.1f}%\n")

    print("── Search ──")
    for i in range(n_iterations):
        scored.sort(key=lambda x: x[1]["avg"], reverse=True)
        parent = scored[0][0]

        reset_cost_tracking()
        child = mutate_meta(parent)
        result = eval_meta(child, gsm_samples, he_samples)

        marker = ""
        if result["avg"] > best_score:
            best_score = result["avg"]
            best_meta = child
            marker = " *** NEW BEST ***"

        scored.append((child, result))
        scored.sort(key=lambda x: x[1]["avg"], reverse=True)
        scored = scored[:4]

        if (i + 1) % 2 == 0 or marker:
            print(f"  [{i+1:3d}/{n_iterations}] GSM={result['gsm8k']:5.1f}% HE={result['humaneval']:5.1f}% "
                  f"avg={result['avg']:.1f}%{marker}")

    print(f"\n{'═'*50}")
    print(f"AUTOFLOW-CONVERGE COMPLETE")
    print(f"{'═'*50}")
    print(f"Best avg: {best_score:.1f}%")
    print(f"Best meta: {best_meta.name}")
    print(f"  Initial: {best_meta.initial_instruction[:80]}...")
    print(f"  Decision: {best_meta.decision_prompt[:80]}...")
    print(f"  Max turns: {best_meta.max_turns}")

    return {"avg": best_score, "meta": best_meta}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gsm", type=int, default=20)
    p.add_argument("--he", type=int, default=10)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()
    run_autoflow(args.gsm, args.he, args.iters)
