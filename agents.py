"""Agent representation and baseline agent implementations.

Agents are represented as structured configs (not raw code). Each agent config
specifies a pipeline of modules that process a query and produce an answer.

Module types:
- reasoning: How the agent thinks (CoT, self-consistency, decomposition, etc.)
- planning: How the agent breaks down problems (none, step-by-step, recursive)
- reflection: Whether/how the agent reviews its own output
- ensemble: How multiple attempts are combined (none, majority-vote, best-of-n)
- persona: System prompt / role framing

The config is a JSON-serializable dict that fully specifies agent behavior.
"""

import json
from typing import Optional
from pydantic import BaseModel, Field
from llm import call_llm, CHEAP, MID


class AgentConfig(BaseModel):
    """Structured configuration for an agent."""
    name: str = "unnamed"
    reasoning: str = "direct"  # direct, cot, cot_sc, decompose, analogy, abstract
    planning: str = "none"  # none, step_by_step, recursive, divide_conquer
    reflection: str = "none"  # none, self_check, self_refine, critic
    ensemble: str = "none"  # none, majority_vote, best_of_n, debate
    ensemble_n: int = 1  # number of samples for ensemble methods
    persona: str = ""  # system prompt prefix
    output_format: str = "free"  # free, structured, step_numbered
    model: str = CHEAP  # which LLM model to use
    temperature: float = 0.7
    max_tokens: int = 2048

    # Advanced
    decompose_strategy: str = "sequential"  # sequential, parallel, tree
    reflection_rounds: int = 1
    custom_instructions: str = ""  # additional instructions appended to prompt


def build_prompt(config: AgentConfig, question: str, passage: str = "") -> tuple[str, str]:
    """Build system and user prompts from an agent config.

    Returns (system_prompt, user_prompt).
    """
    system_parts = []

    # Persona
    if config.persona:
        system_parts.append(config.persona)
    else:
        system_parts.append("You are a helpful AI assistant that solves problems accurately.")

    # Reasoning instructions
    reasoning_instructions = {
        "direct": "Answer the question directly and concisely.",
        "cot": "Think step by step before giving your final answer. Show your reasoning.",
        "cot_sc": "Think step by step carefully. Show your reasoning in detail.",
        "decompose": "Break this problem into smaller sub-problems. Solve each sub-problem, then combine the results.",
        "analogy": "Think of an analogous, simpler problem first. Solve that, then apply the approach to the original problem.",
        "abstract": "First identify the abstract structure of this problem, then solve it using that structure.",
    }
    system_parts.append(reasoning_instructions.get(config.reasoning, reasoning_instructions["direct"]))

    # Planning
    if config.planning == "step_by_step":
        system_parts.append("Create a plan before solving. List your steps, then execute each one.")
    elif config.planning == "recursive":
        system_parts.append("If the problem is complex, recursively break it down until each piece is simple.")
    elif config.planning == "divide_conquer":
        system_parts.append("Divide the problem into independent parts, solve each, then combine.")

    # Output format
    if config.output_format == "structured":
        system_parts.append("Structure your response with clear sections: Reasoning, Steps, Answer.")
    elif config.output_format == "step_numbered":
        system_parts.append("Number each step of your reasoning. Put your final answer after ####.")

    # Custom instructions
    if config.custom_instructions:
        system_parts.append(config.custom_instructions)

    # Final answer format
    system_parts.append("Always put your final numeric answer after #### on its own line. Example: #### 42")

    system_prompt = "\n\n".join(system_parts)

    # User prompt
    if passage:
        user_prompt = f"Passage: {passage}\n\nQuestion: {question}"
    else:
        user_prompt = f"Question: {question}"

    return system_prompt, user_prompt


def run_agent(config: AgentConfig, question: str, passage: str = "") -> str:
    """Run an agent on a single question and return the response text."""
    system_prompt, user_prompt = build_prompt(config, question, passage)

    if config.ensemble == "none" or config.ensemble_n <= 1:
        result = call_llm(
            prompt=user_prompt,
            system=system_prompt,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        response = result["content"]

        # Reflection
        if config.reflection == "self_check":
            response = _self_check(config, question, response, passage)
        elif config.reflection == "self_refine":
            for _ in range(config.reflection_rounds):
                response = _self_refine(config, question, response, passage)
        elif config.reflection == "critic":
            response = _critic_refine(config, question, response, passage)

        return response

    elif config.ensemble == "majority_vote":
        return _majority_vote(config, question, passage)
    elif config.ensemble == "best_of_n":
        return _best_of_n(config, question, passage)
    elif config.ensemble == "debate":
        return _debate(config, question, passage)

    return ""


def _self_check(config: AgentConfig, question: str, response: str, passage: str = "") -> str:
    """Have the agent check its own answer."""
    check_prompt = (
        f"You previously answered this question:\n\n"
        f"Question: {question}\n\n"
        f"Your answer: {response}\n\n"
        f"Check your work carefully. If you find any errors, provide the corrected answer. "
        f"If your answer is correct, restate it. Put your final answer after ####."
    )
    result = call_llm(
        prompt=check_prompt,
        system="You are a careful reviewer. Check the reasoning and arithmetic step by step.",
        model=config.model,
        temperature=0.3,
        max_tokens=config.max_tokens,
    )
    return result["content"]


def _self_refine(config: AgentConfig, question: str, response: str, passage: str = "") -> str:
    """Self-refine: critique then improve."""
    # Critique
    critique_prompt = (
        f"Question: {question}\n\n"
        f"Proposed answer:\n{response}\n\n"
        f"Provide specific, actionable feedback on this answer. "
        f"Point out any errors in reasoning or calculation."
    )
    critique = call_llm(
        prompt=critique_prompt,
        system="You are a critical reviewer. Find flaws in reasoning and calculation.",
        model=config.model,
        temperature=0.3,
        max_tokens=1024,
    )

    # Refine
    refine_prompt = (
        f"Question: {question}\n\n"
        f"Your previous answer:\n{response}\n\n"
        f"Feedback:\n{critique['content']}\n\n"
        f"Using this feedback, provide an improved answer. Put your final answer after ####."
    )
    result = call_llm(
        prompt=refine_prompt,
        system="Improve your answer based on the feedback. Be precise.",
        model=config.model,
        temperature=0.3,
        max_tokens=config.max_tokens,
    )
    return result["content"]


def _critic_refine(config: AgentConfig, question: str, response: str, passage: str = "") -> str:
    """Have a separate critic agent review, then refine."""
    critic_prompt = (
        f"A student answered this question:\n\n"
        f"Question: {question}\n\n"
        f"Student's answer:\n{response}\n\n"
        f"As a teacher, review this answer. Identify specific errors and suggest corrections."
    )
    critique = call_llm(
        prompt=critic_prompt,
        system="You are an expert teacher. Identify errors precisely.",
        model=config.model,
        temperature=0.3,
        max_tokens=1024,
    )

    refine_prompt = (
        f"Question: {question}\n\n"
        f"Your previous answer:\n{response}\n\n"
        f"Teacher's feedback:\n{critique['content']}\n\n"
        f"Incorporate the feedback and provide your final corrected answer after ####."
    )
    result = call_llm(
        prompt=refine_prompt,
        system="You are a diligent student correcting your work.",
        model=config.model,
        temperature=0.3,
        max_tokens=config.max_tokens,
    )
    return result["content"]


def _majority_vote(config: AgentConfig, question: str, passage: str = "") -> str:
    """Generate multiple answers and take majority vote."""
    import re
    from collections import Counter

    system_prompt, user_prompt = build_prompt(config, question, passage)
    answers = []

    for _ in range(config.ensemble_n):
        result = call_llm(
            prompt=user_prompt,
            system=system_prompt,
            model=config.model,
            temperature=max(config.temperature, 0.7),  # need diversity
            max_tokens=config.max_tokens,
        )
        text = result["content"]
        # Extract numeric answer
        match = re.search(r'####\s*([\-\d,\.]+)', text)
        if match:
            answers.append(match.group(1).replace(',', ''))

    if not answers:
        # Fallback: return last response
        return text

    # Majority vote
    counter = Counter(answers)
    most_common = counter.most_common(1)[0][0]
    return f"#### {most_common}"


def _best_of_n(config: AgentConfig, question: str, passage: str = "") -> str:
    """Generate N answers, have the model pick the best."""
    system_prompt, user_prompt = build_prompt(config, question, passage)
    responses = []

    for _ in range(config.ensemble_n):
        result = call_llm(
            prompt=user_prompt,
            system=system_prompt,
            model=config.model,
            temperature=max(config.temperature, 0.7),
            max_tokens=config.max_tokens,
        )
        responses.append(result["content"])

    # Have the model pick the best
    selection_prompt = (
        f"Question: {question}\n\n"
        f"Here are {len(responses)} candidate answers:\n\n"
    )
    for i, r in enumerate(responses):
        selection_prompt += f"--- Answer {i+1} ---\n{r}\n\n"
    selection_prompt += (
        "Which answer is most likely correct? Analyze each carefully, "
        "then provide the best answer after ####."
    )
    result = call_llm(
        prompt=selection_prompt,
        system="You are an expert judge. Pick the most correct answer.",
        model=config.model,
        temperature=0.2,
        max_tokens=config.max_tokens,
    )
    return result["content"]


def _debate(config: AgentConfig, question: str, passage: str = "") -> str:
    """Multi-agent debate: agents argue, then a judge decides."""
    system_prompt, user_prompt = build_prompt(config, question, passage)

    # Generate diverse initial answers
    agent_responses = []
    personas = [
        "You are a careful, methodical problem solver who checks every step.",
        "You are an intuitive problem solver who looks for patterns and shortcuts.",
        "You are a skeptical problem solver who questions every assumption.",
    ]
    n = min(config.ensemble_n, len(personas))

    for i in range(n):
        result = call_llm(
            prompt=user_prompt,
            system=personas[i] + "\n\n" + system_prompt,
            model=config.model,
            temperature=0.7,
            max_tokens=config.max_tokens,
        )
        agent_responses.append(result["content"])

    # Debate round: each agent sees others' answers
    debate_responses = []
    for i in range(n):
        others = "\n\n".join(
            f"Agent {j+1}'s answer:\n{agent_responses[j]}"
            for j in range(n) if j != i
        )
        debate_prompt = (
            f"Question: {question}\n\n"
            f"Your original answer:\n{agent_responses[i]}\n\n"
            f"Other agents' answers:\n{others}\n\n"
            f"Consider the other perspectives. Do you want to change your answer? "
            f"Provide your final answer after ####."
        )
        result = call_llm(
            prompt=debate_prompt,
            system=personas[i],
            model=config.model,
            temperature=0.3,
            max_tokens=config.max_tokens,
        )
        debate_responses.append(result["content"])

    # Judge
    judge_prompt = (
        f"Question: {question}\n\n"
        f"After debate, the agents gave these final answers:\n\n"
    )
    for i, r in enumerate(debate_responses):
        judge_prompt += f"--- Agent {i+1} ---\n{r}\n\n"
    judge_prompt += "As the final judge, determine the correct answer. Put it after ####."

    result = call_llm(
        prompt=judge_prompt,
        system="You are an impartial judge. Analyze all arguments and determine the correct answer.",
        model=config.model,
        temperature=0.1,
        max_tokens=config.max_tokens,
    )
    return result["content"]


# ─── Predefined baseline configs ─────────────────────────────

BASELINE_CONFIGS = {
    "direct": AgentConfig(
        name="direct",
        reasoning="direct",
        model=CHEAP,
        temperature=0.0,
    ),
    "cot": AgentConfig(
        name="cot",
        reasoning="cot",
        output_format="step_numbered",
        model=CHEAP,
        temperature=0.0,
    ),
    "cot_sc5": AgentConfig(
        name="cot_sc5",
        reasoning="cot",
        output_format="step_numbered",
        ensemble="majority_vote",
        ensemble_n=5,
        model=CHEAP,
        temperature=0.7,
    ),
    "self_refine": AgentConfig(
        name="self_refine",
        reasoning="cot",
        output_format="step_numbered",
        reflection="self_refine",
        reflection_rounds=1,
        model=CHEAP,
        temperature=0.0,
    ),
    "debate_3": AgentConfig(
        name="debate_3",
        reasoning="cot",
        output_format="step_numbered",
        ensemble="debate",
        ensemble_n=3,
        model=CHEAP,
        temperature=0.7,
    ),
}
