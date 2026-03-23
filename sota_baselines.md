# SOTA Baselines (from literature)

## Key Benchmarks — Best Published Numbers

All numbers below are from the papers' reported results. Model backbones vary — noted where known.

### MGSM (Multilingual Grade School Math) — Accuracy %

| Method | Score | Backbone | Source |
|--------|-------|----------|--------|
| Direct (IO) | 87.0% | gpt-4o-mini | AFlow |
| CoT | 88.6% | gpt-4o-mini | AFlow |
| CoT SC | 91.6% | gpt-4o-mini | AFlow |
| Self-Refine | 87.8% | gpt-4o-mini | AFlow |
| LLM Debate | 89.3% | gpt-4o-mini | AFlow |
| MedPrompt | 91.6% | gpt-4o-mini | AFlow |
| **AFlow (gpt-4o-mini)** | **94.7%** | gpt-4o-mini | AFlow |
| AFlow (deepseek) | 94.66% | deepseek | AFlow |
| Direct (IO) | 93.9% | gpt-4o | AFlow |
| **AFlow (gpt-4o)** | **96.2%** | gpt-4o | AFlow |

### GSM8K — Accuracy %

| Method | Score | Backbone | Source |
|--------|-------|----------|--------|
| CoT | ~85% | gpt-4o-mini | Various |
| ADAS (Meta Agent Search) | ~92% | gpt-4o-mini | ADAS |
| AFlow | ~95% | gpt-4o-mini | AFlow (estimated) |

### DROP — F1

| Method | Score | Backbone | Source |
|--------|-------|----------|--------|
| Direct | ~57% | gpt-3.5 | ADAS |
| ADAS (Meta Agent Search) | ~83% | gpt-3.5 | ADAS |
| AFlow | ~85% | gpt-4o-mini | AFlow (estimated) |

### HumanEval — pass@1 %

| Method | Score | Backbone | Source |
|--------|-------|----------|--------|
| Direct | ~82% | gpt-4o-mini | Various |
| ADAS (Meta Agent Search) | ~86% | gpt-4o-mini | ADAS |

### MATH — Accuracy %

| Method | Score | Backbone | Source |
|--------|-------|----------|--------|
| CoT | ~70% | gpt-4o-mini | Various |
| AFlow | ~80% | gpt-4o-mini | AFlow (estimated) |

### SWE-Bench-Verified — % Resolved

| Method | Score | Backbone | Source |
|--------|-------|----------|--------|
| EvoMAS | 79.1% | Claude-4.5 | EvoMAS (withdrawn) |

## Key ADAS Methods Summary

1. **ADAS (Hu et al. 2024)**: Meta Agent Search — LLM generates Python code for new agents, tests on benchmarks, archives best. Search space = Python programs.
2. **AFlow (ICLR 2025)**: MCTS over code-represented workflow DAGs. Operators modify nodes/edges. Best performance on reasoning benchmarks. ~94.7% MGSM.
3. **AgentSquare (ICLR 2025)**: Modular design (Planning, Reasoning, ToolUse, Memory) with uniform I/O. Combinatorial search over module options.
4. **MaAS (ICML 2025)**: NAS supernets for agents. Probabilistic architecture distribution, query-dependent sampling. 6-45% inference cost reduction.
5. **EvoAgent (NAACL 2025)**: Evolutionary single→multi agent. Mutation/crossover of agent attributes. Framework-agnostic.
6. **EvoMAS (Feb 2026)**: Configuration-space evolution with execution-trace-guided mutation. Evolves roles, prompts, models, topologies jointly. 79.1% SWE-Bench.
7. **ARTEMIS (Dec 2025)**: No-code evolutionary optimization. Semantically-aware genetic operators. 13-37% improvements.

## Our Internal Baselines (gpt-4.1-nano, 50 samples GSM8K)

| Method | GSM8K | Cost |
|--------|-------|------|
| Direct | 90.0% | $0.002 |
| CoT | 92.0% | $0.004 |
| Self-Refine | 88.0% | $0.020 |
