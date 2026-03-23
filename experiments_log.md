# Experiments Log

## Session 1 — 2026-03-23

### Setup Phase
- Initialized project, installed dependencies
- Built core infrastructure: llm.py, agents.py, evaluate.py, archive.py, search.py, run_experiment.py
- Confirmed OpenAI API key works, end-to-end loop functional

### Baseline Results (GSM8K, 50 samples, gpt-4.1-nano)
| Agent | Score | Cost | Time |
|-------|-------|------|------|
| Direct | 90.0% | $0.002 | 70s |
| CoT | 92.0% | $0.004 | 136s |
| Self-Refine | 88.0% | $0.020 | 465s |

**Key insight**: Self-refine is WORSE than direct while costing 10x more. The reflection step sometimes introduces errors. CoT gives a reliable +2% boost.

### Design Phase
- Designed AIDE (Adaptive Immune-inspired Design Evolution)
- Core idea: error-diagnosis-driven mutation + quality-diversity archive + cost-aware Pareto selection
- Wrote design.md with full approach specification

### AIDE Search (LLM-guided proposal)
- Added parallel evaluation (16x concurrent LLM calls) → 11.5x speedup
- Ran 10 search steps with LLM proposals
- **Best found: "cot_decompose_reflect_vote" at 96% on GSM8K (50 samples)**
  - decompose reasoning + divide_conquer planning + self_refine + majority_vote(3)
  - math olympiad coach persona
- Validated on 200 samples: 93% (the 96% was partly noise)

### Evolutionary Search (V1)
- Ran full evolutionary search: 6 population, 5 generations, 100 samples
- Converged at 95% — simple CoT seed was never beaten
- **Key insight: for gpt-4.1-nano on GSM8K, simple CoT hits a ceiling (~95%).
  Fancy prompt engineering can't exceed the model's inherent capability.**
- Average population fitness rose (91.5% → 94.3%) showing evolution works,
  but the ceiling prevents breakthroughs

### V2 Architecture Expansion
- Added code_solve, plan_solve_verify, classify_route, ensemble_diverse, progressive_refine
- Added ARC-Challenge, DROP, MMLU benchmarks

### Multi-Benchmark Results (50 samples, gpt-4.1-nano)
| Method | GSM8K | ARC | DROP F1 | MMLU | Notes |
|--------|-------|-----|---------|------|-------|
| CoT V2 | **96.0%** | 86.0% | **53.5%** | **74.0%** | Best overall |
| Progressive Refine | 92.0% | **88.0%** | 22.5% | - | Best on ARC |
| Ensemble Diverse | 94.0% | 86.0% | - | - | Expensive |
| Code Solve | 92.0% | 82.0% | - | - | Good for math |
| Plan-Solve-Verify | 82.0% | - | 2.2% | - | Too complex |

### Key Patterns Discovered
1. **Majority vote > best_of_n > debate** — best_of_n adds a judge that introduces errors
2. **Self-refine with strong personas works** — but only for 1-2 rounds
3. **Code generation is competitive for math** but not for knowledge/reading tasks
4. **Progressive refine helps on knowledge tasks** (ARC +2% over CoT)
5. **DROP needs text-specific answer formatting** — math format destroys F1

### V2 Evolution Running...
- Multi-benchmark (GSM8K + ARC) evolutionary search
- Error-trace-conditioned mutation at architecture level
- Population of 5, 5 generations
