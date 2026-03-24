# SOTA Baselines (from literature) — Updated with Full Paper Research

## Comprehensive Comparison Table (gpt-4o-mini backbone)

| Method | GSM8K | MATH | HumanEval | MBPP | HotpotQA | DROP | Source |
|--------|-------|------|-----------|------|----------|------|--------|
| **AutoMaAS** | **95.4%** | 57.1% | **97.2%** | **88.8%** | — | — | Preprint Oct 2025 |
| **AIDE (ours)** | 94.16% | **58.0%** | 93.29% | 87.16% | — | — | This work |
| MaAS | 92.30% | 51.82% | 92.85% | 82.17% | — | — | ICML 2025 Oral |
| AFlow | 93.5% | 56.2% | 94.7% | 83.4% | 73.5 | 80.6 | ICLR 2025 Oral |
| AgentSquare | 87.6% | 48.5% | 89.1% | 78.5% | — | — | ICLR 2025 |
| ADAS | 86.1% | 43.2% | 84.2% | 68.1% | — | — | ICLR 2025 |
| DyLAN | 90.0% | 48.6% | 90.4% | 77.3% | — | — | — |
| GPTSwarm | 89.1% | 47.9% | 89.3% | 77.4% | — | — | — |
| CoT SC (5) | 92.7% | 50.4% | 91.6% | 73.6% | 68.9 | 78.8 | — |
| CoT | 92.4% | 48.8% | 88.6% | 71.8% | 67.9 | 78.5 | — |
| IO (direct) | 92.7% | 48.6% | 87.0% | 71.8% | 68.1 | 68.3 | — |

## Notes on Benchmark Alignment

- **MATH**: AFlow uses MATH level 5 only (617 problems from 4 types). We use lighteval/MATH-Hard (1324 problems). Scores may not be directly comparable.
- **GSM8K**: Full test set (1319 samples) used by all papers. Our 94.16% is on full test set.
- **HumanEval**: Full 164 problems. AFlow's 94.7% is their highest; AutoMaAS claims 97.2%.
- **MBPP**: AFlow uses full MBPP (500 test). We use sanitized version (257 test). Different test sets.

## Key Papers Summary

### AutoMaAS (Oct 2025) — HIGHEST PUBLISHED NUMBERS
- Self-evolving multi-agent NAS with dynamic operator lifecycle
- Cost-aware 5-dimensional cost tensor
- GSM8K 95.4%, MATH 57.1%, HumanEval 97.2%, MBPP 88.8%
- Uses operator fusion (CoT+Self-Refine: 92% success, +4.2%)
- Preprint — numbers may not be fully verified

### AFlow (ICLR 2025 Oral)
- MCTS over code-represented workflow space
- Optimizer: Claude-3.5-sonnet, Executor: GPT-4o-mini
- 20 max rounds, early stopping after 5 without improvement
- Average 80.3% across 6 benchmarks
- Key: code-as-edges representation, predefined operators (Generate, Review, Revise, Ensemble, Test, Programmer)

### MaAS (ICML 2025 Oral, top ~1%)
- Agentic supernet with query-dependent sampling
- Early-exit for easy queries, 6-45% inference cost
- Training cost $3.38 vs AFlow $22.50
- Key: probabilistic distribution over architectures, not single architecture

### EvoMAS (Feb 2026, withdrawn)
- Configuration-space evolution with execution-trace-guided mutation
- SWE-Bench-Verified: 79.1% (Claude-4.5-Sonnet), 63.8% (Claude-3.5-Sonnet)
- BBEH: 58.7%, WorkBench: 48.9%
- Key: evolves configs not code, 98%+ execution reliability

### ADAS (Hu et al., ICLR 2025)
- Meta Agent Search — LLM generates Python code for agents
- Evaluated on GPT-3.5: DROP 79.4 F1, MGSM 53.4%, MMLU 69.6%, GPQA 34.6%
- Cost: ~$300-500 per search run
- Key: Turing-complete search space (Python code)
