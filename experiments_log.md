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

### Entering Experiment Loop...
