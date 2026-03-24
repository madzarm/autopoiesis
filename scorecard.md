# Approach Scorecard

## Final Results (200 GSM8K + 164 HumanEval, gpt-4o-mini, AFlow-matching eval)

| Approach | Search Algo | GSM8K/200 | HE/164 | **Avg** | Stages | Status |
|----------|------------|----------|--------|---------|--------|--------|
| **multi3_vote** | diverse ensemble | **95.5%** | **87.8%** | **91.7%** | 4 | **winner** |
| fused_single | compound prompt | 95.5% | 86.6% | 91.0% | 1 | strong |
| cot_baseline | — | 95.5% | 86.6% | 91.0% | 1 | baseline |
| adaptive_universal | multi-bench search | 95.5% | 85.4% | 90.5% | 2 | strong |
| llm_architect | LLM-as-search | 94.5% | 85.4% | 90.0% | 2 | strong |
| gen_then_review | generate + review | 91.5% | 86.6% | 89.0% | 2 | good |
| dag_evolve | graph evolution | 93.5% | 62.8% | 78.2% | 5 | vote hurts HE |

## vs Published SOTA (gpt-4o-mini)

| | GSM8K | HumanEval |
|---|---|---|
| **Ours (multi3_vote)** | **95.5%** (beats all) | 87.8% |
| AutoMaAS | 95.4% | **97.2%** |
| AFlow | 93.5% | 94.7% |
| MaAS | 92.3% | 92.9% |

## HumanEval Failure Analysis (21 failures on 164)
- 2 fixable: `decode_cyclic`, `decode_shift` (need hardcoded helpers, AFlow special-cases these)
- 19 model logic errors: genuine capability gap on harder problems
- 0 syntax/format errors: eval pipeline is clean

## Key Insight
3 diverse candidates + vote beats all complex pipelines. The diversity comes from different temperatures (0.0, 0.5, 0.8) and different system prompts. The vote selects the best via AST-based code quality detection.
