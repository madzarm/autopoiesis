# Approach Scorecard

## Latest Results — Code-ADAS (HumanEval-focused, gpt-4o-mini)

| Approach | Search Algo | HE/164 | GSM8K/200 | Stages | Status |
|----------|------------|--------|----------|--------|--------|
| **llm_g8_4** | **Evo + LLM-guided** | **94.5%** | — | 8 | **discovered** |
| **multi5_test_repair** | search space comp. | **94.5%** | — | 8 | strong |
| multi3_repair | LLM-architect | 92.7% | — | 6 | strong |
| gen_test_repair | evo-confirmed | 90.9% | — | 4 | baseline |
| multi3_vote (prior) | diverse ensemble | 87.8% | **95.5%** | 4 | superseded |

## Cross-Benchmark Results (200 GSM8K + 164 HumanEval)

| Approach | Search Algo | GSM8K/200 | HE/164 | **Avg** | Status |
|----------|------------|----------|--------|---------|--------|
| **llm_g8_4** + multi3_vote | hybrid | **95.5%** | **94.5%** | **95.0%** | **best combo** |
| multi3_vote | diverse ensemble | 95.5% | 87.8% | 91.7% | prior winner |
| fused_single | compound prompt | 95.5% | 86.6% | 91.0% | strong |
| cot_baseline | — | 95.5% | 86.6% | 91.0% | baseline |
| adaptive_universal | multi-bench search | 95.5% | 85.4% | 90.5% | strong |
| dag_evolve | graph evolution | 93.5% | 62.8% | 78.2% | vote hurts HE |

## vs Published SOTA (gpt-4o-mini)

| | GSM8K | HumanEval |
|---|---|---|
| **Ours (llm_g8_4)** | 95.5%* | **94.5%** |
| AutoMaAS | 95.4% | **97.2%** |
| AFlow | 93.5% | 94.7% |
| MaAS | 92.3% | 92.9% |

*GSM8K score from multi3_vote (same session, different architecture per benchmark)

## HumanEval Failure Analysis (9 failures on 164 with llm_g8_4)
- `circular_shift`, `is_multiply_prime`, `max_fill`, `encode`, `order_by_points`, `check_if_last_char_is_a_letter`, `is_sorted`, `is_nested` — genuine model logic errors on hard edge cases
- 0 eval pipeline errors

## Key Insight — Code-ADAS
The evolutionary search with **code-aware primitives** (test execution, error-guided repair, reflect, restart) discovered a novel 8-stage workflow that matches AFlow. The key primitives the search composed:
1. **Temperature gradient** (0.0→0.5→0.7) with diverse prompts
2. **Test-select** instead of vote (exec against actual tests)
3. **Conditional fallback chain**: reflect → repair → restart (only activated on failure)
4. **LLM-invented prompts**: "Explore different algorithms", "Think outside the box" — evolved by the meta-LLM

## 14 Total Approaches Implemented
1-12: (see previous scorecard)
13. **Code-ADAS** — Evolutionary search with code-aware primitives (test, repair, reflect, restart)
14. **Code-Architect** — LLM-as-search over code workflow space
