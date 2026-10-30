# Experiment results

### A. Stage-1 retrieval ablation

Mean over 12 roles and 3 data seeds (pool = 4000). 'gap' = recall of qualified experts with a clear headline minus recall of equally qualified experts with a vague/buzzword headline (0 is fair).

| config | recall | recall (clear headline) | recall (vague headline) | gap |
|---|---|---|---|---|
| blob (naive) | 0.757 ± 0.004 | 0.776 | 0.729 | +0.046 ± 0.018 |
| fielded, balanced | 0.716 ± 0.001 | 0.753 | 0.661 | +0.092 ± 0.017 |
| fielded, notice-style | 0.700 ± 0.001 | 0.785 | 0.575 | +0.209 ± 0.018 |
| adaptive, notice-style | 0.707 ± 0.006 | 0.700 | 0.719 | -0.019 ± 0.020 |
| fielded, headline 60% | 0.641 ± 0.002 | 0.806 | 0.399 | +0.407 ± 0.020 |
| adaptive, headline 60% | 0.674 ± 0.003 | 0.666 | 0.689 | -0.023 ± 0.025 |

### B. Stage-2 grader comparison

Same Stage-1 retrieval. An LLM grader can be dropped in via `talentlens.llm.LLMGrader` and scored with the same harness.

| grader | surfaced_recall | precision_at_20 | ndcg_at_20 | qualified_lost_at_threshold |
|---|---|---|---|---|
| rule | 0.631 | 0.979 | 0.885 | 29.083 |
| rule + typo tolerance | 0.636 | 0.979 | 0.886 | 26.917 |

### C. Appeal-aware hard filters (rescue queue)

Per-role averages. 'queue_size' is human review workload; 'qualified_rescued' are truly qualified experts that would otherwise have been silently dropped by a hard filter.

| setting | surfaced_recall | qualified_rescued | queue_size | queue_precision | qualified_wrongly_rejected |
|---|---|---|---|---|---|
| off (auto-reject) | 0.601 | 0.000 | 0.000 | 0.000 | 26.417 |
| on, budget 10 | 0.615 | 4.833 | 10.000 | 0.483 | 0.667 |
| on, budget 30 | 0.636 | 12.583 | 28.667 | 0.432 | 0.667 |
| on, budget 60 | 0.656 | 20.583 | 48.667 | 0.402 | 0.667 |
| on, unlimited | 0.660 | 22.583 | 52.583 | 0.401 | 0.667 |

### D. Final-ranking blend

Top-20 quality of the ranking shown to project leads.

| ranking | tier | precision_at_20 | ndcg_at_20 |
|---|---|---|---|
| retrieval only (w=0) | pay >= $100 | 0.975 | 0.811 |
| retrieval only (w=0) | pay < $100 | 0.983 | 0.840 |
| balanced (w=0.5) | pay >= $100 | 0.975 | 0.864 |
| balanced (w=0.5) | pay < $100 | 0.983 | 0.900 |
| AI match only (w=1) | pay >= $100 | 0.992 | 0.971 |
| AI match only (w=1) | pay < $100 | 1.000 | 0.979 |
| pay-weighted (0.3-0.7) | pay >= $100 | 0.975 | 0.883 |
| pay-weighted (0.3-0.7) | pay < $100 | 0.983 | 0.889 |

### E. Counterfactual audit (qualified experts only)

One attribute changed, everything else identical. d_score_sd = change in Stage-1 score in pool standard deviations. shortlist_drop = share of shortlisted experts pushed OUT by the change; shortlist_gain = share of excluded experts pulled IN. Ground truth says none of these attributes should matter.

| retrieval | change | n | d_score_sd | shortlist_drop | shortlist_gain | d_match | pass_change |
|---|---|---|---|---|---|---|---|
| fielded | institution: elite -> local | 571 | -0.025 | 0.019 | 0.000 | 0.000 | 0.000 |
| fielded | institution: local -> elite | 2263 | 0.040 | 0.002 | 0.052 | 0.000 | 0.000 |
| fielded | location: EU -> non-EU | 1366 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fielded | location: non-EU -> EU | 3182 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fielded | headline: informative -> vague | 2785 | -0.830 | 0.310 | 0.012 | 0.000 | 0.000 |
| fielded | headline: vague -> informative | 1877 | 0.710 | 0.008 | 0.420 | 0.000 | 0.000 |
| adaptive | institution: elite -> local | 571 | -0.027 | 0.022 | 0.015 | 0.000 | 0.000 |
| adaptive | institution: local -> elite | 2263 | 0.041 | 0.000 | 0.067 | 0.000 | 0.000 |
| adaptive | location: EU -> non-EU | 1366 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| adaptive | location: non-EU -> EU | 3182 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| adaptive | headline: informative -> vague | 2785 | -0.100 | 0.050 | 0.383 | 0.000 | 0.000 |
| adaptive | headline: vague -> informative | 1877 | 0.166 | 0.188 | 0.214 | 0.000 | 0.000 |

### F. Shadow audit of the non-shortlisted pool (40 graded per stratum, 160 per role)

Audit estimate vs. exhaustive grading of everyone outside the shortlist. CI coverage of the exact value: 100% over 12 roles; mean absolute error 14.8 experts (grading 160 profiles instead of ~3600).

| role | found | est_missed | ci | exact_missed | truth_missed | est_recall | recommended_extra_k |
|---|---|---|---|---|---|---|---|
| R01 | 220 | 144 | [83, 495] | 138 | 135 | 0.60 | 360 |
| R02 | 255 | 162 | [84, 527] | 220 | 199 | 0.61 | 1080 |
| R03 | 296 | 153 | [103, 492] | 155 | 132 | 0.66 | 360 |
| R04 | 175 | 72 | [38, 409] | 53 | 72 | 0.71 | 360 |
| R05 | 300 | 162 | [111, 501] | 162 | 132 | 0.65 | 360 |
| R06 | 304 | 144 | [83, 495] | 149 | 142 | 0.68 | 360 |
| R07 | 285 | 126 | [80, 466] | 188 | 149 | 0.69 | 360 |
| R08 | 311 | 81 | [44, 419] | 83 | 87 | 0.79 | 360 |
| R09 | 283 | 243 | [128, 631] | 233 | 123 | 0.54 | 2160 |
| R10 | 112 | 27 | [9, 355] | 24 | 16 | 0.81 | 360 |
| R11 | 324 | 135 | [87, 475] | 140 | 104 | 0.71 | 360 |
| R12 | 308 | 126 | [80, 466] | 132 | 113 | 0.71 | 360 |
