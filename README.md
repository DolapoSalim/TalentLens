# TalentLens: an auditable two-stage expert-matching pipeline

A capstone project that rebuilds the *kind* of staged AI matching system described in public automated-decision-making
disclosures from expert marketplaces (semantic retrieval, then AI rubric scoring, then hard filters, then human review),
and then asks the question those disclosures leave open: **who does this funnel silently lose, and how would you know?**

> Independent project. It is reconstructed from a public disclosure's *description* of the process, not from any company's
> code or data. All data is synthetic; no real people, names, photos or biometrics are used anywhere.

## What it does

```
 Role + criteria ──► "ideal candidate" query
                          │
   Stage 1  FieldIndex (headline | role | experience | education | skills | location)
            hybrid dense+sparse similarity, two-phase (recently active first, then broad)
            modes: blob · fielded (headline-heavy) · ADAPTIVE (this project)
                          │  shortlist (~10% of pool)
   Stage 2  Grader → per criterion: score 0/1/2 + CONFIDENCE + evidence
            hard filters · average → match score
                          │
            confident hard-fail ──► reject
            low-confidence hard-fail ──► RESCUE QUEUE (human decides)   ← innovation
                          │
   Rank     final = (1-w)·retrieval + w·match,  w grows with pay, bounded [0.3, 0.7]
                          │
   Human    project lead reviews / overrides (logged)   ·   audit log (JSONL)

 Audits   counterfactual bias audit · shadow audit of the non-shortlisted pool · per-expert explanation & contest flow
```

## Innovations (what is new relative to a plain re-implementation)

1. **Adaptive headline weighting.** Headline-heavy retrieval rewards people who write good headlines, not people who are
   qualified. Each profile's headline weight is scaled by how consistent the headline is with the rest of the profile.
2. **Appeal-aware hard filters.** Graders emit a confidence. A hard-filter 0 that rests on *absence of evidence*
   ("language not listed") goes to a human rescue queue instead of auto-rejecting.
3. **Shadow audit.** Stage 1 hides everyone outside the shortlist from every human. Grading a small stratified sample of
   the hidden pool estimates how many passing experts were dropped (with CIs) and recommends a larger `k`.
4. **Counterfactual audit harness.** Change one attribute that should not matter (institution prestige, location,
   headline style) on otherwise identical qualified experts, and measure Stage-1 and Stage-2 effects. Works with any grader, including an LLM.
5. **Contestability.** For any expert, shortlisted or not: which stage eliminated them, by how much, which fields drove the
   score, what evidence would change each grade, a headline what-if, and a logged human-review request.
6. **Synthetic benchmark with hidden ground truth**, so every claim above is measured rather than asserted.

## Results (pool 4,000 · 12 roles · shortlist ≈ pool/10 · `python scripts/run_experiments.py`, fully reproducible)

**A. Stage-1 retrieval** (mean of 3 data seeds; "gap" = recall of qualified experts with a clear headline minus equally qualified experts with a vague headline)

| config | recall | clear headline | vague headline | gap |
|---|---|---|---|---|
| blob (naive concatenation) | 0.757 | 0.776 | 0.729 | +0.046 |
| fielded, balanced | 0.716 | 0.753 | 0.661 | +0.092 |
| fielded, headline-heavy (notice-style) | 0.700 | 0.785 | 0.575 | **+0.209** |
| **adaptive**, notice-style | 0.707 | 0.700 | 0.719 | **−0.019** |
| fielded, headline 60% | 0.641 | 0.806 | 0.399 | +0.407 |
| adaptive, headline 60% | 0.674 | 0.666 | 0.689 | −0.023 |

Headline-heavy weighting retrieves **21 points fewer** of the qualified experts with vague headlines than of those with clear ones (41 points fewer at 60% headline weight).
Adaptive weighting removes the gap and slightly raises overall recall versus the same weights without it (0.707 vs 0.700; 0.674 vs 0.641 at 60%). Honest caveat: the naive
blob baseline has the best raw recall; adaptive trades some of it for fairness among fielded approaches. The adaptive
percentile (35) was tuned on a separate seed (101) from the evaluation seeds (7, 11, 23).

**C. Rescue queue** (per-role averages)

| setting | surfaced recall | qualified rescued | queue size | queue precision |
|---|---|---|---|---|
| auto-reject | 0.600 | 0 | 0 | n/a |
| rescue, budget 30 | 0.637 | 12.6 | 28.7 | 0.43 |
| rescue, unlimited | 0.660 | 22.6 | 52.6 | 0.40 |

Routing low-confidence hard-filter failures to humans recovers qualified experts at a quantified review cost.

**E. Counterfactual audit** (qualified experts only). Changing a clear headline to a vague one moves the Stage-1 score by
−0.83 pool SD under headline-heavy weighting (31% of shortlisted experts fall out) versus −0.10 SD with adaptive weighting.
Institution prestige causes small but real shortlist flips at Stage 1 (≈5% of excluded qualified experts are pulled in when
`local → elite`) even though ground truth says it is irrelevant. Stage-2 effects are zero here only because the rule grader never sees institution; the harness
exists to test graders where that is not guaranteed (e.g. LLMs).

**F. Shadow audit.** Grading 160 non-shortlisted profiles per role (≈4% of the hidden pool) produced estimates whose 95% CI
contained the exhaustive-grading answer in 12/12 roles (mean absolute error 14.8 experts). It indicates Stage 1 hides
roughly 20-45% of grader-passing experts. Intervals are wide when losses are spread thinly; a larger sample tightens them.

**D. Ranking blend.** Always using the AI match score (w = 1) gave the best nDCG@20 (≈0.97); the pay-weighted blend gave ≈0.88.
Read this cautiously: ground truth here is defined by the same rubric the grader applies (see limitations).

Full tables: `results/results.md`; figures: `results/fig_*.png`.

## CV readiness check (candidate-facing)

The same pipeline, pointed the other way: a candidate uploads a job description and their CV, and the checker runs that
one CV through the reconstructed funnel against the synthetic pool, the way any pool member would be scored.

```
 JD (pdf/docx/txt) ──► parse_jd ──► Role: domain, target title, criteria (must-have vs nice-to-have), pay
 CV (pdf/docx/txt) ──► parse_cv ──► Profile: headline, dated roles, degrees, skills (in order), languages, location
                                        │  (both editable in the UI: the checker only sees what the parser read)
 Stage 1   where the CV ranks in the 4,000-expert pool, shortlist cutoff, per-field similarity vs the shortlist
 Stage 2   same grader, hard filters and 1.2 pass bar → per-criterion 0/1/2 with evidence
 Rank      position among the pool's passers (reviewers see the top 20 first)
 What-ifs  formatting-only changes (clear headline from your own title + skills, role skills first): no new claims
       ──► readiness 0-100 + verdict (Ready · Competitive · Hidden · At risk · Not ready) + prioritised fixes
```

The advice comes from the reverse-engineered mechanics: only the first 15 skills and the first 200 characters of each of
the 5 most recent roles are indexed; a skill scores 2 only when it is both listed and evidenced in work; a missing
language level reads as "not met" on a hard filter; non-standard job titles count only 50% toward years of experience.
On the bundled example, a vague headline keeps an otherwise passing CV off the shortlist ("Hidden", top 22% of the pool);
rewriting the headline from the CV's own title and skills moves it to the top 5%.

Readiness = 35% Stage 1 + 45% Stage 2 + 20% rank, capped at 25 for a confident must-have failure and 50 for a must-have
that rests on missing evidence. The percentile says how a CV *reads* to this kind of system, not how it compares with real
applicants. The parsers are rule-based and only know the taxonomy's seven domains; a JD outside them gets a warning.

## Mapping from the disclosure to code

| Described behaviour | Where |
|---|---|
| Stage-1 fields (headline, role, 5 experiences truncated to 200 chars, 3 education, 15 skills) | `retrieval.field_texts` |
| Headline weighted most, then role, experience, education, skills | `retrieval.NOTICE_WEIGHTS` (exact values are my assumption; only the ordering is public) |
| "Ideal candidate" query from role + criteria | `retrieval.ideal_candidate_query`, `llm.LLMQueryWriter` |
| Two-phase retrieval: recently active, then broader pool | `FieldIndex.search` |
| Stage-2 `CandidateInfo` (no raw resume) | `schema.CandidateInfo` |
| 0/1/2 per criterion, equal-weight average, hard filters | `grader.py`, `llm.py`, `Pipeline.decide` |
| Match-score influence rises with pay, bounded | `pipeline.rank_weight` |
| Humans make the final call, can override | `app.py`, `explain.record_override` |
| Human review, contest, add information | `explain.py` |
| Not replicated on purpose | biometrics / fraud scoring, prestige and conversion scores |

## Run it

```bash
pip install -r requirements.txt
pytest -q                              # 21 tests
python scripts/demo.py                 # 20-second tour incl. contesting an expert
python scripts/run_experiments.py      # all tables + figures (~5 min; --quick for ~1 min)
streamlit run app.py                   # project-lead dashboard
streamlit run cv_checker.py            # candidate CV readiness check (upload JD + CV)
python scripts/check_cv.py --cv examples/sample_cv.txt --jd examples/sample_jd.txt   # same, from the terminal
```

LLM grading (optional): `export ANTHROPIC_API_KEY=...`, then `Pipeline(P, LLMGrader(), ...)`. The model is set by
`TALENTLENS_MODEL`. The LLM grader is unit-tested against a fake client (parsing, caching, missing-criterion handling),
but **I have not run it against the live API**, so treat its accuracy and cost as untested.

## Limitations (read before citing any number)

- **Synthetic, and partly circular.** Ground truth is generated from latent facts using the same rubric the graders apply,
  which flatters Stage 2 and the "match score only" ranking result. Validity on real data needs external outcome labels
  (client ratings, retention).
- **Assumed effect sizes.** 40% of headlines are uninformative and prestige is independent of skill by construction.
  Directions are plausible; magnitudes are illustrative.
- **Retrieval is TF-IDF + SVD (LSA), not neural embeddings.** Swap `FieldIndex._dense` for sentence-transformers to test
  whether conclusions hold; absolute recall will change.
- **The rule grader is a baseline,** not a stand-in for an LLM; its ceiling is set by unlisted skills and typos.
- **Single synthetic population per seed;** only experiment A is replicated over seeds.
- **Rescue and shadow audit are sensitive to confidence calibration;** thresholds here are hand-set, not learned.

## Extension ideas

Run `LLMGrader` through the same harness and compare against the rule grader; calibrate confidences (isotonic) and learn
the rescue threshold; blend blob and adaptive scores; sentence-transformer embeddings; active-learning sampling for the
shadow audit; intersectional counterfactuals.

## Layout

```
talentlens/  schema · taxonomy · synth · retrieval · grader · llm · pipeline · evaluate · audit · explain · parse · cv_check
scripts/     run_experiments.py · demo.py · check_cv.py
tests/       test_talentlens.py · test_cv_check.py
examples/    sample_cv.txt · sample_jd.txt (fictional)
app.py       Streamlit review dashboard
cv_checker.py  Streamlit CV readiness check
results/     tables, figures, audit-log samples
```
