# TalentLens

A rebuild of the kind of two-stage AI hiring pipeline that expert marketplaces describe publicly:
**Stage 1** finds a shortlist by text similarity, **Stage 2** grades each shortlisted person against the role's criteria.
TalentLens asks who that pipeline quietly misses, and gives candidates a tool to check their own CV against it.

> Independent project, rebuilt from a public *description* of the process, not from any company's code or data.
> All data is synthetic. No real people are used.

## What's inside

### 1. The pipeline and its audits (`streamlit run app.py`)

- Stage 1 retrieval and Stage 2 grading, with a human review dashboard.
- Audits that measure who gets lost: experts with vague headlines, experts never shortlisted, and bias from things that
  should not matter (university prestige, location).

### 2. CV readiness check (`streamlit run cv_checker.py`)

- Upload a job description and your CV (PDF, Word or text).
- Get a readiness score out of 100, a verdict, and a ranked list of fixes.

## Quick start

```bash
pip install -r requirements.txt
streamlit run cv_checker.py        # check a CV
streamlit run app.py               # project-lead dashboard
pytest -q                          # run the tests
```

From the terminal:

```bash
python scripts/check_cv.py --cv examples/sample_cv.txt --jd examples/sample_jd.txt
```

## How the CV score works

Readiness = **60% Requirements + 40% Visibility**

- **Requirements:** does the CV show what the job asks for? Each requirement is graded 0, 1 or 2. A skill only gets
  full marks when it is listed *and* shown in your work. Failing a must-have caps the score.
- **Visibility:** will the search stage find you? Stage 1 mainly reads your headline, current job title, the first
  200 characters of your 5 most recent roles, and your first 15 skills. The checklist scores how well those parts
  match the job.

Verdicts: **Ready**, **Nearly ready**, **Hard to find**, **At risk**, **Not ready**.

The score does not compare you with other people. An optional tab ranks your CV against a synthetic pool of 4,000
experts, to show how much the fixes are worth.

**Example:** the sample CV meets the requirements but its headline is a slogan ("Passionate problem solver"), so it
scores 61/100, *Hard to find*. Rewriting the headline from the CV's own job title and skills lifts it to 73/100.

## Main findings (synthetic data)

- **Headlines matter too much.** With headline-heavy search, qualified experts with vague headlines are found 21 points
  less often than equally qualified experts with clear ones. TalentLens' adaptive weighting closes that gap.
- **Hard filters drop people on missing information.** Sending "not stated" failures to a human, instead of rejecting
  them, recovers about 13 qualified experts per role.
- **Stage 1 hides 20-45% of people who would pass Stage 2.** A small sample of the hidden pool estimates this reliably.

Full tables and figures: `results/`. Reproduce with `python scripts/run_experiments.py`.

## Limitations

- All data is synthetic, so the numbers show direction, not real-world size.
- Search uses simple text matching (TF-IDF), not neural embeddings.
- The CV and job parsers are rule-based and know seven fields: data science, backend, frontend, finance, legal,
  clinical and marketing.
- The optional AI (LLM) grader is tested with a fake client only, not the live API.

## Project layout

```
talentlens/     the pipeline: retrieval, grading, audits, CV parsing and checking
scripts/        demo, experiments, CV check from the terminal
tests/          tests
examples/       sample CV and job description (fictional)
results/        tables and figures
app.py          project-lead dashboard
cv_checker.py   CV readiness check
```
