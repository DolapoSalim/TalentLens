"""Candidate-facing CV readiness check.   streamlit run cv_checker.py"""
import dataclasses
import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from talentlens.schema import Experience, Education
from talentlens.synth import make_roles
from talentlens.taxonomy import DOMAINS, DEGREE_RANK, LEVEL_ORDER
from talentlens.parse import parse_cv, parse_jd, read_document, make_criterion, criterion_value, KINDS
from talentlens.cv_check import CVChecker, CheckConfig, READINESS_WEIGHTS, to_text

st.set_page_config(page_title="TalentLens CV check", layout="wide")
st.title("TalentLens: CV readiness check")
st.caption("Runs your CV through a reconstruction of a two-stage AI matching pipeline (retrieval, then rubric grading) "
           "against a synthetic pool of 4,000 experts, and shows where it would be filtered out and what to change. "
           "It is not any platform's real score. Your CV is processed in memory and is not stored or logged.")

with st.sidebar:
    mode = st.radio("Stage-1 retrieval to test against", ["fielded", "adaptive"],
                    format_func={"fielded": "Platform as disclosed (headline-heavy)",
                                 "adaptive": "TalentLens adaptive (headline counts less when vague)"}.get)


@st.cache_resource(show_spinner="Building the benchmark pool (first run only, ~15 s)...")
def checker(mode):
    return CVChecker(CheckConfig(mode=mode))


def document(label, key):
    up = st.file_uploader(f"Upload {label}", type=["pdf", "docx", "txt", "md"], key=key + "_file")
    pasted = st.text_area(f"...or paste {label}", key=key + "_text", height=200)
    if up is None:
        return pasted
    try:
        return read_document(up.name, up.getvalue())
    except ImportError as e:
        st.error(str(e))
        return pasted


def tag(text):
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def txt(v):
    return "" if pd.isna(v) else str(v)


# ------------------------------------------------------------------ 1. inputs
st.header("1. Job and CV")
left, right = st.columns(2)
with left:
    st.subheader("The job")
    source = st.radio("Source", ["Job description", "Sample role"], horizontal=True, label_visibility="collapsed")
    if source == "Sample role":
        sample = st.selectbox("Sample role", make_roles(), format_func=lambda r: f"{r.id} - {r.title} (${r.pay_rate:.0f}/h)")
        role, jd_notes, jd_key = sample, [], sample.id
    else:
        jd_text = document("the job description", "jd")
        role, jd_notes = parse_jd(jd_text) if jd_text.strip() else (None, [])
        jd_key = tag(jd_text)
with right:
    st.subheader("Your CV")
    cv_text = document("your CV", "cv")

if role is None or not cv_text.strip():
    st.info("Add a job description (or pick a sample role) and your CV to start.")
    st.stop()
profile, cv_notes = parse_cv(cv_text)
cv_key = tag(cv_text)

# ------------------------------------------------------------------ 2. review what the parser read
st.header("2. Check what was read")
st.caption("The checker only sees what the parser extracted. Fix anything it got wrong; the results update as you edit.")
for n in jd_notes + cv_notes:
    st.warning(n)

jcol, ccol = st.columns(2)
with jcol, st.expander("Job requirements", expanded=True):
    a, b, c = st.columns([2, 2, 1])
    domain = a.selectbox("Field", list(DOMAINS), index=list(DOMAINS).index(role.domain), key=f"dom_{jd_key}")
    titles = DOMAINS[domain]["titles"]
    target = b.selectbox("Closest standard title", titles, key=f"tt_{jd_key}",
                         index=titles.index(role.target_title) if role.target_title in titles else 0)
    pay = c.number_input("Rate $/h", 5.0, 1000.0, float(role.pay_rate), key=f"pay_{jd_key}")
    crit_df = st.data_editor(
        pd.DataFrame([dict(kind=x.kind, value=criterion_value(x), must_have=x.hard) for x in role.criteria],
                     columns=["kind", "value", "must_have"]),
        num_rows="dynamic", width="stretch", hide_index=True, key=f"crit_{jd_key}",
        column_config={"kind": st.column_config.SelectboxColumn(options=KINDS, required=True),
                       "value": st.column_config.TextColumn(help="skill name · years (number) · degree (BSc/MSc/PhD or 1-3) · "
                                                                 "cert name · language as English:B2 · country (EU or names)"),
                       "must_have": st.column_config.CheckboxColumn(default=False)})
    criteria, bad = [], []
    for r in crit_df.dropna(subset=["kind"]).itertuples():
        try:
            criteria.append(make_criterion(r.kind, txt(r.value), bool(r.must_have), domain))
        except ValueError as e:
            bad.append(f"{r.kind} '{r.value}': {e}")
    for msg in bad:
        st.error(msg)
    role = dataclasses.replace(role, domain=domain, target_title=target, pay_rate=pay, criteria=criteria)

with ccol, st.expander("Your CV", expanded=True):
    headline = st.text_input("Headline (the line under your name)", profile.headline, key=f"hl_{cv_key}")
    a, b = st.columns(2)
    city = a.text_input("City", profile.city, key=f"city_{cv_key}")
    country = b.text_input("Country", profile.country, key=f"ctry_{cv_key}")
    exp_df = st.data_editor(
        pd.DataFrame([dataclasses.asdict(e) for e in profile.experiences],
                     columns=["title", "company", "description", "start_year", "end_year"]),
        num_rows="dynamic", width="stretch", hide_index=True, key=f"exp_{cv_key}")
    edu_df = st.data_editor(
        pd.DataFrame([dataclasses.asdict(e) for e in profile.education], columns=["degree", "field", "institution"]),
        num_rows="dynamic", width="stretch", hide_index=True, key=f"edu_{cv_key}",
        column_config={"degree": st.column_config.SelectboxColumn(options=list(DEGREE_RANK), required=True)})
    lang_df = st.data_editor(
        pd.DataFrame([dict(language=k, level=v) for k, v in profile.languages.items()], columns=["language", "level"]),
        num_rows="dynamic", width="stretch", hide_index=True, key=f"lang_{cv_key}",
        column_config={"level": st.column_config.SelectboxColumn(options=LEVEL_ORDER, required=True)})
    skills = st.text_area("Skills, in the order they appear (one per line)", "\n".join(profile.skills), key=f"sk_{cv_key}")
    certs = st.text_area("Certifications (one per line)", "\n".join(profile.certifications), key=f"cert_{cv_key}")

    exps = []
    for r in exp_df.dropna(subset=["start_year"]).itertuples():
        start = int(r.start_year)
        end = start if pd.isna(r.end_year) else int(r.end_year)
        exps.append(Experience(txt(r.title), txt(r.company), txt(r.description), start, end))
    exps.sort(key=lambda e: (e.end_year, e.start_year), reverse=True)
    edus = [Education(r.degree, txt(r.field), txt(r.institution)) for r in edu_df.dropna(subset=["degree"]).itertuples()]
    edus.sort(key=lambda e: -DEGREE_RANK.get(e.degree, 0))
    profile = dataclasses.replace(
        profile, headline=headline, city=city, country=country, experiences=exps, education=edus,
        current_title=exps[0].title if exps else headline, current_company=exps[0].company if exps else "",
        languages={r.language: r.level for r in lang_df.dropna().itertuples()},
        skills=[s.strip() for s in skills.splitlines() if s.strip()],
        certifications=[s.strip() for s in certs.splitlines() if s.strip()])

if not role.criteria:
    st.error("No job requirements to check against. Add at least one row in the requirements table.")
    st.stop()

# ------------------------------------------------------------------ 3. results
rep = checker(mode).check(profile, role)
s1, s2, rk = rep["stage1"], rep["stage2"], rep["rank"]
st.header(f"3. Result: {rep['verdict']}")
m = st.columns(4)
m[0].metric("Readiness", f"{rep['readiness']}/100")
m[1].metric("Stage 1: retrieval", f"top {100 * (1 - s1['percentile']):.0f}%",
            "shortlisted" if s1["shortlisted"] else f"cutoff is top {100 * (1 - s1['cutoff_percentile']):.0f}%",
            delta_color="normal" if s1["shortlisted"] else "inverse")
m[2].metric("Stage 2: AI grading", f"{s2['match']:.2f} / 2", f"{s2['match'] - s2['threshold']:+.2f} vs pass bar {s2['threshold']}")
m[3].metric("Final rank", f"#{rk['rank']} of {rk['passers']}" if rk else "not ranked",
            ("if shortlisted" if rk["hypothetical"] else f"reviewers see top {rk['review_size']}") if rk else s2["outcome"].replace("_", " "),
            delta_color="off")
st.write(rep["verdict_detail"])

st.subheader("What to change, most important first")
if rep["recommendations"]:
    st.dataframe(pd.DataFrame(rep["recommendations"]), width="stretch", hide_index=True)
else:
    st.success("Nothing to change: every criterion is fully evidenced and the CV clears Stage 1.")

t1, t2, t3, t4 = st.tabs(["Stage 1 detail", "Stage 2 detail", "Presentation what-ifs", "How the score works"])
with t1:
    st.caption("Similarity of each CV field to the role's 'ideal candidate' query, next to the median of the shortlisted pool. "
               f"Headline consistency with the rest of your CV: {s1['headline_consistency']:.2f}.")
    st.dataframe(pd.DataFrame(s1["fields"]).round(3), width="stretch", hide_index=True)
with t2:
    st.caption("0 = not met, 1 = partly (listed OR used, not both), 2 = fully. Low confidence means the score rests on missing evidence.")
    st.dataframe(pd.DataFrame(s2["grades"]), width="stretch", hide_index=True)
with t3:
    st.caption("Formatting-only changes that reuse your own title and skills. Nothing new is claimed.")
    if rep["what_ifs"]:
        wi = pd.DataFrame(rep["what_ifs"])
        wi["position"] = wi.percentile.map(lambda p: f"top {100 * (1 - p):.0f}%")
        st.dataframe(wi[["change", "detail", "position", "shortlisted", "delta"]].round(4), width="stretch", hide_index=True)
    else:
        st.write("No formatting changes to try: your headline and skills order already match the role.")
with t4:
    w = READINESS_WEIGHTS
    st.markdown(f"""
Readiness = {w['stage1']:.0%} Stage 1 + {w['stage2']:.0%} Stage 2 + {w['rank']:.0%} rank, then capped by the outcome.

- **Stage 1** is 1 when you clear the shortlist cutoff, otherwise your pool percentile divided by the cutoff percentile (at most 0.95).
- **Stage 2** is your average criterion score divided by 2.
- **Rank** is 1 inside the top {rk['review_size'] if rk else 20} passers reviewers see first, otherwise 20 / your rank (0 if not shortlisted).
- A confident must-have failure caps readiness at 25; a must-have that rests on missing evidence caps it at 50.

The pool is synthetic, so the percentile tells you how your CV *reads* to this kind of system, not how you compare
with real applicants. Retrieval mode: **{rep['benchmark']['mode']}**; pool {rep['benchmark']['pool']:,}; shortlist {rep['benchmark']['shortlist']}.
""")

st.download_button("Download report (.txt)", to_text(rep), file_name="cv_readiness_report.txt")
