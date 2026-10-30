"""Project-lead review dashboard.   streamlit run app.py"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from talentlens.synth import generate_profiles, make_roles, true_summary
from talentlens.grader import RuleGrader
from talentlens.pipeline import Pipeline, PipelineConfig
from talentlens.retrieval import RetrievalConfig
from talentlens.evaluate import build_truth, funnel
from talentlens.explain import explain_decision, to_text, request_review, record_override
from talentlens.audit import shadow_audit

st.set_page_config(page_title="TalentLens", layout="wide")
st.title("TalentLens: auditable expert matching")

with st.sidebar:
    n = st.slider("Synthetic pool size", 1000, 6000, 3000, 500)
    mode = st.selectbox("Stage-1 retrieval", ["adaptive", "fielded", "blob"])
    rescue = st.checkbox("Appeal-aware hard filters (rescue queue)", True)
    budget = st.slider("Human review budget (rescue queue)", 5, 100, 30)
    grader_name = st.selectbox("Stage-2 grader", ["rule + typo tolerance", "rule"] + (["LLM (Anthropic)"] if os.getenv("ANTHROPIC_API_KEY") else []))
    roles = make_roles()
    role = st.selectbox("Role", roles, format_func=lambda r: f"{r.id} - {r.title} (${r.pay_rate:.0f}/h)")


@st.cache_resource(show_spinner="Building index...")
def build(n, mode, rescue, budget, grader_name):
    P = generate_profiles(n, 7)
    if grader_name.startswith("LLM"):
        from talentlens.llm import LLMGrader
        grader = LLMGrader()
    else:
        grader = RuleGrader(fuzzy="typo" in grader_name)
    k = (int(0.0375 * n), int(0.0625 * n))
    cfg = PipelineConfig(retrieval=RetrievalConfig(mode=mode, k_active=k[0], k_broad=k[1]), rescue_enabled=rescue, rescue_max=budget)
    log = Path(__file__).parent / "results" / "dashboard_audit_log.jsonl"
    return P, Pipeline(P, grader, cfg, log_path=log), build_truth(P, roles)


P, pipe, truth = build(n, mode, rescue, budget, grader_name)
by = {p.id: p for p in P}
res = pipe.run(role)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pool", len(P)); c2.metric("Shortlisted (Stage 1)", len(res.records))
c3.metric("Passed both stages", len(res.ranked)); c4.metric("Rescue queue", len(res.rescue))
st.caption(f"AI-match weight in final rank for this pay rate: {res.records[0].rank_weight:.2f} (bounded 0.30-0.70)")

tab1, tab2, tab3, tab4 = st.tabs(["Shortlist", "Rescue queue", "Contest an expert", "Audits"])


def table(records):
    rows = []
    for r in records:
        p = by[r.profile_id]
        row = {"id": r.profile_id, "title": p.current_title, "match": round(r.match_score, 2),
               "final": None if r.final_score is None else round(r.final_score, 3), "stage1": round(r.retrieval_score, 3)}
        row.update({g["criterion_id"].split(":", 1)[1]: g["score"] for g in r.grades})
        rows.append(row)
    return pd.DataFrame(rows)


def detail(records, key):
    if not records:
        st.info("Nothing here."); return
    pid = st.selectbox("Inspect expert", [r.profile_id for r in records], key=key)
    rec = next(r for r in records if r.profile_id == pid)
    st.dataframe(pd.DataFrame(rec.grades), width="stretch", hide_index=True)
    a, b, c = st.columns([1, 1, 3])
    reason = c.text_input("Reason (logged)", key=key + "r")
    if a.button("Approve", key=key + "a"):
        record_override(pipe, role, pid, "project_lead", "approve", reason); st.success("Override logged")
    if b.button("Reject", key=key + "b"):
        record_override(pipe, role, pid, "project_lead", "reject", reason); st.success("Override logged")


with tab1:
    st.dataframe(table(res.ranked[:50]), width="stretch", hide_index=True)
    detail(res.ranked[:50], "s")
with tab2:
    st.write("Hard filter failed on *thin evidence* (absence of evidence, not proof of absence). A human decides before any rejection.")
    st.dataframe(table(res.rescue), width="stretch", hide_index=True)
    detail(res.rescue, "r")
with tab3:
    pid = st.text_input("Expert id (any id in the pool, shortlisted or not)", "P00003")
    if pid in by:
        e = explain_decision(pipe, role, pid, res)
        st.code(to_text(e))
        st.json(e["field_breakdown"], expanded=False)
        stmt = st.text_area("Expert statement / extra information")
        if st.button("Request human review"):
            request_review(pipe, role, pid, stmt); st.success("Review request logged")
    else:
        st.warning("Unknown id")
with tab4:
    f = funnel(res, truth[role.id], by)
    st.subheader("Funnel vs ground truth (synthetic data only)")
    st.json({k: round(v, 3) if isinstance(v, float) else v for k, v in f.items()}, expanded=False)
    if st.button("Run shadow audit (grades ~160 non-shortlisted experts)"):
        s = shadow_audit(pipe, role, res, per_bucket=40, truth_role=truth[role.id])
        st.write(f"Estimated passing experts never shown to a human: **{s['est_missed']:.0f}** "
                 f"(95% CI {s['est_missed_lo']:.0f}-{s['est_missed_hi']:.0f}); estimated Stage-1 recall **{s['est_stage1_recall']:.0%}**.")
        st.dataframe(pd.DataFrame(s["buckets"]), width="stretch", hide_index=True)
