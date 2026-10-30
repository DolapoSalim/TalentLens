"""Reproduce every number and figure in the README.

    python scripts/run_experiments.py            # full run (~2-4 min)
    python scripts/run_experiments.py --quick    # smaller pool, ~30 s
"""
from __future__ import annotations
import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from talentlens.synth import generate_profiles, make_roles
from talentlens.grader import RuleGrader
from talentlens.pipeline import Pipeline, PipelineConfig
from talentlens.retrieval import RetrievalConfig, NOTICE_WEIGHTS, BALANCED_WEIGHTS
from talentlens.evaluate import build_truth, funnel, average
from talentlens.audit import counterfactual_audit, shadow_audit, TRANSFORMS

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
md_parts: list[str] = []


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(rows)


def emit(title: str, df: pd.DataFrame, note: str = "", name: str = "", fmt="{:.3f}"):
    md_parts.append(f"### {title}\n\n{note}\n\n{md_table(df, fmt)}\n")
    if name:
        df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"\n== {title} ==\n{df.to_string(index=False)}")


def heavy(w_head: float) -> dict:
    rest = {k: v for k, v in NOTICE_WEIGHTS.items() if k != "headline"}
    s = sum(rest.values())
    return {"headline": w_head, **{k: v / s * (1 - w_head) for k, v in rest.items()}}


def kcfg(n: int, **kw) -> RetrievalConfig:
    """Shortlist size scales with pool size so the funnel ratio (~10% of pool) is constant."""
    return RetrievalConfig(k_active=int(150 * n / 4000), k_broad=int(250 * n / 4000), **kw)


def retrieval_configs(n: int) -> dict:
    return {
        "blob (naive)": kcfg(n, mode="blob"),
        "fielded, balanced": kcfg(n, mode="fielded", weights=BALANCED_WEIGHTS),
        "fielded, notice-style": kcfg(n, mode="fielded"),
        "adaptive, notice-style": kcfg(n, mode="adaptive"),
        "fielded, headline 60%": kcfg(n, mode="fielded", weights=heavy(0.6)),
        "adaptive, headline 60%": kcfg(n, mode="adaptive", weights=heavy(0.6)),
    }


def run_all(pipe: Pipeline, roles, truth, by) -> pd.DataFrame:
    return pd.DataFrame([funnel(pipe.run(r), truth[r.id], by) for r in roles])


# ------------------------------------------------------------------ A. retrieval ablation (3 seeds)
def exp_retrieval(n: int, seeds=(7, 11, 23)):
    rows = []
    for seed in seeds:
        P = generate_profiles(n, seed)
        R = make_roles()
        by = {p.id: p for p in P}
        truth = build_truth(P, R)
        grader = RuleGrader(fuzzy=True)
        for name, rc in retrieval_configs(n).items():
            pipe = Pipeline(P, grader, PipelineConfig(retrieval=rc))
            a = average([funnel(pipe.run(r), truth[r.id], by) for r in R])
            rows.append(dict(config=name, seed=seed, recall=a["stage1_recall"], precision=a["stage1_precision"],
                             recall_informative=a["stage1_recall_informative"],
                             recall_uninformative=a["stage1_recall_uninformative"]))
    df = pd.DataFrame(rows)
    df["headline_gap"] = df.recall_informative - df.recall_uninformative
    g = df.groupby("config", sort=False)
    out = pd.DataFrame({
        "config": list(g.groups.keys()),
        "recall": [f"{g.recall.mean()[k]:.3f} ± {g.recall.std()[k]:.3f}" for k in g.groups],
        "recall (clear headline)": [f"{g.recall_informative.mean()[k]:.3f}" for k in g.groups],
        "recall (vague headline)": [f"{g.recall_uninformative.mean()[k]:.3f}" for k in g.groups],
        "gap": [f"{g.headline_gap.mean()[k]:+.3f} ± {g.headline_gap.std()[k]:.3f}" for k in g.groups],
    })
    emit("A. Stage-1 retrieval ablation", out,
         f"Mean over 12 roles and {len(seeds)} data seeds (pool = {n}). 'gap' = recall of qualified experts with a clear "
         "headline minus recall of equally qualified experts with a vague/buzzword headline (0 is fair).", "A_retrieval")
    mean = df.groupby("config", sort=False)[["recall_informative", "recall_uninformative"]].mean()
    ax = mean.plot.bar(figsize=(8, 4), rot=25, color=["#3b7ddd", "#e07b39"])
    ax.set_ylabel("Stage-1 recall of qualified experts"); ax.set_ylim(0.3, 0.9)
    ax.legend(["clear headline", "vague headline"]); ax.set_title("Headline dependence of Stage-1 retrieval")
    plt.tight_layout(); plt.savefig(OUT / "fig_headline_gap.png", dpi=150); plt.close()


# ------------------------------------------------------------------ B-D, E, F on seed 7
def exp_main(n: int):
    P = generate_profiles(n, 7)
    R = make_roles()
    by = {p.id: p for p in P}
    truth = build_truth(P, R)

    # B. grader comparison
    rows = []
    for gname, grader in {"rule": RuleGrader(), "rule + typo tolerance": RuleGrader(fuzzy=True)}.items():
        pipe = Pipeline(P, grader, PipelineConfig(retrieval=kcfg(n)))
        a = average([funnel(pipe.run(r), truth[r.id], by) for r in R])
        rows.append(dict(grader=gname, surfaced_recall=a["surfaced_recall"], precision_at_20=a["precision_at_k"],
                         ndcg_at_20=a["ndcg_at_k"], qualified_lost_at_threshold=a["lost_threshold"]))
    emit("B. Stage-2 grader comparison", pd.DataFrame(rows),
         "Same Stage-1 retrieval. An LLM grader can be dropped in via `talentlens.llm.LLMGrader` and scored with the same harness.", "B_graders")

    grader = RuleGrader(fuzzy=True)
    base = PipelineConfig(retrieval=kcfg(n))
    pipe = Pipeline(P, grader, base)

    # C. rescue queue
    rows = []
    for label, kw in {"off (auto-reject)": dict(rescue_enabled=False), "on, budget 10": dict(rescue_max=10),
                      "on, budget 30": dict(rescue_max=30), "on, budget 60": dict(rescue_max=60),
                      "on, unlimited": dict(rescue_max=None)}.items():
        pipe.cfg = dataclasses.replace(base, **kw)
        a = average([funnel(pipe.run(r), truth[r.id], by) for r in R])
        rows.append(dict(setting=label, surfaced_recall=a["surfaced_recall"], qualified_rescued=a["rescued"],
                         queue_size=a["rescue_queue"], queue_precision=a["rescue_precision"],
                         qualified_wrongly_rejected=a["hard_reject_false"]))
    emit("C. Appeal-aware hard filters (rescue queue)", pd.DataFrame(rows),
         "Per-role averages. 'queue_size' is human review workload; 'qualified_rescued' are truly qualified experts that "
         "would otherwise have been silently dropped by a hard filter.", "C_rescue")

    # D. pay-weighted ranking
    rows = []
    for label, kw in {"retrieval only (w=0)": dict(fixed_w=0.0), "balanced (w=0.5)": dict(fixed_w=0.5),
                      "AI match only (w=1)": dict(fixed_w=1.0), "pay-weighted (0.3-0.7)": dict()}.items():
        pipe.cfg = dataclasses.replace(base, **kw)
        for tier, sel in {"pay >= $100": [r for r in R if r.pay_rate >= 100], "pay < $100": [r for r in R if r.pay_rate < 100]}.items():
            a = average([funnel(pipe.run(r), truth[r.id], by) for r in sel])
            rows.append(dict(ranking=label, tier=tier, precision_at_20=a["precision_at_k"], ndcg_at_20=a["ndcg_at_k"]))
    emit("D. Final-ranking blend", pd.DataFrame(rows), "Top-20 quality of the ranking shown to project leads.", "D_ranking")

    # funnel figure (default config)
    pipe.cfg = base
    f = average([funnel(pipe.run(r), truth[r.id], by) for r in R])
    parts = {"surfaced to a human": f["qualified"] * f["surfaced_recall"], "lost: Stage 1 shortlist": f["lost_stage1"],
             "lost: hard-filter reject": f["lost_hard_filter"], "lost: below threshold": f["lost_threshold"]}
    plt.figure(figsize=(6, 3.2))
    plt.barh(list(parts)[::-1], list(parts.values())[::-1], color=["#c0504d", "#e0a339", "#e07b39", "#3b7ddd"])
    plt.xlabel("qualified experts per role (avg)"); plt.title("Where qualified experts are lost")
    plt.tight_layout(); plt.savefig(OUT / "fig_funnel.png", dpi=150); plt.close()

    # E. counterfactual bias audit, notice-style vs adaptive
    rows = []
    for mode in ["fielded", "adaptive"]:
        p2 = Pipeline(P, grader, PipelineConfig(retrieval=kcfg(n, mode=mode)))
        for name in TRANSFORMS:
            res = [counterfactual_audit(p2, r, truth[r.id]["qualified"], name) for r in R]
            res = [x for x in res if x]
            if not res:
                continue
            w = np.array([x["n"] for x in res], float)
            wm = lambda k: float(np.average([x[k] for x in res], weights=w))
            rows.append(dict(retrieval=mode, change=name, n=int(w.sum()),
                             d_score_sd=float(np.average([x["d_retrieval_z"][0] for x in res], weights=w)),
                             shortlist_drop=wm("shortlist_drop_rate"), shortlist_gain=wm("shortlist_gain_rate"),
                             d_match=wm("d_match"), pass_change=wm("pass_change_rate")))
    dfE = pd.DataFrame(rows)
    emit("E. Counterfactual audit (qualified experts only)", dfE,
         "One attribute changed, everything else identical. d_score_sd = change in Stage-1 score in pool standard deviations. "
         "shortlist_drop = share of shortlisted experts pushed OUT by the change; shortlist_gain = share of excluded experts pulled IN. "
         "Ground truth says none of these attributes should matter.", "E_counterfactual")
    sub = dfE[dfE.change.str.startswith("headline")]
    ax = sub.pivot(index="change", columns="retrieval", values="d_score_sd").plot.bar(figsize=(6, 3.5), rot=0, color=["#e07b39", "#3b7ddd"])
    ax.set_ylabel("Stage-1 score change (pool SD)"); ax.set_title("Headline edits move Stage-1 scores")
    plt.tight_layout(); plt.savefig(OUT / "fig_counterfactual_headline.png", dpi=150); plt.close()

    # F. shadow audit
    pipe.cfg = base
    rows = []
    for r in R:
        res = pipe.run(r)
        s = shadow_audit(pipe, r, res, per_bucket=40, truth_role=truth[r.id], validate=True)
        rows.append(dict(role=r.id, found=s["found_pass"], est_missed=round(s["est_missed"]), ci=f"[{s['est_missed_lo']:.0f}, {s['est_missed_hi']:.0f}]",
                         exact_missed=s["exact_grader_missed"], truth_missed=s["truth_missed"],
                         est_recall=s["est_stage1_recall"], recommended_extra_k=s["recommended_extra_k"] if s["recommended_extra_k"] is not None else "all"))
        rows[-1]["_in_ci"] = s["est_missed_lo"] <= s["exact_grader_missed"] <= s["est_missed_hi"]
        rows[-1]["_lo"], rows[-1]["_hi"] = s["est_missed_lo"], s["est_missed_hi"]
    dfF = pd.DataFrame(rows)
    rc = kcfg(n)
    cover = dfF["_in_ci"].mean()
    err = (dfF.est_missed - dfF.exact_missed).abs().mean()
    emit("F. Shadow audit of the non-shortlisted pool (40 graded per stratum, 160 per role)", dfF.drop(columns=["_in_ci", "_lo", "_hi"]),
         f"Audit estimate vs. exhaustive grading of everyone outside the shortlist. CI coverage of the exact value: {cover:.0%} "
         f"over {len(dfF)} roles; mean absolute error {err:.1f} experts (grading 160 profiles instead of ~{n - rc.k_active - rc.k_broad}).", "F_shadow", fmt="{:.2f}")
    plt.figure(figsize=(4.5, 4.5))
    plt.errorbar(dfF.exact_missed, dfF.est_missed, yerr=[dfF.est_missed - dfF._lo, dfF._hi - dfF.est_missed], fmt="o", capsize=3)
    m = max(dfF.exact_missed.max(), dfF._hi.max()) * 1.05
    plt.plot([0, m], [0, m], "k--", lw=1); plt.xlabel("exact (grade everyone)"); plt.ylabel("shadow-audit estimate")
    plt.title("Estimating hidden Stage-1 losses"); plt.tight_layout(); plt.savefig(OUT / "fig_shadow_audit.png", dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n = 1500 if a.quick else 4000
    exp_retrieval(n, seeds=(7,) if a.quick else (7, 11, 23))
    exp_main(n)
    (OUT / "results.md").write_text("# Experiment results\n\n" + "\n".join(md_parts), encoding="utf-8")
    print(f"\nWrote {OUT}/results.md and figures")
