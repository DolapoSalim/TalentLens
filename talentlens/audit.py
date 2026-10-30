"""Built-in audits. These are the project's main departure from the notice, which describes the
funnel but offers no way to measure what it silently loses or whom it treats differently.

1. counterfactual_audit  -- change ONE attribute that should be irrelevant to skill (institution prestige,
                            country, headline style) on otherwise identical qualified experts and measure
                            what happens at Stage 1 (score, shortlist membership) and Stage 2 (match, outcome).
2. shadow_audit          -- Stage 1 hides everyone outside the shortlist from every human. Grade a small
                            stratified sample of the *non-shortlisted* pool to estimate how many qualified
                            experts were dropped, with confidence intervals, and recommend a larger k.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from .schema import Profile, Role
from .retrieval import ideal_candidate_query
from .synth import ELITE, LOCAL, VAGUE, INSTITUTION_TIER
from .taxonomy import EU_COUNTRIES

PASS = {"passed", "rescue_review"}


# ---------------------------------------------------------------- transformations
def _inst(direction: str):
    src, dst = ("elite", LOCAL) if direction == "elite_to_local" else ("local", ELITE)

    def applies(p): return INSTITUTION_TIER.get(p.education[0].institution) == src

    def apply(p):
        q = copy.deepcopy(p)
        for k, e in enumerate(q.education):
            if INSTITUTION_TIER.get(e.institution) == src:
                e.institution = dst[k % len(dst)]
        return q
    return applies, apply


def _country(direction: str):
    eu = direction == "eu_to_non_eu"
    city, country = ("Lagos", "Nigeria") if eu else ("Berlin", "Germany")

    def applies(p): return (p.country in EU_COUNTRIES) == eu

    def apply(p):
        q = copy.deepcopy(p); q.city, q.country = city, country
        return q
    return applies, apply


def _headline(direction: str):
    to_vague = direction == "informative_to_vague"

    def applies(p): return (p.hidden["headline_quality"] == "informative") == to_vague

    def apply(p):
        q = copy.deepcopy(p)
        q.headline = VAGUE[int(p.id[1:]) % len(VAGUE)] if to_vague else f"{p.current_title} | {' | '.join(p.skills[:2])}"
        return q
    return applies, apply


TRANSFORMS = {
    "institution: elite -> local": _inst("elite_to_local"),
    "institution: local -> elite": _inst("local_to_elite"),
    "location: EU -> non-EU": _country("eu_to_non_eu"),
    "location: non-EU -> EU": _country("non_eu_to_eu"),
    "headline: informative -> vague": _headline("informative_to_vague"),
    "headline: vague -> informative": _headline("vague_to_informative"),
}


def boot_ci(x: np.ndarray, n: int = 400, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- counterfactual audit
def counterfactual_audit(pipe, role: Role, qualified_ids: set[str], name: str, query=None) -> dict | None:
    applies, apply = TRANSFORMS[name]
    if "location" in name and any(c.kind == "country" for c in role.criteria):
        return None  # location is a legitimate criterion for this role; not a bias test
    q = query or ideal_candidate_query(role)
    ret = pipe.index.search(q)
    pool_sd = float(ret.all_scores.std()) or 1.0
    base = [pipe.by_id[i] for i in qualified_ids if applies(pipe.by_id[i])]
    if len(base) < 5:
        return None
    cf = [apply(p) for p in base]
    s0, s1 = pipe.index.score_variants(base, q), pipe.index.score_variants(cf, q)
    act = np.array([p.last_active_days <= pipe.cfg.retrieval.active_days for p in base])
    in0 = np.array([pipe.index.would_shortlist(s, a, ret.cutoff) for s, a in zip(s0, act)])
    in1 = np.array([pipe.index.would_shortlist(s, a, ret.cutoff) for s, a in zip(s1, act)])
    # Stage 2 (bypass the grader cache by giving counterfactuals their own ids)
    from .schema import CandidateInfo
    m0, m1, o0, o1 = [], [], [], []
    for p, c in zip(base, cf):
        g0 = pipe.grader.grade(p.id, CandidateInfo.from_profile(p), role)
        g1 = pipe.grader.grade(f"{p.id}#cf:{name}", CandidateInfo.from_profile(c), role)
        a, ma, _ = pipe.decide(g0, role); b, mb, _ = pipe.decide(g1, role)
        m0.append(ma); m1.append(mb); o0.append(a in PASS); o1.append(b in PASS)
    d_z = (s1 - s0) / pool_sd
    drop = (in0 & ~in1).sum() / max(1, in0.sum())
    gain = (~in0 & in1).sum() / max(1, (~in0).sum())
    return dict(transform=name, role=role.id, n=len(base), d_retrieval_z=boot_ci(d_z),
                shortlist_drop_rate=float(drop), shortlist_gain_rate=float(gain),
                d_match=float(np.mean(np.array(m1) - np.array(m0))),
                pass_change_rate=float(np.mean(np.array(o0) != np.array(o1))))


# ---------------------------------------------------------------- shadow audit
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def shadow_audit(pipe, role: Role, res, per_bucket: int = 30, seed: int = 0, target_recall: float = 0.95,
                 truth_role: dict | None = None, validate: bool = False) -> dict:
    """Estimate how many grader-qualified experts Stage 1 dropped, using a stratified sample of the rest."""
    rng = np.random.default_rng(seed)
    scores = res.retrieved.all_scores
    in_short = np.zeros(len(pipe.profiles), bool)
    in_short[res.retrieved.idx] = True
    rest = np.where(~in_short)[0]
    rest = rest[np.argsort(-scores[rest])]
    M = len(rest)
    edges = [0, int(.1 * M), int(.3 * M), int(.6 * M), M]

    def passes(i):
        g = pipe.grade_profile(pipe.profiles[i], role)
        return pipe.decide(g, role)[0] in PASS

    found = sum(r.outcome in PASS for r in res.records)
    buckets, est_total = [], 0.0
    for b in range(4):
        seg = rest[edges[b]:edges[b + 1]]
        if len(seg) == 0:
            continue
        samp = rng.choice(seg, min(per_bucket, len(seg)), replace=False)
        k = sum(passes(i) for i in samp)
        lo, hi = wilson(k, len(samp))
        buckets.append(dict(bucket=b, size=len(seg), sampled=len(samp), passed=k, rate=k / len(samp),
                            est_missed=k / len(samp) * len(seg), est_lo=lo * len(seg), est_hi=hi * len(seg),
                            extra_k=edges[b + 1]))
        est_total += buckets[-1]["est_missed"]
    total = found + est_total
    cum, rec_k = found, (0 if found / max(total, 1e-9) >= target_recall else None)
    for bk in buckets:
        cum += bk["est_missed"]
        if rec_k is None and cum / max(total, 1e-9) >= target_recall:
            rec_k = bk["extra_k"]
    out = dict(role=role.id, shortlist=int(in_short.sum()), found_pass=found, est_missed=est_total,
               est_missed_lo=sum(b["est_lo"] for b in buckets), est_missed_hi=sum(b["est_hi"] for b in buckets),
               est_stage1_recall=found / max(total, 1e-9), buckets=buckets, recommended_extra_k=rec_k)
    if validate:  # only possible on synthetic data: grade everyone outside the shortlist
        out["exact_grader_missed"] = int(sum(passes(i) for i in rest))
        if truth_role is not None:
            out["truth_missed"] = len(truth_role["qualified"] - res.shortlisted_ids)
    return out
