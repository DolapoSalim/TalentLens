"""Evaluation against the synthetic ground truth."""
from __future__ import annotations
import math
import numpy as np
from .schema import Profile, Role
from .synth import true_summary


def build_truth(profiles: list[Profile], roles: list[Role], threshold: float = 1.2) -> dict:
    """truth[role_id] = dict(rel: {pid: relevance}, qualified: set(pid))"""
    out = {}
    for role in roles:
        rel, qual = {}, set()
        for p in profiles:
            t = true_summary(p, role, threshold)
            rel[p.id] = t["relevance"]
            if t["qualified"]:
                qual.add(p.id)
        out[role.id] = dict(rel=rel, qualified=qual)
    return out


def ndcg_at_k(ranked_ids: list[str], rel: dict[str, float], k: int = 20) -> float:
    dcg = sum(rel[pid] / math.log2(i + 2) for i, pid in enumerate(ranked_ids[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def funnel(res, truth_role: dict, profiles_by_id: dict, k: int = 20) -> dict:
    """Where do truly-qualified experts get lost? Also split by headline quality."""
    Q = truth_role["qualified"]
    rel = truth_role["rel"]
    S = res.shortlisted_ids
    ranked = [r.profile_id for r in res.ranked]
    rescue = {r.profile_id for r in res.rescue}
    rejected = {r.profile_id for r in res.by("rejected_hard_filter")}
    below = {r.profile_id for r in res.by("below_threshold")}
    q_inf = {x for x in Q if profiles_by_id[x].hidden["headline_quality"] == "informative"}
    q_uninf = Q - q_inf
    surfaced = set(ranked) | rescue
    return dict(
        qualified=len(Q), shortlist=len(S),
        stage1_recall=len(Q & S) / len(Q),
        stage1_precision=len(Q & S) / len(S),
        stage1_recall_informative=len(q_inf & S) / max(1, len(q_inf)),
        stage1_recall_uninformative=len(q_uninf & S) / max(1, len(q_uninf)),
        lost_stage1=len(Q - S), lost_hard_filter=len(Q & rejected), lost_threshold=len(Q & below),
        rescued=len(Q & rescue), rescue_queue=len(rescue), rescue_precision=len(Q & rescue) / max(1, len(rescue)),
        surfaced_recall=len(Q & surfaced) / len(Q),
        surfaced_recall_uninformative=len(q_uninf & surfaced) / max(1, len(q_uninf)),
        precision_at_k=len(Q & set(ranked[:k])) / max(1, min(k, len(ranked))),
        ndcg_at_k=ndcg_at_k(ranked, rel, k),
        hard_rejects=len(rejected), hard_reject_false=len(Q & rejected),
    )


def average(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
