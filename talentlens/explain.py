"""Contestability layer (the notice's 'right to human review / contest / provide more information').

`explain_decision` answers, for any expert -- even one who was NEVER shortlisted:
  * at which stage they were eliminated and by how much,
  * which fields drove their Stage-1 score and how much their headline counted,
  * which criteria cost them points and what evidence would change the grade,
  * a what-if: would a clearer headline have changed the Stage-1 outcome?
`request_review` writes an auditable event so a human reviewer can pick it up.
"""
from __future__ import annotations
import copy
import numpy as np
from .retrieval import ideal_candidate_query

HINTS = {
    "skill": "Describe concrete work using {name} in a role or project (and mention it in your skills list).",
    "years": "List start/end dates and clear job titles for every relevant role in {domain}.",
    "degree": "Add your highest completed degree and field of study.",
    "cert": "Add the {name} certification if you hold it.",
    "language": "Add your {lang} proficiency level to your profile.",
    "country": "This role requires a listed location in an eligible country.",
}


def explain_decision(pipe, role, profile_id: str, res) -> dict:
    p = pipe.by_id[profile_id]
    q = ideal_candidate_query(role)
    i = pipe.profiles.index(p)
    ret = res.retrieved
    score = float(ret.all_scores[i])
    rank = int((ret.all_scores > score).sum()) + 1
    out = dict(profile_id=profile_id, role=role.id, retrieval_score=round(score, 4), global_rank=rank,
               shortlist_size=len(ret.idx), field_breakdown=pipe.index.explain_row(p, q))
    if profile_id not in res.shortlisted_ids:
        out["stage_eliminated"] = "stage1_retrieval"
        cut = ret.cutoff["broad"] if not pipe.index.active[i] else min(ret.cutoff.values())
        out["score_gap_to_cutoff"] = round(cut - score, 4)
        alt = copy.deepcopy(p)
        alt.headline = f"{p.current_title} | {' | '.join(p.skills[:2])}"
        s_alt = float(pipe.index.score_variants([alt], q)[0])
        out["what_if_clear_headline"] = dict(
            new_score=round(s_alt, 4), would_be_shortlisted=bool(pipe.index.would_shortlist(s_alt, bool(pipe.index.active[i]), ret.cutoff)))
        out["note"] = ("Not surfaced to any reviewer for this role. Request human review to have a person assess "
                       "your profile directly.")
        return out
    rec = next(r for r in res.records if r.profile_id == profile_id)
    out.update(stage_eliminated=None if rec.outcome == "passed" else "stage2_grading", outcome=rec.outcome,
               match_score=round(rec.match_score, 3), hard_failed=rec.hard_failed)
    gaps = []
    for g, c in zip(rec.grades, role.criteria):
        if g["score"] < 2:
            gaps.append(dict(criterion=c.text, hard_filter=c.hard, score=g["score"], confidence=g["confidence"],
                             evidence=g["evidence"], how_to_improve=HINTS[c.kind].format(**c.params)))
    out["criteria_gaps"] = gaps
    if rec.outcome == "rescue_review":
        out["note"] = "A hard filter failed on thin evidence, so a human reviews this case before any rejection."
    return out


def to_text(e: dict) -> str:
    lines = [f"Expert {e['profile_id']} / role {e['role']}"]
    lines.append(f"Stage-1 rank {e['global_rank']} of pool (shortlist = top {e['shortlist_size']}, score {e['retrieval_score']})")
    if e.get("stage_eliminated") == "stage1_retrieval":
        lines.append(f"-> Eliminated at Stage 1 (short of cutoff by {e['score_gap_to_cutoff']}).")
        w = e["what_if_clear_headline"]
        lines.append(f"   With a clearer headline: score {w['new_score']}, shortlisted = {w['would_be_shortlisted']}.")
    else:
        lines.append(f"-> Stage 2 outcome: {e['outcome']} (match {e['match_score']}/2)")
        for g in e["criteria_gaps"]:
            tag = " [HARD FILTER]" if g["hard_filter"] else ""
            lines.append(f"   - {g['criterion']}{tag}: got {g['score']} (confidence {g['confidence']}). {g['evidence']}")
            lines.append(f"     To improve: {g['how_to_improve']}")
    if e.get("note"):
        lines.append(e["note"])
    return "\n".join(lines)


def request_review(pipe, role, profile_id: str, statement: str = "") -> dict:
    ev = dict(event="human_review_requested", role_id=role.id, profile_id=profile_id,
              expert_statement=statement, status="open")
    pipe.log_event(ev)
    return ev


def record_override(pipe, role, profile_id: str, reviewer: str, decision: str, reason: str) -> dict:
    ev = dict(event="human_override", role_id=role.id, profile_id=profile_id, reviewer=reviewer,
              decision=decision, reason=reason)
    pipe.log_event(ev)
    return ev
