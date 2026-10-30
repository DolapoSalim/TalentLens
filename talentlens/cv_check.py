"""CV readiness check: put ONE real CV through the reverse-engineered pipeline for ONE job description.

The CV is scored against a synthetic benchmark pool exactly as a pool member would be:
  Stage 1  where it lands in the retrieval ranking and whether it clears the shortlist cutoff
  Stage 2  per-criterion grades, hard filters and match score (same grader, same thresholds)
  Rank     where a passing CV would sit among the pool's passers (reviewers see the top 20 first)
It also scores formatting-only what-ifs (no new claims) to show what presentation alone would change.
Nothing about the CV is logged.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import numpy as np
from .schema import Profile, Role, CandidateInfo
from .synth import generate_profiles
from .grader import RuleGrader, _terms, _hit, _within_one_edit
from .pipeline import Pipeline, PipelineConfig, rank_weight
from .retrieval import RetrievalConfig, FIELDS, ideal_candidate_query
from .explain import HINTS
from .taxonomy import DOMAINS

FIELD_NOTES = {
    "headline": "The tagline under your name. Weighted most heavily at Stage 1.",
    "current_role": "Current job title and employer, compared with the role's target title.",
    "experience": "Only your 5 most recent roles are read, and only the first 200 characters of each description.",
    "education": "Top 3 education entries: degree, field and institution.",
    "skills": "Only the first 15 skills in your list are read.",
    "location": "City and country.",
}
READINESS_WEIGHTS = {"stage1": 0.35, "stage2": 0.45, "rank": 0.20}
PRIORITY = {"high": 0, "medium": 1, "low": 2}


@dataclasses.dataclass
class CheckConfig:
    pool_size: int = 4000
    seed: int = 7
    mode: str = "fielded"      # "fielded" = the platform as disclosed (headline-heavy); "adaptive" = this project's fix
    review_size: int = 20


def fingerprint(role: Role) -> str:
    """Role id derived from its content, so an edited role never reuses cached grades."""
    blob = json.dumps([role.title, role.target_title, role.domain, role.description, role.pay_rate,
                       [dataclasses.asdict(c) for c in role.criteria]], sort_keys=True)
    return "JD-" + hashlib.sha1(blob.encode()).hexdigest()[:10]


def _evidence(cv: Profile, skill: str) -> tuple[bool, bool]:
    """(listed in skills, shown in a role or project) for one skill."""
    pats = _terms(skill)
    listed = any(_hit(pats, s) for s in cv.skills)
    used = any(_hit(pats, e.description) for e in cv.experiences) or any(_hit(pats, p) for p in cv.projects)
    return listed, used


class CVChecker:
    def __init__(self, cfg: CheckConfig | None = None, profiles: list[Profile] | None = None, grader=None):
        """`grader` grades both the CV and the pool's shortlist (~10% of the pool), so an LLM grader means
        hundreds of calls per new role."""
        self.cfg = cfg or CheckConfig()
        if self.cfg.mode not in ("fielded", "adaptive"):
            raise ValueError("mode must be 'fielded' or 'adaptive'")
        P = profiles if profiles is not None else generate_profiles(self.cfg.pool_size, self.cfg.seed)
        n = len(P)
        rc = RetrievalConfig(mode=self.cfg.mode, k_active=int(150 * n / 4000), k_broad=int(250 * n / 4000))
        self.pipe = Pipeline(P, grader or RuleGrader(fuzzy=True), PipelineConfig(retrieval=rc, review_size=self.cfg.review_size))
        self._runs = {}

    # ------------------------------------------------------------ main entry
    def check(self, cv: Profile, role: Role) -> dict:
        if not role.criteria:
            raise ValueError("The job description has no recognisable criteria to grade against.")
        if role.domain not in DOMAINS:
            raise ValueError(f"Unknown domain '{role.domain}'; use one of {list(DOMAINS)}.")
        role = dataclasses.replace(role, id=fingerprint(role))
        cv = dataclasses.replace(cv, last_active_days=0)          # applying now -> 'recently active' phase
        q = ideal_candidate_query(role)
        if role.id not in self._runs:
            self._runs[role.id] = self.pipe.run(role, q)
        res = self._runs[role.id]

        s1 = self._stage1(cv, q, res.retrieved)
        s2 = self._stage2(cv, role)
        rank = self._rank(s1, s2, role, res)
        what_ifs = self._what_ifs(cv, role, q, res.retrieved, s1["score"])
        readiness, verdict, detail = _readiness(s1, s2, rank)
        return dict(
            role=dict(title=role.title, target_title=role.target_title, domain=role.domain, pay_rate=role.pay_rate,
                      criteria=[dict(criterion=c.text, must_have=c.hard) for c in role.criteria]),
            benchmark=dict(pool=len(self.pipe.profiles), mode=self.cfg.mode, shortlist=len(res.retrieved.idx)),
            readiness=readiness, verdict=verdict, verdict_detail=detail,
            stage1=s1, stage2=s2, rank=rank, what_ifs=what_ifs,
            recommendations=_recommendations(cv, role, s1, s2, what_ifs))

    # ------------------------------------------------------------ stages
    def _position(self, score: float, ret) -> dict:
        return dict(score=float(score), percentile=float((ret.all_scores < score).mean()),
                    shortlisted=bool(self.pipe.index.would_shortlist(score, True, ret.cutoff)))

    def _stage1(self, cv: Profile, q: dict, ret) -> dict:
        ix = self.pipe.index
        cut = min(ret.cutoff.values())
        S, _ = ix.field_scores(q)
        row = ix.explain_row(cv, q)
        fields = [dict(field=f, weight=row[f]["weight"], you=row[f]["similarity"],
                       shortlist_median=float(np.median(S[ret.idx, j])), contribution=row[f]["contribution"],
                       what_is_read=FIELD_NOTES[f]) for j, f in enumerate(FIELDS)]
        return dict(**self._position(float(ix.score_variants([cv], q)[0]), ret), cutoff=cut,
                    cutoff_percentile=float((ret.all_scores < cut).mean()), shortlist_size=len(ret.idx),
                    pool_size=len(ret.all_scores), fields=fields, headline_consistency=row["_headline_consistency"])

    def _stage2(self, cv: Profile, role: Role) -> dict:
        info = CandidateInfo.from_profile(cv)
        key = "CV-" + hashlib.sha1(json.dumps(info.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:12]
        grades = self.pipe.grader.grade(key, info, role)
        outcome, decision_score, failed = self.pipe.decide(grades, role)
        rows = [dict(criterion=c.text, kind=c.kind, must_have=c.hard, score=g.score, confidence=g.confidence,
                     evidence=g.evidence) for c, g in zip(role.criteria, grades)]
        return dict(outcome=outcome, match=float(np.mean([g.score for g in grades])), decision_score=decision_score,
                    threshold=self.pipe.cfg.pass_threshold, hard_failed=failed, grades=rows)

    def _rank(self, s1: dict, s2: dict, role: Role, res) -> dict | None:
        """Final-rank position among the pool's passers. If the CV is not shortlisted this is hypothetical."""
        if s2["outcome"] != "passed":
            return None
        w = rank_weight(role.pay_rate, self.pipe.cfg)
        lo, hi = float(res.retrieved.score.min()), float(res.retrieved.score.max())
        final = (1 - w) * float(np.clip((s1["score"] - lo) / (hi - lo + 1e-9), 0, 1)) + w * s2["match"] / 2
        passers = res.ranked
        rank = 1 + sum(r.final_score > final for r in passers)
        return dict(final=final, rank=rank, passers=len(passers) + 1, ai_weight=w, review_size=self.cfg.review_size,
                    in_review=rank <= self.cfg.review_size, hypothetical=not s1["shortlisted"])

    def _what_ifs(self, cv: Profile, role: Role, q: dict, ret, base: float) -> list[dict]:
        """Formatting-only changes: reuse the CV's own title and skills, never add a claim."""
        skills = [c.params["name"] for c in role.criteria if c.kind == "skill"]
        shown = [s for s in skills if any(_evidence(cv, s))]
        variants = {}
        if cv.current_title:
            h = f"{cv.current_title} | {' | '.join(shown[:2])}" if shown else cv.current_title
            if h.strip().lower() != cv.headline.strip().lower():
                variants["clear headline"] = (dataclasses.replace(cv, headline=h), f'Use the headline "{h}"')
        first = [x for x in cv.skills if any(_hit(_terms(s), x) for s in skills)]
        reordered = first + [x for x in cv.skills if x not in first]
        if reordered[:15] != cv.skills[:15]:
            variants["role skills first"] = (dataclasses.replace(cv, skills=reordered),
                                             "Move to the top of your skills list: " + ", ".join(first))
        if len(variants) == 2:
            (a, da), (b, db) = variants.values()
            variants["both"] = (dataclasses.replace(a, skills=b.skills), f"{da}; {db[0].lower() + db[1:]}")
        if not variants:
            return []
        scores = self.pipe.index.score_variants([p for p, _ in variants.values()], q)
        return [dict(change=name, detail=detail, delta=float(s - base), **self._position(float(s), ret))
                for (name, (_, detail)), s in zip(variants.items(), scores)]


# ------------------------------------------------------------ scoring and advice
def _readiness(s1: dict, s2: dict, rank: dict | None) -> tuple[int, str, str]:
    stage1 = 1.0 if s1["shortlisted"] else min(0.95, s1["percentile"] / max(s1["cutoff_percentile"], 1e-9))
    placed = 0.0 if rank is None or not s1["shortlisted"] else (1.0 if rank["in_review"] else rank["review_size"] / rank["rank"])
    W = READINESS_WEIGHTS
    score = 100 * (W["stage1"] * stage1 + W["stage2"] * s2["match"] / 2 + W["rank"] * placed)
    top = f"top {100 * (1 - s1['percentile']):.0f}%"
    o = s2["outcome"]
    if o == "rejected_hard_filter":
        return min(round(score), 25), "Not ready", "Your CV clearly fails a must-have requirement, so it is rejected automatically."
    if o == "below_threshold":
        return round(score), "Not ready", (f"Your CV's evidence scores {s2['match']:.2f}/2 against the criteria; "
                                           f"the pass bar is {s2['threshold']}.")
    if o == "rescue_review":
        return min(round(score), 50), "At risk", ("A must-have requirement is not visible in your CV. On the platform as "
                                                  "disclosed this is an automatic rejection; TalentLens would send it to a human instead.")
    if not s1["shortlisted"]:
        return round(score), "Hidden", (f"Your CV would pass the AI grading, but Stage 1 places it in the {top} of the pool, "
                                        f"below the shortlist cutoff (top {100 * (1 - s1['cutoff_percentile']):.0f}%). "
                                        "No grader or person would ever see it. Fix how it is presented first.")
    if not rank["in_review"]:
        return round(score), "Competitive", (f"Shortlisted and passing, ranked #{rank['rank']} of {rank['passers']} passers. "
                                             f"Reviewers start with the top {rank['review_size']}.")
    return round(score), "Ready", (f"Shortlisted, passing, and ranked #{rank['rank']} of {rank['passers']}: inside the "
                                   f"top {rank['review_size']} that reviewers see first.")


def _skill_action(cv: Profile, name: str) -> str:
    listed, used = _evidence(cv, name)
    if not listed:
        typo = next((s for s in cv.skills if len(name) >= 5 and _within_one_edit(s.lower(), name.lower())), None)
        if typo:
            return f"Fix the spelling of '{typo}' in your skills list (should be '{name}')."
    if listed and not used:
        return (f"Show where you used {name}: name it in the description of a role or project. A skill gets full "
                "marks only when it is both listed and backed by work.")
    if used and not listed:
        return f"Add {name} to your skills list. It appears in your work history but not among your skills."
    return (f"{name} does not appear anywhere in your CV. If you have it, list it and describe a role or project "
            "where you used it.")


def _recommendations(cv: Profile, role: Role, s1: dict, s2: dict, what_ifs: list[dict]) -> list[dict]:
    recs = []

    def add(priority, stage, issue, action):
        recs.append(dict(priority=priority, stage=stage, issue=issue, action=action))

    for c, g in zip(role.criteria, s2["grades"]):
        if g["score"] == 2:
            continue
        priority = "high" if c.hard and g["score"] == 0 else "medium" if c.hard or g["score"] == 0 else "low"
        if c.kind == "skill":
            action = _skill_action(cv, c.params["name"])
        elif c.kind == "years":
            titles = ", ".join(DOMAINS[c.params["domain"]]["titles"][:3])
            action = (f"Give every relevant role a start and end year and a standard job title (e.g. {titles}). "
                      "Roles with non-standard titles count only 50%, and only when the description shows the field's skills.")
        else:
            action = HINTS[c.kind].format(**c.params)
        tag = " (must-have)" if c.hard else ""
        add(priority, "Stage 2", f"{c.text}{tag}: scored {g['score']}/2. {g['evidence']}", action)

    if not s1["shortlisted"]:
        add("high", "Stage 1", f"Below the shortlist cutoff (you: top {100 * (1 - s1['percentile']):.0f}% of the pool; "
                               f"cutoff: top {100 * (1 - s1['cutoff_percentile']):.0f}%).",
            "Nothing at Stage 2 matters until this is fixed. Apply the presentation fixes below.")
    for w in what_ifs:
        if w["change"] == "both" or w["delta"] <= 0:
            continue
        flips = w["shortlisted"] and not s1["shortlisted"]
        add("high" if flips else "medium" if w["delta"] > 0.01 else "low", "Stage 1",
            f"Presentation: '{w['change']}' moves you from top {100 * (1 - s1['percentile']):.0f}% to top "
            f"{100 * (1 - w['percentile']):.0f}%" + (" and onto the shortlist." if flips else "."), w["detail"] + ".")

    skills = [c.params["name"] for c in role.criteria if c.kind == "skill"]
    late = {}
    for e in cv.experiences[:5]:
        for s in skills:
            pats = _terms(s)
            if _hit(pats, e.description[200:]) and not _hit(pats, e.description[:200]):
                late.setdefault(e.title or "a role", []).append(s)
    for title, ss in late.items():
        add("medium", "Stage 1", f"In '{title}', {', '.join(ss)} appear only after the first 200 characters.",
            f"Mention {', '.join(ss)} in the first sentence of that description; Stage 1 reads only the first 200 characters.")
    old = [s for s in skills if any(_hit(_terms(s), e.description) for e in cv.experiences[5:])
           and not any(_hit(_terms(s), e.description) for e in cv.experiences[:5])]
    if old:
        add("low", "Stage 1", f"{', '.join(old)} appear only in roles older than your 5 most recent.",
            "Stage 1 reads only your 5 most recent roles. Merge older short roles or mention these skills in a recent one.")
    cur = next(f for f in s1["fields"] if f["field"] == "current_role")
    if cur["you"] < 0.5 * cur["shortlist_median"]:
        add("low", "Stage 1", f"Your current title '{cv.current_title}' reads far from '{role.target_title}'.",
            "If your title is non-standard (e.g. 'Data Wizard'), write the standard equivalent next to it.")
    recs.sort(key=lambda r: PRIORITY[r["priority"]])
    return recs


def to_text(rep: dict) -> str:
    s1, s2, rk = rep["stage1"], rep["stage2"], rep["rank"]
    lines = [f"CV readiness for: {rep['role']['title']}  (benchmark: {rep['benchmark']['pool']} synthetic experts, "
             f"{rep['benchmark']['mode']} retrieval)",
             f"READINESS {rep['readiness']}/100  ->  {rep['verdict']}", rep["verdict_detail"], "",
             f"Stage 1: top {100 * (1 - s1['percentile']):.0f}% of the pool; shortlist cutoff top "
             f"{100 * (1 - s1['cutoff_percentile']):.0f}%; shortlisted = {s1['shortlisted']}",
             f"Stage 2: {s2['outcome']}, match {s2['match']:.2f}/2 (pass bar {s2['threshold']})"]
    if rk:
        lines.append(f"Rank:    #{rk['rank']} of {rk['passers']} passers" + (" (if shortlisted)" if rk["hypothetical"] else ""))
    lines += ["", "Criteria:"]
    lines += [f"  [{g['score']}/2]{' MUST' if g['must_have'] else '     '} {g['criterion']}: {g['evidence']}" for g in s2["grades"]]
    lines += ["", "What to change (most important first):"]
    lines += [f"  {i}. [{r['priority']}, {r['stage']}] {r['issue']}\n     -> {r['action']}"
              for i, r in enumerate(rep["recommendations"], 1)]
    return "\n".join(lines)
