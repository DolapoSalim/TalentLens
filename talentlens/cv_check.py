"""CV readiness check: how ready is ONE CV for ONE job description, judged by the reverse-engineered pipeline.

The main score is absolute (no comparison with other candidates):
  Requirements  the Stage-2 grader, hard filters and 1.2 pass bar -> per-criterion 0/1/2 with evidence
  Visibility    a checklist of how Stage 1 reads a CV (headline, current title, first 200 characters of the
                5 most recent roles, first 15 skills, ...), weighted in the disclosed field order
`PoolSimulator` is optional: it ranks the CV against a synthetic pool to show what the checklist items are worth.
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
from .pipeline import Pipeline, PipelineConfig, decide, rank_weight
from .retrieval import RetrievalConfig, FIELDS, ideal_candidate_query
from .explain import HINTS
from .taxonomy import DOMAINS

READINESS_WEIGHTS = {"requirements": 0.60, "visibility": 0.40}
# Visibility check weights follow the disclosed Stage-1 field order (headline > current role > experience > skills > ...).
VISIBILITY_WEIGHTS = {"headline": 0.30, "current title": 0.20, "skills in recent work": 0.20, "skills list order": 0.12,
                      "dated roles": 0.06, "standard titles": 0.06, "language levels": 0.03, "location": 0.03}
STRONG_MATCH = 1.5           # requirement score (of 2) needed for a "Ready" verdict; the pass bar is 1.2
PRIORITY = {"high": 0, "medium": 1, "low": 2}
FIELD_NOTES = {
    "headline": "The tagline under your name. Weighted most heavily at Stage 1.",
    "current_role": "Current job title and employer, compared with the role's target title.",
    "experience": "Only your 5 most recent roles are read, and only the first 200 characters of each description.",
    "education": "Top 3 education entries: degree, field and institution.",
    "skills": "Only the first 15 skills in your list are read.",
    "location": "City and country.",
}


def fingerprint(role: Role) -> str:
    """Role id derived from its content, so an edited role never reuses cached grades."""
    blob = json.dumps([role.title, role.target_title, role.domain, role.description, role.pay_rate,
                       [dataclasses.asdict(c) for c in role.criteria]], sort_keys=True)
    return "JD-" + hashlib.sha1(blob.encode()).hexdigest()[:10]


def _prepare(role: Role) -> Role:
    if not role.criteria:
        raise ValueError("The job description has no recognisable criteria to grade against.")
    if role.domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{role.domain}'; use one of {list(DOMAINS)}.")
    return dataclasses.replace(role, id=fingerprint(role))


def _role_skills(role: Role) -> list[str]:
    return [c.params["name"] for c in role.criteria if c.kind == "skill"]


def _evidence(cv: Profile, skill: str) -> tuple[bool, bool]:
    """(listed in skills, shown in a role or project) for one skill."""
    pats = _terms(skill)
    listed = any(_hit(pats, s) for s in cv.skills)
    used = any(_hit(pats, e.description) for e in cv.experiences) or any(_hit(pats, p) for p in cv.projects)
    return listed, used


def _standard_title(title: str, domain: str) -> bool:
    return any(t.lower() in title.lower() for t in DOMAINS[domain]["titles"])


def suggested_headline(cv: Profile, role: Role) -> str:
    """Formatting-only rewrite: the CV's own current title plus up to two role skills the CV already shows."""
    shown = [s for s in _role_skills(role) if any(_evidence(cv, s))]
    return " | ".join([cv.current_title] + shown[:2]) if cv.current_title else ""


# ------------------------------------------------------------ visibility checklist
def visibility(cv: Profile, role: Role) -> dict:
    skills = _role_skills(role)
    titles = DOMAINS[role.domain]["titles"]
    checks = []

    def add(name, score, detail, action):
        checks.append(dict(check=name, weight=VISIBILITY_WEIGHTS[name], score=score, detail=detail, action=action))

    has_title = _standard_title(cv.headline, role.domain) or role.target_title.lower() in cv.headline.lower()
    named = [s for s in skills if _hit(_terms(s), cv.headline)]
    score = (0.5 * has_title + 0.5 * bool(named)) if skills else float(has_title)
    h = suggested_headline(cv, role)
    add("headline", score,
        f"'{cv.headline or '(none)'}': " + ("names a job title for this field" if has_title else "no job title for this field")
        + (f", names {', '.join(named)}" if named else ", names none of the role's skills"),
        f'Lead with your job title and the role\'s skills you have, e.g. "{h}".' if h else
        "Add a one-line headline under your name: your job title plus two key skills.")

    ok = bool(cv.current_title) and _standard_title(cv.current_title, role.domain)
    add("current title", float(ok), f"Most recent title: '{cv.current_title or '(none)'}'",
        f"If your title is non-standard, write the standard equivalent next to it (e.g. {', '.join(titles[:3])}).")

    used = [s for s in skills if any(_hit(_terms(s), e.description) for e in cv.experiences)]
    early = [s for s in used if any(_hit(_terms(s), e.description[:200]) for e in cv.experiences[:5])]
    late = [s for s in used if s not in early]
    add("skills in recent work", len(early) / len(used) if used else 0.0,
        (f"Read by Stage 1: {', '.join(early) or 'none'}" + (f"; missed (too far down or in old roles): {', '.join(late)}" if late else ""))
        if used else "None of the role's skills appear in any role description.",
        f"Name {', '.join(late)} in the first sentence of one of your 5 most recent roles; only the first 200 "
        "characters of each are read." if late else "Describe the role's key skills you used in your recent role descriptions.")

    listed = {s: next(i for i, x in enumerate(cv.skills) if _hit(_terms(s), x)) for s in skills if _evidence(cv, s)[0]}
    deep = [s for s, i in listed.items() if i >= 15]
    add("skills list order", (len(listed) - len(deep)) / len(listed) if listed else 0.0,
        (f"Role skills at positions {', '.join(f'{s} #{i + 1}' for s, i in listed.items())}" if listed
         else "None of the role's skills are in your skills list."),
        f"Move {', '.join(deep)} into your first 15 skills; the rest are not read." if deep
        else "Put the role's skills you have at the top of your skills list.")

    add("dated roles", float(bool(cv.experiences)), f"{len(cv.experiences)} roles with start and end years",
        "Give every role a start and end year; years of experience are counted only from dated roles.")

    titles_low = [t.lower() for t in titles]
    dom_skills = [_terms(s) for s in DOMAINS[role.domain]["skills"]]
    std = sum(e.years for e in cv.experiences if any(t in e.title.lower() for t in titles_low))
    amb = sum(e.years for e in cv.experiences if not any(t in e.title.lower() for t in titles_low)
              and sum(_hit(p, e.description) for p in dom_skills) >= 2)
    add("standard titles", std / (std + amb) if std + amb else None,
        f"{std} years under standard titles" + (f", {amb} under non-standard ones (counted at 50%)" if amb else ""),
        f"Use standard job titles for relevant roles (e.g. {', '.join(titles[:3])}).")

    langs = [c.params["lang"] for c in role.criteria if c.kind == "language"]
    missing = [l for l in langs if l not in cv.languages]
    add("language levels", (len(langs) - len(missing)) / len(langs) if langs else None,
        f"Required: {', '.join(langs)}; levels stated for {', '.join(l for l in langs if l not in missing) or 'none'}" if langs else "",
        f"State a level for {', '.join(missing)} (e.g. C1, fluent, native); a missing level reads as 'not met'.")

    needs_place = any(c.kind == "country" for c in role.criteria)
    add("location", float(bool(cv.country)) if needs_place else None, f"Country: {cv.country or '(not found)'}",
        "Put your city and country in the CV header; this role has a location rule.")

    scored = [c for c in checks if c["score"] is not None]
    total = sum(c["weight"] * c["score"] for c in scored) / sum(c["weight"] for c in scored)
    return dict(score=float(total), checks=checks)


# ------------------------------------------------------------ checker
class CVChecker:
    def __init__(self, grader=None, cfg: PipelineConfig | None = None):
        self.grader = grader or RuleGrader(fuzzy=True)
        self.cfg = cfg or PipelineConfig()

    def requirements(self, cv: Profile, role: Role) -> dict:
        """Stage 2 exactly as the pipeline runs it. `role` must already carry its fingerprint id."""
        info = CandidateInfo.from_profile(cv)
        key = "CV-" + hashlib.sha1(json.dumps(info.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:12]
        grades = self.grader.grade(key, info, role)
        outcome, decision_score, failed = decide(grades, role, self.cfg)
        rows = [dict(criterion=c.text, kind=c.kind, must_have=c.hard, score=g.score, confidence=g.confidence,
                     evidence=g.evidence) for c, g in zip(role.criteria, grades)]
        return dict(outcome=outcome, match=float(np.mean([g.score for g in grades])), decision_score=decision_score,
                    threshold=self.cfg.pass_threshold, hard_failed=failed, grades=rows)

    def check(self, cv: Profile, role: Role) -> dict:
        role = _prepare(role)
        req, vis = self.requirements(cv, role), visibility(cv, role)
        readiness, verdict, detail = _readiness(req, vis)
        return dict(role=dict(title=role.title, target_title=role.target_title, domain=role.domain, pay_rate=role.pay_rate,
                              criteria=[dict(criterion=c.text, must_have=c.hard) for c in role.criteria]),
                    readiness=readiness, verdict=verdict, verdict_detail=detail, requirements=req, visibility=vis,
                    recommendations=_recommendations(cv, role, req, vis))


def _readiness(req: dict, vis: dict) -> tuple[int, str, str]:
    W = READINESS_WEIGHTS
    score = round(100 * (W["requirements"] * req["match"] / 2 + W["visibility"] * vis["score"]))
    o, v = req["outcome"], vis["score"]
    if o == "rejected_hard_filter":
        return min(score, 25), "Not ready", "Your CV clearly fails a must-have requirement, so it would be rejected automatically."
    if o == "below_threshold":
        return score, "Not ready", f"Your CV's evidence scores {req['match']:.2f}/2 against the criteria; the pass bar is {req['threshold']}."
    if o == "rescue_review":
        return min(score, 50), "At risk", ("A must-have requirement is not visible in your CV. On the platform as disclosed "
                                           "this is an automatic rejection; TalentLens would send it to a human instead.")
    if v < 0.6:
        return score, "Hard to find", (f"Your CV meets the requirements, but scores {v:.0%} on visibility: retrieval reads "
                                       "the headline, current title and the start of recent roles first, and yours undersell you.")
    weak = ([f"only just clears the pass bar ({req['match']:.2f}/2; strong is {STRONG_MATCH}+)"] if req["match"] < STRONG_MATCH else []) \
        + ([f"visibility is {v:.0%}"] if v < 0.8 else [])
    if weak:
        return score, "Nearly ready", f"Your CV meets the requirements, but {' and '.join(weak)}. The fixes below close the gap."
    return score, "Ready", (f"Your CV meets the requirements strongly ({req['match']:.2f}/2) and is presented the way "
                            f"retrieval reads it (visibility {v:.0%}).")


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
    return f"{name} does not appear anywhere in your CV. If you have it, list it and describe a role or project where you used it."


def _recommendations(cv: Profile, role: Role, req: dict, vis: dict) -> list[dict]:
    recs = []
    for c, g in zip(role.criteria, req["grades"]):
        if g["score"] == 2:
            continue
        if c.kind == "skill":
            action = _skill_action(cv, c.params["name"])
        elif c.kind == "years":
            titles = ", ".join(DOMAINS[c.params["domain"]]["titles"][:3])
            action = (f"Give every relevant role a start and end year and a standard job title (e.g. {titles}). "
                      "Roles with non-standard titles count only 50%, and only when the description shows the field's skills.")
        else:
            action = HINTS[c.kind].format(**c.params)
        recs.append(dict(priority="high" if c.hard and g["score"] == 0 else "medium" if c.hard or g["score"] == 0 else "low",
                         area="Requirements", issue=f"{c.text}{' (must-have)' if c.hard else ''}: scored {g['score']}/2. {g['evidence']}",
                         action=action))
    for ch in vis["checks"]:
        if ch["score"] is None or ch["score"] >= 1:
            continue
        lost = ch["weight"] * (1 - ch["score"])
        recs.append(dict(priority="high" if lost >= 0.1 else "medium" if lost >= 0.03 else "low", area="Visibility",
                         issue=f"{ch['check']}: {ch['score']:.0%}. {ch['detail']}", action=ch["action"]))
    recs.sort(key=lambda r: PRIORITY[r["priority"]])
    return recs


def to_text(rep: dict) -> str:
    req, vis = rep["requirements"], rep["visibility"]
    lines = [f"CV readiness for: {rep['role']['title']}",
             f"READINESS {rep['readiness']}/100  ->  {rep['verdict']}", rep["verdict_detail"], "",
             f"Requirements: {req['outcome']}, match {req['match']:.2f}/2 (pass bar {req['threshold']})"]
    lines += [f"  [{g['score']}/2]{' MUST' if g['must_have'] else '     '} {g['criterion']}: {g['evidence']}" for g in req["grades"]]
    lines += ["", f"Visibility: {vis['score']:.0%}"]
    lines += [f"  [{'n/a' if c['score'] is None else format(c['score'], '.0%'):>4}] {c['check']} (weight {c['weight']:.0%}): {c['detail']}"
              for c in vis["checks"]]
    lines += ["", "What to change (most important first):"]
    lines += [f"  {i}. [{r['priority']}, {r['area']}] {r['issue']}\n     -> {r['action']}" for i, r in enumerate(rep["recommendations"], 1)]
    if rep.get("simulation"):
        s = rep["simulation"]
        lines += ["", f"Simulation vs a synthetic pool of {s['pool']} ({s['mode']}): top {100 * (1 - s['percentile']):.0f}%, "
                      f"shortlist cutoff top {100 * (1 - s['cutoff_percentile']):.0f}%, shortlisted = {s['shortlisted']}"]
        lines += [f"  what-if {w['change']}: top {100 * (1 - w['percentile']):.0f}%, shortlisted = {w['shortlisted']}" for w in s["what_ifs"]]
    return "\n".join(lines)


# ------------------------------------------------------------ optional: synthetic-pool simulation
class PoolSimulator:
    """Rank the CV against a synthetic pool, as any pool member would be. The competitors are synthetic, so the
    percentile shows how a CV *reads* to this kind of system, not how it compares with real applicants.
    The grader grades the pool's shortlist (~10% of the pool), so an LLM grader means hundreds of calls per role."""

    def __init__(self, pool_size: int = 4000, seed: int = 7, mode: str = "fielded", review_size: int = 20,
                 profiles: list[Profile] | None = None, grader=None):
        if mode not in ("fielded", "adaptive"):
            raise ValueError("mode must be 'fielded' (the platform as disclosed) or 'adaptive' (TalentLens)")
        P = profiles if profiles is not None else generate_profiles(pool_size, seed)
        n = len(P)
        rc = RetrievalConfig(mode=mode, k_active=int(150 * n / 4000), k_broad=int(250 * n / 4000))
        self.mode, self.review_size = mode, review_size
        self.pipe = Pipeline(P, grader or RuleGrader(fuzzy=True), PipelineConfig(retrieval=rc, review_size=review_size))
        self.checker = CVChecker(self.pipe.grader, self.pipe.cfg)
        self._runs = {}

    def _position(self, score: float, ret) -> dict:
        return dict(score=float(score), percentile=float((ret.all_scores < score).mean()),
                    shortlisted=bool(self.pipe.index.would_shortlist(score, True, ret.cutoff)))

    def simulate(self, cv: Profile, role: Role) -> dict:
        role = _prepare(role)
        cv = dataclasses.replace(cv, last_active_days=0)          # applying now -> 'recently active' phase
        ix, q = self.pipe.index, ideal_candidate_query(role)
        if role.id not in self._runs:
            self._runs[role.id] = self.pipe.run(role, q)
        res = self._runs[role.id]
        ret = res.retrieved
        cut = min(ret.cutoff.values())
        score = float(ix.score_variants([cv], q)[0])
        S, _ = ix.field_scores(q)
        row = ix.explain_row(cv, q)
        fields = [dict(field=f, weight=row[f]["weight"], you=row[f]["similarity"], shortlist_median=float(np.median(S[ret.idx, j])),
                       what_is_read=FIELD_NOTES[f]) for j, f in enumerate(FIELDS)]
        out = dict(pool=len(self.pipe.profiles), mode=self.mode, shortlist_size=len(ret.idx), cutoff=cut,
                   cutoff_percentile=float((ret.all_scores < cut).mean()), fields=fields,
                   **self._position(score, ret), what_ifs=self._what_ifs(cv, role, q, ret, score))

        req = self.checker.requirements(cv, role)
        out["rank"] = None
        if req["outcome"] == "passed":
            w = rank_weight(role.pay_rate, self.pipe.cfg)
            lo, hi = float(ret.score.min()), float(ret.score.max())
            final = (1 - w) * float(np.clip((score - lo) / (hi - lo + 1e-9), 0, 1)) + w * req["match"] / 2
            rank = 1 + sum(r.final_score > final for r in res.ranked)
            out["rank"] = dict(rank=rank, passers=len(res.ranked) + 1, review_size=self.review_size,
                               in_review=rank <= self.review_size, hypothetical=not out["shortlisted"])
        return out

    def _what_ifs(self, cv: Profile, role: Role, q: dict, ret, base: float) -> list[dict]:
        """Formatting-only changes: reuse the CV's own title and skills, never add a claim."""
        skills = _role_skills(role)
        variants = {}
        h = suggested_headline(cv, role)
        if h and h.lower() != cv.headline.strip().lower():
            variants["clear headline"] = (dataclasses.replace(cv, headline=h), f'Headline "{h}"')
        first = [x for x in cv.skills if any(_hit(_terms(s), x) for s in skills)]
        reordered = first + [x for x in cv.skills if x not in first]
        if reordered[:15] != cv.skills[:15]:
            variants["role skills first"] = (dataclasses.replace(cv, skills=reordered), "Skills list starts with " + ", ".join(first))
        if len(variants) == 2:
            (a, da), (b, db) = variants.values()
            variants["both"] = (dataclasses.replace(a, skills=b.skills), f"{da}; {db[0].lower() + db[1:]}")
        if not variants:
            return []
        scores = self.pipe.index.score_variants([p for p, _ in variants.values()], q)
        return [dict(change=name, detail=detail, delta=float(s - base), **self._position(float(s), ret))
                for (name, (_, detail)), s in zip(variants.items(), scores)]
