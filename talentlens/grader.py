"""Stage 2 graders. Every grader returns, per criterion: score (0/1/2), confidence, evidence.

`confidence` is the key addition over the notice's description: a score of 0 that rests on
*absence of evidence* (e.g. skill never mentioned, language not listed) gets LOW confidence,
which the pipeline uses to route hard-filter rejections to a human instead of auto-rejecting.
"""
from __future__ import annotations
import re
from typing import Protocol
from .schema import CandidateInfo, Role, Criterion, CriterionGrade
from .taxonomy import (DOMAINS, ALIASES, DEGREE_RANK, score_years, score_degree, score_language)


class Grader(Protocol):
    name: str
    def grade(self, profile_id: str, info: CandidateInfo, role: Role) -> list[CriterionGrade]: ...


def _pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)")


def _terms(skill: str) -> list[re.Pattern]:
    return [_pattern(t) for t in [skill] + ALIASES.get(skill, [])]


def _within_one_edit(a: str, b: str) -> bool:
    """True if a and b differ by <=1 substitution/insertion/deletion or one adjacent transposition."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def _hit(pats: list[re.Pattern], text: str) -> bool:
    t = text.lower()
    return any(p.search(t) for p in pats)


class RuleGrader:
    """Deterministic, transparent baseline. Cannot resolve typos or unknown paraphrases,
    which is exactly the gap an LLM grader is meant to close."""
    def __init__(self, fuzzy: bool = False):
        self.fuzzy = fuzzy
        self.name = "rule-v1+fuzzy" if fuzzy else "rule-v1"
        self._cache: dict[tuple[str, str], list[CriterionGrade]] = {}

    def grade(self, profile_id: str, info: CandidateInfo, role: Role) -> list[CriterionGrade]:
        key = (profile_id, role.id)
        if key not in self._cache:
            self._cache[key] = [self._one(c, info, role) for c in role.criteria]
        return self._cache[key]

    # ------------------------------------------------------------------ criteria
    def _one(self, c: Criterion, info: CandidateInfo, role: Role) -> CriterionGrade:
        fn = getattr(self, f"_{c.kind}")
        score, conf, ev = fn(c, info, role)
        return CriterionGrade(c.id, score, conf, ev)

    def _skill(self, c, info, role):
        pats = _terms(c.params["name"])
        listed = any(_hit(pats, s) for s in info.skills)
        if not listed and self.fuzzy:   # typo tolerance on single-word skill names
            nm = c.params["name"].lower()
            listed = len(nm) >= 5 and " " not in nm and any(_within_one_edit(s.lower(), nm) for s in info.skills)
        in_exp = [e.title for e in info.experiences if _hit(pats, e.description)]
        in_proj = any(_hit(pats, p) for p in info.projects)
        in_int = _hit(pats, info.interview_summary)
        parts = []
        if listed: parts.append("listed in skills")
        if in_exp: parts.append(f"used in '{in_exp[0]}'")
        if in_proj: parts.append("project work")
        if in_int: parts.append("confirmed in interview")
        if (in_exp or in_proj) and (listed or in_int):
            return 2, 0.9, f"{c.params['name']}: " + "; ".join(parts)
        if listed or in_exp or in_proj or in_int:
            return 1, 0.65, f"{c.params['name']}: only " + "; ".join(parts) + " (no corroboration)"
        return 0, 0.55, f"{c.params['name']}: no mention anywhere (absence of evidence, not proof of absence)"

    def _years(self, c, info, role):
        dom, need = c.params["domain"], c.params["min"]
        titles = [t.lower() for t in DOMAINS[dom]["titles"]]
        skills = [_terms(s) for s in DOMAINS[dom]["skills"]]
        y_title = sum(e.years for e in info.experiences if any(t in e.title.lower() for t in titles))
        amb = [e for e in info.experiences
               if not any(t in e.title.lower() for t in titles)
               and sum(_hit(p, e.description) for p in skills) >= 2]
        y_amb = sum(e.years for e in amb)
        obs = y_title + 0.5 * y_amb
        score = score_years(obs, need)
        note = f"{y_title} yrs under matching titles" + (f" + {y_amb} yrs ambiguous titles counted at 50%" if y_amb else "")
        if score == 2:
            conf = 0.9 if obs - need >= 2 else 0.65
        elif score == 1:
            conf = 0.6
        else:
            conf = 0.5 if y_amb else (0.9 if obs < 0.6 * need - 1 else 0.6)
        return score, conf, f"{dom}: {note}; required {need}"

    def _degree(self, c, info, role):
        if not info.education:
            return 0, 0.5, "no education listed"
        rank = max(DEGREE_RANK.get(e.degree, 0) for e in info.education)
        return score_degree(rank, c.params["min_rank"]), 0.9, f"highest degree rank {rank}"

    def _cert(self, c, info, role):
        found = any(c.params["name"].lower() == x.lower() for x in info.certifications)
        return (2, 0.95, f"{c.params['name']} listed") if found else (0, 0.6, f"{c.params['name']} not listed")

    def _language(self, c, info, role):
        lvl = info.languages.get(c.params["lang"])
        if lvl is None:
            return 0, 0.5, f"{c.params['lang']} not listed (absence of evidence)"
        return score_language(lvl, c.params["min"]), 0.95, f"{c.params['lang']} listed at {lvl}"

    def _country(self, c, info, role):
        ok = info.country in c.params["countries"]
        return (2 if ok else 0), 0.95, f"located in {info.country}"
