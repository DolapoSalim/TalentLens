"""Optional LLM-backed components (require `pip install anthropic` and ANTHROPIC_API_KEY).

Both classes accept an injected `client` so they can be unit-tested with a fake and so you can
swap providers. Model name is configurable via TALENTLENS_MODEL.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import statistics
from .schema import CandidateInfo, Role, CriterionGrade

DEFAULT_MODEL = os.environ.get("TALENTLENS_MODEL", "claude-sonnet-5-5")

GRADER_SYSTEM = """You are a rigorous, fair evaluator of expert candidates for project work.
Grade the candidate against EACH criterion using only the candidate data provided.
Scale: 0 = not met, 1 = partially met, 2 = fully met.
Rules:
- Cite concrete evidence from the data. Do not invent facts.
- If there is no evidence either way, give 0 with LOW confidence (< 0.6): absence of evidence is not proof of absence.
- Tolerate typos and paraphrases (e.g. 'Pyhton' = Python).
- Never use institution prestige, country, or writing style as a proxy for skill unless the criterion is explicitly about it.
Return ONLY a JSON array: [{"criterion_id": str, "score": 0|1|2, "confidence": 0..1, "evidence": str}, ...]"""


def parse_grades(text: str, role: Role) -> list[CriterionGrade]:
    """Robustly parse model output; missing criteria become low-confidence zeros (-> human review)."""
    m = re.search(r"\[.*\]", text, flags=re.S)
    raw = []
    if m:
        try:
            raw = json.loads(m.group(0))
        except json.JSONDecodeError:
            raw = []
    by_id = {str(x.get("criterion_id")): x for x in raw if isinstance(x, dict)}
    out = []
    for c in role.criteria:
        x = by_id.get(c.id)
        if x is None:
            out.append(CriterionGrade(c.id, 0, 0.3, "model returned no grade for this criterion"))
            continue
        try:
            score = min(2, max(0, int(x.get("score", 0))))
            conf = min(1.0, max(0.0, float(x.get("confidence", 0.5))))
        except (TypeError, ValueError):
            score, conf = 0, 0.3
        out.append(CriterionGrade(c.id, score, conf, str(x.get("evidence", ""))[:300]))
    return out


class LLMGrader:
    name = "llm"

    def __init__(self, client=None, model: str = DEFAULT_MODEL, n_samples: int = 1):
        if client is None:
            import anthropic  # lazy import so the rest of the package works without it
            client = anthropic.Anthropic()
        self.client, self.model, self.n = client, model, n_samples
        self._cache: dict[str, list[CriterionGrade]] = {}
        self.name = f"llm:{model}" + (f"x{n_samples}" if n_samples > 1 else "")

    def _prompt(self, info: CandidateInfo, role: Role) -> str:
        crit = [{"criterion_id": c.id, "criterion": c.text, "hard_filter": c.hard} for c in role.criteria]
        return (f"ROLE: {role.title}\n{role.description}\n\nCRITERIA:\n{json.dumps(crit, indent=1)}\n\n"
                f"CANDIDATE (structured, no raw resume):\n{json.dumps(info.to_dict(), indent=1, default=str)}")

    def _call(self, prompt: str) -> str:
        kw = {"temperature": 0.7} if self.n > 1 else {}
        msg = self.client.messages.create(model=self.model, max_tokens=2000, system=GRADER_SYSTEM,
                                          messages=[{"role": "user", "content": prompt}], **kw)
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    def grade(self, profile_id: str, info: CandidateInfo, role: Role) -> list[CriterionGrade]:
        prompt = self._prompt(info, role)
        key = hashlib.sha256((self.model + str(self.n) + prompt).encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        runs = [parse_grades(self._call(prompt), role) for _ in range(self.n)]
        merged = []
        for j, c in enumerate(role.criteria):
            scores = [r[j].score for r in runs]
            conf = statistics.mean(r[j].confidence for r in runs)
            if self.n > 1:  # self-consistency: disagreement across samples lowers confidence
                conf *= 1 - min(1.0, statistics.pstdev(scores))
            merged.append(CriterionGrade(c.id, int(round(statistics.median(scores))), round(conf, 3), runs[0][j].evidence))
        self._cache[key] = merged
        return merged


class LLMQueryWriter:
    """HyDE-style: ask the model to write the 'ideal candidate' profile fields for a role."""
    def __init__(self, client=None, model: str = DEFAULT_MODEL):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model = client, model

    def write(self, role: Role) -> dict[str, str]:
        prompt = ("Write the ideal candidate's profile for this role as JSON with string keys "
                  "headline, current_role, experience, education, skills, location. Return ONLY JSON.\n"
                  f"ROLE: {role.title}\n{role.description}\n"
                  f"CRITERIA: {[c.text for c in role.criteria]}")
        msg = self.client.messages.create(model=self.model, max_tokens=800,
                                          messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}", text, flags=re.S)
        return {k: str(v) for k, v in json.loads(m.group(0)).items()} if m else {}
