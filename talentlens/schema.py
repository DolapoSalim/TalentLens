"""Data structures. Profiles carry a `hidden` ground-truth dict that the pipeline never reads."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

CURRENT_YEAR = 2026


@dataclass
class Experience:
    title: str
    company: str
    description: str
    start_year: int
    end_year: int

    @property
    def years(self) -> int:
        return max(0, self.end_year - self.start_year)


@dataclass
class Education:
    degree: str
    field: str
    institution: str


@dataclass
class Profile:
    id: str
    headline: str
    current_title: str
    current_company: str
    city: str
    country: str
    experiences: list[Experience]          # most recent first
    education: list[Education]             # highest degree first
    skills: list[str]
    certifications: list[str]
    languages: dict[str, str]
    awards: list[str]
    projects: list[str]
    interview_summary: str
    last_active_days: int
    hidden: dict[str, Any] = field(default_factory=dict, repr=False)  # GROUND TRUTH, evaluation only


@dataclass
class CandidateInfo:
    """Stage-2 view of a profile. Mirrors the notice's 'CandidateInfo' object:
    no raw resume text, no headline, no name/photo/identifiers."""
    experiences: list[Experience]
    education: list[Education]
    awards: list[str]
    certifications: list[str]
    projects: list[str]
    skills: list[str]
    languages: dict[str, str]
    city: str
    country: str
    interview_summary: str

    @classmethod
    def from_profile(cls, p: Profile) -> "CandidateInfo":
        return cls(p.experiences, p.education, p.awards, p.certifications, p.projects,
                   p.skills, p.languages, p.city, p.country, p.interview_summary)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Criterion:
    id: str
    text: str
    kind: str                 # skill | years | degree | cert | language | country
    params: dict
    hard: bool = False


@dataclass
class Role:
    id: str
    title: str
    target_title: str
    domain: str
    description: str
    pay_rate: float           # USD / hour
    criteria: list[Criterion]


@dataclass
class CriterionGrade:
    criterion_id: str
    score: int                # 0 not met, 1 partially, 2 fully
    confidence: float         # 0..1; low when the score rests on absence of evidence
    evidence: str


@dataclass
class DecisionRecord:
    decision_id: str
    role_id: str
    profile_id: str
    outcome: str              # passed | rescue_review | rejected_hard_filter | below_threshold
    retrieval_score: float
    retrieval_phase: str
    match_score: float
    final_score: float | None
    rank_weight: float
    hard_failed: list[str]
    grades: list[dict]
    config_hash: str
    grader: str
    timestamp: str
