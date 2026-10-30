"""Two-stage pipeline: retrieval -> AI grading -> hard filters -> pay-weighted ranking -> human review.

INNOVATION (appeal-aware hard filters): a hard-filter 0 only auto-rejects when the grader is
*confident* the criterion fails. If the 0 rests on absence of evidence (confidence < rescue_conf),
the candidate goes to a `rescue_review` queue for a human instead of silently disappearing.
This directly targets the notice's weakest point: automated stages decide whether a human sees you at all.
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from .schema import Profile, Role, CandidateInfo, DecisionRecord, CriterionGrade
from .retrieval import FieldIndex, RetrievalConfig, ideal_candidate_query, Retrieved


@dataclass
class PipelineConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    pass_threshold: float = 1.2        # advisory, as in the notice
    rescue_enabled: bool = True
    rescue_conf: float = 0.70
    w_min: float = 0.30                # bounded pay weighting
    w_max: float = 0.70
    pay_lo: float = 30.0
    pay_hi: float = 150.0
    fixed_w: float | None = None       # set to override pay weighting (ablations)
    review_size: int = 20
    rescue_max: int | None = 30        # human review budget for the rescue queue

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:10]


def rank_weight(pay: float, cfg: PipelineConfig) -> float:
    """Influence of the AI match score on final rank grows with pay, bounded to [w_min, w_max]."""
    if cfg.fixed_w is not None:
        return cfg.fixed_w
    t = (pay - cfg.pay_lo) / (cfg.pay_hi - cfg.pay_lo)
    return float(np.clip(cfg.w_min + (cfg.w_max - cfg.w_min) * t, cfg.w_min, cfg.w_max))


def decide(grades: list[CriterionGrade], role: Role, cfg: PipelineConfig) -> tuple[str, float, list[str]]:
    """Return (outcome, match_score, hard_failed_ids)."""
    match = float(np.mean([g.score for g in grades]))
    hard_zero = [(g, c) for g, c in zip(grades, role.criteria) if c.hard and g.score == 0]
    if hard_zero:
        ids = [c.id for _, c in hard_zero]
        confident = any(g.confidence >= cfg.rescue_conf for g, _ in hard_zero)
        if confident or not cfg.rescue_enabled:
            return "rejected_hard_filter", match, ids
        # optimistic score: impute the uncertain zeros as 'partially met'
        opt = float(np.mean([1 if (c.hard and g.score == 0) else g.score for g, c in zip(grades, role.criteria)]))
        return ("rescue_review" if opt >= cfg.pass_threshold else "below_threshold"), opt, ids
    return ("passed" if match >= cfg.pass_threshold else "below_threshold"), match, []


@dataclass
class RunResult:
    role: Role
    retrieved: Retrieved
    records: list[DecisionRecord]
    n_pool: int

    def by(self, outcome: str) -> list[DecisionRecord]:
        return [r for r in self.records if r.outcome == outcome]

    @property
    def ranked(self) -> list[DecisionRecord]:
        return sorted(self.by("passed"), key=lambda r: -r.final_score)

    @property
    def rescue(self) -> list[DecisionRecord]:
        return sorted(self.by("rescue_review"), key=lambda r: -r.match_score)

    @property
    def shortlisted_ids(self) -> set[str]:
        return {r.profile_id for r in self.records}


class Pipeline:
    def __init__(self, profiles: list[Profile], grader, cfg: PipelineConfig | None = None,
                 log_path: str | Path | None = None):
        self.profiles, self.grader, self.cfg = profiles, grader, cfg or PipelineConfig()
        self.index = FieldIndex(self.cfg.retrieval).build(profiles)
        self.by_id = {p.id: p for p in profiles}
        self.log_path = Path(log_path) if log_path else None

    # ------------------------------------------------------------ stage 2 decision logic
    def decide(self, grades: list[CriterionGrade], role: Role) -> tuple[str, float, list[str]]:
        """Return (outcome, match_score, hard_failed_ids)."""
        return decide(grades, role, self.cfg)

    def grade_profile(self, p: Profile, role: Role) -> list[CriterionGrade]:
        return self.grader.grade(p.id, CandidateInfo.from_profile(p), role)

    # ------------------------------------------------------------ run
    def run(self, role: Role, query: dict[str, str] | None = None) -> RunResult:
        query = query or ideal_candidate_query(role)
        ret = self.index.search(query)
        w = rank_weight(role.pay_rate, self.cfg)
        lo, hi = float(ret.score.min()), float(ret.score.max())
        chash, ts = self.cfg.hash(), time.strftime("%Y-%m-%dT%H:%M:%S")
        records = []
        for rank, (i, s, ph) in enumerate(zip(ret.idx, ret.score, ret.phase)):
            p = self.profiles[i]
            grades = self.grade_profile(p, role)
            outcome, match, failed = self.decide(grades, role)
            r_norm = (s - lo) / (hi - lo + 1e-9)
            final = (1 - w) * r_norm + w * (match / 2) if outcome == "passed" else None
            did = hashlib.sha1(f"{role.id}|{p.id}|{chash}".encode()).hexdigest()[:12]
            records.append(DecisionRecord(did, role.id, p.id, outcome, float(s), ph, match, final, w, failed,
                                          [asdict(g) for g in grades], chash, self.grader.name, ts))
        if self.cfg.rescue_max is not None:   # demote rescue candidates beyond the review budget
            queue = sorted((r for r in records if r.outcome == "rescue_review"), key=lambda r: -r.match_score)
            for r in queue[self.cfg.rescue_max:]:
                r.outcome = "below_threshold"
        res = RunResult(role, ret, records, len(self.profiles))
        self._log(res)
        return res

    def _log(self, res: RunResult):
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            for r in res.records:
                f.write(json.dumps(asdict(r)) + "\n")

    def log_event(self, event: dict):
        """Append an arbitrary audit event (e.g. a human override or a contestation request)."""
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps({"event_time": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}) + "\n")
