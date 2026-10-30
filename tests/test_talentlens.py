import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from talentlens.schema import CandidateInfo, CriterionGrade, Experience, Education
from talentlens.synth import generate_profiles, make_roles, true_summary, VAGUE
from talentlens.grader import RuleGrader, _within_one_edit
from talentlens.llm import LLMGrader, parse_grades
from talentlens.pipeline import Pipeline, PipelineConfig, rank_weight
from talentlens.retrieval import RetrievalConfig, ideal_candidate_query
from talentlens.audit import shadow_audit, wilson, counterfactual_audit
from talentlens.explain import explain_decision, to_text, request_review
from talentlens.evaluate import build_truth, funnel


@pytest.fixture(scope="module")
def world():
    P = generate_profiles(1500, seed=3)
    R = make_roles()
    cfg = PipelineConfig(retrieval=RetrievalConfig(k_active=60, k_broad=100))
    return P, R, Pipeline(P, RuleGrader(fuzzy=True), cfg)


def test_generator_is_deterministic():
    a, b = generate_profiles(20, 5), generate_profiles(20, 5)
    assert [p.headline for p in a] == [p.headline for p in b]
    assert [p.skills for p in a] == [p.skills for p in b]


def test_stage2_view_hides_headline_and_ground_truth(world):
    P, _, _ = world
    d = CandidateInfo.from_profile(P[0]).to_dict()
    assert "headline" not in d and "hidden" not in d and "id" not in d


def _info(skills=(), exp_desc="", interview=""):
    return CandidateInfo(
        [Experience("Senior Data Scientist", "X", exp_desc, 2018, 2026)], [Education("MSc", "Statistics", "Y")],
        [], [], [], list(skills), {"English": "C1"}, "Pisa", "Italy", interview)


def test_rule_grader_skill_levels():
    role = make_roles()[0]
    g = RuleGrader()
    full = g.grade("a", _info(["Python"], "Built models using Python daily."), role)[0]
    listed = g.grade("b", _info(["Python"]), role)[0]
    none = g.grade("c", _info([]), role)[0]
    assert (full.score, listed.score, none.score) == (2, 1, 0)
    assert none.confidence < 0.7 <= full.confidence          # absence of evidence -> low confidence


def test_typo_tolerance():
    assert _within_one_edit("pyhton", "python") and not _within_one_edit("java", "python")
    role = make_roles()[0]
    assert RuleGrader().grade("t1", _info(["Pyhton"]), role)[0].score == 0
    assert RuleGrader(fuzzy=True).grade("t2", _info(["Pyhton"]), role)[0].score == 1


def _grades(role, low_conf_hard_zero: bool):
    out = []
    for c in role.criteria:
        if c.hard:
            out.append(CriterionGrade(c.id, 0, 0.5 if low_conf_hard_zero else 0.95, "x") if c.id == "skill:Python"
                       else CriterionGrade(c.id, 2, 0.9, "x"))
        else:
            out.append(CriterionGrade(c.id, 2, 0.9, "x"))
    return out


def test_hard_filter_rescue_logic(world):
    _, R, pipe = world
    role = R[0]
    assert pipe.decide(_grades(role, True), role)[0] == "rescue_review"
    assert pipe.decide(_grades(role, False), role)[0] == "rejected_hard_filter"
    off = Pipeline.__new__(Pipeline)
    off.cfg = PipelineConfig(rescue_enabled=False)
    assert off.decide(_grades(role, True), role)[0] == "rejected_hard_filter"


def test_rank_weight_bounded_and_monotone():
    cfg = PipelineConfig()
    ws = [rank_weight(p, cfg) for p in (10, 30, 60, 100, 150, 400)]
    assert ws == sorted(ws) and min(ws) == cfg.w_min and max(ws) == cfg.w_max


def test_retrieval_two_phase(world):
    P, R, pipe = world
    ret = pipe.index.search(ideal_candidate_query(R[0]))
    assert len(ret.idx) == len(set(ret.idx)) == 160
    act = [i for i, ph in zip(ret.idx, ret.phase) if ph == "active"]
    assert all(pipe.index.active[i] for i in act)


def test_pipeline_end_to_end_beats_random(world):
    P, R, pipe = world
    by = {p.id: p for p in P}
    truth = build_truth(P, R)
    f = funnel(pipe.run(R[0]), truth[R[0].id], by)
    base_rate = len(truth[R[0].id]["qualified"]) / len(P)
    assert f["stage1_precision"] > 3 * base_rate
    assert f["precision_at_k"] > 0.8


def test_adaptive_reduces_headline_sensitivity(world):
    P, R, _ = world
    q = ideal_candidate_query(R[0])
    drops = {}
    for mode in ("fielded", "adaptive"):
        pipe = Pipeline(P, RuleGrader(), PipelineConfig(retrieval=RetrievalConfig(mode=mode)))
        informative = [p for p in P if p.hidden["headline_quality"] == "informative"][:200]
        import copy
        vague = []
        for p in informative:
            c = copy.deepcopy(p); c.headline = VAGUE[0]; vague.append(c)
        drops[mode] = float((pipe.index.score_variants(informative, q) - pipe.index.score_variants(vague, q)).mean())
    assert drops["adaptive"] < 0.5 * drops["fielded"]


def test_counterfactual_audit_runs(world):
    P, R, pipe = world
    truth = build_truth(P, R)
    res = counterfactual_audit(pipe, R[0], truth[R[0].id]["qualified"], "institution: local -> elite")
    assert res and res["d_match"] == 0.0           # rule grader never sees institution


def test_shadow_audit_and_wilson(world):
    P, R, pipe = world
    run = pipe.run(R[1])
    s = shadow_audit(pipe, R[1], run, per_bucket=15, validate=True)
    assert s["est_missed_lo"] <= s["est_missed"] <= s["est_missed_hi"]
    lo, hi = wilson(0, 30)
    assert lo == 0.0 and 0.05 < hi < 0.2


def test_explain_never_shortlisted_expert(world, tmp_path):
    P, R, pipe = world
    pipe.log_path = tmp_path / "audit.jsonl"
    res = pipe.run(R[0])
    out = next(p for p in P if p.id not in res.shortlisted_ids)
    e = explain_decision(pipe, R[0], out.id, res)
    assert e["stage_eliminated"] == "stage1_retrieval" and "what_if_clear_headline" in e
    assert "Eliminated at Stage 1" in to_text(e)
    request_review(pipe, R[0], out.id, "please re-check my profile")
    lines = [json.loads(l) for l in pipe.log_path.read_text().splitlines()]
    assert any(l.get("event") == "human_review_requested" for l in lines)
    assert any(l.get("outcome") for l in lines)


class _FakeClient:
    def __init__(self, text):
        self.calls, self._text = 0, text
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._text)])


def test_llm_grader_parses_caches_and_handles_gaps():
    role = make_roles()[0]
    ids = [c.id for c in role.criteria]
    payload = [{"criterion_id": i, "score": 2, "confidence": 0.9, "evidence": "e"} for i in ids[:-1]]  # last one missing
    client = _FakeClient("```json\n" + json.dumps(payload) + "\n```")
    g = LLMGrader(client=client, model="fake")
    info = _info(["Python"])
    out = g.grade("p1", info, role)
    g.grade("p1", info, role)
    assert client.calls == 1                                   # cached
    assert [x.score for x in out[:-1]] == [2] * (len(ids) - 1)
    assert out[-1].score == 0 and out[-1].confidence < 0.5     # missing -> low-confidence zero -> human review
    assert parse_grades("not json", role)[0].confidence < 0.5
