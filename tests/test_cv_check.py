import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from talentlens.schema import CURRENT_YEAR
from talentlens.synth import generate_profiles, make_roles
from talentlens.parse import parse_cv, parse_jd, read_document, make_criterion, criterion_value
from talentlens.cv_check import CVChecker, PoolSimulator, suggested_headline, to_text

SAMPLE_CV = (ROOT / "examples" / "sample_cv.txt").read_text(encoding="utf-8")
SAMPLE_JD = (ROOT / "examples" / "sample_jd.txt").read_text(encoding="utf-8")


def test_parse_jd_sample():
    role, notes = parse_jd(SAMPLE_JD)
    crit = {c.id: c.hard for c in role.criteria}
    assert role.domain == "data_science" and role.pay_rate == 120.0 and not notes
    assert crit["skill:Python"] and crit["skill:PyTorch"] and not crit["skill:Spark"]
    assert crit["years:data_science"] and crit["degree:2"] and crit["lang:English:C1"]
    assert crit["cert:AWS ML Specialty"] is False
    assert "skill:AWS" not in crit and "skill:Statistics" not in crit    # cert name and field of study are not skills


def test_parse_cv_sample():
    p, notes = parse_cv(SAMPLE_CV)
    assert p.headline == "Passionate problem solver | Lifelong learner"     # name line skipped
    assert [e.title for e in p.experiences] == ["Senior Data Scientist", "Data Scientist", "Data Analyst"]
    assert p.experiences[0].company == "Helios Analytics" and p.experiences[0].end_year == CURRENT_YEAR
    assert (p.education[0].degree, p.education[0].field, p.education[0].institution) == ("MSc", "Statistics", "University of Pisa")
    assert p.languages == {"Italian": "native", "English": "C1"} and (p.city, p.country) == ("Pisa", "Italy")
    assert p.skills[-1] == "Statistics" and len(p.skills) == 19 and not notes


def test_parse_other_layouts():
    cv = """Backend Engineer | Go | Kubernetes
Manchester, United Kingdom
WORK HISTORY
Kestrel Cloud, Manchester
Platform Engineer
Mar 2019 – Present
• Run Kubernetes clusters
Education
Master of Science in Computer Science, Coastal University
Languages
Fluent in English and Spanish
"""
    p, _ = parse_cv(cv)
    assert p.headline.startswith("Backend Engineer") and p.country == "UK"
    assert (p.experiences[0].title, p.experiences[0].company, p.experiences[0].start_year) == ("Platform Engineer", "Kestrel Cloud", 2019)
    assert p.education[0].degree == "MSc" and p.education[0].field == "Computer Science"
    assert p.languages == {"English": "C1", "Spanish": "C1"}

    role, _ = parse_jd("Growth Analyst\nOwn our go-to-market reporting in MS Excel and SQL.\nMust be based in the EU.")
    ids = {c.id for c in role.criteria}
    assert {"skill:Excel", "skill:SQL", "country:EU"} <= ids
    assert "skill:Go" not in ids and not any(i.startswith("degree") for i in ids)


def test_make_criterion_round_trips_builtin_roles():
    for role in make_roles():
        for c in role.criteria:
            assert make_criterion(c.kind, criterion_value(c), c.hard, role.domain).id == c.id
    with pytest.raises(ValueError):
        make_criterion("language", "English:fluent")


def test_read_document_text_with_bom():
    assert read_document("cv.txt", "﻿Hello".encode("utf-8")) == "Hello"


def test_check_report_on_sample():
    cv, role = parse_cv(SAMPLE_CV)[0], parse_jd(SAMPLE_JD)[0]
    rep = CVChecker().check(cv, role)
    checks = {c["check"]: c["score"] for c in rep["visibility"]["checks"]}
    assert rep["requirements"]["outcome"] == "passed" and rep["verdict"] == "Hard to find"
    assert checks["headline"] == 0 and checks["skills list order"] == 0 and checks["location"] is None   # no location rule
    assert rep["recommendations"][0]["area"] == "Visibility" and "READINESS" in to_text(rep)

    fixed = dataclasses.replace(cv, headline=suggested_headline(cv, role))       # own title + skills, nothing new claimed
    after = CVChecker().check(fixed, role)
    assert after["visibility"]["score"] > rep["visibility"]["score"] and after["readiness"] > rep["readiness"]
    assert after["requirements"] == rep["requirements"]                          # Stage 2 never sees the headline


def test_edited_role_is_regraded():
    cv, _ = parse_cv(SAMPLE_CV)
    role, _ = parse_jd(SAMPLE_JD)
    checker = CVChecker()
    before = checker.check(cv, role)["requirements"]["outcome"]
    harder = dataclasses.replace(role, criteria=role.criteria + [make_criterion("cert", "CKA", True)])   # same role.id
    after = checker.check(cv, harder)
    assert before == "passed" and after["requirements"]["outcome"] != "passed"
    assert after["readiness"] <= 50 and after["recommendations"][0]["priority"] == "high"


@pytest.fixture(scope="module")
def sim():
    return PoolSimulator(profiles=generate_profiles(1500, seed=3))


def test_simulation_matches_pipeline_for_a_pool_member(sim):
    role = make_roles()[0]
    res = sim.pipe.run(role)
    rec = next(r for r in res.ranked if r.retrieval_phase == "active")
    out = sim.simulate(sim.pipe.by_id[rec.profile_id], role)
    assert out["shortlisted"] and out["score"] == pytest.approx(rec.retrieval_score) and out["rank"]


def test_simulation_rewards_the_visibility_fixes(sim):
    out = sim.simulate(parse_cv(SAMPLE_CV)[0], parse_jd(SAMPLE_JD)[0])
    head = next(w for w in out["what_ifs"] if w["change"] == "clear headline")
    assert head["percentile"] > out["percentile"]
