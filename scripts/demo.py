"""A 20-second tour: run a role, look at the shortlist, then contest the decision of an expert
who was qualified but never surfaced.   python scripts/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from talentlens.synth import generate_profiles, make_roles, true_summary
from talentlens.grader import RuleGrader
from talentlens.pipeline import Pipeline, PipelineConfig
from talentlens.explain import explain_decision, to_text, request_review
from talentlens.audit import shadow_audit

P = generate_profiles(4000, seed=7)
role = make_roles()[0]
log = Path(__file__).resolve().parents[1] / "results" / "demo_audit_log.jsonl"
log.unlink(missing_ok=True)
pipe = Pipeline(P, RuleGrader(fuzzy=True), PipelineConfig(), log_path=log)
res = pipe.run(role)

print(f"ROLE: {role.title} (pay ${role.pay_rate:.0f}/h, AI-match weight in final rank = {res.records[0].rank_weight:.2f})")
print(f"Pool {len(P)} -> shortlist {len(res.records)} -> passed {len(res.ranked)} | rescue queue {len(res.rescue)} | "
      f"hard-filter rejects {len(res.by('rejected_hard_filter'))}\n")
print("Top 5 for the project lead:")
for r in res.ranked[:5]:
    p = pipe.by_id[r.profile_id]
    print(f"  {r.profile_id}  final={r.final_score:.2f} match={r.match_score:.2f}  {p.current_title}")

print("\nTop 3 in the human-review RESCUE queue (hard filter failed on thin evidence):")
for r in res.rescue[:3]:
    weak = [g["evidence"] for g in r.grades if g["score"] == 0][:1]
    print(f"  {r.profile_id}  optimistic match={r.match_score:.2f}  <- {weak[0] if weak else ''}")

# a qualified expert who never reached a human, with an uninformative headline
victim = next(p for p in P if p.id not in res.shortlisted_ids and true_summary(p, role)["qualified"]
              and p.hidden["headline_quality"] == "uninformative")
print(f"\nCONTESTING expert {victim.id} (truly qualified, headline: '{victim.headline}')\n")
print(to_text(explain_decision(pipe, role, victim.id, res)))
request_review(pipe, role, victim.id, "My profile undersells my ML experience; please review manually.")

s = shadow_audit(pipe, role, res, per_bucket=40)
print(f"\nSHADOW AUDIT: grading only {sum(b['sampled'] for b in s['buckets'])} non-shortlisted profiles estimates "
      f"{s['est_missed']:.0f} more passing experts were never shown to a human "
      f"(95% CI {s['est_missed_lo']:.0f}-{s['est_missed_hi']:.0f}); estimated Stage-1 recall {s['est_stage1_recall']:.0%}.")
print(f"\nAudit log written to {log}")
