"""Check how ready a CV is for a role, using the reverse-engineered pipeline as the yardstick.

    python scripts/check_cv.py --cv examples/sample_cv.txt --jd examples/sample_jd.txt
    python scripts/check_cv.py --cv my_cv.pdf --role R01          # one of the built-in sample roles
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from talentlens.synth import make_roles
from talentlens.parse import parse_cv, parse_jd, read_document
from talentlens.cv_check import CVChecker, CheckConfig, to_text


def read(path: str) -> str:
    p = Path(path)
    return read_document(p.name, p.read_bytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True, help=".pdf, .docx, .txt or .md")
    job = ap.add_mutually_exclusive_group(required=True)
    job.add_argument("--jd", help="job description file")
    job.add_argument("--role", help="built-in sample role id, e.g. R01")
    ap.add_argument("--pay", type=float, help="hourly rate in USD (overrides the JD)")
    ap.add_argument("--mode", choices=["fielded", "adaptive"], default="fielded",
                    help="fielded = the platform as disclosed; adaptive = TalentLens' headline fix")
    ap.add_argument("--pool", type=int, default=4000, help="synthetic benchmark pool size")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    a = ap.parse_args()

    profile, cv_notes = parse_cv(read(a.cv))
    if a.jd:
        role, jd_notes = parse_jd(read(a.jd), pay_rate=a.pay)
    else:
        role, jd_notes = next(r for r in make_roles() if r.id == a.role), []
    rep = CVChecker(CheckConfig(pool_size=a.pool, mode=a.mode)).check(profile, role)
    rep["parser_notes"] = dict(cv=cv_notes, job=jd_notes)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print(to_text(rep))
        for note in cv_notes + jd_notes:
            print(f"  note: {note}")
