"""Synthetic experts + roles with *hidden ground truth*.

Design choices that make the evaluation meaningful:
  * Every profile is generated from latent facts (true skill levels, true years per domain,
    true degree rank, ...). The observable profile is a noisy rendering of those facts
    (aliases, typos, omitted fields, inflated skills, vague headlines, creative job titles).
  * Institution prestige and headline quality are generated INDEPENDENTLY of skill, so under
    the ground truth neither should matter. Any sensitivity to them is a bias finding.
  * No names, photos or other identifiers exist anywhere (privacy by design).
"""
from __future__ import annotations
import random
from .schema import (Profile, Experience, Education, Role, Criterion, CURRENT_YEAR)
from .taxonomy import (DOMAINS, ALIASES, LEVEL_ORDER, DEGREE_RANK, EU_COUNTRIES,
                       score_years, score_degree, score_language)

CITIES = [("London", "UK"), ("Berlin", "Germany"), ("Pisa", "Italy"), ("Madrid", "Spain"),
          ("Warsaw", "Poland"), ("Paris", "France"), ("Lagos", "Nigeria"), ("Nairobi", "Kenya"),
          ("Bangalore", "India"), ("Sao Paulo", "Brazil"), ("Toronto", "Canada"), ("Austin", "USA"),
          ("Manila", "Philippines"), ("Cairo", "Egypt"), ("Singapore", "Singapore"), ("Sydney", "Australia")]
ENGLISH_NATIVE = {"UK", "USA", "Canada", "Australia"}
ENGLISH_STRONG = {"India", "Nigeria", "Kenya", "Philippines", "Singapore"}

ELITE = ["Stanford University", "MIT", "University of Oxford", "ETH Zurich", "Harvard University", "University of Cambridge"]
MID = ["University of Pisa", "University of Warsaw", "University of Toronto", "University of Lagos", "Technical University of Berlin", "University of Madrid"]
LOCAL = ["Regional State University", "City Technical College", "Metropolitan University", "Valley Institute of Technology", "Coastal University"]
INSTITUTION_TIER = {**{n: "elite" for n in ELITE}, **{n: "mid" for n in MID}, **{n: "local" for n in LOCAL}}

COMPANIES = ["Northwind Labs", "Helios Analytics", "BlueFern Systems", "Quartz Health", "Lumen Capital", "Orbit Commerce",
             "Atlas Legal Group", "Pinecone Media", "Summit Biotech", "Kestrel Cloud", "Marlow & Finch", "Vertex Logistics",
             "Ardent Pharma", "Cobalt Retail", "Nimbus Software", "Harbor Advisory", "Sable Robotics", "Verity Insurance"]
CREATIVE_TITLES = ["Data Wizard", "Code Ninja", "Growth Hacker", "Numbers Guru", "Contracts Whisperer", "Pixel Alchemist"]
VAGUE = ["Passionate professional open to new opportunities", "Results-driven team player", "Lifelong learner | Dreamer | Builder",
         "Experienced professional seeking growth"]
BUZZ = ["Visionary thought leader disrupting the future", "Innovation catalyst and synergy architect",
        "Digital transformation evangelist", "Game changer | Value creator | Storyteller"]


def _typo(s: str, r: random.Random) -> str:
    if len(s) < 5 or " " in s:
        return s
    i = r.randrange(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def _surface(skill: str, r: random.Random, p_alias: float = 0.2) -> str:
    if skill in ALIASES and r.random() < p_alias:
        return r.choice(ALIASES[skill])
    return skill


def _prefix(cum_years: int) -> str:
    return "Junior " if cum_years < 3 else "" if cum_years < 6 else "Senior " if cum_years < 10 else "Lead " if cum_years < 15 else "Principal "


def generate_profile(i: int, r: random.Random) -> Profile:
    dom = r.choice(list(DOMAINS))
    D = DOMAINS[dom]
    years = min(25, 1 + int(r.gammavariate(2.5, 3.2)))
    prior = r.choice([d for d in DOMAINS if d != dom]) if years >= 4 and r.random() < 0.15 else None

    # ---- career timeline (oldest first) ----
    n_pos = min(max(1, min(5, 1 + years // 4 + r.randint(0, 1))), years)
    cuts = sorted(r.sample(range(1, years), n_pos - 1)) if n_pos > 1 else []
    bounds = [0] + cuts + [years]
    pos = []
    for k in range(n_pos):
        d = prior if (prior and bounds[k] < 0.4 * years and k < n_pos - 1) else dom
        pos.append(dict(dom=d, start=CURRENT_YEAR - years + bounds[k], end=CURRENT_YEAR - years + bounds[k + 1], cum=bounds[k + 1]))
    years_by_domain: dict[str, int] = {}
    for q in pos:
        years_by_domain[q["dom"]] = years_by_domain.get(q["dom"], 0) + (q["end"] - q["start"])

    # ---- latent skill levels ----
    levels: dict[str, int] = {}
    for d, y in years_by_domain.items():
        p_strong = min(0.9, max(0.2, 0.2 + 0.04 * y)) if d == dom else min(0.4, 0.1 + 0.03 * y)
        for s in DOMAINS[d]["skills"]:
            roll = r.random()
            lv = 2 if roll < p_strong else 1 if roll < p_strong + 0.35 else 0
            levels[s] = max(levels.get(s, 0), lv)

    # ---- observed experiences (most recent first) ----
    exps = []
    for q in pos:
        Dq = DOMAINS[q["dom"]]
        strong = [s for s in Dq["skills"] if levels.get(s, 0) == 2]
        title = r.choice(CREATIVE_TITLES) if r.random() < 0.08 else _prefix(q["cum"]) + r.choice(Dq["titles"])
        task = r.choice(Dq["tasks"])
        if strong:
            mention = r.sample(strong, min(len(strong), r.randint(1, 3)))
            names = [_surface(s, r) for s in mention]
            desc = r.choice([
                f"Led work on {task} using {', '.join(names)}; partnered with product and operations to ship measurable improvements.",
                f"Designed and delivered {task} with {' and '.join(names[:2])}, mentoring colleagues and documenting best practices.",
                f"Owned {task} end to end, applying {', '.join(names)} to improve reliability and business outcomes.",
            ])
        else:
            desc = f"Contributed to {task}; supported senior colleagues on delivery, reporting and stakeholder updates."
        exps.append(Experience(title, r.choice(COMPANIES), desc, q["start"], q["end"]))
    exps.reverse()

    # ---- education ----
    rank = r.choices([1, 2, 3], weights=[0.45, 0.38, 0.17])[0]
    ladder = {"legal": ["LLB", "LLM", "JD"], "finance": ["BSc", r.choice(["MSc", "MBA"]), "PhD"]}.get(dom, ["BSc", "MSc", "PhD"])
    edu = []
    for lv in range(rank, 0, -1):
        tier = r.choices(["elite", "mid", "local"], weights=[0.12, 0.38, 0.5])[0]
        inst = r.choice({"elite": ELITE, "mid": MID, "local": LOCAL}[tier])
        edu.append(Education(ladder[lv - 1], r.choice(D["fields"]), inst))
    top_tier = INSTITUTION_TIER[edu[0].institution]

    # ---- observed skills (noisy rendering of levels) ----
    skills = []
    for s, lv in levels.items():
        listed = r.random() < (0.88 if lv == 2 else 0.85) if lv > 0 else r.random() < 0.05  # level 0 -> inflation
        if listed:
            sk = _surface(s, r)
            skills.append(_typo(sk, r) if r.random() < 0.05 else sk)
    r.shuffle(skills)

    certs = [c for c in D["certs"] if r.random() < min(0.6, 0.15 + 0.02 * years)]
    city, country = r.choice(CITIES)
    if country in ENGLISH_NATIVE:
        eng = "native"
    elif country in ENGLISH_STRONG:
        eng = r.choices(["C2", "C1", "B2", "B1"], weights=[0.3, 0.4, 0.25, 0.05])[0]
    else:
        eng = r.choices(["C1", "B2", "B1", "A2"], weights=[0.2, 0.3, 0.3, 0.2])[0]
    langs = {} if r.random() < 0.10 else {"English": eng}          # 10% omit English -> absence of evidence
    if r.random() < 0.4:
        langs[r.choice(["Spanish", "French", "German", "Italian", "Portuguese"])] = r.choice(LEVEL_ORDER[2:])

    top_strong = [s for s in D["skills"] if levels.get(s, 0) == 2]
    q = r.random()
    if q < 0.6:
        hq, headline = "informative", f"{exps[0].title}" + (f" | {' | '.join(top_strong[:2])}" if top_strong else "")
    else:
        hq, headline = "uninformative", r.choice(VAGUE if q < 0.85 else BUZZ)

    mention = [s for s in top_strong if r.random() < 0.6][:2]
    interview = (f"AI interview summary: candidate showed {r.choice(['solid', 'strong', 'confident'])} command of "
                 f"{' and '.join(mention) if mention else 'core concepts'}; reasoning about {r.choice(D['tasks'])} was "
                 f"{r.choice(['well structured', 'uneven', 'thorough'])}.")
    awards = [r.choice(["Employee of the Year", "Dean's List", "Industry Innovation Award"])] if r.random() < 0.15 else []
    projects = [f"Open-source tooling around {r.choice(top_strong)}"] if top_strong and r.random() < 0.4 else []

    hidden = dict(domain=dom, prior_domain=prior, years_by_domain=years_by_domain, skill_levels=levels,
                  degree_rank=rank, certs=certs, english=eng, country=country, headline_quality=hq, top_tier=top_tier)
    return Profile(
        id=f"P{i:05d}", headline=headline, current_title=exps[0].title, current_company=exps[0].company,
        city=city, country=country, experiences=exps, education=edu, skills=skills, certifications=certs,
        languages=langs, awards=awards, projects=projects, interview_summary=interview,
        last_active_days=int(r.expovariate(1 / 60)), hidden=hidden)


def generate_profiles(n: int = 4000, seed: int = 7) -> list[Profile]:
    r = random.Random(seed)
    return [generate_profile(i, r) for i in range(n)]


# ---------------------------------------------------------------- roles
def _skill(n, hard=False): return Criterion(f"skill:{n}", f"Hands-on proficiency in {n}", "skill", {"name": n}, hard)
def _years(d, n, hard=False): return Criterion(f"years:{d}", f"At least {n} years of professional experience in {d.replace('_', ' ')}", "years", {"domain": d, "min": n}, hard)
def _degree(m, hard=False): return Criterion(f"degree:{m}", f"Degree level rank >= {m} (1=BSc, 2=MSc, 3=PhD)", "degree", {"min_rank": m}, hard)
def _eng(m="B2", hard=True): return Criterion(f"lang:English:{m}", f"English proficiency at least {m}", "language", {"lang": "English", "min": m}, hard)
def _cert(n): return Criterion(f"cert:{n}", f"Holds the {n} certification", "cert", {"name": n}, False)
def _country(cs): return Criterion("country:EU", "Based in the EU (data residency requirement)", "country", {"countries": sorted(cs)}, True)


def _role(i, title, target, dom, pay, crit, desc):
    return Role(f"R{i:02d}", title, target, dom, desc, pay, crit)


def make_roles() -> list[Role]:
    return [
        _role(1, "Senior ML Engineer, LLM Evaluation", "Machine Learning Engineer", "data_science", 140,
              [_skill("Python", True), _skill("PyTorch"), _skill("NLP"), _skill("Deep Learning"), _years("data_science", 5, True), _eng(), _degree(2)],
              "Design evaluation harnesses for large language models."),
        _role(2, "Data Analyst (contract)", "Data Analyst", "data_science", 35,
              [_skill("SQL", True), _skill("Python"), _skill("Statistics"), _years("data_science", 2), _eng()],
              "Build dashboards and ad-hoc analyses for a retail client."),
        _role(3, "Experimentation Scientist", "Applied Scientist", "data_science", 95,
              [_skill("Experiment Design", True), _skill("Statistics"), _skill("SQL"), _years("data_science", 4, True), _degree(2)],
              "Own A/B testing methodology for a marketplace."),
        _role(4, "Platform Engineer", "Platform Engineer", "backend", 110,
              [_skill("Kubernetes", True), _skill("Docker", True), _skill("Go"), _skill("AWS"), _years("backend", 5, True), _cert("CKA")],
              "Run and scale a multi-region Kubernetes platform."),
        _role(5, "API Developer", "API Developer", "backend", 55,
              [_skill("Python"), _skill("REST APIs", True), _skill("PostgreSQL"), _years("backend", 3), _eng()],
              "Build public REST APIs for a fintech start-up."),
        _role(6, "React Engineer", "Frontend Engineer", "frontend", 90,
              [_skill("React", True), _skill("TypeScript"), _skill("CSS"), _skill("Accessibility"), _years("frontend", 4, True)],
              "Ship accessible React interfaces for a design-system team."),
        _role(7, "Web Performance Consultant", "Web Developer", "frontend", 130,
              [_skill("Web Performance", True), _skill("JavaScript"), _skill("TypeScript"), _years("frontend", 6, True)],
              "Audit and speed up high-traffic e-commerce front ends."),
        _role(8, "Valuation Analyst", "Investment Analyst", "finance", 100,
              [_skill("Valuation", True), _skill("Financial Modeling"), _skill("Excel"), _years("finance", 3, True), _cert("CFA")],
              "Support M&A valuation work for a boutique advisory."),
        _role(9, "FP&A Freelancer", "FP&A Manager", "finance", 40,
              [_skill("Excel", True), _skill("Forecasting"), _years("finance", 3), _eng()],
              "Maintain rolling forecasts for a SaaS company."),
        _role(10, "GDPR Counsel (EU)", "Privacy Counsel", "legal", 125,
              [_skill("GDPR", True), _skill("Privacy Law"), _skill("Contract Drafting"), _years("legal", 5, True), _country(EU_COUNTRIES), _degree(2)],
              "Advise on data protection compliance for EU programmes."),
        _role(11, "Clinical Biostatistician", "Biostatistician", "clinical", 105,
              [_skill("Biostatistics", True), _skill("SAS"), _skill("CDISC"), _years("clinical", 4, True), _degree(2, True)],
              "Write statistical analysis plans for phase II trials."),
        _role(12, "Growth Lead", "Growth Marketer", "marketing", 70,
              [_skill("Paid Acquisition", True), _skill("A/B Testing"), _skill("SQL"), _skill("Attribution Modeling"), _years("marketing", 4, True)],
              "Scale paid acquisition for a consumer subscription app."),
    ]


# ---------------------------------------------------------------- ground truth
def true_grades(p: Profile, role: Role) -> list[int]:
    h = p.hidden
    out = []
    for c in role.criteria:
        q = c.params
        if c.kind == "skill":
            out.append(h["skill_levels"].get(q["name"], 0))
        elif c.kind == "years":
            out.append(score_years(h["years_by_domain"].get(q["domain"], 0), q["min"]))
        elif c.kind == "degree":
            out.append(score_degree(h["degree_rank"], q["min_rank"]))
        elif c.kind == "cert":
            out.append(2 if q["name"] in h["certs"] else 0)
        elif c.kind == "language":
            out.append(score_language(h["english"], q["min"]))
        elif c.kind == "country":
            out.append(2 if h["country"] in q["countries"] else 0)
    return out


def true_summary(p: Profile, role: Role, threshold: float = 1.2) -> dict:
    g = true_grades(p, role)
    hard_ok = all(s > 0 for s, c in zip(g, role.criteria) if c.hard)
    mean = sum(g) / len(g)
    return dict(mean=mean, hard_ok=hard_ok, relevance=mean if hard_ok else 0.0, qualified=hard_ok and mean >= threshold)
