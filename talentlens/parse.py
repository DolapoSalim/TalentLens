"""Turn free text (a candidate's CV, a job description) into the pipeline's own objects.

Rule-based and deterministic. It recognises what `taxonomy` knows (skills and aliases, job titles,
degrees, certifications, CEFR levels) plus a few generic patterns (date ranges, "5+ years", section
headings). Anything it cannot recognise is simply not seen -- which is also how a fielded index
behaves -- so both parsers return notes saying what was guessed or missing instead of silent defaults.
"""
from __future__ import annotations
import hashlib
import io
import re
from pathlib import Path
from .schema import Profile, Experience, Education, Role, Criterion, CURRENT_YEAR
from .taxonomy import DOMAINS, DEGREE_RANK, LEVEL_ORDER
from .grader import _pattern, _terms, _hit
from .synth import CITIES

EU27 = {"Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France",
        "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands",
        "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"}
COUNTRY_ALIASES = {"United Kingdom": "UK", "Great Britain": "UK", "England": "UK", "Scotland": "UK", "Wales": "UK",
                   "Northern Ireland": "UK", "United States of America": "USA", "United States": "USA", "U.S.": "USA",
                   "US": "USA", "Czech Republic": "Czechia", "The Netherlands": "Netherlands", "Holland": "Netherlands"}
OTHER_COUNTRIES = {"UK", "USA", "Canada", "Australia", "India", "Nigeria", "Kenya", "Brazil", "Philippines", "Egypt",
                   "Singapore", "Switzerland", "Norway", "South Africa", "Ghana", "Mexico", "Japan", "China", "Israel",
                   "United Arab Emirates", "New Zealand", "Pakistan", "Turkey", "Argentina"}
_COUNTRY_NAMES = sorted(set(COUNTRY_ALIASES) | EU27 | OTHER_COUNTRIES, key=len, reverse=True)
COUNTRY_RE = re.compile(r"(?<![\w.])(" + "|".join(map(re.escape, _COUNTRY_NAMES)) + r")(?!\w)")

LANGS = ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Polish", "Arabic", "Mandarin",
         "Chinese", "Japanese", "Korean", "Hindi", "Russian", "Turkish", "Swahili", "Yoruba", "Hausa", "Igbo", "Swedish"]
LANG_RE = re.compile(r"\b(" + "|".join(LANGS) + r")\b")
LEVEL_WORDS = [("mother tongue", "native"), ("native", "native"), ("bilingual", "native"), ("first language", "native"),
               ("full professional", "C1"), ("fluent", "C1"), ("fluency", "C1"), ("proficient", "C1"), ("advanced", "C1"),
               ("excellent", "C1"), ("upper intermediate", "B2"), ("upper-intermediate", "B2"),
               ("professional working", "B2"), ("limited working", "A2"), ("professional", "B2"), ("strong", "B2"),
               ("good", "B2"), ("intermediate", "B1"), ("conversational", "B1"), ("working knowledge", "B1"),
               ("elementary", "A2"), ("basic", "A2"), ("beginner", "A1")]

CERT_NAMES = {c: [] for D in DOMAINS.values() for c in D["certs"]}
CERT_NAMES.update({
    "AWS Solutions Architect": ["aws certified solutions architect"], "CKA": ["certified kubernetes administrator"],
    "AWS ML Specialty": ["aws certified machine learning", "aws machine learning specialty"],
    "CFA": ["chartered financial analyst", "cfa charterholder"], "FRM": ["financial risk manager"],
    "CPA": ["certified public accountant"], "Bar Admission": ["admitted to the bar", "bar admitted"],
})

# Degree patterns, most advanced first. Abbreviations are case-sensitive; spelled-out forms are not.
DEGREE_PATTERNS = [
    ("PhD", r"(?i:\bph\.?\s?d\b|\bd\.?phil\b|\bdoctor(?:ate| of philosophy)\b)"),
    ("JD", r"\bJ\.D\.?|\bJD\b|(?i:\bjuris doctor\b)"),
    ("MBA", r"\bMBA\b|(?i:\bmaster of business administration\b)"),
    ("LLM", r"\bLL\.?M\b\.?|(?i:\bmaster of laws\b)"),
    ("MSc", r"\bM\.?Sc\b\.?|\bMS\b(?!\s*(?:Excel|Office|Word|Teams|SQL|Access|Project|PowerPoint|Azure))|\bM\.S\.|\bM\.?Eng\b|\bMRes\b|\bMPhil\b|(?i:\bmaster of (?:science|engineering|research)\b)"),
    ("MA", r"\bM\.?A\b\.?|(?i:\bmaster of arts\b)"),
    ("LLB", r"\bLL\.?B\b\.?|(?i:\bbachelor of laws\b)"),
    ("BSc", r"\bB\.?Sc\b\.?|\bBS\b|\bB\.S\.|\bB\.?Eng\b|\bB\.?Tech\b|(?i:\bbachelor of (?:science|engineering|technology)\b)"),
    ("BA", r"\bB\.?A\b\.?|(?i:\bbachelor of arts\b)"),
    ("MSc", r"(?i:\bmaster'?s\b)"),
    ("BSc", r"(?i:\bbachelor'?s\b)"),
]
DEGREE_RES = [(name, re.compile(p)) for name, p in DEGREE_PATTERNS]
GENERIC_DEGREE = re.compile(r"(?i)\b(advanced|graduate|postgraduate|university|college|undergraduate|relevant|bachelor'?s)\s+degree\b|\bdegree in\b|\bdegree required\b")
INST_RE = re.compile(r"(?i:\b(?:university|universit[àäé]|college|institute|school|academy|polytechnic|politecnico|hochschule)\b)|\b(?:MIT|ETH|UCL|LSE|EPFL|KTH|NYU|UCLA)\b")

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_WHEN = rf"(?:(?:{_MONTH}|\d{{1,2}})[\s/.-]*)?"
DATE_RANGE = re.compile(rf"(?i){_WHEN}((?:19|20)\d{{2}})\s*(?:-|–|—|to|until|till)\s*"
                        rf"(?:{_WHEN}((?:19|20)\d{{2}})|(present|current|now|today|ongoing|date))")
BULLET = re.compile(r"^\s*(?:[•●▪◦‣·]\s*|[-*–—>]\s+|\d+[.)]\s+)")
SPLIT = re.compile(r"\s+(?:at|@)\s+|\s*[|•·,;]\s*|\s+[-–—]\s+|\s*[–—]\s*")
CONTACT = re.compile(r"(?i)@|https?://|www\.|linkedin|github\.com|\+?\d[\d ()-]{7,}")

TITLE_NOUNS = {t.split()[-1].lower() for D in DOMAINS.values() for t in D["titles"]} | {
    "engineer", "scientist", "analyst", "manager", "developer", "consultant", "counsel", "associate", "lead", "director",
    "specialist", "officer", "writer", "designer", "marketer", "intern", "head", "architect", "researcher", "strategist",
    "advisor", "adviser", "partner", "lawyer", "solicitor", "attorney", "accountant", "administrator", "coordinator",
    "statistician", "programmer", "owner", "executive", "founder", "principal", "fellow", "assistant", "technician",
    "editor", "lecturer", "professor", "monitor", "vp", "cto", "ceo", "cfo"}
LOCATION_WORDS = {c.lower() for c, _ in CITIES} | {c.lower() for c in _COUNTRY_NAMES} | {"remote", "hybrid", "on-site", "onsite"}

CV_HEADINGS = {
    "summary": ["summary", "profile", "professional summary", "professional profile", "about me", "about", "objective",
                "career objective", "personal statement"],
    "experience": ["experience", "work experience", "professional experience", "employment", "employment history",
                   "work history", "career history", "relevant experience"],
    "education": ["education", "academic background", "education and training", "academic qualifications"],
    "skills": ["skills", "technical skills", "core skills", "key skills", "core competencies", "competencies", "tools",
               "technologies", "skills and tools", "tools and technologies", "skills and technologies"],
    "certifications": ["certifications", "certificates", "certification", "licenses", "licences",
                       "licenses and certifications", "licences and certifications", "certifications and licenses"],
    "languages": ["languages", "language skills", "language"],
    "projects": ["projects", "selected projects", "personal projects", "publications", "projects and publications"],
    "awards": ["awards", "honours", "honors", "achievements", "awards and honours", "awards and honors"],
}

SOFT_HEAD = re.compile(r"(?i)\b(nice[- ]to[- ]haves?|preferred|bonus|desirable|desired|pluses|good to have|ideally)\b")
HARD_HEAD = re.compile(r"(?i)\b(requirements?|required|must[- ]haves?|essential|minimum qualifications|basic qualifications|"
                       r"what you(?:'ll)? need|you (?:have|bring|will have)|who you are|qualifications)\b")
SOFT_LINE = re.compile(r"(?i)\b(nice to have|preferred|bonus|a plus|desirable|ideally|advantage(?:ous)?|beneficial|familiarity)\b")
HARD_LINE = re.compile(r"(?i)\b(must|required|essential|mandatory|minimum|at least|non-negotiable)\b")
YEARS_RE = re.compile(r"(?i)\b(\d{1,2})\s*\+?\s*(?:(?:-|–|to)\s*\d{1,2}\s*\+?\s*)?(?:years?|yrs?)\b")
PAY_RE = re.compile(r"[$€£]\s?(\d{2,4}(?:\.\d+)?)\s*(?:(?:-|–|to)\s*[$€£]?\s?(\d{2,4}(?:\.\d+)?))?\s*(?:/|per\s+)\s*(?:h|hr|hour)\b", re.I)
EU_RE = re.compile(r"(?i)\b(?:based|located|resident|reside|living)\s+(?:in|within)\s+(?:the\s+)?(?:EU|European Union|EEA)\b|\bEU[- ]based\b")
BASED_IN_RE = re.compile(r"(?i)\b(?:based|located|resident|reside|living)\s+(?:in|within)\s+(?:the\s+)?")

ALL_SKILLS = list(dict.fromkeys(s for D in DOMAINS.values() for s in D["skills"]))
CASE_SENSITIVE_SKILLS = {"R", "Go", "Excel", "React", "Spark", "Figma"}   # also ordinary words / letters
JD_EXTRA_ALIASES = {"Go": ["golang"]}
KINDS = ["skill", "years", "degree", "cert", "language", "country"]


# ------------------------------------------------------------------ documents
def read_document(name: str, data: bytes) -> str:
    """Plain text from an uploaded .pdf, .docx, .txt or .md file."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("Reading PDFs needs `pip install pypdf` (or paste the text instead).") from e
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if ext == ".docx":
        try:
            import docx
        except ImportError as e:
            raise ImportError("Reading .docx files needs `pip install python-docx` (or paste the text instead).") from e
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        parts += [" | ".join(c.text for c in row.cells) for t in d.tables for row in t.rows]
        return "\n".join(parts)
    return data.decode("utf-8-sig", errors="replace")


# ------------------------------------------------------------------ shared helpers
def _lines(text: str) -> list[str]:
    return [l.strip() for l in text.replace("\t", " | ").splitlines() if l.strip()]


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z]+", " ", s.lower().replace("&", " and ")).split())


def _sections(lines: list[str], headings: dict[str, list[str]]) -> dict[str, list[str]]:
    lookup = {_norm(h): name for name, hs in headings.items() for h in hs}
    out, cur = {"_top": []}, "_top"
    for l in lines:
        if len(l) <= 50 and _norm(l) in lookup:
            cur = lookup[_norm(l)]
            out.setdefault(cur, [])
            continue
        out.setdefault(cur, []).append(l)
    return out


def _strip_bullet(l: str) -> str:
    return BULLET.sub("", l).strip()


def _looks_like_title(s: str) -> bool:
    return any(w in TITLE_NOUNS for w in re.findall(r"[a-z]+", s.lower()))


def _is_location(s: str) -> bool:
    return s.strip(" ()").lower() in LOCATION_WORDS


def canonical_country(name: str) -> str:
    return COUNTRY_ALIASES.get(name, name)


def _level(s: str) -> str | None:
    m = re.search(r"\b([ABC][12])\b", s)
    if m:
        return m.group(1)
    low = s.lower()
    return next((lvl for words, lvl in LEVEL_WORDS if words in low), None)


def _degree(line: str):
    """(degree, match) for the most advanced degree named on the line, else (None, None)."""
    for name, rx in DEGREE_RES:
        m = rx.search(line)
        if m:
            return name, m
    return None, None


def _mentioned(skill: str, text: str) -> bool:
    if skill in CASE_SENSITIVE_SKILLS:
        hit = re.search(r"(?<![\w&])" + re.escape(skill) + r"(?![\w&+#])", text) is not None
    else:
        hit = _hit(_terms(skill), text)
    return hit or any(_pattern(a).search(text.lower()) for a in JD_EXTRA_ALIASES.get(skill, []))


# ------------------------------------------------------------------ CV
def parse_cv(text: str) -> tuple[Profile, list[str]]:
    """Free-text CV -> Profile (+ notes on what could not be read). The CV has no AI-interview summary."""
    notes = []
    lines = _lines(text)
    sec = _sections(lines, CV_HEADINGS)
    top = sec.get("_top", [])

    exp_lines = sec.get("experience")
    if not exp_lines:
        exp_lines = [l for k, v in sec.items() if k not in ("education", "skills", "languages", "certifications") for l in v]
        notes.append("No 'Experience' heading found; looked for dated roles anywhere in the CV.")
    exps = _experiences(exp_lines)
    if not exps:
        notes.append("No dated roles found (e.g. 'Data Scientist, Acme | 2019 - 2023'). Years of experience can only "
                     "be counted from roles with start and end years.")

    edu = _education(sec.get("education", []))
    if not edu:
        notes.append("No degree recognised (BSc, MSc, MBA, PhD, ...).")

    skills = _list_items(sec.get("skills", []))
    if not skills:
        notes.append("No 'Skills' section found. Stage 2 gives full marks only to skills that are listed AND evidenced in your work.")
    certs = _certs(sec.get("certifications", []), text)
    langs, unleveled = _languages(sec.get("languages") or lines, in_section=bool(sec.get("languages")))
    for lang in unleveled:
        notes.append(f"{lang} is listed without a level; graders cannot verify it. Add a level (e.g. C1, fluent, native).")
    if "English" not in langs:
        notes.append("English level not stated. Many roles have an English hard filter; a missing level reads as 'not met'.")

    city, country = _location(top or lines[:8])
    if not country:
        notes.append("No country recognised in the CV header; location-restricted roles cannot be checked.")
    headline = _headline(top, sec.get("summary", []))
    if not headline:
        notes.append("No headline/tagline found under your name; Stage 1 weights the headline most heavily.")

    profile = Profile(
        id="CV-" + hashlib.sha1(text.encode()).hexdigest()[:10], headline=headline,
        current_title=exps[0].title if exps else headline, current_company=exps[0].company if exps else "",
        city=city, country=country, experiences=exps, education=edu, skills=skills, certifications=certs,
        languages=langs, awards=[_strip_bullet(l) for l in sec.get("awards", [])],
        projects=[_strip_bullet(l) for l in sec.get("projects", [])], interview_summary="", last_active_days=0)
    return profile, notes


def _is_header(l: str) -> bool:
    return not BULLET.match(l) and len(l) <= 90 and not l.endswith(".") and not DATE_RANGE.search(l)


def _experiences(lines: list[str]) -> list[Experience]:
    spans, floor = [], 0
    for i in (i for i, l in enumerate(lines) if DATE_RANGE.search(l)):
        h = i
        while h > floor and i - h < 2 and _is_header(lines[h - 1]):
            h -= 1
        spans.append((h, i))
        floor = i + 1
    out = []
    for k, (h, i) in enumerate(spans):
        end = spans[k + 1][0] if k + 1 < len(spans) else len(lines)
        m = DATE_RANGE.search(lines[i])
        start = int(m.group(1))
        stop = int(m.group(2)) if m.group(2) else CURRENT_YEAR
        title, company = _title_company(lines[h:i] + [DATE_RANGE.sub(" ", lines[i])])
        desc = " ".join(_strip_bullet(l) for l in lines[i + 1:end])
        out.append(Experience(title, company, desc, start, max(start, stop)))
    out.sort(key=lambda e: (e.end_year, e.start_year), reverse=True)
    return out


def _title_company(head: list[str]) -> tuple[str, str]:
    parts = [p.strip(" ()[],.-–—|·•") for p in SPLIT.split(" | ".join(head))]
    parts = [p for p in parts if p and not _is_location(p)]
    if not parts:
        return "", ""
    title = next((p for p in parts if _looks_like_title(p)), parts[0])
    return title, next((p for p in parts if p != title), "")


def _institution(line: str) -> str:
    line = re.sub(r"\(?\b(?:19|20)\d{2}\b\)?", "", line)
    return next((p.strip(" ()") for p in SPLIT.split(line) if INST_RE.search(p)), "")


def _field(line: str, m) -> str:
    rest = re.sub(r"^[\s.,:()-]*(?:(?:in|of)\s+)?", "", line[m.end():])
    cand = re.sub(r"\(?\b(?:19|20)\d{2}\b.*$", "", SPLIT.split(rest)[0]).strip(" .()")
    if cand and not INST_RE.search(cand):
        return cand
    known = [f for D in DOMAINS.values() for f in D["fields"]]
    return next((f for f in known if f.lower() in line.lower()), "")


def _education(lines: list[str]) -> list[Education]:
    out = []
    for j, l in enumerate(lines):
        deg, m = _degree(l)
        if not deg:
            continue
        near = [l] + lines[j + 1:j + 2] + lines[max(0, j - 1):j]
        inst = next((x for x in map(_institution, near) if x), "")
        out.append(Education(deg, _field(l, m), inst))
    out.sort(key=lambda e: -DEGREE_RANK.get(e.degree, 0))
    return out


def _list_items(lines: list[str]) -> list[str]:
    items = []
    for l in lines:
        l = _strip_bullet(l)
        head, sep, tail = l.partition(":")
        if sep and len(head) <= 30:
            l = tail
        items += [x.strip(" .") for x in re.split(r"[,;|•·]| {2,}", l)]
    return [x for x in dict.fromkeys(items) if 1 <= len(x) <= 40]


def _certs(lines: list[str], text: str) -> list[str]:
    found = [_strip_bullet(l) for l in lines]
    low = text.lower()
    for canon, alts in CERT_NAMES.items():
        if canon not in found and any(_pattern(a).search(low) for a in [canon] + alts):
            found.append(canon)
    return found


def _languages(lines: list[str], in_section: bool) -> tuple[dict[str, str], list[str]]:
    out, unleveled = {}, []
    for l in lines:
        ms = list(LANG_RE.finditer(l))
        if not ms:
            continue
        shared = _level(l[:ms[0].start()])            # "Fluent in English and Spanish"
        for k, m in enumerate(ms):
            seg = l[m.end(): ms[k + 1].start() if k + 1 < len(ms) else len(l)]
            lvl = _level(seg) or shared
            if lvl:
                out.setdefault(m.group(1), lvl)
            elif in_section:
                unleveled.append(m.group(1))
    return out, [u for u in dict.fromkeys(unleveled) if u not in out]


def _location(lines: list[str]) -> tuple[str, str]:
    for l in lines:
        for city, country in CITIES:
            if re.search(rf"\b{re.escape(city)}\b", l):
                return city, country
        m = COUNTRY_RE.search(l)
        if m:
            city = next((p.strip() for p in SPLIT.split(l[:m.start()]) if p.strip() and not CONTACT.search(p)), "")
            return (city if len(city.split()) <= 3 else ""), canonical_country(m.group(1))
    return "", ""


def _headline(top: list[str], summary: list[str]) -> str:
    cands = [l for l in top if not CONTACT.search(l) and not all(_is_location(p) for p in SPLIT.split(l) if p.strip())]
    for k, l in enumerate(cands):
        words = l.split()
        if len(words) < 2 or len(l) > 140:
            continue
        if k == 0 and len(words) <= 4 and "|" not in l and not _looks_like_title(l):
            continue                                   # first short line without a job noun: probably the name
        return l
    if summary:
        return re.split(r"(?<=[.!?])\s", summary[0])[0][:140]
    return ""


# ------------------------------------------------------------------ job description
def make_criterion(kind: str, value, hard: bool = False, domain: str = "") -> Criterion:
    """Build a criterion with the same ids/texts the synthetic roles use (shared by the JD parser and the UI)."""
    v = str(value).strip()
    if not v:
        raise ValueError(f"{kind} criterion needs a value")
    if kind == "skill":
        return Criterion(f"skill:{v}", f"Hands-on proficiency in {v}", "skill", {"name": v}, hard)
    if kind == "years":
        n = int(float(v))
        return Criterion(f"years:{domain}", f"At least {n} years of professional experience in {domain.replace('_', ' ')}",
                         "years", {"domain": domain, "min": n}, hard)
    if kind == "degree":
        m = DEGREE_RANK.get(v) or int(float(v))
        return Criterion(f"degree:{m}", f"Degree level rank >= {m} (1=BSc, 2=MSc, 3=PhD)", "degree", {"min_rank": m}, hard)
    if kind == "cert":
        return Criterion(f"cert:{v}", f"Holds the {v} certification", "cert", {"name": v}, hard)
    if kind == "language":
        lang, _, lvl = v.partition(":")
        lang, lvl = lang.strip(), (lvl.strip() or "B2")
        if lvl not in LEVEL_ORDER:
            raise ValueError(f"language level must be one of {LEVEL_ORDER}, e.g. English:B2")
        return Criterion(f"lang:{lang}:{lvl}", f"{lang} proficiency at least {lvl}", "language", {"lang": lang, "min": lvl}, hard)
    if kind == "country":
        if v.upper() == "EU":
            return Criterion("country:EU", "Based in the EU (data residency requirement)", "country", {"countries": sorted(EU27)}, hard)
        cs = sorted(canonical_country(x.strip()) for x in v.split(",") if x.strip())
        return Criterion(f"country:{'/'.join(cs)}", f"Based in {', '.join(cs)}", "country", {"countries": cs}, hard)
    raise ValueError(f"unknown criterion kind '{kind}' (use one of {KINDS})")


def criterion_value(c: Criterion) -> str:
    """Inverse of make_criterion's `value`, for editing criteria in a table."""
    p = c.params
    if c.kind == "language":
        return f"{p['lang']}:{p['min']}"
    if c.kind == "country":
        return "EU" if c.id == "country:EU" else ", ".join(p["countries"])
    return str({"skill": p.get("name"), "years": p.get("min"), "degree": p.get("min_rank"), "cert": p.get("name")}[c.kind])


def _requirement_context(lines: list[str]) -> list[str | None]:
    """Per line: 'hard' (must-have), 'soft' (nice-to-have) or None, from headings and the line's own wording."""
    out, section = [], None
    for l in lines:
        heading = not BULLET.match(l) and len(l) <= 60 and (l.endswith(":") or (len(l.split()) <= 5 and not re.search(r"\d", l)))
        if heading and SOFT_HEAD.search(l):
            section = "soft"
        elif heading and HARD_HEAD.search(l):
            section = "hard"
        elif heading:
            section = None                              # 'About us', 'Benefits', 'Responsibilities', ...
        own = "soft" if SOFT_LINE.search(l) else "hard" if HARD_LINE.search(l) else None
        out.append(own or section)
    return out


def parse_jd(text: str, title: str | None = None, pay_rate: float | None = None) -> tuple[Role, list[str]]:
    """Free-text job description -> Role (+ notes). Must-have vs nice-to-have comes from headings and wording."""
    notes = []
    lines = _lines(text)
    ctx = _requirement_context(lines)
    low = text.lower()

    dom_score = {d: sum(_mentioned(s, text) for s in D["skills"]) + 2 * sum(t.lower() in low for t in D["titles"])
                 for d, D in DOMAINS.items()}
    dom = max(dom_score, key=dom_score.get)
    if dom_score[dom] < 3:
        notes.append(f"Weak match to the benchmark's domains (best guess: {dom.replace('_', ' ')}). The Stage-1 "
                     "comparison is only meaningful for data science, backend, frontend, finance, legal, clinical or marketing roles.")
    if title is None:
        title = re.sub(r"(?i)^(job title|position|role)\s*:\s*", "", lines[0])[:100] if lines else "Untitled role"
    tw = set(re.findall(r"[a-z]+", title.lower()))
    target = max(DOMAINS[dom]["titles"], key=lambda t: (len(tw & set(t.lower().split())), t.lower() in low))

    crit: list[Criterion] = []
    skills: dict[str, bool] = {}
    cert_rx = re.compile("|".join(re.escape(a) for c, alts in CERT_NAMES.items() for a in [c] + alts), re.I)
    for i, l in enumerate(lines):
        if _degree(l)[0] or GENERIC_DEGREE.search(l):
            continue                                   # "degree in Statistics" names a field of study, not a skill
        l = cert_rx.sub(" ", l)                        # "AWS ML Specialty" is a certification, not the skill AWS
        for s in ALL_SKILLS:
            if _mentioned(s, l):
                skills[s] = skills.get(s, False) or ctx[i] == "hard"
    crit += [make_criterion("skill", s, h) for s, h in skills.items()]

    yrs = [(int(m.group(1)), ctx[i]) for i, l in enumerate(lines)
           if "experience" in l.lower() or ctx[i] for m in YEARS_RE.finditer(l) if 0 < int(m.group(1)) <= 20]
    if yrs:
        n = max(y for y, _ in yrs)
        crit.append(make_criterion("years", n, any(c == "hard" for y, c in yrs if y == n), dom))

    degs = []
    for i, l in enumerate(lines):
        ranks = [DEGREE_RANK[name] for name, rx in DEGREE_RES if rx.search(l)]
        g = None if ranks else GENERIC_DEGREE.search(l)
        if g:
            ranks.append(2 if (g.group(1) or "").lower() in ("advanced", "graduate", "postgraduate") else 1)
        if ranks:
            degs.append((min(ranks), ctx[i]))
    if degs:
        hard = [r for r, c in degs if c == "hard"]
        crit.append(make_criterion("degree", min(hard) if hard else min(r for r, _ in degs), bool(hard)))

    langs: dict[str, tuple[str, bool]] = {}
    for i, l in enumerate(lines):
        ms = list(LANG_RE.finditer(l))
        shared = _level(l[:ms[0].start()]) if ms else None
        for k, m in enumerate(ms):
            lvl = _level(l[m.end(): ms[k + 1].start() if k + 1 < len(ms) else len(l)]) or shared
            if lvl and m.group(1) not in langs:
                langs[m.group(1)] = (lvl, ctx[i] != "soft")
    crit += [make_criterion("language", f"{lang}:{lvl}", hard) for lang, (lvl, hard) in langs.items()]

    if EU_RE.search(text):
        crit.append(make_criterion("country", "EU", True))
    else:
        for l in lines:
            b = BASED_IN_RE.search(l)
            m = COUNTRY_RE.match(l, b.end()) if b else None
            if m:
                crit.append(make_criterion("country", canonical_country(m.group(1)), True))
                break

    for canon, alts in CERT_NAMES.items():
        hits = [i for i, l in enumerate(lines) if any(_pattern(a).search(l.lower()) for a in [canon] + alts)]
        if hits:
            crit.append(make_criterion("cert", canon, any(ctx[i] == "hard" for i in hits)))

    if pay_rate is None:
        m = PAY_RE.search(text)
        if m:
            pay_rate = (float(m.group(1)) + float(m.group(2) or m.group(1))) / 2
        else:
            pay_rate = 60.0
            notes.append("No hourly rate found; assumed $60/h (it only changes how much the AI match counts in the final rank).")
    if not crit:
        notes.append("No requirements recognised. Add criteria by hand before checking.")
    if not any(c.hard for c in crit):
        notes.append("No must-have requirements recognised, so no hard filters apply. Mark must-haves by hand if needed.")

    desc = next((l for l in lines[1:] if len(l.split()) >= 12), title)[:300]
    role = Role("JD-" + hashlib.sha1(text.encode()).hexdigest()[:10], title, target, dom, desc, float(pay_rate), crit)
    return role, notes
