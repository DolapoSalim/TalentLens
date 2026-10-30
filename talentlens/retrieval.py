"""Stage 1: semantic retrieval over a fielded index (mirrors the notice's Section 2(a)/3).

Three scoring modes, so they can be ablated against each other:
  blob      one concatenated text per profile (headline first), the naive baseline
  fielded   per-field similarities combined with fixed weights (headline-heavy, as the notice describes)
  adaptive  INNOVATION: headline weight is scaled by a per-profile *headline consistency* score
            (how well the headline agrees with the rest of the profile). Vague or buzzword
            headlines lose influence; the freed weight goes to the evidence-bearing fields.

Similarity is a hybrid: dense (TF-IDF -> truncated SVD, i.e. LSA) mixed with sparse TF-IDF cosine.
Swap in `sentence-transformers` embeddings by replacing `_dense` if you want neural semantics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from .schema import Profile, Role
from .taxonomy import DOMAINS

FIELDS = ["headline", "current_role", "experience", "education", "skills", "location"]
NOTICE_WEIGHTS = {"headline": 0.34, "current_role": 0.24, "experience": 0.20, "education": 0.08, "skills": 0.11, "location": 0.03}
BALANCED_WEIGHTS = {f: 1 / len(FIELDS) for f in FIELDS}


def field_texts(p: Profile) -> dict[str, str]:
    """Exactly the Stage-1 fields: top-5 experiences (descriptions cut to 200 chars), top-3 education, top-15 skills."""
    return {
        "headline": p.headline,
        "current_role": f"{p.current_title} at {p.current_company}",
        "experience": " ".join(f"{e.title} at {e.company}: {e.description[:200]}" for e in p.experiences[:5]),
        "education": "; ".join(f"{e.degree} {e.field} {e.institution}" for e in p.education[:3]),
        "skills": ", ".join(p.skills[:15]),
        "location": f"{p.city} {p.country}",
    }


def blob_text(p: Profile) -> str:
    t = field_texts(p)
    return " . ".join(t[f] for f in FIELDS)


def ideal_candidate_query(role: Role) -> dict[str, str]:
    """Template 'ideal candidate' (HyDE-style). `llm.LLMQueryWriter` can replace this."""
    skills = [c.params["name"] for c in role.criteria if c.kind == "skill"]
    yrs = next((c.params["min"] for c in role.criteria if c.kind == "years"), 3)
    senior = "Senior " if yrs >= 5 else ""
    dom = DOMAINS[role.domain]
    degree = next((c.params["min_rank"] for c in role.criteria if c.kind == "degree"), 1)
    countries = next((c.params["countries"] for c in role.criteria if c.kind == "country"), [])
    return {
        "headline": f"{senior}{role.target_title} | {' | '.join(skills[:2])}",
        "current_role": f"{senior}{role.target_title}",
        "experience": f"{role.target_title}: {role.description} {' '.join(dom['tasks'][:2])} using {', '.join(skills)}",
        "education": ("MSc PhD " if degree >= 2 else "BSc ") + " ".join(dom["fields"][:2]),
        "skills": ", ".join(skills),
        "location": " ".join(countries),
    }


@dataclass
class RetrievalConfig:
    mode: str = "adaptive"                      # blob | fielded | adaptive
    weights: dict = field(default_factory=lambda: dict(NOTICE_WEIGHTS))
    dense_mix: float = 0.5
    k_active: int = 150
    k_broad: int = 250
    active_days: int = 30
    adapt_floor: float = 0.25
    adapt_pct: float = 35.0                     # percentile of headline consistency that gets full headline weight
    svd_dims: int = 128


@dataclass
class Retrieved:
    idx: np.ndarray            # pool indices in retrieval order
    score: np.ndarray
    phase: list[str]
    all_scores: np.ndarray
    cutoff: dict


class FieldIndex:
    def __init__(self, cfg: RetrievalConfig | None = None):
        self.cfg = cfg or RetrievalConfig()

    # ------------------------------------------------------------ build
    def build(self, profiles: list[Profile]) -> "FieldIndex":
        self.profiles = profiles
        texts = [field_texts(p) for p in profiles]
        blobs = [blob_text(p) for p in profiles]
        corpus = [t[f] for t in texts for f in FIELDS] + blobs
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(corpus)
        self.svd = TruncatedSVD(self.cfg.svd_dims, random_state=0).fit(self.vec.transform(corpus))
        self.X, self.D = self._matrices(texts, blobs)
        self.active = np.array([p.last_active_days <= self.cfg.active_days for p in profiles])
        c = self.headline_consistency(self.D)
        self.tau = float(np.percentile(c, self.cfg.adapt_pct))
        return self

    def _dense(self, X) -> np.ndarray:
        Z = self.svd.transform(X)
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)

    def _matrices(self, texts, blobs):
        X = {f: self.vec.transform([t[f] for t in texts]) for f in FIELDS}
        X["blob"] = self.vec.transform(blobs)
        return X, {k: self._dense(v) for k, v in X.items()}

    # ------------------------------------------------------------ scoring
    @staticmethod
    def headline_consistency(D: dict) -> np.ndarray:
        ctx = D["experience"] + D["current_role"]
        ctx = ctx / np.maximum(np.linalg.norm(ctx, axis=1, keepdims=True), 1e-9)
        return (D["headline"] * ctx).sum(1)

    def _weights(self, D: dict) -> np.ndarray:
        base = np.array([self.cfg.weights[f] for f in FIELDS], dtype=float)
        base /= base.sum()
        n = D["headline"].shape[0]
        W = np.tile(base, (n, 1))
        if self.cfg.mode == "adaptive":
            a = np.clip(self.headline_consistency(D) / max(self.tau, 1e-9), self.cfg.adapt_floor, 1.0)
            freed = base[0] * (1 - a)
            others = base[1:] / base[1:].sum()
            W[:, 0] = base[0] * a
            W[:, 1:] = base[1:] + freed[:, None] * others
        return W

    def _components(self, X: dict, D: dict, query: dict[str, str]):
        qtexts = {f: query.get(f, "") for f in FIELDS}
        qX = {f: self.vec.transform([qtexts[f]]) for f in FIELDS}
        qX["blob"] = self.vec.transform([" . ".join(qtexts[f] for f in FIELDS)])
        qD = {k: self._dense(v)[0] for k, v in qX.items()}
        m = self.cfg.dense_mix

        def sim(f):
            return m * (D[f] @ qD[f]) + (1 - m) * (X[f] @ qX[f].T).toarray().ravel()

        if self.cfg.mode == "blob":
            s = sim("blob")[:, None]
            return s, np.ones_like(s)
        return np.stack([sim(f) for f in FIELDS], 1), self._weights(D)

    def score_pool(self, query: dict[str, str]) -> np.ndarray:
        S, W = self._components(self.X, self.D, query)
        return (S * W).sum(1)

    def field_scores(self, query: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        """Per-field similarities and weights for the whole pool (rows = profiles, columns = FIELDS)."""
        return self._components(self.X, self.D, query)

    def score_variants(self, profiles: list[Profile], query: dict[str, str]) -> np.ndarray:
        """Score modified copies of profiles with the already-fitted index (for counterfactual audits)."""
        X, D = self._matrices([field_texts(p) for p in profiles], [blob_text(p) for p in profiles])
        S, W = self._components(X, D, query)
        return (S * W).sum(1)

    def explain_row(self, profile: Profile, query: dict[str, str]) -> dict:
        X, D = self._matrices([field_texts(profile)], [blob_text(profile)])
        S, W = self._components(X, D, query)
        if self.cfg.mode == "blob":
            return {"blob": {"similarity": float(S[0, 0]), "weight": 1.0, "contribution": float(S[0, 0])}}
        out = {f: {"similarity": float(S[0, j]), "weight": float(W[0, j]), "contribution": float(S[0, j] * W[0, j])}
               for j, f in enumerate(FIELDS)}
        out["_headline_consistency"] = float(self.headline_consistency(D)[0])
        return out

    # ------------------------------------------------------------ search
    def search(self, query: dict[str, str]) -> Retrieved:
        scores = self.score_pool(query)
        order = np.argsort(-scores)
        act = [i for i in order if self.active[i]][: self.cfg.k_active]
        taken = set(act)
        broad = [i for i in order if i not in taken][: self.cfg.k_broad]
        idx = np.array(act + broad)
        phase = ["active"] * len(act) + ["broad"] * len(broad)
        cutoff = {"active": float(scores[act[-1]]) if act else float("inf"),
                  "broad": float(scores[broad[-1]]) if broad else float("inf")}
        return Retrieved(idx, scores[idx], phase, scores, cutoff)

    def would_shortlist(self, score: float, is_active: bool, cutoff: dict) -> bool:
        return score >= (min(cutoff["active"], cutoff["broad"]) if is_active else cutoff["broad"])
