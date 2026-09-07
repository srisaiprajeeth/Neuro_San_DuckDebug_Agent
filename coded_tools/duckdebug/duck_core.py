# Copyright © 2025-2026 Cognizant Technology Solutions Corp, www.cognizant.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# END COPYRIGHT
"""Core logic for DuckDebug, grounded in the Kaggle "60k Stack Overflow Questions
with Quality Rating" dataset (Annamoradnejad et al., 2022 — Apache 2.0 / MIT).

Pure Python + scikit-learn; no neuro-san imports here so it can be unit-tested
standalone. Both the CodedTool wrappers (``clarity_scorer.py`` /
``echo_retriever.py``) and the MCP server (``servers/mcp/duck_mcp.py``) call into
this module, so the scoring logic has exactly one home.

The clarity score is an **ensemble vote** of three independent judges rather than
a single heuristic — see :func:`score_clarity`.

Two design notes that the data forced, both worth keeping in mind before tuning:

* **The corpus is indexed on a "framing view"** — the title plus the first
  ``_FRAMING_CHARS`` characters of the body — not the whole post. A user types
  the duck two sentences, while a real HQ post is a long HTML document with code
  blocks. Comparing those directly is a domain mismatch that pushes every short
  input toward "low quality". Truncating the corpus to roughly the length of
  what a user types removes that skew, and it also strips the trailing code
  dumps that otherwise dominate TF-IDF retrieval.
* **Raw classifier probabilities are compressed, so they are percentile-ranked**
  against the corpus's own score distribution. The resulting judge score answers
  "you framed this better than X% of real Stack Overflow posts", which spreads
  usefully across 0-100 where the raw probability does not.
"""

from __future__ import annotations

import functools
import hashlib
import html
import os
import re
from typing import Any
from typing import Dict
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# Dataset location, in priority order:
#   1. DUCK_DATA_CSV, if set.
#   2. data/stackoverflow_train.csv — the real 45k-row labeled corpus from
#      https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate
#      (15k HQ / 15k LQ_CLOSE / 15k LQ_EDIT).
#   3. data/stackoverflow_sample.csv — the tiny bundled sample, so the network
#      still runs if the full CSV was never downloaded.
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# coded_tools/duckdebug/ -> repo_root/data/
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "data"))
_CSV_CANDIDATES = (
    os.path.join(_DATA_DIR, "stackoverflow_train.csv"),
    os.path.join(_DATA_DIR, "stackoverflow_sample.csv"),
)


def _resolve_csv() -> str:
    """Pick the corpus to load, preferring the full dataset over the sample."""
    explicit = os.environ.get("DUCK_DATA_CSV")
    if explicit:
        return explicit
    for candidate in _CSV_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    # Nothing on disk: return the sample path so the error names a real expectation.
    return _CSV_CANDIDATES[-1]


DATA_CSV = _resolve_csv()

# How much of each post counts as its "framing" — see the module docstring.
_FRAMING_CHARS = 300

# Number of labeled neighbours the kNN judge consults.
_KNN_K = 5

# TF-IDF sizing. 8k features over bigrams keeps the fit ~15s on the 45k-row
# corpus while leaving the classifier enough signal (see _JUDGE_WEIGHTS note).
_MAX_FEATURES = 8000

_CLARITY_SIGNALS = {
    "expected_vs_actual": r"\bexpected\b|\bactual\b|\binstead\b|\bbut\s+got\b|\bshould\s+(return|be|show)\b",
    "error_or_symptom": (
        r"\berror\b|\bexception\b|\btraceback\b|\bstack\s*trace\b|\bcrash\b|\bnull\b|\bnan\b|\bfails?\b"
    ),
    "context_or_code": r"\bi\s+(tried|have|am|use|build|call)\b|\bcode\b|\bquery\b|\bfunction\b|\bline\s*\d+\b|```",
    "specificity": r"\b(python|sql|java|javascript|c#|react|docker|git|pandas|api|decimal)\b|\b\d+\b",
}

_MISSING_HINT = {
    "expected_vs_actual": "State what you EXPECTED to happen vs what ACTUALLY happened.",
    "error_or_symptom": "Include the exact error message / symptom you observed.",
    "context_or_code": "Add the relevant code, query, or what you already TRIED.",
    "specificity": "Name the concrete tech/values involved (language, table, numbers).",
}

# Ensemble weights; they sum to 1.0 so the blended score stays in [0, 1].
#
# rule_signals and learned_quality are weighted equally: the rule judge is the
# only one that can explain *which* piece is missing, and the learned judge is
# the only one actually trained on the HQ/LQ labels (~0.86 holdout accuracy on
# the 45k corpus, measured with a stratified 80/20 split). The kNN judge is a
# weaker tie-breaker because TF-IDF neighbours track topic as much as quality,
# so it gets the smallest share even after base-rate correction.
_JUDGE_WEIGHTS = {"rule_signals": 0.4, "learned_quality": 0.4, "knn_neighbours": 0.2}


def _clean(text: str) -> str:
    """Strip HTML entities/tags and collapse whitespace so corpus and query match.

    The real corpus stores Body as raw HTML (``<p>...</p>``), so tag-stripping is
    what makes it comparable to the plain text a user types at the duck.
    """
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_tags(raw: str) -> str:
    """Render a tag field as ``"a, b, c"`` regardless of source format.

    The Kaggle CSV uses ``<java><repeat>`` while the bundled sample uses
    ``java|repeat``; normalising both keeps the agent-facing output uniform and
    lets the substring tag filter work either way.
    """
    raw = str(raw or "").strip()
    if not raw:
        return ""
    parts = re.findall(r"<([^>]+)>", raw)
    if not parts:
        parts = re.split(r"[|,]", raw)
    return ", ".join(part.strip() for part in parts if part.strip())


def _framing_view(titles: pd.Series, bodies: pd.Series) -> pd.Series:
    """Build the truncated title+body text the corpus is indexed on."""
    joined = titles.fillna("").astype(str) + ". " + bodies.fillna("").astype(str)
    return joined.map(_clean).str.slice(0, _FRAMING_CHARS)


def _cache_path() -> str:
    """Where the fitted vectoriser/classifier for the current CSV is cached.

    Keyed by corpus identity (path, size, mtime), the indexing parameters, and the
    scikit-learn version — a pickled estimator is not portable across versions, so
    an upgrade must invalidate rather than silently load a stale object.
    """
    try:
        stat = os.stat(DATA_CSV)
        fingerprint = f"{os.path.abspath(DATA_CSV)}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        fingerprint = os.path.abspath(DATA_CSV)
    fingerprint += f"|{_FRAMING_CHARS}|{_MAX_FEATURES}|sklearn{sklearn_version}"
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_DATA_DIR, ".duckdebug_cache", f"corpus_{digest}.joblib")


@functools.lru_cache(maxsize=1)
def _load_corpus() -> Dict[str, Any]:
    """Load the CSV, fit TF-IDF, train the quality classifier, and build the
    percentile calibration table.

    Fitting over the full 45k-row corpus is slow enough (tens of seconds, and
    markedly worse under a live request) that the result is cached two ways: in
    process via ``lru_cache``, and on disk via joblib so a restart does not refit.
    """
    cache_file = _cache_path()
    if os.path.exists(cache_file):
        try:
            return joblib.load(cache_file)
        except Exception:  # pragma: no cover - stale/corrupt cache, just refit
            pass

    corpus = _build_corpus()

    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        joblib.dump(corpus, cache_file, compress=3)
    except Exception:  # pragma: no cover - a read-only data dir must not be fatal
        pass

    return corpus


def _build_corpus() -> Dict[str, Any]:
    """Read the CSV and fit everything from scratch (the expensive path)."""
    df = pd.read_csv(DATA_CSV)
    cols = {c.lower(): c for c in df.columns}
    title_c = cols.get("title", "Title")
    body_c = cols.get("body", "Body")
    tag_c = cols.get("tags")
    label_c = cols.get("y")

    df["_title"] = df[title_c].fillna("").astype(str) if title_c in df.columns else ""
    df["_text"] = _framing_view(df["_title"], df[body_c] if body_c in df.columns else pd.Series("", index=df.index))
    df["_tags"] = df[tag_c].fillna("").astype(str).map(_normalize_tags) if tag_c else ""
    # The dataset labels rows HQ / LQ_CLOSE / LQ_EDIT; anything not HQ is low quality.
    df["_label"] = df[label_c].fillna("").astype(str) if label_c else "HQ"

    vect = TfidfVectorizer(stop_words="english", max_features=_MAX_FEATURES, ngram_range=(1, 2))
    matrix = vect.fit_transform(df["_text"])

    hq_mask = df["_label"].str.upper().str.startswith("HQ").values

    # Train the supervised judge. Needs both classes present; the bundled sample
    # is tiny but still mixed, and a single-class CSV degrades to the other judges.
    clf = None
    calibration = None
    if hq_mask.any() and not hq_mask.all():
        clf = LogisticRegression(max_iter=1000)
        clf.fit(matrix, hq_mask.astype(int))
        # Sorted in-corpus probabilities: the reference distribution a user's
        # score is percentile-ranked against.
        calibration = np.sort(clf.predict_proba(matrix)[:, 1])

    # Keep only the columns retrieval actually reports. The raw HTML Body is the
    # bulk of the 49MB CSV and is already baked into `matrix`, so dropping it
    # keeps both the disk cache and the resident footprint small.
    slim = df[["_title", "_tags", "_label"]].reset_index(drop=True)

    return {
        "df": slim,
        "vect": vect,
        "matrix": matrix,
        "clf": clf,
        "calibration": calibration,
        "hq_mask": hq_mask,
        "hq_rate": float(hq_mask.mean()) if len(hq_mask) else 0.5,
    }


def warm_up() -> Dict[str, Any]:
    """Force the corpus/model cache to build now, and report what was loaded.

    Callers that serve requests (the MCP server) should call this at import time:
    building the cache lazily inside the first request is several times slower
    than building it up front, and it strands the first user turn behind it.
    """
    corpus = _load_corpus()
    return {
        "corpus": DATA_CSV,
        "rows": int(len(corpus["df"])),
        "hq_rate": round(corpus["hq_rate"], 3),
        "classifier": corpus["clf"] is not None,
    }


def _judge_rule_signals(low: str) -> Dict[str, Any]:
    """Judge 1: how many of the four hallmarks of a well-framed question are present."""
    present, missing = [], []
    for name, pattern in _CLARITY_SIGNALS.items():
        (present if re.search(pattern, low) else missing).append(name)
    return {
        "score": len(present) / len(_CLARITY_SIGNALS),
        "present": present,
        "missing": missing,
    }


def _judge_learned_quality(vec, corpus: Dict[str, Any]) -> Dict[str, Any]:
    """Judge 2: the trained HQ/LQ classifier, percentile-ranked against the corpus.

    Returns the percentile in [0, 1] as the judge score, plus the raw probability
    for transparency. 0.5 means "no opinion" when the classifier is unavailable.
    """
    clf, calibration = corpus["clf"], corpus["calibration"]
    if clf is None or calibration is None or not len(calibration):
        return {"score": 0.5, "available": False}

    probability = float(clf.predict_proba(vec)[0][1])
    percentile = float(np.searchsorted(calibration, probability) / len(calibration))
    return {
        "score": percentile,
        "available": True,
        "hq_probability": round(probability, 3),
        "percentile": round(percentile, 3),
    }


def _judge_knn(vec, corpus: Dict[str, Any], k: int = _KNN_K) -> Dict[str, Any]:
    """Judge 3: base-rate-corrected, similarity-weighted vote of the k nearest posts.

    Weighting by similarity stops a barely-related neighbour counting as much as
    a near-duplicate. The base-rate correction matters because only ~1/3 of the
    corpus is HQ: without it, "2 of 5 neighbours are HQ" would look like evidence
    of low quality when it is actually *above* the corpus average.
    """
    sims = cosine_similarity(vec, corpus["matrix"])[0]
    top = np.argsort(sims)[::-1][:k]
    hq_mask = corpus["hq_mask"]

    weight_total = float(sum(sims[i] for i in top))
    if weight_total <= 0:
        return {"score": 0.5, "neighbours": int(len(top)), "hq_neighbours": 0}

    fraction = float(sum(sims[i] for i in top if hq_mask[i])) / weight_total

    # Rescale the observed HQ fraction by the corpus prior via a likelihood ratio,
    # so "exactly the base rate" maps to 0.5.
    prior = corpus["hq_rate"]
    eps = 1e-6
    fraction = min(max(fraction, eps), 1 - eps)
    prior = min(max(prior, eps), 1 - eps)
    odds = (fraction / (1 - fraction)) / (prior / (1 - prior))
    score = odds / (1 + odds)

    return {
        "score": float(score),
        "neighbours": int(len(top)),
        "hq_neighbours": int(sum(1 for i in top if hq_mask[i])),
        "hq_fraction_weighted": round(fraction, 3),
        "corpus_hq_rate": round(prior, 3),
    }


def score_clarity(text: str) -> Dict[str, Any]:
    """Rate how clearly a stuck problem is articulated (0-100 + guidance).

    Runs three independent judges and combines them by weighted ensemble vote:

    1. ``rule_signals``    — regex hallmarks of a good bug report.
    2. ``learned_quality`` — a logistic-regression HQ/LQ classifier trained on the
       corpus labels, percentile-ranked against the corpus score distribution.
    3. ``knn_neighbours``  — base-rate-corrected vote of the k nearest posts.

    Each judge returns a score in [0, 1] and casts a HQ/LQ vote. The returned
    ``clarity_score`` is the weighted blend; ``ensemble`` reports the breakdown so
    the Synthesizer can explain *why* it scored that way, and flag borderline
    framings where the judges disagreed.
    """
    clean = _clean(text)
    low = clean.lower()

    rule = _judge_rule_signals(low)
    judges: Dict[str, float] = {"rule_signals": rule["score"]}
    details: Dict[str, Any] = {}

    # The data-backed judges need the corpus; if it is missing or unreadable we
    # degrade to the rule judge alone rather than failing the whole turn.
    corpus_ok = True
    try:
        corpus = _load_corpus()
        vec = corpus["vect"].transform([clean])

        learned = _judge_learned_quality(vec, corpus)
        if learned["available"]:
            judges["learned_quality"] = learned["score"]
            details["learned"] = {
                "hq_probability": learned["hq_probability"],
                "percentile": learned["percentile"],
            }

        knn = _judge_knn(vec, corpus)
        judges["knn_neighbours"] = knn["score"]
        details["knn"] = {
            "neighbours": knn["neighbours"],
            "hq_neighbours": knn["hq_neighbours"],
            "corpus_hq_rate": knn.get("corpus_hq_rate"),
        }
    except Exception as exc:  # pragma: no cover - defensive, keeps the duck talking
        details["corpus_error"] = str(exc)
        corpus_ok = False

    # Renormalise the weights over whichever judges actually voted.
    weight_sum = sum(_JUDGE_WEIGHTS[name] for name in judges)
    blended = sum(_JUDGE_WEIGHTS[name] * score for name, score in judges.items()) / weight_sum

    final = round(100 * blended)
    band = "Clear" if final >= 70 else "Needs Work" if final >= 40 else "Unclear"

    votes = {name: ("HQ" if score >= 0.5 else "LQ") for name, score in judges.items()}
    hq_votes = sum(1 for v in votes.values() if v == "HQ")
    consensus = "HQ" if hq_votes * 2 > len(votes) else "LQ"

    return {
        "clarity_score": final,
        "band": band,
        "present": rule["present"],
        "missing": [{"signal": m, "hint": _MISSING_HINT[m]} for m in rule["missing"]],
        "ensemble": {
            "judges": {name: round(score, 3) for name, score in judges.items()},
            "weights": {name: _JUDGE_WEIGHTS[name] for name in judges},
            "votes": votes,
            "consensus": consensus,
            "agreement": f"{max(hq_votes, len(votes) - hq_votes)}/{len(votes)}",
            "unanimous": hq_votes in (0, len(votes)),
            "details": details,
        },
        "grounded_in": os.path.basename(DATA_CSV),
        "corpus_available": corpus_ok,
    }


def find_similar(text: str, tag: str = "", k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve up to k real questions framed like this problem + how they scored."""
    corpus = _load_corpus()
    df, vect, matrix = corpus["df"], corpus["vect"], corpus["matrix"]
    vec = vect.transform([_clean(text)])
    sims = cosine_similarity(vec, matrix)[0]

    results: List[Dict[str, Any]] = []
    for idx in np.argsort(sims)[::-1]:
        row = df.iloc[idx]
        if tag and tag.lower() not in str(row["_tags"]).lower():
            continue
        results.append(
            {
                "title": row["_title"],
                "tags": row["_tags"],
                "quality": row["_label"],
                "similarity": round(float(sims[idx]), 3),
            }
        )
        if len(results) >= k:
            break
    return results


if __name__ == "__main__":
    print("corpus:", DATA_CSV)
    demo = (
        "My SQL join returns duplicate rows and the totals are doubled. "
        "Expected one row per order but I get one per item."
    )
    print("SCORE :", score_clarity(demo))
    print("SIMILAR:", find_similar(demo, tag="sql"))
    print("VAGUE :", score_clarity("plz fix my code it doesnt work"))
