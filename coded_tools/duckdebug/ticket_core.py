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
"""Ticket analytics for DuckDebug's incident-triage chain.

Backs three of the four new stages with real aggregation over a 100k-row IT
support ticket corpus:

    ticket_search  -> find_tickets()     : similar historical tickets
    root_cause     -> failure_pattern()  : the common pattern across them
    metric         -> sla_metrics()      : SLA breach + reopen/CSAT/volume trends

Pure Python + scikit-learn, no neuro-san imports, so it can be exercised
standalone (``python coded_tools/duckdebug/ticket_core.py``).

Two things the data dictated, both worth knowing before tuning anything:

* **SLA is keyed on ``priority``, not ``sla_plan``.** The corpus's ``sla_plan``
  column (standard/gold/platinum) carries no signal — all three plans show a
  ~30h median resolution time and an identical ~5% reopen rate. ``priority``, by
  contrast, is cleanly tiered (urgent ~5h, high ~15h, medium ~30h, low ~46h
  median). Keying SLA on the plan would report every plan as identical and say
  nothing; keying on priority measures something real.
* **Only ~60% of tickets have a resolution time.** The rest are still open /
  in_progress / on_hold. Rate calculations state their denominator rather than
  silently treating unresolved tickets as fast ones.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
from collections import Counter
from typing import Any
from typing import Dict
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "data"))
_DEFAULT_CSV = os.path.join(_DATA_DIR, "it_support_tickets.csv")
TICKET_CSV = os.environ.get("DUCK_TICKET_CSV", _DEFAULT_CSV)

# SLA targets in hours, keyed on priority. Override with DUCK_SLA_TARGETS, e.g.
# "urgent=4,high=12,medium=24,low=48". These defaults sit below each priority's
# p75 resolution time, which puts the corpus-wide breach rate near 55% — a
# deliberately demanding target, not a description of the corpus's own behaviour.
_DEFAULT_SLA_TARGETS = {"urgent": 4.0, "high": 12.0, "medium": 24.0, "low": 48.0}

# How many neighbours the pattern/metric aggregation looks at. Larger than the
# handful shown to the user: a pattern claim over 5 tickets is noise.
_PATTERN_NEIGHBOURS = 60

_MAX_FEATURES = 20000

# CSAT is recorded 0-5 where 0 means "no survey returned", not "rated zero".
# Averaging the zeros in would drag every score down by roughly a full point.
_CSAT_MISSING = 0


def _sla_targets() -> Dict[str, float]:
    """SLA targets per priority, allowing an env override."""
    raw = os.environ.get("DUCK_SLA_TARGETS", "").strip()
    if not raw:
        return dict(_DEFAULT_SLA_TARGETS)

    targets = dict(_DEFAULT_SLA_TARGETS)
    for part in raw.split(","):
        if "=" not in part:
            continue
        name, _, value = part.partition("=")
        try:
            targets[name.strip().lower()] = float(value)
        except ValueError:
            continue
    return targets


def _clean(text: str) -> str:
    """Collapse whitespace so corpus text and typed queries are comparable."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _cache_path() -> str:
    """Disk-cache location, invalidated by corpus identity and sklearn version."""
    try:
        stat = os.stat(TICKET_CSV)
        fingerprint = f"{os.path.abspath(TICKET_CSV)}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        fingerprint = os.path.abspath(TICKET_CSV)
    fingerprint += f"|{_MAX_FEATURES}|sklearn{sklearn_version}"
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_DATA_DIR, ".duckdebug_cache", f"tickets_{digest}.joblib")


@functools.lru_cache(maxsize=1)
def _load_tickets() -> Dict[str, Any]:
    """Load the ticket corpus and fit the retrieval index (cached on disk)."""
    cache_file = _cache_path()
    if os.path.exists(cache_file):
        try:
            return joblib.load(cache_file)
        except Exception:  # pragma: no cover - stale/corrupt cache, refit
            pass

    corpus = _build_tickets()

    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        joblib.dump(corpus, cache_file, compress=3)
    except Exception:  # pragma: no cover - read-only data dir must not be fatal
        pass

    return corpus


def _build_tickets() -> Dict[str, Any]:
    """Read the CSV and fit the TF-IDF index from scratch (the expensive path)."""
    df = pd.read_csv(TICKET_CSV)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ("initial_message", "resolution_summary", "product_area", "issue_type", "priority", "status"):
        if col not in df.columns:
            df[col] = ""

    df["_text"] = df["initial_message"].fillna("").astype(str).map(_clean)
    df["_created"] = pd.to_datetime(df.get("created_at"), errors="coerce")
    df["_month"] = df["_created"].dt.to_period("M").astype(str)

    # Retrieval runs on the customer's own words; the resolution summary is an
    # outcome, so folding it into the query text would leak the answer into the
    # match and inflate similarity for already-solved tickets.
    vect = TfidfVectorizer(stop_words="english", max_features=_MAX_FEATURES, ngram_range=(1, 2))
    matrix = vect.fit_transform(df["_text"])

    keep = [
        "ticket_id",
        "product_area",
        "issue_type",
        "priority",
        "status",
        "sla_plan",
        "initial_message",
        "resolution_summary",
        "resolution_time_hours",
        "reopened",
        "csat_score",
        "customer_segment",
        "region",
        "platform",
        "channel",
        "_text",
        "_month",
    ]
    slim = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    return {"df": slim, "vect": vect, "matrix": matrix, "rows": int(len(slim))}


def warm_up() -> Dict[str, Any]:
    """Force the ticket index to build now and report what was loaded."""
    corpus = _load_tickets()
    return {"corpus": TICKET_CSV, "rows": corpus["rows"], "sla_targets": _sla_targets()}


def _apply_filters(df: pd.DataFrame, filters: Dict[str, str]) -> pd.Series:
    """Build a boolean mask for the given equality filters, ignoring blanks."""
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        if not value or column not in df.columns:
            continue
        mask &= df[column].astype(str).str.lower() == str(value).strip().lower()
    return mask


def _ticket_row(row: pd.Series, similarity: float) -> Dict[str, Any]:
    """Shape one ticket for agent consumption."""
    resolution_hours = row.get("resolution_time_hours")
    csat = row.get("csat_score")
    return {
        "ticket_id": row.get("ticket_id", ""),
        "product_area": row.get("product_area", ""),
        "issue_type": row.get("issue_type", ""),
        "priority": row.get("priority", ""),
        "status": row.get("status", ""),
        "initial_message": row.get("initial_message", ""),
        "resolution_summary": row.get("resolution_summary") if pd.notna(row.get("resolution_summary")) else None,
        "resolution_time_hours": round(float(resolution_hours), 2) if pd.notna(resolution_hours) else None,
        "reopened": bool(row.get("reopened")) if pd.notna(row.get("reopened")) else None,
        "csat_score": int(csat) if pd.notna(csat) and int(csat) != _CSAT_MISSING else None,
        "similarity": round(float(similarity), 3),
    }


def _rank(text: str, filters: Dict[str, str], limit: int) -> "tuple[pd.DataFrame, np.ndarray]":
    """Return the `limit` best-matching rows for `text` within `filters`."""
    corpus = _load_tickets()
    df, vect, matrix = corpus["df"], corpus["vect"], corpus["matrix"]

    mask = _apply_filters(df, filters)
    candidate_idx = np.flatnonzero(mask.values)
    if candidate_idx.size == 0:
        # Filters excluded everything; fall back to the whole corpus rather than
        # returning nothing, and let the caller see filters_applied in the output.
        candidate_idx = np.arange(len(df))

    sims = cosine_similarity(vect.transform([_clean(text)]), matrix[candidate_idx])[0]
    order = np.argsort(sims)[::-1][:limit]
    chosen = candidate_idx[order]
    return df.iloc[chosen], sims[order]


def find_tickets(
    text: str,
    product_area: str = "",
    issue_type: str = "",
    priority: str = "",
    k: int = 5,
) -> Dict[str, Any]:
    """Stage 2 — find historical tickets that look like this problem."""
    filters = {"product_area": product_area, "issue_type": issue_type, "priority": priority}
    rows, sims = _rank(text, filters, max(1, min(int(k), 25)))

    tickets = [_ticket_row(row, sim) for (_, row), sim in zip(rows.iterrows(), sims)]
    return {
        "tickets": tickets,
        "matched": len(tickets),
        "filters_applied": {k2: v for k2, v in filters.items() if v},
        "corpus_rows": _load_tickets()["rows"],
        "grounded_in": os.path.basename(TICKET_CSV),
    }


def _top_counts(series: pd.Series, n: int = 3) -> List[Dict[str, Any]]:
    """Top-n value counts with shares, for reporting a dominant pattern."""
    total = int(series.notna().sum())
    if not total:
        return []
    out = []
    for value, count in series.value_counts().head(n).items():
        out.append({"value": str(value), "count": int(count), "share_pct": round(100 * count / total, 1)})
    return out


def _resolution_themes(summaries: pd.Series, n: int = 5) -> List[Dict[str, Any]]:
    """Most frequent resolution phrasings among the neighbourhood.

    The corpus's resolution summaries are templated, so exact-text counts are a
    more honest "what actually fixed this" signal than keyword extraction would be.
    """
    cleaned = summaries.dropna().astype(str).map(_clean)
    cleaned = cleaned[cleaned.str.len() > 0]
    if cleaned.empty:
        return []
    total = len(cleaned)
    return [
        {"resolution": text, "count": int(count), "share_pct": round(100 * count / total, 1)}
        for text, count in Counter(cleaned).most_common(n)
    ]


def failure_pattern(
    text: str,
    product_area: str = "",
    issue_type: str = "",
    priority: str = "",
    neighbours: int = _PATTERN_NEIGHBOURS,
) -> Dict[str, Any]:
    """Stage 3 — extract the common failure pattern across similar tickets.

    Aggregates over a wider neighbourhood than the user sees, because a pattern
    claimed from 5 tickets is noise.
    """
    filters = {"product_area": product_area, "issue_type": issue_type, "priority": priority}
    n = max(10, min(int(neighbours), 500))
    rows, sims = _rank(text, filters, n)

    resolved = rows[rows["resolution_time_hours"].notna()] if "resolution_time_hours" in rows else rows.iloc[0:0]
    reopened = rows["reopened"].dropna() if "reopened" in rows else pd.Series(dtype=float)

    return {
        "neighbourhood_size": int(len(rows)),
        "mean_similarity": round(float(np.mean(sims)), 3) if len(sims) else 0.0,
        "dominant_product_area": _top_counts(rows["product_area"]),
        "dominant_issue_type": _top_counts(rows["issue_type"]),
        "priority_mix": _top_counts(rows["priority"], n=4),
        "status_mix": _top_counts(rows["status"], n=5),
        "common_resolutions": _resolution_themes(rows["resolution_summary"]),
        "resolved_share_pct": round(100 * len(resolved) / len(rows), 1) if len(rows) else 0.0,
        "median_resolution_hours": round(float(resolved["resolution_time_hours"].median()), 2)
        if len(resolved)
        else None,
        "reopen_rate_pct": round(100 * float(reopened.mean()), 1) if len(reopened) else None,
        "filters_applied": {k2: v for k2, v in filters.items() if v},
        "grounded_in": os.path.basename(TICKET_CSV),
    }


def _breach_stats(frame: pd.DataFrame, targets: Dict[str, float]) -> Dict[str, Any]:
    """SLA breach rate over the rows that actually have a resolution time."""
    resolved = frame[frame["resolution_time_hours"].notna()]
    if resolved.empty:
        return {"measured_on": 0, "breach_rate_pct": None}

    target_hours = resolved["priority"].astype(str).str.lower().map(targets)
    breached = resolved["resolution_time_hours"] > target_hours
    # Rows whose priority has no configured target can't be judged either way.
    judged = target_hours.notna()
    if not bool(judged.any()):
        return {"measured_on": 0, "breach_rate_pct": None}

    return {
        "measured_on": int(judged.sum()),
        "breach_rate_pct": round(100 * float(breached[judged].mean()), 1),
        "median_resolution_hours": round(float(resolved.loc[judged, "resolution_time_hours"].median()), 2),
    }


def sla_metrics(
    product_area: str = "",
    issue_type: str = "",
    priority: str = "",
    recent_months: int = 6,
) -> Dict[str, Any]:
    """Stage 4 — verify a slice against SLA targets and incident trends.

    Every rate is reported against the corpus-wide baseline for the same measure,
    because "55% breach" only means something next to the global 55%.
    """
    corpus = _load_tickets()
    df = corpus["df"]
    targets = _sla_targets()

    filters = {"product_area": product_area, "issue_type": issue_type, "priority": priority}
    mask = _apply_filters(df, filters)
    slice_df = df[mask]
    scoped = bool(any(filters.values())) and not slice_df.empty
    if not scoped:
        slice_df = df

    slice_sla = _breach_stats(slice_df, targets)
    baseline_sla = _breach_stats(df, targets)

    def _rate(frame: pd.DataFrame, column: str) -> Any:
        if column not in frame:
            return None
        values = frame[column].dropna()
        return round(100 * float(values.mean()), 1) if len(values) else None

    def _csat(frame: pd.DataFrame) -> Any:
        if "csat_score" not in frame:
            return None
        rated = frame["csat_score"].dropna()
        rated = rated[rated != _CSAT_MISSING]
        return round(float(rated.mean()), 2) if len(rated) else None

    # Volume trend: the most recent N months against the preceding baseline.
    months = sorted(m for m in slice_df["_month"].dropna().unique() if m and m != "NaT")
    window = max(1, int(recent_months))
    recent_months_list = months[-window:]
    prior_months_list = months[:-window]
    recent_per_month = len(slice_df[slice_df["_month"].isin(recent_months_list)]) / max(1, len(recent_months_list))
    prior_per_month = (
        len(slice_df[slice_df["_month"].isin(prior_months_list)]) / len(prior_months_list)
        if prior_months_list
        else None
    )
    trend_pct = round(100 * (recent_per_month - prior_per_month) / prior_per_month, 1) if prior_per_month else None

    return {
        "scope": {k2: v for k2, v in filters.items() if v} or "entire corpus",
        "scope_tickets": int(len(slice_df)),
        "sla_targets_hours": targets,
        "sla_keyed_on": "priority",
        "sla": slice_sla,
        "sla_corpus_baseline": baseline_sla,
        "reopen_rate_pct": _rate(slice_df, "reopened"),
        "reopen_rate_corpus_pct": _rate(df, "reopened"),
        "csat_mean": _csat(slice_df),
        "csat_mean_corpus": _csat(df),
        "volume_trend": {
            "recent_months": len(recent_months_list),
            "recent_tickets_per_month": round(recent_per_month, 1),
            "prior_tickets_per_month": round(prior_per_month, 1) if prior_per_month else None,
            "change_pct": trend_pct,
        },
        "note": (
            "SLA is keyed on priority because the corpus's sla_plan column is non-discriminative "
            "(identical ~30h medians across standard/gold/platinum). Rates are measured only on "
            "tickets that have a resolution time; ~40% of the corpus is still open."
        ),
        "grounded_in": os.path.basename(TICKET_CSV),
    }


if __name__ == "__main__":
    import json

    print("warm:", json.dumps(warm_up(), indent=2, default=str))
    query = "Users cannot log in, password reset emails are never delivered."

    found = find_tickets(query, product_area="login_auth", k=3)
    print("\nSEARCH:", json.dumps(found, indent=2, default=str)[:1200])

    pattern = failure_pattern(query, product_area="login_auth")
    print("\nPATTERN:", json.dumps(pattern, indent=2, default=str)[:1400])

    metrics = sla_metrics(product_area="login_auth")
    print("\nMETRICS:", json.dumps(metrics, indent=2, default=str)[:1400])
