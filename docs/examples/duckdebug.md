# DuckDebug

**DuckDebug** is the rubber duck that knows what good debugging looks like.

It runs two workflows off one front-man, and always starts by challenging assumptions rather than
accepting the problem as stated:

- **Coaching** (MapReduce) — you explain a stuck problem. It scores how clearly you framed it against
  45,000 labeled Stack Overflow questions, shows how other people framed the same issue, challenges your
  assumptions, and reflects the problem back sharper than you stated it. Often that reflection is the fix.
- **Incident triage** (sequential chain) — you report an operational incident. It finds similar historical
  tickets in a 100,000-row IT support corpus, extracts the common failure pattern, checks the slice
  against SLA targets and volume trends, and produces an action plan.

---

## Files

- Network: [duckdebug.hocon](../../registries/duckdebug.hocon)
- Coded tools: [`coded_tools/duckdebug/`](../../coded_tools/duckdebug/)
    - [duck_core.py](../../coded_tools/duckdebug/duck_core.py) — clarity scoring + retrieval, no neuro-san imports
    - [ticket_core.py](../../coded_tools/duckdebug/ticket_core.py) — ticket search, pattern aggregation, SLA metrics
    - [clarity_scorer.py](../../coded_tools/duckdebug/clarity_scorer.py) — coaching MAP 1
    - [echo_retriever.py](../../coded_tools/duckdebug/echo_retriever.py) — coaching MAP 2
    - [ticket_search.py](../../coded_tools/duckdebug/ticket_search.py) — chain stage 2
    - [root_cause.py](../../coded_tools/duckdebug/root_cause.py) — chain stage 3
    - [metric_agent.py](../../coded_tools/duckdebug/metric_agent.py) — chain stage 4
    - [duck_mcp_client.py](../../coded_tools/duckdebug/duck_mcp_client.py) — MCP client + local fallback
- MCP server: [servers/mcp/duck_mcp.py](../../servers/mcp/duck_mcp.py) — exposes all five data tools
- Data: `data/stackoverflow_train.csv`, `data/stackoverflow_sample.csv` (fallback),
  `data/it_support_tickets.csv`

---

## Prerequisites

```bash
pip install pandas scikit-learn
```

The network runs with no further setup: it prefers `data/stackoverflow_train.csv` and falls back to the
tiny bundled sample if that file is absent.

### Data

The full corpus is the training split of
[60k Stack Overflow Questions with Quality Rating](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate)
(Annamoradnejad et al., 2022 — Apache 2.0 / MIT): 45,000 rows, evenly split across `HQ`, `LQ_CLOSE`, and
`LQ_EDIT`. Place it at `data/stackoverflow_train.csv`, or point anywhere with:

```bash
export DUCK_DATA_CSV=/path/to/60k_stackoverflow.csv
```

### Optional: route the data tools through MCP

The two data tools call an MCP server when one is running, and run in-process otherwise. To make the MCP
hop real and visible in the nsflow graph, start the server from the repo root:

```bash
python servers/mcp/duck_mcp.py
```

It serves `score_clarity` and `find_similar` over streamable HTTP on `http://localhost:8100/mcp`. Set
`DUCK_USE_MCP=0` to skip MCP entirely, or `DUCK_MCP_URL` to point at a different address.

Both paths return identical results, but they are not equally fast. In-process calls take ~80ms once the
cache is warm. Over MCP, the first call in a process pays ~15-25s of client-side tool discovery and each
subsequent round trip costs ~2s, because `langchain-mcp-adapters` opens a fresh session per invocation.
MCP is the default so the data hop is real and visible in the graph; set `DUCK_USE_MCP=0` if you would
rather have the latency back.

---

## Architecture Overview

Nine agents, one front-man, two orchestration shapes.

```
                          the_duck  (front-man)
                              |
        WORKFLOW A - COACHING (MapReduce)
        +---------------------+---------------------+
        |                     |                     |
   clarity_scorer      echo_retriever        lateral_thinker      <- MAP
   (coded tool)         (coded tool)           (llm agent)
        |                     |                     |
        +---------------------+---------------------+
                              |
                        synthesizer                               <- REDUCE
                         (llm agent)

        WORKFLOW B - INCIDENT TRIAGE (sequential chain)
   lateral_thinker  ->  ticket_search  ->  root_cause  ->  metric_agent  ->  recommendation
    (llm agent)         (coded tool)      (coded tool)     (coded tool)      (llm agent)
   challenges           finds similar     extracts common   verifies vs      produces
   assumptions          tickets           failure pattern   SLA + trends     action plan
```

The chain is genuinely sequential, not fan-out: `root_cause` re-uses the text and filters given to
`ticket_search` so both describe the same neighbourhood, and `metric_agent` is scoped by the
`product_area`/`issue_type` that `root_cause` found *dominant* — so the metrics measure the pattern the
data actually shows, not the front-man's opening guess.

### Front-man: `the_duck`

Takes the problem, fans it out to the three MAP agents, then hands all three results to the REDUCE step.
If clarity comes back `Unclear` or `Needs Work`, it leads with a request for the single most important
missing piece instead of guessing at an answer.

### MAP 1: `clarity_scorer` (coded tool)

Returns a 0-100 score, a band, the missing pieces, and the ensemble breakdown. Three independent judges
each score the framing in `[0, 1]` and cast an HQ/LQ vote:

| Judge | Weight | What it measures |
| --- | --- | --- |
| `rule_signals` | 0.4 | Regex hallmarks: expected-vs-actual, error text, code/context, specificity. The only judge that can say *which* piece is missing. |
| `learned_quality` | 0.4 | A logistic-regression HQ/LQ classifier trained on the corpus labels (~0.86 holdout accuracy, stratified 80/20 split), percentile-ranked against the corpus. |
| `knn_neighbours` | 0.2 | Similarity-weighted vote of the 5 nearest labeled posts, base-rate corrected. Weakest weight because TF-IDF neighbours track topic as much as quality. |

The score is the weighted blend; `ensemble.agreement` exposes disagreement so the Synthesizer can flag a
borderline framing rather than pretending to precision it does not have.

### MAP 2: `echo_retriever` (coded tool)

TF-IDF cosine retrieval over the same corpus, with an optional tag filter. Returns each match's title,
tags, quality label, and similarity — so the user can see an HQ and an LQ framing of their own problem
side by side.

### MAP 3: `lateral_thinker` (llm agent)

Names 2-3 hidden assumptions and a couple of "have you checked...?" probes. Deliberately does not solve
the problem; its disagreement is what makes the duck useful.

### REDUCE: `synthesizer` (llm agent)

Folds all three MAP outputs into four sections: the problem restated clearly, the clarity check, how
others framed it, and concrete next steps.

---

## Workflow B: the incident chain

Grounded in `data/it_support_tickets.csv` — 100,000 synthetic IT support tickets spanning 2022-2025,
with product area, issue type, priority, status, resolution summary, resolution time, reopen flag and
CSAT.

| Stage | Agent | What it does |
| --- | --- | --- |
| 1 | `lateral_thinker` (llm) | Challenges how the incident was framed, *before* any data is touched |
| 2 | `ticket_search` (coded) | TF-IDF retrieval over the customer's own words, optionally filtered by product area / issue type / priority |
| 3 | `root_cause` (coded) | Aggregates ~60 neighbours into a pattern: dominant area and issue type, what actually resolved them, resolved share, median resolution time, reopen rate |
| 4 | `metric_agent` (coded) | SLA breach rate, reopen rate, CSAT and month-over-month volume, each beside the corpus baseline |
| 5 | `recommendation` (llm) | Prioritised action plan, restricted to numbers the earlier stages produced |

### Two data findings that shaped the design

1. **SLA is keyed on `priority`, not `sla_plan`.** The corpus's `sla_plan` column looks like the obvious
   SLA key, but it carries no signal: standard, gold and platinum all show a ~30h median resolution time
   and an identical ~5% reopen rate. `priority` is cleanly tiered (urgent ~5h, high ~15h, medium ~30h,
   low ~46h median), so that is what the targets key on. Keying on the plan would have produced three
   identical numbers and called it an SLA report.
2. **Only ~60% of tickets have a resolution time.** The remainder are still open, in progress, or on
   hold. Every rate states the denominator it was measured on rather than quietly treating unresolved
   tickets as fast ones.

Default SLA targets are urgent 4h / high 12h / medium 24h / low 48h, which puts the corpus-wide breach
rate at 54.8%. These are a deliberately demanding target, not a description of the corpus. Override with:

```bash
export DUCK_SLA_TARGETS="urgent=8,high=24,medium=48,low=72"
```

### Why the baseline comparison matters

`metric_agent` always reports the corpus-wide figure next to the slice figure, because a breach rate is
meaningless alone. For `login_auth` overall the slice sits at 55.1% against a 54.8% baseline — i.e. no
worse than everything else, which is a real finding. Narrow to `login_auth` + `account_access` and it
separates properly: **57.9% breach vs 54.8%, reopen 6.0% vs 5.0%, CSAT 2.78 vs 3.20, volume +9%** — worse
on every measure. The `recommendation` agent is explicitly instructed to say "in line with baseline" when
that is what the numbers show, rather than manufacturing a finding.

---

## Two things the data forced

Both are load-bearing, and worth knowing before tuning anything:

1. **The corpus is indexed on a "framing view"** — title plus the first 300 characters of the body, not
   the whole post. A user types the duck two sentences; a real HQ post is a long HTML document full of
   code blocks. Comparing those directly is a domain mismatch that drags every short input toward "low
   quality", and the trailing code dumps swamp TF-IDF retrieval. Truncating fixed both.
2. **Raw classifier probabilities are compressed, so they are percentile-ranked** against the corpus's own
   score distribution. A well-framed two-sentence problem only scores ~0.3 in absolute probability, which
   reads as "bad" but is actually the 60th percentile of real posts. The judge reports the percentile, so
   the number means "you framed this better than X% of real Stack Overflow questions".

---

## Performance

Fitting TF-IDF plus the classifier over 45k rows takes ~10s, so the result is cached both in-process and
on disk (`data/.duckdebug_cache/`, keyed by corpus fingerprint, indexing parameters, and scikit-learn
version). A warm start loads in ~0.3s. The MCP server warms the cache at import time, before it accepts
connections — building it lazily inside the first request is several times slower and would strand the
user's first turn behind it.

Delete `data/.duckdebug_cache/` to force a refit.

---

## Debugging Hints

- `pandas` and `scikit-learn` installed
- `data/stackoverflow_train.csv` present, or `DUCK_DATA_CSV` set — `score_clarity` reports which corpus it
  used in `grounded_in`, and `corpus_available` is `false` if it could not load one
- If scores look oddly uniform, check `ensemble.judges`: a judge stuck at exactly `0.5` is abstaining
- MCP path: server running on port 8100, and `langchain-mcp-adapters` installed. The tools fall back to
  in-process silently, so check the logs for `MCP call ... unavailable` if you expected an MCP hop
