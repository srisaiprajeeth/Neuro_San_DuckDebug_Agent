# DuckDebug — Architecture

Internal structure and design decisions. For the user-facing feature set see
[FUNCTIONALITY.md](FUNCTIONALITY.md).

---

## 1. Agent topology

Nine agents, one front-man, two orchestration shapes.

```
                            the_duck  (front-man, llm_agent)
                                 |
   WORKFLOW A — COACHING (MapReduce)
   +-----------------------+-----------------------+
   |                       |                       |
clarity_scorer      echo_retriever         lateral_thinker        <- MAP
 (coded_tool)         (coded_tool)            (llm_agent)
   |                       |                       |
   +-----------------------+-----------------------+
                                 |
                           synthesizer                            <- REDUCE
                            (llm_agent)

   WORKFLOW B — INCIDENT TRIAGE (sequential chain)
lateral_thinker -> ticket_search -> root_cause -> metric_agent -> recommendation
  (llm_agent)      (coded_tool)     (coded_tool)   (coded_tool)    (llm_agent)

   WORKFLOW C — FULL SWEEP (both of the above, two separated reads)
```

### Routing

The front-man selects one of three workflows per turn:

| Workflow | Shape | Trigger | Nodes touched |
| --- | --- | --- | --- |
| A — Coaching | MapReduce | A problem the user is personally stuck on | 5 |
| B — Incident triage | Sequential chain | An operational incident affecting users | 6 |
| C — Full sweep | Both | A code defect that is *also* generating tickets, or an explicit "full analysis / both angles" | 9 |

Workflow C exists because the two corpora answer different questions about the same event: Stack Overflow
speaks to the code-level defect, the ticket corpus to the operational impact. It runs A's MAP stages and
B's chain over one problem, then returns "The engineering problem" and "The operational picture"
separately, with a stated view on which to act on first. `lateral_thinker` is shared by all three
workflows, which is why 5 + 6 covers 9 rather than 11 agents.

| Agent | Type | Role |
| --- | --- | --- |
| `the_duck` | LLM (front-man) | Routes each turn to one of three workflows; orchestrates its stages |
| `clarity_scorer` | Coded tool | Ensemble clarity score + missing-pieces checklist |
| `echo_retriever` | Coded tool | Retrieves labeled Stack Overflow questions |
| `lateral_thinker` | LLM | Challenges hidden assumptions (shared by both workflows) |
| `synthesizer` | LLM | REDUCE step — folds the three MAP outputs into one answer |
| `ticket_search` | Coded tool | Similar historical tickets |
| `root_cause` | Coded tool | Aggregated failure pattern |
| `metric_agent` | Coded tool | SLA breach, reopen, CSAT, volume trend vs baseline |
| `recommendation` | LLM | Prioritised action plan |

The front-man is identified by having **no `parameters` block** on its `function` — that is neuro-san's
marker for the agent that talks to the outside world. Every other agent declares explicit parameters, which
is what forces data to flow between stages rather than being paraphrased.

### Why the chain is sequential, not fan-out

This is the load-bearing design decision of workflow B:

- `root_cause` re-uses the **exact text and filters** given to `ticket_search`, so both describe the same
  neighbourhood.
- `metric_agent` is scoped by the `product_area` / `issue_type` that `root_cause` reported as
  **dominant** — not by the front-man's opening guess.

So the metrics measure the pattern the data actually shows. Fanning these out in parallel would have let
stage 4 measure a slice that stage 3 had already disproved.

---

## 2. Layering

```
registries/duckdebug.hocon          <- agent declarations, instructions, tool schemas
        |
coded_tools/duckdebug/*.py          <- CodedTool wrappers (arg coercion, guards, MCP routing)
        |
   +----+----------------------------+
   |                                 |
duck_core.py                   ticket_core.py      <- pure logic, no neuro-san imports
   |                                 |
Stack Overflow corpus          IT ticket corpus
```

`duck_core.py` and `ticket_core.py` hold **all** scoring and aggregation logic and import nothing from
neuro-san, so they can be run standalone:

```bash
python coded_tools/duckdebug/duck_core.py
python coded_tools/duckdebug/ticket_core.py
```

The CodedTool wrappers stay thin on purpose: argument coercion (the LLM often sends `k` as a string),
empty-input guards, clamping, and the MCP-vs-local decision. Both the wrappers and the MCP server call the
same core functions, so there is exactly one implementation of every number.

---

## 3. MCP integration

All five data tools are exposed by `servers/mcp/duck_mcp.py` over **streamable HTTP** on port 8100:

| MCP tool | Corpus |
| --- | --- |
| `score_clarity`, `find_similar` | Stack Overflow |
| `find_tickets`, `failure_pattern`, `sla_metrics` | IT tickets |

The coded tools act as MCP **clients** via `langchain-mcp-adapters`, and fall back to calling the core
modules in-process if the server is not running. Verified: both paths return identical results.

Design notes:

- **Warm-up happens at import time**, before the server accepts connections. Building the caches lazily
  inside the first request was several times slower and stranded the user's first turn behind it.
- **The tool list is discovered once per process.** Re-listing tools on every call doubled latency.
- **The URL has no trailing slash** — `/mcp/` triggers a 307 redirect, costing an extra round trip per
  call.

Measured: first call ~15–25s (client-side discovery), then ~2s per MCP round trip, versus ~80ms
in-process. `DUCK_USE_MCP=0` disables MCP entirely.

---

## 4. Algorithms

### Ensemble voting (clarity score)

Three independent judges each score the framing in `[0, 1]` and cast an HQ/LQ vote:

| Judge | Weight | Method |
| --- | --- | --- |
| `rule_signals` | 0.4 | Regex hallmarks of a good bug report. The only judge that can explain *which* piece is missing. |
| `learned_quality` | 0.4 | Logistic regression on TF-IDF, trained on the corpus HQ/LQ labels — **86% holdout accuracy** (stratified 80/20) — then percentile-ranked against the corpus score distribution. |
| `knn_neighbours` | 0.2 | Similarity-weighted vote of the 5 nearest labeled posts, base-rate corrected. Lowest weight because TF-IDF neighbours track topic as much as quality. |

`clarity_score` is the weighted blend; `consensus` and `agreement` come from the votes. Weights are
renormalised over whichever judges actually voted, so the score degrades gracefully if the corpus is
unavailable.

**Why voting beats a single number:** "SQL join returns duplicate rows… expected one row per order" scores
rule 0.50, learned 0.60, knn 0.00 — a 2/3 split. The kNN dissent is a *topic* effect (SQL beginner posts
skew low-quality), not a framing flaw. Surfacing the split lets the Synthesizer flag it as borderline; a
blended scalar would have buried it.

### Retrieval

TF-IDF (unigrams + bigrams, English stop words) with cosine similarity over both corpora. The ticket
index is built on the customer's `initial_message` only — folding in `resolution_summary` would leak the
outcome into the match and inflate similarity for already-solved tickets.

### SLA measurement

Targets are keyed on `priority` (urgent 4h / high 12h / medium 24h / low 48h, overridable via
`DUCK_SLA_TARGETS`). Every rate is reported next to the corpus-wide baseline for the same measure, because
a breach rate is uninterpretable alone.

---

## 5. Four decisions the data forced

Each was discovered by measurement, and each changed the implementation:

1. **The corpus is indexed on a "framing view"** — title plus the first 300 characters, not whole posts.
   A user types two sentences; an HQ post is a long HTML document full of code. Comparing them directly is
   a domain mismatch that dragged *every* short input toward "low quality", and trailing code dumps
   swamped retrieval.
2. **Classifier probabilities are percentile-ranked.** A well-framed two-sentence problem scores only ~0.31
   raw probability — which reads as "bad" but is the 60th percentile of real posts. The judge reports the
   percentile, so the number means "better framed than X% of real Stack Overflow questions".
3. **SLA is keyed on `priority`, not `sla_plan`.** The obvious SLA column carries no signal: standard, gold
   and platinum all show a ~30h median resolution time and identical ~5% reopen rates. `priority` is
   cleanly tiered (urgent ~5h → low ~46h). Keying on the plan would have produced three identical numbers
   and called it an SLA report.
4. **Only ~60% of tickets have a resolution time.** The rest are open, in progress, or on hold. Every rate
   states its denominator instead of treating unresolved tickets as fast ones.

---

## 6. Performance

Fitting TF-IDF and the classifier over both corpora is expensive, so results are cached twice: in-process
via `lru_cache`, and on disk via joblib under `data/.duckdebug_cache/`. The cache key is a fingerprint of
corpus path, size and mtime, plus the indexing parameters and the **scikit-learn version** — a pickled
estimator is not portable across versions, so an upgrade must invalidate rather than silently load a stale
object.

| Path | Cost |
| --- | --- |
| Cold build (both corpora) | ~10s each |
| Warm load from disk cache | **0.26s** |
| Clarity score, warm | ~80ms |
| Ticket metrics, warm | ~50ms |

This removed a 75s stall on the first user turn. Delete `data/.duckdebug_cache/` to force a refit.

---

## 7. Stack and configuration

- **Framework:** neuro-san / Neuro AI Multi-Agent Accelerator
- **LLM:** Azure OpenAI `gpt-4o-mini`, configured in `config/llm_config.hocon`. Endpoint, deployment and
  API version are set inline there rather than as env vars, because neuro-san's `AzureLlmPolicy` reads
  config first and its expected env-var names are easy to get wrong (`AZURE_OPENAI_DEPLOYMENT_NAME`, not
  `..._DEPLOYMENT`; `OPENAI_API_VERSION`, not `AZURE_OPENAI_API_VERSION`). Only the API key lives in
  `.env`. Gemini is retained as a commented fallback.
- **ML:** scikit-learn, pandas, numpy, joblib
- **MCP:** `langchain-mcp-adapters`, FastMCP over streamable HTTP

### Key files

| Path | Purpose |
| --- | --- |
| `registries/duckdebug.hocon` | The 9-agent network |
| `coded_tools/duckdebug/duck_core.py` | Clarity ensemble + Stack Overflow retrieval |
| `coded_tools/duckdebug/ticket_core.py` | Ticket search, pattern aggregation, SLA metrics |
| `servers/mcp/duck_mcp.py` | MCP server for all five data tools |
| `config/llm_config.hocon` | Single LLM switch for every agent |
| `docs/examples/duckdebug.md` | Run instructions and debugging hints |

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `AZURE_OPENAI_API_KEY` | — | Required; the only secret |
| `DUCK_USE_MCP` | `1` | `0` disables MCP, running data tools in-process |
| `DUCK_MCP_URL` | `http://localhost:8100/mcp` | MCP server address |
| `DUCK_DATA_CSV` | `data/stackoverflow_train.csv` | Stack Overflow corpus, falls back to bundled sample |
| `DUCK_TICKET_CSV` | `data/it_support_tickets.csv` | Ticket corpus |
| `DUCK_SLA_TARGETS` | `urgent=4,high=12,medium=24,low=48` | SLA policy, in hours |
