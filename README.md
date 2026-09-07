# DuckDebug in Neuro-San Studio

DuckDebug is a multi-agent debugging and incident-triage assistant built on Neuro-San Studio.
It combines **developer coaching** (how clearly a problem is framed and what to ask next) with
**operational incident analysis** (historical ticket patterns, SLA posture, and action planning).

This README is a practical entry point. For deeper detail, see:

- Feature behavior: [`FUNCTIONALITY.md`](FUNCTIONALITY.md)
- Internal design: [`ARCHITECTURE.md`](ARCHITECTURE.md)

## What DuckDebug does

DuckDebug supports three workflows selected per user turn:

1. **Workflow A — Coaching**
   - Scores problem clarity (0–100 with a quality band)
   - Flags missing bug-report hallmarks (expected vs actual, error text, context/code, specificity)
   - Retrieves similar labeled developer questions
   - Challenges hidden assumptions
   - Produces a clearer restatement plus concrete next steps

2. **Workflow B — Incident triage**
   - Challenges initial framing before analysis
   - Finds similar historical support tickets
   - Aggregates dominant failure pattern and successful resolutions
   - Computes SLA/CSAT/reopen/trend metrics against corpus baselines
   - Generates a prioritized plan constrained to measured evidence

3. **Workflow C — Full sweep**
   - Runs both coaching and incident lenses on the same problem
   - Separates results into:
     - **Engineering problem** (code-level framing/diagnosis)
     - **Operational picture** (customer impact/incident posture)
   - Recommends what to act on first and why

## Agent architecture at a glance

DuckDebug is implemented as a 9-agent network with one front-man orchestrator.

- **Front-man (`the_duck`)** routes each turn to Workflow A, B, or C.
- **Coded tools** provide deterministic scoring/retrieval/metrics.
- **LLM agents** handle assumption-challenge, synthesis, and recommendation narrative.

### Core agents

- `the_duck` (LLM): workflow routing and orchestration
- `clarity_scorer` (coded tool): ensemble clarity score + missing-pieces checklist
- `echo_retriever` (coded tool): similar Stack Overflow retrieval
- `lateral_thinker` (LLM): assumption challenge (shared across workflows)
- `synthesizer` (LLM): combines Workflow A map outputs
- `ticket_search` (coded tool): similar incident retrieval
- `root_cause` (coded tool): dominant failure pattern aggregation
- `metric_agent` (coded tool): SLA/reopen/CSAT/volume metrics vs baseline
- `recommendation` (LLM): prioritized evidence-bounded action plan

## Data and scoring model

### Data sources

- **Stack Overflow quality corpus** (~45k rows used in-project) for framing quality and developer analogues
- **Synthetic IT support tickets** (100k rows) for operational triage patterns

### Clarity score algorithm

The 0–100 clarity score is an ensemble of three judges:

- `rule_signals` (weight 0.4): explicit framing hallmarks and explainable missing pieces
- `learned_quality` (weight 0.4): TF-IDF + logistic regression quality estimate (percentile-ranked)
- `knn_neighbours` (weight 0.2): similarity-weighted nearest-neighbor quality vote

Weights are re-normalized when one judge is unavailable.
Agreement/disagreement between judges is surfaced to avoid false confidence.

## Execution layers

DuckDebug is layered so logic is reusable and consistent across local and MCP execution:

1. **Registry layer**: network declarations, prompts, tool schemas
2. **Coded-tool wrapper layer**: argument coercion, guards, MCP-vs-local routing
3. **Core logic layer**: pure scoring/retrieval/aggregation code
4. **Data layer**: Stack Overflow + ticket corpora

Key implementation paths:

- `registries/duckdebug.hocon`
- `coded_tools/duckdebug/duck_core.py`
- `coded_tools/duckdebug/ticket_core.py`
- `servers/mcp/duck_mcp.py`
- `config/llm_config.hocon`

## MCP support

DuckDebug tools can run:

- **In-process** (fast local path)
- **Over MCP** via streamable HTTP (`duck_mcp` server)

MCP tools include:

- `score_clarity`, `find_similar`
- `find_tickets`, `failure_pattern`, `sla_metrics`

Behavioral guarantee: both paths share the same core functions for consistent outputs.

## Configuration

Primary runtime configuration is in [`config/llm_config.hocon`](config/llm_config.hocon).

Common environment variables:

- `AZURE_OPENAI_API_KEY` (required)
- `DUCK_USE_MCP` (`1` by default; set `0` to force local execution)
- `DUCK_MCP_URL` (default: `http://localhost:8100/mcp`)
- `DUCK_DATA_CSV` (Stack Overflow training CSV path)
- `DUCK_TICKET_CSV` (ticket CSV path)
- `DUCK_SLA_TARGETS` (SLA policy override, hours per priority)

## Guarantees and limits

### Guarantees

- Numeric outputs are produced by coded tools, not free-form model invention.
- Rates are reported with explicit population context.
- Local and MCP paths are designed to return consistent results.

### Limits

- Incident “root cause” is historical-pattern correlation, not causal proof.
- Ticket corpus is synthetic.
- Coaching and incident corpora represent different domains; cross-domain similarity can be weak and is reported as such.

## Repository orientation

This repository contains the full Neuro-San Studio framework plus apps, plugins, servers, middleware,
and tests. DuckDebug-specific logic primarily lives under:

- `coded_tools/duckdebug/`
- `servers/mcp/`
- `registries/`
- `docs/examples/duckdebug.md`

## Where to go next

- Read [`FUNCTIONALITY.md`](FUNCTIONALITY.md) for user-facing behavior and examples.
- Read [`ARCHITECTURE.md`](ARCHITECTURE.md) for design decisions, algorithms, and performance notes.
- Use `docs/` for broader framework usage and integration guides.
