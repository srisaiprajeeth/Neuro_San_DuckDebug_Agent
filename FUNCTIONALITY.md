# DuckDebug — Functionality

What the system does, from the user's side. For internal structure see
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## The premise

Rubber-duck debugging works because explaining a problem out loud forces you to structure it. A real
rubber duck, however, knows nothing. DuckDebug is a rubber duck with **145,000 worked examples** of how
problems get solved — and how they get solved badly.

It never opens by answering. It opens by challenging how the problem was framed, and grounds every number
it reports in labeled historical data rather than in the model's opinion.

---

## Workflow A — Coaching

**Use it when:** you are personally stuck on a coding or reasoning problem.

**Trigger phrases:** "my join returns duplicates", "why is this undefined", "I can't figure out why X".

### What you get

1. **A clarity score, 0–100, with a band** — Clear / Needs Work / Unclear.
2. **A missing-pieces checklist** — specifically which of the four hallmarks of a good bug report you left
   out, each with a hint:
   - `expected_vs_actual` — state what you expected vs what actually happened
   - `error_or_symptom` — include the exact error message
   - `context_or_code` — add the code, query, or what you already tried
   - `specificity` — name the concrete technology and values
3. **Real questions framed like yours**, each with its quality label, so you can compare a good and a bad
   framing of your own problem.
4. **Challenged assumptions** — 2–3 hidden assumptions plus "have you checked…?" probes.
5. **Your problem, restated clearly** — one crisp paragraph, plus 2–3 concrete next steps.

### The clarity score in practice

| Your input | Score | Band | Judge agreement |
| --- | --- | --- | --- |
| "Docker container exits immediately with code 0. Expected a long-running service, actual: exits right away." | 79 | Clear | 3/3 |
| "My pandas merge gives NaN columns and I dont know why" | 64 | Needs Work | 3/3 |
| "My SQL join returns duplicate rows and totals are doubled. Expected one row per order but I get one per item." | 44 | Needs Work | 2/3 |
| "plz fix my code it doesnt work" | 17 | Unclear | 3/3 |

The score is an **ensemble vote of three judges**, and disagreement is reported rather than hidden. The
SQL row above scores 2/3 because one judge dissented — the system says so instead of projecting false
confidence.

If your framing is Unclear or Needs Work, the duck leads with a warm request for the single most important
missing piece before answering, so the next round scores better.

---

## Workflow B — Incident triage

**Use it when:** you are investigating an operational incident affecting users or customers.

**Trigger phrases:** "login is broken for everyone", "API latency spiked", "are we breaching SLA", "ticket
volume is climbing".

### What you get

A five-stage chain over 100,000 historical support tickets:

1. **Challenged framing** — before any data is touched.
2. **Similar historical tickets** — with their resolutions, resolution times, reopen flags and CSAT.
3. **The common failure pattern** — dominant product area and issue type, what actually resolved those
   tickets, the share that got resolved at all, median resolution time, reopen rate. Aggregated over ~60
   neighbours, not the handful you were shown.
4. **SLA and trend verification** — breach rate, reopen rate, CSAT and month-over-month volume, each
   reported **beside the corpus-wide baseline** for the same measure.
5. **A prioritised action plan** — restricted to the numbers the earlier stages produced.

### A real finding

Asking *"Users can't log in and password reset emails are never delivered. Are we breaching SLA, and is it
getting worse?"* scopes to `login_auth` + `account_access` and returns:

| Measure | This slice | Corpus baseline | Verdict |
| --- | --- | --- | --- |
| SLA breach | **57.9%** | 54.8% | worse |
| Reopen rate | **6.0%** | 5.0% | worse |
| CSAT | **2.78** | 3.20 | worse |
| Volume trend | **+9.0%** | — | rising |

Root cause finds that **88.9%** of the 60 nearest tickets were resolved by *"Reset account credentials and
confirmed successful login"* — an obvious automation target, which is what the action plan leads with.

### It also reports the absence of a finding

Ask *"Are billing tickets performing worse than everything else?"* and it returns 55.0% breach against a
54.8% baseline, CSAT 3.2 vs 3.2, volume −1.8% — **in line with baseline**. The recommendation agent is
explicitly instructed to say so rather than manufacture urgency.

### It will contradict a false premise

Ask *"Our platinum SLA customers get worse service than standard — confirm it"* and the data refutes the
premise: platinum 54.5% breach, gold 54.8%, standard 54.9%, with CSAT 3.21 / 3.19 / 3.20. The premium tier
is not being underserved; the tiers are indistinguishable.

---

## Workflow C — Full sweep

**Use it when:** the problem is *both* a code-level defect and a customer-facing incident.

**Trigger phrases:** "give me the full analysis", "both angles", "everything you have" — or simply
describing a bug that is also generating tickets.

This runs both corpora over one problem and returns two clearly separated reads:

- **The engineering problem** — clarity score, comparable developer questions, challenged assumptions.
- **The operational picture** — similar tickets, failure pattern, SLA and trend position, action plan.

The duck then says which one it would act on first, and why.

### Worked example

> *"Customers cannot log in — the system says their password is incorrect, and password reset emails never
> arrive. Our Django auth code raises an 'Invalid token' error on the reset link; I tried regenerating the
> token. Expected the reset link to authenticate the user, actual: it fails with invalid credentials.
> Tickets are piling up — give me the full analysis, both angles."*

| Stage | Real output |
| --- | --- |
| Clarity score | **74/100, Clear**, 3/3 judges agreeing; flags `specificity` as the remaining gap |
| Comparable questions | Auth/reset questions, but only **~0.4 similarity** — reported as weak |
| Similar tickets | **0.843 similarity** to real "cannot log in / password incorrect" tickets |
| Failure pattern | `account_access` **100%**; **88.9%** resolved by "reset account credentials" |
| Metrics | **57.9% breach vs 54.8%**, CSAT **2.78 vs 3.20**, volume **+9%** |

Note the deliberate honesty in row 2. The Stack Overflow corpus is developer Q&A and contains nothing
about customer-facing outages, so its matches on an operational question are genuinely weak. The duck is
instructed to say so rather than present a 0.4 similarity as though it were evidence.

---

## Guarantees and limits

**Guarantees**

- Every figure shown to the user comes from a coded tool, never from the language model. The
  recommendation agent is instructed to use only numbers present in its inputs and never to invent a
  ticket ID, percentage or hour count.
- Rates state the population they were measured on. Roughly 40% of tickets are still open and have no
  resolution time; they are excluded rather than counted as fast.
- Results are identical whether the data tools are called in-process or over MCP.

**Limits**

- The chain reports **correlated historical resolutions, not a verified causal root cause.** The
  recommendation agent is required to say "the pattern indicates", not "the root cause is proven to be".
- The ticket corpus is **synthetic**. Its aggregate behaviour is realistic but it is not a record of real
  incidents.
- SLA targets are a configured policy (urgent 4h / high 12h / medium 24h / low 48h), not a property of the
  data. They are overridable via `DUCK_SLA_TARGETS`.
- **The two corpora cover different domains.** Stack Overflow speaks to code-level defects; the ticket
  corpus speaks to operational impact. Workflow C runs both, but on a purely operational question the
  developer-corpus matches will be weak — by design, it reports that rather than hiding it.
- One turn runs one workflow. A vaguely-stated incident is fine in two turns: let the duck coach you into
  a sharper statement, then run the chain.

---

## Data sources

| Corpus | Rows | Source |
| --- | --- | --- |
| Stack Overflow questions, labeled HQ / LQ_CLOSE / LQ_EDIT | 45,000 | [60k Stack Overflow Questions with Quality Rating](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate) (Apache 2.0 / MIT) |
| Synthetic IT support tickets, 2022–2025 | 100,000 | [Synthetic IT Support Tickets](https://www.kaggle.com/datasets/ahsanneural/synthetic-it-support-tickets) |

The coaching workflow ships with a small bundled sample, so it runs before either dataset is downloaded.
