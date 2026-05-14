# langgraph-supervisor-research

> ⚠️ **STALE — UNDER RE-EVALUATION (2026-05-14).** This README describes
> the pre-fix fixture and run. The fixture has since changed (`writer`
> role → `reviewer`, made single-terminal, eval reworded) and the
> measurement bug behind its old `verdict_flip_rate` was fixed. The
> entry has **not** yet been regenerated under the fixed runner — all
> numbers and the `writer`/`FINISH` topology description below are
> obsolete; do not quote them. See `.planning/verdict-flip-rootcause/`
> and the step-20a session handoff.

> 4-agent supervisor pattern (supervisor + researcher + writer +
> critic) deciding whether a draft article is publication-ready.
> Inspired by the LangGraph
> [hierarchical agent teams tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/) (the `agent_supervisor` tutorial it succeeded on the LangGraph docs site).

The fixture uses `add_conditional_edges` with an explicit `path_map`
to route the supervisor's decision into one of three workers (or
`FINISH`). The topofuzz adapter accepts this shape; the topofuzz
runner walks `variant_ir.nodes` linearly at execution time per
`packages/topofuzz/src/topofuzz/runner.py:392`, so every node runs
once per cell regardless of routing.

## Headline metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| `verdict_flip_rate.point` | **0.250** | 25% of (variant, eval) pairs flipped publish/reject |
| `verdict_flip_rate.ci_lower` | **0.105** | 95% CI lower bound — passes the curation gate (>0.05) |
| `verdict_flip_rate.ci_upper` | 0.368 | 95% CI upper bound |
| `kappa_original_vs_permuted` | **0.432** | Just inside the Landis-Koch "moderate" band (0.41–0.60), barely. Far below "substantial" (0.61) |
| `judge_failure_rate` | **0.125** | 4 of 32 judge calls failed — **exceeds the 10% threshold the gallery README warns about (caveat #1)**. Read the 0.250 point estimate as "flippy with high judge noise," not a precision number |
| `n_successful_cells` | 32 / 32 | Full matrix completion |
| `total_cost_usd` | $1.25 | Slightly over the BLUEPRINT 19a `<$1` target |
| `status` | `degraded` | Reflects a structural metadata bug (eval-axis ordering mismatch in attestation path), NOT cell failures |

## A real flip — `art_001`

The article: a 1200-word technical explainer on why TypeScript's
strict mode catches more bugs than JavaScript's runtime checks. Cites
three peer-reviewed studies, links the official TS handbook, written
by an author with 8 years of engineering tenure. **By any reasonable
editorial standard, this is publishable.**

(`report.json`'s `voo_per_eval[art_001]` carries `degenerate_flag=true`
because one of the four variants' judge calls failed to emit a
parseable score for this row. The two contrasting scores below are
real successful judge calls on the surviving variants — see
`runs/<run_id>/run.sqlite` table `judge_results` to verify.)

Two variants of the SAME 4-agent topology reached opposite critic
conclusions:

**APPROVE variant** (judge score: 0.8)
> *Critic: Overall Assessment: Both agents have demonstrated a strong
> understanding of the task. The researcher laid a solid technical
> foundation, and the writer skillfully translated that into a
> detailed, structured article plan that meets all specified
> criteria. There are no apparent gaps, factual errors, or
> misinterpretations. The pipeline is progressing very effectively.*

**DENY variant** (judge score: 0.1)
> *Critic: ... the supervisor and researcher agents must re-submit
> their complete, untruncated outputs to allow the writer to proceed
> with an informed and optimal draft. The writer's concerns are
> valid and directly impact the quality and completeness of the
> final article. The system needs to ensure that inter-agent
> communication is complete and unambiguous.*

The DENY variant's failure mode is **process breakdown**, not editorial
judgment — under one ordering the chain decides "the inputs are
incomplete, we can't proceed." Under the other ordering, the chain
decides "the work is done well." Same article, same Gemini Flash
model, same temperature (0.0), same prompt template. Only the agent
ordering changed.

This is a particularly informative flip because the chain didn't
disagree about the article — it disagreed about whether it had enough
information to ship. **A real editorial workflow built on this
topology would silently reject good articles whenever the worker
ordering happened to land in the wrong configuration.**

## Run details

```
Run id:        2026-05-14-01KRJ37GT0QB9175G36HSWGQDD
Fixture:       examples/gallery/langgraph-supervisor-research/fixture.py:build_graph
Eval set:      examples/gallery/langgraph-supervisor-research/eval.csv (8 of 10 rows used)
Rubric:        examples/gallery/langgraph-supervisor-research/rubric.txt
Judge model:   google/gemini-2.5-flash (single judge — see gallery README caveats)
Variants:      4 (capped per --variants 4)
RNG seed:      42
Cost cap:      $2.00
Actual cost:   $1.25
Wall-clock:    147s
```

## Reproduce

```powershell
$env:GEMINI_API_KEY = "<your-key>"
.venv\Scripts\python.exe -m topofuzz run `
    examples/gallery/langgraph-supervisor-research/fixture.py:build_graph `
    --eval examples/gallery/langgraph-supervisor-research/eval.csv `
    --rubric examples/gallery/langgraph-supervisor-research/rubric.txt `
    --n 8 --variants 4 `
    --judge google/gemini-2.5-flash `
    --rng-seed 42 `
    --out runs/supervisor-repro `
    --max-cost-usd 2.00 --yes
.venv\Scripts\python.exe -m topofuzz report runs/supervisor-repro/<run_id>
```

(Replace `<run_id>` with the directory name printed by `topofuzz run`
— for the published run that's `2026-05-14-01KRJ37GT0QB9175G36HSWGQDD`.)

## Source attribution

- Fixture topology: inspired by the LangGraph
  [hierarchical agent teams tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/)
  (which succeeded the older `agent_supervisor` tutorial on the
  LangGraph docs site). The published tutorial wires LLM-driven
  routing; this fixture uses a stub router so the topology *shape* is
  captured but the topofuzz runner can dispatch real LLM calls per
  node — see `fixture.py` for details. **What is under test here is
  the topology shape under variant generation, NOT a faithful replay
  of the tutorial's runtime routing logic.** A reader running
  `build_graph()` directly through LangGraph (not topofuzz) will see
  deterministic routing to `researcher` only, because the router
  function is intentionally stubbed.
- Rubric: hand-written for this gallery entry
- Eval rows: synthetic, illustrative editorial scenarios — any
  resemblance to real article submissions is coincidental
