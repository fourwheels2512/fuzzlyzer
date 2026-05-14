# langgraph-content-moderation — DISCARDED

> 4-agent linear pipeline (extract → classify → policy_check →
> decision) for user-generated content moderation. **Discarded from
> the v0.5 gallery on 2026-05-13** because it failed the
> `verdict_flip_rate.ci_lower > 0.05` curation gate. Source code
> retained as a transparency artifact.

This entry lives in the gallery as evidence that we did not
cherry-pick. The published gallery's "examples where ordering matters"
narrative is more credible if you can see what we tried that didn't
qualify.

## Why it was discarded

The run completed cleanly (32/32 cells, 6.25% judge failure rate,
$0.79 cost), but the headline metrics flagged genuine stability.

> **Note**: the numbers below come from a local 2026-05-13 run on
> the publishing host. The `runs/` directory is intentionally NOT
> committed (no `report.html` to host), so these numbers are not
> independently auditable from the gallery itself. To verify, follow
> the "If you want to investigate" reproducer at the bottom — your
> numbers will differ slightly because Gemini Flash responses are
> not deterministic across calls, but the **stability finding**
> (kappa near 1.0, low verdict_flip_rate.point) should reproduce.

| Metric | Value (2026-05-13 local run) | Reading |
|--------|------------------------------|---------|
| `verdict_flip_rate.point` | 0.0 | No (variant, eval) pairs disagreed on the binary verdict |
| `verdict_flip_rate.degenerate_flag` | true | Indicates the metric is not meaningful |
| `kappa_original_vs_permuted` | 1.0 | Perfect agreement across all variants — Landis-Koch "almost perfect" |

Per BLUEPRINT_v5plus.md line 439:

> *Curation note: if a tutorial chain shows low variance, do not
> include it — the gallery is "examples where ordering matters,"
> not a benchmark.*

A "stable" example in a "flippy examples" gallery is a credibility
loss, so this candidate was excluded.

## Why it was stable — and what it teaches about the metric

A surface reading would conclude *"4-agent content moderation is
robust to reordering."* But physically inspecting the judge scores
reveals something more nuanced. Per-variant judge scores per eval row
were uniformly high (0.7–1.0) — the judge gave high marks to **every**
variant on **every** post, regardless of whether the variant decided
REMOVE or KEEP.

This is a real conceptual edge of `verdict_flip_rate`: it measures
**flips of the judge's quality score above/below 0.5**, NOT flips of
the pipeline's actual remove/keep decision.

When the rubric scores moderation *reasoning quality* (which is what
this entry's rubric and the packaged `lending_judge` rubric do), and
the LLM judge is generally satisfied with all reasonably-articulated
moderation decisions, then `verdict_flip_rate` is structurally
incapable of detecting REMOVE↔KEEP disagreement. A chain could be
flipping its actual moderation verdicts while the gallery records
"stable."

This is the strongest concrete justification for the gallery README's
caveat #2 ("`verdict_flip_rate` measures judge-score flips, not
pipeline-output flips") and a v0.5.1 backlog item to either:

1. Detect and surface raw-verdict extraction from agent text (when
   feasible) and report flip rates on both axes, or
2. Add an opinionated rubric variant ("Score 1 if remove, 0 if keep —
   match the pipeline's stated decision exactly") that aligns the
   judge metric with raw verdict, plus a methodology note.

## What is in this directory

- [`fixture.py`](fixture.py) — the discarded fixture source
- [`eval.csv`](eval.csv) — the 10-row UGC moderation eval set
- [`rubric.txt`](rubric.txt) — the moderation-quality rubric

The `runs/` directory was deliberately not committed — there is no
`report.html` to host because there is no headline finding to host.

## If you want to investigate

Reproduce the run and inspect the per-cell judge scores yourself:

```powershell
$env:GEMINI_API_KEY = "<your-key>"
.venv\Scripts\python.exe -m topofuzz run `
    examples/gallery/langgraph-content-moderation/fixture.py:build_graph `
    --eval examples/gallery/langgraph-content-moderation/eval.csv `
    --rubric examples/gallery/langgraph-content-moderation/rubric.txt `
    --n 8 --variants 4 `
    --judge google/gemini-2.5-flash `
    --rng-seed 42 `
    --out runs/content-mod-repro `
    --max-cost-usd 2.00 --yes
```

Then the per-eval per-variant judge scores live in
`runs/content-mod-repro/<run_id>/run.sqlite` table `judge_results`
(columns: `variant_id`, `eval_id`, `score_value`).
