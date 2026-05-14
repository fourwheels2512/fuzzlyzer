# langgraph-lending-4role

> ⚠️ **NUMBERS STALE — UNDER RE-EVALUATION (2026-05-14).** Regenerated
> under the fixed runner (run `2026-05-14-01KRKK2CHTNFNA48YG07D3WRVM`),
> this entry's `verdict_flip_rate` is `0` (degenerate): honestly
> measured, this topology is robust under permutation. The old
> "~0.3 fragility" was a measurement artifact — the runner judged the
> last-*walked* node, not the declared `decision` node. **All numbers
> below predate the fix; do not quote them.** See
> `.planning/verdict-flip-rootcause/`.

> 4-agent linear chain (intake → risk → policy → decision) deciding
> whether a mortgage application should be approved. The same
> applicant data, same Gemini Flash model, same prompt structure
> across all variants — only the agent **ordering** changes.

This is the home-reference gallery entry: it reuses topofuzz's own
canonical demo fixture (`topofuzz.fixtures.lending_4role`) so reviewers
can verify the gallery infrastructure on the most heavily-tested
fixture in the repo. The gallery wrapper at
[`fixture.py`](fixture.py) overrides `LLM_CONFIGS` to all-Flash to keep
per-entry cost under $1.50; the canonical fixture uses a mixed
flash/pro setup tuned for `role_swap` pair coverage.

## Headline metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| `verdict_flip_rate.point` | **0.300** | 30% of (variant, eval) pairs disagree on APPROVE/DENY with the original variant (binary verdict = judge score ≥ 0.5) |
| `verdict_flip_rate.ci_lower` | **0.100** | 95% CI lower bound — passes the curation gate (>0.05) |
| `verdict_flip_rate.ci_upper` | 0.500 | 95% CI upper bound — wide because n=8 evals × 4 variants |
| `kappa_original_vs_permuted` | **0.077** | Landis-Koch "slight" agreement (0.01–0.20) — well below the "fair" floor (0.21) and the "moderate" floor (0.41). `verdict_flip_rate=0.300` and `kappa=0.077` measure related but distinct things: flip-rate is the per-pair rate of binary verdict disagreement (judge score ≥ 0.5 vs < 0.5); kappa is chance-corrected pairwise variant agreement on the same binary verdict. They can diverge when many of the disagreements concentrate on a small subset of evals, as is the case here |
| `judge_failure_rate` | **0.063** | 2 of 32 judge calls failed to emit a parseable score |
| `n_successful_cells` | 32 / 32 | Full matrix completion |
| `total_cost_usd` | $1.20 | Slightly over the BLUEPRINT 19a `<$1` target |
| `status` | `degraded` | Reflects a structural metadata bug (eval-axis ordering mismatch in attestation path), NOT cell failures |

## A real flip — `lending_008`

The applicant: 30 years old, $55K annual income, 702 credit score,
34% DTI, requesting a $195K 30-year mortgage with 15% down. Income
history: 2 years of 1099 income preceded by 4 years of W-2.

Two variants of the SAME 4-agent topology produced opposite verdicts:

**APPROVE variant** (judge score: 0.9)
> *Decision: Approve with Conditions ... Strong Creditworthiness:
> The applicant's credit score of 702 is considered "Good" and meets
> minimum requirements ... Excellent Repayment Capacity: A
> Debt-to-Income (DTI) ratio of 34% is well within acceptable limits.*

**DENY variant** (judge score: 0.2)
> *Decision: Decline the mortgage application as currently structured
> ... Policy Analysis: High Risk - Unapprovable as Structured ...
> the current mortgage request for $195,000 is unapprovable under
> standard lending guidelines.*

The two variants used the same applicant data, the same Gemini Flash
model on every node, the same temperature (0.0), the same prompt
template — only the agent role-swap was different. See
[`runs/<run_id>/run.sqlite`](runs/2026-05-14-01KRJ2WADFPMF6W1MJSF777EW0/run.sqlite)
for the full per-cell trace.

## Run details

```
Run id:        2026-05-14-01KRJ2WADFPMF6W1MJSF777EW0
Fixture:       examples/gallery/langgraph-lending-4role/fixture.py:build_graph
Eval set:      examples/gallery/langgraph-lending-4role/eval.csv (8 of 50 rows used)
Rubric:        examples/gallery/langgraph-lending-4role/rubric.txt
Judge model:   google/gemini-2.5-flash (single judge — see gallery README caveats)
Variants:      4 (capped per --variants 4)
RNG seed:      42
Cost cap:      $2.00
Actual cost:   $1.20
Wall-clock:    139s
```

## Reproduce

```powershell
$env:GEMINI_API_KEY = "<your-key>"
.venv\Scripts\python.exe -m topofuzz run `
    examples/gallery/langgraph-lending-4role/fixture.py:build_graph `
    --eval examples/gallery/langgraph-lending-4role/eval.csv `
    --rubric examples/gallery/langgraph-lending-4role/rubric.txt `
    --n 8 --variants 4 `
    --judge google/gemini-2.5-flash `
    --rng-seed 42 `
    --out runs/lending-repro `
    --max-cost-usd 2.00 --yes
.venv\Scripts\python.exe -m topofuzz report runs/lending-repro/<run_id>
```

(Replace `<run_id>` with the directory name printed by `topofuzz run`
— for the published run that's `2026-05-14-01KRJ2WADFPMF6W1MJSF777EW0`.)

Your published flip rate may not match exactly because Gemini Flash
responses are not deterministic across calls — but the **direction**
(genuine APPROVE/DENY disagreement under reordering) should reproduce
on borderline applicants like `lending_008`.

## Source attribution

- Fixture topology: `topofuzz.fixtures.lending_4role` shipped with
  topofuzz v0.5
- Rubric: `topofuzz.prompts.lending_judge` shipped with topofuzz v0.5
  (BLUEPRINT_v5plus.md §13 underwriting heuristic)
- Eval rows: synthetic, generated for v0.5 demo purposes
