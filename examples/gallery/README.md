# topofuzz gallery

> ⚠️ **UNDER RE-EVALUATION (2026-05-14).** A measurement bug was found
> and fixed: the runner judged the last-*walked* node instead of the
> declared verdict node, so `order_swap` permutations that landed a
> non-verdict agent last fabricated verdict-flips. With the bug fixed,
> both gallery entries' headline `verdict_flip_rate` collapses to `0`
> (degenerate) — the "fragility" they appeared to show was the artifact.
> **The numbers below predate the fix and must not be quoted.** The
> gallery's headline framing is being re-decided — see
> `.planning/verdict-flip-rootcause/` and the step-20a session handoff.

Pre-built topofuzz runs against well-known LangGraph multi-agent
patterns. Each gallery entry is a complete, reproducible artifact: the
fixture source, eval rows, judge rubric, and the full run directory
(including the rendered Variant D `report.html`).

The gallery exists to answer one question that arrives in the first 30
seconds of every Show HN comment thread:

> *"Does this work on real chains, or just your contrived demo?"*

## What "passes the curation gate"

Per `BLUEPRINT_v5plus.md` line 439, the gallery is **examples where
ordering matters**, NOT a benchmark. A candidate fixture only ships in
the gallery if `headline.verdict_flip_rate.ci_lower > 0.05` after a
small-N run (`--n 8 --variants 4`). Stable topologies are excluded — a
"this is stable" entry in a "examples where ordering matters" gallery
is a credibility loss.

This release ships 2 of 3 candidate fixtures. The third was discarded
on the gate; its source code remains in
[`langgraph-content-moderation/`](langgraph-content-moderation/) as a
transparency artifact (read its README to understand why).

## Cross-vendor scope (v0.5 vs v0.6)

The current release ships **3 LangGraph entries** because topofuzz
today only has a LangGraph IR adapter
(`packages/topofuzz/src/topofuzz/adapter.py:from_langgraph`). The
**canonical final form is a 1 LangGraph + 1 CrewAI + 1 Autogen
gallery**, gated on building `from_crewai` + `from_autogen` adapters in
v0.6. See `packages/topofuzz/BACKLOG.md` § "v0.6 — cross-vendor gallery
(final form)" for the commitment.

## Entries

| Slug | Topology | Domain | Verdict-flip rate | Status |
|------|----------|--------|-------------------|--------|
| [`langgraph-supervisor-research/`](langgraph-supervisor-research/) | Supervisor + 3 workers (`add_conditional_edges` + path_map) | Article publication-readiness | **0.250** [CI 0.105–0.368] | Shipped |
| [`langgraph-lending-4role/`](langgraph-lending-4role/) | 4-agent linear chain | Loan underwriting | **0.300** [CI 0.100–0.500] | Shipped |
| [`langgraph-content-moderation/`](langgraph-content-moderation/) | 4-agent linear chain | UGC moderation | 0.00 (degenerate) | Discarded — see entry README |

### Published rendered reports

Once the gallery is hosted on GitHub Pages (see `docs/gallery-build.md`
for the republish ritual and `gh-pages` branch contents), the rendered
Variant D `report.html` for each shipped entry is reachable at:

* `https://fourwheels2512.github.io/topofuzz/gallery/langgraph-supervisor-research/report.html`
* `https://fourwheels2512.github.io/topofuzz/gallery/langgraph-lending-4role/report.html`

These URLs go live only after the repo is made public AND GitHub Pages
is configured to serve the `gh-pages` branch — see `docs/gallery-build.md`
§ "Activation". Until then, render the reports locally from the
committed `runs/<run_id>/report.html` files.

## Reproducing a gallery run

The committed `runs/<run_id>/` directory is **structurally**
reproducible from the fixture + eval + rubric + `--rng-seed`. The LLM
**responses** will be yours, not ours — you bring your own API key.

### Bring your own keys (BYO-keys)

topofuzz dispatches LLM calls via [LiteLLM](https://docs.litellm.ai/),
so any provider LiteLLM supports works. The published gallery runs
used a single Gemini judge (`google/gemini-2.5-flash`). Get a key from
https://aistudio.google.com/app/apikey and set it in your shell:

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY = "<your-key>"
```

```bash
# macOS / Linux
export GEMINI_API_KEY="<your-key>"
```

Alternative judges if you have other vendor keys (PowerShell shown,
bash equivalent uses `export`):

```powershell
$env:OPENAI_API_KEY = "<your-key>"
$env:ANTHROPIC_API_KEY = "<your-key>"
```

`topofuzz run` with `--jury-of-3` requires Anthropic + OpenAI + Gemini
keys all set.

### Reproduce a single entry

```powershell
# From the repo root, replace <slug> with one of the entry directories
$slug = "langgraph-lending-4role"
.venv\Scripts\python.exe -m topofuzz run `
    examples/gallery/$slug/fixture.py:build_graph `
    --eval examples/gallery/$slug/eval.csv `
    --rubric examples/gallery/$slug/rubric.txt `
    --n 8 --variants 4 `
    --judge google/gemini-2.5-flash `
    --rng-seed 42 `
    --out runs/$slug-repro `
    --max-cost-usd 2.00 --yes
```

Then render the HTML report:

```powershell
.venv\Scripts\python.exe -m topofuzz report runs/$slug-repro/<run_id>
```

### Verify the DSSE attestation

Each shipped run dir contains `attestation.json` (DSSE-signed envelope
of `report.json`). The signature is from a development key —
third-party verifiers should treat it as *integrity-of-publication*
evidence (the file hasn't been tampered with since publication), not
*identity-of-publisher* evidence (the dev key is not registered with
any public root of trust).

The matching dev public key is committed alongside the gallery at
`examples/gallery/dev-attestor.pub`, so a fresh clone can verify
without first running `topofuzz keygen`. Point `--pubkey` at it and
pass a committed run dir:

```powershell
.venv\Scripts\python.exe -m topofuzz verify `
    examples/gallery/langgraph-lending-4role/runs/2026-05-14-01KRKK2CHTNFNA48YG07D3WRVM `
    --pubkey examples/gallery/dev-attestor.pub
```

A successful run prints a `PASS [DEV-KEY]` banner and exits `0`. The
`[DEV-KEY]` tag is a reminder that this is the unregistered
development key — see the integrity-vs-identity caveat above.

## Caveats and known limitations

This gallery surfaces real findings but with real measurement noise.
Read these before quoting any number:

1. **The judge is itself a noisy LLM.** All gallery entries used a
   single Gemini Flash judge. `judge_failure_rate` is reported per
   entry; treat any flip-rate point estimate with skepticism if the
   judge failure rate is above 10%. The published 95% CIs are
   cluster-bootstrap on the eval-id axis (n=5000 resamples) — they
   capture sampling variation across eval rows but NOT judge
   measurement error. With small N (8 evals × 4 variants) the bootstrap
   has very few unique resamples and the resulting CIs are wide.
2. **`verdict_flip_rate` measures judge-score flips, not pipeline-output
   flips.** A pipeline that flips REMOVE↔KEEP but the judge rates both
   as "good reasoning" will register as stable — see
   [`langgraph-content-moderation/`](langgraph-content-moderation/) for
   why this matters and why that entry was discarded.
3. **Same-vendor scoring inflates apparent agreement** (SECURITY.md §8).
   When agents and judge share a vendor (here: Google), small-N
   correlation may overstate stability. v0.6 cross-vendor entries will
   address this directly.
4. **Cost estimator under-estimates real Gemini cost** by ~3–5x.
   Configure `--max-cost-usd` with headroom; the cost cap is enforced
   on actual spend, not the upfront estimate.
5. **Curation gate runs at small N (`--n 8 --variants 4`).** A larger
   sample would tighten CIs but ship `<$1` per entry per BLUEPRINT
   `19a`. Real cold-email runs (BLUEPRINT step 22) use `--n 50
   --variants 8 --jury-of-3`.
6. **`manifest.json` records the publishing host's local file paths**
   (e.g. `examples\gallery\...\eval.csv`, `.cache\topofuzz`). Not a
   secret, but it does expose that the v0.5 reference runs were
   published from a Windows machine and the cache layout. Re-runs on
   your own machine will record your paths instead.

## License

The gallery contents are released under the same Business Source
License 1.1 as topofuzz itself — see the repo-root `LICENSE` and the
canonical 4-line BSL header at the top of every gallery `fixture.py`.
The CI gate at `.github/workflows/ci.yml` enforces header presence
via `scripts/add_copyright_headers.py --check`. Eval rows are
synthetic; any resemblance to real applicants, posts, or articles is
illustrative only.
