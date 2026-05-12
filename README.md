# pramana-eval

Zero-shot evaluation of the Pramana 6-phase Navya-Nyaya reasoning schema across frontier LLMs. Single Python file, three providers, ten problems.

## Headline result

On n=10 test problems, Gemini 2.5 Pro hit 100% format adherence and 100% semantic correctness, zero-shot, no fine-tuning. The paper's fine-tuned DeepSeek-R1-Distill 8B (Stage 1) hit 40% format and 100% semantic on the same n=10.

| Provider  | Model                               | Format | Semantic | Wilson 95% CI (format) |
| --------- | ----------------------------------- | ------ | -------- | ---------------------- |
| Google    | gemini-2.5-pro (zero-shot)          | 100%   | 100%     | ~[69%, 100%]           |
| Paper     | DeepSeek-R1-Distill 8B (fine-tuned) | 40%    | 100%     | ~[17%, 69%]            |

**The 95% confidence intervals overlap.** A 100% vs 40% gap on n=10 is suggestive, not statistically separated. Read this as a single data point that motivates a larger evaluation, not as a benchmark result. See **Limitations -> Statistical caveats** below.

## What this is

`pramana-eval` is a single-file Python harness that asks a frontier LLM to produce the Pramana paper's six-phase epistemic reasoning structure on ten test problems, with no fine-tuning and no few-shot examples. It scores each response for structural format adherence (does the output contain all six required sections, sub-bullets, and YAML fields?) and semantic correctness (does the final answer contain the required ground-truth keywords?). It works against Anthropic, Google Gemini, and OpenAI APIs.

## Why it matters

The Pramana paper trains an 8B model with supervised fine-tuning to produce a specific Sanskrit-named reasoning schema. The resulting model hits 40% format adherence and 100% semantic correctness on a 10-problem test set. The implicit claim is that fine-tuning is the way to get this schema out of an LLM.

This repo tests an alternative hypothesis: a frontier zero-shot model, given the same schema as a system prompt, will match or exceed those numbers. If true, schema portability is suggested - the structure could be a system-prompt artifact rather than a weights artifact - and further evaluation at larger n is recommended before drawing strong conclusions. The Gemini 2.5 Pro result (100% format, 100% semantic) is consistent with the portable-schema reading at this sample size, but on n=10 the confidence intervals overlap with the paper's, so this is a motivating signal, not a settled question. It does not refute the paper; the paper's contribution is the schema itself, which is what does the work in both runs.

## How it works

**The schema.** Six sections, in order: Samshaya (doubt classification), Pramana (evidence sources broken into Pratyaksha/Anumana/Upamana/Shabda), Pancha Avayava (five-member syllogism: Pratijna, Hetu, Udaharana, Upanaya, Nigamana), Tarka (counterfactual reductio), Hetvabhasa (five fallacy checks), Nirnaya (final answer with confidence). Plus YAML frontmatter with id, problem type, difficulty, variable count, and a quoted ground_truth string. The system prompt encodes this verbatim.

**The test set.** 10 problems across three types: constraint satisfaction (logic puzzles), boolean SAT (propositional satisfiability), and multi-step deduction (transitive inference). 3 are taken verbatim from the paper; 7 are constructed by us in the same paper-described problem types.

**The format scorer.** A regex pass that verifies YAML frontmatter is present and contains `ground_truth`, all six section headers exist, the Pramana section names all four evidence sub-types, the Pancha Avayava section names all five syllogism parts, the Udaharana line uses the literal "Wherever ... there is" template, the Hetvabhasa section names all five fallacy types, and the Nirnaya section has Answer/Confidence/Epistemic Status fields. All checks must pass for the response to count as format-passing.

**The semantic scorer.** Extracts the Nirnaya section text and checks that every ground-truth keyword group for that problem appears (case-insensitive, with word boundaries and digit-to-word normalization). Each problem's keywords are an AND-of-OR structure (`list[list[str]]`): the outer list is required tokens (all must appear), each inner list is a synonym set (any one synonym counts). Strict by design: a missing token group fails the problem even if the surrounding answer is correct in spirit.

The scorer is intentionally conservative on the format dimension and intentionally simple on the semantic dimension. Honest false-pass risks remain: substring matches without word boundaries were possible in earlier versions (now addressed), the scorer is blind to negation in the answer text (a "not X" sentence containing keyword X still counts), and an empty keyword list previously short-circuited to a pass (now raises). False-fail risks: synonyms not enumerated in the keyword groups, unicode hyphen variants, and unusual phrasing. See `SCORING_VALIDATION.md` for the full list.

**The decision rule.** >=80% format adherence suggests the schema is portable enough to use as a system prompt; 40-80% suggests the prompt needs iteration; <40% suggests fine-tuning is likely required. These thresholds are heuristics, not precise cut-offs - on n=10 with overlapping CIs, treat any decision as provisional pending a larger run.

## Reproducing

Install only the SDK(s) for the provider(s) you intend to test. Each is optional and lazy-imported.

```bash
pip install anthropic       # for Claude
pip install google-genai    # for Gemini
pip install openai          # for GPT
```

Set whichever provider key you want to test:

```bash
# PowerShell
$env:GEMINI_API_KEY = "..."        # or ANTHROPIC_API_KEY / OPENAI_API_KEY

# bash/zsh
export GEMINI_API_KEY="..."
```

Rotate any API key that has been pasted into a chat, terminal history, or shared screen. Treat them as compromised.

Run:

```bash
python pramana_test.py
```

CLI flags: `--provider {anthropic,gemini,openai}`, `--model MODEL`, `--output PATH`, `--limit N`, `--problems PATH`. The script auto-detects a provider from environment variables in the order Anthropic, Gemini, OpenAI; the first match wins unless you override with `--provider`.

Expected output (abbreviated):

```
Provider: gemini
Model: gemini-2.5-pro
Test problems: 10 (3 paper-verified, 7 constructed)
------------------------------------------------------------------------------
[ 1/10] test-001 (paper       ) FORMAT-PASS  SEM-PASS  (16.7s)
...
------------------------------------------------------------------------------
Format adherence:    10/10 = 100%  (Wilson 95% CI: 69-100%)
Semantic correctness: 10/10 = 100%  (Wilson 95% CI: 69-100%)
Subgroup: paper-verified (n=3)  format 3/3, semantic 3/3
Subgroup: constructed (n=7)     format 7/7, semantic 7/7

DECISION: schema portability suggested at this sample size; further evaluation
recommended at larger n before drawing strong conclusions.
Full traces saved to pramana_test_results.json
```

### Expected wall time and cost

Rough order-of-magnitude estimates for a single 10-problem run as of May 2026. Real numbers depend on prompt length, response length, and current pricing. Treat as approximate.

| Provider  | Model            | Wall time | Estimated cost |
| --------- | ---------------- | --------- | -------------- |
| Google    | gemini-2.5-pro   | ~3 min    | ~$0.10         |
| Anthropic | claude-sonnet-4-6 | ~3 min   | ~$0.30         |
| OpenAI    | gpt-5            | ~3 min    | ~$0.20         |

Verify current per-token pricing on the provider's pricing page before running.

### Reproducibility

The script pins `temperature=0` across all providers and writes a `meta` block into the JSON output containing: provider, model, SDK version, Python version, OS, timestamp, and the test-set fingerprint. To reproduce a published result, match the model version (e.g. `gemini-2.5-pro` at the dated revision in `meta`), the SDK version, and the test-set fingerprint. Note that frontier model providers may silently update model weights even at a pinned model name; the SDK version and timestamp in `meta` are the best record we can capture.

## Test set

Three problems are verbatim from the paper (test-001 through test-003: pet-assignment, rain-cancellation, four-seat ordering). Seven are constructed by us (test-004 through test-010): box/metal assignment, two boolean SAT instances (one unsat, one sat), a syllogism on whales, an arithmetic-divisibility chain, a race-ordering puzzle, and a counting problem on disjoint sets.

We disclose this split because reusing only paper problems would conflate "the model memorized the paper's examples" with "the model can do this kind of problem." Mixing in fresh-but-same-type problems controls for that. We do not claim the constructed seven are perfectly difficulty-matched to the paper's seven unseen problems - the paper does not list its full test set, so exact matching is not possible.

## Extending

Add your own problems by appending to the `TEST_PROBLEMS` list in `pramana_test.py`:

```python
# Without synonyms: each inner list has one token. Outer list = AND.
Problem(
    id="test-011",
    question="...",
    ground_truth_keywords=[["alice"], ["fish"], ["bob"], ["cat"], ["carol"], ["dog"]],
    source="constructed",
)

# With synonyms: any one inner-list token counts. Outer list = AND, inner list = OR.
Problem(
    id="test-012",
    question="How many students passed history?",
    ground_truth_keywords=[["10", "ten"]],
    source="constructed",
)
```

`ground_truth_keywords` has shape `list[list[str]]` (AND-of-OR). The outer list is the set of required answer components - every group must match. Each inner list is a synonym set - any one synonym in the group counts as a match. Pick small, unambiguous tokens that any correct answer must contain. For multi-part answers (e.g. an ordering), include each part as its own outer-list group; for numeric answers, include digit and word forms in the same inner-list group.

To add a new provider, edit `_detect_provider`, `_build_client`, and `call_model` in the single file. The shape is intentionally small: one branch per provider, no plugin system.

To use a custom problem file, pass `--problems mypath.json`. The file should be a JSON list of objects matching the `Problem` dataclass fields.

## Limitations

- **n = 10.** Same as the paper. Indicative, not definitive. A larger test set could move the numbers in either direction.
- **Keyword-based semantic scorer.** Strict but brittle: a model that answers correctly in unusual phrasing can false-fail. We chose this over LLM-as-judge to avoid the judge silently grading on style. The trade-off is conservative scoring.
- **Single-author preprint base.** The Pramana paper is a single-author preprint (arXiv:2604.04937) and has not been peer-reviewed at the time of writing. The schema is what it is; we are not endorsing the broader theoretical claims.
- **One frontier model on one schema.** This is one data point against one preprint. It is not a comprehensive benchmark of frontier reasoning ability and should not be cited as one.

### Statistical caveats

The headline numbers come from a 10-problem run. The Wilson 95% confidence interval for 100% on n=10 is approximately [69%, 100%]; for 40% on n=10 it is approximately [17%, 69%]. **These intervals overlap.** A 100% vs 40% gap on this sample size is suggestive, not statistically separated.

A single-model anecdote on 10 problems is not a benchmark. To turn this into a real comparison, future work should:

- Extend to n >= 50 (preferably n >= 100) with held-out problems written before any model run.
- Pre-register keyword groups before scoring.
- Run each model multiple times to estimate temperature-0 variance (most providers still have small non-determinism even at temp=0).
- Compare against the paper's exact problem set rather than a paper-plus-constructed mix, where possible.

Until something like the above is done, treat the result here as a motivating signal for further evaluation, not a verdict.

## Troubleshooting

- **Rate limits / 429 errors.** All three providers rate-limit free and low-tier accounts. The script retries with exponential backoff on 429s. If you hit a hard quota, wait or switch tiers. A 10-problem run should not exhaust normal paid-tier quotas, but burst limits can still kick in.
- **Billing / quota errors.** "Quota exceeded", "billing not enabled", or "credits exhausted" usually means the API project does not have a payment method or has hit its monthly cap. Fix in the provider's billing console.
- **Region restrictions on the Gemini API.** `google-genai` is unavailable in some regions. If you see `403 PERMISSION_DENIED` or "API not available in your region", use a supported region or run from a hosted environment.
- **Model not available.** If `claude-sonnet-4-6`, `gemini-2.5-pro`, or `gpt-5` returns `404` or "model not found", the model name has rotated or your account does not have access. Pass a current model name with `--model`. Check each provider's model list page; frontier model names change every few months.
- **Encoding issues on Windows.** Output contains Sanskrit terms in transliterated ASCII so plain `cp1252` consoles should be fine, but if your terminal mangles output or `--output` writes garbled text, set `PYTHONIOENCODING=utf-8` and run from Windows Terminal or PowerShell 7+.
- **Empty or truncated responses.** Hitting the model's max-output-tokens limit is the usual cause. The script sets a generous default; if you change `max_tokens`, keep parity across providers so comparisons stay fair.

## Citation

The original schema:

```bibtex
@article{sathish2026pramana,
  title   = {Pramana: Fine-Tuning Large Language Models for Epistemic
             Reasoning through Navya-Nyaya},
  author  = {Sathish, Sharath},
  journal = {arXiv preprint arXiv:2604.04937},
  year    = {2026}
}
```

This repo:

```bibtex
@misc{nayudu2026pramanaeval,
  title  = {pramana-eval: Zero-shot evaluation of the Pramana six-phase
            reasoning schema on frontier LLMs},
  author = {Nayudu, Kiran},
  year   = {2026},
  note   = {ModelBrew AI. github.com/modelbrew/pramana-eval}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE) for the license text and [NOTICE](NOTICE) for required attribution and the explicit disclaimer that ModelBrew AI claims nothing over the Navya-Nyaya tradition or the schema as described in the Sathish paper.

## Acknowledgments

Sharath Sathish designed the six-phase Navya-Nyaya schema and ran the original fine-tuning evaluation in the Pramana paper. This repo only tests whether his schema generalizes to zero-shot frontier inference; the conceptual contribution is his.

## About

ModelBrew AI ([modelbrew.ai](https://modelbrew.ai)) ran this evaluation to answer one question: is the Pramana schema portable enough to drop into a system prompt, or does it need fine-tuning? The Gemini 2.5 Pro result on n=10 suggests portability is plausible, but the confidence intervals overlap with the paper's, so further evaluation at larger n is recommended before drawing strong conclusions. We are publishing the harness so the question can be re-asked against other models, larger test sets, and stricter scorers. This is not a competing product to the Pramana paper - it is a reproducibility probe of one specific claim.
