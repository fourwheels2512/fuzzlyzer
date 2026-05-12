# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Audit-pass fixes from a two-reviewer round. No version bump.

### Reproducibility

- Pinned `temperature=0` across all three providers.
- Added `max_tokens` parity across providers so response budgets compare fairly.
- Added per-request timeouts and retry-with-backoff on transient errors (429s, 5xx).

### Scorer correctness

- Semantic scorer: keyword matching now uses word boundaries instead of raw substring search.
- Semantic scorer: digit/word normalization (e.g. `10` matches `ten` when both are listed in the same synonym group).
- Format scorer: minimum non-empty-content check per section so a header with no body no longer counts as present.
- Format scorer: section order is now verified (six headers must appear in the prescribed order).
- An empty `ground_truth_keywords` group now raises at problem-load time instead of silently passing.

### Output

- Per-problem subgroup breakdown in the JSON: paper-verified vs constructed.
- Wilson 95% confidence interval on format and semantic rates.
- New `meta` block in the JSON output: provider, model, SDK version, Python version, OS, timestamp, test-set fingerprint.

### Hygiene

- API key sanitization in error strings (keys redacted before logging or writing to JSON).
- `--output` parent directory is validated up front; clear error if it does not exist.
- Decision-rule output uses neutral phrasing rather than product-name-leaking copy.

### Documentation

- Corrected `ground_truth_keywords` shape examples in README and CONTRIBUTING (was flat list, is `list[list[str]]`).
- Added Troubleshooting section to README (rate limits, billing, region restrictions, model-not-available, Windows encoding).
- Added Statistical caveats subsection to README (Wilson CIs, n>=50 recommendation, pre-registration).
- Added Reproducibility subsection to README (temperature pinning, model versions, SDK versions, JSON `meta` block).
- Added expected wall time and approximate per-run cost per provider.
- Toned down the headline result: 100% vs 40% on n=10 has overlapping Wilson CIs and is now described as suggestive, not statistically separated.
- Replaced TraceGate productization language with neutral "schema portability suggested; further evaluation recommended" framing.
- Install instructions now show one provider per user with separate one-liners.
- Added `NOTICE` file with required Apache 2.0 attribution and explicit disclaimer about the Navya-Nyaya tradition and the schema's source.
- Added `SAMPLE_OUTPUT.md`, `SCORING_VALIDATION.md`, `requirements.txt`.

## [0.1.0] - 2026-05-08

### Added

- Initial release.
- Multi-provider zero-shot evaluation of the Pramana six-phase Navya-Nyaya
  reasoning schema (Samshaya, Pramana, Pancha Avayava, Tarka, Hetvabhasa,
  Nirnaya) via a single Python file.
- Provider support: Anthropic, Google Gemini, OpenAI. Auto-detection from
  environment variables with `--provider` and `--model` overrides.
- 10-problem test set: 3 verbatim from the Pramana paper, 7 constructed in
  the same paper-described problem types (constraint satisfaction,
  boolean SAT, multi-step deduction). Each problem is tagged with `source`
  for paper-verified vs constructed analysis.
- Format scorer replicating the paper's structural validator: YAML
  frontmatter, six required section headers, Pramana sub-types, Pancha
  Avayava parts, Udaharana template phrase, Hetvabhasa fallacy types,
  Nirnaya fields.
- Keyword-group semantic scorer over the Nirnaya section with YAML
  frontmatter fallback.
- Decision rule for productization vs prompt iteration vs fine-tune.
- JSON output with full per-problem traces and aggregate report.
- Apache 2.0 license.
