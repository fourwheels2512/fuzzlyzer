# Contributing to pramana-eval

Thanks for considering a contribution. This project is intentionally small - one Python file, one test set, one scorer. Single-file simplicity is a feature, not an oversight, so the bar for adding code is high.

## Filing issues

Use GitHub issues for:

- **Bugs**: include the provider, model, Python version, command run, full console output, and (if non-sensitive) the relevant section of `pramana_test_results.json`.
- **Enhancements**: describe the use case before the change. "I want to add X" lands better than "here is X."
- **New test problems**: open an issue with the problem text, the ground-truth answer, the keyword group you propose, and a one-line note on which paper-described category it falls into (constraint satisfaction, boolean SAT, multi-step deduction). PRs welcome after discussion.

## Pull request philosophy

- Small, focused, one concern per PR.
- Keep `pramana_test.py` a single file. Splitting it across modules requires a strong justification.
- No new runtime dependencies without discussion. The current set (`anthropic`, `google-genai`, `openai`) is the ceiling, and each is optional at runtime.
- No new dev dependencies without discussion. Standard library is preferred.

## Code style

- `ruff` and `black` for formatting and lint.
- Type hints on all public functions and dataclasses.
- Docstrings on anything non-obvious. The existing module docstring sets the tone.

## Adding a new provider

The provider seam is intentionally small. Touch three places in `pramana_test.py`:

1. `_detect_provider` - add an environment variable check that returns the new provider name.
2. `_build_client` - construct the SDK client.
3. `call_model` - add a branch that issues the request with `SYSTEM_PROMPT` as the system instruction and the user message as the prompt, and returns the response text as a string.

Add the SDK to the install line in the README and to the `--provider` choices in `argparse`. No plugin registry, no abstract base class.

## Adding test problems

Append to `TEST_PROBLEMS`. `ground_truth_keywords` has shape `list[list[str]]`: outer list is required groups (AND), each inner list is synonyms within a group (OR).

```python
# Without synonyms - each inner list has one token. Outer list = AND.
Problem(
    id="test-NNN",
    question="...",
    ground_truth_keywords=[["alice"], ["fish"], ["bob"], ["cat"]],
    source="constructed",
)

# With synonyms - any one token in an inner list counts as a match.
Problem(
    id="test-MMM",
    question="How many students passed history?",
    ground_truth_keywords=[["10", "ten"]],
    source="constructed",
)
```

Pick keywords that any correct answer must contain. For multi-part answers, list each part as its own outer-list group. For numeric answers, include digit and word forms in the same inner-list group. Mark `source="paper"` only if the problem is verbatim from the Pramana paper.

**Pre-register your keywords.** PRs that add new test problems should not modify keyword groups after seeing model output. Decide your AND-of-OR groups from the problem statement and ground truth alone, then run the model. Adjusting keywords post-hoc to make a model pass (or fail) defeats the point.

## DCO / sign-off

Sign-off via `git commit -s` is recommended but not enforced. If you want your contribution attributed, use a real name and a reachable email.
