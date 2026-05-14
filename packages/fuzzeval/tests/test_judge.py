from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from fuzzeval.judge import (
    BinaryScore,
    ContinuousScore,
    JudgeResult,
    JudgeUnparseableError,
    SingleJudge,
    binary_from_continuous,
    _parse_score,
)


# ---------------------------------------------------------------------------
# §17a type contract
# ---------------------------------------------------------------------------


def test_continuous_score_holds_value():
    s = ContinuousScore(value=0.7)
    assert s.value == 0.7


def test_continuous_score_rationale_defaults_empty():
    """rationale is optional — pre-rationale callers still construct cleanly."""
    assert ContinuousScore(value=0.7).rationale == ""


def test_continuous_score_holds_rationale():
    s = ContinuousScore(value=0.7, rationale="meets rubric")
    assert s.value == 0.7
    assert s.rationale == "meets rubric"


def test_continuous_score_old_pickle_missing_rationale_falls_back():
    """A ContinuousScore pickled before the rationale field existed has no
    'rationale' key in its stored __dict__. The dataclass default is a real
    class attribute, so such an instance resolves ``.rationale`` to "" via the
    class-attribute fallback instead of raising AttributeError — this is what
    keeps pre-existing JudgeCache (pickle-backed) entries round-tripping after
    the field was added."""
    import pickle

    # A current instance must survive a real pickle round-trip cleanly.
    fresh = pickle.loads(pickle.dumps(ContinuousScore(value=0.4, rationale="x")))
    assert fresh == ContinuousScore(value=0.4, rationale="x")

    # Simulate the on-disk shape of a pre-field pickle: a __dict__ carrying
    # only 'value'. object.__setattr__ is required because the dataclass is
    # frozen; this mirrors what pickle.loads would reconstruct from old bytes.
    old = ContinuousScore(value=0.4)
    object.__setattr__(old, "__dict__", {"value": 0.4})
    assert "rationale" not in old.__dict__
    assert old.rationale == ""  # class-attribute fallback, no AttributeError
    assert old == ContinuousScore(value=0.4)  # __eq__ still resolves cleanly


def test_continuous_score_is_frozen():
    s = ContinuousScore(value=0.7)
    with pytest.raises(FrozenInstanceError):
        s.value = 0.8  # type: ignore[misc]


def test_binary_score_holds_rationale():
    b = BinaryScore(passed=True, rationale="meets rubric")
    assert b.passed
    assert b.rationale == "meets rubric"


def test_binary_score_is_frozen():
    b = BinaryScore(passed=False, rationale="off topic")
    with pytest.raises(FrozenInstanceError):
        b.passed = True  # type: ignore[misc]


def test_judge_result_union_accepts_both():
    results: list[JudgeResult] = [
        ContinuousScore(value=0.5),
        BinaryScore(passed=False, rationale="off-topic"),
    ]
    assert len(results) == 2


def test_judge_result_supports_runtime_isinstance():
    """PEP 604 union — isinstance(x, JudgeResult) must work at runtime so
    consumers can narrow Judge.score() returns without typing.TypeGuard tricks.
    """
    assert isinstance(ContinuousScore(value=0.3), JudgeResult)
    assert isinstance(BinaryScore(passed=True, rationale="ok"), JudgeResult)
    assert not isinstance("a string", JudgeResult)
    assert not isinstance(0.5, JudgeResult)


def test_judge_protocol_structural_conformance():
    from fuzzeval.judge import Judge

    class FakeJudge:
        def score(self, payload: dict) -> JudgeResult:
            return ContinuousScore(value=0.42)

    j: Judge = FakeJudge()
    result = j.score({"any": "payload"})
    assert isinstance(result, ContinuousScore)
    assert result.value == 0.42


# ---------------------------------------------------------------------------
# _parse_score — the regex contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_score, expected_rationale",
    [
        ("Score: 7\nRationale: looks good", 7, "looks good"),
        ("Score: 10\nRationale: perfect", 10, "perfect"),
        ("Score: 0\nRationale: clearly wrong", 0, "clearly wrong"),
        ("  Score:   5  \n  Rationale: trimmed", 5, "trimmed"),
        # No rationale → empty string, score still parses.
        ("Score: 3", 3, ""),
        # Score with /10 suffix — we parse the int, accept the slash.
        ("Score: 7/10\nRationale: ok", 7, "ok"),
        # Second-pass M-A1: natural English sentence period after score.
        ("Score: 7.\nRationale: ok", 7, "ok"),
        # Tail commas / parens / trailing whitespace.
        ("Score: 6,\nRationale: comma", 6, "comma"),
        ("Score: 9)\nRationale: paren", 9, "paren"),
    ],
)
def test_parse_score_accepts_contract_format(text, expected_score, expected_rationale):
    raw, rationale = _parse_score(text)
    assert raw == expected_score
    assert rationale == expected_rationale


@pytest.mark.parametrize(
    "text",
    [
        "I think this is a 7 but no Score line",
        "rationale: missing score",
        "",
        "Score: 11",  # out of range — regex alternation rejects
        "Score: 100",  # three-digit run — lookahead rejects
        "Score: -1",  # negative — regex requires \d so the - won't match
        # Audit M1: half-step floats must NOT silently truncate.
        "Score: 8.5\nRationale: borderline",
        "Score: 7.5\nRationale: half",
        # Second-pass MAJOR-2: leading-zero ints must NOT parse as the
        # stripped int (Score: 07 → engine matches 0, then 7 dangles
        # — `(?!\d)` rejects).
        "Score: 07\nRationale: padded",
    ],
)
def test_parse_score_raises_on_bad_format(text):
    with pytest.raises(JudgeUnparseableError):
        _parse_score(text)


def test_parse_score_rejects_non_string():
    with pytest.raises(JudgeUnparseableError):
        _parse_score(None)  # type: ignore[arg-type]


# Audit M2: common markdown decoration must parse, not burn retries.
@pytest.mark.parametrize(
    "text, expected_score",
    [
        ("**Score: 7**\nRationale: bold", 7),
        ("**Score:** 7\nRationale: split bold", 7),
        ("## Score: 8\nRationale: heading", 8),
        ("- Score: 6\nRationale: bullet", 6),
        ("> Score: 9\nRationale: blockquote", 9),
    ],
)
def test_parse_score_handles_markdown_decoration(text, expected_score):
    raw, _ = _parse_score(text)
    assert raw == expected_score


# Second-pass MINOR-1: trailing bold `**` on rationale gets stripped.
def test_parse_score_strips_trailing_rationale_bold():
    raw, rationale = _parse_score("Score: 7\n**Rationale: solid reasoning**")
    assert raw == 7
    assert rationale == "solid reasoning"


# Gallery-regression: real Gemini Flash wraps the rationale across
# physical lines despite the rubric asking for one sentence. The old
# single-line `$`-anchored capture truncated those to the first
# fragment ("The", "The decision"). The capture now runs from the label
# to the next blank line, with internal whitespace collapsed.
def test_parse_score_captures_multiline_rationale():
    text = (
        "Score: 6\n"
        "Rationale: The pipeline correctly\n"
        "approved the applicant given the strong\n"
        "credit score and low DTI."
    )
    raw, rationale = _parse_score(text)
    assert raw == 6
    assert rationale == (
        "The pipeline correctly approved the applicant given the "
        "strong credit score and low DTI."
    )


def test_parse_score_rationale_body_on_next_line():
    """Judge puts the label on its own line, body on the following lines."""
    raw, rationale = _parse_score("Score: 4\nRationale:\nthin reasoning here")
    assert raw == 4
    assert rationale == "thin reasoning here"


def test_parse_score_stops_rationale_at_blank_line():
    """A trailing sign-off / postscript after a blank line must NOT be
    swallowed into the rationale (RT pass-3 MAJOR-1)."""
    text = (
        "Score: 6\n"
        "Rationale: The decision is well reasoned.\n"
        "\n"
        "---\n"
        "Note: evaluation generated by judge v2."
    )
    raw, rationale = _parse_score(text)
    assert raw == 6
    assert rationale == "The decision is well reasoned."


def test_parse_score_stops_rationale_at_crlf_blank_line():
    """CRLF responses (\\r\\n\\r\\n) must still register as a paragraph
    break so the postscript isn't swallowed."""
    text = (
        "Score: 5\r\n"
        "Rationale: reasoning covers the main factors.\r\n"
        "\r\n"
        "Note: trailing postscript."
    )
    raw, rationale = _parse_score(text)
    assert raw == 5
    assert rationale == "reasoning covers the main factors."


def test_parse_score_self_correction_without_trailing_rationale():
    """A self-correction that emits a later 'Score:' with NO following
    'Rationale:' must NOT pair the new score with block-1's rationale
    (RT pass-3 MAJOR-2). Rationale is anchored to the last Score."""
    text = (
        "Score: 3\n"
        "Rationale: thin reasoning\n"
        "\n"
        "On reflection the chain is stronger than I credited.\n"
        "Score: 8"
    )
    raw, rationale = _parse_score(text)
    assert raw == 8
    assert rationale == ""


def test_parse_score_rationale_anchored_after_last_score():
    """A judge echoing the rubric emits an earlier 'Rationale:' label; the
    judge's own answer is the first 'Rationale:' AFTER the last 'Score:'."""
    text = (
        "Rationale: ONE_SHORT_SENTENCE\n"   # echoed rubric instruction
        "...\n"
        "Score: 8\n"
        "Rationale: the decision is well reasoned"
    )
    raw, rationale = _parse_score(text)
    assert raw == 8
    assert rationale == "the decision is well reasoned"


# Second-pass BLOCKER-1, BLOCKER-2: last-match wins so rubric echoes
# and agent_output echoes can't trip the parser.
@pytest.mark.parametrize(
    "text, expected_score",
    [
        # Model echoes the rubric example then emits its own score.
        (
            "The rubric says Score: 7 means well-reasoned.\n\n"
            "Score: 4\nRationale: thin reasoning",
            4,
        ),
        # Agent_output mentioned 'Score: 9' which the judge paraphrases,
        # then judge emits its own conclusion.
        (
            "The agent reasoning included 'Score: 9 for credit history'.\n"
            "Looking at the whole pipeline:\n"
            "Score: 6\nRationale: middling",
            6,
        ),
        # Self-correction: model changes its mind.
        (
            "Initially I thought Score: 3 but on reflection\n"
            "Score: 8\nRationale: nuanced handling",
            8,
        ),
        # Postscript clarification doesn't change the answer.
        (
            "Score: 7\nRationale: ok\n\nP.S. To be clear, Score: 7 is final.",
            7,
        ),
    ],
)
def test_parse_score_last_match_wins(text, expected_score):
    raw, _ = _parse_score(text)
    assert raw == expected_score


# ---------------------------------------------------------------------------
# SingleJudge — construction + rubric loading
# ---------------------------------------------------------------------------


def _write_rubric(path: Path, body: str | None = None) -> Path:
    """Write a minimal rubric template. Returns the path."""
    if body is None:
        body = (
            "Variant: {variant_id}\n"
            "Applicant: {applicant_blurb}\n"
            "Truth: {ground_truth_decision}\n"
            "Output: {agent_output}\n"
            "\n"
            "Score: <int 0-10>\nRationale: <text>\n"
        )
    path.write_text(body, encoding="utf-8")
    return path


def test_single_judge_loads_rubric_and_exposes_hash(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    j = SingleJudge(model="google/gemini-2.5-pro", rubric_path=rubric)
    assert j.model == "google/gemini-2.5-pro"
    assert j.rubric_path == rubric
    assert isinstance(j.rubric_hash, str) and len(j.rubric_hash) == 64
    assert j.dry_run is False


def test_single_judge_hash_is_stable_across_constructions(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    h1 = SingleJudge(model="m", rubric_path=rubric).rubric_hash
    h2 = SingleJudge(model="m", rubric_path=rubric).rubric_hash
    assert h1 == h2


def test_single_judge_hash_changes_on_rubric_edit(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt", body="A {variant_id} {applicant_blurb} {ground_truth_decision} {agent_output}")
    h1 = SingleJudge(model="m", rubric_path=rubric).rubric_hash
    rubric.write_text("B {variant_id} {applicant_blurb} {ground_truth_decision} {agent_output}", encoding="utf-8")
    h2 = SingleJudge(model="m", rubric_path=rubric).rubric_hash
    assert h1 != h2


# Audit B3: rubric_hash must be EOL-agnostic so Windows clones under
# `core.autocrlf=true` produce the same cache key as Linux/macOS clones.
def test_single_judge_hash_is_eol_agnostic(tmp_path):
    body_lf = "Hello {variant_id}\nApplicant: {applicant_blurb}\nGround: {ground_truth_decision}\nOutput: {agent_output}\n"
    body_crlf = body_lf.replace("\n", "\r\n")
    body_cr = body_lf.replace("\n", "\r")

    p_lf = tmp_path / "lf.txt"
    p_lf.write_bytes(body_lf.encode("utf-8"))
    p_crlf = tmp_path / "crlf.txt"
    p_crlf.write_bytes(body_crlf.encode("utf-8"))
    p_cr = tmp_path / "cr.txt"
    p_cr.write_bytes(body_cr.encode("utf-8"))

    h_lf = SingleJudge(model="m", rubric_path=p_lf).rubric_hash
    h_crlf = SingleJudge(model="m", rubric_path=p_crlf).rubric_hash
    h_cr = SingleJudge(model="m", rubric_path=p_cr).rubric_hash
    assert h_lf == h_crlf == h_cr


# Audit B3 (defense in depth): UTF-8 BOM must be stripped before hashing,
# so a Notepad-saved rubric matches a CLI-edited rubric.
def test_single_judge_hash_strips_utf8_bom(tmp_path):
    body = "Hello {variant_id} {applicant_blurb} {ground_truth_decision} {agent_output}"
    p_plain = tmp_path / "plain.txt"
    p_plain.write_bytes(body.encode("utf-8"))
    p_bom = tmp_path / "bom.txt"
    p_bom.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    assert (
        SingleJudge(model="m", rubric_path=p_plain).rubric_hash
        == SingleJudge(model="m", rubric_path=p_bom).rubric_hash
    )


def test_single_judge_rejects_missing_rubric(tmp_path):
    with pytest.raises(FileNotFoundError):
        SingleJudge(model="m", rubric_path=tmp_path / "nope.txt")


def test_single_judge_rejects_empty_rubric(tmp_path):
    rubric = tmp_path / "empty.txt"
    rubric.write_text("   \n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        SingleJudge(model="m", rubric_path=rubric)


def test_single_judge_rejects_non_utf8_rubric(tmp_path):
    rubric = tmp_path / "binary.txt"
    rubric.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(ValueError, match="UTF-8"):
        SingleJudge(model="m", rubric_path=rubric)


def test_single_judge_rejects_empty_model(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    with pytest.raises(ValueError, match="model"):
        SingleJudge(model="", rubric_path=rubric)


def test_single_judge_rejects_bad_max_tokens(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    with pytest.raises(ValueError, match="max_tokens"):
        SingleJudge(model="m", rubric_path=rubric, max_tokens=0)


# ---------------------------------------------------------------------------
# SingleJudge.render_prompt
# ---------------------------------------------------------------------------


def _payload(**overrides) -> dict:
    base = {
        "variant_id": "v-abc",
        "eval_row": {
            "id": "e1",
            "input": "applicant text here",
            "ground_truth_decision": "approve",
        },
        "agent_output": "decision: APPROVE — reason",
    }
    base.update(overrides)
    return base


def test_render_prompt_substitutes_placeholders(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    j = SingleJudge(model="m", rubric_path=rubric)
    text = j.render_prompt(_payload())
    assert "v-abc" in text
    assert "applicant text here" in text
    assert "approve" in text
    assert "decision: APPROVE — reason" in text


def test_render_prompt_prefers_applicant_blurb_over_input(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    j = SingleJudge(model="m", rubric_path=rubric)
    payload = _payload()
    payload["eval_row"]["applicant_blurb"] = "BLURB WINS"
    text = j.render_prompt(payload)
    assert "BLURB WINS" in text
    assert "applicant text here" not in text


def test_render_prompt_falls_back_to_input(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    j = SingleJudge(model="m", rubric_path=rubric)
    payload = _payload()
    payload["eval_row"].pop("input", None)
    payload["eval_row"]["applicant_blurb"] = "via blurb"
    text = j.render_prompt(payload)
    assert "via blurb" in text


def test_render_prompt_raises_when_no_applicant_text(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    j = SingleJudge(model="m", rubric_path=rubric)
    payload = _payload()
    payload["eval_row"].pop("input", None)
    with pytest.raises(KeyError, match="applicant_blurb"):
        j.render_prompt(payload)


def test_render_prompt_raises_when_eval_row_not_dict(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    j = SingleJudge(model="m", rubric_path=rubric)
    with pytest.raises(TypeError, match="eval_row"):
        j.render_prompt({"variant_id": "v", "eval_row": "no", "agent_output": "x"})


def test_render_prompt_surfaces_template_brace_typo(tmp_path):
    bad = "Variant: {variant_id} but stray {nope} brace"
    rubric = _write_rubric(tmp_path / "r.txt", body=bad)
    j = SingleJudge(model="m", rubric_path=rubric)
    with pytest.raises(ValueError, match="brace"):
        j.render_prompt(_payload())


# ---------------------------------------------------------------------------
# SingleJudge.score — happy path, dry-run, retry, give-up
# ---------------------------------------------------------------------------


class _StubLLM:
    """Sync stub mirroring the LiteLLM .completion call shape."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        try:
            text = self._responses.pop(0)
        except IndexError:
            raise AssertionError("StubLLM ran out of responses")
        return {"choices": [{"message": {"content": text}}], "model": kwargs["model"]}


def test_single_judge_score_happy_path(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    stub = _StubLLM(["Score: 8\nRationale: solid reasoning\n"])
    j = SingleJudge(model="m", rubric_path=rubric, llm_callable=stub)
    result = j.score(_payload())
    assert isinstance(result, ContinuousScore)
    assert result.value == pytest.approx(0.8)
    assert len(stub.calls) == 1


def test_single_judge_default_max_tokens_reaches_llm_call(tmp_path):
    """The configured max_tokens is what gets sent to the LLM. Regression
    guard for the 256-default truncation bug: the judge reasons before
    emitting Score/Rationale, so the budget must cover that reasoning.
    The floor assertion catches a regression to a too-small default
    without pinning the exact value."""
    from fuzzeval.judge import _DEFAULT_MAX_TOKENS

    assert _DEFAULT_MAX_TOKENS >= 1000
    rubric = _write_rubric(tmp_path / "r.txt")
    stub = _StubLLM(["Score: 5\nRationale: ok\n"])
    j = SingleJudge(model="m", rubric_path=rubric, llm_callable=stub)
    j.score(_payload())
    assert stub.calls[0]["max_tokens"] == _DEFAULT_MAX_TOKENS


def test_single_judge_threads_parsed_rationale_into_score(tmp_path):
    """The Rationale: line the LLM emits is carried on the ContinuousScore,
    not just debug-logged — that is what makes judge_results.rationale
    non-NULL for continuous cells."""
    rubric = _write_rubric(tmp_path / "r.txt")
    stub = _StubLLM(["Score: 7\nRationale: applicant meets all criteria\n"])
    j = SingleJudge(model="m", rubric_path=rubric, llm_callable=stub)
    result = j.score(_payload())
    assert result.rationale == "applicant meets all criteria"


def test_single_judge_missing_rationale_line_yields_empty_string(tmp_path):
    """A judge response with no Rationale: line still scores; rationale is ""."""
    rubric = _write_rubric(tmp_path / "r.txt")
    stub = _StubLLM(["Score: 5\n"])
    j = SingleJudge(model="m", rubric_path=rubric, llm_callable=stub)
    result = j.score(_payload())
    assert result.value == pytest.approx(0.5)
    assert result.rationale == ""


def test_single_judge_rationale_is_length_capped(tmp_path):
    """A pathologically long Rationale: line is capped before it reaches the
    ContinuousScore — bounding what lands in run.sqlite / the signed
    attestation."""
    from fuzzeval.judge import _RATIONALE_MAX_CHARS

    rubric = _write_rubric(tmp_path / "r.txt")
    long_rationale = "x" * (_RATIONALE_MAX_CHARS + 500)
    stub = _StubLLM([f"Score: 6\nRationale: {long_rationale}\n"])
    j = SingleJudge(model="m", rubric_path=rubric, llm_callable=stub)
    result = j.score(_payload())
    assert len(result.rationale) == _RATIONALE_MAX_CHARS


def test_single_judge_dry_run_skips_llm(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")

    def boom(**kwargs):
        raise AssertionError("dry-run must not call the LLM")

    j = SingleJudge(model="m", rubric_path=rubric, dry_run=True, llm_callable=boom)
    result = j.score(_payload())
    assert isinstance(result, ContinuousScore)
    assert result.value == 0.5
    assert result.rationale == "[DRY-RUN]"


def test_single_judge_dry_run_still_renders_for_validation(tmp_path):
    """Dry-run must surface template errors instead of silently returning 0.5."""
    rubric = _write_rubric(tmp_path / "r.txt", body="bad: {nope}")
    j = SingleJudge(model="m", rubric_path=rubric, dry_run=True)
    with pytest.raises((KeyError, ValueError)):
        j.score(_payload())


def test_single_judge_does_not_retry_on_parser_failure(tmp_path):
    """Second-pass MAJOR-1: parser failures are deterministic at
    temperature=0; retry burns a billed call to produce the same result.
    Skip the retry for JudgeUnparseableError and surface immediately."""
    rubric = _write_rubric(tmp_path / "r.txt")
    stub = _StubLLM(["I forgot the score"])
    j = SingleJudge(model="m", rubric_path=rubric, llm_callable=stub)
    with pytest.raises(JudgeUnparseableError):
        j.score(_payload())
    assert len(stub.calls) == 1  # no retry on parser failure


def test_single_judge_does_not_swallow_keyboard_interrupt(tmp_path):
    """KeyboardInterrupt from the LLM call must propagate, not become a parse fail."""
    rubric = _write_rubric(tmp_path / "r.txt")

    def boom(**kwargs):
        raise KeyboardInterrupt

    j = SingleJudge(model="m", rubric_path=rubric, llm_callable=boom)
    with pytest.raises(KeyboardInterrupt):
        j.score(_payload())


# Audit M5: transient bad-envelope (TypeError from _extract_judge_text)
# must retry, not bypass the retry contract.
def test_single_judge_retries_on_bad_envelope_then_succeeds(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")

    calls = {"n": 0}

    def two_shapes(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # First call: provider returns a content-filter empty-choices
            # response. _extract_judge_text raises TypeError.
            return {"choices": [], "model": kwargs["model"]}
        return {
            "choices": [{"message": {"content": "Score: 7\nRationale: ok"}}],
            "model": kwargs["model"],
        }

    j = SingleJudge(model="m", rubric_path=rubric, llm_callable=two_shapes)
    result = j.score(_payload())
    assert isinstance(result, ContinuousScore)
    assert result.value == pytest.approx(0.7)
    assert calls["n"] == 2


def test_single_judge_raises_after_two_bad_envelopes(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")

    def always_bad(**kwargs):
        return {"choices": [], "model": kwargs["model"]}

    j = SingleJudge(model="m", rubric_path=rubric, llm_callable=always_bad)
    with pytest.raises(JudgeUnparseableError, match="twice"):
        j.score(_payload())


# ---------------------------------------------------------------------------
# binary_from_continuous
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score_value, threshold, expected_passed",
    [
        (0.7, 0.5, True),
        (0.4, 0.5, False),
        (0.5, 0.5, True),   # >= is the contract
        (1.0, 0.99, True),
        (0.0, 0.0, True),
    ],
)
def test_binary_from_continuous_thresholds(score_value, threshold, expected_passed):
    s = ContinuousScore(value=score_value)
    b = binary_from_continuous(s, threshold=threshold)
    assert isinstance(b, BinaryScore)
    assert b.passed is expected_passed


def test_binary_from_continuous_rejects_non_continuous():
    with pytest.raises(TypeError):
        binary_from_continuous(BinaryScore(passed=True, rationale="x"))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_binary_from_continuous_rejects_threshold_out_of_range(bad):
    with pytest.raises(ValueError):
        binary_from_continuous(ContinuousScore(value=0.5), threshold=bad)


# ---------------------------------------------------------------------------
# JuryJudge — Weekend 4 step 12b
#
# Tests use stub LLM callables per the locked step-12b decision: only
# Gemini key is confirmed available, so the unit tests are vendor-name-
# agnostic and inject responses per-vendor. Real-keys integration is
# deferred to v0.5.1.
# ---------------------------------------------------------------------------


from fuzzeval.judge import JuryJudge  # noqa: E402


def _score_response(score: int, rationale: str = "stub") -> str:
    return f"Score: {score}\nRationale: {rationale}\n"


def _jury_stubs(
    responses_by_vendor: dict[str, str],
) -> dict[str, _StubLLM]:
    """Build a fresh _StubLLM per vendor from a single-response mapping."""
    return {v: _StubLLM([r]) for v, r in responses_by_vendor.items()}


def _make_jury(
    tmp_path: Path,
    *,
    callables: dict[str, _StubLLM] | None = None,
    dry_run: bool = False,
    vendor_models: dict[str, str] | None = None,
) -> JuryJudge:
    rubric = _write_rubric(tmp_path / "r.txt")
    return JuryJudge(
        rubric_path=rubric,
        vendor_models=vendor_models,
        llm_callables=callables,
        dry_run=dry_run,
    )


# --- construction + validation -------------------------------------------


def test_jury_judge_constructs_with_default_vendors(tmp_path):
    j = _make_jury(tmp_path)
    assert set(j.vendors) == {"anthropic", "openai", "gemini"}
    assert j.model == "jury_of_3:anthropic+gemini+openai"  # sorted
    # rubric_hash equals what a SingleJudge over the same rubric produces.
    single = SingleJudge(model="m", rubric_path=j.rubric_path)
    assert j.rubric_hash == single.rubric_hash
    assert j.dry_run is False
    assert j.judge_failure_rate == 0.0
    assert j.judge_independence == {"vendor_overlap": []}


def test_jury_judge_rejects_wrong_vendor_count(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    with pytest.raises(ValueError, match="exactly 3"):
        JuryJudge(
            rubric_path=rubric,
            vendor_models={"anthropic": "a/x", "openai": "o/y"},
        )
    with pytest.raises(ValueError, match="exactly 3"):
        JuryJudge(
            rubric_path=rubric,
            vendor_models={
                "anthropic": "a/x",
                "openai": "o/y",
                "gemini": "g/z",
                "extra": "e/w",
            },
        )


def test_jury_judge_rejects_empty_vendor_name(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    with pytest.raises(ValueError, match="non-empty"):
        JuryJudge(
            rubric_path=rubric,
            vendor_models={"": "a/x", "openai": "o/y", "gemini": "g/z"},
        )


def test_jury_judge_rejects_unknown_callable_vendor(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")
    with pytest.raises(ValueError, match="not in vendor_models"):
        JuryJudge(
            rubric_path=rubric,
            llm_callables={"meta": _StubLLM(["Score: 5"])},
        )


def test_jury_judge_uses_same_rubric_path_for_all_vendors(tmp_path):
    j = _make_jury(tmp_path)
    # All three SingleJudges share the same on-disk rubric, so the
    # public rubric_path equals the constructor's argument.
    expected = (tmp_path / "r.txt").resolve()
    assert j.rubric_path.resolve() == expected


def test_jury_judge_render_prompt_matches_single_judge(tmp_path):
    j = _make_jury(tmp_path)
    single = SingleJudge(model="m", rubric_path=j.rubric_path)
    assert j.render_prompt(_payload()) == single.render_prompt(_payload())


# --- aggregation ---------------------------------------------------------


def test_jury_judge_means_three_parseable_votes(tmp_path):
    stubs = _jury_stubs(
        {
            "anthropic": _score_response(4),  # 0.4
            "openai": _score_response(6),  # 0.6
            "gemini": _score_response(8),  # 0.8
        }
    )
    j = _make_jury(tmp_path, callables=stubs)
    result = j.score(_payload())
    assert isinstance(result, ContinuousScore)
    # Mean of 0.4, 0.6, 0.8 == 0.6 (exact in float arithmetic for these
    # decimals but use approx anyway for forward-compat).
    assert result.value == pytest.approx(0.6)
    assert j.judge_failure_rate == 0.0
    assert j.judge_independence == {
        "vendor_overlap": ["anthropic", "gemini", "openai"],
    }


def test_jury_judge_aggregates_vendor_rationales(tmp_path):
    """The persisted rationale reflects the whole jury — each parseable
    vendor's Rationale: line, vendor-prefixed and joined."""
    stubs = _jury_stubs(
        {
            "anthropic": _score_response(4, rationale="weak evidence"),
            "openai": _score_response(6, rationale="adequate"),
            "gemini": _score_response(8, rationale="strong case"),
        }
    )
    j = _make_jury(tmp_path, callables=stubs)
    result = j.score(_payload())
    # Vendors iterate in dict order; assert each vendor's line is present
    # rather than pinning the exact join order.
    assert "anthropic: weak evidence" in result.rationale
    assert "openai: adequate" in result.rationale
    assert "gemini: strong case" in result.rationale
    # ...joined with the " | " delimiter (3 fragments → 2 delimiters).
    assert result.rationale.count(" | ") == 2


def test_jury_judge_joined_rationale_is_length_capped(tmp_path):
    """Each vendor fragment is capped by _parse_score, but the JOINED result
    must also be capped — 3 max-length fragments would otherwise blow past
    _RATIONALE_MAX_CHARS into the signed run.sqlite."""
    from fuzzeval.judge import _RATIONALE_MAX_CHARS

    big = "x" * (_RATIONALE_MAX_CHARS + 500)
    stubs = _jury_stubs(
        {
            "anthropic": _score_response(4, rationale=big),
            "openai": _score_response(6, rationale=big),
            "gemini": _score_response(8, rationale=big),
        }
    )
    j = _make_jury(tmp_path, callables=stubs)
    result = j.score(_payload())
    assert len(result.rationale) == _RATIONALE_MAX_CHARS


def test_jury_judge_rationale_excludes_failed_vendor(tmp_path):
    """A dropped (unparseable) vendor contributes no rationale fragment."""
    stubs = _jury_stubs(
        {
            "anthropic": "no score line here",
            "openai": _score_response(6, rationale="adequate"),
            "gemini": _score_response(8, rationale="strong case"),
        }
    )
    j = _make_jury(tmp_path, callables=stubs)
    result = j.score(_payload())
    assert "anthropic" not in result.rationale
    assert "openai: adequate" in result.rationale
    assert "gemini: strong case" in result.rationale


def test_jury_judge_all_empty_rationales_yield_empty_string(tmp_path):
    """When no parseable vendor emitted a Rationale: line, the aggregate is
    "" — not a string of bare "vendor: " delimiters that would persist as a
    non-NULL noise value."""
    stubs = _jury_stubs(
        {
            "anthropic": "Score: 4\n",
            "openai": "Score: 6\n",
            "gemini": "Score: 8\n",
        }
    )
    j = _make_jury(tmp_path, callables=stubs)
    result = j.score(_payload())
    assert result.value == pytest.approx(0.6)
    assert result.rationale == ""


def test_jury_judge_drops_unparseable_vote_and_means_survivors(tmp_path):
    stubs = _jury_stubs(
        {
            "anthropic": "I refuse to give a score (no Score: line)",
            "openai": _score_response(6),  # 0.6
            "gemini": _score_response(8),  # 0.8
        }
    )
    j = _make_jury(tmp_path, callables=stubs)
    result = j.score(_payload())
    # Mean of 0.6 and 0.8 == 0.7 — anthropic's vote dropped.
    assert result.value == pytest.approx(0.7)
    assert j.judge_failure_rate == pytest.approx(1 / 3)
    assert j.judge_independence == {
        "vendor_overlap": ["gemini", "openai"],
    }


def test_jury_judge_raises_when_all_vendors_unparseable(tmp_path):
    stubs = _jury_stubs(
        {
            "anthropic": "no score line",
            "openai": "still no score",
            "gemini": "neither here",
        }
    )
    j = _make_jury(tmp_path, callables=stubs)
    with pytest.raises(JudgeUnparseableError, match="all 3 vendors failed"):
        j.score(_payload())
    assert j.judge_failure_rate == 1.0
    assert j.judge_independence == {"vendor_overlap": []}


def test_jury_judge_treats_sdk_exception_as_failure(tmp_path):
    """Vendor SDK / network error counts as a failed vendor call — not a
    crash. The BLUEPRINT §8 judge_failure_rate is the right surface."""

    def explode(**kwargs):
        raise ConnectionError("provider timeout")

    callables = {
        "anthropic": explode,  # not a _StubLLM — a raw callable that raises
        "openai": _StubLLM([_score_response(6)]),
        "gemini": _StubLLM([_score_response(8)]),
    }
    j = _make_jury(tmp_path, callables=callables)
    result = j.score(_payload())
    assert result.value == pytest.approx(0.7)
    assert j.judge_failure_rate == pytest.approx(1 / 3)
    assert "anthropic" not in j.judge_independence["vendor_overlap"]


# --- accounting across multiple calls ------------------------------------


def test_jury_judge_accounting_accumulates_across_calls(tmp_path):
    """3 score() calls, anthropic fails on each → failure_rate == 1/3 of 9
    vendor calls. Other two vendors parse all three calls."""
    rubric = _write_rubric(tmp_path / "r.txt")
    callables = {
        "anthropic": _StubLLM(["no score"] * 3),
        "openai": _StubLLM([_score_response(6)] * 3),
        "gemini": _StubLLM([_score_response(8)] * 3),
    }
    j = JuryJudge(rubric_path=rubric, llm_callables=callables)
    for _ in range(3):
        r = j.score(_payload())
        assert r.value == pytest.approx(0.7)
    assert j.judge_failure_rate == pytest.approx(3 / 9)
    assert j.judge_independence == {
        "vendor_overlap": ["gemini", "openai"],
    }


def test_jury_judge_vendor_overlap_records_any_parseable_vote(tmp_path):
    """A vendor that parses on call 1 but fails on call 2 still appears in
    vendor_overlap — the field is "vendors with at least one parseable
    vote", not "vendors that parsed every call"."""
    rubric = _write_rubric(tmp_path / "r.txt")
    callables = {
        "anthropic": _StubLLM([_score_response(5), "no score"]),
        "openai": _StubLLM([_score_response(6), _score_response(7)]),
        "gemini": _StubLLM([_score_response(8), _score_response(9)]),
    }
    j = JuryJudge(rubric_path=rubric, llm_callables=callables)
    j.score(_payload())
    j.score(_payload())
    assert sorted(j.judge_independence["vendor_overlap"]) == [
        "anthropic", "gemini", "openai",
    ]
    # 1 fail out of 6 calls = 1/6.
    assert j.judge_failure_rate == pytest.approx(1 / 6)


# --- dry-run -------------------------------------------------------------


def test_jury_judge_dry_run_returns_half_with_no_llm(tmp_path):
    def boom(**kwargs):
        raise AssertionError("dry-run must not call any LLM")

    callables = {
        "anthropic": boom,
        "openai": boom,
        "gemini": boom,
    }
    j = _make_jury(tmp_path, callables=callables, dry_run=True)
    r = j.score(_payload())
    assert r.value == 0.5
    # Dry-run is parseable by construction — vendor_overlap has all three.
    assert j.judge_independence == {
        "vendor_overlap": ["anthropic", "gemini", "openai"],
    }
    assert j.judge_failure_rate == 0.0


# --- rubric_hash invariants ----------------------------------------------


def test_jury_judge_rubric_hash_matches_single_judge_on_same_rubric(tmp_path):
    body = "X {variant_id} {applicant_blurb} {ground_truth_decision} {agent_output}"
    rubric = _write_rubric(tmp_path / "r.txt", body=body)
    j = JuryJudge(rubric_path=rubric)
    single = SingleJudge(model="m", rubric_path=rubric)
    assert j.rubric_hash == single.rubric_hash


def test_jury_judge_rubric_hash_is_eol_agnostic(tmp_path):
    body_lf = (
        "{variant_id}\n{applicant_blurb}\n"
        "{ground_truth_decision}\n{agent_output}\n"
    )
    p_lf = tmp_path / "lf.txt"
    p_lf.write_bytes(body_lf.encode("utf-8"))
    p_crlf = tmp_path / "crlf.txt"
    p_crlf.write_bytes(body_lf.replace("\n", "\r\n").encode("utf-8"))
    assert (
        JuryJudge(rubric_path=p_lf).rubric_hash
        == JuryJudge(rubric_path=p_crlf).rubric_hash
    )


# --- protocol structural conformance -------------------------------------


def test_jury_judge_conforms_to_judge_protocol(tmp_path):
    from fuzzeval.judge import Judge

    j: Judge = _make_jury(
        tmp_path,
        callables=_jury_stubs(
            {
                "anthropic": _score_response(5),
                "openai": _score_response(5),
                "gemini": _score_response(5),
            }
        ),
    )
    r = j.score(_payload())
    assert isinstance(r, ContinuousScore)
    assert r.value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Pass-1 audit-fix coverage
# ---------------------------------------------------------------------------


# RT-M2 — vendor name normalization (strip + lowercase) and dup-VALUE rejection.


def test_jury_judge_normalizes_vendor_names(tmp_path):
    """Case-variant + whitespace-decorated keys collapse to canonical form."""
    rubric = _write_rubric(tmp_path / "r.txt")
    j = JuryJudge(
        rubric_path=rubric,
        vendor_models={
            "  Anthropic ": "anthropic/claude-opus-4-7",
            "OpenAI": "openai/gpt-5",
            "GEMINI": "google/gemini-2.5-pro",
        },
    )
    assert set(j.vendors) == {"anthropic", "openai", "gemini"}
    assert j.model == "jury_of_3:anthropic+gemini+openai"  # sorted


def test_jury_judge_rejects_case_variant_duplicates(tmp_path):
    """Two raw keys normalizing to the same canonical name → loud failure."""
    rubric = _write_rubric(tmp_path / "r.txt")
    with pytest.raises(ValueError, match="collide after normalization"):
        JuryJudge(
            rubric_path=rubric,
            vendor_models={
                "anthropic": "anthropic/claude-opus-4-7",
                "Anthropic ": "anthropic/claude-3",  # collides post-strip+lower
                "openai": "openai/gpt-5",
            },
        )


@pytest.mark.parametrize(
    "vendor_models",
    [
        # All three same value.
        {
            "anthropic": "google/gemini-2.5-pro",
            "openai": "google/gemini-2.5-pro",
            "gemini": "google/gemini-2.5-pro",
        },
        # Two same, one different — pass-2 RT MINOR-1 audit fix:
        # the original test only covered the unanimous-collision
        # case. Adding the partial-collision parametrization guards
        # a future regression that loosened the check to "all three
        # same".
        {
            "anthropic": "google/gemini-2.5-pro",
            "openai": "google/gemini-2.5-pro",
            "gemini": "openai/gpt-5",
        },
    ],
)
def test_jury_judge_rejects_duplicate_model_values(tmp_path, vendor_models):
    """Vendors pointing at the same LiteLLM model id defeat the
    independence claim. Rejected at construction in both all-same and
    partial-collision configurations."""
    rubric = _write_rubric(tmp_path / "r.txt")
    with pytest.raises(ValueError, match="vendor models must be distinct"):
        JuryJudge(rubric_path=rubric, vendor_models=vendor_models)


def test_jury_judge_normalizes_callables_keys(tmp_path):
    """Caller passes case-variant callable keys → they bind to the
    same canonical vendors, not silently mis-route."""
    rubric = _write_rubric(tmp_path / "r.txt")
    stubs = {
        "Anthropic ": _StubLLM([_score_response(4)]),
        "OPENAI": _StubLLM([_score_response(6)]),
        "Gemini": _StubLLM([_score_response(8)]),
    }
    j = JuryJudge(rubric_path=rubric, llm_callables=stubs)
    result = j.score(_payload())
    assert result.value == pytest.approx(0.6)
    assert j.judge_failure_rate == 0.0


# CR-M2 / RT-M4 — concurrency lock smoke test. We don't try to *force*
# a race (that's flaky); we instead prove the accumulators stay
# internally consistent under threaded fan-out.


def test_jury_judge_accounting_is_consistent_under_threaded_score(tmp_path):
    """**Coherence smoke test** — NOT a race-reproducer.

    Drives 50 concurrent ``score()`` calls × 3 vendors via an 8-worker
    thread pool. Verifies the accumulators land at exactly the expected
    cumulative values: ``total_vendor_calls == 150``,
    ``failed_vendor_calls == 0``, ``judge_failure_rate == 0.0``,
    ``vendor_overlap`` covers all three.

    **Caveat (pass-2 RT MAJOR-2):** the stub does pure-Python compute
    with no I/O, so the GIL serializes most of the per-call work; the
    test would likely PASS even without the accounting lock because
    Python's ``int.__iadd__`` is effectively atomic for the single
    ``self._total_vendor_calls += local_total`` line. The right
    falsifiability surface for a race would inject ``time.sleep`` mid-
    call to release the GIL — that's deferred to v0.5.1 if a real
    contention issue surfaces in production. The lock is documented at
    the write site (``score()``) and the read sites (the four property
    accessors) so a future audit can trust the contract by inspection
    rather than via this test.
    """
    import concurrent.futures

    rubric = _write_rubric(tmp_path / "r.txt")

    class _AlwaysParseable:
        """Thread-safe stub returning a parseable response every call."""

        def __call__(self, **kwargs):
            return {
                "choices": [{"message": {"content": _score_response(5)}}],
                "model": kwargs["model"],
            }

    j = JuryJudge(
        rubric_path=rubric,
        llm_callables={
            "anthropic": _AlwaysParseable(),
            "openai": _AlwaysParseable(),
            "gemini": _AlwaysParseable(),
        },
    )

    n_calls = 50
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: j.score(_payload()), range(n_calls)))

    assert j.total_vendor_calls == 3 * n_calls
    assert j.failed_vendor_calls == 0
    assert j.judge_failure_rate == 0.0
    assert j.judge_independence == {
        "vendor_overlap": ["anthropic", "gemini", "openai"],
    }


# RT-M1 — by-design acknowledgement: 3 vendors unanimously returning
# the same low score IS a legitimate consensus, NOT a failure. This
# test pins the contract so a future RT specialist doesn't re-raise it
# as a silent-pass hole.


def test_jury_judge_unanimous_low_score_is_legitimate(tmp_path):
    """3 stubs all returning "Score: 0" → mean 0.0, vendor_overlap full,
    judge_failure_rate 0. This is by-design per the mean-of-parseable
    aggregation contract — unanimous consensus is a strong signal even
    when the consensus value is low."""
    stubs = _jury_stubs(
        {
            "anthropic": _score_response(0),
            "openai": _score_response(0),
            "gemini": _score_response(0),
        }
    )
    j = _make_jury(tmp_path, callables=stubs)
    result = j.score(_payload())
    assert result.value == 0.0
    assert j.judge_failure_rate == 0.0
    assert j.judge_independence == {
        "vendor_overlap": ["anthropic", "gemini", "openai"],
    }


# RT-9 — BinaryScore from a vendor is treated as a failure (the
# JuryJudge contract is mean-of-CONTINUOUS scores; a vendor returning a
# different result type can't be aggregated into the mean).


def test_jury_judge_treats_non_continuous_score_as_failure(tmp_path):
    """A vendor returning a non-ContinuousScore (e.g., BinaryScore via a
    custom Judge subclass) gets its vote dropped and accumulates as a
    failed vendor call."""
    from fuzzeval.judge import BinaryScore

    rubric = _write_rubric(tmp_path / "r.txt")

    class _BinaryReturningJudge:
        """Stub that returns a BinaryScore — not the ContinuousScore
        that SingleJudge would return. Bypasses the SingleJudge layer
        entirely by injecting via the per-vendor SingleJudge's
        llm_callable... actually no, we need a custom path here:
        we mock the SingleJudge.score by patching the instance method
        after construction."""

        pass

    j = JuryJudge(
        rubric_path=rubric,
        llm_callables={
            "anthropic": _StubLLM([_score_response(5)]),
            "openai": _StubLLM([_score_response(7)]),
            "gemini": _StubLLM([_score_response(9)]),
        },
    )
    # Patch one judge's score method to return a BinaryScore.
    j._judges["anthropic"].score = lambda payload: BinaryScore(
        passed=True, rationale="off-protocol"
    )
    result = j.score(_payload())
    # Mean of the surviving two parseable continuous votes: (0.7 + 0.9) / 2 = 0.8.
    assert result.value == pytest.approx(0.8)
    assert j.judge_failure_rate == pytest.approx(1 / 3)
    assert j.judge_independence == {
        "vendor_overlap": ["gemini", "openai"],
    }


# RT-pass-1 implicit — KeyboardInterrupt propagation. A user Ctrl+C in
# the middle of a jury call must NOT be swallowed as a vendor failure.


def test_jury_judge_does_not_swallow_keyboard_interrupt(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")

    def boom(**kwargs):
        raise KeyboardInterrupt

    j = JuryJudge(
        rubric_path=rubric,
        llm_callables={
            "anthropic": boom,
            "openai": _StubLLM([_score_response(6)]),  # never called
            "gemini": _StubLLM([_score_response(8)]),  # never called
        },
    )
    with pytest.raises(KeyboardInterrupt):
        j.score(_payload())
    # And: the partial state is NOT committed — total_vendor_calls
    # stays 0 because the lock-batched apply never ran.
    assert j.total_vendor_calls == 0
