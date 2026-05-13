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


def test_single_judge_dry_run_skips_llm(tmp_path):
    rubric = _write_rubric(tmp_path / "r.txt")

    def boom(**kwargs):
        raise AssertionError("dry-run must not call the LLM")

    j = SingleJudge(model="m", rubric_path=rubric, dry_run=True, llm_callable=boom)
    result = j.score(_payload())
    assert isinstance(result, ContinuousScore)
    assert result.value == 0.5


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
