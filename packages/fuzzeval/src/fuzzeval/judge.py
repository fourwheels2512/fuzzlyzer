"""Judge protocol, result types, and the v0.5 single-judge implementation.

Type contract (``ContinuousScore`` / ``BinaryScore`` / ``Judge``) lives in
this module because it is the BLUEPRINT_v5plus.md §17a frozen public API:
both topofuzz and (future) pramana import from here. Cache code in
:mod:`fuzzeval.cache` references ``JudgeResult`` for round-trip type
guards; runner code in :mod:`topofuzz.runner` injects ``Judge``-shaped
objects.

The :class:`SingleJudge` implementation (Weekend 2 step 7) renders a
plain-text rubric template, calls one LLM via :mod:`litellm`, parses
``Score: <int 0-10>`` and ``Rationale: <text>`` from the response, and
returns a :class:`ContinuousScore` with value in ``[0, 1]``. Failure
modes:

* **UNPARSEABLE** (regex match fails or score out of range) — retry once
  at ``temperature=0`` (the original request may itself have been at
  ``temperature=0``; the retry is a redundant assertion of determinism
  intended to ride out a transient nondeterministic provider hiccup),
  then raise :class:`JudgeUnparseableError`. The runner catches this and
  marks the cell ``judge_failure=True`` per
  :func:`topofuzz.runner._run_judge`.
* **Dry-run** — when ``dry_run=True`` the LLM is never called; every
  payload returns a deterministic ``ContinuousScore(0.5)`` with
  rationale ``"[DRY-RUN]"``. Cost is $0. Useful for exercising the
  rubric-loading + rubric-hash + cache-key path without billing.

``SingleJudge.rubric_hash`` is computed once at construction from the
rubric file's bytes and exposed as a read-only property. The orchestrator
threads this value into ``RunPlan.rubric_hash`` so the JudgeCache key is
stable across cells in one run and invalidates cleanly when the rubric
text changes.

``binary_from_continuous`` is a metrics-layer helper (called in step 9
when computing Cohen's κ vs the ``ground_truth_decision`` column). It
lives here rather than in ``fuzzeval.stats`` because the threshold
contract belongs with the score type it operates on.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from fuzzeval.canonical import canonical_hash

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# §17a frozen types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousScore:
    """A real-valued judge score in [0, 1]."""

    value: float


@dataclass(frozen=True)
class BinaryScore:
    """A pass/fail judge score with a rationale string."""

    passed: bool
    rationale: str


# PEP 604 union — runtime-checkable so ``isinstance(x, JudgeResult)`` works.
# Equivalent to typing.Union[ContinuousScore, BinaryScore] but cleaner and
# drops a typing import. Python 3.10+ required (we require 3.11+).
JudgeResult = ContinuousScore | BinaryScore


class Judge(Protocol):
    """Judge protocol. Implementations: :class:`SingleJudge`, JuryOf3 (step 12b).

    ``payload`` is a rubric-specific dict. Topofuzz will pass:
        {"variant_id", "eval_row", "agent_output"}
    Pramana (future) will pass:
        {"problem", "ground_truth", "candidate_answer", "nirnaya_section"}
    """

    def score(self, payload: dict) -> JudgeResult: ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class JudgeUnparseableError(RuntimeError):
    """Raised by :class:`SingleJudge` when the LLM response cannot be parsed
    into a ``Score: <int 0-10>`` line after one retry at ``temperature=0``.

    The runner converts this into ``Cell.judge_failure=True`` +
    ``judge_failure_reason`` per :func:`topofuzz.runner._run_judge`. The
    §8 ``judge_failure_rate > 5% → degraded`` accounting consumes that
    flag — single-judge has no jury fallback so one unparseable response
    fails the cell.
    """


# ---------------------------------------------------------------------------
# Score-extraction regex
# ---------------------------------------------------------------------------

# Match a line that starts (optionally after common markdown decoration —
# bold ``**``, headings ``#``, blockquote ``>``, bullets ``-``) with
# ``Score:`` and captures an int 0-10 that follows.
#
# Integer alternation ``(?:10|[0-9])`` accepts exactly the contract range
# (0-9 single digits, plus 10) — rejects leading zeros like ``Score: 07``
# (matched as ``0`` followed by stray ``7``) and three-digit scores like
# ``Score: 100`` (lookahead fails).
#
# Trailing-character lookaheads pin the integer:
#   ``(?!\.\d)`` — rejects half-step floats like ``Score: 8.5`` (audit
#     M1). A trailing period that ENDS the sentence (``Score: 7.``) is
#     accepted, matching natural English (audit second-pass M-A1).
#   ``(?!\d)``  — rejects three-digit runs like ``Score: 100``.
#
# Bold-pair patterns (``**Score:** 7``, ``Score: **7**``) are accepted
# by allowing optional ``*`` runs around the colon and the integer.
#
# Anchored to a line start so ``"my score is 7"`` does NOT match — we
# only accept the contract format spelled in the rubric.
_SCORE_RE = re.compile(
    r"(?im)^[\s*#>\-]*\**\s*Score:\**\s*\**\s*(10|[0-9])\**(?!\.\d)(?!\d)"
)
# Rationale capture. ``[\s*#>\-]*`` allows the same markdown leaders as
# Score. ``\**`` at end strips a trailing ``**`` from bold-wrapped
# rationales like ``**Rationale: ok**`` (second-pass MINOR-1).
_RATIONALE_RE = re.compile(
    r"(?im)^[\s*#>\-]*\**\s*Rationale:\**\s*(.+?)\**\s*$"
)


def _parse_score(response_text: str) -> tuple[int, str]:
    """Return (raw_score in [0, 10], rationale) parsed from a judge response.

    Raises :class:`JudgeUnparseableError` on:

    * No ``Score: N`` line at all.
    * (Currently unreachable — the regex pins ``N`` to ``0-10`` literally;
      kept as defense in depth.)

    **Last-match wins.** A judge response often contains echoes of the
    rubric (which itself shows the contract format) and / or echoes of
    ``agent_output`` (which may legitimately mention scores in its own
    reasoning). Both would surface as additional ``Score: N`` matches
    earlier in the response. The model's OWN final answer always appears
    LAST because the rubric instructs it to. So we take the last regex
    match as the authoritative score (second-pass BLOCKER-1, BLOCKER-2).

    Rationale is optional — a missing rationale becomes an empty string.
    """
    if not isinstance(response_text, str):
        raise JudgeUnparseableError(
            f"Judge response must be str; got {type(response_text).__name__}"
        )
    matches = _SCORE_RE.findall(response_text)
    if not matches:
        raise JudgeUnparseableError(
            "Judge response missing 'Score: <int>' line. Response was:\n"
            f"{response_text[:500]}"
        )
    raw = int(matches[-1])
    # Range check kept as defense in depth even though the regex
    # alternation pins N to 0-10 by construction. A future regex edit
    # that broadens the integer class would otherwise silently accept
    # out-of-range values.
    if not (0 <= raw <= 10):
        raise JudgeUnparseableError(
            f"Judge score {raw} out of range [0, 10]. Response was:\n"
            f"{response_text[:500]}"
        )
    rm = _RATIONALE_RE.search(response_text)
    rationale = rm.group(1).strip() if rm else ""
    return raw, rationale


# ---------------------------------------------------------------------------
# SingleJudge
# ---------------------------------------------------------------------------


# Type alias for the synchronous LLM callable. Synchronous because
# Judge.score is sync per the §17a Protocol; the runner offloads it to
# ``asyncio.to_thread`` so a slow judge doesn't block the event loop.
# Must accept (model, messages, temperature, max_tokens) kwargs and
# return an object/dict with ``.choices[0].message.content`` (LiteLLM
# shape — same as :mod:`topofuzz.runner._call_llm`).
SyncLLMCallable = Callable[..., Any]


# Default ``max_tokens`` for the judge response. The contract format is
# ~2 lines of text; 256 is generous. Lower than the runner's default
# because a verbose judge response is wasteful — we only parse the
# Score and one rationale line.
_DEFAULT_MAX_TOKENS = 256


class SingleJudge:
    """One LLM call per cell. Returns ``ContinuousScore`` in ``[0, 1]``.

    Implements the :class:`Judge` protocol (sync ``.score``). Step 7's
    blueprint position: "Single Claude call per (variant, eval).
    Cached via ``mbcore.cache.JudgeCache``. Rubric prompt drafted and
    committed." V0.5 defaults to ``google/gemini-2.5-pro`` rather than
    Claude per ``project_llm_api_access`` (only Gemini key confirmed).

    Args:
        model: LiteLLM model id (e.g. ``"google/gemini-2.5-pro"``).
        rubric_path: Path to the rubric template. Read as UTF-8 bytes at
            construction; :attr:`rubric_hash` is the SHA-256 over the
            JCS-canonical form of the raw text. The template uses
            Python ``str.format`` substitution with keys:
            ``variant_id``, ``applicant_blurb``, ``ground_truth_decision``,
            ``agent_output``. Missing keys in the payload raise
            ``KeyError`` from ``str.format`` at score time.
        dry_run: When ``True``, ``score`` skips the LLM entirely and
            returns ``ContinuousScore(0.5)`` + rationale ``"[DRY-RUN]"``.
        llm_callable: Optional sync callable for tests. Default is
            ``litellm.completion`` (lazy-imported so verify-only installs
            still load this module).
        max_tokens: Cap on judge-response length. Default 256.
    """

    def __init__(
        self,
        model: str,
        rubric_path: Path | str,
        *,
        dry_run: bool = False,
        llm_callable: SyncLLMCallable | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError(f"model must be a non-empty str; got {model!r}")
        path = Path(rubric_path)
        try:
            raw = path.read_bytes()
        except OSError as e:
            raise FileNotFoundError(
                f"Rubric template not found at {path!s}: {e}"
            ) from e
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Rubric template at {path!s} is not valid UTF-8: {e}"
            ) from e
        # Normalize line endings before hashing AND before substituting.
        # Without this, a Windows clone under `core.autocrlf=true` writes
        # the rubric to disk as CRLF; the hash of CRLF bytes differs from
        # the hash of LF bytes for the same logical template, splitting
        # the cache across platforms (audit B3). Also strip a UTF-8 BOM
        # if one snuck in via Notepad. `.gitattributes` pins LF for *.txt
        # as the first line of defense; this normalize is the second.
        if text.startswith("﻿"):
            text = text[1:]
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            raise ValueError(f"Rubric template at {path!s} is empty")

        self._model = model
        self._rubric_path = path
        self._rubric_text = text
        # JCS-canonical hash over the EOL-normalized text. Mirrors the
        # rest of the hash chain (Topology, LLMConfig, cache identity)
        # which all route through :func:`fuzzeval.canonical.canonical_hash`.
        # A whitespace-only edit to the rubric WILL change this hash —
        # exactly the cache-invalidation behavior we want.
        self._rubric_hash = canonical_hash(text)
        self._dry_run = dry_run
        self._llm_callable = llm_callable
        self._max_tokens = int(max_tokens)
        if self._max_tokens < 1:
            raise ValueError(
                f"max_tokens must be positive; got {max_tokens!r}"
            )

    # ------------------------------------------------------------------
    # Public read-only state
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def rubric_path(self) -> Path:
        return self._rubric_path

    @property
    def rubric_hash(self) -> str:
        return self._rubric_hash

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    # ------------------------------------------------------------------
    # Prompt rendering — also used by the orchestrator to compute the
    # per-cell JudgeCache prompt_hash slot (step 7 contract refinement;
    # replaces step-5's hash-the-whole-eval_row stopgap).
    # ------------------------------------------------------------------

    def render_prompt(self, payload: dict) -> str:
        """Render the rubric template against ``payload``.

        Required payload keys:
            variant_id              — str (typically a canonical_hash)
            eval_row                — dict; must contain ``input`` (the
                                       applicant blurb in lending fixtures)
                                       and ``ground_truth_decision``.
                                       The runner's standard loader
                                       (``topofuzz.loaders.load_eval_rows``)
                                       produces dicts with ``id`` + ``input``
                                       + any additional CSV columns.
                                       ``applicant_blurb`` is accepted as
                                       a §13-vocabulary alias for ``input``;
                                       when both are present the dedicated
                                       ``applicant_blurb`` value wins.
            agent_output            — str, the pipeline's final response

        Raises:
            KeyError: A required key is missing from payload or the
                referenced eval_row.
            TypeError: ``eval_row`` is not a dict; or a referenced value
                is not stringifiable.

        The rendered text is what the LLM sees AND what the JudgeCache
        ``prompt_hash`` slot hashes. Identical payload → identical
        prompt → identical cache key.
        """
        if not isinstance(payload, dict):
            raise TypeError(
                f"payload must be dict; got {type(payload).__name__}"
            )
        eval_row = payload.get("eval_row")
        if not isinstance(eval_row, dict):
            raise TypeError(
                "payload['eval_row'] must be a dict; got "
                f"{type(eval_row).__name__}"
            )
        # The applicant text lives under either the §13 vocabulary name
        # ``applicant_blurb`` (when a fixture explicitly carries it) or
        # under ``input`` (the loader's standard column name). Prefer the
        # explicit blurb when present; fall back to input. Treating these
        # as interchangeable here lets the rubric template stay
        # §13-faithful while CSVs follow the loader contract.
        applicant_text = eval_row.get("applicant_blurb")
        if applicant_text is None:
            applicant_text = eval_row.get("input")
        if applicant_text is None:
            raise KeyError(
                "Rubric payload eval_row missing 'applicant_blurb' (or 'input') "
                "— need at least one for the applicant text."
            )
        try:
            substitutions = {
                "variant_id": str(payload["variant_id"]),
                "applicant_blurb": str(applicant_text),
                "ground_truth_decision": str(eval_row["ground_truth_decision"]),
                "agent_output": str(payload["agent_output"]),
            }
        except KeyError as e:
            raise KeyError(
                f"Rubric payload missing required key: {e}. "
                "Required: variant_id, agent_output, "
                "eval_row['ground_truth_decision'] "
                "(plus 'applicant_blurb' OR 'input' on eval_row)."
            ) from e
        try:
            return self._rubric_text.format(**substitutions)
        except (KeyError, IndexError, ValueError) as e:
            # str.format failures from unexpected ``{...}`` tokens in the
            # rubric file. Surface clearly — the rubric author needs to
            # double-brace literal braces.
            raise ValueError(
                f"Failed to render rubric at {self._rubric_path!s}: {e}. "
                "Are there stray `{` / `}` in the template? "
                "Double them (`{{`, `}}`) to emit literal braces."
            ) from e

    # ------------------------------------------------------------------
    # score() — the Judge.score contract
    # ------------------------------------------------------------------

    def score(self, payload: dict) -> ContinuousScore:
        """Score one (variant, eval_row, agent_output) → ContinuousScore.

        See module docstring for failure modes and dry-run behavior.
        """
        if self._dry_run:
            # Render the prompt anyway so a typo'd template surfaces in
            # dry-run rather than waiting for the first real call. Cheap
            # ($0) and catches the most common rubric-author mistake.
            self.render_prompt(payload)
            return ContinuousScore(value=0.5)

        prompt = self.render_prompt(payload)
        messages = [{"role": "user", "content": prompt}]

        # Retry policy (second-pass audit MAJOR-1):
        #
        # * ``TypeError`` from :func:`_extract_judge_text` is a transient
        #   envelope-shape failure (content-filter empty-choices, provider
        #   timeout returning ``{"choices": []}``). Retry at the same
        #   config — those failures recover stochastically.
        #
        # * ``JudgeUnparseableError`` from :func:`_parse_score` is a
        #   *parser-side* failure on a well-formed response that doesn't
        #   match the rubric format. Under ``temperature=0`` the retry
        #   is byte-identical → same failure. Skip the retry to save the
        #   billed call and surface the failure immediately.
        #
        # The runner converts this into ``judge_failure=True`` per
        # :func:`topofuzz.runner._run_judge`; cells aren't lost.
        try:
            text = self._one_call(messages, temperature=0.0)
        except TypeError as e1:
            _log.warning(
                "SingleJudge envelope bad (%s: %s); retrying once at temp=0.",
                type(e1).__name__, e1,
            )
            try:
                text = self._one_call(messages, temperature=0.0)
            except TypeError as e2:
                raise JudgeUnparseableError(
                    f"SingleJudge envelope bad twice; first: "
                    f"{type(e1).__name__}: {e1}; retry: "
                    f"{type(e2).__name__}: {e2}"
                ) from e2
        raw, rationale = _parse_score(text)

        # 0-10 → [0, 1]. Integer division would zero everything but the
        # 10 case; explicit float division.
        value = raw / 10.0
        # Rationale is currently logged but not threaded into the
        # ContinuousScore (the §17a type only carries `value`). The
        # rationale is in the LLM response body and would surface via
        # debug logging if needed; we deliberately don't widen the §17a
        # type to keep Pramana's BinaryScore the only rationale carrier.
        if rationale:
            _log.debug(
                "SingleJudge rationale (raw=%d): %s",
                raw, rationale[:200],
            )
        return ContinuousScore(value=value)

    # ------------------------------------------------------------------
    # Internal: one LLM call returning the response text
    # ------------------------------------------------------------------

    def _one_call(self, messages: list[dict], *, temperature: float) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if self._llm_callable is not None:
            response = self._llm_callable(**kwargs)
        else:
            try:
                from litellm import completion
            except ImportError as e:
                raise ImportError(
                    "fuzzeval.judge.SingleJudge requires litellm for the "
                    "default llm_callable; install via "
                    "`pip install topofuzz[full]` or inject a custom "
                    "llm_callable into the SingleJudge constructor."
                ) from e
            response = completion(**kwargs)
        return _extract_judge_text(response)


def _extract_judge_text(response: Any) -> str:
    """Pull the response text from a LiteLLM-shaped sync response.

    Mirrors :func:`topofuzz.runner._extract_text_and_served` but returns
    only the text (the judge does not need the served-version slot;
    judges are cached by ``judge_model_version`` which is a caller
    contract). Raises TypeError on shape mismatch — those failures
    should not silently produce an empty string the parser then rejects
    as unparseable.
    """
    try:
        if isinstance(response, dict):
            text = response["choices"][0]["message"]["content"]
        else:
            text = response.choices[0].message.content
    except (KeyError, IndexError, AttributeError) as e:
        raise TypeError(
            f"Judge LLM response shape unexpected: "
            f"{type(response).__name__}; could not extract "
            f"choices[0].message.content: {e}"
        ) from e
    if not isinstance(text, str):
        raise TypeError(
            f"Judge LLM response content is {type(text).__name__}, "
            "expected str."
        )
    return text


# ---------------------------------------------------------------------------
# binary_from_continuous — metrics-layer helper for step 9 Cohen's κ
# ---------------------------------------------------------------------------


def binary_from_continuous(
    score: ContinuousScore,
    *,
    threshold: float = 0.5,
    pass_rationale: str = "score >= threshold",
    fail_rationale: str = "score < threshold",
) -> BinaryScore:
    """Derive a :class:`BinaryScore` from a :class:`ContinuousScore`.

    Step 9 metrics consume this for Cohen's κ vs the lending fixture's
    binary ``ground_truth_decision`` column. The threshold defaults to
    0.5; the v0.5 metrics layer hard-codes 0.5 (handoff decision A)
    and does not expose it as a CLI flag. Exposing the threshold here
    keeps the helper testable in isolation without touching the metrics
    module.

    Raises:
        TypeError: ``score`` is not a ContinuousScore.
        ValueError: ``threshold`` not in [0, 1].
    """
    if not isinstance(score, ContinuousScore):
        raise TypeError(
            f"score must be ContinuousScore; got {type(score).__name__}"
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be in [0, 1]; got {threshold!r}"
        )
    passed = score.value >= threshold
    return BinaryScore(
        passed=passed,
        rationale=pass_rationale if passed else fail_rationale,
    )


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "BinaryScore",
    "ContinuousScore",
    "Judge",
    "JudgeResult",
    "JudgeUnparseableError",
    "SingleJudge",
    "binary_from_continuous",
]
