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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Protocol

from fuzzeval.canonical import canonical_hash

_log = logging.getLogger(__name__)

# Per-LLM-call timeout (seconds). Env-overridable so an operator can
# tighten / loosen without code edits. 60s is the default — long enough
# to absorb a slow Gemini 2.5-pro response on a complex eval, short
# enough that a hung provider doesn't pin the whole run (no parent
# supervisor would reap a stuck cell otherwise). v0.5.1 audit-batch
# MAJOR fix.
#
# Hard ceiling at 3600s (1 hour) as defense-in-depth — TOPOFUZZ_
# LLM_TIMEOUT_SECS=99999999 would otherwise restore the DoS surface
# the cap was meant to close. The pass-2 red-team also caught that
# `float("inf")` and `float("nan")` slip past a naive `<= 0` guard
# (NaN comparisons return False), so the predicate is
# `math.isfinite(value) AND value > 0` AND `value <= ceiling`. Shared
# implementation lives in :mod:`fuzzeval._timeout` (DRY'd 2026-05-14;
# closes BACKLOG: DRY _resolve_llm_timeout_secs + truncate env-var
# value in warning log).
from fuzzeval._timeout import (
    LLM_TIMEOUT_MAX_SECS as _LLM_TIMEOUT_MAX_SECS,
    resolve_llm_timeout_secs as _shared_resolve,
)


def _resolve_llm_timeout_secs() -> float:
    return _shared_resolve(_log, prefix="judge")


_LLM_TIMEOUT_SECS = _resolve_llm_timeout_secs()


# ---------------------------------------------------------------------------
# §17a frozen types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousScore:
    """A real-valued judge score in [0, 1] with the judge's rationale.

    ``rationale`` carries the ``Rationale:`` line the judge LLM emitted
    (empty string when the judge gave none, or for the dry-run / pre-
    rationale-field cache entries). Defaulted so old pickled cache
    entries round-trip via the class-attribute fallback.
    """

    value: float
    rationale: str = ""


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
    """Judge protocol. Implementations: :class:`SingleJudge`, :class:`JuryJudge`.

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
# Rationale LABEL. ``[\s*#>\-]*`` allows the same markdown leaders as
# Score; ``\**`` tolerates ``**Rationale:**`` bold-wrapping. This only
# matches the *label* — _parse_score takes the body as everything from
# the label to the first blank line. A single-line ``$``-anchored
# capture was insufficient: real judge LLMs (Gemini Flash especially)
# wrap the "one short sentence" the rubric asks for across physical
# lines, and the old regex truncated those to the first fragment
# (often 1-3 words like "The" / "The decision").
_RATIONALE_LABEL_RE = re.compile(
    r"(?im)^[\s*#>\-]*\**\s*Rationale:\**[ \t]*"
)

# Paragraph break — a blank (whitespace-only) line. The rationale body
# is captured only up to here so a judge's trailing sign-off / "Note:" /
# "---" postscript after the rationale is not swallowed into it. ``\r``
# is in the inner class so a CRLF response (``\r\n\r\n``) still counts
# as a break.
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t\r]*\n")

# Persisted-rationale length cap. The rubric contract is a single short
# rationale sentence; this bounds a pathological / adversarial LLM
# response from inflating ``run.sqlite`` (which is SHA-256'd into the
# signed attestation) without limit. Generous — a real rationale is
# well under this. _parse_score also collapses internal whitespace
# (incl. newlines) so the persisted value is single-line.
_RATIONALE_MAX_CHARS: Final[int] = 1000


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
    When present, it is the text from the first ``Rationale:`` label
    *after the authoritative score* through the next blank line,
    whitespace-collapsed to a single line and length-capped (see
    :data:`_RATIONALE_MAX_CHARS`).
    """
    if not isinstance(response_text, str):
        raise JudgeUnparseableError(
            f"Judge response must be str; got {type(response_text).__name__}"
        )
    score_matches = list(_SCORE_RE.finditer(response_text))
    if not score_matches:
        raise JudgeUnparseableError(
            "Judge response missing 'Score: <int>' line. Response was:\n"
            f"{response_text[:500]}"
        )
    last_score = score_matches[-1]
    raw = int(last_score.group(1))
    # Range check kept as defense in depth even though the regex
    # alternation pins N to 0-10 by construction. A future regex edit
    # that broadens the integer class would otherwise silently accept
    # out-of-range values.
    if not (0 <= raw <= 10):
        raise JudgeUnparseableError(
            f"Judge score {raw} out of range [0, 10]. Response was:\n"
            f"{response_text[:500]}"
        )
    # Rationale belongs to the SAME answer block as the authoritative
    # score. The rubric format is "Score:" then "Rationale:", so we take
    # the FIRST "Rationale:" label at or after the LAST "Score:" match.
    # Anchoring to the score (rather than "last Rationale: anywhere")
    # keeps the two extractors agreeing on one block: a self-correction
    # like a trailing "...Score: 8" with no following Rationale yields an
    # empty rationale, not block-1's rationale mis-paired with block-2's
    # score. The body runs from the label to the first blank line — so a
    # trailing sign-off / "Note:" / "---" postscript isn't swallowed —
    # with internal whitespace (incl. the newlines Gemini wraps the
    # rationale across) collapsed to single spaces and a trailing
    # markdown ``**`` stripped.
    rationale = ""
    label_match = _RATIONALE_LABEL_RE.search(response_text, last_score.end())
    if label_match:
        tail = response_text[label_match.end():]
        tail = _PARAGRAPH_BREAK_RE.split(tail, maxsplit=1)[0]
        rationale = " ".join(tail.split()).rstrip("*").strip()
    # Bound the persisted rationale (see _RATIONALE_MAX_CHARS): it lands
    # in run.sqlite, which is digested into the signed attestation.
    return raw, rationale[:_RATIONALE_MAX_CHARS]


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


# Default ``max_tokens`` for the judge response. Gemini 2.5 Flash is a
# *reasoning* model — it spends tokens on internal "thinking" that count
# against this budget before any visible Score:/Rationale: text is
# emitted. The original 256 truncated it mid-response: the ``Score:``
# line sometimes never appeared (→ JudgeUnparseableError →
# judge_failure → §8 degraded), and when it did the ``Rationale:`` was
# cut to 1-3 words ("The", "The decision"). 1500 covered the simpler
# lending fixture but still starved the supervisor fixture's longer
# prompts (6/32 cells failed). 4000 leaves headroom for the thinking
# budget plus the full response on both gallery fixtures — empirically
# finish_reason='stop'. The cap costs nothing when unused (a reasoning
# model stops when done); only actually-emitted tokens are billed.
_DEFAULT_MAX_TOKENS = 4000


class SingleJudge:
    """One LLM call per cell. Returns ``ContinuousScore`` in ``[0, 1]``.

    Implements the :class:`Judge` protocol (sync ``.score``). Step 7's
    blueprint position: "Single Claude call per (variant, eval). Cached
    via ``fuzzeval.cache.JudgeCache``. Rubric prompt drafted and
    committed." V0.5 defaults to ``google/gemini-2.5-pro`` rather than
    Claude per ``project_llm_api_access`` (only Gemini key confirmed).
    (BLUEPRINT_v5plus.md predates the rename and still spells
    ``mbcore.cache.JudgeCache``; the real module is ``fuzzeval.cache``
    per [[project_topofuzz_names]].)

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
        max_tokens: Cap on judge-response length. Default 4000 (see
            :data:`_DEFAULT_MAX_TOKENS` — must cover a reasoning model's
            thinking budget, not just the two contract lines).
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
            return ContinuousScore(value=0.5, rationale="[DRY-RUN]")

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
        if rationale:
            _log.debug(
                "SingleJudge rationale (raw=%d): %s",
                raw, rationale[:200],
            )
        return ContinuousScore(value=value, rationale=rationale)

    # ------------------------------------------------------------------
    # Internal: one LLM call returning the response text
    # ------------------------------------------------------------------

    def _one_call(self, messages: list[dict], *, temperature: float) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
            # v0.5.1 audit MAJOR fix: bound the per-call wall-clock.
            # LiteLLM's default is 600s; without a cap a single slow
            # provider can pin every cell in the run. Operators can
            # override via TOPOFUZZ_LLM_TIMEOUT_SECS env (see module).
            "timeout": _LLM_TIMEOUT_SECS,
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
            # Translate the project-canonical model id (e.g.
            # ``"google/gemini-2.5-flash"``) to LiteLLM's provider-
            # prefixed form (``"gemini/gemini-2.5-flash"``) JUST at
            # the call boundary. ``self._model`` is the canonical name
            # for cache / attestation / §8 vendor-overlap purposes;
            # only the outbound LiteLLM call sees the translation.
            from fuzzeval._litellm_alias import litellm_model_id
            kwargs["model"] = litellm_model_id(kwargs["model"])
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
# JuryJudge — Weekend 4 step 12b
# ---------------------------------------------------------------------------


# BLUEPRINT_v5plus.md §6 line 220 names the opt-in jury trio. Vendor
# names are the LiteLLM provider prefixes (matching what ``model``
# starts with) so they're independent of the *specific* model id chosen
# for that vendor — a future config update from "claude-opus-4-7" to
# "claude-opus-5" preserves the "anthropic" vendor name in the
# accumulator. Stored as a ``dict[str, str]`` because both the names
# AND the model ids are needed at construction (see
# :class:`JuryJudge.__init__`).
_DEFAULT_JURY_VENDOR_MODELS: dict[str, str] = {
    "anthropic": "anthropic/claude-opus-4-7",
    "openai": "openai/gpt-5",
    "gemini": "google/gemini-2.5-pro",
}


# Number of judges in the jury. BLUEPRINT §6 line 220 is explicit: "Three
# judges". Promoted to a named constant so a future v0.6 "jury-of-N"
# experiment can lift this without searching for magic threes.
_JURY_SIZE = 3


# Sentinel-typed mapping for the optional per-vendor test injection. Each
# value is the same ``SyncLLMCallable`` shape as ``SingleJudge``'s
# ``llm_callable`` arg — supplied per-vendor so a test can simulate a
# parseable Anthropic response and an unparseable Gemini response in the
# same JuryJudge.
LLMCallables = dict[str, SyncLLMCallable]


class JuryJudge:
    """Three-vendor jury wrapper around :class:`SingleJudge`.

    Fans one ``score(payload)`` call out to three single-vendor judges
    (default: Anthropic + OpenAI + Gemini per BLUEPRINT_v5plus.md §6
    line 220), aggregates parseable continuous scores by **arithmetic
    mean**, and exposes run-level accounting for the §12b DoD #14
    falsifiability targets.

    Aggregation contract (handoff decision locked at step-12b kickoff):

    * **Mean of parseable scores** in ``[0, 1]``. Each vendor's
      :class:`SingleJudge` returns ``ContinuousScore``; we collect
      ``.value`` from the calls that did not raise
      :class:`JudgeUnparseableError`, then mean.
    * **UNPARSEABLE fallback.** If 1 or 2 vendors raise
      ``JudgeUnparseableError``, drop their votes and mean the
      survivors. If **all** vendors raise, re-raise
      ``JudgeUnparseableError`` so the runner (:mod:`topofuzz.runner`)
      marks the cell ``judge_failure=True`` (§8 ``judge_failure_rate >
      5% → degraded`` accounting).

    Accounting (DoD #14, BLUEPRINT line 445 — INFORMATIONAL only at
    v0.5, promoted to HARD at TDS-Regulator):

    * :attr:`judge_failure_rate` — cumulative ratio
      ``failed_vendor_calls / total_vendor_calls`` across **all**
      ``score()`` calls. So 1-of-3 failures on each of 10 calls reports
      ``10/30 = 0.333…`` (not 1.0).
    * :attr:`judge_independence` — ``{"vendor_overlap": [...]}`` where
      the list is the sorted, deduplicated set of vendor names that
      produced at least one parseable vote across the run. Per the
      locked field-shape decision this is *minimal* — the list-of-pairs
      variant is deferred to a future v0.5.x schema bump if needed.

    Sequencing: the three vendor calls happen **sequentially** on the
    calling thread. The :class:`Judge` protocol is sync (the runner
    offloads to ``asyncio.to_thread``), so parallelizing within
    JuryJudge would force a sync↔async boundary right where the cache
    layer is. Step 12b ships the sequential version per
    [[feedback_minimalist_scope]]; async fan-out is a v0.5.1
    optimization.

    Cache compatibility: the v0.5 :class:`JudgeCache` keys on a single
    ``judge_model_version`` string, which doesn't model "this row was
    produced by a jury". The CLI side disables the JudgeCache when
    ``--jury-of-3`` is set (see ``topofuzz.cli.run``) — jury runs are
    rare and expensive enough that uncached is acceptable. Per-vendor
    cache slicing is a v0.5.1 follow-up.

    Args:
        rubric_path: Path to the rubric template. Same arg as
            :class:`SingleJudge`; all three vendor judges read the same
            file at construction so :attr:`rubric_hash` is a single
            scalar (and we double-check the per-vendor hashes match —
            see ``_rubric_hash_pin``).
        vendor_models: Optional ``{vendor_name: litellm_model_id}``
            mapping. Default is the BLUEPRINT trio. Pass-1 RT-M2 audit
            fix: vendor names are stripped and lowercased before the
            uniqueness check, so ``"  Anthropic "`` and ``"anthropic"``
            collide and raise. Model VALUES must also be distinct
            (three vendors pointing at the same LiteLLM model id
            defeats the independence claim of
            :attr:`judge_independence`).
        dry_run: When ``True``, every vendor judge is dry-run
            (deterministic ``ContinuousScore(0.5)`` per
            :class:`SingleJudge.score`). The jury mean is therefore
            also 0.5 and no real LLM calls are billed.
        llm_callables: Optional per-vendor sync LLM callable for tests.
            Same shape as :class:`SingleJudge.llm_callable` but keyed
            by vendor name (same normalization rules as
            ``vendor_models``). Unsupplied vendors fall back to
            :mod:`litellm`.
        max_tokens: Cap passed through to each vendor's
            :class:`SingleJudge`. Default 4000 (see
            :data:`_DEFAULT_MAX_TOKENS`).

    Raises:
        ValueError: ``vendor_models`` has the wrong count
            (``!= _JURY_SIZE``); a vendor name is non-``str`` or empty
            after strip; two names collide after strip+lowercase
            normalization; two vendors point at the same model id; an
            ``llm_callables`` key is not present in ``vendor_models``;
            or any vendor's :class:`SingleJudge` rejects its model /
            max_tokens.
        FileNotFoundError / ValueError: forwarded from the per-vendor
            :class:`SingleJudge` rubric load (bad path / empty / non-UTF8).
    """

    def __init__(
        self,
        rubric_path: Path | str,
        *,
        vendor_models: dict[str, str] | None = None,
        dry_run: bool = False,
        llm_callables: LLMCallables | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        raw_models = (
            dict(_DEFAULT_JURY_VENDOR_MODELS)
            if vendor_models is None
            else dict(vendor_models)
        )
        # Pass-1 RT-M2 audit fix: normalize vendor keys (strip whitespace,
        # lowercase) before the uniqueness check so "Anthropic" and
        # "anthropic " can't both appear in vendor_overlap as case-variant
        # duplicates. The BLUEPRINT line 445 falsifiability claim — "vendor_overlap
        # includes 'anthropic' if Claude is in the jury" — depends on the
        # lowercase convention; normalize at the gate, not at the read site.
        models: dict[str, str] = {}
        for raw_vendor, model in raw_models.items():
            if not isinstance(raw_vendor, str):
                raise ValueError(
                    f"JuryJudge vendor names must be str; got {type(raw_vendor).__name__}"
                )
            vendor = raw_vendor.strip().lower()
            if not vendor:
                raise ValueError(
                    f"JuryJudge vendor name must be non-empty after strip; got {raw_vendor!r}"
                )
            if vendor in models:
                # Two raw keys normalized to the same canonical name
                # (e.g., "anthropic" + "Anthropic ") — fail loud rather
                # than silently collapse.
                raise ValueError(
                    f"JuryJudge vendor names collide after normalization: "
                    f"{raw_vendor!r} → {vendor!r} already present."
                )
            models[vendor] = model
        if len(models) != _JURY_SIZE:
            raise ValueError(
                f"JuryJudge requires exactly {_JURY_SIZE} vendor entries; "
                f"got {len(models)}: {sorted(models)!r}"
            )

        # Pass-1 RT-M2 audit fix: reject duplicate model VALUES. Three
        # vendors all pointing at the same LiteLLM model id (e.g., all
        # three set to "google/gemini-2.5-pro") would produce a "jury"
        # whose three votes are statistically dependent — the
        # judge_independence.vendor_overlap signal becomes a lie.
        if len(set(models.values())) != _JURY_SIZE:
            raise ValueError(
                f"JuryJudge vendor models must be distinct; got duplicates in "
                f"{models!r}. Three vendors pointing at the same model defeats "
                "the independence claim of vendor_overlap."
            )

        # Normalize llm_callables keys the same way so caller-side
        # casing inconsistencies surface here, not at score() time when
        # the callable would silently never be invoked.
        raw_callables = dict(llm_callables) if llm_callables else {}
        callables: dict[str, SyncLLMCallable] = {}
        for raw_vendor, cb in raw_callables.items():
            if not isinstance(raw_vendor, str):
                raise ValueError(
                    "llm_callables keys must be str; got "
                    f"{type(raw_vendor).__name__}"
                )
            vendor = raw_vendor.strip().lower()
            if vendor not in models:
                raise ValueError(
                    f"llm_callables key {raw_vendor!r} not in vendor_models; "
                    f"valid keys: {sorted(models)!r}"
                )
            callables[vendor] = cb

        # Per-vendor SingleJudge. Stable iteration order: the dict's
        # insertion order (Python 3.7+ guarantee). Reports rely on this
        # for reproducible ``vendor_overlap`` listings.
        self._judges: dict[str, SingleJudge] = {}
        for vendor, model in models.items():
            self._judges[vendor] = SingleJudge(
                model=model,
                rubric_path=rubric_path,
                dry_run=dry_run,
                llm_callable=callables.get(vendor),
                max_tokens=max_tokens,
            )

        # Pin the rubric_hash from the first vendor's SingleJudge and
        # *assert* every other vendor produced the same hash. They share
        # ``rubric_path``, so the only way they could differ is if the
        # SingleJudge constructor's EOL/BOM normalization stopped being
        # deterministic — a defensive check that would catch a future
        # regression in :class:`SingleJudge` rather than silently
        # accepting a split cache key.
        first = next(iter(self._judges.values()))
        self._rubric_hash = first.rubric_hash
        self._rubric_path = first.rubric_path
        self._dry_run = dry_run
        for vendor, judge in self._judges.items():
            if judge.rubric_hash != self._rubric_hash:
                raise RuntimeError(
                    f"JuryJudge rubric_hash mismatch: vendor {vendor!r} "
                    f"produced {judge.rubric_hash!r}, "
                    f"expected {self._rubric_hash!r}. This is a SingleJudge "
                    "regression — file a bug."
                )

        # Run-level accounting accumulators. Mutated only inside
        # :meth:`score`; readers should treat them as snapshots — the
        # values reflect calls completed up to the most recent
        # ``score()`` return.
        #
        # Pass-1 CR-M2 / RT-M4 audit fix: the runner offloads
        # :meth:`score` to ``asyncio.to_thread`` and the orchestrator
        # runs cells concurrently under a semaphore. Two cells may
        # therefore execute :meth:`score` on different worker threads
        # at the same time, racing the counters. A ``threading.Lock``
        # makes the per-call accounting update atomic — at the cost of
        # serializing the *bookkeeping*, not the vendor calls themselves
        # (those release the GIL while waiting on litellm). The lock is
        # held for microseconds at a time.
        self._total_vendor_calls: int = 0
        self._failed_vendor_calls: int = 0
        self._parseable_vendors: set[str] = set()
        self._accounting_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public read-only state — mirrors :class:`SingleJudge`'s surface so
    # the runner / CLI / cache layer can treat the two interchangeably
    # except where the cache key is concerned (see class docstring).
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Synthetic combined model id for manifest / cli_args display.

        Format: ``"jury_of_3:<vendor1>+<vendor2>+<vendor3>"`` with
        vendors sorted alphabetically for stability. NOT a real LiteLLM
        id — never pass this string to ``litellm.completion``.
        """
        return "jury_of_3:" + "+".join(sorted(self._judges))

    @property
    def rubric_path(self) -> Path:
        return self._rubric_path

    @property
    def rubric_hash(self) -> str:
        return self._rubric_hash

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def vendors(self) -> tuple[str, ...]:
        """Tuple of vendor names in construction order."""
        return tuple(self._judges)

    # ------------------------------------------------------------------
    # Prompt rendering — delegates to the first vendor. All three vendor
    # SingleJudges share the same rubric file and render arguments, so
    # the rendered prompt is byte-identical across them; the runner's
    # JudgeCache prompt-hash slot is therefore well-defined even though
    # the cache itself is disabled for jury runs.
    # ------------------------------------------------------------------

    def render_prompt(self, payload: dict) -> str:
        first = next(iter(self._judges.values()))
        return first.render_prompt(payload)

    # ------------------------------------------------------------------
    # score() — the Judge.score contract
    # ------------------------------------------------------------------

    def score(self, payload: dict) -> ContinuousScore:
        """Fan ``payload`` out to all three vendor judges; mean parseable.

        See class docstring for aggregation + UNPARSEABLE fallback +
        accounting semantics.
        """
        # Buffer the per-call decisions then apply them under the
        # accounting lock at the end so the accumulators advance
        # atomically as one batch (CR-M2 / RT-M4 audit fix). Without
        # batching, two threads interleaving their per-vendor
        # increments could produce a transient state like
        # ``failed_vendor_calls > total_vendor_calls`` if a reader
        # snapshots between the two ``+=`` statements.
        votes: list[float] = []
        vote_rationales: list[tuple[str, str]] = []
        parseable_this_call: list[str] = []
        per_call_errors: list[tuple[str, str]] = []
        local_total = 0
        local_failed = 0
        for vendor, judge in self._judges.items():
            local_total += 1
            try:
                result = judge.score(payload)
            except JudgeUnparseableError as e:
                local_failed += 1
                per_call_errors.append((vendor, f"{type(e).__name__}: {e}"))
                _log.warning(
                    "JuryJudge vendor %s unparseable: %s", vendor, e,
                )
                continue
            except (KeyboardInterrupt, SystemExit):
                # RT pass-1 implicit: don't swallow user-initiated
                # cancellation as a vendor failure. Mirrors
                # :class:`SingleJudge.score`'s behavior and the runner's
                # cancellation handling. Note: the partial state
                # accumulated so far stays in the local buffers and is
                # NOT applied — the JuryJudge accumulators do not see
                # this aborted call.
                raise
            except Exception as e:  # noqa: BLE001 — boundary: vendor SDK
                # Network / SDK / quota failures get the same treatment as
                # UNPARSEABLE for accounting: drop the vote, record the
                # vendor as failed, keep going. Without this, an Anthropic
                # 429 would crash the whole jury run mid-cell; the
                # BLUEPRINT's "judge_failure_rate" contract is the right
                # surface for surfacing these and §A's degraded badge is
                # the right escalation.
                local_failed += 1
                per_call_errors.append((vendor, f"{type(e).__name__}: {e}"))
                _log.warning(
                    "JuryJudge vendor %s raised %s: %s",
                    vendor, type(e).__name__, e,
                )
                continue
            # Defensive: SingleJudge always returns ContinuousScore in
            # its current implementation, but the Judge Protocol allows
            # BinaryScore. The mean operation isn't defined on a mix, so
            # bail out clearly rather than silently dropping the value
            # and producing a misleading mean.
            if not isinstance(result, ContinuousScore):
                local_failed += 1
                per_call_errors.append(
                    (
                        vendor,
                        f"non-ContinuousScore {type(result).__name__}",
                    )
                )
                _log.warning(
                    "JuryJudge vendor %s returned %s; expected ContinuousScore. "
                    "Dropping vote.",
                    vendor, type(result).__name__,
                )
                continue
            votes.append(result.value)
            vote_rationales.append((vendor, result.rationale))
            parseable_this_call.append(vendor)

        # Apply the per-call accumulator updates atomically.
        with self._accounting_lock:
            self._total_vendor_calls += local_total
            self._failed_vendor_calls += local_failed
            self._parseable_vendors.update(parseable_this_call)

        if not votes:
            raise JudgeUnparseableError(
                f"JuryJudge: all {len(self._judges)} vendors failed. "
                f"Errors: {per_call_errors!r}"
            )
        mean = sum(votes) / len(votes)
        # Mean of values in [0, 1] is itself in [0, 1] — no clamp needed.
        # Rationale aggregates only the parseable vendors that actually
        # emitted a Rationale: line, vendor-prefixed and joined. Vendors
        # with an empty rationale contribute nothing — so an all-empty
        # jury yields "" (NULL once persisted), not a string of bare
        # "vendor: " delimiters. Each fragment is already single-line and
        # per-vendor capped by _parse_score; the JOINED result is capped
        # again here — N fragments × the per-vendor cap would otherwise
        # exceed _RATIONALE_MAX_CHARS and inflate the signed run.sqlite.
        jury_rationale = " | ".join(
            f"{vendor}: {text}" for vendor, text in vote_rationales if text
        )[:_RATIONALE_MAX_CHARS]
        return ContinuousScore(value=mean, rationale=jury_rationale)

    # ------------------------------------------------------------------
    # Run-level accounting — DoD #14 falsifiability surface.
    # ------------------------------------------------------------------

    @property
    def total_vendor_calls(self) -> int:
        """Cumulative count of vendor calls (parseable + failed)."""
        with self._accounting_lock:
            return self._total_vendor_calls

    @property
    def failed_vendor_calls(self) -> int:
        """Cumulative count of vendor calls that did not produce a vote.

        Includes both :class:`JudgeUnparseableError` and other vendor SDK
        exceptions (network / quota / non-ContinuousScore return).
        """
        with self._accounting_lock:
            return self._failed_vendor_calls

    @property
    def judge_failure_rate(self) -> float:
        """Fraction of vendor calls (across the run) that did not parse.

        Returns 0.0 before any ``score()`` call so a fresh JuryJudge in
        an aborted run reports a finite, well-defined value rather than
        NaN. Reads the two counters under :attr:`_accounting_lock` so the
        snapshot is internally consistent — without the lock, a reader
        interleaved between a writer's two ``+=`` increments could see a
        ratio > 1.0.
        """
        with self._accounting_lock:
            total = self._total_vendor_calls
            failed = self._failed_vendor_calls
        if total == 0:
            return 0.0
        return failed / total

    @property
    def judge_independence(self) -> dict[str, list[str]]:
        """Minimal vendor-overlap record per the locked field shape.

        ``vendor_overlap`` is the sorted, deduplicated list of vendor
        names that produced at least one parseable vote across the run.
        BLUEPRINT line 445 makes this falsifiable: "includes
        'anthropic' if Claude is in the jury" — i.e., if the Anthropic
        single-judge parsed at least once, "anthropic" must appear.
        """
        with self._accounting_lock:
            snapshot = sorted(self._parseable_vendors)
        return {"vendor_overlap": snapshot}


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
    "JuryJudge",
    "SingleJudge",
    "binary_from_continuous",
]
