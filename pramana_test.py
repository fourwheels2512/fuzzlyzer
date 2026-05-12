"""
pramana_test.py

Tests whether a frontier model (Claude, Gemini, or GPT) can produce the
Pramana paper's 6-phase reasoning schema zero-shot at quality matching or
exceeding the paper's fine-tuned 8B model (no training, just a system prompt).

Paper: Sharath Sathish, "Pramana: Fine-Tuning LLMs for Epistemic Reasoning
through Navya-Nyaya" - arXiv:2604.04937, Feb 14 2026.

Paper's reported results (n=10 test problems):
  Stage 0 (Llama-3.2-3B fine-tuned):    40% format / 50%  semantic
  Stage 1 (DeepSeek-R1-Distill 8B FT):  40% format / 100% semantic

Decision rule for our zero-shot frontier run:
  >= 80% format adherence -> productize as TraceGate semantic-eval extension
  40-80%                  -> iterate prompt, retest
  < 40%                   -> fine-tune required (use modelbrew CRMA pipeline)

Test set: 3 problems verified verbatim from the paper, 7 constructed in the
same paper-described problem types (constraint satisfaction, boolean SAT,
multi-step deduction). Each is tagged with `source` so paper-verified vs
constructed can be broken out in the analysis.

Provider auto-detect order (first match wins):
  - ANTHROPIC_API_KEY set  -> Claude (default model: claude-sonnet-4-6)
  - GEMINI_API_KEY  set    -> Gemini (default model: gemini-2.5-pro)
  - GOOGLE_API_KEY  set    -> Gemini (alias for GEMINI_API_KEY)
  - OPENAI_API_KEY  set    -> OpenAI (default model: gpt-5)

Override with --provider / --model flags or the PRAMANA_MODEL env var.

Install:
  pip install anthropic google-genai openai     # any subset you'll use

Run:
  # bash:       export GEMINI_API_KEY=...   (or ANTHROPIC_API_KEY / OPENAI_API_KEY)
  # PowerShell: $env:GEMINI_API_KEY = "..." (or ANTHROPIC_API_KEY / OPENAI_API_KEY)
  # cmd.exe:    set GEMINI_API_KEY=...      (or ANTHROPIC_API_KEY / OPENAI_API_KEY)
  python pramana_test.py

CLI flags:
  --provider {anthropic,gemini,openai}   Force provider (skips env detection)
  --model MODEL                          Override model (overrides PRAMANA_MODEL)
  --output PATH                          JSON results path
                                         (default: pramana_test_results.json)
  --limit N                              Run only the first N problems
  --problems PATH                        Load problems from a JSON file
                                         (advanced; see Problem schema below)

Output:
  Console: per-problem format/semantic pass + summary table + decision
  JSON: full traces and reports for later inspection
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants for retries, timeouts, and secret sanitization
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT_SECONDS = 120.0
RETRY_TRIES = 3
RETRY_BASE_DELAY_SECONDS = 2.0
SCHEMA_VERSION = "1.0"

# Common API key shapes -- strip from any error string before storing or printing.
# Matches Anthropic/OpenAI-style "sk-..." / "sk_test_..." / "sk_live_..." / "pk-..."
# tokens and Google API keys ("AIza" + 35 url-safe chars).
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk_(?:live|test)_[A-Za-z0-9_\-]{8,}"),
    re.compile(r"pk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{35}"),
]


def _sanitize(s: str) -> str:
    """Redact anything that looks like an API key from a string."""
    if not s:
        return s
    out = s
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion. Returns (low, high) in [0,1]."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return (low, high)


def _format_rate_with_ci(successes: int, n: int) -> str:
    """Format a rate with Wilson 95% CI like '100% [69%, 100%]'."""
    if n == 0:
        return "n/a"
    rate = successes / n
    low, high = _wilson_interval(successes, n)
    return f"{rate:.0%} [{low:.0%}, {high:.0%}]"


def _retry_call(fn: Any) -> Any:
    """Run `fn()` with exponential backoff for transient errors.

    Retries up to RETRY_TRIES times on network / rate-limit / 5xx errors.
    Base delay 2s, doubling each attempt. Non-transient errors raise immediately.
    """
    last_exc: BaseException | None = None
    for attempt in range(RETRY_TRIES):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK-specific errors vary
            if not _is_transient_error(exc):
                raise
            last_exc = exc
            if attempt == RETRY_TRIES - 1:
                break
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _is_transient_error(exc: BaseException) -> bool:
    """Heuristic for transient errors safe to retry across SDKs.

    SDKs differ in exception types, so we match on class names and on common
    HTTP-status / message substrings rather than importing each SDK's
    exception hierarchy.
    """
    name = type(exc).__name__.lower()
    transient_names = (
        "timeout", "connection", "ratelimit", "rate_limit",
        "apiconnection", "apitimeout", "serviceunavailable",
        "internalserver", "overloaded", "retry",
    )
    if any(n in name for n in transient_names):
        return True
    msg = str(exc).lower()
    if any(s in msg for s in ("timeout", "timed out", "rate limit", "429",
                              "500 ", "502 ", "503 ", "504 ",
                              "temporarily unavailable", "overloaded")):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    return False


def _sdk_version(module_name: str) -> str | None:
    """Return the installed version of an SDK package, or None if not installed."""
    try:
        return importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# System prompt - encodes the paper's 6-phase Navya-Nyaya schema verbatim
# in shape. Plain English where the paper uses Sanskrit, with the Sanskrit
# names in headers since the paper's validator checks for them.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a reasoning system that answers questions using the Navya-Nyaya
six-phase epistemic methodology. You MUST emit your answer as Markdown with
YAML frontmatter, in the exact structure below. No prose outside this format.

YAML frontmatter (required fields, in order):
---
id: <given problem id>
problem_type: constraint_satisfaction | boolean_sat | multi_step_deduction
difficulty: simple | moderate | complex
variables: <integer count of distinct variables in the problem>
ground_truth: "<your final answer here>"
---

Then exactly six Markdown sections, in this order, each starting with "## ":

## Samshaya (Doubt Analysis)
Classify the doubt into one of: Samana Dharma Upapatti, Aneka Dharma Upapatti,
Vipratipatti, Upalabdhi Avyavastha, Anupalabdhi Avyavastha. Give the chosen
doubt type and a one-sentence justification.

## Pramana (Evidence Sources)
Use exactly these four sub-bullets. Fill each with the relevant evidence from
the problem; if a category does not apply, write "(none in this problem)".
- Pratyaksha: <directly stated facts or observations>
- Anumana: <inferences derived from the stated facts>
- Upamana: <analogies to known cases, if any>
- Shabda: <authoritative principles or axioms invoked>

## Pancha Avayava (Five-Member Syllogism)
Use exactly these five sub-bullets:
- Pratijna: <thesis / claim>
- Hetu: <reason / evidence supporting the claim>
- Udaharana: Wherever <X>, there is <Y>. Example: <concrete instance>
- Upanaya: <application of the rule to this specific case>
- Nigamana: <conclusion drawn>

The Udaharana line MUST contain the literal phrase "Wherever" and "there is".

## Tarka (Counterfactual Testing)
Suppose the conclusion is false. Derive the resulting contradiction in one or
two sentences. This must be a real reductio - if no contradiction follows, say
so explicitly.

## Hetvabhasa (Fallacy Detection)
Use exactly these five sub-bullets. For each, write "absent" or describe how it
applies in this problem:
- Savyabhichara: <inconclusive reason>
- Viruddha: <contradictory reason>
- Prakaranasama: <unestablished reason>
- Sadhyasama: <begging the question>
- Kalaatita: <mistimed reason>

## Nirnaya (Definitive Conclusion)
Answer: <the final answer to the question>
Confidence: <0.0 to 1.0>
Epistemic Status: Definitive Knowledge | Hypothesis Requiring Verification

Do not include any text after the Nirnaya section. Do not skip sections. Do
not add sections."""


# ---------------------------------------------------------------------------
# Test set - 10 problems
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """A single test problem.

    `ground_truth_keywords` is structured as AND-of-OR groups:
      outer list = AND (every group must match)
      inner list = OR  (any one synonym in the group satisfies that group)

    Example: [["yes"], ["10", "ten", "10 students"]] means the answer must
    contain "yes" AND must contain at least one of "10", "ten", or "10 students".
    Number normalization (digit <-> word for 0-100) is applied on top, so a
    group containing "10" will also match the word "ten" automatically.
    """

    id: str
    question: str
    ground_truth_keywords: list[list[str]]
    source: str  # "paper" or "constructed"


TEST_PROBLEMS: list[Problem] = [
    # Verified verbatim from the paper:
    Problem(
        id="test-001",
        question=(
            "Alice, Bob, and Carol each have one pet: a cat, a dog, or a fish. "
            "Constraints: (1) Alice does not have the dog. (2) Bob has the cat. "
            "(3) Carol does not have the fish. "
            "Question: Who has which pet?"
        ),
        ground_truth_keywords=[
            ["alice"], ["alice has the fish", "alice: fish", "alice has fish"],
            ["bob"], ["bob has the cat", "bob: cat", "bob has cat"],
            ["carol"], ["carol has the dog", "carol: dog", "carol has dog"],
        ],
        source="paper",
    ),
    Problem(
        id="test-002",
        question=(
            "Given: (1) If it rains, the match is canceled. (2) If the match is "
            "canceled, the stadium is empty. (3) If it rains, the ground is wet. "
            "(4) It rains. "
            "Question: What is the state of the ground, the match, and the stadium?"
        ),
        ground_truth_keywords=[["wet"], ["canceled", "cancelled"], ["empty"]],
        source="paper",
    ),
    Problem(
        id="test-003",
        question=(
            "Alex, Ben, Cara, and Dana sit in seats 1, 2, 3, and 4 (left to right). "
            "Constraints: (1) Dana sits to the left of Ben. (2) Cara sits between "
            "Ben and Alex. (3) Alex sits in seat 4. (4) Ben does not sit in seat 1. "
            "Question: Who sits in which seat?"
        ),
        ground_truth_keywords=[
            ["dana: 1", "dana - 1", "dana in seat 1", "dana sits in seat 1",
             "seat 1: dana", "1: dana"],
            ["ben: 2", "ben - 2", "ben in seat 2", "ben sits in seat 2",
             "seat 2: ben", "2: ben"],
            ["cara: 3", "cara - 3", "cara in seat 3", "cara sits in seat 3",
             "seat 3: cara", "3: cara"],
            ["alex: 4", "alex - 4", "alex in seat 4", "alex sits in seat 4",
             "seat 4: alex", "4: alex"],
        ],
        source="paper",
    ),
    # Constructed in the same paper-described problem types:
    Problem(
        id="test-004",
        question=(
            "Three boxes are labeled A, B, C. One contains gold, one silver, one "
            "bronze. Constraints: (1) Box A does not contain silver. (2) Box C "
            "does not contain gold. (3) Box B does not contain bronze. "
            "Question: What does each box contain?"
        ),
        ground_truth_keywords=[
            ["a: gold", "a - gold", "a contains gold", "box a: gold",
             "box a contains gold", "a = gold"],
            ["b: silver", "b - silver", "b contains silver", "box b: silver",
             "box b contains silver", "b = silver"],
            ["c: bronze", "c - bronze", "c contains bronze", "box c: bronze",
             "box c contains bronze", "c = bronze"],
        ],
        source="constructed",
    ),
    Problem(
        id="test-005",
        question=(
            "Given the propositional formula: (P or Q) and (not P or R) and "
            "(not Q or R) and (not R). "
            "Question: Is this formula satisfiable? If yes, give a satisfying "
            "assignment; if no, explain why not."
        ),
        ground_truth_keywords=[["unsatisfiable"]],
        source="constructed",
    ),
    Problem(
        id="test-006",
        question=(
            "Given: (1) All mammals are warm-blooded. (2) All whales are mammals. "
            "(3) Moby is a whale. "
            "Question: Is Moby warm-blooded?"
        ),
        ground_truth_keywords=[["yes", "warm-blooded", "warm blooded"]],
        source="constructed",
    ),
    Problem(
        id="test-007",
        question=(
            "Given: (1) If a number is divisible by 6, it is divisible by 3. "
            "(2) If a number is divisible by 3, the sum of its digits is "
            "divisible by 3. (3) The number 18 is divisible by 6. "
            "Question: Is the sum of the digits of 18 divisible by 3?"
        ),
        ground_truth_keywords=[["yes"]],
        source="constructed",
    ),
    Problem(
        id="test-008",
        question=(
            "Four students - Pat, Quinn, Riley, Sam - finished a race in some "
            "order. Constraints: (1) Pat finished before Quinn. (2) Riley "
            "finished after Sam. (3) Quinn finished before Sam. "
            "Question: What is the finishing order from first to last?"
        ),
        # Order-aware: the canonical ordering Pat, Quinn, Sam, Riley must
        # appear as a contiguous sequence in the answer.
        ground_truth_keywords=[
            ["pat, quinn, sam, riley", "pat -> quinn -> sam -> riley",
             "pat > quinn > sam > riley", "pat, then quinn, then sam, then riley",
             "1. pat", "1) pat"],
            ["quinn", "2. quinn", "2) quinn"],
            ["sam", "3. sam", "3) sam"],
            ["riley", "4. riley", "4) riley"],
        ],
        source="constructed",
    ),
    Problem(
        id="test-009",
        question=(
            "Given the propositional formula: (A or B) and (A or not B) and "
            "(not A or C). "
            "Question: Is this formula satisfiable? If yes, give a satisfying "
            "assignment for A, B, and C."
        ),
        ground_truth_keywords=[
            ["satisfiable"],
            ["a = true", "a: true", "a is true", "a=true"],
            ["c = true", "c: true", "c is true", "c=true"],
        ],
        source="constructed",
    ),
    Problem(
        id="test-010",
        question=(
            "Given: (1) Every student in the class passed either math or "
            "history. (2) No student passed both. (3) Twenty students passed "
            "math. (4) The class has thirty students. "
            "Question: How many students passed history?"
        ),
        ground_truth_keywords=[["10", "ten", "10 students", "ten students"]],
        source="constructed",
    ),
]


def _problems_from_json(path: str) -> list[Problem]:
    """Load problems from a JSON file. Schema mirrors the Problem dataclass.

    Raises ValueError on a malformed file. Callers (e.g. main) decide whether
    to translate that into sys.exit.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"--problems file {path} must contain a JSON list.")
    out: list[Problem] = []
    for i, item in enumerate(raw):
        try:
            out.append(Problem(
                id=str(item["id"]),
                question=str(item["question"]),
                ground_truth_keywords=[
                    [str(s) for s in group]
                    for group in item["ground_truth_keywords"]
                ],
                source=str(item.get("source", "external")),
            ))
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"--problems file {path} item {i} invalid: {exc}"
            ) from exc
    return out


# ---------------------------------------------------------------------------
# Format scoring - replicates the paper's structural validator
# ---------------------------------------------------------------------------

YAML_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADER_RE = re.compile(r"^##\s+(.+?)$", re.MULTILINE)

REQUIRED_HEADERS_PREFIXES = [
    "samshaya",
    "pramana",
    "pancha avayava",
    "tarka",
    "hetvabhasa",
    "nirnaya",
]
REQUIRED_PRAMANA_SUBTYPES = ["pratyaksha", "anumana", "upamana", "shabda"]
REQUIRED_AVAYAVA_PARTS = ["pratijna", "hetu", "udaharana", "upanaya", "nigamana"]
REQUIRED_HETVABHASA_TYPES = [
    "savyabhichara", "viruddha", "prakaranasama", "sadhyasama", "kalaatita",
]
REQUIRED_NIRNAYA_FIELDS = ["answer", "confidence", "epistemic status"]


@dataclass
class FormatReport:
    yaml_present: bool = False
    yaml_has_ground_truth: bool = False
    headers_found: list[str] = field(default_factory=list)
    missing_headers: list[str] = field(default_factory=list)
    headers_in_order: bool = False
    pramana_complete: bool = False
    avayava_complete: bool = False
    udaharana_uses_template: bool = False
    hetvabhasa_complete: bool = False
    nirnaya_complete: bool = False
    placeholders_clean: bool = False

    @property
    def passes(self) -> bool:
        return (
            self.yaml_present
            and self.yaml_has_ground_truth
            and not self.missing_headers
            and self.headers_in_order
            and self.pramana_complete
            and self.avayava_complete
            and self.udaharana_uses_template
            and self.hetvabhasa_complete
            and self.nirnaya_complete
            and self.placeholders_clean
        )


# Sentinel strings explicitly permitted by the system prompt for the two
# sections where the model is told it can opt out of providing detail.
_PRAMANA_NONE_MARKER = "(none in this problem)"
_HETVABHASA_ABSENT_MARKER = "absent"
_MIN_CONTENT_TOKENS = 1  # >=1 real token; rejects empty / pure-placeholder content
                         # but accepts terse-but-correct answers like "It rains."


def _bullet_value(section_text: str, label: str) -> str | None:
    """Return the text after `- <label>:` on its bullet line, or None.

    Only the first occurrence of the label inside the section is considered.
    The match is case-insensitive and tolerates leading whitespace before
    the bullet marker.
    """
    pattern = re.compile(
        rf"^\s*[-*]\s*{re.escape(label)}\s*:\s*(.*?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pattern.search(section_text)
    if not m:
        return None
    return m.group(1)


def _is_placeholder_value(value: str, allowed_sentinels: tuple[str, ...]) -> bool:
    """True if `value` is an explicit allowed sentinel.

    Comparison is case-insensitive and ignores surrounding punctuation/whitespace.
    """
    stripped = value.strip().strip(".").strip().lower()
    return any(stripped == s.lower() for s in allowed_sentinels)


def _has_real_content(value: str | None,
                      allowed_sentinels: tuple[str, ...] = ()) -> bool:
    """True if `value` has at least _MIN_CONTENT_TOKENS non-placeholder tokens.

    A response that fills a bullet with literal `<...>` placeholder syntax,
    a single `x`, or repeated `x` placeholders does NOT count as real content.
    Allowed sentinels (e.g. "absent", "(none in this problem)") count as real
    content.
    """
    if value is None:
        return False
    if _is_placeholder_value(value, allowed_sentinels):
        return True

    # Strip out angle-bracketed placeholders like "<final answer>".
    cleaned = re.sub(r"<[^>]*>", " ", value)
    tokens = [t for t in re.split(r"\s+", cleaned) if t]
    # Reject tokens that are pure punctuation or just "x"/"xxx" placeholders.
    real_tokens = [
        t for t in tokens
        if not re.fullmatch(r"x+", t, re.IGNORECASE)
        and not re.fullmatch(r"[^\w]+", t)
    ]
    return len(real_tokens) >= _MIN_CONTENT_TOKENS


def score_format(text: str) -> FormatReport:
    report = FormatReport()

    yaml_match = YAML_RE.search(text)
    if yaml_match:
        report.yaml_present = True
        report.yaml_has_ground_truth = "ground_truth:" in yaml_match.group(1).lower()

    headers = [h.strip().lower() for h in HEADER_RE.findall(text)]
    report.headers_found = headers
    for prefix in REQUIRED_HEADERS_PREFIXES:
        if not any(h.startswith(prefix) for h in headers):
            report.missing_headers.append(prefix)

    # Section-order check: the system prompt says "in this order".
    # Walk the found headers and confirm their first occurrences appear in the
    # same relative order as REQUIRED_HEADERS_PREFIXES.
    order_indices: list[int] = []
    for h in headers:
        for idx, prefix in enumerate(REQUIRED_HEADERS_PREFIXES):
            if h.startswith(prefix):
                order_indices.append(idx)
                break
    # Headers in order iff order_indices is strictly increasing AND covers all
    # required prefixes (any missing prefix already fails via missing_headers).
    report.headers_in_order = (
        not report.missing_headers
        and order_indices == sorted(order_indices)
        and len(set(order_indices)) == len(REQUIRED_HEADERS_PREFIXES)
    )

    pramana_section = _section_text(text, "pramana")
    pramana_lower = pramana_section.lower()
    report.pramana_complete = all(
        s in pramana_lower for s in REQUIRED_PRAMANA_SUBTYPES
    )

    avayava_section = _section_text(text, "pancha avayava")
    avayava_lower = avayava_section.lower()
    report.avayava_complete = all(p in avayava_lower for p in REQUIRED_AVAYAVA_PARTS)
    report.udaharana_uses_template = (
        "wherever" in avayava_lower and "there is" in avayava_lower
    )

    hetvabhasa_section = _section_text(text, "hetvabhasa")
    hetvabhasa_lower = hetvabhasa_section.lower()
    report.hetvabhasa_complete = all(
        h in hetvabhasa_lower for h in REQUIRED_HETVABHASA_TYPES
    )

    nirnaya_section = _section_text(text, "nirnaya")
    nirnaya_lower = nirnaya_section.lower()
    report.nirnaya_complete = all(
        f in nirnaya_lower for f in REQUIRED_NIRNAYA_FIELDS
    )

    # Placeholder / minimum-content check: every required sub-bullet must
    # either contain >= 3 real tokens, or be the explicitly allowed sentinel.
    placeholders_clean = True
    for label in REQUIRED_PRAMANA_SUBTYPES:
        if not _has_real_content(
            _bullet_value(pramana_section, label),
            allowed_sentinels=(_PRAMANA_NONE_MARKER,),
        ):
            placeholders_clean = False
            break
    if placeholders_clean:
        for label in REQUIRED_AVAYAVA_PARTS:
            if not _has_real_content(_bullet_value(avayava_section, label)):
                placeholders_clean = False
                break
    if placeholders_clean:
        for label in REQUIRED_HETVABHASA_TYPES:
            if not _has_real_content(
                _bullet_value(hetvabhasa_section, label),
                allowed_sentinels=(_HETVABHASA_ABSENT_MARKER,),
            ):
                placeholders_clean = False
                break
    report.placeholders_clean = placeholders_clean

    return report


def _section_text(text: str, header_prefix: str) -> str:
    """Return the text under the first ## header whose name starts with prefix.

    The header prefix is matched at a word boundary so that a prefix like
    "pramana" does NOT match a header named "Pramanavada".
    """
    pattern = re.compile(
        rf"^##\s+{re.escape(header_prefix)}\b.*?$(.*?)(?=^##\s|\Z)",
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Semantic scoring - extract Nirnaya answer, match keywords with synonym
# groups (AND-of-OR) plus digit/word number normalization for 0-100.
# ---------------------------------------------------------------------------

# Hand-rolled digit <-> word map for 0-100. Avoids a new dependency on
# inflect / num2words. Uses American English ("one hundred").
_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
]


def _int_to_word(n: int) -> str:
    if 0 <= n < 20:
        return _ONES[n]
    if 20 <= n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"
    if n == 100:
        return "one hundred"
    return str(n)  # out of range, leave as digits


_DIGIT_TO_WORD: dict[str, str] = {str(n): _int_to_word(n) for n in range(0, 101)}
_WORD_TO_DIGIT: dict[str, str] = {v: k for k, v in _DIGIT_TO_WORD.items()}


def _expand_number_synonyms(token: str) -> list[str]:
    """If token contains a 0-100 digit run or a number word, return both forms."""
    out = {token}
    # Digit -> word (only if the entire token is a bare integer)
    if token.isdigit() and token in _DIGIT_TO_WORD:
        out.add(_DIGIT_TO_WORD[token])
    # Word -> digit (whole-token match on the word form)
    if token in _WORD_TO_DIGIT:
        out.add(_WORD_TO_DIGIT[token])
    # Substring digit replacement (e.g. "10 students" -> "ten students").
    # Only swap whole-word digit runs to avoid mangling "10x" or version strings.
    def _sub_digits(match: re.Match[str]) -> str:
        d = match.group(0)
        return _DIGIT_TO_WORD.get(d, d)

    swapped = re.sub(r"\b\d{1,3}\b", _sub_digits, token)
    if swapped != token:
        out.add(swapped)
    # Substring word replacement: ten students -> 10 students
    for word, digit in _WORD_TO_DIGIT.items():
        # Skip the empty tens slots (word == "")
        if word and re.search(rf"\b{re.escape(word)}\b", token):
            out.add(re.sub(rf"\b{re.escape(word)}\b", digit, token))
    return [s.lower() for s in out]


def _group_matches(haystack: str, group: list[str]) -> bool:
    """Return True if any synonym in `group` (with number normalization) appears
    in haystack as a whole word. `haystack` is expected to already be lowercased.

    Word-boundary matching prevents `satisfiable` from matching inside
    `unsatisfiable`, which is critical for the SAT-style test problems.
    """
    for raw in group:
        for variant in _expand_number_synonyms(raw.lower()):
            if not variant:
                continue
            if re.search(rf"\b{re.escape(variant)}\b", haystack):
                return True
    return False


def score_semantic(text: str, problem: Problem) -> bool:
    """Score a response semantically against a problem's ground-truth keywords.

    Raises ValueError if the problem has an empty ground_truth_keywords list,
    which would otherwise pass vacuously via `all([])`.
    """
    if not problem.ground_truth_keywords:
        raise ValueError(
            f"Problem {problem.id!r} has empty ground_truth_keywords; "
            "an empty keyword list cannot be scored semantically."
        )
    nirnaya = _section_text(text, "nirnaya").lower()
    if not nirnaya:
        # Fall back to YAML ground_truth field if Nirnaya missing
        yaml_match = YAML_RE.search(text)
        nirnaya = (yaml_match.group(1) if yaml_match else "").lower()
    if not nirnaya:
        return False
    # AND across groups, OR within each group.
    return all(_group_matches(nirnaya, group) for group in problem.ground_truth_keywords)


# ---------------------------------------------------------------------------
# Provider-agnostic model call
# ---------------------------------------------------------------------------

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.5-pro",
    "openai": "gpt-5",
}


def _detect_provider() -> str:
    """Return the auto-detected provider based on which API key is in env.

    Precedence (first match wins): Anthropic, Gemini, OpenAI.
    Existing users with ANTHROPIC_API_KEY or GEMINI_API_KEY are unaffected
    by the addition of OpenAI.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    sys.exit(
        "No API key found. Set ANTHROPIC_API_KEY, GEMINI_API_KEY, or "
        "OPENAI_API_KEY (or pass --provider with the matching key set)."
    )


def _build_user_msg(problem: Problem) -> str:
    return (
        f"Problem id: {problem.id}\n\n"
        f"{problem.question}\n\n"
        f"Produce the six-phase Navya-Nyaya analysis exactly as specified."
    )


def _build_client(provider: str) -> Any:
    """Lazily import the SDK for the chosen provider and construct a client.

    Importing inside the function lets users install only the SDK they need.
    """
    if provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError:
            sys.exit("Missing dependency. Run: pip install anthropic")
        return Anthropic()

    if provider == "gemini":
        try:
            from google import genai
        except ImportError:
            sys.exit("Missing dependency. Run: pip install google-genai")
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key)

    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("Missing dependency. Run: pip install openai")
        return OpenAI()

    sys.exit(f"Unknown provider: {provider!r}")


def call_model(client: Any, provider: str, model: str, problem: Problem) -> str:
    """Call the chosen provider with deterministic settings and retry on transient
    errors. All three providers share temperature=0.0, max_tokens=4096, and a
    REQUEST_TIMEOUT_SECONDS per-request timeout for symmetry."""
    user_msg = _build_user_msg(problem)

    if provider == "anthropic":
        def _invoke_anthropic() -> str:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            # Filter for text blocks - the first content block isn't guaranteed
            # to be text (could be tool_use, etc.). Concatenate all text blocks.
            text_parts: list[str] = []
            for block in getattr(resp, "content", []) or []:
                if getattr(block, "type", None) == "text":
                    text_parts.append(getattr(block, "text", "") or "")
            if not text_parts:
                stop_reason = getattr(resp, "stop_reason", "unknown")
                raise RuntimeError(
                    f"Anthropic response had no text blocks "
                    f"(stop_reason={stop_reason!r})."
                )
            return "".join(text_parts)
        return _retry_call(_invoke_anthropic)

    if provider == "gemini":
        # Imported here so the function works even if google-genai isn't installed
        # for non-gemini runs.
        from google.genai import types as genai_types

        def _invoke_gemini() -> str:
            resp = client.models.generate_content(
                model=model,
                contents=user_msg,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=4096,
                    temperature=0.0,
                    # google-genai accepts http_options for per-request timeout
                    # (in milliseconds).
                    http_options=genai_types.HttpOptions(
                        timeout=int(REQUEST_TIMEOUT_SECONDS * 1000),
                    ),
                ),
            )
            # resp.text raises if the response was safety-filtered. Wrap it
            # and surface prompt_feedback / finish_reason instead of crashing.
            try:
                text = resp.text
            except Exception as exc:  # noqa: BLE001 - SDK raises ValueError-like
                feedback = getattr(resp, "prompt_feedback", None)
                candidates = getattr(resp, "candidates", None) or []
                finish_reason = (
                    getattr(candidates[0], "finish_reason", None)
                    if candidates else None
                )
                raise RuntimeError(
                    f"Gemini response unavailable (likely safety-filtered): "
                    f"finish_reason={finish_reason!r} prompt_feedback={feedback!r} "
                    f"underlying={type(exc).__name__}: {exc}"
                ) from exc
            return text or ""
        return _retry_call(_invoke_gemini)

    if provider == "openai":
        def _invoke_openai() -> str:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            choice = resp.choices[0]
            content = getattr(choice.message, "content", None)
            if content is None:
                # Refusal / reasoning-only responses come back with content=None.
                # Surface finish_reason in the error so the caller sees it.
                finish_reason = getattr(choice, "finish_reason", "unknown")
                raise RuntimeError(
                    f"OpenAI response had null content "
                    f"(finish_reason={finish_reason!r})."
                )
            return content
        return _retry_call(_invoke_openai)

    raise ValueError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot test of the Pramana 6-phase reasoning schema.",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "gemini", "openai"],
        default=None,
        help="Force provider (skips env-based auto-detection).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (overrides PRAMANA_MODEL env var).",
    )
    parser.add_argument(
        "--output",
        default="pramana_test_results.json",
        help="JSON output path (default: pramana_test_results.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N problems (smoke test).",
    )
    parser.add_argument(
        "--problems",
        default=None,
        help=("Path to a JSON file of problems (advanced). Each item must have "
              "id, question, ground_truth_keywords (list[list[str]]), and "
              "optionally source."),
    )
    return parser.parse_args(argv)


def _resolve_provider_and_model(args: argparse.Namespace) -> tuple[str, str]:
    provider = args.provider or _detect_provider()
    model = (
        args.model
        or os.environ.get("PRAMANA_MODEL")
        or PROVIDER_DEFAULT_MODELS[provider]
    )
    return provider, model


def _print_table(rows: list[tuple[str, ...]]) -> None:
    """Print a small ASCII table. First row is the header."""
    if not rows:
        return
    ncols = len(rows[0])
    widths = [max(len(r[i]) for r in rows) for i in range(ncols)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    for i, row in enumerate(rows):
        cells = " | ".join(f"{row[j]:<{widths[j]}}" for j in range(ncols))
        print(f"| {cells} |")
        if i == 0:
            print(sep)
    print(sep)


def _print_summary_table(
    n: int,
    format_passes: int,
    semantic_passes: int,
    paper_format_pct: int = 40,
    paper_semantic_pct: int = 100,
) -> None:
    """Print the main summary table comparing ours to the paper, with Wilson CIs."""
    format_str = (
        f"{format_passes}/{n} ({_format_rate_with_ci(format_passes, n)})"
    )
    semantic_str = (
        f"{semantic_passes}/{n} ({_format_rate_with_ci(semantic_passes, n)})"
    )
    rows: list[tuple[str, ...]] = [
        ("Metric", "Ours (95% CI)", "Paper Stage 1"),
        ("Format adherence", format_str, f"{paper_format_pct}%"),
        ("Semantic correctness", semantic_str, f"{paper_semantic_pct}%"),
    ]
    _print_table(rows)


def _print_subgroup_table(results: list[dict[str, Any]]) -> None:
    """Break out paper-verbatim vs constructed problems so readers see both."""
    groups = [
        ("paper-verbatim", "paper"),
        ("constructed", "constructed"),
    ]
    rows: list[tuple[str, ...]] = [("Subgroup", "n", "Format", "Semantic")]
    for label, src in groups:
        items = [r for r in results if r.get("source") == src]
        n = len(items)
        if n == 0:
            continue
        f_pass = sum(1 for r in items if r.get("format_pass"))
        s_pass = sum(1 for r in items if r.get("semantic_pass"))
        rows.append((
            label,
            str(n),
            f"{f_pass}/{n} ({_format_rate_with_ci(f_pass, n)})",
            f"{s_pass}/{n} ({_format_rate_with_ci(s_pass, n)})",
        ))
    if len(rows) > 1:
        _print_table(rows)


def _build_meta_block() -> dict[str, Any]:
    """Capture environment metadata so a JSON output is reproducible."""
    sdk_versions: dict[str, str] = {}
    for pkg in ("anthropic", "google-genai", "openai"):
        v = _sdk_version(pkg)
        if v is not None:
            # Use the conventional import name (underscore) as the JSON key.
            key = pkg.replace("-", "_")
            sdk_versions[key] = v
    return {
        "schema_version": SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "sdk_versions": sdk_versions,
    }


def _validate_output_path(output: str) -> None:
    """Exit early if --output's parent directory does not exist.

    Done before any API calls so a misconfigured path can't waste a run.
    """
    parent = Path(output).expanduser().resolve().parent
    if not parent.exists() or not parent.is_dir():
        sys.exit(
            f"--output parent directory does not exist: {parent} "
            f"(create it first or pass an existing path)."
        )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_output_path(args.output)
    provider, model = _resolve_provider_and_model(args)

    try:
        problems = (
            _problems_from_json(args.problems)
            if args.problems else list(TEST_PROBLEMS)
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"Failed to load problems: {exc}")

    if args.limit is not None:
        if args.limit < 1:
            sys.exit("--limit must be >= 1")
        problems = problems[: args.limit]
    if not problems:
        sys.exit("No problems to run.")

    client = _build_client(provider)

    paper_count = sum(1 for p in problems if p.source == "paper")
    constructed_count = sum(1 for p in problems if p.source == "constructed")
    other_count = len(problems) - paper_count - constructed_count

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print("=" * 78, flush=True)
    print(f"Provider:    {provider}", flush=True)
    print(f"Model:       {model}", flush=True)
    print(f"Started:     {started_at}", flush=True)
    print(f"Problems:    {len(problems)} "
          f"({paper_count} paper-verified, {constructed_count} constructed"
          + (f", {other_count} external" if other_count else "")
          + ")", flush=True)
    print(f"Output:      {args.output}", flush=True)
    print("=" * 78, flush=True)

    results: list[dict[str, Any]] = []
    format_passes = 0
    semantic_passes = 0

    for i, problem in enumerate(problems, 1):
        t0 = time.time()
        response = ""
        error: str | None = None
        error_type: str | None = None
        try:
            response = call_model(client, provider, model, problem)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK errors vary by provider
            error_type = type(exc).__name__
            error = _sanitize(f"{error_type}: {exc!s}")
        elapsed = time.time() - t0

        format_report = score_format(response) if response else FormatReport()
        format_pass = format_report.passes
        try:
            semantic_pass = (
                score_semantic(response, problem) if response else False
            )
        except ValueError as exc:
            # Empty ground_truth_keywords should fail loudly. Surface as an
            # error on this row but don't crash the whole run.
            semantic_pass = False
            extra = _sanitize(f"ValueError: {exc!s}")
            error = f"{error}; {extra}" if error else extra
            error_type = error_type or "ValueError"

        if format_pass:
            format_passes += 1
        if semantic_pass:
            semantic_passes += 1

        marker_f = "FORMAT-PASS" if format_pass else "FORMAT-FAIL"
        marker_s = "SEM-PASS   " if semantic_pass else "SEM-FAIL   "
        line = (
            f"[{i:2d}/{len(problems)}] {problem.id} ({problem.source:12s}) "
            f"{marker_f}  {marker_s}  ({elapsed:5.1f}s)"
        )
        if error:
            line += f"  ERROR: {error_type}"
        print(line, flush=True)

        results.append({
            "id": problem.id,
            "source": problem.source,
            "question": problem.question,
            "ground_truth_keywords": problem.ground_truth_keywords,
            "response": response,
            "format_report": asdict(format_report),
            "format_pass": format_pass,
            "semantic_pass": semantic_pass,
            "error": error,
            "error_type": error_type,
            "elapsed_seconds": round(elapsed, 2),
        })

    n = len(problems)
    format_rate = format_passes / n
    semantic_rate = semantic_passes / n

    print("-" * 78, flush=True)
    _print_summary_table(n, format_passes, semantic_passes)
    print(flush=True)
    _print_subgroup_table(results)
    print(flush=True)

    if format_rate >= 0.80:
        decision = ("Schema is portable: frontier model produces the "
                    "schema zero-shot.")
    elif format_rate >= 0.40:
        decision = ("Schema is partially portable: prompt iteration "
                    "recommended before any fine-tune path.")
    else:
        decision = ("Schema does not travel zero-shot: fine-tuning "
                    "likely required.")
    print(f"DECISION: {decision}", flush=True)

    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload: dict[str, Any] = {
        "meta": _build_meta_block(),
        "provider": provider,
        "model": model,
        "started_at": started_at,
        "finished_at": finished_at,
        "n": n,
        "format_passes": format_passes,
        "semantic_passes": semantic_passes,
        "format_rate": format_rate,
        "semantic_rate": semantic_rate,
        "decision": decision,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nFull traces saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
