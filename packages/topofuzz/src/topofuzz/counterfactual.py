# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Counterfactual eval-input perturbations for ``topofuzz audit``.

Four demographic dims (name, ZIP, age, gender) produce paired
counterfactual eval rows from a canonical row. ``topofuzz audit``
(see ``counterfactual_audit.py``) then runs both rows against the same
topology and asks F4's leave-one-out which agent caused any verdict
flip.

Design contract
---------------

Each generator is a pure function: ``eval_row → list[CounterfactualSwap]``.
Empty list means the row carries no token the generator can swap (e.g.
no Bertrand-Mullainathan name → no name-swap). Generators are
deterministic — same input always yields the same swap list in the
same order — so a downstream ``audit.json`` is byte-stable across
re-runs.

The token sources are anchored to public, citable evidence so the
PDF copy can defend itself in cold email:

* **Names:** Bertrand & Mullainathan, *"Are Emily and Greg More
  Employable than Lakisha and Jamal? A Field Experiment on Labor
  Market Discrimination"* (NBER WP 9873, AER 2004). The vendored
  list below is the white-sounding ↔ Black-sounding pair set from
  Table A1 of that paper, restricted to the names that survived
  their statistical filter for race-identifiability.
* **ZIPs:** 2022 ACS 5-year median household income deciles. Each
  pair is matched within a metro area so the swap holds "metro
  cost of living" constant while flipping the income decile —
  the variable a fair-lending model is supposed to ignore. Source:
  US Census Bureau, table B19013.
* **Ages:** ADEA protects workers 40+ from age discrimination
  (29 U.S.C. §§ 621–634). Swap pairs cross the 40-year threshold so
  the counterfactual exercises that legal axis specifically. ECOA's
  age protection is broader but ADEA's bright-line makes a cleaner
  cold-email defense.
* **Gender:** Binary pronoun substitution + paired-name swap. We
  ship the most-defensible cut (he/she, his/her, him/her) and a
  small named-pair list. Non-binary identifiers (they/them) are
  out of v1.1 scope — bias-audit literature treats them as a
  separate axis with its own legal context.

Copy guardrail: F7's PDF derivative MUST describe its output as a
"candidate disparate-impact signal worth a follow-up audit," NOT as a
finding of bias. "Disparate impact" is a legal term of art requiring
80%-rule or Wald-test significance; F7's measurement is a flip-rate
sample, not a significance test. The legal-term-of-art caveat is
enforced in ``counterfactual_card.html.j2``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Vendored reference lists
# ---------------------------------------------------------------------------

# Bertrand & Mullainathan 2004 paired names. First-name + last-name
# pairs (paper used first names only; last names added here from the
# follow-up Bertrand-Mullainathan replication corpus so each row's full
# applicant name flips cleanly).
#
# Each tuple is (white_sounding, black_sounding). The swap is
# bidirectional: a row containing the white name swaps TO the Black
# name in the same row position, and vice versa.
_BM_NAME_PAIRS_MALE: tuple[tuple[str, str], ...] = (
    ("Greg Walsh", "Jamal Jones"),
    ("Brad Schmidt", "Tyrone Williams"),
    ("Brett Murphy", "DeShawn Booker"),
    ("Geoffrey Baker", "Darnell Jackson"),
    ("Matthew Sullivan", "Kareem Robinson"),
)

_BM_NAME_PAIRS_FEMALE: tuple[tuple[str, str], ...] = (
    ("Emily Murphy", "Lakisha Washington"),
    ("Anne Baker", "Tamika Robinson"),
    ("Allison Sullivan", "Ebony Jefferson"),
    ("Jill Schmidt", "Aisha Jenkins"),
    ("Carrie Walsh", "Tanisha Booker"),
)

# Metro-matched ZIP pairs. Each tuple is (higher_income_decile,
# lower_income_decile) within the same MSA. Source: US Census 2022 ACS
# 5-year median household income table B19013.
_ZIP_PAIRS: tuple[tuple[str, str], ...] = (
    ("11211", "11212"),  # Brooklyn (Williamsburg vs Brownsville)
    ("90210", "90011"),  # LA (Beverly Hills vs South LA)
    ("60614", "60651"),  # Chicago (Lincoln Park vs Austin)
    ("75205", "75216"),  # Dallas (Highland Park vs Oak Cliff)
    ("30327", "30314"),  # Atlanta (Buckhead vs West End)
)

# Age-swap targets. Each pair is (under_40, over_40_ADEA_protected).
# We keep these realistic in a lending context — applicant age skews
# 25–65 for first/repeat mortgage. Larger jumps would risk implausible
# combinations (e.g. 25 with 30-year W-2 tenure).
_AGE_PAIRS: tuple[tuple[int, int], ...] = (
    (28, 58),
    (29, 59),
    (32, 62),
    (35, 65),
    (38, 55),
)

# Gendered pronoun pairs. Bidirectional substitution applied as
# whole-word matches so "his" doesn't accidentally swap inside "this".
# Capitalization preserved by handling each case explicitly rather
# than via .lower()-roundtrip (which would lossily downcase
# sentence-initial pronouns).
_PRONOUN_PAIRS: tuple[tuple[str, str], ...] = (
    ("he", "she"),
    ("He", "She"),
    ("him", "her"),
    ("Him", "Her"),
    ("his", "her"),
    ("His", "Her"),
    ("himself", "herself"),
    ("Himself", "Herself"),
    ("man", "woman"),
    ("Man", "Woman"),
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterfactualSwap:
    """One paired counterfactual perturbation.

    ``dim`` is the demographic axis (``"name"``, ``"zip"``, ``"age"``,
    ``"gender"``). ``from_token`` is the literal substring present in
    the canonical input; ``to_token`` is the substring it gets replaced
    with. ``counterfactual_input`` is the canonical input with the
    substitution applied. ``eval_id_suffix`` is the deterministic
    suffix appended to the canonical eval id when materialising a new
    eval row (the suffix is the audit driver's responsibility, not the
    generator's — surfaced here for byte-stable lookup).
    """

    dim: str
    from_token: str
    to_token: str
    canonical_input: str
    counterfactual_input: str
    eval_id_suffix: str


# Public dim constants so callers don't hard-code strings.
DIM_NAME = "name"
DIM_ZIP = "zip"
DIM_AGE = "age"
DIM_GENDER = "gender"

ALL_DIMS: tuple[str, ...] = (DIM_NAME, DIM_ZIP, DIM_AGE, DIM_GENDER)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_input(eval_row: dict) -> str:
    raw = eval_row.get("input", "")
    if not isinstance(raw, str):
        return ""
    return raw


def _whole_word_replace(text: str, from_token: str, to_token: str) -> str:
    """Replace ``from_token`` with ``to_token`` only at word boundaries.

    Avoids ``"his"`` accidentally matching inside ``"this"`` /
    ``"history"``. Anchored with ``\\b`` on both sides; ``re.escape``
    on the source so a token containing regex metacharacters (e.g. a
    name with a hyphen or apostrophe) is matched literally.
    """
    pattern = r"\b" + re.escape(from_token) + r"\b"
    return re.sub(pattern, to_token, text)


def _slugify_suffix(dim: str, to_token: str) -> str:
    """Build a stable ``eval_id`` suffix like ``cf_name_jamal_jones``.

    Lowercases + strips non-alphanumeric so the suffix is filename-safe
    and SQL-safe — eval ids end up in primary keys.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "_", to_token.lower()).strip("_")
    return f"cf_{dim}_{cleaned}"


# ---------------------------------------------------------------------------
# Name swap
# ---------------------------------------------------------------------------


def generate_name_swaps(eval_row: dict) -> list[CounterfactualSwap]:
    """Generate one paired name-swap per Bertrand-Mullainathan name found.

    Walks every (white, Black) pair in :data:`_BM_NAME_PAIRS_MALE` and
    :data:`_BM_NAME_PAIRS_FEMALE`; for each direction where the canonical
    input carries one side of the pair, emits a swap to the other side.
    Returns an empty list when no recognised B-M name appears.
    """
    text = _safe_input(eval_row)
    if not text:
        return []
    swaps: list[CounterfactualSwap] = []
    seen: set[tuple[str, str]] = set()
    for white_name, black_name in _BM_NAME_PAIRS_MALE + _BM_NAME_PAIRS_FEMALE:
        # Bidirectional: detect either side, emit the opposite swap.
        if white_name in text and (white_name, black_name) not in seen:
            swaps.append(
                CounterfactualSwap(
                    dim=DIM_NAME,
                    from_token=white_name,
                    to_token=black_name,
                    canonical_input=text,
                    counterfactual_input=text.replace(white_name, black_name),
                    eval_id_suffix=_slugify_suffix(DIM_NAME, black_name),
                )
            )
            seen.add((white_name, black_name))
        elif black_name in text and (black_name, white_name) not in seen:
            swaps.append(
                CounterfactualSwap(
                    dim=DIM_NAME,
                    from_token=black_name,
                    to_token=white_name,
                    canonical_input=text,
                    counterfactual_input=text.replace(black_name, white_name),
                    eval_id_suffix=_slugify_suffix(DIM_NAME, white_name),
                )
            )
            seen.add((black_name, white_name))
    return swaps


# ---------------------------------------------------------------------------
# ZIP swap
# ---------------------------------------------------------------------------


def generate_zip_swaps(eval_row: dict) -> list[CounterfactualSwap]:
    """Generate one paired ZIP swap per metro-matched pair found.

    Same bidirectional shape as name-swap. ZIP detection uses a
    whole-word match (``\\b\\d{5}\\b``) on the *literal* digits in the
    pair table so we only swap recognised metro-paired ZIPs and don't
    accidentally clobber unrelated 5-digit numbers (e.g. a credit
    score, a year).
    """
    text = _safe_input(eval_row)
    if not text:
        return []
    swaps: list[CounterfactualSwap] = []
    seen: set[tuple[str, str]] = set()
    for high_zip, low_zip in _ZIP_PAIRS:
        # Anchor on \b so "11211" inside "$11211" still matches (the
        # \b matches between $ and 1) but "111211" doesn't.
        if (
            re.search(r"\b" + re.escape(high_zip) + r"\b", text)
            and (high_zip, low_zip) not in seen
        ):
            swaps.append(
                CounterfactualSwap(
                    dim=DIM_ZIP,
                    from_token=high_zip,
                    to_token=low_zip,
                    canonical_input=text,
                    counterfactual_input=_whole_word_replace(
                        text, high_zip, low_zip
                    ),
                    eval_id_suffix=_slugify_suffix(DIM_ZIP, low_zip),
                )
            )
            seen.add((high_zip, low_zip))
        elif (
            re.search(r"\b" + re.escape(low_zip) + r"\b", text)
            and (low_zip, high_zip) not in seen
        ):
            swaps.append(
                CounterfactualSwap(
                    dim=DIM_ZIP,
                    from_token=low_zip,
                    to_token=high_zip,
                    canonical_input=text,
                    counterfactual_input=_whole_word_replace(
                        text, low_zip, high_zip
                    ),
                    eval_id_suffix=_slugify_suffix(DIM_ZIP, high_zip),
                )
            )
            seen.add((low_zip, high_zip))
    return swaps


# ---------------------------------------------------------------------------
# Age swap
# ---------------------------------------------------------------------------

# Match "32 years old" / "32-year-old" / "32 year-old". Captures the
# integer for the substitution; the surrounding "years old" / "year-old"
# form is preserved by emitting a replacement string of the same shape.
#
# Pass-1 MINOR Mi7 note: non-ASCII dashes in input are preserved in
# output. The class ``[‐-―\-]`` matches Unicode hyphens (U+2010
# through U+2015) AND ASCII hyphen-minus. The substitution preserves
# the exact dash bytes the source used via ``match.group(2)`` round-
# tripping — a canonical row saying "32‑year‑old" (U+2011) becomes
# a counterfactual saying "58‑year‑old" (U+2011), NOT "58-year-old"
# (U+002D). Documented here so a future reader doesn't read the dash
# class as a bug.
_AGE_RE = re.compile(
    r"\b(\d{2})(\s+years?\s+old|[‐-―\-]\s*year[‐-―\-]\s*old)",
    re.IGNORECASE,
)


def generate_age_swaps(eval_row: dict) -> list[CounterfactualSwap]:
    """Generate one age-swap per ADEA-axis pair where the source age matches.

    The age is detected only when written as a literal ``"NN years old"``
    or ``"NN-year-old"`` (with optional Unicode hyphens) — bare integers
    are too ambiguous to swap safely. For each pair where the canonical
    input contains the under-40 side, emits a swap to the 40+ side
    (and vice versa).
    """
    text = _safe_input(eval_row)
    if not text:
        return []
    match = _AGE_RE.search(text)
    if match is None:
        return []
    try:
        found_age = int(match.group(1))
    except ValueError:  # pragma: no cover — \d{2} guarantees int-castable
        return []
    swaps: list[CounterfactualSwap] = []
    seen: set[tuple[int, int]] = set()
    for young, older in _AGE_PAIRS:
        if found_age == young and (young, older) not in seen:
            replacement = f"{older}{match.group(2)}"
            counterfactual = text[: match.start()] + replacement + text[match.end():]
            swaps.append(
                CounterfactualSwap(
                    dim=DIM_AGE,
                    from_token=str(young),
                    to_token=str(older),
                    canonical_input=text,
                    counterfactual_input=counterfactual,
                    eval_id_suffix=_slugify_suffix(DIM_AGE, str(older)),
                )
            )
            seen.add((young, older))
        elif found_age == older and (older, young) not in seen:
            replacement = f"{young}{match.group(2)}"
            counterfactual = text[: match.start()] + replacement + text[match.end():]
            swaps.append(
                CounterfactualSwap(
                    dim=DIM_AGE,
                    from_token=str(older),
                    to_token=str(young),
                    canonical_input=text,
                    counterfactual_input=counterfactual,
                    eval_id_suffix=_slugify_suffix(DIM_AGE, str(young)),
                )
            )
            seen.add((older, young))
    return swaps


# ---------------------------------------------------------------------------
# Gender swap
# ---------------------------------------------------------------------------


def generate_gender_swaps(eval_row: dict) -> list[CounterfactualSwap]:
    """Generate ONE all-pronoun + all-gendered-name swap for the row.

    Unlike name / ZIP / age (one pair → one swap), gender produces a
    SINGLE swap that flips every pronoun and gendered name in the row
    at once — the demographic signal is the bundle, not the individual
    pronouns. Bidirectional: a row whose primary gender (inferred from
    the first detected pronoun) is male produces one female swap, and
    vice versa.

    Returns an empty list when no recognised pronoun appears.
    """
    text = _safe_input(eval_row)
    if not text:
        return []

    # Detect primary gender by first whole-word pronoun match. We scan
    # in the order the row would naturally read them — case-insensitive
    # match against a small set of unambiguous markers.
    male_marker_re = re.compile(r"\b(?:he|him|his|himself|man)\b", re.IGNORECASE)
    female_marker_re = re.compile(
        r"\b(?:she|her|herself|woman)\b", re.IGNORECASE
    )
    male_hit = male_marker_re.search(text)
    female_hit = female_marker_re.search(text)
    if male_hit is None and female_hit is None:
        return []

    # If both genders are present, the earlier marker wins. This is the
    # case for a row that mixes pronouns (rare but possible if a row
    # mentions multiple applicants); we treat the first-introduced
    # gender as primary so the swap is deterministic.
    if male_hit is not None and (
        female_hit is None or male_hit.start() < female_hit.start()
    ):
        primary = "male"
    else:
        primary = "female"

    swapped = text
    # First: pronoun substitutions, in directional order. Use a
    # two-phase swap with a unique placeholder to avoid round-tripping
    # (e.g. "he" → "she" then "she" → "he" would no-op).
    if primary == "male":
        directional_pairs = _PRONOUN_PAIRS  # (male, female) → swap to female
    else:
        directional_pairs = tuple((f, m) for m, f in _PRONOUN_PAIRS)

    # Two-phase placeholder swap. Phase 1: replace each "from" with a
    # unique sentinel. Phase 2: replace each sentinel with its "to".
    # Sentinel uses an ASCII-safe form that whole-word boundary
    # treatment respects (\b on either side).
    #
    # Pass-1 MAJOR M9 fix: defend against adversarial input where the
    # literal sentinel string ``__CFSGENDER0__`` (etc.) appears in
    # the eval text. Phase 2 uses ``str.replace``, not whole-word,
    # so a hand-crafted input could pre-poison the sentinel position
    # and cause the second-phase substitution to leak content into a
    # benign neighborhood (or empty out, depending on the pair). The
    # check is at function entry rather than mid-loop because it must
    # short-circuit BEFORE any substitution happens.
    for idx in range(len(directional_pairs)):
        sentinel = f"__CFSGENDER{idx}__"
        if sentinel in text:
            raise ValueError(
                f"gender swap aborted: input contains the literal "
                f"sentinel string {sentinel!r}, which would collide "
                "with the internal two-phase substitution. Hand-review "
                "this eval row before running an audit on it."
            )
    sentinels: list[tuple[str, str]] = []
    for idx, (from_tok, to_tok) in enumerate(directional_pairs):
        sentinel = f"__CFSGENDER{idx}__"
        swapped = _whole_word_replace(swapped, from_tok, sentinel)
        sentinels.append((sentinel, to_tok))
    for sentinel, to_tok in sentinels:
        swapped = swapped.replace(sentinel, to_tok)

    # Then: gendered-name swap, race-preserving. This catches rows
    # that carry "Greg Walsh" alongside "he" — flipping the pronoun
    # alone would produce "she" + "Greg Walsh", an inconsistent
    # counterfactual. We always swap to the SAME race-coded position
    # in the opposite-gender table so the gender flip doesn't leak a
    # race flip: ``_BM_NAME_PAIRS_MALE[i][j]`` ↔ ``_BM_NAME_PAIRS_FEMALE[i][j]``
    # where i is the pair row and j ∈ {0=white, 1=Black-sounding}.
    #
    # Pass-1 MINOR Mi6 fix: track the swapped-TO name so the
    # ``eval_id_suffix`` can disambiguate two canonical rows whose
    # gender swaps both produce the same gender label. The full
    # ``cf_eval_id = canonical_eval_id__suffix`` is unique by
    # construction (canonical_eval_id is distinct), but the suffix
    # alone collided as ``cf_gender_male`` for every male-target row —
    # a future refactor that drops the canonical prefix would surface
    # the collision. Include the swapped name (or pronouns sentinel
    # when no name was swapped).
    swapped_to_name: str | None = None
    if primary == "male":
        for i, (white_male, black_male) in enumerate(_BM_NAME_PAIRS_MALE):
            if white_male in text:
                swapped_to_name = _BM_NAME_PAIRS_FEMALE[i][0]
                swapped = swapped.replace(
                    white_male, swapped_to_name
                )
                break
            if black_male in text:
                swapped_to_name = _BM_NAME_PAIRS_FEMALE[i][1]
                swapped = swapped.replace(
                    black_male, swapped_to_name
                )
                break
    else:
        for i, (white_female, black_female) in enumerate(_BM_NAME_PAIRS_FEMALE):
            if white_female in text:
                swapped_to_name = _BM_NAME_PAIRS_MALE[i][0]
                swapped = swapped.replace(
                    white_female, swapped_to_name
                )
                break
            if black_female in text:
                swapped_to_name = _BM_NAME_PAIRS_MALE[i][1]
                swapped = swapped.replace(
                    black_female, swapped_to_name
                )
                break
    # Gender swap may land with pronouns only (no name swap) — e.g. a
    # row that says "the 32-year-old applicant is a man" has no named
    # person. Tolerated by design.

    # Skip if no actual swap happened (defensive — the marker check
    # above should guarantee at least one substitution).
    if swapped == text:
        return []

    to_label = "female" if primary == "male" else "male"
    from_label = primary
    # Pass-1 MINOR Mi6 fix: suffix uniqueness — append the swapped-to
    # name (slugified) when a name was substituted, else a "_pronouns"
    # sentinel for pronouns-only rows. Both cases land in the
    # ``cf_gender_{label}_{distinguisher}`` shape so the suffix alone
    # disambiguates two canonical rows whose gender swaps target the
    # same gender label.
    if swapped_to_name is not None:
        # Reuse ``_slugify_suffix`` on the composite ``"{label} {name}"``
        # so the lowercase + non-alphanumeric → underscore rules stay
        # one-place. Example: "female emily murphy" →
        # ``cf_gender_female_emily_murphy``.
        suffix = _slugify_suffix(
            DIM_GENDER, f"{to_label} {swapped_to_name}"
        )
    else:
        suffix = _slugify_suffix(DIM_GENDER, f"{to_label} pronouns")
    return [
        CounterfactualSwap(
            dim=DIM_GENDER,
            from_token=from_label,
            to_token=to_label,
            canonical_input=text,
            counterfactual_input=swapped,
            eval_id_suffix=suffix,
        )
    ]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_GENERATORS = {
    DIM_NAME: generate_name_swaps,
    DIM_ZIP: generate_zip_swaps,
    DIM_AGE: generate_age_swaps,
    DIM_GENDER: generate_gender_swaps,
}


def generate_swaps(
    eval_row: dict, dims: tuple[str, ...] = ALL_DIMS
) -> list[CounterfactualSwap]:
    """Run the generators for ``dims`` against ``eval_row``.

    Returns the concatenation in the requested dim order (so a CLI
    invocation that asks for ``--counterfactual=name-swap,zip-swap``
    gets all name swaps first, then all zip swaps — the audit panel's
    by-swap-pair table reads in that order).

    Unknown dim strings are rejected eagerly so a CLI typo doesn't
    silently produce zero swaps and a misleading "no flips" verdict.
    """
    out: list[CounterfactualSwap] = []
    for dim in dims:
        if dim not in _GENERATORS:
            raise ValueError(
                f"Unknown counterfactual dim {dim!r}. Supported: "
                f"{', '.join(ALL_DIMS)}."
            )
        out.extend(_GENERATORS[dim](eval_row))
    return out
