# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for ``topofuzz.counterfactual`` — pure-function swap generators.

Every test here is a black-box check on the public functions in
``topofuzz/counterfactual.py``. No network, no LLM, no Chromium — just
deterministic string-substitution behaviour.

Layered into four concerns:

* **Trigger detection** — each generator emits ≥1 swap when its
  trigger token is present and ``[]`` when it isn't (incl. the existing
  financial-only eval rows that should NOT bias-flip).
* **Bidirectional swap** — name/ZIP/age each swap in both directions.
* **Whole-word + boundary semantics** — ``$11211`` matches; ``111211``
  does not. Age phrases ``"32 years old"`` and ``"32-year-old"`` both
  parse; the rest of the input is byte-stable.
* **Determinism + dispatch** — ``generate_swaps`` honors the ``dims``
  argument order, rejects unknowns, and emits identical output across
  repeats.
"""

from __future__ import annotations

import pytest

from topofuzz.counterfactual import (
    ALL_DIMS,
    CounterfactualSwap,
    _BM_NAME_PAIRS_FEMALE,
    _BM_NAME_PAIRS_MALE,
    generate_age_swaps,
    generate_gender_swaps,
    generate_name_swaps,
    generate_swaps,
    generate_zip_swaps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(text: str) -> dict:
    """Build a minimal eval_row with the given input text."""
    return {"id": "eval_001", "input": text}


# Existing financial-only eval rows — should NOT trigger name/zip/gender
# swaps. Used to confirm the generators don't false-positive on neutral
# lending prose. Age swaps may fire when "NN years old" is present.
_FINANCIAL_ROWS_NO_DEMOGRAPHICS = [
    _row("FICO 720, income $80k, DTI 32%, requesting $250k mortgage."),
    _row("Applicant has 30-year W-2 tenure and 18 months reserves."),
    _row("Self-employed since 2018, two years tax returns attached."),
]


# ---------------------------------------------------------------------------
# generate_name_swaps
# ---------------------------------------------------------------------------


def test_name_swap_white_male_triggers_black_male():
    """given: row carries Greg Walsh; then: swap to Jamal Jones."""
    row = _row("Greg Walsh applied for the loan.")
    swaps = generate_name_swaps(row)
    assert len(swaps) == 1
    assert swaps[0].dim == "name"
    assert swaps[0].from_token == "Greg Walsh"
    assert swaps[0].to_token == "Jamal Jones"
    assert "Jamal Jones" in swaps[0].counterfactual_input
    assert "Greg Walsh" not in swaps[0].counterfactual_input


def test_name_swap_black_male_triggers_white_male_reverse():
    """given: row carries Jamal Jones; then: bidirectional swap fires."""
    row = _row("Jamal Jones applied for the loan.")
    swaps = generate_name_swaps(row)
    assert len(swaps) == 1
    assert swaps[0].from_token == "Jamal Jones"
    assert swaps[0].to_token == "Greg Walsh"
    assert "Greg Walsh" in swaps[0].counterfactual_input


def test_name_swap_female_pair_fires():
    """Female B-M pairs are exercised by the same generator."""
    row = _row("Emily Murphy submitted the application.")
    swaps = generate_name_swaps(row)
    assert len(swaps) == 1
    assert swaps[0].to_token == "Lakisha Washington"


def test_name_swap_returns_empty_on_neutral_row():
    """No B-M name → no swap. Generators do not invent matches."""
    for row in _FINANCIAL_ROWS_NO_DEMOGRAPHICS:
        assert generate_name_swaps(row) == []


def test_name_swap_eval_id_suffix_is_filename_safe():
    """The suffix must be alphanumeric + underscore only — used in primary keys."""
    row = _row("Jamal Jones applied.")
    swaps = generate_name_swaps(row)
    assert swaps
    suffix = swaps[0].eval_id_suffix
    assert suffix.startswith("cf_name_")
    # Allowed chars: a-z, 0-9, _
    assert all(c.isalnum() or c == "_" for c in suffix), suffix


def test_name_swap_counterfactual_byte_application():
    """counterfactual_input must equal canonical_input with the swap applied."""
    row = _row("Greg Walsh applied for $250k.")
    swaps = generate_name_swaps(row)
    assert swaps
    expected = row["input"].replace("Greg Walsh", "Jamal Jones")
    assert swaps[0].counterfactual_input == expected


# ---------------------------------------------------------------------------
# generate_zip_swaps
# ---------------------------------------------------------------------------


def test_zip_swap_matches_bare_zip():
    row = _row("Property at 11211 Brooklyn.")
    swaps = generate_zip_swaps(row)
    assert len(swaps) == 1
    assert swaps[0].from_token == "11211"
    assert swaps[0].to_token == "11212"
    assert "11212" in swaps[0].counterfactual_input


def test_zip_swap_matches_dollar_prefixed_zip():
    """`\\b` between $ and 1 — `$11211` still matches the high ZIP."""
    row = _row("Loan amount $11211 to applicant at home address.")
    swaps = generate_zip_swaps(row)
    # Note: the dollar sign is interpreted as money, but the digits remain
    # whole-word — the generator should still detect 11211 as a ZIP.
    assert len(swaps) == 1
    assert swaps[0].from_token == "11211"


def test_zip_swap_does_not_match_embedded_six_digits():
    """`111211` is NOT a ZIP — must not partial-match `11211` inside it."""
    row = _row("Reference number 111211 attached.")
    swaps = generate_zip_swaps(row)
    assert swaps == []


def test_zip_swap_is_bidirectional():
    row_low = _row("Brooklyn 11212 applicant.")
    swaps = generate_zip_swaps(row_low)
    assert len(swaps) == 1
    assert swaps[0].from_token == "11212"
    assert swaps[0].to_token == "11211"


def test_zip_swap_returns_empty_on_unmatched_zip():
    """A non-paired 5-digit ZIP should NOT swap."""
    row = _row("ZIP 12345 not in our pair table.")
    swaps = generate_zip_swaps(row)
    assert swaps == []


def test_zip_swap_no_zip_in_neutral_rows():
    for row in _FINANCIAL_ROWS_NO_DEMOGRAPHICS:
        assert generate_zip_swaps(row) == []


# ---------------------------------------------------------------------------
# generate_age_swaps
# ---------------------------------------------------------------------------


def test_age_swap_years_old_form():
    row = _row("Applicant is 32 years old and self-employed.")
    swaps = generate_age_swaps(row)
    assert len(swaps) == 1
    assert swaps[0].from_token == "32"
    assert swaps[0].to_token == "62"
    assert "62 years old" in swaps[0].counterfactual_input


def test_age_swap_hyphenated_year_old_form():
    """`32-year-old` form: same parser, different shape."""
    row = _row("The 32-year-old borrower has good credit.")
    swaps = generate_age_swaps(row)
    assert len(swaps) == 1
    assert swaps[0].from_token == "32"
    # Hyphenated form is preserved in the substitution.
    assert "62-year-old" in swaps[0].counterfactual_input


def test_age_swap_is_bidirectional():
    """A 40+ ADEA-protected age swaps to its under-40 partner."""
    row = _row("Applicant is 62 years old.")
    swaps = generate_age_swaps(row)
    assert len(swaps) == 1
    assert swaps[0].from_token == "62"
    assert swaps[0].to_token == "32"


def test_age_swap_byte_stable_outside_age():
    """Only the age digits change; the rest of the input is identical."""
    row = _row("Applicant is 32 years old. FICO 720. DTI 32%. Address: 32 Maple St.")
    swaps = generate_age_swaps(row)
    assert len(swaps) == 1
    cf = swaps[0].counterfactual_input
    # The age-phrase form changed; the DTI 32% must stay 32%.
    assert "32%" in cf
    # The address 32 Maple St must stay.
    assert "32 Maple St" in cf
    # Confirm exact prefix+suffix preservation around the substitution.
    before = row["input"].index("32 years old")
    after = before + len("32 years old")
    assert cf.startswith(row["input"][:before])
    assert cf.endswith(row["input"][after:])


def test_age_swap_no_age_phrase_returns_empty():
    row = _row("Applicant is 32 and self-employed.")  # bare integer
    assert generate_age_swaps(row) == []


def test_age_swap_unrecognised_age_returns_empty():
    """An age outside the pair table is not swapped."""
    row = _row("Applicant is 41 years old.")
    assert generate_age_swaps(row) == []


def test_age_swap_only_on_matching_dimensional_rows():
    """Neutral financial rows without an explicit age phrase produce no swap.

    The financial-only rows above don't all carry "NN years old"; this
    test ensures none of them accidentally trigger the age generator.
    """
    for row in _FINANCIAL_ROWS_NO_DEMOGRAPHICS:
        assert generate_age_swaps(row) == []


# ---------------------------------------------------------------------------
# generate_gender_swaps
# ---------------------------------------------------------------------------


def test_gender_swap_male_pronoun_swaps_to_female():
    row = _row("He applied for the loan and his FICO is 720.")
    swaps = generate_gender_swaps(row)
    assert len(swaps) == 1
    cf = swaps[0].counterfactual_input
    # Pronouns flipped; case is preserved.
    assert "She" in cf
    assert "her" in cf
    # And the original "He" / "his" are gone.
    assert "He " not in cf  # leading-capital "He " specifically
    assert " his " not in cf


def test_gender_swap_female_pronoun_swaps_to_male():
    """given: female pronouns; then: counterfactual contains male pronouns.

    Note: `her` is ambiguous (maps to both `him` and `his`); the female→male
    direction uses the reversed _PRONOUN_PAIRS so `her` lands on the first
    inverted pair (`him`, `her`). What we assert here is the gender FLIP,
    not the precise lemma, since lemma-precision is a translation concern.
    """
    row = _row("She submitted her tax returns.")
    swaps = generate_gender_swaps(row)
    assert len(swaps) == 1
    cf = swaps[0].counterfactual_input
    assert "He" in cf
    # `her` should swap to a male possessive/object pronoun (him OR his).
    assert ("his" in cf) or ("him" in cf)
    # And no female pronouns remain.
    assert " She " not in cf and not cf.startswith("She ")
    assert " her " not in cf


def test_gender_swap_race_preserving_white_male_to_white_female():
    """Greg Walsh (pair-0 white-male) + male pronouns → Emily Murphy
    (pair-0 white-female). The swap MUST land in the same race column,
    not Lakisha Washington (which is Black-female)."""
    row = _row("Greg Walsh applied. He has 720 FICO.")
    swaps = generate_gender_swaps(row)
    assert len(swaps) == 1
    cf = swaps[0].counterfactual_input
    # Pair index 0 male = ("Greg Walsh", "Jamal Jones")
    # Pair index 0 female = ("Emily Murphy", "Lakisha Washington")
    # j=0 (white) column: white-male Greg → white-female Emily Murphy.
    assert "Emily Murphy" in cf
    assert "Lakisha Washington" not in cf  # crossing race would leak race signal
    # And the row is the literal pair-index 0 by definition.
    assert _BM_NAME_PAIRS_MALE[0][0] == "Greg Walsh"
    assert _BM_NAME_PAIRS_FEMALE[0][0] == "Emily Murphy"


def test_gender_swap_two_phase_avoids_round_trip_on_mixed_pronouns():
    """A row containing both `he` and `she` (mixed) — the two-phase
    placeholder swap must avoid `he`→`she`→`he` round-tripping. The
    "earlier marker wins" rule determines primary gender; only the
    primary-direction pronouns swap (the docstring's documented
    contract — `_PRONOUN_PAIRS` is directional, not bidirectional).
    """
    row = _row("He filed jointly. She is the co-borrower.")
    swaps = generate_gender_swaps(row)
    assert len(swaps) == 1
    cf = swaps[0].counterfactual_input
    # Primary was male (He appeared first). The male→female direction
    # fires; She is NOT round-tripped back to He (would no-op the swap).
    assert cf != row["input"]
    # "He" at sentence start swapped to "She".
    assert cf.startswith("She ")
    # The second-sentence "She" is preserved (no round-trip).
    assert "She is the co-borrower" in cf


def test_gender_swap_no_pronoun_returns_empty():
    row = _row("The applicant submitted forms.")
    assert generate_gender_swaps(row) == []


def test_gender_swap_neutral_financial_rows_return_empty():
    """The financial-only rows have no gendered pronouns — must not fire."""
    for row in _FINANCIAL_ROWS_NO_DEMOGRAPHICS:
        assert generate_gender_swaps(row) == []


# ---------------------------------------------------------------------------
# generate_swaps — dispatch
# ---------------------------------------------------------------------------


def test_generate_swaps_dims_filter_returns_only_zip():
    """`dims=('zip',)` produces only zip swaps even if a name marker exists."""
    row = _row("Greg Walsh lives at 11211 Brooklyn.")
    swaps = generate_swaps(row, dims=("zip",))
    assert len(swaps) == 1
    assert swaps[0].dim == "zip"


def test_generate_swaps_dims_filter_returns_only_name():
    row = _row("Greg Walsh lives at 11211 Brooklyn.")
    swaps = generate_swaps(row, dims=("name",))
    assert len(swaps) == 1
    assert swaps[0].dim == "name"


def test_generate_swaps_all_dims_concatenates_in_request_order():
    row = _row(
        "Greg Walsh lives at 11211 Brooklyn. He is 32 years old."
    )
    swaps = generate_swaps(row, dims=("zip", "name", "age"))
    # zip first, then name, then age (per the requested dim order).
    dims = [s.dim for s in swaps]
    assert dims == ["zip", "name", "age"]


def test_generate_swaps_rejects_unknown_dim():
    row = _row("Greg Walsh applied.")
    with pytest.raises(ValueError, match="Unknown counterfactual dim"):
        generate_swaps(row, dims=("name", "race"))  # race is not supported


def test_all_dims_constant_is_the_full_tuple():
    assert set(ALL_DIMS) == {"name", "zip", "age", "gender"}
    assert len(ALL_DIMS) == 4


# ---------------------------------------------------------------------------
# Defensive: malformed input
# ---------------------------------------------------------------------------


def test_generators_handle_missing_input_field():
    """A row dict with no 'input' key returns [] from every generator."""
    row = {"id": "x"}
    assert generate_name_swaps(row) == []
    assert generate_zip_swaps(row) == []
    assert generate_age_swaps(row) == []
    assert generate_gender_swaps(row) == []


def test_generators_handle_non_string_input():
    """A row dict whose input is not a string returns [] (defensive)."""
    row = {"id": "x", "input": 12345}
    assert generate_name_swaps(row) == []
    assert generate_zip_swaps(row) == []
    assert generate_age_swaps(row) == []
    assert generate_gender_swaps(row) == []


def test_generators_handle_empty_input():
    row = _row("")
    assert generate_name_swaps(row) == []
    assert generate_zip_swaps(row) == []
    assert generate_age_swaps(row) == []
    assert generate_gender_swaps(row) == []


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_generators_are_deterministic_across_repeats():
    """100 calls with the same input produce byte-identical output lists."""
    row = _row(
        "Greg Walsh lives at 11211 Brooklyn. He is 32 years old."
    )
    first = generate_swaps(row)
    for _ in range(100):
        again = generate_swaps(row)
        assert again == first


def test_swap_dataclass_is_frozen():
    """CounterfactualSwap is frozen — must not be mutable after construction."""
    swap = CounterfactualSwap(
        dim="name",
        from_token="A",
        to_token="B",
        canonical_input="A is here",
        counterfactual_input="B is here",
        eval_id_suffix="cf_name_b",
    )
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        swap.dim = "zip"  # type: ignore[misc]


def test_swap_eval_id_suffix_byte_safe_for_all_pairs():
    """Every B-M name in both tables produces a suffix that is filename-safe."""
    for white, black in _BM_NAME_PAIRS_MALE + _BM_NAME_PAIRS_FEMALE:
        row = _row(f"{white} applied.")
        swaps = generate_name_swaps(row)
        assert len(swaps) == 1
        suffix = swaps[0].eval_id_suffix
        assert all(c.isalnum() or c == "_" for c in suffix), (white, suffix)


def test_no_swap_when_dim_token_missing_but_other_dim_present():
    """A row that triggers ZIP but not name yields zero swaps from
    generate_name_swaps (cross-dim isolation)."""
    row = _row("ZIP 11211 has good schools.")
    assert generate_name_swaps(row) == []
    assert generate_age_swaps(row) == []
    assert generate_gender_swaps(row) == []
    # zip generator does fire.
    assert len(generate_zip_swaps(row)) == 1
