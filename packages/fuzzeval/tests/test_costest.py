import pytest

from fuzzeval.costest import CostEstimate, CostLine, estimate


def _sample_rows(n: int = 5) -> list[dict]:
    return [{"id": i, "text": "x" * 200} for i in range(n)]


def test_estimate_returns_well_formed_estimate():
    r = estimate(_sample_rows(), judge_mode="single", n_total_rows=50, n_variants=8)
    assert isinstance(r, CostEstimate)
    assert len(r.lines) == 2
    assert all(isinstance(line, CostLine) for line in r.lines)
    assert r.total_cold_usd > 0
    assert r.total_warm_usd > 0
    assert r.total_warm_usd < r.total_cold_usd
    assert r.wallclock_estimate_s > 0


def test_estimate_jury_more_expensive_than_single():
    s = estimate(_sample_rows(), "single", 50, 8)
    j = estimate(_sample_rows(), "jury_of_3", 50, 8)
    assert j.total_cold_usd > s.total_cold_usd


def test_estimate_scales_with_cells():
    small = estimate(_sample_rows(), "single", 10, 4)
    large = estimate(_sample_rows(), "single", 100, 16)
    assert large.total_cold_usd > small.total_cold_usd


def test_estimate_rejects_bad_judge_mode():
    with pytest.raises(ValueError):
        estimate(_sample_rows(), "jury_of_5", 50, 8)


def test_estimate_rejects_empty_sample_rows():
    with pytest.raises(ValueError):
        estimate([], "single", 50, 8)


def test_estimate_rejects_nonpositive_counts():
    with pytest.raises(ValueError):
        estimate(_sample_rows(), "single", 0, 8)
    with pytest.raises(ValueError):
        estimate(_sample_rows(), "single", 50, 0)


def test_estimate_is_deterministic_on_same_inputs():
    r1 = estimate(_sample_rows(), "single", 50, 8)
    r2 = estimate(_sample_rows(), "single", 50, 8)
    assert r1.total_cold_usd == r2.total_cold_usd
    assert r1.total_warm_usd == r2.total_warm_usd
    assert r1.wallclock_estimate_s == r2.wallclock_estimate_s


def test_unknown_model_falls_back_to_default_rate():
    r = estimate(_sample_rows(), "single", 50, 8, agent_model="nonexistent/model")
    assert r.total_cold_usd > 0  # no crash, fallback rate applied


def test_estimate_magnitude_regression():
    """Pin the static-rate output magnitude so a typo in a constant
    (`_TOKENS_PER_CALL`, `_RATE_USD_PER_1K`, `_AGENT_CALLS_PER_ROW`) doesn't
    silently 10x the bill. With current statics:
      50 rows × 8 variants × 4 agents × 2_500 tok × 0.045/1K = $180 base cold
      50 × 8 × 1 × 2_500 × 0.045/1K = $45 judge cold
    Total cold ≈ $225 ± $5 of float drift.
    If this assertion fails, audit the changed constant — don't just widen
    the band.
    """
    r = estimate(_sample_rows(), "single", 50, 8)
    assert 220.0 <= r.total_cold_usd <= 230.0, (
        f"cost magnitude drift detected: total_cold={r.total_cold_usd}"
    )
