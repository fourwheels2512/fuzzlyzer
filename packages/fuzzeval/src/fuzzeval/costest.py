"""Cost estimation: static-rate cold + warm-cache estimator.

Step 2 implements data shapes + a static-rate estimator. Real dry-run (actual
LLM probe calls via LiteLLM) is wired up in Weekend 2 step 6.

Per BLUEPRINT_v5plus.md §17a:
    estimate(sample_rows, judge_mode, n_total_rows, n_variants) -> CostEstimate
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostLine:
    name: str
    cold_estimate_usd: float
    cold_ci_half_width_usd: float
    warm_estimate_usd: float
    warm_ci_half_width_usd: float


@dataclass(frozen=True)
class CostEstimate:
    lines: list[CostLine]
    total_cold_usd: float
    total_warm_usd: float
    wallclock_estimate_s: float


# Blended (in+out) USD per 1K tokens. Verify before relying on these for
# real budgeting; step 6 replaces them with measured per-model rates.
_RATE_USD_PER_1K = {
    "anthropic/claude-opus-4-7": 0.045,
    "openai/gpt-5": 0.040,
    "google/gemini-2.5-pro": 0.020,
}
_DEFAULT_RATE = 0.040

# Static heuristics — replaced by measured values in step 6.
_TOKENS_PER_CALL = 2_500             # input + output, one agent
_AGENT_CALLS_PER_ROW = 4             # 4-agent chain (lending fixture)
_JUDGE_CALLS_PER_CELL = {"single": 1, "jury_of_3": 3}
_WARM_CACHE_HIT_RATE = 0.85
_WALLCLOCK_PER_CALL_S = 4.5
_SEMAPHORE_CONCURRENCY = 8           # asyncio.Semaphore(8) per BLUEPRINT §7


def _line_cost(n_calls: int, rate_per_1k: float) -> float:
    return n_calls * _TOKENS_PER_CALL * rate_per_1k / 1000.0


def estimate(
    sample_rows: list,
    judge_mode: str,
    n_total_rows: int,
    n_variants: int,
    agent_model: str = "anthropic/claude-opus-4-7",
    judge_model: str = "anthropic/claude-opus-4-7",
) -> CostEstimate:
    """Static-rate cost estimate. Step 6 replaces statics with measured rates."""
    if judge_mode not in _JUDGE_CALLS_PER_CELL:
        raise ValueError(
            f"judge_mode must be one of {list(_JUDGE_CALLS_PER_CELL)}; "
            f"got {judge_mode!r}"
        )
    if n_total_rows <= 0 or n_variants <= 0:
        raise ValueError("n_total_rows and n_variants must be positive")
    if not sample_rows:
        raise ValueError("sample_rows must be non-empty")

    agent_rate = _RATE_USD_PER_1K.get(agent_model, _DEFAULT_RATE)
    judge_rate = _RATE_USD_PER_1K.get(judge_model, _DEFAULT_RATE)
    judge_calls_per_cell = _JUDGE_CALLS_PER_CELL[judge_mode]

    n_cells = n_total_rows * n_variants
    n_agent_cold = n_cells * _AGENT_CALLS_PER_ROW
    n_agent_warm = int(n_agent_cold * (1.0 - _WARM_CACHE_HIT_RATE))
    n_judge_cold = n_cells * judge_calls_per_cell
    n_judge_warm = int(n_judge_cold * (1.0 - _WARM_CACHE_HIT_RATE))

    base_cold = _line_cost(n_agent_cold, agent_rate)
    base_warm = _line_cost(n_agent_warm, agent_rate)
    judge_cold = _line_cost(n_judge_cold, judge_rate)
    judge_warm = _line_cost(n_judge_warm, judge_rate)

    ci_band = 0.20  # ±20% until real variance data lands in step 6

    lines = [
        CostLine(
            name="base_run",
            cold_estimate_usd=base_cold,
            cold_ci_half_width_usd=base_cold * ci_band,
            warm_estimate_usd=base_warm,
            warm_ci_half_width_usd=base_warm * ci_band,
        ),
        CostLine(
            name="judge",
            cold_estimate_usd=judge_cold,
            cold_ci_half_width_usd=judge_cold * ci_band,
            warm_estimate_usd=judge_warm,
            warm_ci_half_width_usd=judge_warm * ci_band,
        ),
    ]

    wallclock = (
        (n_agent_cold + n_judge_cold) * _WALLCLOCK_PER_CALL_S / _SEMAPHORE_CONCURRENCY
    )

    return CostEstimate(
        lines=lines,
        total_cold_usd=base_cold + judge_cold,
        total_warm_usd=base_warm + judge_warm,
        wallclock_estimate_s=wallclock,
    )
