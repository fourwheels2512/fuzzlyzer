"""fuzzeval — shared eval infrastructure for topofuzz (and siblings)."""

from fuzzeval.cache import JudgeCache, LLMCache
from fuzzeval.costest import CostEstimate, CostLine, estimate
from fuzzeval.judge import BinaryScore, ContinuousScore, Judge, JudgeResult
from fuzzeval.stats import (
    BootstrapResult,
    percentile_bootstrap,
    percentile_cluster_bootstrap,
    wilson_ci,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "BinaryScore",
    "BootstrapResult",
    "ContinuousScore",
    "CostEstimate",
    "CostLine",
    "Judge",
    "JudgeCache",
    "JudgeResult",
    "LLMCache",
    "__version__",
    "estimate",
    "percentile_bootstrap",
    "percentile_cluster_bootstrap",
    "wilson_ci",
]
