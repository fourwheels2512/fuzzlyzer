"""Step 9: §9 metrics on top of one run's persisted ``judge_results``.

Public surface (kept small; step 11's ``topofuzz report`` is the only
in-tree consumer):

* :func:`read_score_matrix` — load a 2D ``(n_variants, n_evals)`` score
  matrix from one run's ``run.sqlite``, with variants ordered by
  ``canonical_hash`` lex (deterministic across re-runs) and evals
  ordered by insertion.  Refuses to load if the run's ``rubric_hash`` is
  empty (legacy / unknown) or to merge across different rubric hashes —
  averaging continuous judge scores under different rubrics is a real
  footgun.
* :func:`variance_across_orderings` — §9 metric #1 point estimate. The
  CI helper :func:`variance_across_orderings_ci` wraps
  :func:`fuzzeval.stats.percentile_cluster_bootstrap` with the canonical
  closure.
* :func:`error_correction_rate` — §9 metric #2. Pure-numpy, no CI.
  Caller-supplied ``upstream_wrong`` mask — v0.5's lending fixture has
  no per-upstream-agent ground-truth column, so the lending demo path
  reports ECR as degenerate (denominator < 5).  See the ``ecr.py``
  module docstring for the v0.6 plan.
* :func:`cohens_kappa_vs_truth` — §9 metric #3 point estimate + CI +
  ``collapse_flag``. Inline numpy: sklearn is intentionally NOT a
  dependency (handoff decision A; would pull ~50 MB transitive for
  two functions of ~10 LOC each).
* :func:`classify_run_status` — §A exit-code thresholds (1/25/50%) for
  the orchestrator to call before ``RunWriter.finalize``. Pure helper
  with no DB dependency so the metrics module owns the rule and the
  orchestrator owns the write — single source of truth per the step-9
  entry-decision answer.
"""

from __future__ import annotations

from topofuzz.metrics.ecr import ECRResult, error_correction_rate
from topofuzz.metrics.kappa import (
    KappaResult,
    balanced_accuracy,
    cohens_kappa,
    cohens_kappa_vs_truth,
)
from topofuzz.metrics.score_matrix import ScoreMatrixResult, read_score_matrix
from topofuzz.metrics.status import classify_run_status
from topofuzz.metrics.voo import (
    variance_across_orderings,
    variance_across_orderings_ci,
)

__all__ = [
    "ECRResult",
    "KappaResult",
    "ScoreMatrixResult",
    "balanced_accuracy",
    "classify_run_status",
    "cohens_kappa",
    "cohens_kappa_vs_truth",
    "error_correction_rate",
    "read_score_matrix",
    "variance_across_orderings",
    "variance_across_orderings_ci",
]
