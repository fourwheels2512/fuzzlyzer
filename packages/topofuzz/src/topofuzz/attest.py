# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""High-level attestation glue for topofuzz runs.

The cryptographic primitives live in :mod:`fuzzeval.attestation` (DSSE
PAE + envelope) and :mod:`fuzzeval.keys` (Ed25519 keygen / load). This
module assembles the v0.5 attestation payload, signs it, writes
``<run_dir>/attestation.json``, and provides the verify-side
``load_and_verify_attestation`` helper used by the ``topofuzz verify``
CLI and the SECURITY.md reference verifier.

Payload shape (DSSE inner payload, JCS-canonicalized before signing):

.. code-block:: json

    {
      "schema_version": 1,
      "run_id": "<id>",
      "manifest": { ...full manifest.json content... },
      "headline": {
        "status": "succeeded|degraded|failed|cost_cap_hit|interrupted",
        "voo_per_eval": [...] | null,
        "ecr": null,
        "kappa": null,
        "n_successful_cells": <int>,
        "n_total_cells":      <int>,
        "n_variants":         <int>,
        "n_evals":            <int>,
        "total_cost_usd":     <float>,
        "elapsed_s":          <float>
      },
      "run_sqlite_sha256": "<hex>",
      "signed_at": "<ISO8601 UTC>"
    }

``payloadType`` is ``application/vnd.topofuzz.run+json;v=0.5``.

Deferred-metrics rationale (v0.5 OSS):

* **voo** is computed as a point-only statistic per eval (no bootstrap
  CI) so the orchestrator's finalize step stays sub-second. Step 11's
  ``report.json.dsse`` will carry full bootstrap CIs.
* **ecr** is ``null`` in v0.5 because the lending fixture has no
  upstream-wrong signal (step-9 handoff line 32: degenerate-by-design).
  ECR lands in v0.6 once eval datasets surface an upstream signal.
* **kappa** is ``null`` in v0.5 because run.sqlite doesn't persist
  ground-truth labels for topology variant rollouts (κ vs. truth is a
  judge-calibration metric, not a per-run-status metric). Lands in v0.6
  when the eval-set schema gains a ``ground_truth`` column.

The headline dict is signed AS-IS (not hashed first), so a third-party
verifier can read voo/status/cell-counts off the attestation without
loading run.sqlite. The envelope payload is bounded (~few KB even at
50 evals × 8 variants).
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import rfc8785

from fuzzeval.attestation import (
    DSSEDecodeError,
    DSSESignatureError,
    DSSEVerifyError,
    Signer,
    dsse_sign,
    dsse_verify,
)

__all__ = [
    "ATTESTATION_FILENAME",
    "AttestationDiscoveryError",
    "PAYLOAD_SCHEMA_VERSION",
    "PAYLOAD_TYPE",
    "REPORT_FILENAME",
    "VerifyOutcome",
    "build_attestation_payload",
    "compute_ba_drop_for_attestation",
    "compute_findings",
    "compute_headline",
    "compute_influence_for_attestation",
    "compute_kappa_orig_vs_perm_for_attestation",
    "compute_noise_floor_for_attestation",
    "compute_verdict_flip_for_attestation",
    "compute_voo_for_attestation",
    "emit_attestation",
    "emit_run_attestation",
    "load_and_verify_attestation",
    "sha256_file",
]

PAYLOAD_TYPE: Final[str] = "application/vnd.topofuzz.run+json;v=0.5"
PAYLOAD_SCHEMA_VERSION: Final[int] = 1
ATTESTATION_FILENAME: Final[str] = "attestation.json"
# Step 15a (h): plain-JSON mirror of attestation.json's inner payload.
# Same bytes as ``base64decode(attestation.payload)``, just unwrapped, so
# step 22's recipient can ``jq '.headline'`` without DSSE-decoding skills.
REPORT_FILENAME: Final[str] = "report.json"


class AttestationDiscoveryError(FileNotFoundError):
    """The expected attestation bundle was not found at the given path."""


@dataclass(frozen=True)
class VerifyOutcome:
    """Result of :func:`load_and_verify_attestation`.

    ``ok`` is True iff every check the function ran passed. ``checks``
    is the ordered list of named checks the function executed (so the
    CLI can render a checklist-style PASS/FAIL summary); each entry is
    ``(name, ok, detail)``. ``payload`` is the verified payload dict
    (only populated when ``ok``).

    ``watermark`` is the human-readable banner the CLI prints. v0.5
    always emits ``"[DEV-KEY]"`` per BLUEPRINT_v5plus.md §11 line 304.
    """

    ok: bool
    checks: tuple[tuple[str, bool, str], ...]
    payload: dict[str, Any] | None
    watermark: str


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase-hex SHA-256 of ``path``'s contents.

    Streamed in 1 MiB chunks so a multi-GB ``run.sqlite`` doesn't blow
    memory.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_headline(
    *,
    status: str,
    n_successful_cells: int,
    n_total_cells: int,
    n_variants: int,
    n_evals: int,
    total_cost_usd: float,
    elapsed_s: float,
    voo_per_eval: list[dict[str, Any]] | None = None,
    verdict_flip_rate: dict[str, Any] | None = None,
    kappa_original_vs_permuted: float | None = None,
    balanced_accuracy_drop: dict[str, Any] | None = None,
    judge_failure_rate: float | None = None,
    judge_independence: dict[str, Any] | None = None,
    verdict_stability: float | None = None,
    confirmed_verdict_flip_rate: dict[str, Any] | None = None,
    voo_noise_floor: float | None = None,
    confirm_note: str | None = None,
) -> dict[str, Any]:
    """Assemble the headline dict.

    Step 15a fields (BLUEPRINT §14):

    * ``voo_per_eval`` — list of ``{eval_id, point, ci_lower, ci_upper,
      degenerate_flag}`` per BLUEPRINT §14 sub-deliverable (b).
    * ``verdict_flip_rate`` — ``{point, ci_lower, ci_upper,
      degenerate_flag, numerator, denominator, wilson_ci_lower,
      wilson_ci_upper, wilson_conf_level}`` per (c). The last five
      fields are F1 additions (numerator/denominator + Wilson CI on
      the binomial proportion + the conf level the Wilson CI was
      computed at).
    * ``kappa_original_vs_permuted`` — scalar float per (d).
    * ``balanced_accuracy_drop`` — ``{original, mean_permuted, drop}`` per (e).
    * ``judge_failure_rate`` — cell-level §8 rate in [0, 1] per (f).
    * ``judge_independence`` — full §8 dict (judge_protocol, judges,
      vendor_overlap, vendor_disjoint, self_preference_warning,
      vendor_call_failure_rate) per (g).

    ECR and κ-vs-truth remain hard-coded ``null`` in v0.5 (see module
    docstring). ``kappa_original_vs_permuted`` is the *new* κ statistic
    introduced by step 15a (d); it is intentionally distinct from the
    legacy ``kappa`` field, which references κ-vs-truth.

    Noise-floor fields (the ``topofuzz confirm`` pass):

    * ``verdict_stability`` — fraction of eval_ids whose K reference
      repeats agree on the verdict (the AFL "stability %" negative
      control). Scalar in [0, 1].
    * ``confirmed_verdict_flip_rate`` — ``{point, ci_lower, ci_upper,
      confirmed, flaky, unstable_baseline, missing_baseline,
      judge_failure_pairs, n_screened_pairs, degenerate_flag,
      wilson_ci_lower, wilson_ci_upper, wilson_conf_level}``: the
      noise-corrected flip rate (majority-vote-vs-majority-vote). The
      Wilson keys are F1 additions, computed on the (confirmed,
      confirmed + flaky) binomial proportion.
    * ``voo_noise_floor`` — VOO over the K reference repeats of the
      original variant; the updated §22 gate input is ``voo_ci_lower >
      voo_noise_floor`` instead of ``> 0`` (no in-tree caller applies
      the gate yet — the §22 CEO-PDF site is not built).
    * ``confirm_note`` — a short human-readable string set ONLY when the
      noise-floor metrics were suppressed for a recoverable reason (a
      partial confirm pass, reconstruction drift, etc.). The recipient
      shows this instead of a number. ``None`` in the normal case
      (metrics populated, or no confirm pass run at all).

    The three metric fields default to ``None`` and are populated only
    when the run has a complete ``confirm`` pass (``confirm.json`` sidecar
    present, ``partial: false``, ``n_unmatched_candidates: 0``). An absent
    sidecar leaves them ``null`` with ``confirm_note=null`` — not a crash.
    A partial pass leaves them ``null`` with a non-null ``confirm_note``.

    All step-15a fields default to ``None`` so pre-step-15a callers
    (and the v0.5 in-tree callsites that haven't been migrated yet)
    continue producing well-shaped headlines without the new metrics.
    """
    return {
        "status": status,
        "voo_per_eval": voo_per_eval,
        "ecr": None,
        "kappa": None,
        "verdict_flip_rate": verdict_flip_rate,
        "kappa_original_vs_permuted": kappa_original_vs_permuted,
        "balanced_accuracy_drop": balanced_accuracy_drop,
        "judge_failure_rate": judge_failure_rate,
        "judge_independence": judge_independence,
        "verdict_stability": verdict_stability,
        "confirmed_verdict_flip_rate": confirmed_verdict_flip_rate,
        "voo_noise_floor": voo_noise_floor,
        "confirm_note": confirm_note,
        "n_successful_cells": int(n_successful_cells),
        "n_total_cells": int(n_total_cells),
        "n_variants": int(n_variants),
        "n_evals": int(n_evals),
        "total_cost_usd": float(total_cost_usd),
        "elapsed_s": float(elapsed_s),
    }


def build_attestation_payload(
    *,
    run_id: str,
    manifest: dict[str, Any],
    headline: dict[str, Any],
    run_sqlite_sha256: str,
    signed_at: str | None = None,
    findings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the inner DSSE payload dict (before JCS canonicalization).

    ``signed_at`` defaults to the current UTC time formatted as
    ``YYYY-MM-DDTHH:MM:SS.ffffffZ``. Callers (mostly tests) can pass a
    fixed string for determinism.

    ``findings`` is an optional dict of post-run analytics that don't
    belong in ``headline`` (they may be empty, may carry CIs, may grow
    in v0.6+). Step 15 (§6a per-agent influence) is the first occupant.
    When omitted, defaults to an empty dict so consumers can always
    read ``payload['findings']`` without a KeyError.
    """
    if signed_at is None:
        # ISO-8601 UTC with 'Z' suffix. Using datetime.now(tz=UTC) (not
        # the deprecated utcnow()) gives true microsecond precision —
        # the previous float-arithmetic implementation could collapse
        # microseconds to 0 near integer-second boundaries.
        now = _datetime.datetime.now(tz=_datetime.timezone.utc)
        signed_at = now.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "run_id": run_id,
        "manifest": manifest,
        "headline": headline,
        "findings": dict(findings) if findings else {},
        "run_sqlite_sha256": run_sqlite_sha256,
        "signed_at": signed_at,
    }


# ---------------------------------------------------------------------------
# Sign
# ---------------------------------------------------------------------------


def emit_attestation(
    payload: dict[str, Any],
    signer: Signer,
    out_path: Path,
    *,
    report_path: Path | None = None,
) -> Path:
    """JCS-canonicalize ``payload``, DSSE-sign, write to ``out_path``.

    The on-disk envelope is the raw bytes returned by
    :func:`fuzzeval.attestation.dsse_sign` (compact, key-sorted JSON).
    No newline is appended — the file is exactly the bytes a verifier
    would re-read with ``Path.read_bytes()``.

    Step 15a (h) — ``report_path``: when supplied, also writes the
    JCS-canonical ``rfc8785.dumps(payload)`` bytes (the same payload
    that is base64-wrapped inside the DSSE envelope) to
    ``report_path``. This is the *plain-JSON mirror* DoD #4 promises:
    grep-extractable without DSSE-decoding skills, but byte-identical
    at the payload level to ``base64decode(attestation.payload)`` by
    construction.

    **Audit Pass-1 CR-M3 / RT-M2 hardening — write ordering:**

    1. Build canonical payload bytes.
    2. DSSE-sign and write ``attestation.json`` to its final path.
    3. Atomic-rename ``report.json.tmp`` → ``report.json``.

    Rationale: if ``dsse_sign`` raises (signer revoked, etc.), step 2
    never lands AND step 3 never runs, so no orphan mirror persists
    on disk. If step 2 succeeds but step 3 races with a concurrent
    reader, the atomic rename guarantees readers see either the prior
    mirror or the new one, never a half-written one. The mirror's
    presence is therefore a strict subset of the envelope's presence:
    consumers running ``jq '.headline' report.json`` MUST also verify
    via the envelope, because the mirror without an envelope is
    structurally impossible in a clean emit.

    The reverse order (mirror-first) leaves a window where a signed
    envelope failure persists an orphan mirror that recipients would
    quote-and-trust — exactly the silent-failure vector the audit
    flagged.
    """
    canonical = rfc8785.dumps(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    envelope = dsse_sign(canonical, PAYLOAD_TYPE, signer)
    out_path.write_bytes(envelope)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via tmp + replace. ``Path.replace`` is atomic on
        # POSIX (rename(2)) and on Windows NTFS (since Python 3.3).
        # The tmp filename appends ``.tmp`` AFTER the existing suffix
        # so a concurrent ``glob("*.json")`` sweep doesn't pick it up.
        tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
        tmp_path.write_bytes(canonical)
        tmp_path.replace(report_path)
    return out_path


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def _discover_bundle(run_dir: Path | None, bundle: Path | None) -> Path:
    """Return the bundle path to read; raise if neither resolves."""
    if bundle is not None:
        if not bundle.exists():
            raise AttestationDiscoveryError(
                f"--bundle path does not exist: {bundle}"
            )
        return bundle
    if run_dir is None:
        raise AttestationDiscoveryError(
            "neither run_dir nor bundle supplied"
        )
    candidate = run_dir / ATTESTATION_FILENAME
    if not candidate.exists():
        raise AttestationDiscoveryError(
            f"no {ATTESTATION_FILENAME} in {run_dir} — "
            f"run was not signed (no dev key) or the file was moved"
        )
    return candidate


def load_and_verify_attestation(
    *,
    run_dir: Path | None,
    bundle: Path | None,
    public_key_pem: bytes,
    payload_type_expected: str = PAYLOAD_TYPE,
) -> VerifyOutcome:
    """Full verify pipeline used by ``topofuzz verify``.

    Checks, in order:

    1. **discover**: locate ``attestation.json`` (or ``--bundle`` path).
    2. **envelope_signature**: dsse_verify decodes + checks Ed25519
       signature. Returns a :class:`fuzzeval.attestation.VerifiedPayload`
       carrying the AUTHENTICATED bytes — we use those bytes for all
       subsequent checks, never re-parsing the envelope (per DSSE v1.0
       spec MUST).
    3. **payload_type**: envelope's payloadType matches
       ``payload_type_expected``.
    4. **jcs_round_trip**: re-canonicalize the verified payload dict
       via JCS; bytes must match the authenticated payload bytes. This
       catches signers that signed non-canonical bytes (the JCS
       guarantee is meaningless without this).
    5. **schema_version**: payload's ``schema_version`` is exactly 1.
    6. **manifest_match** (only when ``run_dir`` supplied and
       ``bundle`` is None — i.e., we're verifying an in-place run):
       ``run_dir/manifest.json`` MUST exist and its JCS must match the
       envelope's ``manifest`` field. Missing-file is a FAIL (defends
       against the replay attack of dropping an orphan attestation into
       a foreign run dir).
    7. **run_sqlite_sha256_match** (same gate as 6, plus the envelope's
       ``run_sqlite_sha256`` is non-empty): ``run.sqlite`` MUST exist
       and re-hash to the signed value.

    ``--bundle`` semantics (when ``bundle is not None``): file-vs-
    envelope checks 6/7 are skipped entirely. The bundle proves the
    producer signed a particular payload; without a run_dir context,
    nothing else is checked. Callers wanting full run-binding verify
    should pass ``run_dir`` and rely on auto-discovery of the
    co-located ``attestation.json``.

    Any failure short-circuits to ``ok=False`` but the function still
    returns a ``VerifyOutcome`` so the CLI can print the full checklist.
    """
    checks: list[tuple[str, bool, str]] = []
    # When --bundle is supplied, file-vs-envelope checks against a
    # run_dir don't make sense (the bundle could be unrelated to that
    # dir). Force run_dir=None for the in-function semantics so steps
    # 6/7 are skipped cleanly.
    if bundle is not None:
        run_dir = None

    # 1. Discover the bundle.
    try:
        bundle_path = _discover_bundle(run_dir, bundle)
    except AttestationDiscoveryError as e:
        checks.append(("discover", False, str(e)))
        return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
    checks.append(("discover", True, f"using {bundle_path}"))

    envelope_bytes = bundle_path.read_bytes()

    # 2. Envelope decode + signature. dsse_verify returns a
    # VerifiedPayload carrying the authenticated payload_bytes and the
    # envelope's payloadType — we use those for steps 3-4 rather than
    # re-parsing the envelope (DSSE v1.0 spec MUST).
    try:
        verified = dsse_verify(envelope_bytes, public_key_pem)
    except DSSESignatureError as e:
        checks.append(("envelope_signature", False, str(e)))
        return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
    except DSSEDecodeError as e:
        checks.append(("envelope_decode", False, str(e)))
        return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
    except DSSEVerifyError as e:  # pragma: no cover — defensive parent catch
        checks.append(("envelope", False, str(e)))
        return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
    checks.append(("envelope_signature", True, "ed25519 ok"))

    payload = verified.payload

    # 3. payloadType match. Read from the verified payload (which was
    # authenticated together with the bytes) rather than re-parsing.
    if verified.payload_type != payload_type_expected:
        checks.append(
            (
                "payload_type",
                False,
                f"expected {payload_type_expected!r}, "
                f"got {verified.payload_type!r}",
            )
        )
        return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
    checks.append(("payload_type", True, verified.payload_type))

    # 4. JCS round-trip. Re-canonicalize the verified dict and compare
    # to the AUTHENTICATED payload bytes (verified.payload_bytes), not
    # to a re-parsed envelope field.
    canonical = rfc8785.dumps(payload)
    if canonical != verified.payload_bytes:
        checks.append(
            (
                "jcs_round_trip",
                False,
                f"signed payload was not JCS-canonical "
                f"(signed {len(verified.payload_bytes)}B, "
                f"canonical {len(canonical)}B)",
            )
        )
        return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
    checks.append(("jcs_round_trip", True, f"{len(canonical)}B canonical"))

    # 5. schema_version.
    if payload.get("schema_version") != PAYLOAD_SCHEMA_VERSION:
        checks.append(
            (
                "schema_version",
                False,
                f"expected {PAYLOAD_SCHEMA_VERSION}, got "
                f"{payload.get('schema_version')!r}",
            )
        )
        return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
    checks.append(("schema_version", True, str(PAYLOAD_SCHEMA_VERSION)))

    # 6 + 7: file-vs-envelope checks (only when verifying in-place — i.e.
    # run_dir supplied AND no --bundle override). REQUIRE the on-disk
    # files; their absence is a FAIL, not a skip — that's the fix for
    # the orphan-attestation replay vector.
    if run_dir is not None:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            checks.append(
                (
                    "manifest_match",
                    False,
                    f"no manifest.json in {run_dir} "
                    f"(attestation may have been moved between run dirs)",
                )
            )
            return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
        try:
            on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            checks.append(("manifest_match", False, f"on-disk read failed: {e}"))
            return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
        envelope_manifest = payload.get("manifest")
        if rfc8785.dumps(on_disk) != rfc8785.dumps(envelope_manifest):
            checks.append(
                (
                    "manifest_match",
                    False,
                    "on-disk manifest.json differs from signed manifest",
                )
            )
            return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
        checks.append(("manifest_match", True, "JCS bytes match"))

        sqlite_path = run_dir / "run.sqlite"
        envelope_sha = payload.get("run_sqlite_sha256")
        if envelope_sha:
            # Signed hash is non-empty → run.sqlite MUST exist and match.
            # If it's gone, that's a real binding-failure (the producer
            # signed a hash of a file that's now missing).
            if not sqlite_path.exists():
                checks.append(
                    (
                        "run_sqlite_sha256_match",
                        False,
                        f"envelope signed a non-empty sha256 but "
                        f"run.sqlite is missing from {run_dir}",
                    )
                )
                return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
            on_disk_sha = sha256_file(sqlite_path)
            if on_disk_sha != envelope_sha:
                checks.append(
                    (
                        "run_sqlite_sha256_match",
                        False,
                        f"on-disk sha256 {on_disk_sha} != signed {envelope_sha}",
                    )
                )
                return VerifyOutcome(False, tuple(checks), None, "[DEV-KEY]")
            checks.append(
                ("run_sqlite_sha256_match", True, on_disk_sha[:12] + "…")
            )
        else:
            # Empty signed hash means the producer didn't have a sqlite at
            # finalize time (aborted run with no DB). The check is then
            # vacuous but we record it so the checklist is consistent.
            checks.append(
                ("run_sqlite_sha256_match", True, "empty sha (no run.sqlite)")
            )

    return VerifyOutcome(True, tuple(checks), payload, "[DEV-KEY]")


# ---------------------------------------------------------------------------
# Orchestrator-facing helpers
# ---------------------------------------------------------------------------


def compute_voo_for_attestation(
    run_dir: Path,
    run_id: str,
    *,
    n_resamples: int = 5000,
    conf: float = 0.95,
    seed: int | None = 0,
) -> list[dict[str, Any]] | None:
    """Best-effort per-eval ``variance_across_orderings`` for the headline.

    Returns ``None`` (and never raises) if the score matrix can't be
    read — empty rubric_hash, no successful cells, numpy missing, etc.
    The orchestrator emits the attestation regardless; the headline
    just gets ``voo_per_eval=None`` and the verifier accepts that.

    The per-eval point estimate is ``np.var(matrix[:, j], ddof=0)`` for
    each eval ``j`` — matching the inner statistic of
    :func:`topofuzz.metrics.variance_across_orderings`. Per-eval cells
    with NaN (failed cell in that variant×eval) propagate to
    ``point=None`` with ``degenerate_flag=True`` so the JSON stays
    well-typed.

    Step 15a (b) — each per-eval entry also carries
    ``{ci_lower, ci_upper, degenerate_flag}``. The CI is a percentile
    bootstrap on the **variant axis** (``n_resamples=5000`` per §9):
    we resample the per-eval column with replacement and recompute
    ``np.var(.., ddof=0)`` on each draw. The §9 paired cluster
    bootstrap is the aggregate-statistic shape (outer=eval, inner=
    variant) — its math doesn't apply to a single-eval column; for
    per-eval CIs the single-axis bootstrap on the variants in that
    eval is the right specialization. Degenerate columns (constant
    values across variants, or any NaN) get ``ci_lower=ci_upper=None``
    and ``degenerate_flag=True``.

    F3 MAJOR (2026-05-16) jury-aware POINT estimate: the per-eval
    ``point`` value dispatches through
    :func:`topofuzz.metrics.per_eval_variance` — on a v6 jury run this
    is the per-vendor continuous variance, escaping the Bernoulli ceiling
    that saturated the 2D ``np.var(matrix, ddof=0)`` closure. The CI
    bootstrap still resamples the binarised majority column ``matrix[:,
    j]`` — that's a documented minimum-scope deferral (re-signing the
    bundle would require a v6 schema change too) tracked as a v0.6
    POLISH item in ``packages/topofuzz/BACKLOG.md``. The asymmetry can
    surface as a non-zero ``point`` with a degenerate ``ci_lower/upper``
    on jury runs with a constant majority but wobbling continuous votes
    — by design until the v0.6 follow-up lands.

    ``seed`` defaults to ``0`` for byte-stable re-signs of the same
    run (matches the §15 influence path).

    The ImportError catches are split by intent: a missing ``numpy`` is
    expected on a ``topofuzz[verify]``-only install (no signing, voo
    irrelevant) and degrades silently to ``None``. A missing
    ``topofuzz.metrics`` would mean the installed wheel is broken — we
    log a WARNING and still return ``None`` so the run doesn't crash,
    but the operator sees the problem.
    """
    log = logging.getLogger(__name__)
    try:
        import numpy as np
    except ImportError:
        # Numpy not installed: expected on verify-only installs. Silent
        # None — the producer wasn't going to compute voo anyway.
        return None
    try:
        import sqlite3

        from topofuzz.metrics import read_score_matrix
    except ImportError as e:
        # topofuzz.metrics import failure means the wheel is broken
        # (a v0.5 contract violation, not a v0.5-design choice).
        # Logging WARNING here so a corrupted install doesn't silently
        # ship unsigned-voo attestations forever.
        log.warning(
            "topofuzz.metrics unavailable; voo skipped in attestation: %s", e
        )
        return None
    try:
        result = read_score_matrix(
            run_dir,
            run_id=run_id,
            allow_empty_rubric_hash=True,
        )
    except (FileNotFoundError, ValueError, OSError, sqlite3.DatabaseError):
        # sqlite3.DatabaseError covers empty / schemaless / corrupted DBs
        # (OperationalError "no such table: runs" is the common one for
        # an aborted run where the writer didn't get to commit).
        return None
    matrix = result.matrix
    if matrix.size == 0 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        return None

    # F3 MAJOR dispatch: on jury runs, compute the per-eval point estimate
    # over per-vendor continuous scores instead of the Bernoulli-bounded
    # majority. The 2D ``matrix`` (majority votes) still drives the
    # variant-axis bootstrap CI; the per-eval point follows the same
    # dispatch as the report bar SVG so both surfaces agree.
    from topofuzz.metrics import per_eval_variance

    per_eval_point_vector = per_eval_variance(matrix, result.per_vendor_matrix)
    n_variants = matrix.shape[0]
    alpha = (1.0 - conf) / 2.0
    out: list[dict[str, Any]] = []
    for j, eid in enumerate(result.eval_ids):
        column = matrix[:, j]
        finite = np.isfinite(column)
        if not finite.all() or column.size == 0:
            # Any NaN in the column → §9 point is NaN; mark degenerate.
            # We don't impute on the attest path (the loader is the
            # right place to surface missing cells; per the §9 docstring).
            out.append({
                "eval_id": eid,
                "point": None,
                "ci_lower": None,
                "ci_upper": None,
                "degenerate_flag": True,
            })
            continue
        # Jury dispatch: per_eval_point_vector[j] is already the
        # per-vendor-averaged variance for jury runs; for single-judge
        # runs it equals ``np.var(column, ddof=0)`` exactly.
        point_arr = per_eval_point_vector[j]
        if not np.isfinite(point_arr):
            # The jury path can return NaN for an eval where every
            # vendor's roster has fewer than 2 non-NaN variants (M1
            # guard); mark degenerate rather than signing a NaN.
            out.append({
                "eval_id": eid,
                "point": None,
                "ci_lower": None,
                "ci_upper": None,
                "degenerate_flag": True,
            })
            continue
        point = float(point_arr)
        if n_variants < 2 or bool(np.all(column == column[0])):
            # Constant column or singleton variant axis: variance is
            # structurally 0 (or trivially defined), bootstrap is a
            # point — mark degenerate so the report doesn't draw a
            # zero-width error bar.
            out.append({
                "eval_id": eid,
                "point": point,
                "ci_lower": None,
                "ci_upper": None,
                "degenerate_flag": True,
            })
            continue
        # Audit RT-M7: per-column RNG seed so each column's bootstrap
        # is byte-stable in isolation. A single shared RNG would mean
        # column j's CI depends on the entire history of columns
        # 0..j-1 — adding or reordering eval columns would silently
        # shift every subsequent column's signed CI across the rc1→GA
        # boundary, defeating step 13's release pipeline contract.
        col_seed: int | None = None if seed is None else int(seed) + j
        col_rng = np.random.default_rng(col_seed)
        indices = col_rng.integers(
            0, n_variants, size=(n_resamples, n_variants)
        )
        resampled = column[indices]
        variances = np.var(resampled, axis=1, ddof=0)
        out.append({
            "eval_id": eid,
            "point": point,
            "ci_lower": float(np.quantile(variances, alpha)),
            "ci_upper": float(np.quantile(variances, 1.0 - alpha)),
            "degenerate_flag": False,
        })
    return out


def compute_influence_for_attestation(
    run_dir: Path,
    run_id: str,
    manifest: dict[str, Any],
    *,
    seed: int | None = 0,
) -> list[dict[str, Any]] | None:
    """Best-effort §6a per-agent influence for the attestation findings.

    Returns ``None`` (and never raises) if any of the following are
    true: numpy missing, ``topofuzz.metrics`` unavailable, manifest
    has no ``variant_orderings`` (old run), score matrix can't be
    read, or no agent has a computable influence row. The orchestrator
    emits the attestation regardless — the only signal is the absence
    of a ``findings.per_agent_influence`` field.

    The output shape matches BLUEPRINT_v5plus.md §6a DoD #12:

    .. code-block:: json

        [
          {"node_id": "A", "influence": 0.67, "ci_lower": 0.51,
           "ci_upper": 0.79, "degenerate": false},
          ...
        ]

    Ranked descending by signed influence (the
    :class:`topofuzz.metrics.influence.AgentInfluence` row order); rows
    with ``degenerate=true`` carry ``influence=null`` /
    ``ci_lower=null`` / ``ci_upper=null`` plus a short
    ``degenerate_reason`` string.

    ``seed`` defaults to ``0`` so the signed attestation is
    byte-stable across re-signs of the same run. Callers wanting a
    fresh RNG can pass ``None`` but lose the rc1→GA byte-identical
    promise that step 13's release pipeline depends on.
    """
    log = logging.getLogger(__name__)
    if seed is None:
        log.warning(
            "compute_influence_for_attestation called with seed=None; the "
            "signed payload will not be byte-stable across re-signs. Pass "
            "seed=0 to preserve the rc1→GA byte-identical contract."
        )
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        import sqlite3

        from topofuzz.metrics import (
            OrderingAlignmentError,
            align_orderings_to_matrix,
            compute_influence,
            read_score_matrix,
        )
    except ImportError as e:
        log.warning(
            "topofuzz.metrics unavailable; influence skipped in attestation: %s",
            e,
        )
        return None

    try:
        sm = read_score_matrix(
            run_dir,
            run_id=run_id,
            allow_empty_rubric_hash=True,
        )
    except (FileNotFoundError, ValueError, OSError, sqlite3.DatabaseError) as e:
        log.warning(
            "score matrix unreadable; influence skipped in attestation: %s",
            e,
        )
        return None

    matrix = sm.matrix
    if matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        return None

    # Align variant_orderings (manifest insertion order) with the score
    # matrix axis (canonical_hash lex). Strict alignment (audit RT-B-1
    # / RT-B-2): duplicate hashes, variant-count mismatches, and unknown
    # hashes all surface as a logged warning and a missing
    # ``findings.per_agent_influence`` field — refusing is safer than
    # signing a wrong-axis chart.
    try:
        aligned = align_orderings_to_matrix(manifest, sm.variant_ids)
    except OrderingAlignmentError as e:
        if e.reason != "missing":
            # 'missing' is the expected pre-step-15 run case; everything
            # else is a real signal the operator should see.
            log.warning(
                "influence skipped in attestation (%s): %s", e.reason, e
            )
        return None

    try:
        comp = compute_influence(matrix, aligned, seed=seed)
    except (ValueError, ImportError) as e:
        log.warning(
            "influence computation failed in attestation path: %s", e
        )
        return None

    out: list[dict[str, Any]] = []
    for row in comp.results:
        entry: dict[str, Any] = {
            "node_id": row.node_id,
            "influence": (
                None if row.influence is None else float(row.influence)
            ),
            "ci_lower": (
                None if row.ci_lower is None else float(row.ci_lower)
            ),
            "ci_upper": (
                None if row.ci_upper is None else float(row.ci_upper)
            ),
            "degenerate": bool(row.degenerate),
        }
        if row.degenerate_reason:
            entry["degenerate_reason"] = row.degenerate_reason
        out.append(entry)
    return out if out else None


def _resolve_original_variant_index(
    manifest: dict[str, Any],
    variant_ids: tuple[str, ...],
) -> int | None:
    """Locate the §6 "original" variant's row index in the score matrix.

    Per the locked step-15a decision: the original is
    ``manifest.variant_canonical_hashes[0]`` (kickoff insertion order).
    The score-matrix variant axis is canonical_hash lex (step 9 handoff
    decision C), so we find the index of that hash in ``variant_ids``.

    **Audit Pass-1 RT-B1 / RT-B2 hardening:** refuses to resolve when
    the manifest hash list contains duplicates (``.index()`` would
    silently pick the first lex-sorted occurrence — a wrong-axis vector
    that gets signed with DSSE confidence) or when the manifest's
    set-of-hashes is not a subset of the score-matrix axis (signals a
    canonical_hash recipe drift between kickoff and finalize). Both
    refusals log at WARNING so a corrupted run shows in operator logs.

    Returns ``None`` (and never raises) on any malformed input. The
    caller (attest path) treats ``None`` as "metric not computable"
    and skips with a stable null in the headline.
    """
    log = logging.getLogger(__name__)
    if not isinstance(manifest, dict):
        return None
    hashes = manifest.get("variant_canonical_hashes")
    if not isinstance(hashes, list) or not hashes:
        return None
    # Audit RT-B1: duplicate hashes in the manifest list mean the
    # `[0]`-resolved original could be ambiguous against a lex-sorted
    # axis that also contains the duplicate. Refuse rather than guess.
    if len(set(hashes)) != len(hashes):
        log.warning(
            "manifest.variant_canonical_hashes has duplicate entries; "
            "headline metrics requiring original_index skipped"
        )
        return None
    original_hash = hashes[0]
    if not isinstance(original_hash, str):
        return None
    # Audit Pass-2 M1 reversal: only require that the ORIGINAL hash is
    # in the score-matrix axis. The original pass-1 fix checked
    # ``set(hashes).issubset(set(variant_ids))`` to catch
    # canonical_hash drift, but ``read_score_matrix`` filters cells
    # with ``failure_reason IS NOT NULL`` (score_matrix.py:153-162),
    # so a variant whose cells all failed is dropped from
    # ``sm.variant_ids``. In the legitimate "12-variant run, 11
    # succeeded, 1 fully aborted" case the manifest is a SUPERSET of
    # the matrix axis — that's normal partial-success behavior, not
    # drift. The strict subset gate killed verdict_flip / κ / BA-drop
    # on every such run.
    #
    # The original-only check still catches the wrong-axis vector
    # RT-B2 was concerned with: if the producer's canonical_hash
    # recipe drifted between kickoff and finalize, the manifest's
    # ``[0]`` will not match anything in the matrix and
    # ``.index()`` returns ValueError → None below.
    if original_hash not in variant_ids:
        log.warning(
            "manifest.variant_canonical_hashes[0]=%r not present in "
            "score-matrix variant_ids; headline metrics skipped "
            "(canonical_hash drift or the original variant fully "
            "failed all cells)",
            original_hash,
        )
        return None
    # Surface (but don't block on) the inverse signal: matrix
    # variant_ids that aren't in the manifest mean a cell was
    # persisted under a hash the kickoff didn't declare —
    # genuinely a drift signal that the operator should investigate.
    extra_in_matrix = set(variant_ids) - set(hashes)
    if extra_in_matrix:
        log.warning(
            "score-matrix variant_ids contains hashes absent from "
            "manifest.variant_canonical_hashes: %r; continuing with "
            "manifest-declared original",
            sorted(extra_in_matrix),
        )
    return variant_ids.index(original_hash)


def _extract_ground_truth_binary(
    manifest: dict[str, Any],
    eval_ids: tuple[str, ...],
) -> "Any":
    """Return a numpy 0/1 array aligned with ``eval_ids``, or None.

    Reads ``manifest.eval_ground_truth`` (parallel to
    ``manifest.eval_ids``) and re-indexes it onto the score-matrix eval
    axis (``eval_ids``) BY eval_id.

    The two axes are NOT positionally aligned: the manifest lists evals
    in plan order, but :func:`topofuzz.metrics.read_score_matrix`
    discovers its eval axis in judge-row insertion order — and the
    concurrent runner does not persist judge rows in plan order. An
    earlier version of this function asserted positional alignment and
    bailed on every real run; alignment must be by eval_id.

    Returns ``None`` (BA-drop then degenerate) when: the manifest
    fields are missing/malformed, the manifest's own ``eval_ids`` /
    ``eval_ground_truth`` arrays disagree in length, the manifest has
    duplicate eval_ids, a score-matrix eval_id is absent from the
    manifest, any ground-truth entry needed for ``eval_ids`` is null,
    or any value is not 0/1.
    """
    log = logging.getLogger(__name__)
    if not isinstance(manifest, dict):
        return None
    gt_raw = manifest.get("eval_ground_truth")
    eids_manifest = manifest.get("eval_ids")
    if not isinstance(gt_raw, list) or not isinstance(eids_manifest, list):
        return None
    if len(gt_raw) != len(eids_manifest):
        # The manifest's two parallel arrays must agree in length — a
        # producer bug if they don't. ERROR so a regression surfaces in
        # operator logs rather than silently degrading BA-drop.
        log.error(
            "manifest.eval_ids (len %d) and manifest.eval_ground_truth "
            "(len %d) disagree — BA-drop skipped (producer bug)",
            len(eids_manifest), len(gt_raw),
        )
        return None
    if len(set(eids_manifest)) != len(eids_manifest):
        # Duplicate eval_ids make the eval_id→ground_truth map ambiguous.
        # Like the length-mismatch and missing-eval cases, this is a
        # producer bug — ERROR so it surfaces in operator logs.
        log.error(
            "manifest.eval_ids has duplicate entries; BA-drop skipped "
            "(cannot unambiguously align ground truth by eval_id)"
        )
        return None
    gt_by_eid = dict(zip(eids_manifest, gt_raw))
    missing = [eid for eid in eval_ids if eid not in gt_by_eid]
    if missing:
        # A score-matrix eval with no manifest ground-truth entry. This
        # is the structural-refactor signal RT-M6 cared about — ERROR so
        # it doesn't silently degrade BA-drop to "no ground-truth
        # signal" on every signed attestation thereafter.
        log.error(
            "score-matrix eval_ids %r absent from manifest.eval_ids — "
            "BA-drop skipped (structural bug, not a per-run condition)",
            missing,
        )
        return None
    aligned = [gt_by_eid[eid] for eid in eval_ids]
    if any(v is None for v in aligned):
        # At least one eval on the score-matrix axis has no ground-truth
        # label. Genuine "this run has no ground-truth signal" condition.
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        arr = np.asarray(aligned, dtype=int)
    except (TypeError, ValueError):
        return None
    if not bool(np.all((arr == 0) | (arr == 1))):
        return None
    return arr


def compute_verdict_flip_for_attestation(
    run_dir: Path,
    run_id: str,
    manifest: dict[str, Any],
    *,
    n_resamples: int = 5000,
    conf: float = 0.95,
    seed: int | None = 0,
) -> dict[str, Any] | None:
    """§14 step 15a (c) attestation wire-up for verdict-flip rate.

    Returns ``{point, ci_lower, ci_upper, degenerate_flag}`` or ``None``
    when the metric isn't computable (numpy missing, score matrix
    unreadable, single-variant fixture, original-hash not in matrix axis).

    Verdict-flip is the THE pull-quote number for step 22's CEO outreach
    PDF (BLUEPRINT line 416). The "original" variant is identified by
    ``manifest.variant_canonical_hashes[0]`` (the §6 contract).
    """
    log = logging.getLogger(__name__)
    try:
        import numpy as np  # noqa: F401 — numpy presence gate
    except ImportError:
        return None
    try:
        import sqlite3

        from topofuzz.metrics import (
            compute_verdict_flip_rate,
            read_score_matrix,
            wilson_score_interval,
        )
    except ImportError as e:
        log.warning(
            "topofuzz.metrics unavailable; verdict_flip skipped in attestation: %s",
            e,
        )
        return None
    try:
        sm = read_score_matrix(
            run_dir,
            run_id=run_id,
            allow_empty_rubric_hash=True,
        )
    except (FileNotFoundError, ValueError, OSError, sqlite3.DatabaseError):
        return None
    if sm.matrix.size == 0 or sm.matrix.shape[0] < 2 or sm.matrix.shape[1] < 1:
        return None
    orig_idx = _resolve_original_variant_index(manifest, sm.variant_ids)
    if orig_idx is None:
        return None
    try:
        # Pass the 3-state class_matrix when present (v7+ runs) so the
        # headline flip rate counts ONLY clear APPROVE↔DENY swings and
        # excludes conditional transitions. ``None`` on pre-3-state runs
        # keeps the legacy score-threshold path byte-identical.
        result = compute_verdict_flip_rate(
            sm.matrix, orig_idx, n_resamples=n_resamples, conf=conf, seed=seed,
            class_matrix=sm.class_matrix,
        )
    except (ValueError, ImportError) as e:
        log.warning(
            "verdict_flip computation failed in attestation path: %s", e
        )
        return None
    # Wilson CI on the flip proportion (count, total). Adds defensible
    # uncertainty bounds independent of the cluster-bootstrap CI above:
    # the bootstrap quantifies "how much would re-sampling the eval set
    # move it"; Wilson quantifies "is the rate non-zero given this many
    # paired observations". Both are surfaced — buyers asked for both.
    # ``wilson_conf_level`` is persisted so the renderers (§17 paragraph,
    # UI subtitle) can show the correct percentage even if a future
    # caller passes a non-default ``conf`` (audit pass-1 Fuzzer-MAJOR-4).
    wilson_lo: float | None = None
    wilson_hi: float | None = None
    if (
        result.numerator is not None
        and result.denominator is not None
        and result.denominator > 0
    ):
        try:
            wilson_lo, wilson_hi = wilson_score_interval(
                result.numerator, result.denominator, conf=conf,
            )
        except (ValueError, TypeError) as e:
            log.warning(
                "wilson CI failed in attestation path: %s", e,
            )
    return {
        "point": result.point,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "degenerate_flag": result.degenerate_flag,
        "numerator": result.numerator,
        "denominator": result.denominator,
        "wilson_ci_lower": wilson_lo,
        "wilson_ci_upper": wilson_hi,
        "wilson_conf_level": conf if wilson_lo is not None else None,
    }


def compute_kappa_orig_vs_perm_for_attestation(
    run_dir: Path,
    run_id: str,
    manifest: dict[str, Any],
) -> float | None:
    """§14 step 15a (d) attestation wire-up for κ original-vs-permuted.

    Returns a scalar in ``[-1, 1]`` or ``None`` when the metric isn't
    computable. Distinct from the legacy ``cohens_kappa_vs_truth`` —
    this κ compares the original variant against the union of all
    permuted variants (no ground-truth column required).
    """
    log = logging.getLogger(__name__)
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return None
    try:
        import sqlite3

        from topofuzz.metrics import (
            cohens_kappa_original_vs_permuted,
            read_score_matrix,
        )
    except ImportError as e:
        log.warning(
            "topofuzz.metrics unavailable; kappa_orig_vs_perm skipped: %s", e
        )
        return None
    try:
        sm = read_score_matrix(
            run_dir, run_id=run_id, allow_empty_rubric_hash=True
        )
    except (FileNotFoundError, ValueError, OSError, sqlite3.DatabaseError):
        return None
    if sm.matrix.size == 0 or sm.matrix.shape[0] < 2 or sm.matrix.shape[1] < 1:
        return None
    orig_idx = _resolve_original_variant_index(manifest, sm.variant_ids)
    if orig_idx is None:
        return None
    try:
        kappa = cohens_kappa_original_vs_permuted(sm.matrix, orig_idx)
    except (ValueError, ImportError) as e:
        log.warning(
            "kappa_orig_vs_perm computation failed in attestation path: %s", e
        )
        return None
    if kappa is None:
        return None
    return float(kappa)


def compute_ba_drop_for_attestation(
    run_dir: Path,
    run_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    """§14 step 15a (e) attestation wire-up for balanced-accuracy drop.

    Returns ``{original, mean_permuted, drop}`` or ``None`` when
    ground_truth is absent from the manifest, the metric is degenerate,
    or the score matrix is unreadable.

    Ground truth is sourced from ``manifest.eval_ground_truth`` (a
    list of 0/1/null parallel to ``manifest.eval_ids``). Step 22
    consumers MUST treat a ``None`` return as "this run has no
    ground-truth signal" and not assert on BA-drop in that case.
    """
    log = logging.getLogger(__name__)
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return None
    try:
        import sqlite3

        from topofuzz.metrics import balanced_accuracy_drop, read_score_matrix
    except ImportError as e:
        log.warning(
            "topofuzz.metrics unavailable; ba_drop skipped in attestation: %s",
            e,
        )
        return None
    try:
        sm = read_score_matrix(
            run_dir, run_id=run_id, allow_empty_rubric_hash=True
        )
    except (FileNotFoundError, ValueError, OSError, sqlite3.DatabaseError):
        return None
    if sm.matrix.size == 0 or sm.matrix.shape[0] < 2 or sm.matrix.shape[1] < 1:
        return None
    orig_idx = _resolve_original_variant_index(manifest, sm.variant_ids)
    if orig_idx is None:
        return None
    gt = _extract_ground_truth_binary(manifest, sm.eval_ids)
    if gt is None:
        return None
    try:
        result = balanced_accuracy_drop(sm.matrix, gt, orig_idx)
    except (ValueError, ImportError) as e:
        log.warning(
            "ba_drop computation failed in attestation path: %s", e
        )
        return None
    if result.degenerate_flag or result.original is None:
        return None
    return {
        "original": result.original,
        "mean_permuted": result.mean_permuted,
        "drop": result.drop,
    }


def compute_noise_floor_for_attestation(
    run_dir: Path,
    run_id: str,
    *,
    n_resamples: int = 5000,
    conf: float = 0.95,
    seed: int | None = 0,
) -> tuple[float | None, dict[str, Any] | None, float | None, str | None]:
    """Best-effort noise-floor metrics from the ``confirm`` pass.

    Returns the 4-tuple ``(verdict_stability, confirmed_verdict_flip_rate,
    voo_noise_floor, note)`` for the headline. The first three elements
    are ``None`` (and the function never raises) in every case below; the
    ``note`` is a short human-readable string when the metrics were
    suppressed for a *recoverable* reason (so the attestation can show it
    instead of a number), else ``None``.

    Suppression cases (per the B-RT-1 / M-RT-2 spec-owner decisions —
    ``emits the inputs for the updated §22 gate; no in-tree caller
    applies the gate yet``):

    * **no ``confirm.json`` sidecar** → ``(None, None, None, None)``: no
      confirm pass has been run. Stable nulls, no note.
    * **``confirm.json`` ``partial: true``** → ``(None, None, None,
      <note>)``: a cost-cap hard-stop left a ragged repeat block; the
      majority vote cannot be taken honestly. The note tells the operator
      to re-run ``topofuzz confirm`` to complete it.
    * **``confirm.json`` ``n_unmatched_candidates > 0``** → same shape:
      fixture / rng_seed drift meant some candidates produced zero
      repeats; the confirm data is incomplete (audit M-RT-2).
    * **untrustworthy sidecar** → ``(None, None, None, <note>)``: a
      sidecar missing the load-bearing ``partial`` or ``repeats`` key, or
      with a non-int / negative ``repeats`` / ``n_unmatched_candidates``,
      cannot be trusted to gate the headline — refuse rather than
      silently degrade to a looser mode (audit RT-3 / CR-MINOR-2/3).
    * **sidecar ``repeats`` disagrees with the table** → same shape: a
      persisted ``repeat_idx`` is out of the sidecar's declared K range,
      so the sidecar is stale or tampered (audit RT-3).
    * **disposition buckets do not sum to ``n_screened_pairs``** → same
      shape: the ``confirmed_verdict_flip`` accounting is internally
      inconsistent (audit N-RT-3).
    * numpy / ``topofuzz.metrics`` unavailable → ``(None, None, None,
      None)``.

    Only when the sidecar exists and is trustworthy, ``partial`` is
    false, ``n_unmatched_candidates`` is 0, and the table is consistent
    with it are the three metrics computed and returned.

    * ``verdict_stability`` — scalar in [0, 1] or ``None``.
    * ``confirmed_verdict_flip_rate`` — ``{point, ci_lower, ci_upper,
      confirmed, flaky, unstable_baseline, missing_baseline,
      judge_failure_pairs, n_screened_pairs, degenerate_flag}`` or
      ``None``.
    * ``voo_noise_floor`` — scalar or ``None``.

    The reads go through :func:`topofuzz.confirm.read_flip_confirmations`
    / :func:`topofuzz.confirm.read_confirm_sidecar`; the metrics are the
    pure functions in :mod:`topofuzz.metrics`.
    """
    log = logging.getLogger(__name__)
    try:
        import numpy as np  # noqa: F401 — numpy presence gate
    except ImportError:
        return None, None, None, None
    try:
        from topofuzz.confirm import (
            ConfirmError,
            read_confirm_sidecar,
            read_flip_confirmations,
        )
        from topofuzz.metrics import (
            compute_confirmed_verdict_flip_rate,
            compute_verdict_stability,
            compute_voo_noise_floor,
        )
        from topofuzz.repro import ReproError, find_all_flip_candidates
    except ImportError as e:
        log.warning(
            "topofuzz.metrics/confirm unavailable; noise-floor metrics "
            "skipped in attestation: %s", e,
        )
        return None, None, None, None

    # B-RT-1: the sidecar is the source of truth for "was a confirm pass
    # run, and did it complete?". Missing sidecar → no confirm pass →
    # stable nulls. The flip_confirmations table existing-but-empty (a
    # RunWriter created the table) is NOT a confirm pass.
    try:
        sidecar = read_confirm_sidecar(run_dir)
    except ConfirmError as e:
        # A corrupt sidecar is a real signal — surface it as a note
        # rather than silently treating the run as un-confirmed.
        log.warning("confirm.json unreadable in %s: %s", run_dir, e)
        return None, None, None, (
            "confirm.json present but unreadable — re-run `topofuzz "
            "confirm` to regenerate it"
        )
    if sidecar is None:
        # No confirm pass run. Stable nulls in the headline, no note.
        return None, None, None, None

    # CR-MINOR-2 / RT-3: ``partial`` is load-bearing — a missing key in
    # an otherwise-valid sidecar means we cannot trust it. A truncated
    # sidecar must REFUSE (note + null headline), not silently fall into
    # the looser "treat as complete" path.
    if "partial" not in sidecar:
        return None, None, None, (
            "confirm.json is missing the `partial` field — cannot trust "
            "this sidecar; re-run `topofuzz confirm` to regenerate it"
        )
    if sidecar.get("partial") is True:
        # B-RT-1(c): a partial confirm pass must NOT ship a confident
        # noise-corrected number. Surface a note instead.
        return None, None, None, (
            "partial confirm pass — re-run `topofuzz confirm` to complete"
        )
    # CR-MINOR-3: JSON ``true``/``false`` are int subclasses, so a bare
    # ``isinstance(x, int)`` accepts ``n_unmatched_candidates: true`` and
    # then prints a cosmetically wrong "True screened candidate variant(s)"
    # note. Require a real (non-bool) int; a bad value is an untrustworthy
    # sidecar — refuse (consistent with RT-3), don't guess.
    n_unmatched = sidecar.get("n_unmatched_candidates", 0)
    if isinstance(n_unmatched, bool) or not isinstance(n_unmatched, int):
        return None, None, None, (
            "confirm.json `n_unmatched_candidates` is not an integer — "
            "cannot trust this sidecar; re-run `topofuzz confirm`"
        )
    if n_unmatched < 0:
        return None, None, None, (
            "confirm.json `n_unmatched_candidates` is negative — cannot "
            "trust this sidecar; re-run `topofuzz confirm`"
        )
    if n_unmatched > 0:
        # M-RT-2: variant-reconstruction drift left some screened
        # candidates with zero repeats — the confirm data is incomplete.
        return None, None, None, (
            f"{n_unmatched} screened candidate variant(s) could not be "
            "reconstructed (fixture/rng_seed drift) — confirm data "
            "incomplete; re-run `topofuzz confirm`"
        )

    # ``repeats`` from the sidecar is threaded into every metric so a
    # ragged repeat block (a unit whose scored-repeat count != K) is
    # DROPPED rather than majority-voted — B-RT-1(b) / RT-2.
    #
    # RT-3 / CR-MINOR-2: ``repeats`` is load-bearing (it decides which
    # units get dropped). A missing or non-positive-int ``repeats`` in an
    # otherwise-valid sidecar means we cannot trust it — REFUSE rather
    # than silently fall back to the looser ``repeats=None`` path (which
    # would majority-vote ragged blocks). Earlier code degraded to
    # ``repeats=None`` here; that was the CR-MINOR-2 silent-looser-mode
    # bug.
    repeats = sidecar.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        return None, None, None, (
            "confirm.json `repeats` is missing or not a positive integer "
            "— cannot trust this sidecar; re-run `topofuzz confirm`"
        )

    try:
        reference_rows = read_flip_confirmations(
            run_dir, run_id=run_id, role="reference"
        )
        candidate_rows = read_flip_confirmations(
            run_dir, run_id=run_id, role="candidate"
        )
    except ConfirmError:
        # The sidecar said a confirm pass ran, but run.sqlite is now
        # missing / schemaless. Treat as un-computable with a note.
        return None, None, None, (
            "confirm.json present but flip_confirmations table is "
            "missing — re-run `topofuzz confirm`"
        )

    if not reference_rows and not candidate_rows:
        # Sidecar present but the table is empty — a no-candidate run
        # (a non-fragile run legitimately has zero candidates). Nulls,
        # no note (this is not an error condition).
        return None, None, None, None

    # RT-3: ``confirm.json`` is an unsigned sidecar and ``repeats`` is
    # load-bearing — it decides which units the metrics drop. Cross-check
    # it against the actual ``repeat_idx`` values in the table: a
    # hand-crafted sidecar whose ``repeats`` disagrees with the persisted
    # rows would silently change the drop logic. We do not sign the
    # sidecar (spec: minimal cross-check, NOT a signing scheme) — but a
    # ``repeat_idx`` at or beyond ``repeats`` is structurally impossible
    # for a pass that actually ran K=``repeats`` (the writer emits
    # ``repeat_idx`` in ``range(repeats)``), so it is an unambiguous
    # tamper / staleness signal. Refuse rather than trust it.
    observed_idxs = [
        r["repeat_idx"]
        for r in (*reference_rows, *candidate_rows)
        if isinstance(r.get("repeat_idx"), int)
    ]
    if observed_idxs and max(observed_idxs) >= repeats:
        return None, None, None, (
            "confirm.json `repeats` disagrees with the flip_confirmations "
            "table (a persisted repeat_idx is out of range) — re-run "
            "`topofuzz confirm`"
        )

    # The screened candidate-pair set — needed so the disposition
    # buckets account for pairs whose every repeat was a judge failure
    # (M-RT-1). Best-effort: if it can't be read, fall through to the
    # observed-pairs fallback in the metric.
    try:
        screened_candidates: list[tuple[str, str]] | None = (
            find_all_flip_candidates(run_dir)
        )
    except ReproError:
        screened_candidates = None

    stability_result = compute_verdict_stability(
        reference_rows, repeats=repeats
    )
    verdict_stability = stability_result.point

    confirmed_result = compute_confirmed_verdict_flip_rate(
        reference_rows,
        candidate_rows,
        candidates=screened_candidates,
        repeats=repeats,
        n_resamples=n_resamples,
        conf=conf,
        seed=seed,
    )
    # N-RT-3: the honesty invariant — every screened pair lands in exactly
    # one disposition bucket — only holds when ``candidates`` was supplied
    # (see ConfirmedVerdictFlipResult). The attest path DOES supply it
    # unless ``find_all_flip_candidates`` raised, so when it is non-None
    # the buckets MUST sum to ``n_screened_pairs``. A mismatch means the
    # metric's accounting is internally inconsistent — refuse to populate
    # the headline + emit a note rather than ``assert`` (a hard crash in
    # attest is worse than note-and-null, the house pattern).
    if screened_candidates is not None:
        bucket_sum = (
            confirmed_result.confirmed
            + confirmed_result.flaky
            + confirmed_result.unstable_baseline
            + confirmed_result.missing_baseline
            + confirmed_result.judge_failure_pairs
        )
        if bucket_sum != confirmed_result.n_screened_pairs:
            return None, None, None, (
                "confirmed_verdict_flip disposition buckets do not sum to "
                "the screened-pair count — confirm data is internally "
                "inconsistent; re-run `topofuzz confirm`"
            )
    # F1: Wilson CI on the noise-corrected rate. The denominator for the
    # confirmed metric is ``confirmed + flaky`` per the module docstring
    # — "candidate pairs with stable baseline". Other buckets
    # (unstable_baseline, missing_baseline, judge_failure_pairs) are NOT
    # candidate-pair observations; they are pair-level data-coverage
    # gaps, excluded by construction. User feedback 2026-05-15:
    # "Wilson must apply to the noise-corrected number, not just the
    # raw rate; otherwise it's a half-fix to flag." Skipped on the
    # degenerate-result branch (no Wilson on a None point).
    conf_wilson_lo: float | None = None
    conf_wilson_hi: float | None = None
    if not confirmed_result.degenerate_flag and confirmed_result.point is not None:
        conf_denom = confirmed_result.confirmed + confirmed_result.flaky
        if conf_denom > 0:
            try:
                from topofuzz.metrics import wilson_score_interval
                conf_wilson_lo, conf_wilson_hi = wilson_score_interval(
                    confirmed_result.confirmed, conf_denom, conf=conf,
                )
            except (ValueError, TypeError, ImportError) as e:
                log.warning(
                    "wilson CI on confirmed rate failed in attestation: %s",
                    e,
                )
    confirmed_dict: dict[str, Any] = {
        "point": confirmed_result.point,
        "ci_lower": confirmed_result.ci_lower,
        "ci_upper": confirmed_result.ci_upper,
        "confirmed": confirmed_result.confirmed,
        "flaky": confirmed_result.flaky,
        "unstable_baseline": confirmed_result.unstable_baseline,
        "missing_baseline": confirmed_result.missing_baseline,
        "judge_failure_pairs": confirmed_result.judge_failure_pairs,
        "n_screened_pairs": confirmed_result.n_screened_pairs,
        "degenerate_flag": confirmed_result.degenerate_flag,
        "wilson_ci_lower": conf_wilson_lo,
        "wilson_ci_upper": conf_wilson_hi,
        "wilson_conf_level": conf if conf_wilson_lo is not None else None,
    }

    voo_noise_floor = compute_voo_noise_floor(reference_rows, repeats=repeats)

    return verdict_stability, confirmed_dict, voo_noise_floor, None


def compute_findings(
    run_dir: Path,
    run_id: str,
    manifest: dict[str, Any],
    *,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Assemble the attestation ``findings`` dict.

    Today carries one key: ``per_agent_influence`` (step 15, §6a). Other
    findings (worst-case viewer summary in §10a, natural-language
    summary in §17) will land here in subsequent steps. Returns an
    empty dict when no field is computable so the payload shape stays
    stable across runs (a missing ``findings`` key would force readers
    to defensive-default).
    """
    findings: dict[str, Any] = {}
    influence = compute_influence_for_attestation(
        run_dir, run_id, manifest, seed=seed
    )
    if influence is not None:
        findings["per_agent_influence"] = influence
    return findings


def emit_run_attestation(
    *,
    run_dir: Path,
    run_id: str,
    manifest: dict[str, Any],
    status: str,
    successful_cells: int,
    total_cells: int,
    n_variants: int,
    n_evals: int,
    total_cost_usd: float,
    elapsed_s: float,
    signer: Signer,
    judge_failure_rate: float | None = None,
    judge_independence: dict[str, Any] | None = None,
) -> Path:
    """Orchestrator's one-call attestation emit.

    Computes the headline (voo + verdict_flip + κ + BA-drop +
    judge_failure_rate + judge_independence on the best-effort path),
    digests ``run.sqlite`` if present, builds the payload, signs, and
    writes ``<run_dir>/attestation.json`` plus the step-15a (h) plain
    ``<run_dir>/report.json`` mirror. Returns the path of the signed
    envelope. Raises only if signing or disk-write fails — the
    orchestrator should call this in a try/except and log-and-continue
    on failure so a bad sign never tanks a successful run.

    Step 15a kwargs (BLUEPRINT §14 (f) + (g)):

    * ``judge_failure_rate`` — cell-level §8 rate sourced from the
      orchestrator's ``cost_state['judge_failure_cells'] / n_total_cells``.
      Defaults to ``None`` to keep backwards-compatibility with
      pre-15a callers (the legacy in-tree call site at
      ``orchestrator._maybe_emit_attestation`` is upgraded to pass it).
    * ``judge_independence`` — pre-assembled §8 dict produced by the
      orchestrator (it has direct access to the judge instance + the
      topology IR for the config-overlap computation). Defaults to
      ``None`` when no judge is attached.
    """
    # Audit RT-M8: pass seed=0 explicitly at every call site so a
    # future refactor of any callee's default doesn't silently lose
    # the rc1→GA byte-stable contract.
    voo = compute_voo_for_attestation(run_dir, run_id, seed=0)
    verdict_flip = compute_verdict_flip_for_attestation(
        run_dir, run_id, manifest, seed=0
    )
    kappa_orig_vs_perm = compute_kappa_orig_vs_perm_for_attestation(
        run_dir, run_id, manifest
    )
    ba_drop = compute_ba_drop_for_attestation(run_dir, run_id, manifest)
    # Noise-floor / verdict-confirmation metrics. All three are None
    # unless a COMPLETE `topofuzz confirm` pass has run (confirm.json
    # sidecar present, not partial, no reconstruction drift). A partial
    # pass yields None metrics + a non-null ``confirm_note``.
    (
        verdict_stability,
        confirmed_verdict_flip,
        voo_noise_floor,
        confirm_note,
    ) = compute_noise_floor_for_attestation(run_dir, run_id, seed=0)
    headline = compute_headline(
        status=status,
        n_successful_cells=successful_cells,
        n_total_cells=total_cells,
        n_variants=n_variants,
        n_evals=n_evals,
        total_cost_usd=total_cost_usd,
        elapsed_s=elapsed_s,
        voo_per_eval=voo,
        verdict_flip_rate=verdict_flip,
        kappa_original_vs_permuted=kappa_orig_vs_perm,
        balanced_accuracy_drop=ba_drop,
        judge_failure_rate=judge_failure_rate,
        judge_independence=judge_independence,
        verdict_stability=verdict_stability,
        confirmed_verdict_flip_rate=confirmed_verdict_flip,
        voo_noise_floor=voo_noise_floor,
        confirm_note=confirm_note,
    )
    findings = compute_findings(run_dir, run_id, manifest, seed=0)
    sqlite_path = run_dir / "run.sqlite"
    if sqlite_path.exists():
        sqlite_sha = sha256_file(sqlite_path)
    else:
        # Audit RT-m1: surface the "no sqlite at finalize" case so the
        # operator's logs distinguish an intentionally-aborted run
        # from a writer-crashed-mid-finalize bug. INFO not WARNING:
        # the verify path treats empty sha as a vacuous pass, so this
        # is not an error — just a data point worth recording.
        logging.getLogger(__name__).info(
            "emit_run_attestation: run.sqlite missing at %s; "
            "signing payload with empty run_sqlite_sha256",
            sqlite_path,
        )
        sqlite_sha = ""
    payload = build_attestation_payload(
        run_id=run_id,
        manifest=manifest,
        headline=headline,
        findings=findings,
        run_sqlite_sha256=sqlite_sha,
    )
    return emit_attestation(
        payload,
        signer,
        run_dir / ATTESTATION_FILENAME,
        report_path=run_dir / REPORT_FILENAME,
    )
