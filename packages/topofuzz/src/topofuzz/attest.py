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
    "VerifyOutcome",
    "build_attestation_payload",
    "compute_findings",
    "compute_headline",
    "compute_influence_for_attestation",
    "compute_voo_for_attestation",
    "emit_attestation",
    "emit_run_attestation",
    "load_and_verify_attestation",
    "sha256_file",
]

PAYLOAD_TYPE: Final[str] = "application/vnd.topofuzz.run+json;v=0.5"
PAYLOAD_SCHEMA_VERSION: Final[int] = 1
ATTESTATION_FILENAME: Final[str] = "attestation.json"


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
) -> dict[str, Any]:
    """Assemble the headline dict.

    ``voo_per_eval`` is the only "expensive" field — callers compute it
    using :mod:`topofuzz.metrics` (score matrix + per-eval variance)
    and pass it in. ECR and κ are hard-coded ``null`` in v0.5; see
    module docstring for the rationale.
    """
    return {
        "status": status,
        "voo_per_eval": voo_per_eval,
        "ecr": None,
        "kappa": None,
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
) -> Path:
    """JCS-canonicalize ``payload``, DSSE-sign, write to ``out_path``.

    The on-disk envelope is the raw bytes returned by
    :func:`fuzzeval.attestation.dsse_sign` (compact, key-sorted JSON).
    No newline is appended — the file is exactly the bytes a verifier
    would re-read with ``Path.read_bytes()``.
    """
    canonical = rfc8785.dumps(payload)
    envelope = dsse_sign(canonical, PAYLOAD_TYPE, signer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(envelope)
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
) -> list[dict[str, Any]] | None:
    """Best-effort per-eval ``variance_across_orderings`` for the headline.

    Returns ``None`` (and never raises) if the score matrix can't be
    read — empty rubric_hash, no successful cells, numpy missing, etc.
    The orchestrator emits the attestation regardless; the headline
    just gets ``voo_per_eval=None`` and the verifier accepts that.

    The per-eval point estimate is ``np.var(matrix[:, j], ddof=0)`` for
    each eval ``j`` — matching the inner statistic of
    :func:`topofuzz.metrics.variance_across_orderings`. Per-eval cells
    with NaN (failed cell in that variant×eval) propagate to ``point=None``
    so the JSON stays well-typed.

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
    # ddof=0 matches the variance_across_orderings convention (BLUEPRINT
    # §9 point estimate). We compute per-eval variance across the variant
    # axis. If any cell in an eval column is NaN, np.var of that column
    # is NaN — we emit point=None for that eval rather than NaN (JSON
    # has no native NaN; rfc8785 would refuse).
    per_eval = np.var(matrix, axis=0, ddof=0)
    out: list[dict[str, Any]] = []
    for eid, v in zip(result.eval_ids, per_eval, strict=True):
        if np.isnan(v) or np.isinf(v):
            out.append({"eval_id": eid, "point": None})
        else:
            out.append({"eval_id": eid, "point": float(v)})
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
) -> Path:
    """Orchestrator's one-call attestation emit.

    Computes the headline (voo best-effort), digests ``run.sqlite`` if
    present, builds the payload, signs, and writes
    ``<run_dir>/attestation.json``. Returns the path written. Raises
    only if signing or disk-write fails — the orchestrator should call
    this in a try/except and log-and-continue on failure so a bad sign
    never tanks a successful run.
    """
    voo = compute_voo_for_attestation(run_dir, run_id)
    headline = compute_headline(
        status=status,
        n_successful_cells=successful_cells,
        n_total_cells=total_cells,
        n_variants=n_variants,
        n_evals=n_evals,
        total_cost_usd=total_cost_usd,
        elapsed_s=elapsed_s,
        voo_per_eval=voo,
    )
    findings = compute_findings(run_dir, run_id, manifest)
    sqlite_path = run_dir / "run.sqlite"
    sqlite_sha = sha256_file(sqlite_path) if sqlite_path.exists() else ""
    payload = build_attestation_payload(
        run_id=run_id,
        manifest=manifest,
        headline=headline,
        findings=findings,
        run_sqlite_sha256=sqlite_sha,
    )
    return emit_attestation(payload, signer, run_dir / ATTESTATION_FILENAME)
