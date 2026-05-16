# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Sigstore Rekor opt-in submission for topofuzz attestations.

BLUEPRINT_v5plus.md §11 line 306 + step 18 (line 436): when a run is
signed via ``topofuzz run --rekor``, this module POSTs the freshly-
written DSSE envelope to the Sigstore Rekor Public Good Instance and
embeds the returned ``LogEntry`` in ``<run_dir>/report.json.dsse.rekor.json``.
On timeout the report banner lights ``[REKOR-DEFERRED]`` (gated by a
``<run_dir>/.rekor-deferred`` marker file) and the user can re-run
``topofuzz run --rekor`` later to retry.

Network discipline (step-18 decision 2): synchronous attempts only,
~30 s ceiling. Up to ``MAX_ATTEMPTS`` POSTs with ``BACKOFF_DELAYS_S``
between them, ``PER_ATTEMPT_TIMEOUT_S`` per attempt. The literal
BLUEPRINT "12 h retry" reading is rejected for v0.5 because blocking
the foreground for half a day is hostile to the demo path; a 30-second
ceiling + a clear deferred marker that the user can re-attest later is
the simplest thing that meets the spirit of the spec without owning a
background-daemon process model.

Dependency discipline (step-18 decision 4): handcrafted ``urllib`` POST.
The DSSE envelope is already signed by ``fuzzeval.attestation`` — Rekor
verifies the signature itself using the supplied public key. We don't
sign anything new here, so we don't need the ``sigstore`` PyPI package
and its ~20 MB of transitive deps (cryptography is already a topofuzz
dep but securesystemslib + sigstore-python's other transitives are not).

Rekor entry shape used:
``{apiVersion: "0.0.1", kind: "dsse", spec: {envelope, publicKey}}``.
``envelope`` is the DSSE JSON-string (NOT base64); ``publicKey`` is the
base64-encoded PEM. Reference:
https://github.com/sigstore/rekor/blob/main/types/dsse/v0.0.1/dsse_v0_0_1_schema.json
"""

from __future__ import annotations

import base64
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

__all__ = [
    "BACKOFF_DELAYS_S",
    "DEFAULT_REKOR_URL",
    "MAX_ATTEMPTS",
    "MAX_RESPONSE_BYTES",
    "PER_ATTEMPT_TIMEOUT_S",
    "REKOR_DEFERRED_MARKER",
    "REKOR_OUTPUT_FILENAME",
    "RekorSubmitOutcome",
    "looks_like_rekor_deferred",
    "looks_like_rekor_logged",
    "submit_for_run_dir",
    "submit_to_rekor",
]

DEFAULT_REKOR_URL: Final[str] = "https://rekor.sigstore.dev/api/v1/log/entries"
REKOR_OUTPUT_FILENAME: Final[str] = "report.json.dsse.rekor.json"
REKOR_DEFERRED_MARKER: Final[str] = ".rekor-deferred"
# 3 attempts × 10 s per-attempt + (1 s + 2 s) backoff = ~33 s ceiling.
# Tuned for the demo path: a developer running ``topofuzz run --rekor``
# from a laptop sees a result inside one minute. If Rekor is genuinely
# down for longer than that, a 12 h foreground block wouldn't help.
MAX_ATTEMPTS: Final[int] = 3
PER_ATTEMPT_TIMEOUT_S: Final[float] = 10.0
BACKOFF_DELAYS_S: Final[tuple[float, ...]] = (1.0, 2.0)
USER_AGENT: Final[str] = (
    "topofuzz/0.5 (+https://github.com/kirannayudu/topofuzz)"
)
# Rekor LogEntry responses are ~few KB (signed entry timestamp,
# inclusion proof hashes, body base64). Cap at 1 MiB so a misrouted
# response (e.g. an HTML error page) doesn't pin memory or fill the
# run dir if a future writer uses the raw bytes.
MAX_RESPONSE_BYTES: Final[int] = 1 << 20

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RekorSubmitOutcome:
    """Result of a single ``submit_to_rekor`` call.

    ``ok=True`` iff a 2xx response was received AND its body parsed as
    JSON. ``log_entry`` is the parsed Rekor response (the
    ``{"<uuid>": LogEntry}`` envelope or an unwrapped LogEntry,
    depending on the API version) — callers should not assume a
    specific shape beyond "JSON dict". ``error`` carries a short
    operator-readable string when ``ok=False``. ``attempts`` is the
    number of network round-trips actually made (1..MAX_ATTEMPTS).
    """

    ok: bool
    log_entry: dict[str, Any] | None
    error: str | None
    attempts: int
    url: str


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_proposed_entry(
    envelope_bytes: bytes, public_key_pem: bytes
) -> bytes:
    """Build the Rekor ``dsse v0.0.1`` proposed-entry JSON body.

    Schema (verified against
    https://github.com/sigstore/rekor/blob/main/pkg/types/dsse/v0.0.1/dsse_v0_0_1_schema.json):

    .. code-block:: json

       {
         "apiVersion": "0.0.1",
         "kind": "dsse",
         "spec": {
           "proposedContent": {
             "envelope": "<DSSE-JSON-string>",
             "verifiers": ["<base64(PEM-public-key)>"]
           }
         }
       }

    ``proposedContent.envelope`` is the DSSE envelope's JSON STRING
    (NOT base64). ``proposedContent.verifiers`` is an array (minItems:
    1) of base64-encoded verification material — for v0.5 we always
    submit a single base64-wrapped PEM public key.

    Audit Pass-1 RT BLOCKER-1 fix: the prior shape
    (``spec.envelope`` + ``spec.publicKey`` as a singular string) was
    not the current Rekor schema and would have produced a 400 on
    every live submission. The dual-pass audit caught it before
    commit; the manual live smoke against the Public Good Instance
    verifies the fix at step-18 commit time.

    Audit Pass-1 CR MAJOR-1 fix: a non-UTF-8 envelope (would only
    happen if a future signer changed the on-disk encoding) used to
    raise ``UnicodeDecodeError`` at module-level decode, which then
    crashed the CLI through a too-narrow exception filter. Re-raise
    as :class:`ValueError` here so :func:`submit_to_rekor`'s
    ValueError catch keeps the failure inside the deferred-marker
    flow.
    """
    try:
        envelope_str = envelope_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"DSSE envelope is not valid UTF-8 ({e}); cannot submit to Rekor"
        ) from e
    proposed = {
        "apiVersion": "0.0.1",
        "kind": "dsse",
        "spec": {
            "proposedContent": {
                "envelope": envelope_str,
                "verifiers": [
                    base64.b64encode(public_key_pem).decode("ascii"),
                ],
            },
        },
    }
    return json.dumps(proposed).encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


def _post_once(
    url: str,
    body: bytes,
    *,
    timeout: float,
    opener: Callable[..., Any] | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    """Single POST attempt. Returns ``(status, body_bytes, headers)``.

    Audit Pass-1 RT MINOR-2 fix: the function now returns response
    headers (lower-cased keys) so the retry loop can honor
    ``Retry-After`` hints from rate-limited responses and so the
    operator-facing log line can carry ``X-Request-ID`` for triage.

    Audit Pass-1 CR MINOR-3 fix: status is read explicitly with a
    -1 sentinel for the "shim missing the attribute" case (real
    ``urlopen`` always sets ``.status``); the retry loop's
    ``ValueError`` branch then catches the sentinel as a network
    error rather than silently retrying through a 0-status hole.

    The ``opener`` seam exists so tests can inject a fake without
    monkey-patching the global :mod:`urllib.request` module — keeps
    the test isolation explicit at the call site.

    Non-2xx responses return their status + body rather than raising;
    the retry loop in :func:`submit_to_rekor` decides whether the
    status is retryable.
    """
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    open_fn = opener if opener is not None else urllib.request.urlopen
    try:
        with open_fn(req, timeout=timeout) as resp:
            status = getattr(resp, "status", -1)
            if not isinstance(status, int) or status <= 0:
                # A real urllib response always carries a positive status.
                # A test shim that forgot to set it should fail loudly,
                # not be silently retried into a hole.
                raise ValueError(
                    f"rekor response missing status (got {status!r})"
                )
            headers = _headers_to_dict(getattr(resp, "headers", None))
            data = resp.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise ValueError(
                    f"rekor response exceeds {MAX_RESPONSE_BYTES}B cap"
                )
            return status, bytes(data), headers
    except urllib.error.HTTPError as e:
        try:
            data = e.read(MAX_RESPONSE_BYTES + 1)[:MAX_RESPONSE_BYTES]
        except OSError:
            data = b""
        headers = _headers_to_dict(getattr(e, "headers", None))
        return int(e.code), bytes(data), headers


def _headers_to_dict(raw: Any) -> dict[str, str]:
    """Normalize a response/HTTPError headers object into a flat dict.

    urllib's response objects expose ``.headers`` as an
    :class:`email.message.Message` (case-insensitive multi-dict) and
    test shims may pass anything dict-like or ``None``. Normalize to
    ``{lowercase_key: value}`` so downstream code can do simple
    ``headers.get("location")`` lookups without wrapping every site.
    """
    if raw is None:
        return {}
    try:
        items = raw.items()
    except AttributeError:
        return {}
    out: dict[str, str] = {}
    for k, v in items:
        if isinstance(k, str) and isinstance(v, str):
            out[k.lower()] = v
    return out


def submit_to_rekor(
    envelope_bytes: bytes,
    public_key_pem: bytes,
    *,
    url: str = DEFAULT_REKOR_URL,
    max_attempts: int = MAX_ATTEMPTS,
    per_attempt_timeout_s: float = PER_ATTEMPT_TIMEOUT_S,
    backoff_delays_s: tuple[float, ...] = BACKOFF_DELAYS_S,
    sleep: Callable[[float], None] = time.sleep,
    opener: Callable[..., Any] | None = None,
) -> RekorSubmitOutcome:
    """POST a signed DSSE envelope to Rekor with bounded retry.

    Per BLUEPRINT line 306 + step-18 decision 2, this is explicitly
    bounded synchronous (~30s ceiling), NOT a 12h background daemon —
    the ``[REKOR-DEFERRED]`` watermark covers the failure case and the
    user can re-run later when the network is healthy.

    Retry policy:

    * Network errors (URLError, TimeoutError, OSError) and 5xx are
      retried with the backoff sequence.
    * 4xx responses are NOT retried — they're terminal (bad request,
      rate limit, etc.). The outcome carries the response body for
      operator triage.
    * 409 Conflict is special-cased (audit Pass-1 RT MAJOR-1): an
      identical envelope already lives in the log, which is a
      successful outcome from the user's perspective even though
      Rekor surfaces it as 4xx. We persist a stub LogEntry that
      carries the existing entry's URL so the run dir still witnesses
      "this run is in Rekor"; full LogEntry recovery (a separate GET
      against the Location header) is deferred to v0.6.
    * 2xx with unparseable body, non-dict body, or empty-dict body is
      treated as terminal failure — the server is misbehaving in a
      way retry won't fix in 30 s, AND ``looks_like_rekor_logged``
      requires a non-empty dict (audit Pass-1 RT MAJOR-4: keep the
      success contract and the gate consistent so an "ok" outcome
      always lights the report banner correctly).

    The ``sleep`` and ``opener`` seams exist for test injection — pass
    ``sleep=lambda _: None`` to avoid real waits, and ``opener`` to
    avoid real network calls. Production callers leave both at default.

    The public key is supplied (not loaded) so this module stays free
    of ``fuzzeval.keys`` imports — :func:`submit_for_run_dir` is the
    convenience wrapper that handles the load.

    Audit Pass-1 CR MAJOR-1 fix: ``_build_proposed_entry`` is wrapped
    in a try/except so a non-UTF-8 envelope (from a future signer
    refactor) returns a structured failure outcome rather than
    crashing the CLI through an uncaught :class:`UnicodeDecodeError`.
    """
    try:
        body = _build_proposed_entry(envelope_bytes, public_key_pem)
    except ValueError as e:
        # CR MAJOR-1: surface as a structured failure so
        # ``submit_for_run_dir`` writes the deferred marker and the
        # CLI prints a clean ``[REKOR-DEFERRED]`` line.
        return RekorSubmitOutcome(
            ok=False, log_entry=None,
            error=f"rekor: {e}",
            attempts=0, url=url,
        )
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            status, data, headers = _post_once(
                url, body,
                timeout=per_attempt_timeout_s, opener=opener,
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            # ValueError covers the response-cap-exceeded and
            # missing-status paths from _post_once. Network and
            # oversized-response failures are both retryable on the
            # assumption that the next attempt may hit a healthier
            # replica or a different proxy.
            last_error = f"network error on attempt {attempt}: {e}"
            _log.warning("rekor: %s url=%s", last_error, url)
        else:
            if 200 <= status < 300:
                try:
                    parsed = json.loads(data.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    last_error = (
                        f"rekor returned {status} but body was unparseable: {e}"
                    )
                    _log.warning("rekor: %s url=%s", last_error, url)
                    return RekorSubmitOutcome(
                        ok=False, log_entry=None, error=last_error,
                        attempts=attempt, url=url,
                    )
                if not isinstance(parsed, dict) or not parsed:
                    # RT MAJOR-4: empty dict is rejected so the on-disk
                    # success file is guaranteed to satisfy
                    # ``looks_like_rekor_logged``'s non-empty-dict gate.
                    last_error = (
                        f"rekor returned {status} with empty / non-dict "
                        f"body (type={type(parsed).__name__})"
                    )
                    _log.warning("rekor: %s url=%s", last_error, url)
                    return RekorSubmitOutcome(
                        ok=False, log_entry=None, error=last_error,
                        attempts=attempt, url=url,
                    )
                return RekorSubmitOutcome(
                    ok=True, log_entry=parsed, error=None,
                    attempts=attempt, url=url,
                )
            if status == 409:
                # RT MAJOR-1: idempotent re-submission. Rekor returns
                # 409 with a ``Location`` header pointing at the
                # already-logged entry. Synthesize a stub LogEntry
                # carrying the URL so the run dir still witnesses
                # "this envelope is in Rekor"; consumers needing the
                # full entry GET it themselves from ``location_url``.
                location = headers.get("location", "")
                stub: dict[str, Any] = {
                    "_rekor_409_already_logged": True,
                    "location_url": location,
                    "url_attempted": url,
                }
                _log.info(
                    "rekor: 409 already-logged at %s — treating as success",
                    location or "(no Location header)",
                )
                return RekorSubmitOutcome(
                    ok=True, log_entry=stub, error=None,
                    attempts=attempt, url=url,
                )
            # The response-body preview belongs in `outcome.error` (the
            # operator-facing return path — they need it to debug a
            # malformed envelope), but NOT in the structured WARNING log
            # — central log shippers would otherwise see server-rendered
            # detail strings carrying operator identifiers / request paths
            # / upstream-component-version diagnostics on a misconfigured
            # Sigstore. Split: full preview goes to the caller, sanitised
            # status+length goes to the log (audit MINOR-1 fix).
            preview = data[:200].decode("utf-8", errors="replace")
            last_error = f"rekor http {status} on attempt {attempt}: {preview}"
            _log.warning(
                "rekor: rekor http %d on attempt %d "
                "(response %d bytes) url=%s",
                status, attempt, len(data), url,
            )
            if 400 <= status < 500:
                # Other 4xx is terminal: a bad payload won't become
                # good on retry. Surface the body so the operator can
                # fix it.
                return RekorSubmitOutcome(
                    ok=False, log_entry=None, error=last_error,
                    attempts=attempt, url=url,
                )
        if attempt < max_attempts:
            # RT MINOR-3: collapsed three-branch fallback to one
            # bounded lookup. ``min(idx, len-1)`` saturates at the
            # last delay if the caller passed fewer delays than
            # attempts; the empty-tuple case falls through to 0.0.
            if backoff_delays_s:
                delay = backoff_delays_s[
                    min(attempt - 1, len(backoff_delays_s) - 1)
                ]
            else:
                delay = 0.0
            if delay > 0:
                sleep(delay)
    return RekorSubmitOutcome(
        ok=False, log_entry=None,
        error=last_error or "rekor: all attempts failed",
        attempts=max_attempts, url=url,
    )


# ---------------------------------------------------------------------------
# Run-dir glue
# ---------------------------------------------------------------------------


def submit_for_run_dir(
    run_dir: Path,
    public_key_pem: bytes,
    *,
    url: str = DEFAULT_REKOR_URL,
    max_attempts: int = MAX_ATTEMPTS,
    per_attempt_timeout_s: float = PER_ATTEMPT_TIMEOUT_S,
    backoff_delays_s: tuple[float, ...] = BACKOFF_DELAYS_S,
    sleep: Callable[[float], None] = time.sleep,
    opener: Callable[..., Any] | None = None,
) -> RekorSubmitOutcome:
    """Submit ``<run_dir>/attestation.json`` to Rekor; persist outcome.

    Side effects on success:

    * Writes ``<run_dir>/report.json.dsse.rekor.json`` atomically (tmp +
      replace) with the parsed Rekor response, sorted-keys + 2-space
      indented for human grep-ability.
    * Removes any prior ``<run_dir>/.rekor-deferred`` marker — a
      successful re-attest supersedes a previous timeout.

    Side effects on failure (any non-ok outcome):

    * Touches ``<run_dir>/.rekor-deferred`` so the report banner gate
      can light ``[REKOR-DEFERRED]``.

    Returns the :class:`RekorSubmitOutcome` for the caller to print.
    Never raises on network or HTTP failure — the deferred marker is
    the user-facing signal.

    Audit Pass-1 RT BLOCKER-2 / MAJOR-2 fix: a successful Rekor POST
    followed by a disk-write failure (ENOSPC, permission denied) or a
    SIGINT used to silently lose the LogEntry — Rekor server-side has
    it, but the local run dir never gets the receipt and a re-run
    would 409. The fix catches both :class:`OSError` and
    :class:`KeyboardInterrupt`/:class:`BaseException` around the
    persistence block and dumps the LogEntry JSON to stderr before
    re-raising / continuing, so the operator can manually save it.
    Re-raising preserves the existing "disk-write failures raise"
    contract for operators wiring their own retry on top.

    Audit Pass-1 CR MINOR-2 fix: ``backoff_delays_s`` is now an
    explicit parameter, removing the asymmetry where a custom
    ``max_attempts`` would silently fall back to the module default.

    Imports :data:`topofuzz.attest.ATTESTATION_FILENAME` lazily to keep
    import cost out of the verify-only path.
    """
    from topofuzz.attest import ATTESTATION_FILENAME

    attest_path = run_dir / ATTESTATION_FILENAME
    if not attest_path.exists():
        outcome = RekorSubmitOutcome(
            ok=False, log_entry=None,
            error=f"no {ATTESTATION_FILENAME} at {attest_path}",
            attempts=0, url=url,
        )
        _write_deferred_marker(run_dir)
        return outcome
    envelope_bytes = attest_path.read_bytes()
    outcome = submit_to_rekor(
        envelope_bytes, public_key_pem,
        url=url,
        max_attempts=max_attempts,
        per_attempt_timeout_s=per_attempt_timeout_s,
        backoff_delays_s=backoff_delays_s,
        sleep=sleep,
        opener=opener,
    )
    if outcome.ok and outcome.log_entry is not None:
        try:
            _write_log_entry(run_dir, outcome.log_entry)
        except BaseException as e:
            # Rekor accepted the envelope (server-side has the entry)
            # but the local persist failed — print the LogEntry JSON
            # to stderr so the operator can manually save it. Then
            # re-raise so the failure mode isn't swallowed (per
            # docstring contract: disk-write failures raise).
            sys.stderr.write(
                "[REKOR] WARNING: Rekor accepted the entry but the local "
                "persist failed (run_dir=%s err=%r). Save this LogEntry "
                "JSON manually:\n%s\n" % (
                    run_dir, e,
                    json.dumps(outcome.log_entry, sort_keys=True, indent=2),
                )
            )
            raise
        marker = run_dir / REKOR_DEFERRED_MARKER
        if marker.exists():
            try:
                marker.unlink()
            except OSError as e:
                _log.warning(
                    "rekor: could not clear stale deferred marker %s: %s",
                    marker, e,
                )
    else:
        _write_deferred_marker(run_dir)
    return outcome


def _write_log_entry(run_dir: Path, log_entry: dict[str, Any]) -> None:
    """Atomic write of the Rekor LogEntry to ``REKOR_OUTPUT_FILENAME``.

    Mirror of the atomic-write pattern in
    :func:`topofuzz.attest.emit_attestation`: tmp file (suffix appended
    AFTER the existing extension so a ``glob('*.json')`` sweep doesn't
    pick it up) then ``Path.replace`` for the rename(2) guarantee.
    """
    out_path = run_dir / REKOR_OUTPUT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    payload = json.dumps(log_entry, sort_keys=True, indent=2)
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(out_path)


def _write_deferred_marker(run_dir: Path) -> None:
    """Touch the ``.rekor-deferred`` marker (idempotent).

    Audit Pass-1 RT MINOR-4 fix: the prior implementation called
    ``run_dir.mkdir(parents=True, exist_ok=True)`` which would
    silently create stray directories on a typo'd run_dir path. This
    function is only ever called from :func:`submit_for_run_dir`
    immediately after a successful ``execute_run`` (the orchestrator
    already minted the directory). If the run dir doesn't exist by
    the time we reach this point, raising :class:`FileNotFoundError`
    surfaces the real bug rather than littering the filesystem.
    """
    marker = run_dir / REKOR_DEFERRED_MARKER
    marker.write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Report banner gates
# ---------------------------------------------------------------------------


def looks_like_rekor_logged(run_dir: Path) -> bool:
    """Return True iff ``run_dir`` carries a parseable Rekor LogEntry.

    Used by the report renderer to know whether to suppress the
    ``[REKOR-DEFERRED]`` banner — if a successful LogEntry is on disk,
    a stale marker (e.g. from a prior failed attempt) shouldn't keep
    the deferred banner up. A non-empty JSON dict is the structural
    check; full Rekor inclusion-proof verification is a downstream
    consumer's job, mirroring the ``_attestation_looks_signed`` gate
    in :mod:`topofuzz.report`.
    """
    path = run_dir / REKOR_OUTPUT_FILENAME
    if not path.exists():
        return False
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        _log.warning(
            "rekor: %s present but unreadable (%s) — banner-gate False",
            REKOR_OUTPUT_FILENAME, e,
        )
        return False
    if not raw:
        return False
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(data, dict) and bool(data)


def looks_like_rekor_deferred(run_dir: Path) -> bool:
    """Return True iff ``run_dir`` carries a deferred marker.

    Note the asymmetry: the marker file's CONTENT isn't read (it's a
    touch-file). Presence is the signal. The report renderer should
    AND this with ``not looks_like_rekor_logged(run_dir)`` so a
    succeeded re-attest doesn't show ``[REKOR-DEFERRED]`` if the
    marker cleanup ever races.
    """
    return (run_dir / REKOR_DEFERRED_MARKER).exists()
