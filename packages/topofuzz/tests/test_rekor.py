"""Step 18: topofuzz.rekor — mocked unit tests for the Rekor opt-in path.

Tests use a fake ``opener`` callable to simulate Rekor responses without
real network. Per step-18 decision 3 the live smoke against the
Sigstore Public Good Instance happens manually before commit; only the
mocked paths run in the test suite (CI deterministic, no Sigstore
outage flakiness).

The ``opener`` seam in :func:`topofuzz.rekor._post_once` lets each test
inject a fake without monkey-patching :mod:`urllib.request` globally —
keeps test scopes explicit and parallel-safe.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
from pathlib import Path
from typing import Any, Callable

import pytest

from topofuzz.rekor import (
    BACKOFF_DELAYS_S,
    DEFAULT_REKOR_URL,
    MAX_ATTEMPTS,
    MAX_RESPONSE_BYTES,
    PER_ATTEMPT_TIMEOUT_S,
    REKOR_DEFERRED_MARKER,
    REKOR_OUTPUT_FILENAME,
    RekorSubmitOutcome,
    _build_proposed_entry,
    looks_like_rekor_deferred,
    looks_like_rekor_logged,
    submit_for_run_dir,
    submit_to_rekor,
)


# ---------------------------------------------------------------------------
# Fake-opener helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Stand-in for the context manager returned by ``urllib.request.urlopen``.

    Mirrors the surface ``_post_once`` actually touches: ``.status``,
    ``.read(N)``, and ``.headers`` (audit pass-1 RT MINOR-2: headers
    are now read for ``Location`` extraction on the 409 path).
    """

    def __init__(
        self,
        status: int,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._buf = io.BytesIO(body)
        self.headers = headers or {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._buf.close()

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)


def _http_error(
    code: int,
    body: bytes,
    headers: dict[str, str] | None = None,
) -> urllib.error.HTTPError:
    """Build a HTTPError that exposes both ``.read()`` body and ``.headers``."""
    err = urllib.error.HTTPError(
        url="x",
        code=code,
        msg=f"HTTP {code}",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body),
    )
    # _headers_to_dict tolerates dict-like .items() so a plain dict
    # works as a stand-in for email.message.Message.
    err.headers = headers or {}  # type: ignore[assignment]
    return err


def _opener_seq(*responses: Any) -> Callable[..., Any]:
    """Build an opener that yields the given responses in order.

    Each entry is either a ``_FakeResponse`` (returned to the caller),
    an Exception subclass to raise, or an Exception instance to raise.
    Raises ``StopIteration`` if the caller exceeds the script — caught
    by the test as a contract violation (we expected fewer attempts).
    """
    it = iter(responses)

    def _open(req: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            nxt = next(it)
        except StopIteration:
            raise AssertionError(
                "fake opener exhausted — code under test made more "
                "POSTs than the test scripted"
            )
        if isinstance(nxt, BaseException):
            raise nxt
        if isinstance(nxt, type) and issubclass(nxt, BaseException):
            raise nxt("scripted")
        return nxt

    return _open


def _no_sleep(_: float) -> None:
    """Replacement for ``time.sleep`` — keeps tests at-speed."""
    return None


_SAMPLE_ENVELOPE = (
    b'{"payloadType":"application/vnd.topofuzz.run+json;v=0.5",'
    b'"payload":"eyJoZWxsbyI6ICJ3b3JsZCJ9",'
    b'"signatures":[{"keyid":"dev","sig":"AAAA"}]}'
)
_SAMPLE_PUBLIC_PEM = (
    b"-----BEGIN PUBLIC KEY-----\n"
    b"MCowBQYDK2VwAyEAVGyiUYBeYj0Y7Idz9KCEbKZAyWX7CSGCmwwTfRoyOhI=\n"
    b"-----END PUBLIC KEY-----\n"
)
_SAMPLE_LOG_ENTRY = {
    "abc123uuid": {
        "logIndex": 4242,
        "integratedTime": 1700000000,
        "body": "base64-canonical-body",
        "verification": {"signedEntryTimestamp": "..."},
    },
}


# ---------------------------------------------------------------------------
# _build_proposed_entry
# ---------------------------------------------------------------------------


def test_build_proposed_entry_matches_rekor_dsse_v0_0_1_schema() -> None:
    """Audit Pass-1 RT BLOCKER-1 fix: shape is
    ``spec.proposedContent.{envelope, verifiers}`` (verified against
    sigstore/rekor's published JSON schema). ``envelope`` is the
    literal DSSE JSON string; ``verifiers`` is a non-empty array of
    base64-encoded PEM verification material. The prior shape
    (``spec.envelope`` + ``spec.publicKey``) would 400 on every live
    submission."""
    body = _build_proposed_entry(_SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM)
    parsed = json.loads(body)
    assert parsed["apiVersion"] == "0.0.1"
    assert parsed["kind"] == "dsse"
    spec = parsed["spec"]
    proposed = spec["proposedContent"]
    # envelope is the literal DSSE JSON, not base64-wrapped.
    assert proposed["envelope"] == _SAMPLE_ENVELOPE.decode("utf-8")
    # verifiers is an ARRAY (minItems: 1) of base64-encoded PEM bytes.
    assert isinstance(proposed["verifiers"], list)
    assert len(proposed["verifiers"]) >= 1
    assert base64.b64decode(proposed["verifiers"][0]) == _SAMPLE_PUBLIC_PEM
    # The legacy shape MUST NOT be present (regression guard).
    assert "envelope" not in spec
    assert "publicKey" not in spec


def test_build_proposed_entry_envelope_invalid_utf8_raises_value_error() -> None:
    """Audit Pass-1 CR MAJOR-1 fix: a non-UTF-8 envelope (would only
    happen if a future signer changed the on-disk encoding) raises a
    :class:`ValueError` with operator-readable context — NOT a bare
    :class:`UnicodeDecodeError` that would crash the CLI through the
    too-narrow exception filter."""
    with pytest.raises(ValueError) as exc_info:
        _build_proposed_entry(b"\xff\xfe\xfd", _SAMPLE_PUBLIC_PEM)
    assert "not valid UTF-8" in str(exc_info.value)


def test_build_proposed_entry_byte_stable() -> None:
    """Audit Pass-1 RT INFO-3: dict insertion order is deterministic in
    CPython 3.7+, so two calls on identical inputs produce byte-identical
    bodies. Locks the property in place for the rc1→GA byte-stable
    contract."""
    a = _build_proposed_entry(_SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM)
    b = _build_proposed_entry(_SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM)
    assert a == b


# ---------------------------------------------------------------------------
# submit_to_rekor — success / retry / failure paths
# ---------------------------------------------------------------------------


def test_submit_to_rekor_success_first_attempt() -> None:
    """200 + parseable JSON dict on attempt 1 → ok=True, log_entry parsed."""
    body = json.dumps(_SAMPLE_LOG_ENTRY).encode("utf-8")
    opener = _opener_seq(_FakeResponse(200, body))
    outcome = submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is True
    assert outcome.attempts == 1
    assert outcome.error is None
    assert outcome.log_entry == _SAMPLE_LOG_ENTRY
    assert outcome.url == DEFAULT_REKOR_URL


def test_submit_to_rekor_retries_on_5xx_then_succeeds() -> None:
    """502 once, then 200 — should burn one retry and report attempts=2."""
    success_body = json.dumps(_SAMPLE_LOG_ENTRY).encode("utf-8")
    opener = _opener_seq(
        _http_error(502, b"upstream unreachable"),
        _FakeResponse(200, success_body),
    )
    outcome = submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is True
    assert outcome.attempts == 2
    assert outcome.log_entry == _SAMPLE_LOG_ENTRY


def test_submit_to_rekor_retries_on_network_error_then_succeeds() -> None:
    """URLError on attempt 1, success on attempt 2."""
    success_body = json.dumps(_SAMPLE_LOG_ENTRY).encode("utf-8")
    opener = _opener_seq(
        urllib.error.URLError("connection refused"),
        _FakeResponse(200, success_body),
    )
    outcome = submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is True
    assert outcome.attempts == 2


def test_submit_to_rekor_4xx_is_terminal_no_retry() -> None:
    """400 should NOT trigger retry — the payload is broken; retry won't
    fix it. Outcome surfaces the body preview so the operator can fix it."""
    opener = _opener_seq(_http_error(400, b"invalid envelope shape"))
    outcome = submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is False
    assert outcome.attempts == 1, "must NOT retry on 4xx"
    assert outcome.error is not None
    assert "400" in outcome.error
    assert "invalid envelope shape" in outcome.error
    assert outcome.log_entry is None


def test_submit_to_rekor_all_attempts_fail() -> None:
    """All three attempts time out → ok=False, attempts=MAX_ATTEMPTS,
    error is a network-error string. The CLI uses this state to write
    the [REKOR-DEFERRED] marker."""
    opener = _opener_seq(
        urllib.error.URLError("network down 1"),
        urllib.error.URLError("network down 2"),
        urllib.error.URLError("network down 3"),
    )
    outcome = submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is False
    assert outcome.attempts == MAX_ATTEMPTS
    assert outcome.log_entry is None
    assert outcome.error is not None
    assert "network down 3" in outcome.error  # last error wins


def test_submit_to_rekor_2xx_with_unparseable_body_is_terminal() -> None:
    """200 with broken JSON is treated as terminal — the server is
    misbehaving and another 10s round-trip won't fix it."""
    opener = _opener_seq(_FakeResponse(200, b"not-json{{{"))
    outcome = submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is False
    assert outcome.attempts == 1
    assert outcome.error is not None
    assert "unparseable" in outcome.error


def test_submit_to_rekor_2xx_with_non_dict_body_is_terminal() -> None:
    """200 with a JSON array (not a dict) is structurally wrong for
    Rekor's response shape. Reject with a clear error."""
    opener = _opener_seq(_FakeResponse(200, b"[\"not\", \"a\", \"dict\"]"))
    outcome = submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is False
    err = outcome.error or ""
    assert "non-dict" in err or "empty" in err


def test_submit_to_rekor_2xx_with_empty_dict_body_is_terminal() -> None:
    """Audit Pass-1 RT MAJOR-4 fix: a 2xx with body ``{}`` is rejected.
    Without this, ``submit_to_rekor`` would return ok=True with an
    empty log_entry, ``submit_for_run_dir`` would persist ``{}``, but
    ``looks_like_rekor_logged`` requires a NON-empty dict so the
    deferred banner would falsely light. Tightening here keeps the
    success-contract and the gate consistent."""
    opener = _opener_seq(_FakeResponse(200, b"{}"))
    outcome = submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is False
    assert outcome.log_entry is None
    assert "empty" in (outcome.error or "").lower()


def test_submit_to_rekor_409_returns_already_logged_stub() -> None:
    """Audit Pass-1 RT MAJOR-1 fix: 409 Conflict is special-cased as a
    success outcome (the envelope is in Rekor server-side; we just
    don't have a fresh LogEntry). The stub carries the Location URL
    so the operator can GET the existing entry."""
    opener = _opener_seq(
        _http_error(
            409,
            b"entry already exists",
            headers={"Location": "/api/v1/log/entries/abc-uuid"},
        )
    )
    outcome = submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is True
    assert outcome.attempts == 1
    assert outcome.log_entry is not None
    assert outcome.log_entry.get("_rekor_409_already_logged") is True
    assert outcome.log_entry.get("location_url") == "/api/v1/log/entries/abc-uuid"


def test_submit_to_rekor_409_without_location_header_still_succeeds() -> None:
    """A 409 without a Location header (theoretical, but not impossible)
    still returns ok=True with an empty location_url. The operator is
    informed but not blocked."""
    opener = _opener_seq(_http_error(409, b"conflict"))
    outcome = submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is True
    assert outcome.log_entry is not None
    assert outcome.log_entry.get("location_url") == ""


def test_submit_to_rekor_unicode_envelope_returns_failure_not_crash() -> None:
    """Audit Pass-1 CR MAJOR-1 fix: a non-UTF-8 envelope returns a
    structured ok=False outcome, NOT an uncaught UnicodeDecodeError.
    The CLI thus prints a clean ``[REKOR-DEFERRED]`` line instead of
    crashing the post-run flow."""
    opener = _opener_seq()  # Must NOT be called.
    outcome = submit_to_rekor(
        b"\xff\xfe\xfd", _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is False
    assert outcome.attempts == 0
    assert outcome.error is not None
    assert "UTF-8" in outcome.error


def test_submit_to_rekor_backoff_delays_invoked() -> None:
    """Sleeps between attempts use BACKOFF_DELAYS_S in order. Verifies
    we honor the bounded-backoff contract (no silent infinite retry)."""
    delays_observed: list[float] = []

    def record_sleep(d: float) -> None:
        delays_observed.append(d)

    opener = _opener_seq(
        urllib.error.URLError("e1"),
        urllib.error.URLError("e2"),
        urllib.error.URLError("e3"),
    )
    submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=record_sleep, opener=opener,
    )
    # Two sleeps between three attempts; sequence matches the constants.
    assert delays_observed == list(BACKOFF_DELAYS_S[:2])


def test_submit_to_rekor_oversized_response_body_is_caught() -> None:
    """A response larger than ``MAX_RESPONSE_BYTES`` is treated as a
    network error (retryable) rather than silently truncating. Without
    this guard, a misrouted response (e.g. an HTML error page from a
    transparent proxy) could pin memory or fill the run dir."""
    big = b"x" * (MAX_RESPONSE_BYTES + 100)
    opener = _opener_seq(
        _FakeResponse(200, big),
        _FakeResponse(200, big),
        _FakeResponse(200, big),
    )
    outcome = submit_to_rekor(
        _SAMPLE_ENVELOPE, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is False
    assert outcome.attempts == MAX_ATTEMPTS
    assert outcome.error is not None
    assert "exceeds" in outcome.error.lower() or "cap" in outcome.error.lower()


# ---------------------------------------------------------------------------
# submit_for_run_dir — file-side persistence
# ---------------------------------------------------------------------------


def _seed_attestation(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "attestation.json").write_bytes(_SAMPLE_ENVELOPE)


def test_submit_for_run_dir_success_writes_log_entry_atomic(
    tmp_path: Path,
) -> None:
    """Success path: writes ``REKOR_OUTPUT_FILENAME`` with parsed entry;
    no ``.tmp`` artifact left over; no deferred marker."""
    run_dir = tmp_path / "run-success"
    _seed_attestation(run_dir)
    body = json.dumps(_SAMPLE_LOG_ENTRY).encode("utf-8")
    opener = _opener_seq(_FakeResponse(200, body))

    outcome = submit_for_run_dir(
        run_dir, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is True

    out_path = run_dir / REKOR_OUTPUT_FILENAME
    assert out_path.exists()
    persisted = json.loads(out_path.read_text(encoding="utf-8"))
    assert persisted == _SAMPLE_LOG_ENTRY
    # No tmp left over.
    assert not (run_dir / (REKOR_OUTPUT_FILENAME + ".tmp")).exists()
    # No deferred marker on a success.
    assert not (run_dir / REKOR_DEFERRED_MARKER).exists()


def test_submit_for_run_dir_failure_writes_deferred_marker(
    tmp_path: Path,
) -> None:
    """Failure path: NO rekor.json output, deferred marker present."""
    run_dir = tmp_path / "run-deferred"
    _seed_attestation(run_dir)
    opener = _opener_seq(
        urllib.error.URLError("net1"),
        urllib.error.URLError("net2"),
        urllib.error.URLError("net3"),
    )
    outcome = submit_for_run_dir(
        run_dir, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is False
    assert not (run_dir / REKOR_OUTPUT_FILENAME).exists()
    assert (run_dir / REKOR_DEFERRED_MARKER).exists()


def test_submit_for_run_dir_success_clears_stale_deferred_marker(
    tmp_path: Path,
) -> None:
    """If a prior attempt left a stale marker, a successful re-attest
    must clear it — otherwise the banner stays stuck on a now-logged run."""
    run_dir = tmp_path / "run-restored"
    _seed_attestation(run_dir)
    # Pretend a prior attempt failed.
    (run_dir / REKOR_DEFERRED_MARKER).write_text("", encoding="utf-8")

    body = json.dumps(_SAMPLE_LOG_ENTRY).encode("utf-8")
    opener = _opener_seq(_FakeResponse(200, body))
    outcome = submit_for_run_dir(
        run_dir, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is True
    assert (run_dir / REKOR_OUTPUT_FILENAME).exists()
    assert not (run_dir / REKOR_DEFERRED_MARKER).exists()


def test_submit_for_run_dir_no_attestation_writes_deferred_marker(
    tmp_path: Path,
) -> None:
    """If attestation.json is missing (run wasn't signed), report a
    structured failure AND drop the deferred marker so the report
    surfaces the gap — silent no-op would be worse UX."""
    run_dir = tmp_path / "run-unsigned"
    run_dir.mkdir(parents=True, exist_ok=True)
    # NO attestation.json.

    outcome = submit_for_run_dir(
        run_dir, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep,
        # Opener must never be called when there's no attestation.
        opener=_opener_seq(),
    )
    assert outcome.ok is False
    assert outcome.attempts == 0
    assert outcome.error is not None
    assert "attestation.json" in outcome.error
    assert (run_dir / REKOR_DEFERRED_MARKER).exists()


def test_submit_for_run_dir_persists_log_entry_sorted_keys(
    tmp_path: Path,
) -> None:
    """Persisted rekor.json is sorted-keys + indented so a human can
    diff two LogEntries without a JSON-aware tool."""
    run_dir = tmp_path / "run-sorted"
    _seed_attestation(run_dir)
    body = json.dumps(_SAMPLE_LOG_ENTRY).encode("utf-8")
    submit_for_run_dir(
        run_dir, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=_opener_seq(_FakeResponse(200, body)),
    )
    raw = (run_dir / REKOR_OUTPUT_FILENAME).read_text(encoding="utf-8")
    # Indented (multi-line) and the first inner key after the uuid
    # wrapper would be the alphabetically-first one (``body``).
    assert "\n" in raw
    # Sort property: re-serialize after parse and compare to raw text
    # modulo trailing newline.
    re_serialized = json.dumps(json.loads(raw), sort_keys=True, indent=2)
    assert raw.rstrip("\n") == re_serialized.rstrip("\n")


def test_submit_for_run_dir_forwards_backoff_delays_s(tmp_path: Path) -> None:
    """Audit Pass-1 CR MINOR-2 fix: ``backoff_delays_s`` MUST flow from
    ``submit_for_run_dir`` through to ``submit_to_rekor`` — a custom
    ``max_attempts`` paired with a default backoff would otherwise
    overshoot the user's intended timing budget."""
    run_dir = tmp_path / "run-backoff"
    _seed_attestation(run_dir)
    delays_observed: list[float] = []
    opener = _opener_seq(
        urllib.error.URLError("e1"),
        urllib.error.URLError("e2"),
    )
    submit_for_run_dir(
        run_dir, _SAMPLE_PUBLIC_PEM,
        max_attempts=2,
        backoff_delays_s=(0.7,),  # custom, non-default
        sleep=delays_observed.append,
        opener=opener,
    )
    # Two attempts → exactly one inter-attempt sleep, value matches.
    assert delays_observed == [0.7]


def test_submit_for_run_dir_baseexception_on_write_dumps_log_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Audit Pass-1 RT BLOCKER-2 / MAJOR-2 fix: a Rekor-accepted entry
    + a disk-write failure (or KeyboardInterrupt mid-persist) must
    print the LogEntry JSON to stderr so the operator can manually
    save it. The signed receipt is unrecoverable from the run dir
    otherwise."""
    from topofuzz import rekor as rekor_mod

    run_dir = tmp_path / "run-baseexc"
    _seed_attestation(run_dir)

    def _boom(_run_dir: Path, _entry: dict[str, Any]) -> None:
        raise KeyboardInterrupt("simulated Ctrl-C between POST and persist")

    monkeypatch.setattr(rekor_mod, "_write_log_entry", _boom)

    body = json.dumps(_SAMPLE_LOG_ENTRY).encode("utf-8")
    opener = _opener_seq(_FakeResponse(200, body))

    with pytest.raises(KeyboardInterrupt):
        submit_for_run_dir(
            run_dir, _SAMPLE_PUBLIC_PEM,
            sleep=_no_sleep, opener=opener,
        )

    captured = capsys.readouterr()
    # Both the warning preamble and the LogEntry JSON should land on
    # stderr so the user has the receipt in their terminal.
    assert "[REKOR] WARNING" in captured.err
    assert "abc123uuid" in captured.err
    assert "\"logIndex\": 4242" in captured.err


def test_submit_for_run_dir_409_persists_stub_and_clears_marker(
    tmp_path: Path,
) -> None:
    """When Rekor returns 409, ``submit_for_run_dir`` persists the stub
    LogEntry (so ``looks_like_rekor_logged`` becomes True and the
    deferred banner is suppressed) and removes any prior marker."""
    run_dir = tmp_path / "run-409"
    _seed_attestation(run_dir)
    (run_dir / REKOR_DEFERRED_MARKER).write_text("", encoding="utf-8")
    opener = _opener_seq(
        _http_error(
            409, b"already",
            headers={"Location": "/api/v1/log/entries/dup-uuid"},
        )
    )
    outcome = submit_for_run_dir(
        run_dir, _SAMPLE_PUBLIC_PEM,
        sleep=_no_sleep, opener=opener,
    )
    assert outcome.ok is True
    persisted_path = run_dir / REKOR_OUTPUT_FILENAME
    assert persisted_path.exists()
    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert persisted["_rekor_409_already_logged"] is True
    assert persisted["location_url"] == "/api/v1/log/entries/dup-uuid"
    assert not (run_dir / REKOR_DEFERRED_MARKER).exists()


def test_write_deferred_marker_raises_when_run_dir_missing(
    tmp_path: Path,
) -> None:
    """Audit Pass-1 RT MINOR-4 fix: ``_write_deferred_marker`` no longer
    silently mints the run dir on a typo'd path. A wrong run_dir
    surfaces as :class:`FileNotFoundError` rather than littering the
    filesystem with empty marker files."""
    from topofuzz.rekor import _write_deferred_marker

    nonexistent = tmp_path / "this-dir-does-not-exist"
    assert not nonexistent.exists()
    with pytest.raises(FileNotFoundError):
        _write_deferred_marker(nonexistent)
    # And it didn't get created as a side-effect.
    assert not nonexistent.exists()


# ---------------------------------------------------------------------------
# Banner-gate helpers
# ---------------------------------------------------------------------------


def test_looks_like_rekor_logged_true_only_on_parseable_dict(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-gate"
    run_dir.mkdir()
    # Absent → False.
    assert looks_like_rekor_logged(run_dir) is False
    # Empty file → False (not a dict).
    out = run_dir / REKOR_OUTPUT_FILENAME
    out.write_text("", encoding="utf-8")
    assert looks_like_rekor_logged(run_dir) is False
    # Garbage → False.
    out.write_text("not-json{{{", encoding="utf-8")
    assert looks_like_rekor_logged(run_dir) is False
    # Empty dict → False (no LogEntry inside).
    out.write_text("{}", encoding="utf-8")
    assert looks_like_rekor_logged(run_dir) is False
    # Real-ish entry → True.
    out.write_text(json.dumps(_SAMPLE_LOG_ENTRY), encoding="utf-8")
    assert looks_like_rekor_logged(run_dir) is True


def test_looks_like_rekor_deferred_pure_presence_check(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-marker"
    run_dir.mkdir()
    assert looks_like_rekor_deferred(run_dir) is False
    (run_dir / REKOR_DEFERRED_MARKER).write_text("", encoding="utf-8")
    assert looks_like_rekor_deferred(run_dir) is True


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_module_constants_match_step18_decisions() -> None:
    """Lock the bounded-retry contract: 3 attempts, 10s per attempt,
    delays (1, 2). A future refactor changing these without updating
    the BLUEPRINT line-306 comment + handoff memory would break the
    documented behavior."""
    assert MAX_ATTEMPTS == 3
    assert PER_ATTEMPT_TIMEOUT_S == 10.0
    assert BACKOFF_DELAYS_S == (1.0, 2.0)
    assert DEFAULT_REKOR_URL == "https://rekor.sigstore.dev/api/v1/log/entries"
    assert REKOR_OUTPUT_FILENAME == "report.json.dsse.rekor.json"
    assert REKOR_DEFERRED_MARKER == ".rekor-deferred"


def test_rekor_submit_outcome_is_frozen() -> None:
    """RekorSubmitOutcome is a frozen dataclass — callers shouldn't
    mutate it (returns flow through CLI prints + report banner gate)."""
    outcome = RekorSubmitOutcome(
        ok=True, log_entry={}, error=None, attempts=1, url="x",
    )
    with pytest.raises((AttributeError, Exception)):
        outcome.ok = False  # type: ignore[misc]
