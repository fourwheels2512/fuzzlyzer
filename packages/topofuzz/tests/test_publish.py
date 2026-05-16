"""Tests for ``topofuzz publish`` — buyer-pitch PDF artifact."""

from __future__ import annotations

import base64
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from topofuzz.cli import app
from topofuzz.publish import (
    DEFAULT_MAX_FLIPS,
    MAX_FLIPS_HARD_LIMIT,
    PublishError,
    _format_footer,
    _html_escape,
    build_publish_pdf,
    load_publish_metadata,
    select_top_flipping_evals,
)


# ---------------------------------------------------------------------------
# Synthetic run fixture builder
# ---------------------------------------------------------------------------


def _hash(seed: str) -> str:
    """Deterministic 64-char hex stand-in for a variant_canonical_hash."""
    return (seed * 64)[:64]


def _make_run(
    tmp_path: Path,
    *,
    n_variants: int = 4,
    eval_verdicts: dict[str, list[bool | None]] | None = None,
    with_attestation: bool = False,
    tds_version: str = "0.5.0",
    signed_at: str = "2026-05-15T02:07:48.945561Z",
) -> Path:
    """Build a minimal valid run_dir.

    ``eval_verdicts``: ``{eval_id: [v0, v1, ..., v_{n_variants-1}]}`` where
    each entry is ``True``/``False`` (scored verdict) or ``None`` (judge
    failure). Variant 0 is canonical.
    """
    if eval_verdicts is None:
        eval_verdicts = {
            "eval_001": [True, False, False, False],  # 3 flips
            "eval_002": [True, True, False, True],     # 1 flip
            "eval_003": [False, True, True, False],    # 2 flips
        }
    hashes = [_hash(chr(ord("a") + i)) for i in range(n_variants)]
    eval_ids = sorted(eval_verdicts.keys())
    run_dir = tmp_path / "synthetic-run"
    run_dir.mkdir()

    manifest = {
        "run_id": "publish-synth-0001",
        "topology_canonical_hash": _hash("h"),
        "topology_name": "synth",
        "tds_version": tds_version,
        "tenant_id": "local",
        "n_variants": n_variants,
        "n_eval_rows": len(eval_ids),
        "variant_canonical_hashes": hashes,
        "variant_orderings": [["n1", "n2"] for _ in range(n_variants)],
        "eval_ids": eval_ids,
        "eval_ground_truth": [0 for _ in eval_ids],
        "agent_vendors": ["google"],
        "judge_model": "google/gemini-2.5-flash",
        "rubric_hash": _hash("r"),
        "semaphore_size": 8,
        "max_cost_usd": 1.0,
        "cli_args": {"fixture": "synth", "eval": "eval.csv"},
        "kickoff_ts": 0,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE cells ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, ir_canonical_hash "
        "TEXT, node_outputs_json TEXT, error TEXT, cell_cost_usd REAL)"
    )
    db.execute(
        "CREATE TABLE judge_results ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, judge_kind TEXT, "
        "score_value REAL, passed INTEGER, rationale TEXT, "
        "failure_reason TEXT)"
    )
    for eid, verdicts in eval_verdicts.items():
        for idx, v in enumerate(verdicts):
            if v is None:
                passed: int | None = None
                fail: str | None = "judge_fail"
            else:
                passed = 1 if v else 0
                fail = None
            db.execute(
                "INSERT INTO judge_results VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "publish-synth-0001",
                    hashes[idx],
                    eid,
                    "binary",
                    None,
                    passed,
                    "rat",
                    fail,
                ),
            )
    db.commit()
    db.close()

    if with_attestation:
        payload = {
            "run_id": "publish-synth-0001",
            "signed_at": signed_at,
            "manifest": {"tds_version": tds_version},
        }
        envelope = {
            "payload": base64.b64encode(
                json.dumps(payload).encode()
            ).decode(),
            "payloadType": (
                "application/vnd.topofuzz.run+json;v=0.5"
            ),
            "signatures": [{"keyid": "test", "sig": "stub"}],
        }
        (run_dir / "attestation.json").write_text(
            json.dumps(envelope), encoding="utf-8"
        )

    return run_dir


# ---------------------------------------------------------------------------
# select_top_flipping_evals
# ---------------------------------------------------------------------------


def test_select_top_flipping_evals_orders_by_flip_count_desc(tmp_path):
    """eval_001 has 3 flips, eval_003 has 2, eval_002 has 1."""
    run_dir = _make_run(tmp_path)
    eids = select_top_flipping_evals(run_dir, max_flips=10)
    assert eids == ["eval_001", "eval_003", "eval_002"]


def test_select_top_flipping_evals_lex_tiebreak(tmp_path):
    """Equal flip counts break ties on lexicographic eval_id."""
    run_dir = _make_run(
        tmp_path,
        eval_verdicts={
            "eval_z": [True, False, True, True],   # 1 flip
            "eval_a": [True, False, True, True],   # 1 flip
            "eval_m": [True, False, True, True],   # 1 flip
        },
    )
    eids = select_top_flipping_evals(run_dir, max_flips=10)
    assert eids == ["eval_a", "eval_m", "eval_z"]


def test_select_top_flipping_evals_respects_max_flips(tmp_path):
    """max_flips=2 → only top 2 returned."""
    run_dir = _make_run(tmp_path)
    eids = select_top_flipping_evals(run_dir, max_flips=2)
    assert eids == ["eval_001", "eval_003"]


def test_select_top_flipping_evals_skips_canonical_judge_failures(tmp_path):
    """Eval with no canonical verdict contributes no entry."""
    run_dir = _make_run(
        tmp_path,
        eval_verdicts={
            "eval_001": [None, True, True, True],  # canonical NULL
            "eval_002": [False, True, True, True],  # 3 flips
        },
    )
    eids = select_top_flipping_evals(run_dir, max_flips=10)
    assert eids == ["eval_002"]


def test_select_top_flipping_evals_excludes_stable_evals(tmp_path):
    """Eval where every variant agrees with canonical is not returned."""
    run_dir = _make_run(
        tmp_path,
        eval_verdicts={
            "eval_001": [True, True, True, True],  # all agree
            "eval_002": [False, True, False, False],  # 1 flip
        },
    )
    eids = select_top_flipping_evals(run_dir, max_flips=10)
    assert eids == ["eval_002"]


def test_select_top_flipping_evals_zero_max_returns_empty(tmp_path):
    run_dir = _make_run(tmp_path)
    assert select_top_flipping_evals(run_dir, max_flips=0) == []


def test_select_top_flipping_evals_negative_max_raises(tmp_path):
    run_dir = _make_run(tmp_path)
    with pytest.raises(PublishError, match="non-negative"):
        select_top_flipping_evals(run_dir, max_flips=-1)


def test_select_top_flipping_evals_missing_sqlite_raises(tmp_path):
    run_dir = _make_run(tmp_path)
    (run_dir / "run.sqlite").unlink()
    with pytest.raises(PublishError, match="run.sqlite"):
        select_top_flipping_evals(run_dir)


def test_select_top_flipping_evals_empty_hashes_raises(tmp_path):
    """An empty variant_canonical_hashes list cannot pin a canonical."""
    run_dir = _make_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variant_canonical_hashes"] = []
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PublishError, match="missing or empty"):
        select_top_flipping_evals(run_dir)


# ---------------------------------------------------------------------------
# load_publish_metadata
# ---------------------------------------------------------------------------


def test_load_publish_metadata_reads_attestation_payload(tmp_path):
    """Attestation.json present → DSSE payload wins over manifest.json."""
    run_dir = _make_run(
        tmp_path, with_attestation=True, tds_version="0.5.0"
    )
    meta = load_publish_metadata(run_dir)
    assert meta["run_id"] == "publish-synth-0001"
    assert meta["tds_version"] == "0.5.0"
    assert meta["signed_at"] == "2026-05-15T02:07:48.945561Z"


def test_load_publish_metadata_falls_back_to_manifest(tmp_path):
    """No attestation → use manifest, mark signed_at='(unsigned)'."""
    run_dir = _make_run(tmp_path, with_attestation=False)
    meta = load_publish_metadata(run_dir)
    assert meta["run_id"] == "publish-synth-0001"
    assert meta["tds_version"] == "0.5.0"
    assert meta["signed_at"] == "(unsigned)"


def test_load_publish_metadata_falls_back_when_attestation_corrupt(
    tmp_path,
):
    """A malformed attestation.json must not crash publish.

    Per RT-pass-2 MINOR-N2 the signal is ``(corrupt-attestation)`` to
    distinguish from a legitimately-unsigned bundle.
    """
    run_dir = _make_run(tmp_path, with_attestation=False)
    (run_dir / "attestation.json").write_text("not valid json {")
    meta = load_publish_metadata(run_dir)
    assert meta["run_id"] == "publish-synth-0001"
    assert meta["signed_at"] == "(corrupt-attestation)"


def test_load_publish_metadata_falls_back_when_payload_unparseable(
    tmp_path,
):
    """Attestation with non-decodable payload → fall back, don't raise."""
    run_dir = _make_run(tmp_path, with_attestation=False)
    envelope = {"payload": "not-base64!", "signatures": []}
    (run_dir / "attestation.json").write_text(json.dumps(envelope))
    meta = load_publish_metadata(run_dir)
    assert meta["signed_at"] == "(corrupt-attestation)"


# ---------------------------------------------------------------------------
# _format_footer
# ---------------------------------------------------------------------------


def test_format_footer_canonical_shape():
    text = _format_footer(
        {
            "run_id": "2026-05-15-01KRMNR",
            "tds_version": "0.5.0",
            "signed_at": "2026-05-15T02:07:48.945561Z",
        }
    )
    # Subseconds stripped; 'Z' preserved.
    assert text == (
        "topofuzz run 2026-05-15-01KRMNR · TDS v0.5.0 · "
        "signed 2026-05-15T02:07:48Z"
    )


def test_format_footer_no_subseconds_preserved_as_is():
    text = _format_footer(
        {
            "run_id": "r1",
            "tds_version": "0.5.0",
            "signed_at": "2026-05-15T02:07:48Z",
        }
    )
    assert "2026-05-15T02:07:48Z" in text


def test_format_footer_unsigned():
    text = _format_footer(
        {
            "run_id": "r1",
            "tds_version": "0.5.0",
            "signed_at": "(unsigned)",
        }
    )
    assert "signed (unsigned)" in text


def test_format_footer_missing_fields_default_to_unknown():
    text = _format_footer({})
    assert "(unknown)" in text
    assert "(unsigned)" in text


# ---------------------------------------------------------------------------
# _html_escape (footer-injection defence)
# ---------------------------------------------------------------------------


def test_html_escape_neutralises_script_tag():
    """A hostile run_id must not break out of the footer div."""
    out = _html_escape("</div><script>alert('x')</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_html_escape_handles_ampersand_and_quotes():
    out = _html_escape("a & b \"c\" 'd'")
    assert "&amp;" in out
    assert "&quot;" in out
    assert "&#x27;" in out


# ---------------------------------------------------------------------------
# build_publish_pdf — guard rails (no actual Chromium launch)
# ---------------------------------------------------------------------------


def test_build_publish_pdf_rejects_negative_max_flips(tmp_path):
    run_dir = _make_run(tmp_path)
    with pytest.raises(PublishError, match="non-negative"):
        build_publish_pdf(
            run_dir, out_path=tmp_path / "out.pdf", max_flips=-1
        )


def test_build_publish_pdf_rejects_max_flips_over_hard_limit(tmp_path):
    run_dir = _make_run(tmp_path)
    with pytest.raises(PublishError, match="exceeds hard limit"):
        build_publish_pdf(
            run_dir,
            out_path=tmp_path / "out.pdf",
            max_flips=MAX_FLIPS_HARD_LIMIT + 1,
        )


def test_build_publish_pdf_rejects_missing_run_dir(tmp_path):
    with pytest.raises(PublishError, match="not a directory"):
        build_publish_pdf(
            tmp_path / "nope", out_path=tmp_path / "out.pdf"
        )


def test_build_publish_pdf_rejects_out_path_is_directory(tmp_path):
    run_dir = _make_run(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(PublishError, match="existing directory"):
        build_publish_pdf(run_dir, out_path=tmp_path / "out")


def test_render_html_pages_to_pdfs_raises_when_playwright_missing(
    tmp_path, monkeypatch
):
    """Simulate a verify-only install: hide playwright at import time.

    Tests the lower-level helper because ``build_publish_pdf``
    re-renders ``report.html`` first; a synthetic run_dir doesn't
    satisfy ``topofuzz.report``'s schema requirements. Verifying the
    helper directly pins the "pip install topofuzz[pdf]" surfacing.
    """
    from topofuzz.publish import _render_html_pages_to_pdfs

    stub = tmp_path / "stub.html"
    stub.write_text("<html><body>stub</body></html>", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(PublishError, match="'pdf' install extra"):
        _render_html_pages_to_pdfs([stub], footer_text="x")


def test_merge_pdfs_raises_when_pypdf_missing(monkeypatch):
    """Hide pypdf — verify the helper wraps the ImportError cleanly."""
    from topofuzz.publish import _merge_pdfs

    monkeypatch.setitem(sys.modules, "pypdf", None)
    monkeypatch.setitem(sys.modules, "pypdf.errors", None)
    with pytest.raises(PublishError, match="'pdf' install extra"):
        # Two distinct stub blobs force the merge code path.
        _merge_pdfs([b"%PDF-1.4 stub", b"%PDF-1.4 stub"])


def test_merge_pdfs_single_blob_returns_unchanged():
    """A list of one PDF skips the merge step entirely."""
    from topofuzz.publish import _merge_pdfs

    blob = b"%PDF-1.4\nfake-single-page"
    assert _merge_pdfs([blob]) is blob


def test_merge_pdfs_empty_list_raises():
    from topofuzz.publish import _merge_pdfs

    with pytest.raises(PublishError, match="no PDF pages to merge"):
        _merge_pdfs([])


def test_merge_pdfs_wraps_corrupt_blob_as_publish_error():
    """A malformed PDF blob → PdfReadError → PublishError translation.

    Skipped on environments without ``topofuzz[pdf]`` installed —
    the lower-level helper short-circuits to the "pdf install extra"
    error before reaching the pypdf parse path.
    """
    pytest.importorskip("pypdf")
    from topofuzz.publish import _merge_pdfs

    # Two blobs (forces merge path), both intentionally not valid PDFs
    # so pypdf raises PdfReadError on the first read.
    with pytest.raises(PublishError, match="pypdf refused"):
        _merge_pdfs([b"not a pdf", b"also not a pdf"])


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_publish_bad_usage_when_run_dir_missing(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "publish",
            str(tmp_path / "nope"),
            "--pdf",
            str(tmp_path / "out.pdf"),
        ],
    )
    assert result.exit_code == 2
    assert "not a directory" in result.output


def test_cli_publish_propagates_publish_error_as_exit_1(tmp_path):
    """A PublishError from the module → CLI prints message, exits 1.

    Triggers the error via ``--pdf`` pointing at an existing directory
    (caught early in ``build_publish_pdf``) — keeps the test free of
    playwright/render_report dependencies.
    """
    run_dir = _make_run(tmp_path)
    (tmp_path / "is_a_dir").mkdir()

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "publish",
            str(run_dir),
            "--pdf",
            str(tmp_path / "is_a_dir"),
            "--max-flips",
            "0",
        ],
    )
    assert result.exit_code == 1
    assert "topofuzz publish:" in result.output
    assert "existing directory" in result.output


def test_cli_publish_default_max_flips_matches_module_constant():
    """The CLI Typer default tracks the module's DEFAULT_MAX_FLIPS.

    Reaches into Click's compiled ``Command`` object so a future
    refactor that drifts the literal `5` away from
    ``DEFAULT_MAX_FLIPS = 5`` fails this test rather than silently
    shipping mismatched docs vs behaviour.
    """
    # Click's Typer compiles the @app.command into the app's
    # registered_commands dict; the param defaults are reachable via
    # ``params``. Look up the publish command and find --max-flips.
    publish_cmd = None
    for cmd in app.registered_commands:
        if cmd.callback.__name__ == "publish":
            publish_cmd = cmd
            break
    assert publish_cmd is not None, "publish command not registered"
    # The Typer ParameterInfo for --max-flips lives on the callback
    # signature as a default value (``typer.Option(5, ...)``).
    import inspect as _inspect
    sig = _inspect.signature(publish_cmd.callback)
    default = sig.parameters["max_flips"].default
    # Typer wraps it in a ParameterInfo; the underlying default is the
    # first positional arg, accessible via .default.
    cli_default = (
        default.default if hasattr(default, "default") else default
    )
    assert cli_default == DEFAULT_MAX_FLIPS


# ---------------------------------------------------------------------------
# Additional adversarial / robustness coverage (audit pass-1)
# ---------------------------------------------------------------------------


def test_select_top_flipping_evals_rejects_hostile_eval_id(tmp_path):
    """BLOCKER (audit RT-pass-1 B1): a ``../`` eval_id in run.sqlite is
    refused before it can land in the inspect_<eid>.html path.
    """
    run_dir = _make_run(
        tmp_path,
        eval_verdicts={
            "../etc/passwd": [True, False, False, False],
        },
    )
    with pytest.raises(PublishError, match="not permitted"):
        select_top_flipping_evals(run_dir)


def test_select_top_flipping_evals_score_value_branch(tmp_path):
    """When ``passed`` is NULL but ``score_value`` is present, the
    threshold path produces the verdict (covers the previously-untested
    elif branch).
    """
    run_dir = _make_run(tmp_path)
    # Overwrite judge_results with score_value-only rows.
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute("DELETE FROM judge_results")
    hashes = json.loads((run_dir / "manifest.json").read_text())[
        "variant_canonical_hashes"
    ]
    # Canonical (idx 0) above threshold; other 3 below → 3 flips.
    rows = [
        ("publish-synth-0001", hashes[0], "eval_a", "binary",
         0.9, None, "r", None),
        ("publish-synth-0001", hashes[1], "eval_a", "binary",
         0.1, None, "r", None),
        ("publish-synth-0001", hashes[2], "eval_a", "binary",
         0.2, None, "r", None),
        ("publish-synth-0001", hashes[3], "eval_a", "binary",
         0.3, None, "r", None),
    ]
    db.executemany(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    db.commit()
    db.close()
    eids = select_top_flipping_evals(run_dir, max_flips=10)
    assert eids == ["eval_a"]


def test_select_top_flipping_evals_silently_drops_unknown_variant_id(
    tmp_path,
):
    """A ``variant_id`` not in ``manifest.variant_canonical_hashes`` is
    silently dropped (mirrors inspect.py forward-compat behaviour). The
    selection still works on the recognised variants.
    """
    run_dir = _make_run(tmp_path)
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("publish-synth-0001", _hash("z"), "eval_001", "binary",
         None, 0, "r", None),  # unknown variant_id
    )
    db.commit()
    db.close()
    eids = select_top_flipping_evals(run_dir, max_flips=10)
    # Same as the default fixture's ordering: unknown row was silently
    # dropped, the recognised flips still rank correctly.
    assert eids == ["eval_001", "eval_003", "eval_002"]


def test_load_publish_metadata_missing_tds_version_in_payload(tmp_path):
    """Payload with ``manifest`` but no ``tds_version`` → empty string,
    not crash. ``_format_footer`` then surfaces ``(unknown)`` for the
    TDS field via its empty-string default-fallback.
    """
    run_dir = _make_run(tmp_path, with_attestation=False)
    payload = {
        "run_id": "rid",
        "signed_at": "2026-05-15T00:00:00Z",
        "manifest": {},  # no tds_version
    }
    envelope = {
        "payload": base64.b64encode(json.dumps(payload).encode()).decode(),
        "payloadType": "application/vnd.topofuzz.run+json;v=0.5",
        "signatures": [{"keyid": "k", "sig": "s"}],
    }
    (run_dir / "attestation.json").write_text(json.dumps(envelope))
    meta = load_publish_metadata(run_dir)
    assert meta["tds_version"] == ""
    # _format_footer surfaces "(unknown)" for an empty string per its
    # ``or "(unknown)"`` fallback.
    footer = _format_footer(meta)
    assert "TDS v(unknown)" in footer


def test_format_footer_malformed_signed_at_passes_through(tmp_path):
    """A non-ISO signed_at (e.g. ``"garbage"`` or ``"not.a.date"``) is
    surfaced verbatim — the subsecond-strip only fires when the string
    looks like ISO-8601 with subseconds + trailing Z.
    """
    # Garbage with no T, no Z: passes through unchanged.
    f1 = _format_footer(
        {"run_id": "r", "tds_version": "v", "signed_at": "garbage"}
    )
    assert "signed garbage" in f1

    # Looks like a date but no T → passes through unchanged
    # (strip requires both T and Z).
    f2 = _format_footer(
        {"run_id": "r", "tds_version": "v", "signed_at": "2026.05.15.foo"}
    )
    assert "2026.05.15.foo" in f2

    # Has T + . but no trailing Z → passes through unchanged
    # (strip requires Z to anchor the rewrite).
    f3 = _format_footer(
        {"run_id": "r", "tds_version": "v",
         "signed_at": "2026-05-15T00:00:00.000+00:00"}
    )
    assert "2026-05-15T00:00:00.000+00:00" in f3


def test_select_top_flipping_evals_uses_readonly_sqlite(tmp_path):
    """publish must not create a ``-journal`` sibling that could race a
    concurrent ``topofuzz run`` writer (audit RT-pass-1 MAJOR-3).

    Strengthened per RT-pass-2 NIT-N4: also probe the connection
    behaviour by re-opening the same path the same way and asserting
    a write attempt raises ``OperationalError`` for "readonly database".
    """
    run_dir = _make_run(tmp_path)
    journal_before = list(run_dir.glob("run.sqlite-*"))
    select_top_flipping_evals(run_dir, max_flips=3)
    journal_after = list(run_dir.glob("run.sqlite-*"))
    assert journal_after == journal_before, (
        f"publish created a sqlite journal file: "
        f"{set(journal_after) - set(journal_before)}"
    )
    # Reproduce publish's connection shape and verify it's read-only.
    sqlite_path = run_dir / "run.sqlite"
    con = sqlite3.connect(
        f"file:{sqlite_path}?mode=ro", uri=True
    )
    try:
        with pytest.raises(
            sqlite3.OperationalError, match="readonly database"
        ):
            con.execute(
                "INSERT INTO judge_results VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)",
                ("x", "x", "x", "x", None, 0, None, None),
            )
    finally:
        con.close()


def test_select_top_flipping_evals_rejects_trailing_newline_eid(tmp_path):
    """Audit RT-pass-2 MINOR-N1: an eval_id ending in ``\\n`` would slip
    past ``re.match(r"^...+$", ...)`` because Python's ``$`` matches
    before a trailing newline. ``fullmatch`` rejects it.
    """
    run_dir = _make_run(
        tmp_path,
        eval_verdicts={
            "lookslegit\n": [True, False, False, False],
        },
    )
    with pytest.raises(PublishError, match="not permitted"):
        select_top_flipping_evals(run_dir)


def test_load_publish_metadata_corrupt_attestation_distinguishable(
    tmp_path,
):
    """Audit RT-pass-2 MINOR-N2: when ``attestation.json`` is present
    but unreadable, the footer breadcrumb reads
    ``(corrupt-attestation)``, NOT ``(unsigned)``, so an attacker who
    swaps a real attestation for a corrupt one cannot silently
    masquerade as a legitimately-unsigned bundle.
    """
    run_dir = _make_run(tmp_path, with_attestation=False)
    (run_dir / "attestation.json").write_text(
        "not valid json {", encoding="utf-8"
    )
    meta = load_publish_metadata(run_dir)
    assert meta["signed_at"] == "(corrupt-attestation)"
    # And the footer surfaces it verbatim.
    footer = _format_footer(meta)
    assert "(corrupt-attestation)" in footer
    assert "(unsigned)" not in footer
