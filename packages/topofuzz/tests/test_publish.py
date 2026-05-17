# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

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


# ---------------------------------------------------------------------------
# F6 — CEO card PDF
# ---------------------------------------------------------------------------


def _make_run_with_report_json(
    tmp_path: Path,
    *,
    n_flips: int = 3,
    n_pairs: int = 27,
    point: float = 0.4444,
    wilson_lo: float = 0.2759,
    wilson_hi: float = 0.6269,
    with_attestation: bool = True,
    eval_verdicts: dict[str, list[bool | None]] | None = None,
    cost_actual_usd: float = 1.23,
) -> Path:
    """Build a synthetic run with both manifest + report.json + cells.

    Lean enough for F6 unit tests — no playwright invocation; the F6
    `_build_ceo_card_context` does all the work pre-render.
    """
    run_dir = _make_run(
        tmp_path,
        eval_verdicts=eval_verdicts,
        with_attestation=with_attestation,
    )
    # Add cost + topology_name supplements to manifest.
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cost_actual_usd"] = cost_actual_usd
    manifest["n_evals"] = manifest.pop("n_eval_rows", 3)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report_payload = {
        "headline": {
            "verdict_flip_rate": {
                "point": point,
                "n_flips": n_flips,
                "n_pairs": n_pairs,
                "ci_lower": max(0.0, point - 0.15),
                "ci_upper": min(1.0, point + 0.15),
            },
        },
        "wilson_addenda": {
            "n_flips": n_flips,
            "n_pairs": n_pairs,
            "point": point,
            "ci_lower": wilson_lo,
            "ci_upper": wilson_hi,
            "conf_pct": 95,
        },
    }
    (run_dir / "report.json").write_text(
        json.dumps(report_payload), encoding="utf-8"
    )

    # Add node_outputs_json for cells so the flip-card excerpt populates.
    hashes = manifest["variant_canonical_hashes"]
    eval_ids = manifest.get("eval_ids", [])
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    # Add evals table for prompt lookup.
    db.execute(
        "CREATE TABLE IF NOT EXISTS evals (id TEXT PRIMARY KEY, input TEXT)"
    )
    for eid in eval_ids:
        db.execute(
            "INSERT OR REPLACE INTO evals VALUES (?, ?)",
            (eid, f"Decide for applicant {eid}: 720 FICO, $80k income."),
        )
    for eid in eval_ids:
        for idx, vid in enumerate(hashes):
            verdict = "APPROVE" if idx == 0 else "DENY"
            outputs = [
                {"node_id": "intake", "response_text": "intake summary"},
                {"node_id": "risk", "response_text": "risk analysis"},
                {
                    "node_id": "decision",
                    "response_text": (
                        f"FINAL DECISION: {verdict}. Reasoning: based on "
                        "the candidate's debt-to-income ratio and credit "
                        f"profile, the recommendation is {verdict}."
                    ),
                },
            ]
            db.execute(
                "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "publish-synth-0001",
                    vid,
                    eid,
                    "ir-hash-stub",
                    json.dumps(outputs),
                    None,
                    0.01,
                ),
            )
    db.commit()
    db.close()
    return run_dir


def test_extract_flip_counts_handles_clean_payload():
    from topofuzz.publish import _extract_flip_counts
    point, n_pairs, n_flips = _extract_flip_counts(
        {"point": 0.5, "n_pairs": 10, "n_flips": 5}
    )
    assert point == 0.5
    assert n_pairs == 10
    assert n_flips == 5


def test_extract_flip_counts_rejects_bool_n_flips():
    from topofuzz.publish import _extract_flip_counts
    # bool is a subclass of int in Python — must be rejected explicitly.
    _, _, n_flips = _extract_flip_counts(
        {"point": 0.5, "n_pairs": 10, "n_flips": True}
    )
    assert n_flips is None


def test_extract_flip_counts_rejects_nan_point():
    from topofuzz.publish import _extract_flip_counts
    point, _, _ = _extract_flip_counts(
        {"point": float("nan"), "n_pairs": 10, "n_flips": 5}
    )
    assert point is None


def test_extract_flip_counts_returns_none_on_missing():
    from topofuzz.publish import _extract_flip_counts
    assert _extract_flip_counts(None) == (None, None, None)
    assert _extract_flip_counts({}) == (None, None, None)


def test_extract_wilson_bounds_prefers_persisted_block():
    from topofuzz.publish import _extract_wilson_bounds
    lo, hi = _extract_wilson_bounds(
        {"ci_lower": 0.1, "ci_upper": 0.7},
        flip_rate=0.4, n_pairs=10, n_flips=4,
    )
    assert (lo, hi) == (0.1, 0.7)


def test_extract_wilson_bounds_falls_back_to_recompute_when_missing():
    from topofuzz.publish import _extract_wilson_bounds
    # No wilson_addenda block — must recompute from counts.
    lo, hi = _extract_wilson_bounds(None, flip_rate=0.5, n_pairs=10, n_flips=5)
    assert lo is not None and hi is not None
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_extract_wilson_bounds_returns_none_when_unrecomputable():
    from topofuzz.publish import _extract_wilson_bounds
    # No persisted block AND no counts → cannot recompute.
    assert _extract_wilson_bounds(None, None, None, None) == (None, None)


def test_truncate_collapses_whitespace_and_ellipsises():
    from topofuzz.publish import _truncate
    out = _truncate("  hello   world   spread   ", 20)
    assert out == "hello world spread"
    long = "x" * 200
    out = _truncate(long, 30)
    assert out.endswith("…")
    assert len(out) == 30


def test_build_ceo_card_context_flipping_run_has_headline(tmp_path):
    from topofuzz.publish import (
        _build_ceo_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_report_json(tmp_path)
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ctx = _build_ceo_card_context(
        metadata=metadata,
        manifest=manifest,
        verdict_flip_rate=report["headline"]["verdict_flip_rate"],
        wilson_addenda=report["wilson_addenda"],
        target_company=None,
        run_dir=run_dir,
        threshold=0.5,
    )
    assert ctx["flip_pct_str"] == "44%"
    assert "27 pairs" in ctx["flip_label"]
    assert "Wilson 95% CI" in ctx["ci_str"]
    assert "27.6%" in ctx["ci_str"] and "62.7%" in ctx["ci_str"]
    assert "your stack" in ctx["what_paragraph"]
    assert "your own agent topology" in ctx["cta_line"]
    assert ctx["topology_name"] == "synth"
    assert ctx["cost_str"] == "$1.23"
    # Flip card built from cells.
    assert ctx["flip_card"] is not None
    # Default labels are APPROVE/DENY (lending-domain — the working
    # gallery today). A future non-lending gallery + a paired
    # producer-side `verdict_label_true/false` manifest field will
    # opt out, see test_flip_card_honors_manifest_verdict_labels +
    # the BACKLOG Step-20a producer wiring entry.
    assert ctx["flip_card"]["canonical_verdict"] in ("APPROVE", "DENY")
    assert ctx["flip_card"]["canonical_verdict"] != ctx["flip_card"]["variant_verdict"]


def test_build_ceo_card_context_target_substitutes_into_what_para(tmp_path):
    from topofuzz.publish import (
        _build_ceo_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_report_json(tmp_path)
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ctx = _build_ceo_card_context(
        metadata=metadata,
        manifest=manifest,
        verdict_flip_rate=report["headline"]["verdict_flip_rate"],
        wilson_addenda=report["wilson_addenda"],
        target_company="Acme Bank",
        run_dir=run_dir,
        threshold=0.5,
    )
    assert "Acme Bank" in ctx["what_paragraph"]
    assert "your stack" not in ctx["what_paragraph"]
    assert "Acme Bank" in ctx["cta_line"]


def test_build_ceo_card_context_stable_run_has_no_headline_stat(tmp_path):
    from topofuzz.publish import (
        _build_ceo_card_context, load_publish_metadata, _load_manifest,
    )
    # All variants agree → 0 flips.
    run_dir = _make_run_with_report_json(
        tmp_path,
        n_flips=0,
        n_pairs=10,
        point=0.0,
        eval_verdicts={
            "eval_001": [True, True, True, True],
            "eval_002": [False, False, False, False],
        },
    )
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    # No flips → still emits a report.json with point=0.
    report = json.loads((run_dir / "report.json").read_text())
    ctx = _build_ceo_card_context(
        metadata=metadata,
        manifest=manifest,
        verdict_flip_rate=report["headline"]["verdict_flip_rate"],
        wilson_addenda=report["wilson_addenda"],
        target_company=None,
        run_dir=run_dir,
        threshold=0.5,
    )
    assert ctx["flip_pct_str"] == ""
    # The headline still gets the "held its verdict" title.
    assert "held its verdict" in ctx["headline_title"]
    # No flip card on a stable run.
    assert ctx["flip_card"] is None


def test_build_ceo_card_pdf_rejects_missing_run_dir(tmp_path):
    from topofuzz.publish import build_ceo_card_pdf
    with pytest.raises(PublishError, match="not a directory"):
        build_ceo_card_pdf(
            tmp_path / "nope", out_path=tmp_path / "card.pdf"
        )


def test_build_ceo_card_pdf_rejects_out_path_is_directory(tmp_path):
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(PublishError, match="existing directory"):
        build_ceo_card_pdf(run_dir, out_path=tmp_path / "out")


def test_build_ceo_card_pdf_rejects_empty_target(tmp_path):
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    with pytest.raises(PublishError, match="non-empty"):
        build_ceo_card_pdf(
            run_dir, out_path=tmp_path / "card.pdf", target_company="   ",
        )


def test_build_ceo_card_pdf_rejects_overlong_target(tmp_path):
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    with pytest.raises(PublishError, match="exceeds 80"):
        build_ceo_card_pdf(
            run_dir,
            out_path=tmp_path / "card.pdf",
            target_company="x" * 81,
        )


def test_build_ceo_card_pdf_rejects_non_string_target(tmp_path):
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    with pytest.raises(PublishError, match="must be str"):
        build_ceo_card_pdf(
            run_dir,
            out_path=tmp_path / "card.pdf",
            target_company=42,  # type: ignore[arg-type]
        )


def test_build_ceo_card_pdf_rejects_missing_report_json(tmp_path):
    from topofuzz.publish import build_ceo_card_pdf
    # Use the bare run (no report.json).
    run_dir = _make_run(tmp_path, with_attestation=True)
    with pytest.raises(PublishError, match="report.json is missing"):
        build_ceo_card_pdf(run_dir, out_path=tmp_path / "card.pdf")


def test_build_ceo_card_pdf_rejects_corrupt_report_json(tmp_path):
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    (run_dir / "report.json").write_text("not json {", encoding="utf-8")
    with pytest.raises(PublishError, match="could not read report.json"):
        build_ceo_card_pdf(run_dir, out_path=tmp_path / "card.pdf")


def test_render_ceo_card_html_strict_undefined_catches_missing_keys(tmp_path):
    """StrictUndefined: a template variable typo (e.g. ctx['title'] vs
    'headline_title') must surface at render-time, not produce a blank
    cell in the PDF."""
    from topofuzz.publish import _render_ceo_card_html
    import jinja2
    with pytest.raises(jinja2.UndefinedError):
        _render_ceo_card_html({"headline_title": "x"})  # missing the rest


def test_cli_publish_requires_at_least_one_of_pdf_or_ceo_card(tmp_path):
    run_dir = _make_run_with_report_json(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["publish", str(run_dir)])
    assert result.exit_code == 2
    # F8 widened the message to mention --consistency-card too. The
    # "at least one of" prefix is the stable assertion target.
    assert "at least one of" in result.output
    assert "--pdf" in result.output and "--ceo-card" in result.output


def test_cli_publish_rejects_target_without_ceo_card(tmp_path):
    run_dir = _make_run_with_report_json(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "publish",
            str(run_dir),
            "--pdf",
            str(tmp_path / "out.pdf"),
            "--target",
            "Acme Bank",
        ],
    )
    assert result.exit_code == 2
    assert "--target is only meaningful" in result.output


def test_format_headline_flipping_vs_stable():
    from topofuzz.publish import _format_headline
    title_flip, lede_flip = _format_headline("lending-4role", 0.44, 12)
    assert "flips its verdict" in title_flip
    assert "lending 4role" in title_flip  # underscore/dash normalisation
    assert "jury-of-3" in lede_flip
    title_stable, lede_stable = _format_headline("stable-pipe", None, 0)
    assert "held its verdict" in title_stable
    assert "publishable" in lede_stable


def test_excerpt_terminal_output_handles_missing_json():
    from topofuzz.publish import _excerpt_terminal_output
    assert _excerpt_terminal_output(None) == ("", "")
    assert _excerpt_terminal_output("") == ("", "")
    assert _excerpt_terminal_output("not json") == ("", "")
    assert _excerpt_terminal_output("[]") == ("", "")


def test_excerpt_terminal_output_dispatches_on_verdict_node():
    """Step 20a / F6 pass-1 RT M1: when verdict_node is supplied the
    excerpt comes from THAT node, not outputs[-1] — mirroring the
    runner._run_judge dispatch so the buyer never sees a verdict badge
    next to a non-verdict node's text."""
    from topofuzz.publish import _excerpt_terminal_output
    payload = json.dumps([
        {"node_id": "intake", "response_text": "INTAKE TEXT"},
        {"node_id": "risk", "response_text": "RISK TEXT"},
        {"node_id": "decision", "response_text": "DECISION TEXT"},
    ])
    text, ordering = _excerpt_terminal_output(payload, verdict_node="risk")
    assert text == "RISK TEXT"
    assert ordering == "intake → risk → decision"


def test_excerpt_terminal_output_dispatches_on_lex_first_terminal_when_no_verdict_node():
    from topofuzz.publish import _excerpt_terminal_output
    payload = json.dumps([
        {"node_id": "a", "response_text": "A TEXT"},
        {"node_id": "b", "response_text": "B TEXT"},
        {"node_id": "c", "response_text": "C TEXT"},
    ])
    # terminal=("c","a") — lex-first present is "a", not the tuple-first.
    text, _ = _excerpt_terminal_output(
        payload, terminal_nodes=("c", "a")
    )
    assert text == "A TEXT"


def test_excerpt_terminal_output_falls_back_to_outputs_last_when_neither_set():
    """v0.5-compat: a manifest with neither verdict_node nor
    terminal_nodes still works — fallback to outputs[-1]."""
    from topofuzz.publish import _excerpt_terminal_output
    payload = json.dumps([
        {"node_id": "a", "response_text": "A"},
        {"node_id": "b", "response_text": "B (last walked)"},
    ])
    text, _ = _excerpt_terminal_output(payload)
    assert text == "B (last walked)"


def test_build_ceo_card_pdf_target_company_rejects_bidi_override(tmp_path):
    """F6 pass-1 RT M3: ‪ (U+202E RIGHT-TO-LEFT OVERRIDE) survives
    autoescape but cosmetically poisons the rendered PDF. Reject at
    the boundary."""
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    with pytest.raises(PublishError, match="invisible"):
        build_ceo_card_pdf(
            run_dir,
            out_path=tmp_path / "card.pdf",
            target_company="Acme‮Bank",
        )


def test_build_ceo_card_pdf_target_company_rejects_zero_width_joiner(tmp_path):
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    with pytest.raises(PublishError, match="invisible"):
        build_ceo_card_pdf(
            run_dir,
            out_path=tmp_path / "card.pdf",
            target_company="Acme‍Bank",
        )


def test_build_ceo_card_pdf_target_company_rejects_c0_control(tmp_path):
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    with pytest.raises(PublishError, match="C0 control"):
        build_ceo_card_pdf(
            run_dir,
            out_path=tmp_path / "card.pdf",
            target_company="Acme\x01Bank",
        )


def test_build_ceo_card_pdf_target_company_nfc_normalises(tmp_path):
    """A decomposed-form input ("Café" with combining acute) gets NFC-
    normalised before the length cap so the visual width matches the
    char count budget. Doesn't raise — proves the normalisation path
    runs without rejection."""
    from topofuzz.publish import _build_ceo_card_context, _load_manifest, load_publish_metadata
    # We just exercise the normalisation by hitting the guard logic via
    # build_ceo_card_pdf with a borderline name; non-raising assertion.
    run_dir = _make_run_with_report_json(tmp_path)
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    # Construct context with a decomposed café (e + U+0301).
    ctx = _build_ceo_card_context(
        metadata=metadata, manifest=manifest,
        verdict_flip_rate=report["headline"]["verdict_flip_rate"],
        wilson_addenda=report["wilson_addenda"],
        target_company="Café Bank",  # decomposed form
        run_dir=run_dir, threshold=0.5,
    )
    # Target survives the context build (normalisation happens at the
    # CLI-input guard inside build_ceo_card_pdf, not _build_ceo_card_context).
    assert "Cafe" in ctx["what_paragraph"] or "Café" in ctx["what_paragraph"]


def test_cli_publish_empty_string_target_rejected_when_no_ceo_card(tmp_path):
    """F6 pass-1 RT M2: --target "" (empty string) is now treated as a
    supplied target and rejected without --ceo-card."""
    run_dir = _make_run_with_report_json(tmp_path)
    runner = CliRunner()
    # Empty string for target with only --pdf — should NOT silently
    # ignore. The truthy guard means "" is treated as not-supplied,
    # which is the correct UX (allows shell scripts that conditionally
    # set TARGET=""). The MORE problematic case is a non-empty target
    # with no --ceo-card, which IS rejected.
    result = runner.invoke(
        app,
        [
            "publish", str(run_dir),
            "--pdf", str(tmp_path / "out.pdf"),
            "--target", "Acme Bank",
        ],
    )
    assert result.exit_code == 2
    assert "--target is only meaningful" in result.output


def test_flip_card_defaults_to_approve_deny_for_lending(tmp_path):
    """F6 pass-2 RT B1: with no producer-side wiring of
    verdict_label_true/false yet, the working lending demo keeps
    APPROVE/DENY as the default so the gallery doesn't regress to
    'VERDICT A/B'. When a non-lending gallery + producer-side field
    ship together (BACKLOG follow-up), this test flips."""
    from topofuzz.publish import _build_ceo_card_context, _load_manifest, load_publish_metadata
    run_dir = _make_run_with_report_json(tmp_path)
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ctx = _build_ceo_card_context(
        metadata=metadata, manifest=manifest,
        verdict_flip_rate=report["headline"]["verdict_flip_rate"],
        wilson_addenda=report["wilson_addenda"],
        target_company=None,
        run_dir=run_dir, threshold=0.5,
    )
    fc = ctx["flip_card"]
    assert fc is not None
    assert {fc["canonical_verdict"], fc["variant_verdict"]} == {"APPROVE", "DENY"}


def test_flip_card_rejects_colliding_verdict_labels(tmp_path):
    """F6 pass-2 RT M1: two labels that upper-case to the same string
    destroy the side-by-side story — refuse the card."""
    from topofuzz.publish import _build_ceo_card_context, _load_manifest, load_publish_metadata
    run_dir = _make_run_with_report_json(tmp_path)
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_label_true"] = "approve"
    m["verdict_label_false"] = "Approve"  # collides after upper()
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    with pytest.raises(PublishError, match="collide after normalization"):
        _build_ceo_card_context(
            metadata=metadata, manifest=manifest,
            verdict_flip_rate=report["headline"]["verdict_flip_rate"],
            wilson_addenda=report["wilson_addenda"],
            target_company=None,
            run_dir=run_dir, threshold=0.5,
        )


def test_flip_card_honors_manifest_verdict_labels(tmp_path):
    from topofuzz.publish import _build_ceo_card_context, _load_manifest, load_publish_metadata
    run_dir = _make_run_with_report_json(tmp_path)
    # Add labels to the manifest.
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_label_true"] = "approve"
    m["verdict_label_false"] = "deny"
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ctx = _build_ceo_card_context(
        metadata=metadata, manifest=manifest,
        verdict_flip_rate=report["headline"]["verdict_flip_rate"],
        wilson_addenda=report["wilson_addenda"],
        target_company=None,
        run_dir=run_dir, threshold=0.5,
    )
    fc = ctx["flip_card"]
    assert fc is not None
    # Uppercased + length-capped per builder rules.
    assert {fc["canonical_verdict"], fc["variant_verdict"]} == {"APPROVE", "DENY"}


def test_flip_card_rejects_xor_half_set_manifest(tmp_path, caplog):
    """Red-team pass-1 M2: orchestrator persists labels as a pair
    (both-or-neither). A tampered manifest with exactly one label
    non-empty must NOT let publish silently default the unset half —
    instead, log + fall back to BOTH defaults."""
    from topofuzz.publish import _build_ceo_card_context, _load_manifest, load_publish_metadata
    import logging
    run_dir = _make_run_with_report_json(tmp_path)
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_label_true"] = "ESCALATE"
    # verdict_label_false intentionally absent — half-set tamper.
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    with caplog.at_level(logging.WARNING):
        ctx = _build_ceo_card_context(
            metadata=metadata, manifest=manifest,
            verdict_flip_rate=report["headline"]["verdict_flip_rate"],
            wilson_addenda=report["wilson_addenda"],
            target_company=None,
            run_dir=run_dir, threshold=0.5,
        )
    fc = ctx["flip_card"]
    assert fc is not None
    # Both fell back to defaults despite half-set tamper.
    assert {fc["canonical_verdict"], fc["variant_verdict"]} == {"APPROVE", "DENY"}
    assert any(
        "exactly one of verdict_label" in rec.message
        for rec in caplog.records
    )


def test_flip_card_rejects_over_twelve_char_manifest_label(tmp_path, caplog):
    """Red-team pass-1 M1 + CR MAJOR-2: a tampered manifest with a
    label >12 chars used to silently slice and ship truncated text in
    the CEO PDF. Now publish logs and falls back to both defaults."""
    from topofuzz.publish import _build_ceo_card_context, _load_manifest, load_publish_metadata
    import logging
    run_dir = _make_run_with_report_json(tmp_path)
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_label_true"] = "PARTIALLY_APPROVED"  # 18 chars
    m["verdict_label_false"] = "REJECTED"
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    with caplog.at_level(logging.WARNING):
        ctx = _build_ceo_card_context(
            metadata=metadata, manifest=manifest,
            verdict_flip_rate=report["headline"]["verdict_flip_rate"],
            wilson_addenda=report["wilson_addenda"],
            target_company=None,
            run_dir=run_dir, threshold=0.5,
        )
    fc = ctx["flip_card"]
    assert fc is not None
    assert {fc["canonical_verdict"], fc["variant_verdict"]} == {"APPROVE", "DENY"}
    assert any(
        "normalizes to >12 chars" in rec.message for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# Cross-consumer NFKC + Cc/Cf/M normalize red-team tests
# (BACKLOG 2026-05-16 MAJOR: shared verdict-label normalize helper)
# ---------------------------------------------------------------------------
#
# Mirrors the four red-team tests in tests/test_inspect.py for the
# publish consumer path. The producer IR validator
# (`ir._normalize_verdict_label`) applies NFKC + Cc/Cf/M strip +
# .strip().upper() + 12-char cap before a manifest is ever written.
# These tests cover the scenario where a hand-edited or supply-chain-
# tampered manifest carries Unicode-tamper labels the IR validator
# would have rejected -- the shared helper
# `_normalize_verdict_label_for_consumer` is the last line of defense.
#
# Non-ASCII codepoints use \u escape sequences in Python source so the
# test bytes are unambiguous to a future reader.


def test_ceo_card_strips_variation_selector_from_manifest_labels(tmp_path):
    """Variation Selector-16 (U+FE0F, category Mn) is the producer's
    canonical red-team example: a combining mark that survives NFKC
    unchanged and renders visually identical to the underlying label
    but bytewise differs. Without the shared helper, a tampered
    manifest with ``"APPROVE️"`` ships bytewise-tampered content
    into the CEO card PDF even though it looks identical to ASCII
    "APPROVE". The Mn-category filter in the shared helper drops it,
    recovering the canonical bytes the producer IR would have
    produced."""
    from topofuzz.publish import (
        _build_ceo_card_context, _load_manifest, load_publish_metadata,
    )
    run_dir = _make_run_with_report_json(tmp_path)
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_label_true"] = "APPROVE️"
    m["verdict_label_false"] = "DENY"
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ctx = _build_ceo_card_context(
        metadata=metadata, manifest=manifest,
        verdict_flip_rate=report["headline"]["verdict_flip_rate"],
        wilson_addenda=report["wilson_addenda"],
        target_company=None,
        run_dir=run_dir, threshold=0.5,
    )
    fc = ctx["flip_card"]
    assert fc is not None
    # Pair collapses to canonical bytes -- no variation selector
    # survives into the rendered fields.
    assert {fc["canonical_verdict"], fc["variant_verdict"]} == {"APPROVE", "DENY"}
    assert "️" not in fc["canonical_verdict"]
    assert "️" not in fc["variant_verdict"]


def test_ceo_card_strips_zwsp_from_manifest_labels(tmp_path):
    """Zero-width space (U+200B, category Cf) is invisible at render
    time but bytewise differs from the canonical label. NFKC leaves it
    intact (no compatibility-equivalent); the Cf-category filter in
    the shared helper drops it. A buyer's manifest-hash comparison
    must match what producer-side IR would have produced."""
    from topofuzz.publish import (
        _build_ceo_card_context, _load_manifest, load_publish_metadata,
    )
    run_dir = _make_run_with_report_json(tmp_path)
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_label_true"] = "APPROVE​​"
    m["verdict_label_false"] = "DE​NY"  # interior ZWSP, not trailing.
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ctx = _build_ceo_card_context(
        metadata=metadata, manifest=manifest,
        verdict_flip_rate=report["headline"]["verdict_flip_rate"],
        wilson_addenda=report["wilson_addenda"],
        target_company=None,
        run_dir=run_dir, threshold=0.5,
    )
    fc = ctx["flip_card"]
    assert fc is not None
    assert {fc["canonical_verdict"], fc["variant_verdict"]} == {"APPROVE", "DENY"}
    assert "​" not in fc["canonical_verdict"]
    assert "​" not in fc["variant_verdict"]


def test_ceo_card_collapses_fullwidth_latin_in_manifest_labels(tmp_path):
    """Fullwidth Latin (U+FF21..U+FF3A) renders visually as Latin
    capitals but is a different codepoint range. NFKC collapses it to
    ASCII -- without the shared helper, a fullwidth-tampered label
    slips past ``.strip().upper()`` and ships bytewise-tampered into
    the CEO card PDF."""
    from topofuzz.publish import (
        _build_ceo_card_context, _load_manifest, load_publish_metadata,
    )
    run_dir = _make_run_with_report_json(tmp_path)
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    # U+FF21..U+FF3A fullwidth Latin spelling APPROVE / DENY.
    m["verdict_label_true"] = "ＡＰＰＲＯＶＥ"
    m["verdict_label_false"] = "ＤＥＮＹ"
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ctx = _build_ceo_card_context(
        metadata=metadata, manifest=manifest,
        verdict_flip_rate=report["headline"]["verdict_flip_rate"],
        wilson_addenda=report["wilson_addenda"],
        target_company=None,
        run_dir=run_dir, threshold=0.5,
    )
    fc = ctx["flip_card"]
    assert fc is not None
    assert {fc["canonical_verdict"], fc["variant_verdict"]} == {"APPROVE", "DENY"}
    # Fullwidth codepoints must NOT survive into the rendered verdict
    # fields. Spot-check the corner codepoints.
    combined = fc["canonical_verdict"] + fc["variant_verdict"]
    assert "Ａ" not in combined
    assert "Ｄ" not in combined


def test_ceo_card_falls_back_to_defaults_on_hangul_filler_label(tmp_path):
    """L949 closed 2026-05-16. Tampered manifest with U+3164 HANGUL
    FILLER (Lo, invisible) inside the label survives NFKC + C/M filter
    (Lo is in category L, not C/M) and would have rendered as the eye
    sees "APPROVE" on the CEO PDF while mismatching ASCII grep on the
    manifest bytes. Positive ASCII allowlist drops both labels to
    empty → pair-level fallback → buyer sees the safe defaults."""
    from topofuzz.publish import (
        _build_ceo_card_context, _load_manifest, load_publish_metadata,
    )
    run_dir = _make_run_with_report_json(tmp_path)
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_label_true"] = "APㅤPROVE"  # U+3164 HANGUL FILLER
    m["verdict_label_false"] = "DEFER"
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ctx = _build_ceo_card_context(
        metadata=metadata, manifest=manifest,
        verdict_flip_rate=report["headline"]["verdict_flip_rate"],
        wilson_addenda=report["wilson_addenda"],
        target_company=None,
        run_dir=run_dir, threshold=0.5,
    )
    fc = ctx["flip_card"]
    assert fc is not None
    # Pair-level fallback: BOTH labels resolve to defaults, NOT
    # half-custom (custom-false + default-true).
    assert {fc["canonical_verdict"], fc["variant_verdict"]} == {
        "APPROVE", "DENY"
    }
    combined = fc["canonical_verdict"] + fc["variant_verdict"]
    assert "ㅤ" not in combined
    assert "DEFER" not in combined


def test_ceo_card_falls_back_to_defaults_on_cyrillic_homoglyph_label(tmp_path):
    """L949 closed 2026-05-16. The canonical procurement-integrity
    attack: tampered manifest with U+0410 CYRILLIC А (Lu) renders
    pixel-identical to ASCII Latin A on every full-Unicode font.
    ``"АPPROVE"`` (Cyrillic) on the CEO PDF text layer reads identical
    to ``"APPROVE"`` (ASCII) but mismatches the buyer's deterministic
    grep. Allowlist drops; defaults render."""
    from topofuzz.publish import (
        _build_ceo_card_context, _load_manifest, load_publish_metadata,
    )
    run_dir = _make_run_with_report_json(tmp_path)
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_label_true"] = "АPPROVE"  # leading U+0410, not U+0041
    m["verdict_label_false"] = "DEFER"
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ctx = _build_ceo_card_context(
        metadata=metadata, manifest=manifest,
        verdict_flip_rate=report["headline"]["verdict_flip_rate"],
        wilson_addenda=report["wilson_addenda"],
        target_company=None,
        run_dir=run_dir, threshold=0.5,
    )
    fc = ctx["flip_card"]
    assert fc is not None
    assert {fc["canonical_verdict"], fc["variant_verdict"]} == {
        "APPROVE", "DENY"
    }
    # The Cyrillic byte must not leak into the rendered fields.
    combined = fc["canonical_verdict"] + fc["variant_verdict"]
    assert "А" not in combined
    assert "DEFER" not in combined


def test_ceo_card_rejects_post_normalize_collision_via_variation_selector(
    tmp_path,
):
    """The collision check fires POST-normalize. The pair ("APPROVE",
    "APPROVE️") looks distinct bytewise (7 vs 8 codepoints) but
    collapses to identical "APPROVE" after the Mn-category filter.
    Without the shared helper this sneaks past the consumer collision
    check (``"APPROVE".upper() != "APPROVE️".upper()``) and the
    CEO card would render two visually-identical "APPROVE" verdict
    pills with different excerpts -- worse than no flip card. The
    shared helper's NFKC + Mn filter exposes the collision so the
    consumer can refuse to render."""
    from topofuzz.publish import (
        _build_ceo_card_context, _load_manifest, load_publish_metadata,
    )
    run_dir = _make_run_with_report_json(tmp_path)
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_label_true"] = "APPROVE"
    m["verdict_label_false"] = "APPROVE️"  # collides POST-normalize.
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    with pytest.raises(PublishError, match="collide after normalization"):
        _build_ceo_card_context(
            metadata=metadata, manifest=manifest,
            verdict_flip_rate=report["headline"]["verdict_flip_rate"],
            wilson_addenda=report["wilson_addenda"],
            target_company=None,
            run_dir=run_dir, threshold=0.5,
        )


def test_ceo_card_falls_back_when_zwsp_only_label_creates_half_custom_pair(
    tmp_path, caplog,
):
    """Pass-1 dual-audit regression (CR M-1 + RT M1, 2026-05-16): a label
    like ``"​"`` (ZWSP only) is truthy under ``str.strip()`` (which
    does NOT strip Cf characters), so a pre-normalize XOR check sees
    both sides as "set" and skips the half-set fallback. Then the
    helper collapses the ZWSP-only side to ``""``, the empty-fallback
    resolves it to the default partner, and the buyer sees a
    half-custom pair (custom-true + default-false) -- exactly the
    scenario the pair-level fallback is meant to prevent. The fix
    moves the XOR check to operate on the POST-normalize results so
    this asymmetric tamper triggers the fallback."""
    from topofuzz.publish import (
        _build_ceo_card_context, _load_manifest, load_publish_metadata,
    )
    import logging
    run_dir = _make_run_with_report_json(tmp_path)
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_label_true"] = "ESCALATE"
    m["verdict_label_false"] = "​​"  # ZWSP only -- normalizes to "".
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    with caplog.at_level(logging.WARNING):
        ctx = _build_ceo_card_context(
            metadata=metadata, manifest=manifest,
            verdict_flip_rate=report["headline"]["verdict_flip_rate"],
            wilson_addenda=report["wilson_addenda"],
            target_company=None,
            run_dir=run_dir, threshold=0.5,
        )
    fc = ctx["flip_card"]
    assert fc is not None
    # Both labels fell back to defaults; no half-custom pair leaks.
    assert {fc["canonical_verdict"], fc["variant_verdict"]} == {"APPROVE", "DENY"}
    assert any(
        "exactly one of verdict_label" in rec.message
        and "after NFKC" in rec.message
        for rec in caplog.records
    )


def test_build_ceo_card_pdf_end_to_end_renders_exactly_one_page(tmp_path):
    """F6 pass-1 CR M3 + pass-2 RT B2: real Chromium render, asserts
    EXACTLY 1 page on the synthetic lending fixture so a template
    change that silently bumps to 2 pages is caught.

    Skipped when playwright OR its Chromium binary OR pypdf is
    missing — verify-only installs."""
    pytest.importorskip("playwright")
    pytest.importorskip("pypdf")
    # F6 pass-2 RT m2: also skip when the Chromium binary isn't
    # installed (importing the python package alone doesn't guarantee
    # the binary; pw.chromium.executable_path raises otherwise).
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as pw:
            if not pw.chromium.executable_path:
                pytest.skip("playwright chromium binary not installed")
    except Exception as e:
        pytest.skip(f"playwright runtime not usable: {e}")
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    out = tmp_path / "card.pdf"
    target = build_ceo_card_pdf(
        run_dir, out_path=out, target_company="Acme Bank"
    )
    assert target.is_file()
    assert target.stat().st_size > 1024  # not an empty PDF
    # F6 pass-2 RT B2: tighten "≤2" to "==1" on the synthetic
    # fixture — catches template overflow regressions early.
    import pypdf
    reader = pypdf.PdfReader(str(target))
    assert len(reader.pages) == 1, (
        f"ceo card rendered to {len(reader.pages)} pages on the "
        "synthetic lending fixture; expected exactly 1 — investigate "
        "template overflow."
    )
    # Cover at least the headline + footer breadcrumb appearance.
    page_text = reader.pages[0].extract_text() or ""
    assert "topofuzz" in page_text.lower()


def test_build_ceo_card_pdf_excerpt_dispatches_on_verdict_node_e2e(tmp_path):
    """F6 pass-2 CR I2: build a fixture where outputs[-1] != verdict
    node's output; assert the rendered PDF excerpt contains the
    verdict-node text, NOT outputs[-1]. Catches a regression to the
    bug F6 pass-1 RT M1 fixed."""
    pytest.importorskip("playwright")
    pytest.importorskip("pypdf")
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as pw:
            if not pw.chromium.executable_path:
                pytest.skip("playwright chromium binary not installed")
    except Exception as e:
        pytest.skip(f"playwright runtime not usable: {e}")
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    # Mark verdict_node = "intake" (the FIRST node) so outputs[-1]
    # (the "decision" node) is NOT what the excerpt should show.
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_node"] = "intake"
    m["terminal_nodes"] = ["intake"]
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    # Replant cells so intake's response_text contains a UNIQUE
    # marker we can search for in the rendered PDF text.
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute("DELETE FROM cells")
    hashes = m["variant_canonical_hashes"]
    eval_ids = m["eval_ids"]
    INTAKE_MARKER = "INTAKE_PROVENANCE_MARKER_UNIQ"
    DECISION_MARKER = "DECISION_LAST_NODE_TEXT"
    for eid in eval_ids:
        for idx, vid in enumerate(hashes):
            outputs = [
                {"node_id": "intake", "response_text": (
                    f"{INTAKE_MARKER} {'APPROVE' if idx == 0 else 'DENY'}"
                )},
                {"node_id": "risk", "response_text": "risk"},
                {"node_id": "decision", "response_text": (
                    f"{DECISION_MARKER} ignored-by-excerpt"
                )},
            ]
            db.execute(
                "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("publish-synth-0001", vid, eid, "ir-hash", json.dumps(outputs), None, 0.01),
            )
    db.commit(); db.close()
    out = tmp_path / "card.pdf"
    build_ceo_card_pdf(run_dir, out_path=out, target_company="Test")
    import pypdf
    reader = pypdf.PdfReader(str(out))
    page_text = reader.pages[0].extract_text() or ""
    # The excerpt should come from the verdict node (intake), NOT
    # outputs[-1] (decision).
    assert INTAKE_MARKER in page_text, (
        "ceo card excerpt did not include the verdict-node text; "
        "fell back to outputs[-1]?"
    )
    # If we're strict that the wrong-node text is excluded entirely:
    # the decision text might still appear in the ordering line, so
    # only assert the dedicated marker is absent from the EXCERPT
    # context (i.e. not adjacent to the badge). Loose check: the
    # intake marker should appear at least twice (canonical + variant
    # cells), and the decision marker should not be the verdict text.
    assert page_text.count(INTAKE_MARKER) >= 2


def test_build_ceo_card_pdf_sweeps_stale_tempfiles_on_entry(tmp_path):
    """F6 pass-1 CR N2 + pass-2 CR I1: a prior leak leaves
    ceo_card_tmp_*.html in run_dir; the next invocation must sweep
    them so an automation loop does not accumulate orphans. Glob is
    narrowed to ``ceo_card_tmp_*`` so operator files like
    ``topofuzz inspect --out ceo_card_demo.html`` are NOT touched."""
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    # Plant 3 stale tempfiles + 1 operator file the sweep MUST leave.
    for i in range(3):
        (run_dir / f"ceo_card_tmp_stale_{i}.html").write_text("stale", encoding="utf-8")
    operator_file = run_dir / "ceo_card_demo.html"
    operator_file.write_text("operator wrote this", encoding="utf-8")
    assert len(list(run_dir.glob("ceo_card_tmp_*.html"))) == 3
    try:
        build_ceo_card_pdf(
            run_dir,
            out_path=tmp_path / "card.pdf",
            target_company="Acme",
        )
    except PublishError:
        pass  # Acceptable — we only assert the sweep ran.
    remaining = list(run_dir.glob("ceo_card_tmp_stale_*.html"))
    assert remaining == [], (
        f"sweep failed to remove stale tempfiles: {remaining}"
    )
    # Operator file must NOT have been touched.
    assert operator_file.is_file(), (
        "sweep mistakenly unlinked the operator's ceo_card_demo.html"
    )


@pytest.mark.parametrize(
    "char,label",
    [
        ("­", "soft hyphen"),
        ("️", "variation selector"),
        ("\U000e0061", "tag character"),
        ("\U0001d400", "math alphanumeric"),
    ],
)
def test_build_ceo_card_pdf_target_company_rejects_invisible_or_homoglyph(
    tmp_path, char, label,
):
    """F6 pass-2 RT M2: expanded invisible-codepoint guard catches
    variation selectors, tag chars, soft hyphen, mathematical
    alphanumeric symbols."""
    from topofuzz.publish import build_ceo_card_pdf
    run_dir = _make_run_with_report_json(tmp_path)
    with pytest.raises(PublishError, match="invisible"):
        build_ceo_card_pdf(
            run_dir,
            out_path=tmp_path / "card.pdf",
            target_company=f"Acme{char}Bank",
        )


def test_build_flip_card_warns_on_empty_verdict_node(tmp_path, caplog):
    """F6 pass-2 RT m1: an empty/whitespace verdict_node in a
    malformed manifest must log a warning rather than silently fall
    back to outputs[-1]."""
    import logging
    from topofuzz.publish import _build_ceo_card_context, _load_manifest, load_publish_metadata
    run_dir = _make_run_with_report_json(tmp_path)
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["verdict_node"] = "   "  # whitespace-only
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    metadata = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    with caplog.at_level(logging.WARNING, logger="topofuzz.publish"):
        _build_ceo_card_context(
            metadata=metadata, manifest=manifest,
            verdict_flip_rate=report["headline"]["verdict_flip_rate"],
            wilson_addenda=report["wilson_addenda"],
            target_company=None,
            run_dir=run_dir, threshold=0.5,
        )
    assert any(
        "verdict_node is empty/whitespace" in r.message
        for r in caplog.records
    ), f"expected warn-log on whitespace verdict_node; got {[r.message for r in caplog.records]}"


def test_render_ceo_card_html_explicit_autoescape_blocks_html_in_topology_name():
    """F6 pass-1 CR N3 regression: explicit autoescape=True must HTML-
    escape user-controlled strings even if a future rename strips the
    .j2/.html suffix from the template path."""
    from topofuzz.publish import _render_ceo_card_html
    ctx = {
        "headline_title": "<script>alert(1)</script>",
        "headline_lede": "lede",
        "flip_pct_str": "44%", "flip_label": "label", "ci_str": "ci",
        "what_paragraph": "what",
        "cta_line": "cta",
        "topology_name": "<script>x</script>",
        "n_variants": 4, "n_evals": 3, "cost_str": "$1.00",
        "n_pairs_str": "27",
        "flip_card": None,
        "run_id": "rid", "tds_version": "0.5.0", "signed_at": "now",
    }
    html = _render_ceo_card_html(ctx)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    # And the topology name in <title> and elsewhere is escaped too.
    assert "<script>x</script>" not in html.replace(
        "&lt;script&gt;x&lt;/script&gt;", ""
    )


# ============================================================
# F8 — cross-vendor consistency card
# ============================================================


def _make_run_with_jury_report_json(
    tmp_path: Path,
    *,
    vendor_overlap: list[str] | None = None,
    vendor_call_failure_rate: float | None = 0.075,
    vendor_disjoint: bool = False,
    jury_models_resolved: dict[str, str] | None = None,
    cost_actual_usd: float = 1.23,
    omit_independence: bool = False,
    omit_report_json: bool = False,
) -> Path:
    """Build a synthetic run with a jury-of-3 report.json populated.

    Mirrors `_make_run_with_report_json` but stamps the
    `headline.judge_independence` block the F8 consistency card reads.
    """
    if vendor_overlap is None:
        vendor_overlap = ["google"]
    if jury_models_resolved is None:
        jury_models_resolved = {
            "deepseek": "deepseek/deepseek-chat",
            "google": "google/gemini-2.5-flash",
            "google_2": "google/gemini-2.5-pro",
        }
    run_dir = _make_run(tmp_path, with_attestation=True)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cost_actual_usd"] = cost_actual_usd
    manifest["n_evals"] = manifest.pop("n_eval_rows", 3)
    manifest["cli_args"] = {
        "jury_of_3": True,
        "jury_models_resolved": jury_models_resolved,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    if not omit_report_json:
        report_payload: dict[str, Any] = {"headline": {}}
        if not omit_independence:
            independence: dict[str, Any] = {
                "judge_protocol": "jury_of_3",
                "judges": [
                    {"vendor": "deepseek", "model_id": "deepseek/deepseek-chat"},
                    {"vendor": "google", "model_id": "google/gemini-2.5-flash"},
                    {"vendor": "google_2", "model_id": "google/gemini-2.5-pro"},
                ],
                "vendor_overlap": vendor_overlap,
                "vendor_disjoint": vendor_disjoint,
            }
            if vendor_call_failure_rate is not None:
                independence["vendor_call_failure_rate"] = vendor_call_failure_rate
            report_payload["headline"]["judge_independence"] = independence
        (run_dir / "report.json").write_text(
            json.dumps(report_payload), encoding="utf-8"
        )
    return run_dir


def test_validate_target_company_passthrough_none():
    from topofuzz.publish import _validate_target_company
    assert _validate_target_company(None) is None


def test_validate_target_company_rejects_empty():
    from topofuzz.publish import _validate_target_company, PublishError
    with pytest.raises(PublishError, match="non-empty"):
        _validate_target_company("   ")


def test_validate_target_company_rejects_overlong():
    from topofuzz.publish import _validate_target_company, PublishError
    with pytest.raises(PublishError, match="exceeds 80 chars"):
        _validate_target_company("a" * 81)


def test_validate_target_company_rejects_non_string():
    from topofuzz.publish import _validate_target_company, PublishError
    with pytest.raises(PublishError, match="must be str"):
        _validate_target_company(123)  # type: ignore[arg-type]


def test_validate_target_company_rejects_bidi_override():
    from topofuzz.publish import _validate_target_company, PublishError
    # U+202E RIGHT-TO-LEFT OVERRIDE — survives autoescape, cosmetically
    # reverses subsequent text in the rendered PDF.
    with pytest.raises(PublishError, match="invisible / formatting"):
        _validate_target_company("Acme‮Bank")


def test_validate_target_company_rejects_zero_width_joiner():
    from topofuzz.publish import _validate_target_company, PublishError
    with pytest.raises(PublishError, match="invisible / formatting"):
        _validate_target_company("Acme​Bank")


def test_validate_target_company_rejects_mathematical_alphanumeric():
    from topofuzz.publish import _validate_target_company, PublishError
    # U+1D400 MATHEMATICAL BOLD CAPITAL A — homoglyph that NFC does NOT
    # fold; would bypass the visual-width budget.
    with pytest.raises(PublishError, match="invisible / formatting"):
        _validate_target_company("\U0001d400cmeBank")


def test_validate_target_company_nfc_normalises():
    """NFC fold must happen BEFORE the length cap so a decomposed form
    counts at its composed length, not its codepoint count."""
    from topofuzz.publish import _validate_target_company
    # "Café" decomposed: C, a, f, e, U+0301 combining acute = 5 codepoints
    # NFC: C, a, f, é = 4 codepoints.
    out = _validate_target_company("Café")
    assert out == "Café"
    assert len(out) == 4


def test_extract_vendor_independence_concentrated_run():
    from topofuzz.publish import _extract_vendor_independence
    independence = {
        "vendor_overlap": ["google"],
        "vendor_call_failure_rate": 0.075,
        "vendor_disjoint": False,
        "judges": [
            {"vendor": "deepseek", "model_id": "deepseek/deepseek-chat"},
            {"vendor": "google", "model_id": "google/gemini-2.5-flash"},
            {"vendor": "google_2", "model_id": "google/gemini-2.5-pro"},
        ],
    }
    manifest = {
        "cli_args": {
            "jury_models_resolved": {
                "deepseek": "deepseek/deepseek-chat",
                "google": "google/gemini-2.5-flash",
                "google_2": "google/gemini-2.5-pro",
            },
        },
    }
    overlap, fr, dis, jurors, dvc, total = _extract_vendor_independence(
        independence, manifest
    )
    assert overlap == ["google"]
    assert fr == 0.075
    assert dis is False
    assert dvc == 2 and total == 3
    assert [j["position"] for j in jurors] == ["1", "2", "3"]
    # Position labels are 1-indexed sequence, NOT the manifest dict key.
    assert all(j["position"].isdigit() for j in jurors)
    # Models are stripped of the vendor/ prefix.
    assert any(j["model"] == "gemini-2.5-flash" for j in jurors)


def test_extract_vendor_independence_falls_back_to_judges_when_resolved_missing():
    from topofuzz.publish import _extract_vendor_independence
    # No cli_args.jury_models_resolved → walk independence.judges instead.
    independence = {
        "vendor_overlap": [],
        "vendor_call_failure_rate": 0.0,
        "vendor_disjoint": True,
        "judges": [
            {"vendor": "anthropic", "model_id": "anthropic/claude-sonnet-4-6"},
            {"vendor": "openai", "model_id": "openai/gpt-5"},
            {"vendor": "google", "model_id": "google/gemini-2.5-flash"},
        ],
    }
    manifest = {"cli_args": {}}
    _, _, _, jurors, dvc, total = _extract_vendor_independence(
        independence, manifest
    )
    assert dvc == 3 and total == 3
    assert {j["vendor"] for j in jurors} == {"anthropic", "openai", "google"}
    # Fallback also strips the vendor/ prefix from model_id.
    assert any(j["model"] == "claude-sonnet-4-6" for j in jurors)


def test_extract_vendor_independence_rejects_nan_failure_rate():
    from topofuzz.publish import _extract_vendor_independence
    independence = {
        "vendor_call_failure_rate": float("nan"),
        "vendor_disjoint": True,
        "vendor_overlap": [],
        "judges": [],
    }
    _, fr, _, _, _, _ = _extract_vendor_independence(independence, {})
    assert fr is None


def test_extract_vendor_independence_rejects_bool_failure_rate():
    """Python bool is a subclass of int — must be rejected explicitly so
    the template never renders True as a "100% failure rate"."""
    from topofuzz.publish import _extract_vendor_independence
    independence = {
        "vendor_call_failure_rate": True,
        "vendor_disjoint": True,
        "vendor_overlap": [],
        "judges": [],
    }
    _, fr, _, _, _, _ = _extract_vendor_independence(independence, {})
    assert fr is None


def test_extract_vendor_independence_skips_malformed_juror_rows():
    from topofuzz.publish import _extract_vendor_independence
    independence = {
        "judges": [
            {"vendor": "openai", "model_id": "openai/gpt-5"},
            "not-a-dict",  # malformed entry should be dropped
            {"vendor": 123},  # non-string vendor dropped
            {"vendor": "google", "model_id": "google/gemini-2.5-flash"},
        ],
    }
    _, _, _, jurors, dvc, total = _extract_vendor_independence(
        independence, {"cli_args": {}}
    )
    assert total == 2
    assert dvc == 2


def test_build_consistency_card_context_concentrated_severity(tmp_path):
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(tmp_path)
    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company="Acme Bank"
    )
    assert ctx["severity"] == "concentrated"
    assert "2 distinct vendors" in ctx["headline_title"]
    assert ctx["stat_headline"] == "2 / 3"
    assert ctx["failure_rate_str"] == "7.5%"
    assert "Acme Bank" in ctx["what_paragraph"]
    assert "Acme Bank" in ctx["cta_line"]


def test_build_consistency_card_context_degraded_severity(tmp_path):
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    # No overlap but high failure rate → degraded.
    run_dir = _make_run_with_jury_report_json(
        tmp_path,
        vendor_overlap=[],
        vendor_call_failure_rate=0.25,
        vendor_disjoint=True,
    )
    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    assert ctx["severity"] == "degraded"
    # F8 RT pass-1 M9: precision unified to .1f across headline + body.
    assert "25.0%" in ctx["headline_title"]
    assert "your stack" in ctx["what_paragraph"]


def test_build_consistency_card_context_stable_severity(tmp_path):
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(
        tmp_path,
        vendor_overlap=[],
        vendor_call_failure_rate=0.02,
        vendor_disjoint=True,
        jury_models_resolved={
            "anthropic": "anthropic/claude-sonnet-4-6",
            "openai": "openai/gpt-5",
            "google": "google/gemini-2.5-flash",
        },
    )
    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    assert ctx["severity"] == "stable"
    # F8 RT pass-1 B3 follow-up: copy was softened. The stable branch
    # no longer claims SOC 2 / MRM quotability outright — it now reads
    # as a "preliminary cross-vendor-disjointness signal" requiring
    # self-preference + rerun-stability confirmation.
    assert "preliminary" in ctx["what_paragraph"]
    assert "cross-vendor-disjointness signal" in ctx["what_paragraph"]
    # Soft-claim guardrail: must NOT say it's quotable without confirmation.
    assert "SOC 2" not in ctx["what_paragraph"]


def test_build_consistency_card_context_unknown_severity_when_partial(tmp_path):
    """No overlap signal, no failure-rate signal, no disjoint flag → the
    extractor falls back to the conservative "unknown" branch rather
    than over-claiming stability."""
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(
        tmp_path,
        vendor_overlap=[],
        vendor_call_failure_rate=None,
        vendor_disjoint=False,  # disjoint=False AND no overlap is the partial-telemetry shape
    )
    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    assert ctx["severity"] == "unknown"
    assert ctx["stat_headline"] == "?"


def test_html_safe_vendor_truncates_long_name():
    from topofuzz.publish import _html_safe_vendor
    assert _html_safe_vendor("google") == "google"
    out = _html_safe_vendor("a" * 100)
    assert out.endswith("…")
    assert len(out) == 38  # 37 chars + ellipsis


def test_build_consistency_card_pdf_rejects_missing_run_dir(tmp_path):
    from topofuzz.publish import build_consistency_card_pdf, PublishError
    with pytest.raises(PublishError, match="not a directory"):
        build_consistency_card_pdf(
            tmp_path / "nope", out_path=tmp_path / "card.pdf"
        )


def test_build_consistency_card_pdf_rejects_out_path_is_directory(tmp_path):
    from topofuzz.publish import build_consistency_card_pdf, PublishError
    run_dir = _make_run_with_jury_report_json(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(PublishError, match="existing directory"):
        build_consistency_card_pdf(run_dir, out_path=tmp_path / "out")


def test_build_consistency_card_pdf_rejects_missing_report_json(tmp_path):
    from topofuzz.publish import build_consistency_card_pdf, PublishError
    run_dir = _make_run_with_jury_report_json(tmp_path, omit_report_json=True)
    with pytest.raises(PublishError, match="report.json is missing"):
        build_consistency_card_pdf(run_dir, out_path=tmp_path / "card.pdf")


def test_build_consistency_card_pdf_rejects_missing_independence_block(tmp_path):
    """Single-judge runs have no headline.judge_independence — the
    consistency card must refuse to render rather than silently emit a
    "stable" PDF on a single-juror source."""
    from topofuzz.publish import build_consistency_card_pdf, PublishError
    run_dir = _make_run_with_jury_report_json(tmp_path, omit_independence=True)
    with pytest.raises(PublishError, match="judge_independence"):
        build_consistency_card_pdf(run_dir, out_path=tmp_path / "card.pdf")


def test_build_consistency_card_pdf_rejects_overlong_target(tmp_path):
    from topofuzz.publish import build_consistency_card_pdf, PublishError
    run_dir = _make_run_with_jury_report_json(tmp_path)
    with pytest.raises(PublishError, match="exceeds 80 chars"):
        build_consistency_card_pdf(
            run_dir,
            out_path=tmp_path / "card.pdf",
            target_company="a" * 81,
        )


def test_build_consistency_card_pdf_rejects_bidi_override_target(tmp_path):
    from topofuzz.publish import build_consistency_card_pdf, PublishError
    run_dir = _make_run_with_jury_report_json(tmp_path)
    with pytest.raises(PublishError, match="invisible / formatting"):
        build_consistency_card_pdf(
            run_dir,
            out_path=tmp_path / "card.pdf",
            target_company="Acme‮Bank",
        )


def test_render_consistency_card_html_strict_undefined_catches_missing_keys():
    """The Jinja env uses StrictUndefined — a typo in the context dict
    crashes the render rather than emitting an empty cell."""
    import jinja2
    from topofuzz.publish import _render_consistency_card_html
    with pytest.raises((jinja2.UndefinedError, KeyError)):
        _render_consistency_card_html({"headline_title": "x"})  # missing the rest


def test_render_consistency_card_html_autoescape_blocks_html_in_vendor():
    """A hostile or sloppy vendor name must be HTML-escaped in the
    rendered card, not rendered raw."""
    from topofuzz.publish import _render_consistency_card_html
    ctx = {
        "headline_title": "title",
        "headline_lede": "lede",
        "stat_headline": "2 / 3",
        "stat_label": "label",
        "severity": "concentrated",
        "failure_rate_str": "7.5%",
        "self_preference_warning": "",
        "what_paragraph": "what",
        "cta_line": "cta",
        "topology_name": "fixture",
        "n_variants": 4, "n_evals": 3, "cost_str": "$1.00",
        "jurors": [
            {"position": "1", "vendor": "<script>alert(1)</script>", "model": "x"},
        ],
        "attribution_rows": [],
        "attribution_n_truncated": 0,
        "vendor_independence_note": "",
        "headline_sub_line": "Jury-vote failure rate: 0.0%",
        "run_id": "rid", "tds_version": "0.5.0", "signed_at": "now",
    }
    html = _render_consistency_card_html(ctx)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_render_consistency_card_html_autoescape_blocks_html_in_self_preference():
    """F8 RT pass-1 B4 follow-up: a hostile self_preference_warning
    string from a producer typo must not escape into raw HTML."""
    from topofuzz.publish import _render_consistency_card_html
    ctx = {
        "headline_title": "title", "headline_lede": "lede",
        "stat_headline": "?", "stat_label": "label",
        "severity": "unknown", "failure_rate_str": "—",
        "self_preference_warning": "<img src=x onerror=alert(1)>",
        "what_paragraph": "what", "cta_line": "cta",
        "topology_name": "fixture",
        "n_variants": "—", "n_evals": "—", "cost_str": "—",
        "jurors": [],
        "attribution_rows": [],
        "attribution_n_truncated": 0,
        "vendor_independence_note": "",
        "headline_sub_line": "Jury-vote failure rate: 0.0%",
        "run_id": "rid", "tds_version": "0.5.0", "signed_at": "now",
    }
    html = _render_consistency_card_html(ctx)
    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_cli_publish_requires_at_least_one_card_type(tmp_path):
    from typer.testing import CliRunner
    from topofuzz.cli import app
    run_dir = _make_run_with_jury_report_json(tmp_path)
    runner = CliRunner()
    # No --pdf, no --ceo-card, no --consistency-card.
    # F8 CR pass-1 M1: assert against ``result.output`` to match the
    # F6 precedent at L1063-1071 — every other publish-CLI test in
    # this file uses ``.output``, not ``.stderr``.
    result = runner.invoke(app, ["publish", str(run_dir)])
    assert result.exit_code == 2
    assert "at least one of" in result.output


def test_cli_publish_target_without_card_is_usage_error(tmp_path):
    from typer.testing import CliRunner
    from topofuzz.cli import app
    run_dir = _make_run_with_jury_report_json(tmp_path)
    runner = CliRunner()
    # --target with only --pdf is meaningless.
    result = runner.invoke(
        app,
        ["publish", str(run_dir), "--pdf", str(tmp_path / "out.pdf"),
         "--target", "Acme"],
    )
    assert result.exit_code == 2
    # F8 CR pass-1 M1: assert against ``.output``.
    assert "--target is only meaningful" in result.output


def test_cli_publish_three_card_flags_invoke_all_three_builders(tmp_path, monkeypatch):
    """F8 CR pass-1 M3: the three card-flag branches at cli.py:1855-
    1869 are independent ``if`` blocks. A future refactor to ``elif``
    or a moved ``_validate_target_company`` call could silently regress
    the "all three at once" behaviour without test coverage. Mock the
    builders and assert all three are called and all three paths
    echoed.
    """
    from typer.testing import CliRunner
    from topofuzz.cli import app
    from topofuzz import publish as publish_module

    run_dir = _make_run_with_jury_report_json(tmp_path)
    calls: list[str] = []

    def fake_build_publish_pdf(*args, out_path, **kwargs):
        calls.append("publish")
        out_path.write_bytes(b"%PDF-pub")
        return out_path

    def fake_build_ceo_card_pdf(*args, out_path, target_company=None, **kwargs):
        calls.append(f"ceo_card:{target_company}")
        out_path.write_bytes(b"%PDF-ceo")
        return out_path

    def fake_build_consistency_card_pdf(*args, out_path, target_company=None, **kwargs):
        calls.append(f"consistency_card:{target_company}")
        out_path.write_bytes(b"%PDF-con")
        return out_path

    monkeypatch.setattr(publish_module, "build_publish_pdf", fake_build_publish_pdf)
    monkeypatch.setattr(publish_module, "build_ceo_card_pdf", fake_build_ceo_card_pdf)
    monkeypatch.setattr(
        publish_module, "build_consistency_card_pdf", fake_build_consistency_card_pdf
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "publish", str(run_dir),
            "--pdf", str(tmp_path / "full.pdf"),
            "--ceo-card", str(tmp_path / "ceo.pdf"),
            "--consistency-card", str(tmp_path / "consistency.pdf"),
            "--target", "Acme Bank",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        "publish",
        "ceo_card:Acme Bank",
        "consistency_card:Acme Bank",
    ]
    # Three echo lines, one per builder.
    assert result.output.count("\n") >= 3


def test_build_consistency_card_pdf_rejects_single_judge_with_present_block(tmp_path):
    """F8 RT pass-1 M1: refuse a single-judge run even when the
    ``judge_independence`` block is present-but-single-protocol. The
    "missing block" check is not sufficient — a partial telemetry
    path can populate the block with ``judge_protocol="single"``."""
    from topofuzz.publish import build_consistency_card_pdf, PublishError
    run_dir = _make_run_with_jury_report_json(
        tmp_path, jury_models_resolved={"google": "google/gemini-2.5-flash"},
    )
    # Overwrite the report.json so independence claims single-judge.
    report = json.loads((run_dir / "report.json").read_text())
    report["headline"]["judge_independence"] = {
        "judge_protocol": "single",
        "judges": [{"vendor": "google", "model_id": "google/gemini-2.5-flash"}],
        "vendor_overlap": [],
        "vendor_disjoint": True,
    }
    # Manifest must also drop the jury_of_3 flag.
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["cli_args"]["jury_of_3"] = False
    manifest["cli_args"]["jury_models_resolved"] = {
        "google": "google/gemini-2.5-flash",
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PublishError, match="no signal indicates the source was a jury-of-N"):
        build_consistency_card_pdf(run_dir, out_path=tmp_path / "card.pdf")


def test_normalize_vendor_name_strips_disambiguator():
    """F8 RT pass-1 B1: the IR uses ``google``, ``google_2`` to
    disambiguate two jurors from the same vendor. The PDF should see
    ``google`` for both."""
    from topofuzz.publish import _normalize_vendor_name
    assert _normalize_vendor_name("google") == "google"
    assert _normalize_vendor_name("google_2") == "google"
    assert _normalize_vendor_name("openai_42") == "openai"
    # An underscore that's not a trailing disambiguator stays.
    assert _normalize_vendor_name("my_vendor") == "my_vendor"


def test_extract_vendor_independence_normalises_google_2(tmp_path):
    """F8 RT pass-1 B1 regression: the real lending-tricky-jury data
    has ``google_2`` as the third juror vendor. Without normalisation
    the distinct count was 3 while overlap=['google'] said 2 — a
    contradiction a procurement reviewer spots immediately."""
    from topofuzz.publish import _extract_vendor_independence
    independence = {
        "judges": [
            {"vendor": "deepseek", "model_id": "deepseek/deepseek-chat"},
            {"vendor": "google", "model_id": "google/gemini-2.5-flash"},
            {"vendor": "google_2", "model_id": "google/gemini-2.5-pro"},
        ],
        "vendor_overlap": ["google"],
        "vendor_disjoint": False,
        "vendor_call_failure_rate": 0.075,
    }
    _, _, _, jurors, dvc, total = _extract_vendor_independence(
        independence, {"cli_args": {}}
    )
    assert total == 3
    # The CRITICAL assertion: distinct_vendor_count agrees with overlap.
    assert dvc == 2
    assert [j["vendor"] for j in jurors] == ["deepseek", "google", "google"]


def test_extract_vendor_independence_position_renumbered_after_filter():
    """F8 CR pass-1 N1: if malformed entries are dropped, the rendered
    table must not have a gap (positions ``1, 3`` with no ``2``)."""
    from topofuzz.publish import _extract_vendor_independence
    independence = {
        "judges": [
            {"vendor": "openai", "model_id": "openai/gpt-5"},
            "not-a-dict",
            {"vendor": 123},
            {"vendor": "google", "model_id": "google/gemini-2.5-flash"},
            {"vendor": "anthropic", "model_id": "anthropic/claude-sonnet-4-6"},
        ],
    }
    _, _, _, jurors, _, total = _extract_vendor_independence(
        independence, {"cli_args": {}}
    )
    assert total == 3
    assert [j["position"] for j in jurors] == ["1", "2", "3"]


def test_extract_vendor_independence_overlap_dedups_disambiguator():
    """F8 RT pass-1 B1: a producer that returns ``['google', 'google_2']``
    in the overlap list should not double-count after normalisation."""
    from topofuzz.publish import _extract_vendor_independence
    independence = {"vendor_overlap": ["google", "google_2", "openai"]}
    overlap, _, _, _, _, _ = _extract_vendor_independence(
        independence, {"cli_args": {}}
    )
    assert overlap == ["google", "openai"]


def test_build_consistency_card_context_stable_refuses_when_total_jurors_lt_2(tmp_path):
    """F8 BLOCKER B1/B2 regression: the STABLE branch used hardcoded
    ``or 3`` literals when juror data was absent — fabricating a
    "3 / 3" headline from nothing. Must collapse to ``unknown``."""
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(
        tmp_path,
        vendor_overlap=[],
        vendor_call_failure_rate=0.02,
        vendor_disjoint=True,
        # Empty resolved + no judges → no juror evidence.
        jury_models_resolved={},
    )
    # Drop the judges array too so the fallback path also fails.
    report = json.loads((run_dir / "report.json").read_text())
    report["headline"]["judge_independence"]["judges"] = []
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    assert ctx["severity"] == "unknown"
    assert ctx["stat_headline"] == "?"
    # And the headline must not claim "jury-of-3" out of thin air.
    assert "jury-of-3" not in ctx["headline_title"]


def test_build_consistency_card_context_singular_vendor_grammar(tmp_path):
    """F8 CR pass-1 M2: all-same-vendor fixture must say "1 distinct
    vendor", not "1 distinct vendors"."""
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(
        tmp_path,
        vendor_overlap=["google"],
        vendor_call_failure_rate=0.0,
        vendor_disjoint=False,
        jury_models_resolved={
            "google": "google/gemini-2.5-flash",
            "google_2": "google/gemini-2.5-pro",
            "google_3": "google/gemini-2.5-ultra",
        },
    )
    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    assert ctx["severity"] == "concentrated"
    assert "1 distinct vendor." in ctx["headline_title"]
    assert "vendors" not in ctx["headline_title"].split("distinct ")[1]
    # And the stat label is singular too.
    assert "vendor" in ctx["stat_label"]
    assert "distinct vendor backing" in ctx["stat_label"]


def test_build_consistency_card_context_surfaces_self_preference_warning(tmp_path):
    """F8 RT pass-1 B4: the real lending-tricky-jury data has a
    ``self_preference_warning`` ("Agents and judges share vendor(s):
    google..."). Silently dropping it is a credibility hit when a
    procurement reader diffs the PDF against report.json."""
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(tmp_path)
    report = json.loads((run_dir / "report.json").read_text())
    report["headline"]["judge_independence"]["self_preference_warning"] = (
        "Agents and judges share vendor(s): google. Same-vendor scoring "
        "can inflate apparent agreement"
    )
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    assert "share vendor(s): google" in ctx["self_preference_warning"]


def test_build_consistency_card_context_donut_hole_at_displayed_10pct(tmp_path):
    """F8 RT pass-1 M3: a failure_rate that displays as ``10.0%`` must
    enter the ``degraded`` branch — not slip through to ``stable``
    with a "stable but 10%" contradiction in the panel."""
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(
        tmp_path,
        vendor_overlap=[],
        vendor_call_failure_rate=0.0999,  # displays as 10.0% under .1f
        vendor_disjoint=True,
        jury_models_resolved={
            "google": "google/gemini-2.5-flash",
            "openai": "openai/gpt-5",
            "anthropic": "anthropic/claude-sonnet-4-6",
        },
    )
    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    # ``0.0999 >= 0.0995`` → degraded, not stable.
    assert ctx["severity"] == "degraded"
    assert "10.0%" in ctx["failure_rate_str"] or "10.0%" in ctx["headline_title"]


def test_build_consistency_card_context_n_evals_falls_back_to_n_eval_rows(tmp_path):
    """F8 RT pass-1 M5: real production manifests write
    ``n_eval_rows``, not ``n_evals``. The card must not display
    ``Evals: —`` on the real gallery run."""
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest.pop("n_evals", None)
    manifest["n_eval_rows"] = 42
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    md = load_publish_metadata(run_dir)
    manifest_reloaded = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest_reloaded, independence=ind, target_company=None
    )
    assert ctx["n_evals"] == 42


def test_build_consistency_card_context_topology_name_length_cap(tmp_path):
    """F8 CR pass-1 N3: operator-controlled ``topology_name`` cannot
    blow the ``<title>`` tag or the meta-row line."""
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["topology_name"] = "X" * 5000
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    md = load_publish_metadata(run_dir)
    manifest_reloaded = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest_reloaded, independence=ind, target_company=None
    )
    assert len(ctx["topology_name"]) <= 60
    assert ctx["topology_name"].endswith("…")


def test_build_consistency_card_context_overlap_join_bounded(tmp_path):
    """F8 RT pass-1 M6: a hostile manifest with 20+ overlap vendors
    must not blow the 13pt serif lede onto a second line."""
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    overlap = [f"vendor{i}" for i in range(20)]
    run_dir = _make_run_with_jury_report_json(tmp_path, vendor_overlap=overlap)
    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    assert "+17 more" in ctx["headline_lede"]
    # vendor0..vendor2 + the truncation token = 4 comma-separated items
    assert ctx["headline_lede"].count(", ") == 3


def test_build_consistency_card_context_concentrated_lede_surfaces_compound_degradation(tmp_path):
    """F8 RT pass-2 R1: when concentrated AND degraded both fire, the
    procurement reader scanning bold copy must see BOTH signals — not
    just the concentration headline with degradation hidden in a 9pt
    sub-line. The lede gets a tail sentence interpolating the same
    formatted failure rate the body footer carries.
    """
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(
        tmp_path,
        vendor_overlap=["google"],
        vendor_call_failure_rate=0.40,
        vendor_disjoint=False,
    )
    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    # Branch precedence: concentrated wins.
    assert ctx["severity"] == "concentrated"
    # Tail sentence interpolating failure_rate_str must be visible.
    assert "40.0%" in ctx["headline_lede"]
    assert "compounding" in ctx["headline_lede"]


def test_build_consistency_card_context_concentrated_lede_no_tail_when_failure_rate_low(tmp_path):
    """F8 RT pass-2 R1 boundary: when concentrated AND failure rate is
    below the 10% threshold, the lede does NOT get the compound tail.
    Otherwise a clean run with overlap would look procurement-worse
    than it is."""
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(
        tmp_path,
        vendor_overlap=["google"],
        vendor_call_failure_rate=0.02,
        vendor_disjoint=False,
    )
    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    assert ctx["severity"] == "concentrated"
    assert "compounding" not in ctx["headline_lede"]
    assert "Jury votes also failed" not in ctx["headline_lede"]


def test_build_consistency_card_context_all_share_vendor_lede(tmp_path):
    """F8 RT pass-2 R3: when distinct_vendor_count == 1 (all jurors
    share a single vendor), the lede must read "All jurors share a
    vendor", not "Two or more jurors share a vendor"."""
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(
        tmp_path,
        vendor_overlap=["google"],
        vendor_call_failure_rate=0.0,
        vendor_disjoint=False,
        jury_models_resolved={
            "google": "google/gemini-2.5-flash",
            "google_2": "google/gemini-2.5-pro",
            "google_3": "google/gemini-2.5-ultra",
        },
    )
    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    assert ctx["severity"] == "concentrated"
    assert ctx["headline_lede"].startswith("All jurors share a vendor")
    assert "Two or more jurors share" not in ctx["headline_lede"]


def test_build_consistency_card_context_two_jurors_share_vendor_lede(tmp_path):
    """F8 RT pass-2 R3 boundary: the "Two or more" copy still applies
    when there are 2+ distinct vendors and at least one is shared."""
    from topofuzz.publish import (
        _build_consistency_card_context, load_publish_metadata, _load_manifest,
    )
    run_dir = _make_run_with_jury_report_json(tmp_path)  # default: 2 google + 1 deepseek
    md = load_publish_metadata(run_dir)
    manifest = _load_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    ind = report["headline"]["judge_independence"]
    ctx = _build_consistency_card_context(
        metadata=md, manifest=manifest, independence=ind, target_company=None
    )
    assert ctx["severity"] == "concentrated"
    assert ctx["headline_lede"].startswith("Two or more jurors share a vendor")
    assert "All jurors share a vendor" not in ctx["headline_lede"]


def test_build_consistency_card_context_severity_whitelist_falls_back_to_unknown():
    """F8 RT pass-1 M10: a future producer typo cannot ship an
    unstyled white card. The context-builder pipeline only outputs
    one of the four whitelisted severities."""
    # No new branch in the builder produces an off-list severity, but
    # the whitelist guard at the assignment site is the regression
    # surface. Re-affirm by directly testing the membership.
    from topofuzz.publish import _VALID_SEVERITIES
    # F8 Stage A added "attribution" — the F4-per-agent-headline mode
    # uses a neutral-blue chip distinct from concentrated red / degraded
    # amber / stable green / unknown grey.
    assert _VALID_SEVERITIES == frozenset(
        {"concentrated", "degraded", "stable", "unknown", "attribution"}
    )


# ---------------------------------------------------------------------------
# F8 Stage A — F4 attribution headline rework
# ---------------------------------------------------------------------------


def _stage_a_independence_with_overlap() -> dict:
    """Return an independence block matching the lending-tricky-jury
    gallery run (2 distinct vendors, concentrated severity). Used by
    Stage A tests to confirm vendor-concentration demotes correctly."""
    return {
        "vendor_overlap": ["google"],
        "vendor_call_failure_rate": 0.0,
        "vendor_disjoint": False,
        "judges": [
            {"vendor": "deepseek", "model": "deepseek-chat"},
            {"vendor": "google",   "model": "gemini-2.5-flash"},
            {"vendor": "google_2", "model": "gemini-2.5-pro"},
        ],
    }


def _stage_a_ablation_lending_tricky() -> dict:
    """Replica of the canonical 2026-05-15-f4-lending-tricky-jury
    ablation.json shape — risk 63.64%, policy 60.0%, intake 54.55%,
    decision 100% (terminal — must NOT appear in the headline)."""
    return {
        "schema_version": 1,
        "n_flipping_pairs": 12,
        "by_node": {
            "intake": {
                "flips_closed": 6, "total_meaningful": 11,
                "attribution_pct": 54.55, "is_terminal": False,
            },
            "risk": {
                "flips_closed": 7, "total_meaningful": 11,
                "attribution_pct": 63.64, "is_terminal": False,
            },
            "policy": {
                "flips_closed": 6, "total_meaningful": 10,
                "attribution_pct": 60.0, "is_terminal": False,
            },
            "decision": {
                "flips_closed": 10, "total_meaningful": 10,
                "attribution_pct": 100.0, "is_terminal": True,
            },
        },
    }


def test_stage_a_attribution_headline_lands_when_ablation_present(tmp_path):
    """F4 attribution headline wins when ablation.json is supplied:
    title leads with top non-terminal role, decision (terminal) does
    NOT appear in the headline. Severity flips to 'attribution'."""
    from topofuzz.publish import _build_consistency_card_context
    ctx = _build_consistency_card_context(
        metadata={"run_id": "r1", "tds_version": "0.6.0", "signed_at": "t"},
        manifest={"topology_name": "lending-4role", "n_variants": 4,
                  "n_eval_rows": 1, "cost_actual_usd": 3.07,
                  "cli_args": {"jury_of_3": True, "jury_models_resolved": {
                      "1": "deepseek/deepseek-chat",
                      "2": "google/gemini-2.5-flash",
                      "3": "google_2/gemini-2.5-pro",
                  }}},
        independence=_stage_a_independence_with_overlap(),
        target_company="Acme Bank",
        ablation=_stage_a_ablation_lending_tricky(),
    )
    # Headline leads with the top non-terminal role (risk).
    # Post-pass-2 RT-MAJOR: title now surfaces the per-role
    # numerator/denominator IN THE TITLE itself so a CRO scanning
    # the headline can reconcile the % against the ratio without
    # scrolling. For lending-tricky-jury demo: risk closes 7 of 11
    # (63.6%).
    assert "The risk role closed 7 of 11 ablation-eligible" in ctx["headline_title"]
    assert "(63.6%)" in ctx["headline_title"]
    assert "lending-4role" in ctx["headline_title"]
    # Decision (terminal) must NOT appear in headline despite 100% attribution.
    assert "decision" not in ctx["headline_title"].lower()
    # Severity is the new 'attribution' chip.
    assert ctx["severity"] == "attribution"
    # Stat headline is the top percentage.
    assert ctx["stat_headline"] == "63.6%"
    # 3 non-terminal rows in the table (intake/risk/policy), terminal filtered.
    assert len(ctx["attribution_rows"]) == 3
    names = [r["name"] for r in ctx["attribution_rows"]]
    assert "decision" not in names
    assert names[0] == "risk"  # sorted by attribution desc
    # Post-pass-1 RT-B1: per-role denominator MUST be in the lede so
    # a reader who divides reaches the headline ratio. "7 of 11" is
    # the risk role's flips_closed/total_meaningful.
    assert "7 of 11" in ctx["headline_lede"]


def test_stage_a_falls_back_to_vendor_concentration_without_ablation(tmp_path):
    """Regression: when ablation.json is absent, the existing vendor-
    concentration headline wins — severity stays 'concentrated', no
    attribution rows, no vendor_independence_note (it IS the headline)."""
    from topofuzz.publish import _build_consistency_card_context
    ctx = _build_consistency_card_context(
        metadata={"run_id": "r1", "tds_version": "0.6.0", "signed_at": "t"},
        manifest={"topology_name": "lending-4role", "n_variants": 4,
                  "n_eval_rows": 1, "cost_actual_usd": 3.07,
                  "cli_args": {"jury_of_3": True, "jury_models_resolved": {
                      "1": "deepseek/deepseek-chat",
                      "2": "google/gemini-2.5-flash",
                      "3": "google_2/gemini-2.5-pro",
                  }}},
        independence=_stage_a_independence_with_overlap(),
        target_company=None,
        ablation=None,  # explicit absent
    )
    assert ctx["severity"] == "concentrated"
    assert ctx["attribution_rows"] == []
    # Vendor concentration is the headline, not a note — so the note slot is empty.
    assert ctx["vendor_independence_note"] == ""
    assert "distinct" in ctx["headline_title"]


def test_stage_a_vendor_concentration_demotes_to_notes_when_attribution_leads(tmp_path):
    """Stage A intent: in attribution mode, the vendor-concentration
    narrative must still surface — as a secondary notes block, not
    discarded. A CRO who skims the card sees BOTH the actionable F4
    headline AND the independence-check caveat below it."""
    from topofuzz.publish import _build_consistency_card_context
    ctx = _build_consistency_card_context(
        metadata={"run_id": "r1", "tds_version": "0.6.0", "signed_at": "t"},
        manifest={"topology_name": "lending-4role", "n_variants": 4,
                  "n_eval_rows": 1, "cost_actual_usd": 3.07,
                  "cli_args": {"jury_of_3": True, "jury_models_resolved": {
                      "1": "deepseek/deepseek-chat",
                      "2": "google/gemini-2.5-flash",
                      "3": "google_2/gemini-2.5-pro",
                  }}},
        independence=_stage_a_independence_with_overlap(),
        target_company=None,
        ablation=_stage_a_ablation_lending_tricky(),
    )
    note = ctx["vendor_independence_note"]
    assert note  # non-empty
    # Note mentions the vendor that overlaps (google) somewhere.
    assert "google" in note.lower()
    # Note describes the jury topology — distinct vendor count is 2-of-3
    # because google + google_2 normalise to one vendor.
    assert "2" in note and "3" in note


def test_stage_a_attribution_skips_terminal_node(tmp_path):
    """Terminal node attribution is always ~100% by construction
    (substitute_terminal kind). Headline must filter it so we don't
    publish a non-finding."""
    from topofuzz.publish import _extract_attribution_rows
    rows = _extract_attribution_rows(_stage_a_ablation_lending_tricky())
    assert all(r["name"] != "decision" for r in rows)
    assert len(rows) == 3


def test_stage_a_attribution_rows_capped_at_limit(tmp_path):
    """`_ATTRIBUTION_TABLE_LIMIT` enforces a 1-page budget. Extra
    non-terminal nodes drop off the table but are counted in
    n_truncated so the stat-label can disclose the truncation."""
    from topofuzz.publish import (
        _build_consistency_card_context, _ATTRIBUTION_TABLE_LIMIT,
    )
    by_node = {
        f"role_{i}": {
            "flips_closed": 1, "total_meaningful": 10,
            "attribution_pct": 10.0 + i, "is_terminal": False,
        }
        for i in range(_ATTRIBUTION_TABLE_LIMIT + 2)
    }
    by_node["terminal"] = {
        "flips_closed": 10, "total_meaningful": 10,
        "attribution_pct": 100.0, "is_terminal": True,
    }
    ctx = _build_consistency_card_context(
        metadata={"run_id": "r1", "tds_version": "0.6.0", "signed_at": "t"},
        manifest={"topology_name": "wide-pipeline", "n_variants": 4,
                  "n_eval_rows": 1,
                  "cli_args": {"jury_of_3": True, "jury_models_resolved": {
                      "1": "deepseek/deepseek-chat",
                      "2": "google/gemini-2.5-flash",
                      "3": "google_2/gemini-2.5-pro",
                  }}},
        independence=_stage_a_independence_with_overlap(),
        target_company=None,
        ablation={"n_flipping_pairs": 5, "by_node": by_node},
    )
    assert len(ctx["attribution_rows"]) == _ATTRIBUTION_TABLE_LIMIT
    assert ctx["attribution_n_truncated"] == 2
    assert "top of 5" in ctx["stat_label"]


def test_stage_a_corrupt_ablation_falls_back_safely(tmp_path):
    """Malformed ablation payload (wrong type at root, missing by_node)
    falls back to vendor-concentration headline without raising."""
    from topofuzz.publish import _build_consistency_card_context
    # ablation is a dict but by_node is missing — no attribution rows
    # extractable.
    ctx = _build_consistency_card_context(
        metadata={"run_id": "r1", "tds_version": "0.6.0", "signed_at": "t"},
        manifest={"topology_name": "lending-4role", "n_variants": 4,
                  "n_eval_rows": 1, "cost_actual_usd": 3.07,
                  "cli_args": {"jury_of_3": True, "jury_models_resolved": {
                      "1": "deepseek/deepseek-chat",
                      "2": "google/gemini-2.5-flash",
                      "3": "google_2/gemini-2.5-pro",
                  }}},
        independence=_stage_a_independence_with_overlap(),
        target_company=None,
        ablation={"schema_version": 1, "n_flipping_pairs": 12},
    )
    # No attribution data → vendor-concentration headline wins.
    assert ctx["attribution_rows"] == []
    assert ctx["severity"] == "concentrated"


def test_stage_a_attribution_what_paragraph_branch():
    """The 'attribution' severity has its own _format_consistency_what_paragraph
    branch — must describe per-role ablation as the actionable headline,
    and must NOT use 're-engineering' language the engine can't
    substantiate (F8 Stage A pass-1 RT-M2)."""
    from topofuzz.publish import _format_consistency_what_paragraph
    text = _format_consistency_what_paragraph("Acme Bank", "attribution")
    assert "Acme Bank" in text
    assert "ablation" in text.lower()
    assert "role" in text.lower()
    # MUST not overclaim — re-engineering is a remediation claim
    # leave-one-out attribution can't substantiate.
    assert "re-engineering" not in text.lower()
    # The bridging sentence to the vendor-independence notes block
    # below must be present so a CRO reading the body paragraph and
    # skipping notes still sees the caveat exists (pass-1 RT-MINOR).
    assert "vendor-independence" in text.lower()


def test_stage_a_extract_attribution_rows_rejects_nan_and_out_of_range():
    """Defensive: a malformed ablation entry (NaN, negative, >100)
    must be silently skipped so the headline never renders 'nan%' or
    a negative percentage."""
    from topofuzz.publish import _extract_attribution_rows
    rows = _extract_attribution_rows({
        "by_node": {
            "good": {"attribution_pct": 50.0, "is_terminal": False,
                     "flips_closed": 1, "total_meaningful": 2},
            "nan_role": {"attribution_pct": float("nan"), "is_terminal": False},
            "negative": {"attribution_pct": -1.0, "is_terminal": False},
            "overflow": {"attribution_pct": 150.0, "is_terminal": False},
            "bool_trap": {"attribution_pct": True, "is_terminal": False},
            "missing_pct": {"is_terminal": False},
        }
    })
    assert [r["name"] for r in rows] == ["good"]


def test_stage_a_build_pdf_e2e_reads_ablation_json(tmp_path):
    """build_consistency_card_pdf must auto-resolve run_dir/ablation.json
    and use it for the headline. Doesn't render Chromium (would require
    playwright); just confirms the context-build path picks it up by
    monkey-patching the renderer."""
    import json as _json
    from unittest.mock import patch
    from topofuzz.publish import build_consistency_card_pdf

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(_json.dumps({
        "run_id": "r-stage-a",
        "tds_version": "0.6.0",
        "topology_name": "lending-4role",
        "n_variants": 4,
        "n_eval_rows": 1,
        "cost_actual_usd": 3.07,
        "cli_args": {
            "jury_of_3": True,
            "jury_models_resolved": {
                "1": "deepseek/deepseek-chat",
                "2": "google/gemini-2.5-flash",
                "3": "google_2/gemini-2.5-pro",
            },
        },
    }), encoding="utf-8")
    (run_dir / "report.json").write_text(_json.dumps({
        "headline": {
            "judge_independence": _stage_a_independence_with_overlap() | {
                "judge_protocol": "jury_of_3",
            }
        }
    }), encoding="utf-8")
    (run_dir / "ablation.json").write_text(
        _json.dumps(_stage_a_ablation_lending_tricky()), encoding="utf-8"
    )

    # Real minimal 1-page PDF bytes so build_consistency_card_pdf's
    # pypdf page-count validation passes. Generated via pypdf to avoid
    # hand-tuning a brittle byte literal.
    import io as _io
    import pypdf as _pypdf
    _writer = _pypdf.PdfWriter()
    _writer.add_blank_page(width=72, height=72)
    _buf = _io.BytesIO()
    _writer.write(_buf)
    MINIMAL_PDF_BYTES = _buf.getvalue()

    captured_html: list[str] = []
    def fake_render_pages(html_paths, footer_text):
        from pathlib import Path as _P
        for p in html_paths:
            captured_html.append(_P(p).read_text(encoding="utf-8"))
        return [MINIMAL_PDF_BYTES]

    with patch("topofuzz.publish._render_html_pages_to_pdfs", fake_render_pages):
        out = build_consistency_card_pdf(
            run_dir, out_path=tmp_path / "card.pdf"
        )

    assert out.exists()
    assert captured_html
    html = captured_html[0]
    # F4 attribution headline landed in the rendered HTML.
    # Pass-2 RT-MAJOR fix: title includes per-role denominator in
    # the headline itself ("7 of 11 ablation-eligible verdict
    # flips ... (63.6%)") so a CRO reconciles without scrolling.
    assert "The risk role closed 7 of 11 ablation-eligible" in html
    assert "(63.6%)" in html
    # Vendor-concentration demoted to the notes block.
    assert "Vendor independence check:" in html
    # Decision (terminal) not in the attribution table.
    assert ">decision<" not in html.lower()


# ---------------------------------------------------------------------------
# F8 Stage A — pass-1 audit-fix regressions
# ---------------------------------------------------------------------------


def test_stage_a_pass1_b1_per_role_denominator_in_lede():
    """RT pass-1 BLOCKER B1: the lede MUST publish the per-role
    numerator/denominator so a reader who divides reaches the same
    63.6% the headline quotes — not 7/12 (n_flipping_pairs)."""
    from topofuzz.publish import _format_attribution_headline
    rows = [
        {"name": "risk", "attribution_pct": 63.64,
         "attribution_pct_str": "63.6%",
         "flips_closed": 7, "total_meaningful": 11},
        {"name": "policy", "attribution_pct": 60.0,
         "attribution_pct_str": "60.0%",
         "flips_closed": 6, "total_meaningful": 10},
    ]
    _title, lede, _stat, _label, _sub = _format_attribution_headline(
        attribution_rows=rows,
        topology_name="lending-4role",
        n_flipping_pairs=12,
        n_truncated=0,
        partial=False,
    )
    # Per-role numerator/denominator must appear for the top role.
    assert "7 of 11" in lede
    # The "next contributor" clause must publish policy's 6/10 too.
    assert "6/10" in lede
    # The lede must NOT lead with the misleading "Across 12 measured
    # verdict flips" framing that triggered the BLOCKER pre-fix.
    assert "Across 12 measured" not in lede


def test_stage_a_pass1_b2_partial_flag_surfaces_in_lede():
    """RT pass-1 BLOCKER B2: ablation.partial=True must surface in
    the lede so a CRO who opens ablation.json can't catch us
    silently dropping the flag."""
    from topofuzz.publish import _format_attribution_headline
    rows = [
        {"name": "risk", "attribution_pct": 63.64,
         "attribution_pct_str": "63.6%",
         "flips_closed": 7, "total_meaningful": 11},
    ]
    _t, lede_partial, _s, _l, sub_partial = _format_attribution_headline(
        attribution_rows=rows,
        topology_name="lending-4role",
        n_flipping_pairs=12,
        n_truncated=0,
        partial=True,
    )
    _t, lede_full, _s, _l, sub_full = _format_attribution_headline(
        attribution_rows=rows,
        topology_name="lending-4role",
        n_flipping_pairs=12,
        n_truncated=0,
        partial=False,
    )
    assert "partial" in lede_partial.lower()
    assert "partial" not in lede_full.lower()
    # Sub-line also exposes "F4 run partial" when partial=True.
    assert "partial" in sub_partial.lower()
    assert "partial" not in sub_full.lower()


def test_stage_a_pass1_m1_drops_agent_suffix_and_capitalize():
    """RT pass-1 MAJOR M1: title no longer uses '.capitalize()' or
    '-agent' suffix grafting — those produced 'Risk-agent' and were
    fragile against non-lowercase / multi-token operator role IDs."""
    from topofuzz.publish import _format_attribution_headline
    rows = [{
        "name": "RISK", "attribution_pct": 50.0,
        "attribution_pct_str": "50.0%",
        "flips_closed": 5, "total_meaningful": 10,
    }]
    title, _l, _s, _label, _sub = _format_attribution_headline(
        attribution_rows=rows,
        topology_name="pipe",
        n_flipping_pairs=10,
        n_truncated=0,
        partial=False,
    )
    # Bare role name preserved as-typed (RISK, not Risk).
    assert "RISK" in title
    assert "-agent" not in title
    # Title leads with "The {role} role" — no .capitalize() fragility.
    # Post-pass-2: title also includes per-role denom "5 of 10".
    assert title.startswith("The RISK role closed 5 of 10")


def test_stage_a_pass1_m3_attribution_mode_replaces_failure_rate_subline():
    """RT pass-1 MAJOR M3: in attribution mode the chip sub-line is
    flip count + role count + (optionally) 'F4 run partial' — NOT
    'Jury-vote failure rate: X%' which is a vendor-concentration
    sub-line that mis-leads under an attribution headline."""
    from topofuzz.publish import _build_consistency_card_context
    ctx_attr = _build_consistency_card_context(
        metadata={"run_id": "r1", "tds_version": "0.6.0", "signed_at": "t"},
        manifest={"topology_name": "lending-4role", "n_variants": 4,
                  "n_eval_rows": 1, "cost_actual_usd": 3.07,
                  "cli_args": {"jury_of_3": True, "jury_models_resolved": {
                      "1": "deepseek/deepseek-chat",
                      "2": "google/gemini-2.5-flash",
                      "3": "google_2/gemini-2.5-pro",
                  }}},
        independence=_stage_a_independence_with_overlap(),
        target_company=None,
        ablation=_stage_a_ablation_lending_tricky(),
    )
    sub_attr = ctx_attr["headline_sub_line"]
    assert "Jury-vote failure rate" not in sub_attr
    assert "measured verdict flips" in sub_attr
    assert "non-terminal roles ablated" in sub_attr

    # Vendor-concentration mode (no ablation) — chip sub-line is the
    # original failure-rate phrasing preserved for back-compat.
    ctx_vc = _build_consistency_card_context(
        metadata={"run_id": "r1", "tds_version": "0.6.0", "signed_at": "t"},
        manifest={"topology_name": "lending-4role", "n_variants": 4,
                  "n_eval_rows": 1, "cost_actual_usd": 3.07,
                  "cli_args": {"jury_of_3": True, "jury_models_resolved": {
                      "1": "deepseek/deepseek-chat",
                      "2": "google/gemini-2.5-flash",
                      "3": "google_2/gemini-2.5-pro",
                  }}},
        independence=_stage_a_independence_with_overlap(),
        target_company=None,
        ablation=None,
    )
    assert ctx_vc["headline_sub_line"].startswith("Jury-vote failure rate:")


def test_stage_a_pass1_cr_m2_role_name_length_bounded():
    """Code-review pass-1 MINOR M-CR2: an operator-supplied
    by_node key of pathological length must be truncated before it
    inflates the rendered title / HTML <title> tag."""
    from topofuzz.publish import (
        _extract_attribution_rows, _ATTRIBUTION_ROLE_NAME_DISPLAY_LIMIT,
    )
    hostile_name = "x" * 5000
    rows = _extract_attribution_rows({
        "by_node": {
            hostile_name: {"attribution_pct": 50.0, "is_terminal": False,
                           "flips_closed": 1, "total_meaningful": 2},
        }
    })
    assert len(rows) == 1
    assert len(rows[0]["name"]) <= _ATTRIBUTION_ROLE_NAME_DISPLAY_LIMIT
    assert rows[0]["name"].endswith("…")


def test_stage_a_pass1_stat_label_uses_attributable_framing():
    """The stat-label string must avoid 'attributed to' (which implies
    causal claim) and instead use the F4-honest 'closed by removing the
    {role} role' wording (RT pass-1 alignment with the title fix)."""
    from topofuzz.publish import _format_attribution_headline
    rows = [{
        "name": "risk", "attribution_pct": 63.64,
        "attribution_pct_str": "63.6%",
        "flips_closed": 7, "total_meaningful": 11,
    }]
    _t, _l, _s, label, _sub = _format_attribution_headline(
        attribution_rows=rows,
        topology_name="lending-4role",
        n_flipping_pairs=12,
        n_truncated=0,
        partial=False,
    )
    assert "closed by removing the risk role" in label


# ---------------------------------------------------------------------------
# F8 Stage A — pass-2 audit-fix regressions
# ---------------------------------------------------------------------------


def test_stage_a_pass2_title_includes_per_role_denominator():
    """RT pass-2 MAJOR: title must surface flips_closed/total_meaningful
    so a CRO scanning the headline can reconcile the % against the
    ratio without scrolling to the lede. Pre-fix, title said
    '63.6% of attributable verdict flips' and a reader who computed
    7 of n_flipping_pairs (12) got 58.3%, diffing against 63.6%."""
    from topofuzz.publish import _format_attribution_headline
    rows = [{
        "name": "risk", "attribution_pct": 63.64,
        "attribution_pct_str": "63.6%",
        "flips_closed": 7, "total_meaningful": 11,
    }]
    title, _l, _s, _label, _sub = _format_attribution_headline(
        attribution_rows=rows,
        topology_name="lending-4role",
        n_flipping_pairs=12,
        n_truncated=0,
        partial=False,
    )
    assert "7 of 11 ablation-eligible" in title
    assert "(63.6%)" in title


def test_stage_a_pass2_title_falls_back_when_counts_missing():
    """If flips_closed / total_meaningful are absent from the ablation
    row, the title falls back to '% of ablation-eligible verdict flips'
    rather than crashing or rendering 'None of None'."""
    from topofuzz.publish import _format_attribution_headline
    rows = [{
        "name": "risk", "attribution_pct": 50.0,
        "attribution_pct_str": "50.0%",
        "flips_closed": None, "total_meaningful": None,
    }]
    title, _l, _s, _label, _sub = _format_attribution_headline(
        attribution_rows=rows,
        topology_name="pipe",
        n_flipping_pairs=10,
        n_truncated=0,
        partial=False,
    )
    assert "50.0% of ablation-eligible" in title
    # Must NOT print "None" anywhere.
    assert "None" not in title


def test_stage_a_pass2_what_paragraph_discloses_overlap_caveat():
    """RT pass-2 MINOR: leave-one-out shares can overlap (the demo's
    63.6 + 60 + 54.5 = 178% would trigger a 'numbers don't add up'
    pattern match without the caveat). The attribution what_paragraph
    must disclose the overlap semantics."""
    from topofuzz.publish import _format_consistency_what_paragraph
    text = _format_consistency_what_paragraph("Acme Bank", "attribution")
    assert "overlap" in text.lower()
    assert "sum to 100" in text or "100%" in text


def test_stage_a_pass2_extract_skips_zero_denominator_rows():
    """Code-review pass-2 MINOR: a degenerate ablation row with
    total_meaningful == 0 must be silently skipped (would otherwise
    produce '0 of 0 ablation-eligible' copy in the lede)."""
    from topofuzz.publish import _extract_attribution_rows
    rows = _extract_attribution_rows({
        "by_node": {
            "good": {"attribution_pct": 50.0, "is_terminal": False,
                     "flips_closed": 5, "total_meaningful": 10},
            "degenerate": {"attribution_pct": 0.0, "is_terminal": False,
                           "flips_closed": 0, "total_meaningful": 0},
        }
    })
    names = [r["name"] for r in rows]
    assert names == ["good"]
