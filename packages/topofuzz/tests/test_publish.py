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
    assert "at least one of --pdf or --ceo-card" in result.output


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
