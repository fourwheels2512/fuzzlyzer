# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Tests for ``topofuzz inspect`` — side-by-side flip inspector."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from topofuzz.cli import app
from topofuzz.inspect import (
    _MAX_RESPONSE_BYTES,
    DEFAULT_VERDICT_THRESHOLD,
    InspectError,
    _highlight_risk_phrases,
    _truncate_for_render,
    build_inspect_html,
    find_inspect_pair,
    write_inspect_html,
)


# ---------------------------------------------------------------------------
# Synthetic run fixture builder
# ---------------------------------------------------------------------------


_CANONICAL_HASH = "a" * 64
_FLIPPING_HASH = "b" * 64


def _node_output(node_id: str, role: str, response_text: str) -> dict:
    """Match the dataclass-asdict shape the runner writes."""
    return {
        "node_id": node_id,
        "role": role,
        "response_text": response_text,
        "model_served_version": "google/gemini-2.5-flash",
        "cache_hit": False,
        "latency_ms": 100,
        "input_tokens": 50,
        "output_tokens": 50,
        "cost_usd": 0.001,
    }


def _make_run(
    tmp_path: Path,
    *,
    with_flip: bool = True,
    extra_canonical_text: str = "We recommend approving this application.",
    extra_flipping_text: str = "We must deny this application — concerning patterns.",
    judge_pass_canonical: bool | None = True,
    judge_pass_flipping: bool | None = False,
    judge_rationale_canonical: str | None = "Approve",
    judge_rationale_flipping: str | None = "Deny — risk profile",
) -> Path:
    """Build a minimal valid run_dir for inspect tests.

    Two variants (canonical + one alternate ordering) and one eval row,
    with `cells` + `judge_results` rows populated. ``with_flip=False``
    makes both variants share the same verdict so find_inspect_pair
    raises.
    """
    run_dir = tmp_path / "synthetic-run"
    run_dir.mkdir()

    manifest = {
        "run_id": "synth-run-0001",
        "topology_canonical_hash": "h" * 64,
        "topology_name": "synth",
        "tds_version": "0.5.0",
        "tenant_id": "local",
        "n_variants": 2,
        "n_eval_rows": 1,
        "variant_canonical_hashes": [_CANONICAL_HASH, _FLIPPING_HASH],
        "variant_orderings": [["n1", "n2"], ["n2", "n1"]],
        "eval_ids": ["eval_001"],
        "eval_ground_truth": [1],
        "agent_vendors": ["google"],
        "judge_model": "google/gemini-2.5-pro",
        "rubric_hash": "r" * 64,
        "semaphore_size": 8,
        "max_cost_usd": 1.0,
        "cli_args": {
            "fixture": "tests/data/fake_fixture.py:build_graph",
            "eval": "eval.csv",
            "rng_seed": 42,
        },
        "kickoff_ts": 0,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE cells ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, ir_canonical_hash TEXT, "
        "node_outputs_json TEXT, error TEXT, cell_cost_usd REAL)"
    )
    db.execute(
        "CREATE TABLE judge_results ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, judge_kind TEXT, "
        "score_value REAL, passed INTEGER, rationale TEXT, failure_reason TEXT)"
    )
    canonical_outputs = [
        _node_output("n1", "underwriter", "Initial analysis: looks healthy."),
        _node_output("n2", "policy_reviewer", extra_canonical_text),
    ]
    flipping_outputs = [
        _node_output("n2", "policy_reviewer", "Pre-review: I see exceptions."),
        _node_output("n1", "underwriter", extra_flipping_text),
    ]
    db.executemany(
        "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "synth-run-0001",
                _CANONICAL_HASH,
                "eval_001",
                "ir-canonical-hash",
                json.dumps(canonical_outputs),
                None,
                0.002,
            ),
            (
                "synth-run-0001",
                _FLIPPING_HASH,
                "eval_001",
                "ir-canonical-hash-2",
                json.dumps(flipping_outputs),
                None,
                0.002,
            ),
        ],
    )
    if with_flip:
        rows = [
            (
                "synth-run-0001", _CANONICAL_HASH, "eval_001", "binary",
                None, 1 if judge_pass_canonical else 0, judge_rationale_canonical, None,
            ),
            (
                "synth-run-0001", _FLIPPING_HASH, "eval_001", "binary",
                None, 1 if judge_pass_flipping else 0, judge_rationale_flipping, None,
            ),
        ]
    else:
        rows = [
            (
                "synth-run-0001", _CANONICAL_HASH, "eval_001", "binary",
                None, 1, "Approve", None,
            ),
            (
                "synth-run-0001", _FLIPPING_HASH, "eval_001", "binary",
                None, 1, "Approve", None,
            ),
        ]
    db.executemany(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    db.commit()
    db.close()
    return run_dir


# ---------------------------------------------------------------------------
# find_inspect_pair
# ---------------------------------------------------------------------------


def test_find_inspect_pair_returns_canonical_left(tmp_path):
    """Canonical is always idx 0; flipping is the smallest other idx."""
    run_dir = _make_run(tmp_path, with_flip=True)
    eid, canonical_idx, flipping_idx = find_inspect_pair(run_dir)
    assert eid == "eval_001"
    assert canonical_idx == 0
    assert flipping_idx == 1


def test_find_inspect_pair_no_flip_raises(tmp_path):
    run_dir = _make_run(tmp_path, with_flip=False)
    with pytest.raises(InspectError, match="No verdict flip"):
        find_inspect_pair(run_dir)


def test_find_inspect_pair_unknown_eval_raises(tmp_path):
    run_dir = _make_run(tmp_path, with_flip=True)
    with pytest.raises(InspectError, match="no scored judge results"):
        find_inspect_pair(run_dir, eval_id="missing")


def test_find_inspect_pair_named_eval_with_no_flip_raises(tmp_path):
    """Naming an eval that exists but doesn't flip raises a clear error."""
    run_dir = _make_run(tmp_path, with_flip=False)
    with pytest.raises(InspectError, match="No verdict flip"):
        find_inspect_pair(run_dir, eval_id="eval_001")


def test_find_inspect_pair_skips_judge_failures(tmp_path):
    """NULL-score + NULL-passed rows contribute no side of the flip."""
    run_dir = _make_run(tmp_path, with_flip=True)
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    # Add a third variant whose judge failed — must not perturb the pair.
    third_hash = "c" * 64
    db.execute(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("synth-run-0001", third_hash, "eval_001", "binary", None, None, None, "judge timeout"),
    )
    db.commit()
    db.close()
    eid, c_idx, f_idx = find_inspect_pair(run_dir)
    assert (eid, c_idx, f_idx) == ("eval_001", 0, 1)


def test_find_inspect_pair_original_judge_failure_named_raises(tmp_path):
    """When the named eval's original variant has no verdict, raise clearly."""
    run_dir = _make_run(tmp_path, with_flip=True)
    # Wipe the canonical's verdict on eval_001 to simulate a judge failure.
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "DELETE FROM judge_results WHERE variant_id=?", (_CANONICAL_HASH,),
    )
    db.execute(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("synth-run-0001", _CANONICAL_HASH, "eval_001", "binary", None, None, None, "judge_fail"),
    )
    db.commit()
    db.close()
    with pytest.raises(InspectError, match="no scored verdict"):
        find_inspect_pair(run_dir, eval_id="eval_001")


def test_find_inspect_pair_empty_variant_hashes_raises(tmp_path):
    run_dir = _make_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["variant_canonical_hashes"] = []
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(InspectError, match="variant_canonical_hashes is empty"):
        find_inspect_pair(run_dir)


# ---------------------------------------------------------------------------
# build_inspect_html
# ---------------------------------------------------------------------------


def test_build_inspect_html_renders_both_panels(tmp_path):
    run_dir = _make_run(tmp_path, with_flip=True)
    html, eid = build_inspect_html(run_dir)
    assert eid == "eval_001"
    # Both verdict pills appear.
    assert "verdict-approve" in html
    assert "verdict-deny" in html
    # Both variant short hashes appear.
    assert _CANONICAL_HASH[:8] in html
    assert _FLIPPING_HASH[:8] in html
    # Both response texts appear (autoescape is fine for plain ASCII).
    # Note: risk-phrase highlighting wraps "deny" in a <span>, so check
    # the surrounding context rather than the whole phrase.
    assert "looks healthy" in html
    assert "must" in html and "this application" in html
    assert '<span class="phrase-deny">deny</span>' in html


def test_build_inspect_html_highlights_risk_phrases(tmp_path):
    run_dir = _make_run(tmp_path, with_flip=True)
    html, _ = build_inspect_html(run_dir)
    # "approving" → phrase-approve, "deny" → phrase-deny, "concerning" → concern.
    assert '<span class="phrase-approve">approving</span>' in html
    assert '<span class="phrase-deny">deny</span>' in html
    assert '<span class="phrase-concern">concerning</span>' in html


def test_build_inspect_html_byte_stable_double_render(tmp_path):
    """Two renders of the same (run_dir, eval_id) must be byte-identical.

    Guards against `datetime.now()` or any process-salted output leaking
    in. This is the design contract that lets `topofuzz inspect` be
    diffed in CI.
    """
    run_dir = _make_run(tmp_path, with_flip=True)
    a, _ = build_inspect_html(run_dir)
    b, _ = build_inspect_html(run_dir)
    assert a == b


def test_build_inspect_html_escapes_hostile_response(tmp_path):
    """LLM-generated text must NOT be able to inject HTML."""
    run_dir = _make_run(
        tmp_path,
        extra_canonical_text="<script>alert('xss')</script>",
        extra_flipping_text="deny <img src=x onerror=alert(1)>",
    )
    html, _ = build_inspect_html(run_dir)
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
    assert "<img src=x onerror" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_build_inspect_html_escapes_hostile_rationale(tmp_path):
    """Judge rationale is also LLM text — must escape, not render."""
    run_dir = _make_run(
        tmp_path,
        judge_rationale_canonical="<b>raw HTML</b>",
        judge_rationale_flipping="deny <script>x</script>",
    )
    html, _ = build_inspect_html(run_dir)
    assert "<b>raw HTML</b>" not in html
    assert "&lt;b&gt;raw HTML&lt;/b&gt;" in html
    assert "<script>x</script>" not in html


def test_build_inspect_html_handles_missing_rationale(tmp_path):
    run_dir = _make_run(
        tmp_path,
        judge_rationale_canonical=None,
        judge_rationale_flipping=None,
    )
    html, _ = build_inspect_html(run_dir)
    # The "Judge rationale" header should not appear when both are None.
    assert "Judge rationale" not in html


def test_build_inspect_html_renders_orderings(tmp_path):
    run_dir = _make_run(tmp_path)
    html, _ = build_inspect_html(run_dir)
    # Each ordering arrow appears between node ids.
    assert "n1" in html and "n2" in html
    # `&rarr;` rendered (arrow between nodes).
    assert "&rarr;" in html or "→" in html


# ---------------------------------------------------------------------------
# Highlight unit tests (boundary correctness)
# ---------------------------------------------------------------------------


def test_highlight_word_boundary_respected():
    """`underwriter` must NOT match `under` from any deny stem; `denial` must."""
    s = "The denial was upheld. underwriter approved."
    out = _highlight_risk_phrases(s)
    assert '<span class="phrase-deny">denial</span>' in out
    # "underwriter" has no embedded risk word; should not be wrapped.
    assert "underwriter" in out
    # "approved" should be wrapped though.
    assert '<span class="phrase-approve">approved</span>' in out


def test_highlight_does_not_double_wrap():
    """Two passes over the same text must not nest spans."""
    s = "deny deny deny"
    out = _highlight_risk_phrases(s)
    # Three matches, each wrapped exactly once.
    assert out.count('<span class="phrase-deny">') == 3
    assert out.count("</span>") == 3


def test_highlight_handles_case_insensitive():
    s = "DENY this application; Approval pending; CONCERNING."
    out = _highlight_risk_phrases(s)
    assert '<span class="phrase-deny">DENY</span>' in out
    assert '<span class="phrase-approve">Approval</span>' in out
    assert '<span class="phrase-concern">CONCERNING</span>' in out


def test_highlight_multi_word_high_risk():
    """`high risk` and `high-risk` both match the concern group."""
    s = "Flagged: high risk and high-risk patterns."
    out = _highlight_risk_phrases(s)
    assert '<span class="phrase-concern">high risk</span>' in out
    assert '<span class="phrase-concern">high-risk</span>' in out


def test_highlight_empty_string_no_op():
    assert _highlight_risk_phrases("") == ""


def test_highlight_no_match_no_change():
    s = "Nothing relevant here, just neutral text."
    assert _highlight_risk_phrases(s) == s


# ---------------------------------------------------------------------------
# write_inspect_html
# ---------------------------------------------------------------------------


def test_write_inspect_html_default_path(tmp_path):
    run_dir = _make_run(tmp_path)
    written = write_inspect_html(run_dir)
    assert written == run_dir / "inspect_eval_001.html"
    assert written.is_file()
    body = written.read_text(encoding="utf-8")
    assert "Where did the AIs diverge?" in body


def test_write_inspect_html_custom_out(tmp_path):
    run_dir = _make_run(tmp_path)
    target = tmp_path / "elsewhere.html"
    written = write_inspect_html(run_dir, out_path=target)
    assert written == target
    assert target.is_file()


def test_write_inspect_html_unsafe_eval_id_refused(tmp_path):
    """Path-traversal-bait eval_ids are rejected when using the default path.

    A caller-provided ``out_path`` still works — the caller owns that path.
    """
    # Build a run whose eval_id has a slash.
    run_dir = tmp_path / "unsafe-run"
    run_dir.mkdir()
    manifest = {
        "run_id": "u",
        "variant_canonical_hashes": ["a" * 64, "b" * 64],
        "variant_orderings": [["n1"], ["n1"]],
        "kickoff_ts": 0,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE cells (run_id TEXT, variant_id TEXT, eval_id TEXT, "
        "ir_canonical_hash TEXT, node_outputs_json TEXT, error TEXT, "
        "cell_cost_usd REAL)"
    )
    db.execute(
        "CREATE TABLE judge_results (run_id TEXT, variant_id TEXT, "
        "eval_id TEXT, judge_kind TEXT, score_value REAL, passed INTEGER, "
        "rationale TEXT, failure_reason TEXT)"
    )
    unsafe_eid = "../../etc/passwd"
    db.executemany(
        "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("u", "a" * 64, unsafe_eid, "h", json.dumps([_node_output("n1", "r", "ok")]), None, 0),
            ("u", "b" * 64, unsafe_eid, "h", json.dumps([_node_output("n1", "r", "ok")]), None, 0),
        ],
    )
    db.executemany(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("u", "a" * 64, unsafe_eid, "binary", None, 1, "ok", None),
            ("u", "b" * 64, unsafe_eid, "binary", None, 0, "ok", None),
        ],
    )
    db.commit()
    db.close()
    with pytest.raises(InspectError, match="characters not permitted"):
        write_inspect_html(run_dir)


# ---------------------------------------------------------------------------
# CLI subcommand
# ---------------------------------------------------------------------------


def test_cli_inspect_happy_path(tmp_path):
    run_dir = _make_run(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(run_dir)])
    assert result.exit_code == 0, result.output
    expected = run_dir / "inspect_eval_001.html"
    assert expected.is_file()
    assert str(expected) in result.output


def test_cli_inspect_no_flip_exit1(tmp_path):
    run_dir = _make_run(tmp_path, with_flip=False)
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(run_dir)])
    assert result.exit_code == 1
    assert "No verdict flip" in (result.stderr or result.output)


def test_cli_inspect_missing_dir_exit2(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(tmp_path / "nope")])
    assert result.exit_code == 2
    assert "not a directory" in (result.stderr or result.output)


def test_cli_inspect_eval_option_plumbed(tmp_path):
    run_dir = _make_run(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(run_dir), "--eval", "eval_001"])
    assert result.exit_code == 0


def test_cli_inspect_out_option(tmp_path):
    run_dir = _make_run(tmp_path)
    custom = tmp_path / "my_inspect.html"
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(run_dir), "--out", str(custom)])
    assert result.exit_code == 0
    assert custom.is_file()


# ---------------------------------------------------------------------------
# Determinism contract regression
# ---------------------------------------------------------------------------


def test_inspect_html_has_no_now_leak(tmp_path):
    """The current year/date must NOT leak into the rendered HTML.

    Belt-and-braces guard against a future template edit that calls
    `now()` and breaks the byte-stability contract that
    `test_build_inspect_html_byte_stable_double_render` enforces.
    """
    import datetime as _dt

    run_dir = _make_run(tmp_path)
    html, _ = build_inspect_html(run_dir)
    year = str(_dt.datetime.now().year)
    # The manifest's kickoff_ts=0 → 1970-01-01, so the current year only
    # appears if a render-time clock leaked in.
    if year != "1970":
        assert year not in html, (
            "Current year leaked into inspect HTML — "
            "template called now()? Byte-stability is broken."
        )


# ---------------------------------------------------------------------------
# Audit-driven hardening tests (RT pass-1 MAJOR/MINOR + CR pass-1)
# ---------------------------------------------------------------------------


def test_load_cell_rejects_non_dict_node_outputs_elements(tmp_path):
    """Audit RT-pass-1 MINOR-2 / CR MAJOR-1: each node_outputs item
    must be a dict so `_agent_view` can `.get(...)` on it. A stray
    primitive raises a clean `InspectError`, not `AttributeError` from
    deep inside the renderer.
    """
    run_dir = _make_run(tmp_path, with_flip=True)
    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "UPDATE cells SET node_outputs_json=? WHERE variant_id=?",
        (json.dumps(["raw string", 42]), _CANONICAL_HASH),
    )
    db.commit()
    db.close()
    with pytest.raises(InspectError, match="expected object"):
        build_inspect_html(run_dir)


def test_build_inspect_html_rejects_truncated_variant_orderings(tmp_path):
    """Audit RT-pass-1 MAJOR-2 / CR MAJOR-2: a manifest with fewer
    `variant_orderings` than `variant_canonical_hashes` raises
    `InspectError`, not `IndexError`.
    """
    run_dir = _make_run(tmp_path, with_flip=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    # Drop the second ordering even though both variant hashes are still
    # present — the inspector should refuse this manifest.
    manifest["variant_orderings"] = [manifest["variant_orderings"][0]]
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(InspectError, match="variant_orderings"):
        build_inspect_html(run_dir)


def test_build_inspect_html_rejects_non_list_variant_orderings(tmp_path):
    run_dir = _make_run(tmp_path, with_flip=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["variant_orderings"] = "not a list"
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(InspectError, match="not a list"):
        build_inspect_html(run_dir)


def test_build_inspect_html_rejects_non_list_ordering_element(tmp_path):
    """Audit RT-pass-2 MINOR-B: a per-element type check refuses a
    string/dict element that `list(...)` would otherwise spread into
    a nonsensical char-list of "node ids" in the template.
    """
    run_dir = _make_run(tmp_path, with_flip=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    # Replace the flipping variant's ordering with a string — `list("ab")`
    # would otherwise produce `["a", "b"]` and template-render nonsense.
    manifest["variant_orderings"][1] = "abc"
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(InspectError, match="expected list"):
        build_inspect_html(run_dir)


def test_response_text_truncated_to_cap(tmp_path):
    """Audit RT-pass-1 MINOR-1: pathological response_text is capped
    with a sentinel so a 100MB cell can't blow the rendered payload."""
    run_dir = _make_run(
        tmp_path,
        extra_canonical_text="A" * (_MAX_RESPONSE_BYTES * 2),
    )
    html, _ = build_inspect_html(run_dir)
    assert "[truncated by topofuzz inspect" in html
    # Sanity: the rendered HTML is bounded — not a multi-megabyte string.
    assert len(html) < _MAX_RESPONSE_BYTES * 4


def test_truncate_for_render_is_utf8_safe():
    """Truncation must land on a UTF-8 codepoint boundary so the
    rendered HTML is always valid UTF-8 (no U+FFFD splash)."""
    # Each '☠' (☠) encodes as 3 bytes — picking a cap at 2 bytes
    # forces a mid-codepoint cut that must back off cleanly.
    out = _truncate_for_render("☠☠☠", max_bytes=2)
    assert "�" not in out
    # Encoded bytes never exceed the cap + sentinel.
    assert "[truncated" in out


def test_truncate_for_render_passes_through_short_text():
    assert _truncate_for_render("short", max_bytes=10) == "short"


def test_rationale_none_renders_no_block(tmp_path):
    """A None rationale must yield Markup('') so the template hides
    the rationale block — not the literal string 'None'."""
    run_dir = _make_run(
        tmp_path,
        judge_rationale_canonical=None,
        judge_rationale_flipping=None,
    )
    html, _ = build_inspect_html(run_dir)
    assert "Judge rationale" not in html
    assert "None" not in html  # not the literal Python repr


def test_rationale_empty_string_renders_no_block(tmp_path):
    """An empty rationale is functionally indistinguishable from None
    for display — hide the block instead of rendering a blank box."""
    run_dir = _make_run(
        tmp_path,
        judge_rationale_canonical="",
        judge_rationale_flipping="",
    )
    html, _ = build_inspect_html(run_dir)
    assert "Judge rationale" not in html


def test_rationale_truncated_at_cap(tmp_path):
    """A pathologically long rationale is capped with the same sentinel."""
    long_rationale = "x" * (_MAX_RESPONSE_BYTES * 2)
    run_dir = _make_run(
        tmp_path,
        judge_rationale_canonical=long_rationale,
        judge_rationale_flipping=long_rationale,
    )
    html, _ = build_inspect_html(run_dir)
    assert "[truncated by topofuzz inspect" in html


def test_atomic_write_no_orphan_tmp_on_success(tmp_path):
    """The atomic-write tmp file must be replaced onto the target, not
    leaked alongside it (audit RT-pass-1 MAJOR-1)."""
    run_dir = _make_run(tmp_path)
    target = write_inspect_html(run_dir)
    assert target.is_file()
    leftovers = [
        p for p in run_dir.iterdir()
        if p.name.startswith(f".{target.name}.tmp.")
    ]
    assert leftovers == [], f"orphan tmp files left behind: {leftovers}"


def test_cli_inspect_rejects_unknown_flag(tmp_path):
    """Audit CR pass-1 MINOR-2: `_reject_unknown_extras` must fire on a
    typo flag like `--variantss`."""
    run_dir = _make_run(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(run_dir), "--variantss", "5"])
    assert result.exit_code != 0


def test_cli_inspect_rejects_reserved_flag(tmp_path):
    """Audit CR pass-1 MINOR-2: `_reject_reserved_flags` must fire on
    a reserved flag like `--sandbox`."""
    run_dir = _make_run(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(run_dir), "--sandbox", "foo"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# UI route shape — sanity-check that the no-flip 200 wrapper exists
# ---------------------------------------------------------------------------


def test_ui_app_has_no_flips_html_constant():
    """Audit RT-pass-1 MINOR-4 / CR MINOR-1: the no-flip HTMLResponse
    fallback exists and is sized like a real page, not an empty string.
    The route's `InspectError → HTMLResponse(_NO_FLIPS_HTML)` branch is
    not worth wiring through the closure-local RunRegistry; this
    constant-shape check guards the friendly-empty-state contract.
    """
    from topofuzz.ui.app import _INSPECT_HEADERS, _NO_FLIPS_HTML

    assert "<!DOCTYPE html>" in _NO_FLIPS_HTML
    assert "Nothing to inspect" in _NO_FLIPS_HTML
    assert "Content-Security-Policy" in _INSPECT_HEADERS
    # Same strict CSP posture as the report route.
    assert "default-src 'none'" in _INSPECT_HEADERS["Content-Security-Policy"]


# ---------------------------------------------------------------------------
# F3 — jury vote panel rendering
# ---------------------------------------------------------------------------


def _make_run_with_jury_votes(
    tmp_path: Path,
    *,
    canonical_votes: list[dict] | None,
    flipping_votes: list[dict] | None,
    canonical_score: float = 0.8,
    flipping_score: float = 0.2,
) -> Path:
    """Build a synthetic run dir whose judge_results carries votes_json.

    Schema is the F3+ shape (v6) — votes_json is the 9th column.
    """
    run_dir = tmp_path / "synthetic-jury-run"
    run_dir.mkdir()
    manifest = {
        "run_id": "synth-jury-0001",
        "topology_canonical_hash": "h" * 64,
        "topology_name": "synth",
        "tds_version": "0.5.0",
        "tenant_id": "local",
        "n_variants": 2,
        "n_eval_rows": 1,
        "variant_canonical_hashes": [_CANONICAL_HASH, _FLIPPING_HASH],
        "variant_orderings": [["n1", "n2"], ["n2", "n1"]],
        "eval_ids": ["eval_001"],
        "eval_ground_truth": [1],
        "agent_vendors": ["google"],
        "judge_model": "jury_of_3:deepseek+google+google_2",
        "rubric_hash": "r" * 64,
        "semaphore_size": 8,
        "max_cost_usd": 1.0,
        "cli_args": {"fixture": "x.py:build", "eval": "e.csv", "jury_of_3": True},
        "kickoff_ts": 0,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    db = sqlite3.connect(str(run_dir / "run.sqlite"))
    db.execute(
        "CREATE TABLE cells ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, ir_canonical_hash TEXT, "
        "node_outputs_json TEXT, error TEXT, cell_cost_usd REAL)"
    )
    db.execute(
        "CREATE TABLE judge_results ("
        "run_id TEXT, variant_id TEXT, eval_id TEXT, judge_kind TEXT, "
        "score_value REAL, passed INTEGER, rationale TEXT, failure_reason TEXT, "
        "votes_json TEXT)"
    )
    canonical_outputs = [
        _node_output("n1", "underwriter", "Initial analysis: looks healthy."),
        _node_output("n2", "policy_reviewer", "We recommend approving."),
    ]
    flipping_outputs = [
        _node_output("n2", "policy_reviewer", "Pre-review: I see exceptions."),
        _node_output("n1", "underwriter", "We must deny — concerning patterns."),
    ]
    db.executemany(
        "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "synth-jury-0001", _CANONICAL_HASH, "eval_001",
                "ir-canonical-hash", json.dumps(canonical_outputs), None, 0.002,
            ),
            (
                "synth-jury-0001", _FLIPPING_HASH, "eval_001",
                "ir-canonical-hash-2", json.dumps(flipping_outputs), None, 0.002,
            ),
        ],
    )
    db.executemany(
        "INSERT INTO judge_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "synth-jury-0001", _CANONICAL_HASH, "eval_001", "continuous",
                canonical_score, None, "approve consensus", None,
                json.dumps(canonical_votes) if canonical_votes is not None else None,
            ),
            (
                "synth-jury-0001", _FLIPPING_HASH, "eval_001", "continuous",
                flipping_score, None, "deny consensus", None,
                json.dumps(flipping_votes) if flipping_votes is not None else None,
            ),
        ],
    )
    db.commit()
    db.close()
    return run_dir


def test_inspect_renders_jury_panel_when_votes_json_present(tmp_path):
    """Per-vendor votes from JuryJudge persistence path show up as a
    jury panel in the inspect HTML; counts match the score-threshold
    binarisation."""
    canonical_votes = [
        {"vendor": "google", "score": 0.85, "rationale": "Strong approve"},
        {"vendor": "google_2", "score": 0.75, "rationale": "Approve"},
        {"vendor": "deepseek", "score": 0.80, "rationale": "Approve"},
    ]
    flipping_votes = [
        {"vendor": "google", "score": 0.25, "rationale": "Deny — risk"},
        {"vendor": "google_2", "score": 0.30, "rationale": "Deny"},
        # Lone dissenter approving for the flipping ordering.
        {"vendor": "deepseek", "score": 0.65, "rationale": "Marginal approve"},
    ]
    run_dir = _make_run_with_jury_votes(
        tmp_path,
        canonical_votes=canonical_votes,
        flipping_votes=flipping_votes,
    )
    html, _eid = build_inspect_html(run_dir)
    # Panel label + per-vendor rows + APPROVE/DENY counts.
    assert "Jury vote breakdown" in html
    # Vendor names (rendered verbatim inside the .jury-vendor span).
    for vendor in ("google", "google_2", "deepseek"):
        assert vendor in html
    # Canonical side: 3 approve / 0 deny.
    assert "3</strong> approve" in html
    # Flipping side: 1 approve / 2 deny (deepseek dissented).
    assert "1</strong> approve" in html
    assert "2</strong> deny" in html


def test_inspect_omits_jury_panel_on_legacy_single_judge_run(tmp_path):
    """Single-judge runs have NULL votes_json — the panel must be hidden,
    not rendered with empty rows."""
    run_dir = _make_run(tmp_path)  # legacy synthetic-run shape, no votes
    html, _eid = build_inspect_html(run_dir)
    assert "Jury vote breakdown" not in html


def test_load_cell_falls_back_when_votes_json_column_absent(tmp_path):
    """An older run.sqlite (schema v5, no votes_json column) must still
    load cleanly — _load_cell's PRAGMA probe falls back to NULL."""
    from topofuzz.inspect import _load_cell

    run_dir = _make_run(tmp_path)  # uses the legacy 8-column schema
    cell = _load_cell(run_dir, _CANONICAL_HASH, "eval_001")
    assert cell["jury_votes"] is None


def test_load_cell_parses_votes_json_when_present(tmp_path):
    """When votes_json is populated, _load_cell returns the structured
    list; entries with the wrong shape are filtered."""
    from topofuzz.inspect import _load_cell

    votes = [
        {"vendor": "google", "score": 0.7, "rationale": "ok"},
        # Malformed entry — missing score — must be filtered out.
        {"vendor": "broken"},
        {"vendor": "deepseek", "score": 0.4, "rationale": "no"},
    ]
    run_dir = _make_run_with_jury_votes(
        tmp_path, canonical_votes=votes, flipping_votes=None
    )
    cell = _load_cell(run_dir, _CANONICAL_HASH, "eval_001")
    assert cell["jury_votes"] == [
        {"vendor": "google", "score": 0.7, "rationale": "ok"},
        {"vendor": "deepseek", "score": 0.4, "rationale": "no"},
    ]
    # Flipping side intentionally has NULL votes_json — single-judge
    # fallback works alongside jury cells in the same run.
    flipping_cell = _load_cell(run_dir, _FLIPPING_HASH, "eval_001")
    assert flipping_cell["jury_votes"] is None


def test_jury_panel_n1_single_survivor_renders_no_tie(tmp_path):
    """F3 RT m5 audit fix: when 2 of 3 vendors failed, the surviving
    single vote renders without a tie label — its own verdict is the
    majority by construction."""
    # Single-survivor canonical vote that says APPROVE; flipping side
    # has DENY so a flip exists for build_inspect_html to find.
    canonical_votes = [
        {"vendor": "google", "score": 0.85, "rationale": "approve"},
    ]
    flipping_votes = [
        {"vendor": "google", "score": 0.20, "rationale": "deny"},
    ]
    run_dir = _make_run_with_jury_votes(
        tmp_path,
        canonical_votes=canonical_votes,
        flipping_votes=flipping_votes,
    )
    html, _eid = build_inspect_html(run_dir)
    # Canonical N=1 → 1 approve / 0 deny / 1 judge — strict majority, no tie.
    assert "1</strong> approve" in html
    assert "0</strong> deny" in html
    assert "TIE" not in html


def test_jury_panel_tie_renders_explicit_tie_label(tmp_path):
    """F3 Fuzzer M2 audit fix: a deadlocked N=2 (1 approve / 1 deny)
    jury surfaces a TIE → DENY label so the user understands the
    headline pill came from the conservative tie default. The flipping
    side is unanimous APPROVE so find_inspect_pair finds the flip."""
    canonical_votes = [
        {"vendor": "google", "score": 0.85, "rationale": "approve"},
        {"vendor": "deepseek", "score": 0.20, "rationale": "deny"},
    ]
    flipping_votes = [
        {"vendor": "google", "score": 0.85, "rationale": "approve"},
        {"vendor": "deepseek", "score": 0.80, "rationale": "approve"},
    ]
    run_dir = _make_run_with_jury_votes(
        tmp_path,
        canonical_votes=canonical_votes,
        flipping_votes=flipping_votes,
        # Majority scalar matches what JuryJudge.score would persist:
        # canonical tie → 0.0 (DENY); flipping unanimous → 1.0 (APPROVE).
        canonical_score=0.0,
        flipping_score=1.0,
    )
    html, _eid = build_inspect_html(run_dir)
    # Canonical panel shows the TIE explicitly.
    assert "TIE" in html


def test_jury_votes_view_handles_n1_correctly(tmp_path):
    """Unit test on the helper directly — N=1 strict approve produces
    APPROVE majority_label (no tie even though total is small)."""
    from topofuzz.inspect import _jury_votes_view

    # Use markupsafe directly so we don't pull jinja2's escape.
    pytest.importorskip("markupsafe")
    from markupsafe import Markup, escape

    view = _jury_votes_view(
        [{"vendor": "x", "score": 0.7, "rationale": "ok"}],
        threshold=0.5,
        escape=escape,
        Markup=Markup,
    )
    assert view is not None
    assert view["approve_count"] == 1
    assert view["deny_count"] == 0
    assert view["total"] == 1
    assert view["majority_label"] == "APPROVE"


# ---------------------------------------------------------------------------
# Step 20a/F6 follow-up — manifest verdict_label_true/false consumer wiring
# (BACKLOG 2026-05-16: inspect.py consumer not wired)
# ---------------------------------------------------------------------------


# Sentinel for "manifest key omitted entirely" — distinct from None which
# would write a literal `null` value into the JSON. Pass-2 RT MIN-7: declared
# BEFORE _patch_manifest_labels so a top-to-bottom read works.
_MISSING = object()


def _patch_manifest_labels(
    run_dir: Path, *, label_true: object, label_false: object
) -> None:
    """Patch manifest.json with the given verdict_label_true/false pair.

    ``object`` rather than ``str`` so tests can plant XOR half-sets
    (None / missing) and other tamper shapes the producer IR would
    reject but a hand-edited manifest can carry.
    """
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if label_true is not _MISSING:
        m["verdict_label_true"] = label_true
    if label_false is not _MISSING:
        m["verdict_label_false"] = label_false
    manifest_path.write_text(json.dumps(m), encoding="utf-8")


def test_inspect_default_labels_when_manifest_omits_pair(tmp_path):
    """Backcompat: a v0.5 manifest with no verdict_label_* keys still
    renders APPROVE/DENY (lending-domain default). Confirms the new
    helper does not regress the existing lending-only behaviour."""
    run_dir = _make_run(tmp_path)
    html, _eid = build_inspect_html(run_dir)
    assert "APPROVE" in html
    assert "DENY" in html
    assert "ESCALATE" not in html
    assert "DEFER" not in html


def test_inspect_renders_custom_manifest_labels(tmp_path):
    """Adversarial-relay shape: manifest opts into ESCALATE/DEFER and the
    side-by-side flip HTML renders the new pair instead of APPROVE/DENY.
    Both panels (binary judge canonical=pass + flipping=fail) must
    swap."""
    run_dir = _make_run(tmp_path)
    _patch_manifest_labels(
        run_dir, label_true="ESCALATE", label_false="DEFER"
    )
    html, _eid = build_inspect_html(run_dir)
    assert "ESCALATE" in html
    assert "DEFER" in html
    # Producer-side wiring strips the old defaults from manifest when
    # both differ from default — at the HTML level neither lending
    # label should appear in any verdict pill. (Risk-phrase highlighting
    # in agent response text can still contain "approve"/"deny"
    # substrings, so we look for the uppercase pill labels specifically.)
    assert ">APPROVE<" not in html
    assert ">DENY<" not in html


def test_inspect_custom_labels_are_lowercase_normalized(tmp_path):
    """Mirrors publish.py: lowercase manifest input gets upper()'d before
    rendering so the inspect HTML matches the CEO PDF byte-for-byte."""
    run_dir = _make_run(tmp_path)
    _patch_manifest_labels(
        run_dir, label_true="escalate", label_false="defer"
    )
    html, _eid = build_inspect_html(run_dir)
    assert "ESCALATE" in html
    assert "DEFER" in html


def test_inspect_falls_back_to_defaults_on_xor_half_set_true_only(
    tmp_path, caplog
):
    """Red-team check: orchestrator persists the label pair as both-or-
    neither. A tampered manifest with exactly one key set must log + fall
    back to BOTH defaults rather than silently defaulting the unset half
    against an explicit other half.

    This variant covers ``true`` set / ``false`` missing.
    """
    import logging
    run_dir = _make_run(tmp_path)
    _patch_manifest_labels(
        run_dir, label_true="ESCALATE", label_false=_MISSING
    )
    # Pass-1 RT-specialist MIN-1: caplog default capture is propagation-
    # dependent; explicit logger= forces propagation under caplog's
    # handler regardless of test-runner config.
    with caplog.at_level(logging.WARNING, logger="topofuzz.inspect"):
        html, _eid = build_inspect_html(run_dir)
    assert "APPROVE" in html
    assert "DENY" in html
    assert "ESCALATE" not in html
    assert any(
        "exactly one of verdict_label" in rec.message
        for rec in caplog.records
    )


def test_inspect_falls_back_to_defaults_on_xor_half_set_false_only(
    tmp_path, caplog
):
    """Pass-1 RT-specialist MIN-3: bidirectional XOR coverage. A future
    code change that branches asymmetrically on ``has_true`` vs
    ``has_false`` would silently regress one direction; both must be
    tested."""
    import logging
    run_dir = _make_run(tmp_path)
    _patch_manifest_labels(
        run_dir, label_true=_MISSING, label_false="DEFER"
    )
    with caplog.at_level(logging.WARNING, logger="topofuzz.inspect"):
        html, _eid = build_inspect_html(run_dir)
    assert "APPROVE" in html
    assert "DENY" in html
    assert "DEFER" not in html
    assert any(
        "exactly one of verdict_label" in rec.message
        for rec in caplog.records
    )


def test_inspect_falls_back_to_defaults_on_over_twelve_char_label(
    tmp_path, caplog
):
    """Mirrors publish.py: a tampered manifest with a >12-char label
    used to silently slice. Now inspect logs and falls back to both
    defaults so the buyer sees the same default pair rather than a
    half-custom pair."""
    import logging
    run_dir = _make_run(tmp_path)
    _patch_manifest_labels(
        run_dir, label_true="PARTIALLY_APPROVED", label_false="REJECTED"
    )
    with caplog.at_level(logging.WARNING, logger="topofuzz.inspect"):
        html, _eid = build_inspect_html(run_dir)
    assert "APPROVE" in html
    assert "DENY" in html
    assert "PARTIALLY_APPROVED" not in html
    # "REJECTED" is ≤12 chars but the partner label triggers pair-level
    # fallback — so the partner is discarded too.
    assert ">REJECTED<" not in html
    # Pass-1 RT-specialist MIN-4: also assert the WARN message names the
    # *actually-too-long* key. A regression that swaps the key/value in
    # the log args would silently rename to "verdict_label_false" and
    # mislead the operator.
    assert any(
        ">12 chars after strip" in rec.message
        and "verdict_label_true" in rec.message
        for rec in caplog.records
    )


def test_inspect_whitespace_only_label_treated_as_unset(tmp_path, caplog):
    """Pass-1 RT-specialist MIN-5: a manifest with whitespace-only labels
    (e.g. accidental hand-edit leaving spaces) must read as "not set"
    rather than as a non-empty pair. Both `"  "` strings strip to empty
    → falsy `has_*` → defaults fallback with NO half-set XOR warning
    (because neither side is "set" post-strip).
    """
    import logging
    run_dir = _make_run(tmp_path)
    _patch_manifest_labels(
        run_dir, label_true="   ", label_false="\t\t"
    )
    with caplog.at_level(logging.WARNING, logger="topofuzz.inspect"):
        html, _eid = build_inspect_html(run_dir)
    assert "APPROVE" in html
    assert "DENY" in html
    # Neither side is treated as "set" — no XOR warning fires.
    assert not any(
        "exactly one of verdict_label" in rec.message
        for rec in caplog.records
    )


def test_inspect_raises_on_colliding_manifest_labels(tmp_path):
    """Two labels that collapse to the same string post-normalize destroy
    the side-by-side story (the inspect view IS the side-by-side flip
    inspector). Refuse to render — mirrors publish.PublishError on
    collision."""
    run_dir = _make_run(tmp_path)
    _patch_manifest_labels(
        run_dir, label_true="approve", label_false="Approve"
    )
    with pytest.raises(InspectError) as exc_info:
        build_inspect_html(run_dir)
    msg = str(exc_info.value)
    assert "collide after normalization" in msg
    # Raw values surface in the message so the operator can see which
    # input pair collided.
    assert "approve" in msg.lower()


def test_inspect_no_flip_error_wins_over_colliding_labels(tmp_path):
    """Pass-2 RT-specialist MAJ-3 fix: a degenerate run (no flip) is the
    dominant real-world failure; manifest tampering is a corner case.
    When BOTH conditions hold, the operator must see the "No verdict
    flip" message — not the less-actionable collision message — so they
    investigate the fixture rather than the manifest. Enforced by
    calling ``find_inspect_pair`` BEFORE ``_resolve_verdict_labels``."""
    run_dir = _make_run(tmp_path, with_flip=False)
    _patch_manifest_labels(
        run_dir, label_true="approve", label_false="Approve"
    )
    with pytest.raises(InspectError) as exc_info:
        build_inspect_html(run_dir)
    msg = str(exc_info.value)
    assert "No verdict flip" in msg
    assert "collide" not in msg.lower()


def test_inspect_jury_panel_renders_custom_labels(tmp_path):
    """Jury panel rows + majority badge honour the manifest pair so a
    non-lending JuryJudge run reads consistently (per-vendor pills +
    summary all switch to ESCALATE/DEFER)."""
    canonical_votes = [
        {"vendor": "google", "score": 0.85, "rationale": "approve"},
        {"vendor": "deepseek", "score": 0.80, "rationale": "approve"},
        {"vendor": "openai", "score": 0.75, "rationale": "approve"},
    ]
    flipping_votes = [
        {"vendor": "google", "score": 0.25, "rationale": "deny"},
        {"vendor": "deepseek", "score": 0.30, "rationale": "deny"},
        {"vendor": "openai", "score": 0.20, "rationale": "deny"},
    ]
    run_dir = _make_run_with_jury_votes(
        tmp_path,
        canonical_votes=canonical_votes,
        flipping_votes=flipping_votes,
    )
    _patch_manifest_labels(
        run_dir, label_true="ESCALATE", label_false="DEFER"
    )
    html, _eid = build_inspect_html(run_dir)
    # Per-vendor row pills + majority badge both render the new pair.
    assert "ESCALATE" in html
    assert "DEFER" in html
    assert ">APPROVE<" not in html
    assert ">DENY<" not in html


def test_inspect_jury_panel_tie_label_honours_custom_false(tmp_path):
    """The TIE → <label_false> majority message follows the manifest pair
    so a deadlocked jury on a non-lending fixture reads as e.g.
    "TIE → DEFER" not "TIE → DENY"."""
    canonical_votes = [
        {"vendor": "google", "score": 0.85, "rationale": "approve"},
        {"vendor": "deepseek", "score": 0.20, "rationale": "deny"},
    ]
    flipping_votes = [
        {"vendor": "google", "score": 0.85, "rationale": "approve"},
        {"vendor": "deepseek", "score": 0.80, "rationale": "approve"},
    ]
    run_dir = _make_run_with_jury_votes(
        tmp_path,
        canonical_votes=canonical_votes,
        flipping_votes=flipping_votes,
        canonical_score=0.0,
        flipping_score=1.0,
    )
    _patch_manifest_labels(
        run_dir, label_true="ESCALATE", label_false="DEFER"
    )
    html, _eid = build_inspect_html(run_dir)
    assert "TIE → DEFER" in html
    assert "TIE → DENY" not in html
    # Pass-2 RT MIN-10: structural backstop independent of the arrow
    # glyph. If a future template refactor switches to HTML-entity
    # encoding (e.g. `&rarr;` / `&#8594;` / `→`) the substring
    # assertion above breaks even though the user-visible behaviour is
    # correct. The CSS class is template-stable; assert it as well.
    assert "verdict-unknown" in html


def test_resolve_verdict_labels_helper_defaults():
    """Direct unit test on the helper — absent keys produce APPROVE/DENY."""
    from topofuzz.inspect import _resolve_verdict_labels

    assert _resolve_verdict_labels({}) == ("APPROVE", "DENY")


def test_resolve_verdict_labels_helper_custom_pair():
    """Direct unit test on the helper — explicit pair is upper()'d +
    trimmed and returned. Bytes match publish's normalize."""
    from topofuzz.inspect import _resolve_verdict_labels

    assert _resolve_verdict_labels({
        "verdict_label_true": " escalate ",
        "verdict_label_false": "defer",
    }) == ("ESCALATE", "DEFER")


def test_resolve_verdict_labels_helper_rejects_collision():
    """Direct unit test on the helper — collision raises InspectError
    with the raw values in the message."""
    from topofuzz.inspect import _resolve_verdict_labels

    with pytest.raises(InspectError) as exc_info:
        _resolve_verdict_labels({
            "verdict_label_true": "yes",
            "verdict_label_false": "YES",
        })
    assert "collide after normalization" in str(exc_info.value)


def test_verdict_label_helper_threads_custom_labels():
    """_verdict_label echoes the passed label_true / label_false on the
    binary-judge path and the score-fallback path alike."""
    from topofuzz.inspect import _verdict_label

    # Binary path.
    assert _verdict_label(True, None, 0.5, "ESCALATE", "DEFER") == "ESCALATE"
    assert _verdict_label(False, None, 0.5, "ESCALATE", "DEFER") == "DEFER"
    # Score-fallback path keeps the "(score)" suffix and threads the label.
    assert _verdict_label(None, 0.8, 0.5, "ESCALATE", "DEFER") == "ESCALATE (0.80)"
    assert _verdict_label(None, 0.2, 0.5, "ESCALATE", "DEFER") == "DEFER (0.20)"
    # Backcompat: omitting label args falls back to APPROVE/DENY.
    assert _verdict_label(True, None, 0.5) == "APPROVE"
    assert _verdict_label(False, None, 0.5) == "DENY"
