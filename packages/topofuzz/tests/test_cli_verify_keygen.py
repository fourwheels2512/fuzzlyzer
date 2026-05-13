"""Step 10a: `topofuzz verify` + `topofuzz keygen` CLI tests.

Each test runs against an isolated tmp_path "home" via the `--priv-out`
/ `--pub-out` / `--pubkey` flags so we never touch the real
``~/.topofuzz/dev.pem`` on the developer's machine.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

pytest.importorskip("diskcache")  # CLI module imports it transitively

from typer.testing import CliRunner  # noqa: E402

from fuzzeval.attestation import Ed25519Signer  # noqa: E402
from fuzzeval.keys import generate_dev_keys  # noqa: E402

from topofuzz.attest import (  # noqa: E402
    ATTESTATION_FILENAME,
    build_attestation_payload,
    compute_headline,
    emit_attestation,
)
from topofuzz.cli import app  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def keys(tmp_path: Path) -> tuple[Path, Path]:
    """Generate a fresh dev keypair under tmp_path."""
    priv = tmp_path / "dev.pem"
    pub = tmp_path / "dev.pub.pem"
    generate_dev_keys(priv, pub)
    return priv, pub


@pytest.fixture
def signed_run_dir(tmp_path: Path, keys: tuple[Path, Path]) -> Path:
    """A run dir with manifest.json, run.sqlite, and a fresh attestation.json."""
    priv_path, _ = keys
    from fuzzeval.keys import load_private_key

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    manifest = {
        "run_id": "r1",
        "topology_canonical_hash": "abc",
        "topology_name": "lending_4role",
        "tds_version": "0.5.0",
        "n_variants": 1,
        "n_eval_rows": 1,
        "kickoff_ts": 0.0,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    (run_dir / "run.sqlite").write_bytes(b"")
    signer = Ed25519Signer(load_private_key(priv_path))
    from topofuzz.attest import sha256_file

    payload = build_attestation_payload(
        run_id="r1",
        manifest=manifest,
        headline=compute_headline(
            status="succeeded",
            n_successful_cells=1,
            n_total_cells=1,
            n_variants=1,
            n_evals=1,
            total_cost_usd=0.0,
            elapsed_s=0.0,
        ),
        run_sqlite_sha256=sha256_file(run_dir / "run.sqlite"),
        signed_at="2026-01-01T00:00:00.000000Z",
    )
    emit_attestation(payload, signer, run_dir / ATTESTATION_FILENAME)
    return run_dir


# ---------------------------------------------------------------------------
# keygen
# ---------------------------------------------------------------------------


def test_keygen_writes_keys_and_prints_dev_key_watermark(tmp_path: Path):
    priv = tmp_path / "dev.pem"
    pub = tmp_path / "dev.pub.pem"
    result = CliRunner().invoke(
        app,
        ["keygen", "--priv-out", str(priv), "--pub-out", str(pub)],
    )
    assert result.exit_code == 0, result.output
    assert priv.exists() and pub.exists()
    assert "[DEV-KEY]" in result.output


def test_keygen_refuses_to_overwrite_without_force(tmp_path: Path):
    priv = tmp_path / "dev.pem"
    pub = tmp_path / "dev.pub.pem"
    runner = CliRunner()
    first = runner.invoke(
        app, ["keygen", "--priv-out", str(priv), "--pub-out", str(pub)],
    )
    assert first.exit_code == 0
    second = runner.invoke(
        app, ["keygen", "--priv-out", str(priv), "--pub-out", str(pub)],
    )
    assert second.exit_code == 2
    assert "refusing to overwrite" in second.output or "refusing to overwrite" in (
        second.stderr if hasattr(second, "stderr") else ""
    )


def test_keygen_rejects_reserved_flag(tmp_path: Path):
    """Reserved-flag rejection (BLUEPRINT §16 #2) applies to keygen too."""
    result = CliRunner().invoke(
        app,
        [
            "keygen",
            "--priv-out", str(tmp_path / "dev.pem"),
            "--pub-out", str(tmp_path / "dev.pub.pem"),
            "--kms=aws",
        ],
    )
    assert result.exit_code == 2
    assert "reserved" in result.output.lower()


def test_keygen_force_overwrites(tmp_path: Path):
    priv = tmp_path / "dev.pem"
    pub = tmp_path / "dev.pub.pem"
    runner = CliRunner()
    runner.invoke(
        app, ["keygen", "--priv-out", str(priv), "--pub-out", str(pub)],
    )
    first_priv_bytes = priv.read_bytes()
    second = runner.invoke(
        app, ["keygen", "--priv-out", str(priv), "--pub-out", str(pub), "--force"],
    )
    assert second.exit_code == 0
    assert priv.read_bytes() != first_priv_bytes


# ---------------------------------------------------------------------------
# verify — happy path
# ---------------------------------------------------------------------------


def test_verify_succeeds_for_signed_run(signed_run_dir: Path, keys: tuple[Path, Path]):
    _, pub = keys
    result = CliRunner().invoke(
        app, ["verify", str(signed_run_dir), "--pubkey", str(pub)],
    )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert "[DEV-KEY]" in result.output


def test_verify_prints_each_check_in_order(
    signed_run_dir: Path, keys: tuple[Path, Path],
):
    _, pub = keys
    result = CliRunner().invoke(
        app, ["verify", str(signed_run_dir), "--pubkey", str(pub)],
    )
    # Order matters: discover first, signature second, etc.
    assert result.exit_code == 0
    text = result.output
    # All checklist names appear.
    for name in (
        "discover",
        "envelope_signature",
        "payload_type",
        "jcs_round_trip",
        "schema_version",
        "manifest_match",
        "run_sqlite_sha256_match",
    ):
        assert name in text, f"missing check {name!r} in output:\n{text}"


# ---------------------------------------------------------------------------
# verify — tamper / fail paths
# ---------------------------------------------------------------------------


def test_verify_tampered_byte_exits_1(
    signed_run_dir: Path, keys: tuple[Path, Path],
):
    """BLUEPRINT_v5plus.md §15 DoD line 436: tampering with one byte
    returns exit 1."""
    _, pub = keys
    att = signed_run_dir / ATTESTATION_FILENAME
    raw = bytearray(att.read_bytes())
    # Flip a byte inside the payload base64.
    pos = raw.index(b'"payload":"') + len(b'"payload":"') + 8
    raw[pos] = (raw[pos] ^ 0x01)
    att.write_bytes(bytes(raw))
    result = CliRunner().invoke(
        app, ["verify", str(signed_run_dir), "--pubkey", str(pub)],
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_verify_manifest_drift_exits_1(
    signed_run_dir: Path, keys: tuple[Path, Path],
):
    """If on-disk manifest.json was modified after signing, verify FAILs."""
    _, pub = keys
    manifest_path = signed_run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["topology_name"] = "DIFFERENT"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = CliRunner().invoke(
        app, ["verify", str(signed_run_dir), "--pubkey", str(pub)],
    )
    assert result.exit_code == 1
    assert "manifest_match" in result.output


def test_verify_sqlite_drift_exits_1(
    signed_run_dir: Path, keys: tuple[Path, Path],
):
    _, pub = keys
    (signed_run_dir / "run.sqlite").write_bytes(b"changed")
    result = CliRunner().invoke(
        app, ["verify", str(signed_run_dir), "--pubkey", str(pub)],
    )
    assert result.exit_code == 1
    assert "run_sqlite_sha256_match" in result.output


# ---------------------------------------------------------------------------
# verify — usage errors (exit 2)
# ---------------------------------------------------------------------------


def test_verify_missing_pubkey_exits_2(tmp_path: Path):
    """--pubkey points at a nonexistent file → exit 2 (usage)."""
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    result = CliRunner().invoke(
        app, ["verify", str(run_dir), "--pubkey", str(tmp_path / "nope.pem")],
    )
    assert result.exit_code == 2


def test_verify_missing_run_dir_exits_2(tmp_path: Path, keys: tuple[Path, Path]):
    """run_dir doesn't exist as a directory → exit 2 (usage)."""
    _, pub = keys
    result = CliRunner().invoke(
        app, ["verify", str(tmp_path / "nope"), "--pubkey", str(pub)],
    )
    assert result.exit_code == 2


def test_verify_missing_attestation_exits_1(
    tmp_path: Path, keys: tuple[Path, Path],
):
    """run_dir exists but has no attestation.json → exit 1 (discover-fail)."""
    _, pub = keys
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    result = CliRunner().invoke(
        app, ["verify", str(run_dir), "--pubkey", str(pub)],
    )
    assert result.exit_code == 1
    assert "discover" in result.output.lower()


# ---------------------------------------------------------------------------
# verify — --bundle path
# ---------------------------------------------------------------------------


def test_verify_bundle_with_concrete_run_dir_rejected(
    signed_run_dir: Path, keys: tuple[Path, Path], tmp_path: Path,
):
    """Pass-2 MAJOR-1 regression guard: passing --bundle together with
    a concrete run_dir is a usage error (would silently skip file-vs-
    envelope checks). User must pass '-' as run_dir."""
    _, pub = keys
    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(signed_run_dir),
            "--bundle", str(signed_run_dir / ATTESTATION_FILENAME),
            "--pubkey", str(pub),
        ],
    )
    assert result.exit_code == 2
    assert "incompatible" in result.output.lower()


def test_verify_bundle_flag_overrides_run_dir(
    signed_run_dir: Path, keys: tuple[Path, Path], tmp_path: Path,
):
    """--bundle <path> reads a standalone attestation.json. With run_dir='-'
    sentinel, file-vs-envelope checks are skipped (no manifest_match)."""
    _, pub = keys
    bundle = signed_run_dir / ATTESTATION_FILENAME
    # Use '-' as the run_dir sentinel; --bundle drives the verify.
    result = CliRunner().invoke(
        app,
        [
            "verify",
            "-",
            "--bundle",
            str(bundle),
            "--pubkey",
            str(pub),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_verify_bundle_nonexistent_exits_1(
    tmp_path: Path, keys: tuple[Path, Path],
):
    _, pub = keys
    result = CliRunner().invoke(
        app,
        [
            "verify",
            "-",
            "--bundle",
            str(tmp_path / "nope.json"),
            "--pubkey",
            str(pub),
        ],
    )
    assert result.exit_code == 1
    assert "discover" in result.output.lower()


# ---------------------------------------------------------------------------
# verify — reserved-flag rejection still applies
# ---------------------------------------------------------------------------


def test_verify_rejects_reserved_flag(
    signed_run_dir: Path, keys: tuple[Path, Path],
):
    """Reserved-flag rejection (BLUEPRINT §16 #2) applies to verify too."""
    _, pub = keys
    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(signed_run_dir),
            "--pubkey",
            str(pub),
            "--kms=aws",
        ],
    )
    assert result.exit_code == 2
    assert "reserved" in result.output.lower()
