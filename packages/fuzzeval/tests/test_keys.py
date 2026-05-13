"""Ed25519 dev-key management tests.

Tests run against a tmp_path home so we never touch the real
``~/.topofuzz/dev.pem``.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from fuzzeval.keys import (
    DEFAULT_DEV_KEY_DIR_NAME,
    DEFAULT_PRIV_FILENAME,
    DEFAULT_PUB_FILENAME,
    KeyAlreadyExistsError,
    KeyNotFoundError,
    default_dev_priv_path,
    default_dev_pub_path,
    generate_dev_keys,
    load_private_key,
    load_public_key,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_default_paths_use_home_topofuzz_dir(tmp_path: Path):
    priv = default_dev_priv_path(home=tmp_path)
    pub = default_dev_pub_path(home=tmp_path)
    assert priv == tmp_path / DEFAULT_DEV_KEY_DIR_NAME / DEFAULT_PRIV_FILENAME
    assert pub == tmp_path / DEFAULT_DEV_KEY_DIR_NAME / DEFAULT_PUB_FILENAME
    assert priv.name == "dev.pem"
    assert pub.name == "dev.pub.pem"


def test_default_paths_default_to_real_home():
    """Default arg uses Path.home() (verify-only smoke; doesn't write)."""
    assert default_dev_priv_path().parent.name == DEFAULT_DEV_KEY_DIR_NAME
    assert default_dev_pub_path().parent.name == DEFAULT_DEV_KEY_DIR_NAME


# ---------------------------------------------------------------------------
# generate_dev_keys
# ---------------------------------------------------------------------------


def test_generate_dev_keys_writes_both_pems(tmp_path: Path):
    priv = tmp_path / "dev.pem"
    pub = tmp_path / "dev.pub.pem"
    generate_dev_keys(priv, pub)
    assert priv.exists() and pub.exists()
    priv_pem = priv.read_bytes()
    pub_pem = pub.read_bytes()
    assert priv_pem.startswith(b"-----BEGIN PRIVATE KEY-----")
    assert pub_pem.startswith(b"-----BEGIN PUBLIC KEY-----")


def test_generate_dev_keys_keypair_is_consistent(tmp_path: Path):
    """Loading both and re-deriving the public from private should match."""
    priv_path = tmp_path / "dev.pem"
    pub_path = tmp_path / "dev.pub.pem"
    generate_dev_keys(priv_path, pub_path)
    priv = load_private_key(priv_path)
    pub = load_public_key(pub_path)
    derived = priv.public_key()
    from cryptography.hazmat.primitives import serialization

    raw_a = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    raw_b = derived.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    assert raw_a == raw_b


def test_generate_dev_keys_refuses_overwrite_without_force(tmp_path: Path):
    priv = tmp_path / "dev.pem"
    pub = tmp_path / "dev.pub.pem"
    generate_dev_keys(priv, pub)
    with pytest.raises(KeyAlreadyExistsError, match="refusing to overwrite"):
        generate_dev_keys(priv, pub)


def test_generate_dev_keys_force_overwrites(tmp_path: Path):
    priv = tmp_path / "dev.pem"
    pub = tmp_path / "dev.pub.pem"
    generate_dev_keys(priv, pub)
    first_priv_bytes = priv.read_bytes()
    generate_dev_keys(priv, pub, force=True)
    # Fresh key bytes — Ed25519PrivateKey.generate is cryptographically
    # random so collision is impossible.
    assert priv.read_bytes() != first_priv_bytes


def test_generate_dev_keys_creates_parent_directory(tmp_path: Path):
    """``generate_dev_keys`` mkdir(parents=True) the parent dir."""
    nested = tmp_path / "nested" / "deep" / "dir"
    priv = nested / "dev.pem"
    pub = nested / "dev.pub.pem"
    generate_dev_keys(priv, pub)
    assert priv.exists() and pub.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only chmod check")
def test_generate_dev_keys_chmods_private_to_0600_on_posix(tmp_path: Path):
    priv = tmp_path / "dev.pem"
    pub = tmp_path / "dev.pub.pem"
    generate_dev_keys(priv, pub)
    priv_mode = stat.S_IMODE(priv.stat().st_mode)
    assert priv_mode == 0o600, f"expected 0o600 got {oct(priv_mode)}"


# ---------------------------------------------------------------------------
# load_private_key / load_public_key
# ---------------------------------------------------------------------------


def test_load_private_key_returns_ed25519(tmp_path: Path):
    priv_path = tmp_path / "dev.pem"
    pub_path = tmp_path / "dev.pub.pem"
    generate_dev_keys(priv_path, pub_path)
    key = load_private_key(priv_path)
    assert isinstance(key, Ed25519PrivateKey)


def test_load_public_key_returns_ed25519(tmp_path: Path):
    priv_path = tmp_path / "dev.pem"
    pub_path = tmp_path / "dev.pub.pem"
    generate_dev_keys(priv_path, pub_path)
    key = load_public_key(pub_path)
    assert isinstance(key, Ed25519PublicKey)


def test_load_private_key_missing_raises_key_not_found(tmp_path: Path):
    with pytest.raises(KeyNotFoundError, match="no private key"):
        load_private_key(tmp_path / "nope.pem")


def test_load_public_key_missing_raises_key_not_found(tmp_path: Path):
    with pytest.raises(KeyNotFoundError, match="no public key"):
        load_public_key(tmp_path / "nope.pub.pem")


def test_load_private_key_rejects_malformed_pem(tmp_path: Path):
    bad = tmp_path / "bad.pem"
    bad.write_bytes(b"not a PEM")
    with pytest.raises(ValueError, match="malformed"):
        load_private_key(bad)


def test_load_public_key_rejects_malformed_pem(tmp_path: Path):
    bad = tmp_path / "bad.pub.pem"
    bad.write_bytes(b"not a PEM")
    with pytest.raises(ValueError, match="malformed"):
        load_public_key(bad)


def test_load_private_key_rejects_non_ed25519(tmp_path: Path):
    """RSA keys are valid PKCS#8 but not Ed25519 — refuse loudly."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = rsa_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    bad = tmp_path / "rsa.pem"
    bad.write_bytes(pem)
    with pytest.raises(ValueError, match="not Ed25519"):
        load_private_key(bad)


def test_load_private_key_rejects_encrypted_pem(tmp_path: Path):
    """v0.5 expects unencrypted PEM; encrypted PEM gets a clear message."""
    from cryptography.hazmat.primitives import serialization

    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.BestAvailableEncryption(b"hunter2"),
    )
    enc = tmp_path / "enc.pem"
    enc.write_bytes(pem)
    with pytest.raises(ValueError, match="encrypted"):
        load_private_key(enc)


# ---------------------------------------------------------------------------
# Atomic write contract
# ---------------------------------------------------------------------------


def test_generate_dev_keys_leaves_no_temp_files_on_success(tmp_path: Path):
    priv = tmp_path / "dev.pem"
    pub = tmp_path / "dev.pub.pem"
    generate_dev_keys(priv, pub)
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Cross-platform path expansion (§12a gate)
#
# These tests pin platform-correct behavior of ``default_dev_priv_path`` /
# ``default_dev_pub_path`` so that the dev-key dir resolves correctly under
# all three of the OSes the v0.5 wheel targets (Windows, macOS, Linux).
#
# The home-env-var coverage matters because:
#   * Windows ``Path.home()`` reads ``USERPROFILE`` (with HOMEDRIVE+HOMEPATH
#     as a fallback). A regression that hard-coded ``HOME`` would silently
#     resolve to the wrong dir on Windows.
#   * POSIX ``Path.home()`` reads ``HOME`` (with the pwd database as
#     fallback).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific USERPROFILE path expansion")
def test_default_dev_paths_use_userprofile_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Path.home() on Windows resolves ~ via USERPROFILE; the dev-key
    paths must honor that env var when ``home=None``."""
    fake_home = tmp_path / "Some User"
    fake_home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    # Clear HOMEDRIVE+HOMEPATH and HOME so a stale test env can't make
    # this test pass for the wrong reason. CPython's Path.home() on
    # Windows currently tries USERPROFILE first, but os.path.expanduser
    # checks HOME first if set — keep this test resilient to either
    # implementation strategy.
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    monkeypatch.delenv("HOME", raising=False)
    assert default_dev_priv_path() == (
        fake_home / DEFAULT_DEV_KEY_DIR_NAME / DEFAULT_PRIV_FILENAME
    )
    assert default_dev_pub_path() == (
        fake_home / DEFAULT_DEV_KEY_DIR_NAME / DEFAULT_PUB_FILENAME
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX-specific HOME path expansion")
def test_default_dev_paths_use_home_on_posix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Path.home() on POSIX resolves ~ via HOME; the dev-key paths must
    honor that env var when ``home=None``."""
    fake_home = tmp_path / "some user"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    assert default_dev_priv_path() == (
        fake_home / DEFAULT_DEV_KEY_DIR_NAME / DEFAULT_PRIV_FILENAME
    )
    assert default_dev_pub_path() == (
        fake_home / DEFAULT_DEV_KEY_DIR_NAME / DEFAULT_PUB_FILENAME
    )


def test_default_paths_handle_home_with_spaces(tmp_path: Path):
    """Home dirs containing spaces (common on Windows: ``C:\\Users\\Some User``)
    must not break path joining or downstream keygen."""
    spaced = tmp_path / "Path With Spaces"
    spaced.mkdir()
    priv = default_dev_priv_path(home=spaced)
    pub = default_dev_pub_path(home=spaced)
    assert priv.parent.parent == spaced
    assert " " in str(priv)
    # Round-trip the spaced path through generate_dev_keys to flush out
    # any subprocess-level quoting bugs in _atomic_write_bytes (none
    # expected — it uses pathlib + os primitives — but worth pinning).
    generate_dev_keys(priv, pub)
    assert priv.exists() and pub.exists()


def test_default_paths_handle_home_with_non_ascii(tmp_path: Path):
    """Non-ASCII home dir names round-trip through path operations.

    Windows + macOS both support Unicode filenames natively; Linux
    filesystems treat names as opaque bytes (UTF-8 by convention). The
    Path object stores the str unchanged; this test exists to catch a
    regression where someone reintroduces ``str.encode('ascii')`` on
    the path.
    """
    # "house" in Greek — covers the BMP path. The 4-byte CJK case is
    # already exercised by the UTF-8 PAE test; we don't need a second
    # filesystem-level case for it here.
    nonascii = tmp_path / "Σπίτι"
    # Skip ONLY on encoding-failure errnos (e.g. EILSEQ / EINVAL caused
    # by a runner with LANG=C against a UTF-8 filename). Other OSErrors
    # (EACCES, ENOSPC, ENAMETOOLONG, ...) propagate as real test
    # failures — the previous broad-except would mask those as "runner
    # limitation" and hide regressions.
    encoding_errnos = {
        getattr(errno, name, -1) for name in ("EILSEQ", "EINVAL")
    } - {-1}
    try:
        nonascii.mkdir()
    except OSError as e:
        if e.errno in encoding_errnos:
            pytest.skip(
                f"filesystem locale cannot encode the non-ASCII name "
                f"(errno={e.errno}); this is a CI-runner locale issue"
            )
        raise
    priv = default_dev_priv_path(home=nonascii)
    pub = default_dev_pub_path(home=nonascii)
    assert priv.parent.parent == nonascii
    generate_dev_keys(priv, pub)
    assert priv.exists() and pub.exists()


def test_default_paths_use_native_separator(tmp_path: Path):
    """Path components are joined with the platform-native separator.

    On Windows ``str(path)`` renders with backslash; on POSIX with
    forward slash. The ``.topofuzz`` segment must appear as a
    separator-delimited component, not as a substring of a longer name.
    """
    priv = default_dev_priv_path(home=tmp_path)
    pub = default_dev_pub_path(home=tmp_path)
    needle = f"{os.sep}{DEFAULT_DEV_KEY_DIR_NAME}{os.sep}"
    assert needle in str(priv)
    assert needle in str(pub)


def test_default_paths_parent_dir_name_is_topofuzz_constant(tmp_path: Path):
    """The parent dir name is exactly the ``.topofuzz`` constant.

    Guards against silent constant renames that would orphan existing
    user keys (someone's already-generated ``~/.topofuzz/dev.pem`` would
    be invisible to a renamed key loader).
    """
    priv = default_dev_priv_path(home=tmp_path)
    pub = default_dev_pub_path(home=tmp_path)
    assert priv.parent.name == ".topofuzz"
    assert pub.parent.name == ".topofuzz"
    # And the leaf filenames are the documented constants — a rename
    # here would orphan existing pubkey files even within an unchanged
    # dir.
    assert priv.name == DEFAULT_PRIV_FILENAME == "dev.pem"
    assert pub.name == DEFAULT_PUB_FILENAME == "dev.pub.pem"
