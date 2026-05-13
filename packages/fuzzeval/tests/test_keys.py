"""Ed25519 dev-key management tests.

Tests run against a tmp_path home so we never touch the real
``~/.topofuzz/dev.pem``.
"""

from __future__ import annotations

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
