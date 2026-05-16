# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Ed25519 dev-key management for topofuzz attestation.

The v0.5 dev key is an **unencrypted** PEM file under
``~/.topofuzz/dev.pem``. The threat model and the reason age-encryption
is intentionally deferred are documented in SECURITY.md (step 10b),
but the short version: this is a developer-grade key that signs runs
with a ``[DEV-KEY]`` watermark; production-grade signing happens at
v1.0 via Cloud KMS adapters (see BLUEPRINT_v5plus.md §11 line 308 for
the reserved ``--kms`` namespace).

Cross-platform layout:

* POSIX (Linux, macOS, WSL): private key is chmod 0o600 after write.
* Windows: filesystem ACLs are NTFS-driven; we do not attempt to twiddle
  them programmatically. The path lives under the user's profile dir
  which is already user-private by default, but operators in shared
  environments should set explicit ACLs and audit them per
  SECURITY.md. The function emits an informational note in that case.

The "verify" surface of fuzzeval needs only :func:`load_public_key`;
keygen + private-key loading live here for symmetry but are also
verify-path-safe (no numpy / sqlite / langgraph imports).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

_log = logging.getLogger(__name__)

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

__all__ = [
    "DEFAULT_DEV_KEY_DIR_NAME",
    "DEFAULT_PRIV_FILENAME",
    "DEFAULT_PUB_FILENAME",
    "KeyAlreadyExistsError",
    "KeyNotFoundError",
    "default_dev_priv_path",
    "default_dev_pub_path",
    "generate_dev_keys",
    "load_private_key",
    "load_public_key",
]

DEFAULT_DEV_KEY_DIR_NAME: Final[str] = ".topofuzz"
DEFAULT_PRIV_FILENAME: Final[str] = "dev.pem"
DEFAULT_PUB_FILENAME: Final[str] = "dev.pub.pem"


class KeyNotFoundError(FileNotFoundError):
    """The PEM file at the requested path does not exist."""


class KeyAlreadyExistsError(FileExistsError):
    """``generate_dev_keys`` was called without ``force=True`` and one of
    the target files already exists. Refusing to overwrite a private
    key without an explicit confirmation is the cardinal UX rule for
    key-management tools."""


def default_dev_priv_path(home: Path | None = None) -> Path:
    """Return ``<home>/.topofuzz/dev.pem``. ``home`` defaults to
    :func:`pathlib.Path.home` and is overridable for testing."""
    return (home or Path.home()) / DEFAULT_DEV_KEY_DIR_NAME / DEFAULT_PRIV_FILENAME


def default_dev_pub_path(home: Path | None = None) -> Path:
    """Return ``<home>/.topofuzz/dev.pub.pem``."""
    return (home or Path.home()) / DEFAULT_DEV_KEY_DIR_NAME / DEFAULT_PUB_FILENAME


def generate_dev_keys(
    priv_path: Path,
    pub_path: Path,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Generate a fresh Ed25519 keypair and write both PEMs to disk.

    The parent directory is created (mode 0o700 on POSIX). On POSIX the
    private key is chmod 0o600. On Windows no permission changes are
    attempted — see module docstring.

    Returns ``(priv_path, pub_path)`` for caller convenience.

    Raises ``KeyAlreadyExistsError`` if either path exists and
    ``force=False`` — this is the safety interlock against accidentally
    rotating a key that already attests some run dirs. ``force=True``
    overwrites both files atomically (best-effort: the private key is
    written first; if pub-key write fails, the priv-key is left in
    place because re-deriving the pub-key from priv is trivial).
    """
    if not force:
        for p in (priv_path, pub_path):
            if p.exists():
                raise KeyAlreadyExistsError(
                    f"refusing to overwrite existing key at {p} "
                    f"(pass force=True to override)"
                )

    parent = priv_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    # 0o700 on the dir is best-effort: mkdir() with exist_ok ignores
    # mode if the directory already exists, so we chmod explicitly
    # after the fact. POSIX-only — Path.chmod on Windows is a no-op for
    # the unix bits but doesn't raise.
    if os.name == "posix":
        try:
            parent.chmod(0o700)
        except OSError as e:
            # Non-fatal but worth a warning: the user's umask may
            # already be tight, but it might also mean we don't own
            # the dir (e.g. created by sudo). The 0o600 chmod on the
            # private key file itself still applies in _atomic_write_bytes,
            # so the key is protected even if the dir isn't.
            _log.warning(
                "could not chmod %s to 0o700 (%s); "
                "manually verify dir permissions",
                parent,
                e,
            )

    priv = Ed25519PrivateKey.generate()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    # Write atomically: temp-file + os.replace. This avoids leaving a
    # truncated PEM on disk if the process is killed mid-write. When
    # ``force=True`` we also clean up any leftover ``.tmp`` from a prior
    # crashed keygen — without this, the second invocation would hit
    # ``FileExistsError`` on O_EXCL with no obvious recovery path.
    if force:
        for p in (priv_path, pub_path):
            tmp = p.with_suffix(p.suffix + ".tmp")
            if tmp.exists():
                _log.info("removing stale tmp file from prior keygen: %s", tmp)
                try:
                    tmp.unlink()
                except OSError as e:  # pragma: no cover — defensive
                    _log.warning("could not remove %s (%s); continuing", tmp, e)
    _atomic_write_bytes(priv_path, priv_pem, mode=0o600 if os.name == "posix" else None)
    _atomic_write_bytes(pub_path, pub_pem, mode=0o644 if os.name == "posix" else None)

    return priv_path, pub_path


def load_private_key(path: Path) -> Ed25519PrivateKey:
    """Read an unencrypted PKCS#8 PEM and return the Ed25519 private key.

    Raises ``KeyNotFoundError`` if the file does not exist (this is a
    distinct exit reason from "the PEM is malformed") and ``ValueError``
    for shape errors. Note: encrypted PEMs raise ``TypeError`` from
    ``cryptography`` because we pass ``password=None``; we surface that
    as ``ValueError`` with a clearer message.
    """
    if not path.exists():
        raise KeyNotFoundError(
            f"no private key at {path} — run `topofuzz keygen` to create one"
        )
    pem_bytes = path.read_bytes()
    try:
        key = serialization.load_pem_private_key(pem_bytes, password=None)
    except TypeError as e:
        raise ValueError(
            f"private key at {path} appears to be encrypted; v0.5 expects "
            f"an unencrypted PEM (see SECURITY.md)"
        ) from e
    except ValueError as e:
        raise ValueError(f"private key at {path} is malformed: {e}") from e
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(
            f"private key at {path} is not Ed25519 (got {type(key).__name__})"
        )
    return key


def load_public_key(path: Path) -> Ed25519PublicKey:
    """Read a SubjectPublicKeyInfo PEM and return the Ed25519 public key."""
    if not path.exists():
        raise KeyNotFoundError(f"no public key at {path}")
    pem_bytes = path.read_bytes()
    try:
        key = serialization.load_pem_public_key(pem_bytes)
    except ValueError as e:
        raise ValueError(f"public key at {path} is malformed: {e}") from e
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(
            f"public key at {path} is not Ed25519 (got {type(key).__name__})"
        )
    return key


def _atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    mode: int | None,
) -> None:
    """Write ``data`` to ``path`` via temp-file + rename.

    On POSIX, ``mode`` is applied to the temp file before rename so the
    final on-disk file never exists with a wider mode. On Windows,
    ``mode=None`` is passed (NTFS ACLs are not unix-mode-shaped) and
    the operating system inherits parent ACLs.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        # Open with O_EXCL so a race with another writer fails loudly
        # rather than overwriting a half-written tmp file.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_BINARY"):  # Windows
            flags |= os.O_BINARY
        if mode is None:
            fd = os.open(tmp, flags)
        else:
            fd = os.open(tmp, flags, mode)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # fsync isn't supported on all filesystems
                    # (e.g. some network mounts); the rename below
                    # is still atomic on POSIX/NTFS.
                    pass
        except BaseException:
            # Ensure we don't leave the tmp file around if the write
            # itself raises mid-stream (e.g. ENOSPC, KeyboardInterrupt).
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        os.replace(tmp, path)
    except FileExistsError:
        # Race: another process is writing the same tmp. Bail loudly —
        # the user can re-run keygen.
        raise
