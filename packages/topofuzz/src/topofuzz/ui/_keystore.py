# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Multi-vendor in-browser key storage for the topofuzz UI.

Four vendors are supported:
  * ``google``     — Gemini      (env: ``GEMINI_API_KEY``,    prefix: ``AIza``)
  * ``openai``     — GPT         (env: ``OPENAI_API_KEY``,    prefix: ``sk-``)
  * ``anthropic``  — Claude      (env: ``ANTHROPIC_API_KEY``, prefix: ``sk-ant-``)
  * ``deepseek``   — DeepSeek    (env: ``DEEPSEEK_API_KEY``,  prefix: ``sk-``)

Per-vendor resolution precedence:
  1. ``_runtime[vendor]``         (set by POST /key/{vendor})
  2. vendor's env var
  3. ``~/.topofuzz/keys/{vendor}`` on disk (opt-in)

Google is the only REQUIRED vendor for the default workflows — the
others are optional and only consumed by jury-of-3.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple


_KEY_DIR: Path = Path.home() / ".topofuzz" / "keys"


class VendorSpec(NamedTuple):
    vendor: str          # internal id used in routes + dict keys
    label: str           # what we show the user
    env_var: str         # env var to read
    prefix: str          # required shape prefix
    min_len: int         # minimum plausible length
    judge_model: str     # model to use when this vendor is in a jury
    required: bool       # gates the run; non-required vendors are optional


VENDORS: dict[str, VendorSpec] = {
    "google": VendorSpec(
        vendor="google",
        label="Google (Gemini)",
        env_var="GEMINI_API_KEY",
        prefix="AIza",
        min_len=30,
        judge_model="google/gemini-2.5-flash",
        required=True,
    ),
    "openai": VendorSpec(
        vendor="openai",
        label="OpenAI",
        env_var="OPENAI_API_KEY",
        prefix="sk-",
        min_len=20,
        judge_model="openai/gpt-4o-mini",
        required=False,
    ),
    "anthropic": VendorSpec(
        vendor="anthropic",
        label="Anthropic (Claude)",
        env_var="ANTHROPIC_API_KEY",
        prefix="sk-ant-",
        min_len=30,
        judge_model="anthropic/claude-3-5-haiku-latest",
        required=False,
    ),
    "deepseek": VendorSpec(
        vendor="deepseek",
        label="DeepSeek",
        env_var="DEEPSEEK_API_KEY",
        prefix="sk-",
        min_len=20,
        judge_model="deepseek/deepseek-chat",
        required=False,
    ),
}


_runtime: dict[str, str] = {}  # vendor -> key


def _disk_path(vendor: str) -> Path:
    return _KEY_DIR / vendor


def _read_disk(vendor: str) -> str | None:
    p = _disk_path(vendor)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def get_key(vendor: str) -> str | None:
    """Return the resolved key for ``vendor`` or None."""
    if vendor not in VENDORS:
        return None
    if key := _runtime.get(vendor):
        return key
    env = os.environ.get(VENDORS[vendor].env_var, "").strip()
    if env:
        return env
    disk = _read_disk(vendor)
    if disk:
        _runtime[vendor] = disk
        return disk
    return None


def get_all_keys() -> dict[str, str]:
    """All resolved vendor keys keyed by env-var name. Used to populate subprocess env."""
    out: dict[str, str] = {}
    for vendor, spec in VENDORS.items():
        if key := get_key(vendor):
            out[spec.env_var] = key
    return out


def is_present(vendor: str) -> bool:
    return bool(get_key(vendor))


def required_present() -> bool:
    """The required vendor (Google) is what gates the run-form."""
    return any(is_present(v) for v, spec in VENDORS.items() if spec.required)


def vendors_present() -> list[str]:
    """List of vendor ids that currently have a key. Order matches VENDORS insertion order."""
    return [v for v in VENDORS if is_present(v)]


def key_tail(key: str) -> str:
    return key[-4:] if len(key) >= 4 else "????"


def source(vendor: str) -> str:
    """Where ``vendor``'s active key came from: ``runtime``, ``env``, ``disk``, or ``none``.

    ``runtime`` vs ``disk`` is a small distinction: both live in
    ``_runtime[vendor]`` at read time, but if ``_runtime[vendor]`` exactly
    matches the on-disk file we report ``disk`` so the user knows it
    will survive a restart.
    """
    if vendor not in VENDORS:
        return "none"
    if _runtime.get(vendor):
        disk = _read_disk(vendor)
        return "disk" if disk and _runtime[vendor] == disk else "runtime"
    if os.environ.get(VENDORS[vendor].env_var, "").strip():
        return "env"
    if _disk_path(vendor).exists():
        return "disk"
    return "none"


def validate_shape(vendor: str, key: str) -> tuple[bool, str | None]:
    """Cheap per-vendor shape check. Returns (ok, error_message)."""
    if vendor not in VENDORS:
        return False, f"Unknown vendor: {vendor}"
    spec = VENDORS[vendor]
    key = (key or "").strip()
    if not key:
        return False, "Paste your key in the box first."
    if not key.startswith(spec.prefix):
        return False, f"A {spec.label} key starts with {spec.prefix}. Try copying it again."
    if len(key) < spec.min_len:
        return False, f"That key looks too short for {spec.label}."
    return True, None


def set_key(vendor: str, key: str, *, remember: bool) -> None:
    """Save a key. ``remember`` writes it to disk; otherwise memory-only."""
    if vendor not in VENDORS:
        raise ValueError(f"unknown vendor: {vendor}")
    key = key.strip()
    _runtime[vendor] = key
    disk = _disk_path(vendor)
    if remember:
        disk.parent.mkdir(parents=True, exist_ok=True)
        disk.write_text(key, encoding="utf-8")
        try:
            os.chmod(disk, 0o600)
        except OSError:
            pass
    else:
        try:
            disk.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def jury_trio_for_subprocess() -> list[str] | None:
    """Return exactly 3 vendor model ids if we can field a jury, else None.

    Preference order: ``anthropic > openai > google > deepseek`` (matches
    the CLI's default jury per BLUEPRINT §6 — Anthropic + OpenAI + Gemini).
    DeepSeek only joins when one of the default three is missing.
    """
    priority = ["anthropic", "openai", "google", "deepseek"]
    picked = [v for v in priority if is_present(v)]
    if len(picked) < 3:
        return None
    return [VENDORS[v].judge_model for v in picked[:3]]


def clear_key(vendor: str) -> None:
    """Forget the browser-set key for ``vendor``. Env vars are not touched."""
    if vendor not in VENDORS:
        return
    _runtime.pop(vendor, None)
    try:
        _disk_path(vendor).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


# ---- backwards-compat shims (legacy single-vendor callers) ----------------

def get_gemini_key() -> str | None:
    return get_key("google")


# Old _KEY_FILE pointer kept for tests that monkeypatch it. Points at the
# google slot under the new layout.
_KEY_FILE: Path = _disk_path("google")
