"""LLMCache + JudgeCache wrapping diskcache.

Two distinct caches so judge-prompt changes don't invalidate agent rollouts
(BLUEPRINT_v5plus.md §7 — "the v4 §2.4 lesson"). `tenant_id` is the Day-1
forward-compat placeholder per §16; v0.5 always uses "local".

Diskcache is imported lazily so `import fuzzeval.cache` works on a
verify-only install. Construction of LLMCache/JudgeCache raises ImportError
with the install hint if the [cache] extra is missing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import rfc8785

from fuzzeval.judge import BinaryScore, ContinuousScore, JudgeResult

if TYPE_CHECKING:
    from diskcache import Cache as _Cache


def _canonical_key(identity: dict, tenant_id: str) -> str:
    """Stable hash of (tenant_id, identity) using RFC 8785 JCS canonicalization.

    JCS (RFC 8785) is required per BLUEPRINT_v5plus.md §16 forward-compat. Using
    `json.dumps(sort_keys=True)` here would silently diverge on float / unicode
    edge cases, causing cache misses between platforms that should hit.

    `identity` must hold only JSON-serializable values (str/int/float/bool/None,
    list, dict). Pass non-serializable values (bytes, sets, numpy scalars) at
    your peril — caller error is surfaced with a clear TypeError.
    """
    try:
        blob = rfc8785.dumps({"tenant_id": tenant_id, "identity": identity})
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Cache identity must be JSON-serializable; got: {e}"
        ) from e
    return hashlib.sha256(blob).hexdigest()


def _open(path: Path) -> _Cache:
    try:
        from diskcache import Cache
    except ImportError as e:
        raise ImportError(
            "fuzzeval.cache requires the [cache] extra: "
            "`pip install fuzzeval[cache]`"
        ) from e
    path.mkdir(parents=True, exist_ok=True)
    return Cache(str(path))


class LLMCache:
    """Cache for raw LLM responses (bytes).

    Identity dict per BLUEPRINT §7 includes:
        ir_canonical_hash, node_id, llm_config_hash, prompt_hash,
        model_served_version
    `attempt_id` is intentionally omitted so retries do not re-bill.

    Trust boundary: `root` is assumed to be a directory the calling process
    controls (typically `./.cache/` or `~/.topofuzz/cache/`). The constructor
    creates `root/llm/` if missing. Passing an attacker-controlled `root` will
    silently create directories there; v0.5 makes this the caller's
    responsibility and v1.0 will tighten this in the multi-tenant path.
    """

    def __init__(self, root: Path, tenant_id: str = "local") -> None:
        self._tenant_id = tenant_id
        self._cache = _open(Path(root) / "llm")

    def get(self, identity: dict) -> bytes | None:
        v = self._cache.get(_canonical_key(identity, self._tenant_id))
        if v is None:
            return None
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        raise TypeError(
            f"LLMCache stored non-bytes value of type {type(v).__name__} "
            f"under key {_canonical_key(identity, self._tenant_id)[:16]}…"
        )

    def set(self, identity: dict, value: bytes) -> None:
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError(
                f"LLMCache values must be bytes; got {type(value).__name__}"
            )
        self._cache.set(_canonical_key(identity, self._tenant_id), bytes(value))

    def close(self) -> None:
        self._cache.close()


class JudgeCache:
    """Cache for judge scores (ContinuousScore | BinaryScore).

    Identity dict per BLUEPRINT §7 includes:
        rubric_hash, prompt_hash, response_hash, judge_model_version
    Separate from LLMCache so rubric changes don't invalidate agent rollouts.
    """

    def __init__(self, root: Path, tenant_id: str = "local") -> None:
        self._tenant_id = tenant_id
        self._cache = _open(Path(root) / "judge")

    def get(self, identity: dict) -> JudgeResult | None:
        v = self._cache.get(_canonical_key(identity, self._tenant_id))
        if v is None:
            return None
        if isinstance(v, (ContinuousScore, BinaryScore)):
            return v
        # Diskcache uses pickle internally. A tampered / corrupted cache file
        # could deserialize to any type. Refuse to round-trip non-JudgeResult
        # objects — pickle-deserialization attack surface stops here.
        raise TypeError(
            f"JudgeCache held unexpected type {type(v).__name__}. "
            f"Cache may be corrupted or tampered; refusing to round-trip."
        )

    def set(self, identity: dict, value: JudgeResult) -> None:
        if not isinstance(value, (ContinuousScore, BinaryScore)):
            raise TypeError(
                f"JudgeCache values must be ContinuousScore or BinaryScore; "
                f"got {type(value).__name__}"
            )
        self._cache.set(_canonical_key(identity, self._tenant_id), value)

    def close(self) -> None:
        self._cache.close()
