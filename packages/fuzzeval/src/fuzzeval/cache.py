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
import json
from pathlib import Path
from typing import TYPE_CHECKING

from fuzzeval.judge import JudgeResult

if TYPE_CHECKING:
    from diskcache import Cache as _Cache


def _canonical_key(identity: dict, tenant_id: str) -> str:
    """Stable order-independent hash of (tenant_id, identity)."""
    blob = json.dumps(
        {"tenant_id": tenant_id, "identity": identity},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
        return self._cache.get(_canonical_key(identity, self._tenant_id))

    def set(self, identity: dict, value: JudgeResult) -> None:
        self._cache.set(_canonical_key(identity, self._tenant_id), value)

    def close(self) -> None:
        self._cache.close()
