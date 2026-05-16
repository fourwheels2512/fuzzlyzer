# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""LLMCache + JudgeCache wrapping diskcache.

Two distinct caches so judge-prompt changes don't invalidate agent rollouts
(BLUEPRINT_v5plus.md §7 — "the v4 §2.4 lesson"). `tenant_id` is the Day-1
forward-compat placeholder per §16; v0.5 always uses "local".

Diskcache is imported lazily so `import fuzzeval.cache` works on a
verify-only install. Construction of LLMCache/JudgeCache raises ImportError
with the install hint if the [cache] extra is missing.

**Eviction policy.** Both caches accept ``size_limit_bytes`` (default
``DEFAULT_SIZE_LIMIT_BYTES`` = 4 GiB). Diskcache uses ``least-recently-
stored`` eviction once the limit is crossed; entries written earlier in
a long run can disappear without warning. Step-5 callers driving large
sweeps should size this deliberately and surface the limit in the run
report so users understand a re-run may be a cache miss.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fuzzeval.canonical import canonical_hash
from fuzzeval.judge import BinaryScore, ContinuousScore, JudgeResult

if TYPE_CHECKING:
    from diskcache import Cache as _Cache


# §7 cache-key producer contract — pin one place so step-5+ runners
# cannot drift. Every field consumed below is sourced as follows:
#
# LLMCache identity (5 fields + tenant_id from constructor):
#   ir_canonical_hash       ← Topology.canonical_hash()
#   node_id                 ← Node.id (str)
#   llm_config_hash         ← LLMConfig.canonical_hash()
#   prompt_hash             ← canonical_hash(<rendered_prompt_str>)
#                              — exact bytes the runner sent to the LLM,
#                                AFTER variable substitution. attempt_id
#                                is intentionally OUT (retries reuse).
#   model_served_version    ← Spec text says response.model (the actual
#                              served id, NOT request.model — providers
#                              alias-route). v0.5 deviation: the runner
#                              keys on a caller-supplied resolver's
#                              prediction so the cache can be looked up
#                              BEFORE calling. Default resolver is
#                              identity (correct for non-aliasing
#                              providers like Gemini direct). Anthropic /
#                              multi-region OpenAI users MUST inject a
#                              resolver that returns the served id, OR
#                              accept that the cache key is the request
#                              id (consistent across runs but technically
#                              §7-deviating). See
#                              ``topofuzz.runner`` module docstring,
#                              "Cache served-version contract".
#   tenant_id               ← LLMCache(tenant_id="local") constructor arg
#                              for v0.5. Per-call tenant overrides are
#                              NOT supported; supply the right cache.
#
# JudgeCache identity (4 fields + tenant_id):
#   rubric_hash             ← caller-supplied. v0.5 callers compute once
#                              from the rubric template at run start —
#                              ``fuzzeval.judge.SingleJudge.rubric_hash``
#                              is the canonical producer.
#   prompt_hash             ← Step 7+: canonical_hash(<rendered_judge_prompt>).
#                              The runner passes a ``judge_prompt_renderer``
#                              (typically ``SingleJudge.render_prompt``)
#                              and hashes its output byte-for-byte. Two
#                              cells whose rendered prompts agree share a
#                              JudgeCache row — exact §7 spec text intent.
#                              Step 5/6 fallback (no renderer supplied)
#                              hashes the full eval_row instead so test
#                              callers that pass a Judge but no renderer
#                              still get stable cache keys; this is the
#                              v0.5 step-5 stopgap, preserved for
#                              compatibility but no longer the default.
#   response_hash           ← canonical_hash(<final_agent_output_str>)
#   judge_model_version     ← caller-supplied (e.g.
#                              "google/gemini-2.5-pro@2026-04").
#
# `attempt_id` is omitted so retries do not re-bill. See BLUEPRINT §7.
#
# Routing through fuzzeval.canonical.canonical_hash keeps every hash in
# the chain (Topology, LLMConfig, _canonical_key) on the same JCS+SHA-256
# recipe. A divergence here would silently re-bucket cache identity and
# corrupt the §7 invariant — caught in test_canonical_chain_consistency.


def _canonical_key(identity: dict, tenant_id: str) -> str:
    """Stable hash of (tenant_id, identity) for cache keying.

    Wraps :func:`fuzzeval.canonical.canonical_hash` so the JCS recipe
    matches :meth:`topofuzz.ir.Topology.canonical_hash` and
    :meth:`topofuzz.ir.LLMConfig.canonical_hash`.

    `identity` must hold only JSON-serializable values
    (str/int/float/bool/None, list, dict). Non-serializable values
    (bytes, sets, numpy scalars) surface as a clear TypeError from the
    shared helper.
    """
    return canonical_hash({"tenant_id": tenant_id, "identity": identity})


# Default eviction-cap. 4 GiB is generous enough that a typical
# `lending_4role` × 50-eval sweep (~14 variants × 50 = 700 LLM calls,
# ≤10 KB each ≈ 7 MB) never evicts; user-supplied massive sweeps should
# override via the constructor's ``size_limit_bytes`` arg.
DEFAULT_SIZE_LIMIT_BYTES = 4 * 1024 * 1024 * 1024


def _open(path: Path, *, size_limit_bytes: int) -> _Cache:
    try:
        from diskcache import Cache
    except ImportError as e:
        raise ImportError(
            "fuzzeval.cache requires the [cache] extra: "
            "`pip install fuzzeval[cache]`"
        ) from e
    if size_limit_bytes < 1:
        raise ValueError(
            f"size_limit_bytes must be a positive int; got {size_limit_bytes!r}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return Cache(str(path), size_limit=size_limit_bytes)


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

    def __init__(
        self,
        root: Path,
        tenant_id: str = "local",
        *,
        size_limit_bytes: int = DEFAULT_SIZE_LIMIT_BYTES,
    ) -> None:
        self._tenant_id = tenant_id
        self._cache = _open(Path(root) / "llm", size_limit_bytes=size_limit_bytes)

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

    Identity dict per BLUEPRINT §7 (see contract block above for v0.5
    semantics of each slot):
        rubric_hash, prompt_hash, response_hash, judge_model_version

    Separate from LLMCache so rubric changes don't invalidate agent rollouts.

    **Hash producer for ``prompt_hash``** is the runner's
    ``judge_prompt_renderer`` (step 7+, typically
    :meth:`fuzzeval.judge.SingleJudge.render_prompt`). The legacy step-5
    fallback (hash the full eval_row) is preserved for test callers that
    supply a Judge but no renderer; mixing the two producers in one cache
    directory is supported but produces two distinct cache rows for the
    same (eval_row, agent_output) — clear the cache directory when
    switching producers if cache-warmth matters.
    """

    def __init__(
        self,
        root: Path,
        tenant_id: str = "local",
        *,
        size_limit_bytes: int = DEFAULT_SIZE_LIMIT_BYTES,
    ) -> None:
        self._tenant_id = tenant_id
        self._cache = _open(Path(root) / "judge", size_limit_bytes=size_limit_bytes)

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
