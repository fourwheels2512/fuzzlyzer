# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

import pytest

pytest.importorskip("diskcache")

from fuzzeval.cache import JudgeCache, LLMCache  # noqa: E402
from fuzzeval.judge import BinaryScore, ContinuousScore  # noqa: E402


def test_llm_cache_round_trip(tmp_path):
    cache = LLMCache(tmp_path)
    identity = {"node_id": "intake", "prompt_hash": "abc"}
    assert cache.get(identity) is None
    cache.set(identity, b"hello world")
    assert cache.get(identity) == b"hello world"
    cache.close()


def test_llm_cache_rejects_non_bytes_set(tmp_path):
    cache = LLMCache(tmp_path)
    with pytest.raises(TypeError):
        cache.set({"k": 1}, "not bytes")  # type: ignore[arg-type]
    cache.close()


def test_llm_cache_key_is_order_independent(tmp_path):
    cache = LLMCache(tmp_path)
    cache.set({"a": 1, "b": 2}, b"value")
    assert cache.get({"b": 2, "a": 1}) == b"value"
    cache.close()


def test_judge_cache_round_trip_continuous(tmp_path):
    cache = JudgeCache(tmp_path)
    identity = {"rubric_hash": "r1", "response_hash": "x"}
    cache.set(identity, ContinuousScore(value=0.75))
    got = cache.get(identity)
    assert isinstance(got, ContinuousScore)
    assert got.value == 0.75
    cache.close()


def test_judge_cache_round_trip_binary(tmp_path):
    cache = JudgeCache(tmp_path)
    identity = {"rubric_hash": "r1", "response_hash": "y"}
    cache.set(identity, BinaryScore(passed=True, rationale="meets"))
    got = cache.get(identity)
    assert isinstance(got, BinaryScore)
    assert got.passed
    assert got.rationale == "meets"
    cache.close()


def test_llm_and_judge_caches_are_key_isolated(tmp_path):
    """Same identity dict stored in both caches must not collide.

    This is the BLUEPRINT §7 contract: judge-prompt changes do not
    invalidate agent rollouts and vice versa.
    """
    llm = LLMCache(tmp_path)
    judge = JudgeCache(tmp_path)
    identity = {"k": "shared"}
    llm.set(identity, b"agent-response")
    assert judge.get(identity) is None
    judge.set(identity, ContinuousScore(value=0.5))
    assert llm.get(identity) == b"agent-response"
    llm.close()
    judge.close()


def test_tenant_id_isolates_keys(tmp_path):
    """tenant_id forward-compat (§16) — distinct tenants get distinct entries
    AND distinct values (not just "other tenant's slot is empty").
    """
    local = LLMCache(tmp_path, tenant_id="local")
    other = LLMCache(tmp_path, tenant_id="customer-42")
    identity = {"k": 1}
    local.set(identity, b"local-resp")
    assert other.get(identity) is None
    # Now write a different value under the SAME identity in the other tenant.
    # If keys weren't tenant-scoped, this would clobber the local entry.
    other.set(identity, b"other-resp")
    assert local.get(identity) == b"local-resp"
    assert other.get(identity) == b"other-resp"
    local.close()
    other.close()


def test_canonical_key_rejects_non_serializable_identity(tmp_path):
    """Identity values that aren't JSON-serializable (bytes, sets, etc.)
    must produce a clear TypeError from _canonical_key, not a confusing
    error from deep inside rfc8785.
    """
    cache = LLMCache(tmp_path)
    with pytest.raises(TypeError, match="JSON / JCS serializable"):
        cache.get({"bad": {1, 2, 3}})  # sets are not JSON
    with pytest.raises(TypeError, match="JSON / JCS serializable"):
        cache.get({"bad": b"raw-bytes"})  # bytes are not JSON
    cache.close()


def test_judge_cache_rejects_non_judgeresult_on_set(tmp_path):
    """JudgeCache.set must reject anything that isn't a ContinuousScore or
    BinaryScore. Boundary guard so corrupt values never reach diskcache.
    """
    cache = JudgeCache(tmp_path)
    with pytest.raises(TypeError, match="ContinuousScore or BinaryScore"):
        cache.set({"k": 1}, {"not": "a JudgeResult"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ContinuousScore or BinaryScore"):
        cache.set({"k": 2}, "string-not-score")  # type: ignore[arg-type]
    cache.close()


def test_judge_cache_rejects_corrupted_diskcache_value(tmp_path):
    """If a tampered / corrupted cache file holds a non-JudgeResult value,
    JudgeCache.get must raise TypeError rather than silently round-trip.

    Pickle-deserialization attack surface stops at the JudgeCache boundary.
    """
    from diskcache import Cache

    from fuzzeval.cache import _canonical_key

    identity = {"k": "tampered"}
    key = _canonical_key(identity, "local")

    judge_dir = tmp_path / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    raw = Cache(str(judge_dir))
    raw.set(key, "not a JudgeResult")
    raw.close()

    cache = JudgeCache(tmp_path)
    with pytest.raises(TypeError, match="unexpected type"):
        cache.get(identity)
    cache.close()


def test_judge_cache_tenant_isolation(tmp_path):
    """tenant_id forward-compat (§16) — JudgeCache must isolate tenants too
    (LLMCache already does — symmetry pin so a future refactor doesn't drop
    the contract on one side).
    """
    local = JudgeCache(tmp_path, tenant_id="local")
    other = JudgeCache(tmp_path, tenant_id="customer-42")
    identity = {"rubric_hash": "r1", "response_hash": "x"}
    local.set(identity, ContinuousScore(value=0.9))
    assert other.get(identity) is None
    other.set(identity, ContinuousScore(value=0.1))
    assert local.get(identity).value == 0.9
    assert other.get(identity).value == 0.1
    local.close()
    other.close()


def test_llm_cache_size_limit_bytes_is_stored_on_underlying_cache(tmp_path):
    """RT M5: eviction policy must be EXPLICIT, not silently inherited from
    diskcache's 1GB default. Pin that constructing with size_limit_bytes
    routes the cap through to the underlying ``diskcache.Cache``.

    This test asserts *storage*, not *behaviour* — diskcache's size_limit
    is advisory (soft-cap, culled lazily on writes). We assert the value
    was stored; the eviction docstring at cache.py:13-17 covers the
    behavioural semantics.
    """
    cache = LLMCache(tmp_path, size_limit_bytes=1024 * 1024)  # 1 MB
    # diskcache exposes size_limit as a settable property; this asserts
    # we actually passed our cap (a regression here would silently revert
    # to 1 GB).
    assert cache._cache.size_limit == 1024 * 1024
    cache.close()


def test_cache_rejects_zero_size_limit(tmp_path):
    """Defence against accidental size_limit_bytes=0 — would write nothing
    and read all-miss, silently producing a useless cache."""
    with pytest.raises(ValueError, match="positive"):
        LLMCache(tmp_path, size_limit_bytes=0)
    with pytest.raises(ValueError, match="positive"):
        LLMCache(tmp_path, size_limit_bytes=-1)


def test_canonical_key_collision_smoke_10k_distinct_identities(tmp_path):
    """RT M5: smoke-test that 10,000 distinct identity dicts produce
    10,000 distinct cache keys. SHA-256 makes the birthday probability
    of a collision in 10k draws astronomically low (~ 2^-220) — if this
    test fails, the canonical-hash recipe is corrupted (likely truncation
    or wrong hash function).
    """
    from fuzzeval.cache import _canonical_key

    keys: set[str] = set()
    for i in range(10_000):
        identity = {
            "ir_canonical_hash": f"hash-{i:08d}",
            "node_id": "intake",
            "llm_config_hash": "cfg",
            "prompt_hash": f"p-{i}",
            "model_served_version": "v1",
        }
        keys.add(_canonical_key(identity, "local"))
    assert len(keys) == 10_000


def test_canonical_key_distinguishes_field_swaps(tmp_path):
    """Swapping field VALUES (not keys) between two identities MUST
    produce distinct keys. Otherwise a runner could cache an `intake`
    response under what looks like a `risk` cache slot — silent
    cross-node poisoning.
    """
    from fuzzeval.cache import _canonical_key

    a = {"node_id": "intake", "prompt_hash": "X"}
    b = {"node_id": "X", "prompt_hash": "intake"}  # values swapped
    assert _canonical_key(a, "local") != _canonical_key(b, "local")
