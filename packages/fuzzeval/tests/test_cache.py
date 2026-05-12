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
    with pytest.raises(TypeError, match="JSON-serializable"):
        cache.get({"bad": {1, 2, 3}})  # sets are not JSON
    with pytest.raises(TypeError, match="JSON-serializable"):
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
