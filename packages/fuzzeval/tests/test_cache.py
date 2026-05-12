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
    """tenant_id forward-compat (§16) — distinct tenants get distinct entries."""
    local = LLMCache(tmp_path, tenant_id="local")
    other = LLMCache(tmp_path, tenant_id="customer-42")
    identity = {"k": 1}
    local.set(identity, b"local-resp")
    assert other.get(identity) is None
    local.close()
    other.close()
