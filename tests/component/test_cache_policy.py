from __future__ import annotations

import pytest

from mlx_inference.cache.policy import CachePolicy, CachePolicyConfig, PagedCacheNotImplementedError


def test_cache_policy_taxonomy_and_request_isolation() -> None:
    policy = CachePolicy(CachePolicyConfig(max_tokens=3, rotating=True))
    first = policy.create_request_cache("a")
    second = policy.create_request_cache("b")

    policy.append_tokens("a", [1, 2, 3, 4])
    policy.append_tokens("b", [9])

    assert first is policy.get_request_cache("a")
    assert second is policy.get_request_cache("b")
    assert policy.get_request_cache("a").tokens == [2, 3, 4]
    assert policy.get_request_cache("b").tokens == [9]

    taxonomy = policy.describe_taxonomy()
    assert taxonomy.native_mlx_lm_cache_wrapper
    assert taxonomy.local_policy_abstraction
    assert taxonomy.prompt_cache_persistence
    assert taxonomy.future_block_paged_design


def test_paged_cache_claim_is_explicitly_rejected() -> None:
    policy = CachePolicy(CachePolicyConfig())

    with pytest.raises(PagedCacheNotImplementedError):
        policy.require_paged_cache()
