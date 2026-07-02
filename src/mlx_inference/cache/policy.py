from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any


class PagedCacheNotImplementedError(NotImplementedError):
    pass


@dataclass(slots=True)
class CacheTaxonomy:
    native_mlx_lm_cache_wrapper: bool = True
    local_policy_abstraction: bool = True
    prompt_cache_persistence: bool = True
    future_block_paged_design: bool = True


@dataclass(slots=True)
class CachePolicyConfig:
    max_tokens: int | None = None
    rotating: bool = False


@dataclass(slots=True)
class RequestCache:
    request_id: str
    native_cache: Any = None
    tokens: list[int] = field(default_factory=list)


class CachePolicy:
    def __init__(self, config: CachePolicyConfig | None = None) -> None:
        self.config = config or CachePolicyConfig()
        self._requests: dict[str, RequestCache] = {}

    def create_request_cache(self, request_id: str, native_cache: Any = None) -> RequestCache:
        cache = RequestCache(request_id=request_id, native_cache=native_cache)
        self._requests[request_id] = cache
        return cache

    def get_request_cache(self, request_id: str) -> RequestCache:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"Unknown request cache: {request_id}") from exc

    def append_tokens(self, request_id: str, token_ids: list[int]) -> None:
        cache = self.get_request_cache(request_id)
        cache.tokens.extend(int(token_id) for token_id in token_ids)
        if self.config.max_tokens is not None and len(cache.tokens) > self.config.max_tokens:
            if not self.config.rotating:
                raise ValueError("Cache token limit exceeded and rotating cache is disabled")
            cache.tokens[:] = cache.tokens[-self.config.max_tokens :]

    def reset_request(self, request_id: str) -> None:
        self.get_request_cache(request_id).tokens.clear()

    def drop_request(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def describe_taxonomy(self) -> CacheTaxonomy:
        return CacheTaxonomy()

    def require_paged_cache(self) -> None:
        raise PagedCacheNotImplementedError(
            "Paged KV cache requires token-index to physical-block mapping and is not implemented in phase 1"
        )

    def save_prompt_cache(self, request_id: str, path: str | Path) -> None:
        cache = self.get_request_cache(request_id)
        Path(path).write_text(json.dumps({"request_id": request_id, "tokens": cache.tokens}, indent=2))

    def load_prompt_cache(self, request_id: str, path: str | Path) -> RequestCache:
        payload = json.loads(Path(path).read_text())
        cache = self.create_request_cache(request_id)
        cache.tokens.extend(int(token_id) for token_id in payload["tokens"])
        return cache
