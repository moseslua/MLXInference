from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from mlx_inference.cache.policy import CachePolicy
from mlx_inference.runtime.decode import select_tokens_from_logits


class TokenBudgetExceededError(ValueError):
    pass


@dataclass(slots=True)
class SchedulerRequest:
    request_id: str
    input_ids: list[int]
    max_new_tokens: int
    eos_token_ids: set[int] = field(default_factory=set)
    generated_tokens: list[int] = field(default_factory=list)
    cancelled: bool = False

    def next_input(self) -> list[int]:
        if self.generated_tokens:
            return [self.generated_tokens[-1]]
        return list(self.input_ids)

    def is_complete(self) -> bool:
        if self.cancelled:
            return True
        if len(self.generated_tokens) >= self.max_new_tokens:
            return True
        return bool(self.generated_tokens and self.generated_tokens[-1] in self.eos_token_ids)


@dataclass(slots=True)
class SchedulerResult:
    request_id: str
    generated_tokens: list[int]
    stop_reason: str


class Scheduler:
    def __init__(self, *, max_active: int = 4, token_budget: int = 1024, cache_policy: CachePolicy | None = None) -> None:
        self.max_active = max_active
        self.token_budget = token_budget
        self.cache_policy = cache_policy or CachePolicy()
        self._pending: list[SchedulerRequest] = []
        self.trace: list[dict[str, Any]] = []

    def submit(self, request: SchedulerRequest) -> None:
        if len(request.input_ids) + request.max_new_tokens > self.token_budget:
            raise TokenBudgetExceededError(f"Request {request.request_id} exceeds scheduler token budget")
        self.cache_policy.create_request_cache(request.request_id)
        self.cache_policy.append_tokens(request.request_id, request.input_ids)
        self._pending.append(request)
        self.trace.append({"event": "submit", "request_id": request.request_id, "cache_tokens": list(request.input_ids)})

    def cancel(self, request_id: str) -> None:
        for request in self._pending:
            if request.request_id == request_id:
                request.cancelled = True
                self.trace.append({"event": "cancel", "request_id": request_id})
                return

    def run(self, model: Any) -> list[SchedulerResult]:
        active = self._pending[: self.max_active]
        self._pending = self._pending[self.max_active :]
        completed: list[SchedulerResult] = []

        while active:
            runnable = [request for request in active if not request.is_complete()]
            if not runnable:
                completed.extend(self._result_for(request) for request in active)
                active = []
                continue

            token_slices = [request.next_input() for request in runnable]
            batch = _pad_token_slices(token_slices)
            self.trace.append(
                {
                    "event": "model_forward_batch",
                    "active_requests": [request.request_id for request in runnable],
                    "token_slice": batch,
                }
            )
            logits = model(mx.array(batch, dtype=mx.int32))
            next_tokens = select_tokens_from_logits(logits, temperature=0.0)
            for request, token_id in zip(runnable, next_tokens, strict=True):
                request.generated_tokens.append(token_id)
                self.cache_policy.append_tokens(request.request_id, [token_id])
                self.trace.append(
                    {
                        "event": "next_token",
                        "request_id": request.request_id,
                        "token_id": token_id,
                    }
                )
                self.trace.append(
                    {
                        "event": "cache_update",
                        "request_id": request.request_id,
                        "token_id": token_id,
                        "cache_tokens": list(self.cache_policy.get_request_cache(request.request_id).tokens),
                    }
                )

            still_active: list[SchedulerRequest] = []
            for request in active:
                if request.is_complete():
                    completed.append(self._result_for(request))
                else:
                    still_active.append(request)
            active = still_active

        return completed

    def _result_for(self, request: SchedulerRequest) -> SchedulerResult:
        if request.cancelled:
            stop_reason = "cancelled"
        elif request.generated_tokens and request.generated_tokens[-1] in request.eos_token_ids:
            stop_reason = "eos"
        else:
            stop_reason = "max_tokens"
        return SchedulerResult(
            request_id=request.request_id,
            generated_tokens=list(request.generated_tokens),
            stop_reason=stop_reason,
        )


def _pad_token_slices(token_slices: list[list[int]]) -> list[list[int]]:
    width = max(len(tokens) for tokens in token_slices)
    batch: list[list[int]] = []
    for tokens in token_slices:
        if len(tokens) == width:
            batch.append(tokens)
        else:
            batch.append(tokens + [tokens[-1]] * (width - len(tokens)))
    return batch
