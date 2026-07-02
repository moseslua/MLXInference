from __future__ import annotations

from mlx_inference.scheduler.core import Scheduler, SchedulerRequest
from mlx_inference.testing.fake import FakeCausalLM


def test_scheduler_drives_real_model_forward_batches() -> None:
    scheduler = Scheduler(max_active=2, token_budget=8)
    scheduler.submit(SchedulerRequest(request_id="r1", input_ids=[2], max_new_tokens=2))
    scheduler.submit(SchedulerRequest(request_id="r2", input_ids=[5], max_new_tokens=2))
    model = FakeCausalLM(vocab_size=16)

    completed = scheduler.run(model)

    assert [r.request_id for r in completed] == ["r1", "r2"]
    assert completed[0].generated_tokens == [3, 4]
    assert completed[1].generated_tokens == [6, 7]
    assert model.batch_call_count == 2
    assert scheduler.cache_policy.get_request_cache("r1").tokens == [2, 3, 4]
    assert scheduler.cache_policy.get_request_cache("r2").tokens == [5, 6, 7]
    assert scheduler.trace
    forward_events = [event for event in scheduler.trace if event["event"] == "model_forward_batch"]
    cache_events = [event for event in scheduler.trace if event["event"] == "cache_update"]
    assert forward_events[0]["active_requests"] == ["r1", "r2"]
    assert cache_events[-1]["request_id"] == "r2"
    assert cache_events[-1]["cache_tokens"] == [5, 6, 7]
