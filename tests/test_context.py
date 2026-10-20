import json

import pytest

from hkgai_agent.context import ContextConfig, ContextError, ContextManager, validate_history
from hkgai_agent.model import FakeModelProvider, ModelError
from hkgai_agent.types import Message, ModelResponse, ToolCall, ToolResult


def exchange(identifier="call", text="data"):
    return [
        Message("assistant", tool_calls=(ToolCall(identifier, "read_file", '{"path":"a.txt"}'),)),
        ToolResult(identifier, "read_file", output=text).to_message(),
    ]


async def test_below_threshold_does_not_call_model():
    manager = ContextManager()
    model = FakeModelProvider([])
    history = [Message("system", "keep"), Message("user", "hello")]
    result = await manager.compact(history, model)
    assert result.messages == tuple(history)
    assert not result.compacted and not model.requests


async def test_compaction_preserves_system_latest_user_and_complete_recent_exchange():
    system = Message("system", "Never write files")
    current_user = Message("user", "Keep the budget under $10; inspect files")
    history = [system, Message("user", "Old task " * 300), *exchange("old", "output " * 300),
               current_user, Message("assistant", "Thinking"), *exchange("recent")]
    manager = ContextManager(ContextConfig(max_tokens=1400, keep_recent_messages=1))
    model = FakeModelProvider([ModelResponse("Old task completed; budget constraint remains.")])
    result = await manager.compact(history, model)
    assert result.compacted
    assert result.estimated_tokens < manager.estimate_tokens(history)
    assert system in result.messages and current_user in result.messages
    assert result.messages[-2:] == tuple(exchange("recent"))
    assert len([message for message in result.messages if message.is_summary]) == 1
    assert not model.requests[0].tools
    prompt = model.requests[0].messages[0].content
    assert "constraints" in prompt and "unresolved" in prompt
    summarized = json.loads(model.requests[0].messages[1].content)
    assert any(entry.get("tool_call_id") == "old" for entry in summarized)
    assert not any(entry.get("tool_call_id") == "recent" for entry in summarized)
    validate_history(result.messages)


async def test_latest_user_is_pinned_even_when_far_from_recent_messages():
    request = Message("user", "Never modify these files")
    history = [Message("system", "system"), request, Message("assistant", "large " * 900), *exchange()]
    manager = ContextManager(ContextConfig(max_tokens=1000, keep_recent_messages=1))
    result = await manager.compact(history, FakeModelProvider([ModelResponse("Read-only review in progress")]))
    assert result.compacted and request in result.messages
    validate_history(result.messages)


async def test_existing_summary_is_replaced_and_supplied_to_summarizer():
    old_summary = Message("system", "Important previous decision", is_summary=True)
    history = [Message("system", "keep"), old_summary, Message("user", "old " * 1000),
               Message("assistant", "finished"), Message("user", "new task")]
    manager = ContextManager(ContextConfig(max_tokens=1000, keep_recent_messages=1))
    model = FakeModelProvider([ModelResponse("Combined memory including previous decision")])
    result = await manager.compact(history, model)
    assert result.compacted
    assert old_summary not in result.messages
    assert len([message for message in result.messages if message.is_summary]) == 1
    assert "Important previous decision" in model.requests[0].messages[1].content


@pytest.mark.parametrize("history", [
    [Message("tool", "orphan", tool_call_id="missing")],
    [Message("assistant", tool_calls=(ToolCall("missing", "echo", "{}"),))],
    [Message("assistant", tool_calls=(ToolCall("call", "echo", "{}"),)), Message("user", "interleaving")],
    [*exchange(), ToolResult("call", "echo", output="duplicate").to_message()],
])
async def test_unmatched_or_interleaved_exchanges_rejected(history):
    with pytest.raises(ContextError):
        await ContextManager().compact(history, FakeModelProvider([]))


@pytest.mark.parametrize("response", [ModelError("failed"), ModelResponse(""),
    ModelResponse(tool_calls=(ToolCall("unexpected", "echo", "{}"),))])
async def test_summary_failure_does_not_mutate_original_history(response):
    history = [Message("system", "keep"), Message("user", "old " * 1000), Message("user", "new")]
    original = list(history)
    manager = ContextManager(ContextConfig(max_tokens=1000, keep_recent_messages=1))
    with pytest.raises(ContextError):
        await manager.compact(history, FakeModelProvider([response]))
    assert history == original


async def test_no_compaction_when_only_pinned_and_recent_messages_remain():
    history = [Message("system", "system"), Message("user", "large " * 1000)]
    model = FakeModelProvider([])
    result = await ContextManager(ContextConfig(max_tokens=100)).compact(history, model)
    assert not result.compacted and not model.requests


async def test_output_reserve_can_trigger_compaction_before_ratio_threshold():
    history = [Message("system", "system"), Message("user", "old " * 300), Message("user", "new")]
    manager = ContextManager(ContextConfig(max_tokens=1000, compact_at_ratio=1, keep_recent_messages=1))
    assert manager.estimate_tokens(history) < 1000
    result = await manager.compact(history, FakeModelProvider([ModelResponse("Old task")]), reserved_tokens=800)
    assert result.compacted


def test_schema_tokens_are_counted():
    history = [Message("user", "Hi")]
    assert ContextManager.estimate_tokens(history, [{"large": "x" * 400}]) > ContextManager.estimate_tokens(history)
