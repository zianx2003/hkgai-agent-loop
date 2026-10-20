import asyncio
import json

import pytest

from hkgai_agent import AgentConfig, AgentLoop, ContextConfig, ContextManager
from hkgai_agent.context import validate_history
from hkgai_agent.model import FakeModelProvider, ModelError, ModelProvider
from hkgai_agent.tools import ExecutorConfig, ToolExecutor, ToolRegistry
from hkgai_agent.types import ModelResponse, ToolCall
from conftest import SlowTool


async def test_final_response_and_multiturn_history(executor, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = FakeModelProvider([ModelResponse("Hello"), ModelResponse("Again")])
    agent = AgentLoop(model, executor, system_prompt="Keep instructions")
    first = await agent.run("Hi")
    second = await agent.run("Continue")
    assert first.termination == second.termination == "final"
    assert first.response == "Hello"
    assert second.iterations == 1
    assert [entry.role for entry in agent.history] == ["system", "user", "assistant", "user", "assistant"]
    assert model.requests[1].messages[2].content == "Hello"
    assert model.requests[0].tools[0]["function"]["name"] == "echo"


async def test_tool_ids_results_and_content_are_returned_to_model(executor):
    model = FakeModelProvider([
        ModelResponse("Working", (
            ToolCall("first", "echo", '{"value":"one"}'),
            ToolCall("second", "echo", '{"value":"two"}'),
        )),
        ModelResponse("Done"),
    ])
    agent = AgentLoop(model, executor)
    result = await agent.run("Run both")
    assert result.termination == "final"
    assert result.iterations == 2
    tool_messages = model.requests[1].messages[-2:]
    assert [message.tool_call_id for message in tool_messages] == ["first", "second"]
    assert json.loads(tool_messages[0].content) == {"ok": True, "output": "one"}
    validate_history(agent.history)


async def test_iteration_limit_leaves_complete_tool_history(executor):
    response = ModelResponse(tool_calls=(ToolCall("call", "echo", '{"value":"loop"}'),))
    model = FakeModelProvider([response, response])
    agent = AgentLoop(model, executor, config=AgentConfig(max_iterations=2))
    result = await agent.run("Loop")
    assert result.termination == "max_iterations"
    assert result.iterations == len(model.requests) == 2
    assert len(result.tool_results) == 2
    validate_history(agent.history)


async def test_model_failure_is_reported_gracefully(executor):
    model = FakeModelProvider([ModelError("credential-like-secret")])
    result = await AgentLoop(model, executor).run("Hi")
    assert result.termination == "model_error"
    assert "ModelError" in result.error
    assert "credential-like-secret" not in result.error


async def test_tool_rejection_is_recoverable(executor):
    model = FakeModelProvider([
        ModelResponse(tool_calls=(ToolCall("blocked", "shell", '{"command":"anything"}'),)),
        ModelResponse("That action is unavailable"),
    ])
    result = await AgentLoop(model, executor).run("Try a missing tool")
    assert result.termination == "final"
    assert result.tool_results[0].error_code == "not_allowed"
    assert json.loads(model.requests[1].messages[-1].content)["ok"] is False


async def test_run_deadline_cancels_tools_and_repairs_history():
    tool = SlowTool()
    executor = ToolExecutor(
        ToolRegistry([tool]), allowed_tools=frozenset({"slow"}), config=ExecutorConfig(tool_timeout=5)
    )
    model = FakeModelProvider([
        ModelResponse(tool_calls=(ToolCall("slow-id", "slow", '{"value":"wait"}'),)),
        ModelResponse("Recovered"),
    ])
    agent = AgentLoop(model, executor, config=AgentConfig(run_timeout=0.1))
    result = await agent.run("Wait")
    assert result.termination == "timeout"
    assert tool.started.is_set() and tool.cancelled.is_set()
    assert result.tool_results[0].error_code == "interrupted"
    assert agent.history[-1].tool_call_id == "slow-id"
    validate_history(agent.history)
    assert (await agent.run("Continue")).response == "Recovered"


async def test_external_cancellation_propagates_and_repairs_history():
    tool = SlowTool()
    executor = ToolExecutor(ToolRegistry([tool]), allowed_tools=frozenset({"slow"}))
    model = FakeModelProvider([
        ModelResponse(tool_calls=(ToolCall("slow-id", "slow", '{"value":"wait"}'),)),
    ])
    agent = AgentLoop(model, executor)
    task = asyncio.create_task(agent.run("Wait"))
    await asyncio.wait_for(tool.started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tool.cancelled.is_set()
    validate_history(agent.history)
    assert agent.history[-1].role == "tool"


async def test_model_timeout(executor):
    class HangingModel(ModelProvider):
        async def complete(self, messages, tools=(), *, max_output_tokens=None):
            await asyncio.Event().wait()

    result = await AgentLoop(
        HangingModel(), executor, config=AgentConfig(model_timeout=0.05)
    ).run("Wait")
    assert result.termination == "model_error"
    assert "TimeoutError" in result.error


@pytest.mark.parametrize("response", [
    ModelResponse(),
    ModelResponse("   "),
    ModelResponse(tool_calls=(ToolCall("same", "echo", "{}"), ToolCall("same", "echo", "{}"))),
    ModelResponse(tool_calls=(ToolCall("", "echo", "{}"),)),
])
async def test_invalid_model_responses_are_not_added_to_history(executor, response):
    agent = AgentLoop(FakeModelProvider([response]), executor)
    result = await agent.run("Hi")
    assert result.termination == "invalid_response"
    assert len(agent.history) == 2


async def test_excessive_tool_batch_rejected_before_execution(executor):
    response = ModelResponse(tool_calls=tuple(
        ToolCall(str(index), "echo", '{"value":"hello"}') for index in range(17)
    ))
    result = await AgentLoop(FakeModelProvider([response]), executor).run("Batch")
    assert result.termination == "invalid_response"
    assert not result.tool_results


async def test_irreducible_context_stops_before_model_request(executor):
    model = FakeModelProvider([ModelResponse("Should not run")])
    agent = AgentLoop(
        model, executor,
        context=ContextManager(ContextConfig(max_tokens=512)),
        config=AgentConfig(max_output_tokens=64),
    )
    result = await agent.run("constraint " * 1000)
    assert result.termination == "context_limit"
    assert not model.requests


async def test_compaction_during_a_run_preserves_tool_protocol(executor):
    model = FakeModelProvider([
        ModelResponse("long previous answer " * 100),
        ModelResponse("Summary: keep files read-only; prior review complete."),
        ModelResponse(tool_calls=(ToolCall("echo-id", "echo", '{"value":"safe"}'),)),
        ModelResponse("Done"),
    ])
    agent = AgentLoop(
        model, executor,
        context=ContextManager(ContextConfig(max_tokens=1200, compact_at_ratio=0.5, keep_recent_messages=1)),
        config=AgentConfig(max_output_tokens=64),
        system_prompt="Stay safe",
    )
    assert (await agent.run("First request: read-only")).termination == "final"
    result = await agent.run("Second request: still read-only")
    assert result.termination == "final"
    assert any(message.is_summary for message in agent.history)
    assert not model.requests[1].tools
    validate_history(agent.history)


async def test_simultaneous_runs_are_serialized(executor):
    model = FakeModelProvider([ModelResponse("first answer"), ModelResponse("second answer")])
    agent = AgentLoop(model, executor)
    outcomes = await asyncio.gather(agent.run("one"), agent.run("two"))
    assert [outcome.response for outcome in outcomes] == ["first answer", "second answer"]
    assert [message.content for message in agent.history if message.role == "user"] == ["one", "two"]


@pytest.mark.parametrize("text", ["", " ", None, 1])
async def test_invalid_user_input(executor, text):
    with pytest.raises(ValueError):
        await AgentLoop(FakeModelProvider([]), executor).run(text)
