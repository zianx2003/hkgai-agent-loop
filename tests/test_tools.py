import asyncio
import json
from typing import Any

import pytest

from hkgai_agent.tools import ExecutorConfig, ToolExecutor, ToolRegistry
from hkgai_agent.types import ToolCall
from conftest import EchoTool, SlowTool


def call(identifier, name="echo", value="value"):
    return ToolCall(identifier, name, json.dumps({"value": value}))


async def test_reads_overlap_and_concurrency_is_bounded():
    class ControlledTool(EchoTool):
        def __init__(self):
            self.active = 0
            self.peak = 0
            self.started = 0
            self.two_started = asyncio.Event()
            self.release = asyncio.Event()

        async def execute(self, arguments):
            self.active += 1
            self.started += 1
            self.peak = max(self.peak, self.active)
            if self.started == 2:
                self.two_started.set()
            try:
                await self.release.wait()
                return arguments["value"]
            finally:
                self.active -= 1

    tool = ControlledTool()
    executor = ToolExecutor(
        ToolRegistry([tool]), allowed_tools=frozenset({"echo"}), config=ExecutorConfig(concurrency=2)
    )
    task = asyncio.create_task(executor.execute_batch([call(str(index)) for index in range(5)]))
    await asyncio.wait_for(tool.two_started.wait(), 1)
    assert tool.active == tool.started == 2
    tool.release.set()
    results = await asyncio.wait_for(task, 1)
    assert len(results) == tool.started == 5
    assert all(result.ok for result in results)
    assert tool.peak == 2 and tool.active == 0


async def test_out_of_order_completion_preserves_result_order_and_ids():
    finished_fast = asyncio.Event()

    class OrderedTool(EchoTool):
        async def execute(self, arguments):
            if arguments["value"] == "slow":
                await finished_fast.wait()
            else:
                finished_fast.set()
            return arguments["value"]

    executor = ToolExecutor(ToolRegistry([OrderedTool()]), allowed_tools=frozenset({"echo"}))
    results = await asyncio.wait_for(executor.execute_batch([
        call("slow-id", value="slow"), call("fast-id", value="fast")
    ]), 1)
    assert [result.tool_call_id for result in results] == ["slow-id", "fast-id"]
    assert [result.output for result in results] == ["slow", "fast"]


async def test_writes_exclude_reads_and_other_writes_across_batches():
    active = {"readers": 0, "writers": 0}
    events = []

    class AccessTool(EchoTool):
        def __init__(self, name, read_only):
            self.name = name
            self.read_only = read_only

        async def execute(self, arguments):
            value = arguments["value"]
            assert active["writers"] == 0
            if not self.read_only:
                assert active["readers"] == 0
            counter = "readers" if self.read_only else "writers"
            active[counter] += 1
            events.append(f"start:{value}")
            try:
                await asyncio.sleep(0)
                assert active["writers"] == (0 if self.read_only else 1)
            finally:
                active[counter] -= 1
                events.append(f"end:{value}")
            return value

    executor = ToolExecutor(
        ToolRegistry([AccessTool("read", True), AccessTool("write", False)]),
        allowed_tools=frozenset({"read", "write"}),
    )
    results = await asyncio.gather(
        executor.execute_batch([
            call("a", "read", "a"), call("b", "read", "b"),
            call("c", "write", "c"), call("d", "read", "d"),
        ]),
        executor.execute_batch([call("e", "write", "e"), call("f", "write", "f")]),
    )
    assert all(result.ok for batch in results for result in batch)
    assert events.index("start:c") > max(events.index("end:a"), events.index("end:b"))
    assert events.index("start:d") > events.index("end:c")
    assert active == {"readers": 0, "writers": 0}


async def test_partial_failure_and_per_tool_timeout():
    class BrokenTool(EchoTool):
        name = "broken"

        async def execute(self, arguments):
            raise RuntimeError("sensitive detail")

    slow = SlowTool()
    executor = ToolExecutor(
        ToolRegistry([EchoTool(), BrokenTool(), slow]),
        allowed_tools=frozenset({"echo", "broken", "slow"}),
        config=ExecutorConfig(tool_timeout=1, tool_timeouts={"slow": 0.05}),
    )
    results = await executor.execute_batch([call("ok"), call("bad", "broken"), call("slow", "slow")])
    assert [result.error_code for result in results] == [None, "execution_error", "timeout"]
    assert "sensitive detail" not in results[1].error_message
    assert slow.cancelled.is_set()
    assert (await executor.execute_batch([call("again")]))[0].ok


@pytest.mark.parametrize("arguments", [
    "not json", "[]", "null", '{"value":1}', '{"value":"ok","extra":1}',
    '{"value":"first","value":"second"}', '{"value":NaN}', '{"value":Infinity}',
    '{"value":1e9999}', '{"value":' + "[" * 25 + '"x"' + "]" * 25 + "}",
])
async def test_invalid_model_arguments_are_rejected(executor, arguments):
    result = (await executor.execute_batch([ToolCall("invalid", "echo", arguments)]))[0]
    assert result.error_code == "invalid_arguments"


async def test_allowlist_is_independent_from_registration():
    executor = ToolExecutor(ToolRegistry([EchoTool()]), allowed_tools=frozenset())
    assert executor.schemas == []
    assert (await executor.execute_batch([call("blocked")]))[0].error_code == "not_allowed"


async def test_argument_and_result_size_limits():
    executor = ToolExecutor(
        ToolRegistry([EchoTool()]), allowed_tools=frozenset({"echo"}),
        config=ExecutorConfig(max_argument_chars=32, max_result_chars=10),
    )
    results = await executor.execute_batch([call("large", value="x" * 100), call("output", value="small")])
    assert [result.error_code for result in results] == ["invalid_arguments", "result_too_large"]


async def test_non_json_tool_output_is_a_failure():
    class NonJsonTool(EchoTool):
        async def execute(self, arguments: dict[str, Any]) -> Any:
            return {"value": float("nan")}

    executor = ToolExecutor(ToolRegistry([NonJsonTool()]), allowed_tools=frozenset({"echo"}))
    assert (await executor.execute_batch([call("bad-output")]))[0].error_code == "execution_error"


@pytest.mark.parametrize("calls", [
    [call("duplicate"), call("duplicate")], [call("")], [call(str(index)) for index in range(17)],
])
async def test_invalid_batch_limits(executor, calls):
    with pytest.raises(ValueError):
        await executor.execute_batch(calls)


def test_registry_rejects_duplicate_names_and_invalid_schemas():
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry([EchoTool(), EchoTool()])

    class UnrestrictedTool(EchoTool):
        parameters = {"type": "object"}

    with pytest.raises(ValueError, match="additionalProperties"):
        ToolRegistry([UnrestrictedTool()])


@pytest.mark.parametrize("config", [
    {"concurrency": 0}, {"tool_timeout": 0}, {"tool_timeout": float("nan")},
    {"tool_timeouts": {"echo": -1}}, {"max_batch_size": 0},
])
def test_invalid_executor_configuration(config):
    with pytest.raises(ValueError):
        ExecutorConfig(**config)
