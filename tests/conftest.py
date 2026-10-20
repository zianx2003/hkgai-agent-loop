import asyncio
from typing import Any

import pytest

from hkgai_agent.tools import Tool, ToolExecutor, ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "Echo a supplied value."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict[str, Any]) -> Any:
        return arguments["value"]


class SlowTool(EchoTool):
    name = "slow"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def execute(self, arguments: dict[str, Any]) -> Any:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()


@pytest.fixture
def executor() -> ToolExecutor:
    return ToolExecutor(ToolRegistry([EchoTool()]), allowed_tools=frozenset({"echo"}))
