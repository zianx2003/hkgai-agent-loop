"""Provider-neutral conversation and tool records."""

from dataclasses import dataclass
import json
from typing import Any, Literal


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    is_summary: bool = False

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported message role")
        if self.content is not None and not isinstance(self.content, str):
            raise ValueError("Message content must be text or None")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages can request tools")
        if (self.role == "tool") != bool(self.tool_call_id):
            raise ValueError("Only tool results must have a tool_call_id")
        if self.is_summary and self.role != "system":
            raise ValueError("Summaries must be system messages")

    def to_wire(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [call.to_wire() for call in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def as_message(self) -> Message:
        return Message("assistant", self.content, self.tool_calls)


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.error_code is None

    def to_message(self) -> Message:
        payload = (
            {"ok": True, "output": self.output}
            if self.ok
            else {
                "ok": False,
                "error": {"code": self.error_code, "message": self.error_message},
            }
        )
        return Message(
            "tool",
            json.dumps(payload, ensure_ascii=False, allow_nan=False),
            tool_call_id=self.tool_call_id,
        )
