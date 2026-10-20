"""Validated tools and a bounded executor with read/write barriers."""

from abc import ABC, abstractmethod
import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .safety import SafetyError, Workspace
from .types import ToolCall, ToolResult


class ToolInputError(ValueError):
    """A tool rejected its validated input."""


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]
    read_only: bool = True

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> Any:
        """Return JSON-serializable data; implementations must cooperate with cancellation."""


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        for tool in tools:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", tool.name):
                raise ValueError("Invalid tool name")
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool: {tool.name}")
            parameters = deepcopy(tool.parameters)
            Draft202012Validator.check_schema(parameters)
            if parameters.get("type") != "object" or parameters.get("additionalProperties") is not False:
                raise ValueError("Tool schemas must be objects with additionalProperties=false")
            self._tools[tool.name] = tool
            self._schemas[tool.name] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
            self._validators[tool.name] = Draft202012Validator(parameters)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self, allowed_tools: frozenset[str]) -> list[dict[str, Any]]:
        return [deepcopy(schema) for name, schema in self._schemas.items() if name in allowed_tools]

    def validate(self, name: str, arguments: dict[str, Any]) -> None:
        error = next(self._validators[name].iter_errors(arguments), None)
        if error is not None:
            raise ToolInputError("Arguments do not match the tool's JSON schema")


@dataclass(frozen=True)
class ExecutorConfig:
    concurrency: int = 4
    tool_timeout: float = 5
    tool_timeouts: Mapping[str, float] = field(default_factory=dict)
    max_batch_size: int = 16
    max_argument_chars: int = 16_384
    max_result_chars: int = 32_768

    def __post_init__(self) -> None:
        if min(self.concurrency, self.max_batch_size, self.max_argument_chars, self.max_result_chars) < 1:
            raise ValueError("Executor size limits must be positive")
        if not math.isfinite(self.tool_timeout) or self.tool_timeout <= 0 or any(
            not math.isfinite(timeout) or timeout <= 0 for timeout in self.tool_timeouts.values()
        ):
            raise ValueError("Tool timeouts must be finite and positive")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ToolInputError("Duplicate JSON argument keys are forbidden")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ToolInputError("Nonfinite JSON numbers are forbidden")


def _check_depth(value: Any, depth: int = 0) -> None:
    if depth > 20:
        raise ToolInputError("Arguments exceed the nesting limit")
    if isinstance(value, dict):
        for child in value.values():
            _check_depth(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _check_depth(child, depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ToolInputError("Nonfinite JSON numbers are forbidden")


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowed_tools: frozenset[str],
        config: ExecutorConfig | None = None,
    ) -> None:
        self.registry = registry
        self.allowed_tools = frozenset(allowed_tools)
        self.config = config or ExecutorConfig()
        if any(registry.get(name) is None for name in self.allowed_tools):
            raise ValueError("Allowlist contains an unregistered tool")
        if any(registry.get(name) is None for name in self.config.tool_timeouts):
            raise ValueError("Timeout configuration contains an unregistered tool")
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        self._batch_lock = asyncio.Lock()

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self.registry.schemas(self.allowed_tools)

    async def execute_batch(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        if len(calls) > self.config.max_batch_size:
            raise ValueError("Too many tool calls in one response")
        if len({call.id for call in calls}) != len(calls) or any(not call.id for call in calls):
            raise ValueError("Tool-call IDs must be nonempty and unique within a batch")
        async with self._batch_lock:
            results: list[ToolResult] = []
            pending_reads: list[ToolCall] = []
            for call in calls:
                tool = self.registry.get(call.name)
                if tool is not None and not tool.read_only and call.name in self.allowed_tools:
                    if pending_reads:
                        results.extend(await self._read_batch(pending_reads))
                        pending_reads = []
                    results.append(await self._invoke(call))
                else:
                    pending_reads.append(call)
            if pending_reads:
                results.extend(await self._read_batch(pending_reads))
            return results

    async def _read_batch(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        return list(await asyncio.gather(*(self._invoke(call) for call in calls)))

    async def _invoke(self, call: ToolCall) -> ToolResult:
        tool = self.registry.get(call.name)
        if call.name not in self.allowed_tools or tool is None:
            return self._error(call, "not_allowed", "Tool is not allowlisted")
        try:
            if not isinstance(call.arguments, str) or len(call.arguments) > self.config.max_argument_chars:
                raise ToolInputError("Arguments must be bounded JSON text")
            arguments = json.loads(
                call.arguments, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
            if not isinstance(arguments, dict):
                raise ToolInputError("Arguments must be a JSON object")
            _check_depth(arguments)
            self.registry.validate(call.name, arguments)
        except (ValueError, RecursionError) as error:
            message = str(error) if isinstance(error, ToolInputError) else "Arguments are not valid JSON"
            return self._error(call, "invalid_arguments", message)
        timeout = self.config.tool_timeouts.get(call.name, self.config.tool_timeout)
        try:
            async with self._semaphore:
                async with asyncio.timeout(timeout):
                    output = await tool.execute(arguments)
            result = ToolResult(call.id, call.name, output=output)
            encoded = result.to_message().content or ""
            if len(encoded) > self.config.max_result_chars:
                return self._error(call, "result_too_large", "Tool output exceeds the configured size limit")
            return result
        except TimeoutError:
            return self._error(call, "timeout", "Tool exceeded its execution time limit")
        except SafetyError as error:
            return self._error(call, "unsafe_path", str(error))
        except ToolInputError as error:
            return self._error(call, "invalid_input", str(error))
        except Exception as error:
            return self._error(call, "execution_error", f"Tool failed ({type(error).__name__})")

    @staticmethod
    def _error(call: ToolCall, code: str, message: str) -> ToolResult:
        return ToolResult(call.id, call.name, error_code=code, error_message=message)


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a small UTF-8 text file using a workspace-relative path."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 1024}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Workspace, *, max_bytes: int = 8192) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.workspace = workspace
        self.max_bytes = max_bytes

    async def execute(self, arguments: dict[str, Any]) -> Any:
        path = self.workspace.resolve(arguments["path"])
        if not path.is_file():
            raise ToolInputError("Path must reference an existing regular file")
        with path.open("rb") as stream:
            data = stream.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ToolInputError("File exceeds the read size limit")
        try:
            return {"path": arguments["path"], "text": data.decode("utf-8")}
        except UnicodeDecodeError as error:
            raise ToolInputError("File is not UTF-8 text") from error


class ListFilesTool(Tool):
    name = "list_files"
    description = "List up to 100 entries in a workspace directory; no recursive traversal."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 1024}},
        "additionalProperties": False,
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def execute(self, arguments: dict[str, Any]) -> Any:
        directory = self.workspace.resolve(arguments.get("path", "."), allow_root=True)
        if not directory.is_dir():
            raise ToolInputError("Path must reference an existing directory")
        entries = sorted(islice(directory.iterdir(), 101), key=lambda entry: entry.name)
        return {"entries": [entry.name for entry in entries[:100]], "truncated": len(entries) > 100}


class WriteFileTool(Tool):
    name = "write_file"
    description = "Atomically write a small UTF-8 workspace file. Existing files require overwrite=true."
    read_only = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1, "maxLength": 1024},
            "text": {"type": "string", "maxLength": 8192},
            "overwrite": {"type": "boolean"},
        },
        "required": ["path", "text"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Workspace, *, max_bytes: int = 8192) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.workspace = workspace
        self.max_bytes = max_bytes

    async def execute(self, arguments: dict[str, Any]) -> Any:
        path = self.workspace.resolve(arguments["path"])
        data = arguments["text"].encode("utf-8")
        if len(data) > self.max_bytes:
            raise ToolInputError("Text exceeds the write size limit")
        if not path.parent.is_dir():
            raise ToolInputError("Parent directory must already exist")
        if path.exists() and not arguments.get("overwrite", False):
            raise ToolInputError("File already exists; explicit overwrite=true is required")
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".agent-", delete=False) as stream:
                temporary_path = Path(stream.name)
                stream.write(data)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return {"path": arguments["path"], "bytes_written": len(data)}
