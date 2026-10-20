"""Observable acceptance demos, with live API execution explicitly selected."""

import argparse
import asyncio
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Sequence

from .agent import AgentConfig, AgentLoop
from .context import ContextConfig, ContextManager, validate_history
from .model import FakeModelProvider, ModelProvider, OpenAICompatibleProvider
from .safety import Workspace
from .skills import LoadSkillTool, SkillRegistry
from .tools import ExecutorConfig, ListFilesTool, ReadFileTool, Tool, ToolExecutor, ToolRegistry, WriteFileTool
from .types import Message, ModelResponse, ToolCall, ToolResult


REVIEW_PROMPT = (
    "Use the code_review skill to review review_sample.py. First load the skill using "
    "load_skill, then read the file using read_file before answering. Report concrete "
    "issues with severity, filename:line, triggering input, impact, and proposed fixes. "
    "Do not execute or modify the code. Do not claim tests were run."
)


def _snapshot(messages: Sequence[Message]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        record = message.to_wire()
        content = message.content or ""
        record["is_summary"] = message.is_summary
        record["content_chars"] = len(content)
        record["content_truncated"] = len(content) > 240 and not message.is_summary
        if record["content_truncated"]:
            record["content"] = content[:240] + "..."
        result.append(record)
    return result


def _tool_records(results: Sequence[ToolResult]) -> list[dict[str, Any]]:
    return [
        {"id": result.tool_call_id, "name": result.name, **json.loads(result.to_message().content or "{}")}
        for result in results
    ]


def _read_target_succeeded(result: ToolResult, workspace: Workspace, target: Path) -> bool:
    if result.name != "read_file" or not result.ok or not isinstance(result.output, dict):
        return False
    try:
        return workspace.resolve(result.output.get("path", "")) == target
    except ValueError:
        return False


async def compression_demo() -> dict[str, Any]:
    system = Message("system", "Use only read-only tools. Never execute reviewed code.")
    latest_request = Message("user", "Continue reviewing review_sample.py; budget <=10; do not modify files.")
    recent_exchange = (
        Message("assistant", tool_calls=(ToolCall("recent-list", "list_files", "{}"),)),
        ToolResult("recent-list", "list_files", output={"entries": ["review_sample.py"]}).to_message(),
    )
    history = [
        system,
        Message("user", "Review review_sample.py; budget <=10; do not modify files."),
        Message("assistant", tool_calls=(ToolCall("old-read", "read_file", '{"path":"review_sample.py"}'),)),
        ToolResult("old-read", "read_file", output={"text": "Earlier source evidence. " * 100}).to_message(),
        Message("assistant", "File read completed; review is pending validation. " * 30),
        latest_request,
        *recent_exchange,
        Message("assistant", "Next: report findings with filename and line numbers."),
    ]
    summary_text = (
        "Task: review review_sample.py. Constraints: budget <=10; do not modify files; "
        "use read-only tools. The file read completed. Review is pending validation."
    )
    model = FakeModelProvider([ModelResponse(summary_text)])
    manager = ContextManager(ContextConfig(
        max_tokens=2048, compact_at_ratio=0.35, keep_recent_messages=4, summary_tokens=128
    ))
    validate_history(history)
    result = await manager.compact(history, model)
    validate_history(result.messages)
    summaries = [message.content or "" for message in result.messages if message.is_summary]
    memory = "\n".join(summaries)
    checks = {
        "compaction_triggered": result.compacted,
        "token_estimate_reduced": result.estimated_tokens < manager.estimate_tokens(history),
        "original_system_preserved": system in result.messages,
        "latest_user_request_preserved_verbatim": latest_request in result.messages,
        "recent_tool_exchange_preserved": tuple(result.messages[-3:-1]) == recent_exchange,
        "important_task_state_preserved": all(
            marker in memory for marker in ("review_sample.py", "budget <=10", "do not modify files", "pending validation")
        ),
        "tool_protocol_valid": True,
    }
    return {
        "demo": "compression",
        "source": "scripted fake-model summary; not a measurement of real-model summary quality",
        "config": asdict(manager.config),
        "before": {"estimated_tokens": manager.estimate_tokens(history), "messages": _snapshot(history)},
        "after": {"estimated_tokens": result.estimated_tokens, "messages": _snapshot(result.messages)},
        "checks": checks,
        "passed": all(checks.values()),
    }


class DelayedValueTool(Tool):
    description = "Return a fixed lookup value after a configured asynchronous delay."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, name: str, value: str, delay_seconds: float, origin: float) -> None:
        if not math.isfinite(delay_seconds) or not 0 < delay_seconds <= 5:
            raise ValueError("Demo delays must be finite and in (0, 5] seconds")
        self.name = name
        self.value = value
        self.delay_seconds = delay_seconds
        self.origin = origin

    async def execute(self, arguments: dict[str, Any]) -> Any:
        started = perf_counter() - self.origin
        await asyncio.sleep(self.delay_seconds)
        return {
            "value": self.value,
            "delay_seconds": self.delay_seconds,
            "started_seconds": started,
            "finished_seconds": perf_counter() - self.origin,
        }


def _overlap(results: Sequence[ToolResult]) -> float:
    if len(results) != 2 or not all(result.ok for result in results):
        return 0
    first, second = (result.output for result in results)
    return max(0, min(first["finished_seconds"], second["finished_seconds"])
               - max(first["started_seconds"], second["started_seconds"]))


async def _timed_batch(concurrency: int, delay_seconds: float) -> tuple[list[ToolResult], dict[str, Any]]:
    origin = perf_counter()
    tools = [
        DelayedValueTool("slow_lookup", "notes", delay_seconds * 1.5, origin),
        DelayedValueTool("fast_lookup", "constraints", delay_seconds, origin),
    ]
    executor = ToolExecutor(
        ToolRegistry(tools), allowed_tools=frozenset(tool.name for tool in tools),
        config=ExecutorConfig(concurrency=concurrency, tool_timeout=10),
    )
    started = perf_counter()
    results = await executor.execute_batch([
        ToolCall("slow-id", "slow_lookup", "{}"), ToolCall("fast-id", "fast_lookup", "{}")
    ])
    return results, {
        "concurrency": concurrency,
        "elapsed_seconds": perf_counter() - started,
        "overlap_seconds": _overlap(results),
        "results": _tool_records(results),
    }


async def parallel_demo(delay_seconds: float = 0.2) -> dict[str, Any]:
    serial_results, serial = await _timed_batch(1, delay_seconds)
    parallel_results, parallel = await _timed_batch(2, delay_seconds)
    checks = {
        "all_tools_succeeded": all(result.ok for result in [*serial_results, *parallel_results]),
        "serial_execution_has_no_overlap": serial["overlap_seconds"] == 0,
        "parallel_execution_has_overlap": parallel["overlap_seconds"] > 0,
        "call_ids_preserved": all(
            [result.tool_call_id for result in results] == ["slow-id", "fast-id"]
            for results in (serial_results, parallel_results)
        ),
        "values_match_call_ids": all(
            {result.tool_call_id: result.output.get("value") if result.ok else None for result in results}
            == {"slow-id": "notes", "fast-id": "constraints"}
            for results in (serial_results, parallel_results)
        ),
    }
    return {
        "demo": "parallel",
        "serial": serial,
        "parallel": parallel,
        "observed_speedup": serial["elapsed_seconds"] / parallel["elapsed_seconds"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def _review_components(workspace_path: Path, skills_path: Path) -> tuple[Workspace, SkillRegistry, ToolExecutor]:
    workspace = Workspace(workspace_path)
    skills = SkillRegistry(skills_path)
    skills.discover()
    skills.get("code_review")
    tools = [ReadFileTool(workspace), ListFilesTool(workspace), LoadSkillTool(skills), WriteFileTool(workspace)]
    executor = ToolExecutor(
        ToolRegistry(tools), allowed_tools=frozenset({"read_file", "list_files", "load_skill"})
    )
    return workspace, skills, executor


async def code_review_demo(workspace_path: Path, skills_path: Path) -> dict[str, Any]:
    workspace, skills, executor = _review_components(workspace_path, skills_path)
    sample_path = workspace.resolve("review_sample.py")
    source_before = sample_path.read_bytes()
    model = FakeModelProvider([
        ModelResponse(tool_calls=(ToolCall("review-skill", "load_skill", '{"name":"code_review"}'),)),
        ModelResponse(tool_calls=(
            ToolCall("review-source", "read_file", '{"path":"review_sample.py"}'),
            ToolCall("outside-path", "read_file", '{"path":"../outside.txt"}'),
            ToolCall("blocked-write", "write_file", '{"path":"review_sample.py","text":"replace","overwrite":true}'),
            ToolCall("blocked-shell", "shell", '{"command":"echo demo"}'),
        )),
        ModelResponse(
            "Scripted example findings (not model-generated):\n"
            "[high] review_sample.py:5: average([]) raises ZeroDivisionError; reject an empty input explicitly.\n"
            "[medium] review_sample.py:8: add_label uses a shared default list; repeated calls leak labels. "
            "Use None and create a new list per call. These are static observations; no code was executed."
        ),
    ])
    agent = AgentLoop(model, executor, skills=skills)
    outcome = await agent.run(REVIEW_PROMPT)
    records = {result.tool_call_id: result for result in outcome.tool_results}
    skill_result = records.get("review-skill")
    source_result = records.get("review-source")
    checks = {
        "final_response_received": outcome.termination == "final",
        "code_review_skill_loaded": bool(skill_result and skill_result.ok and skill_result.output["name"] == "code_review"),
        "source_read": bool(source_result and _read_target_succeeded(source_result, workspace, sample_path)),
        "outside_path_rejected": bool(records.get("outside-path") and records["outside-path"].error_code == "unsafe_path"),
        "write_rejected": bool(records.get("blocked-write") and records["blocked-write"].error_code == "not_allowed"),
        "shell_rejected": bool(records.get("blocked-shell") and records["blocked-shell"].error_code == "not_allowed"),
        "source_unchanged": sample_path.read_bytes() == source_before,
    }
    validate_history(agent.history)
    return {
        "demo": "code-review",
        "source": "scripted fake model; tool results come from the real executor",
        "termination": outcome.termination,
        "iterations": outcome.iterations,
        "results": _tool_records(outcome.tool_results),
        "response": outcome.response,
        "checks": checks,
        "passed": all(checks.values()),
    }


async def live_demo(model: ModelProvider, workspace_path: Path, skills_path: Path) -> dict[str, Any]:
    workspace, skills, executor = _review_components(workspace_path, skills_path)
    sample_path = workspace.resolve("review_sample.py")
    source_before = sample_path.read_bytes()
    agent = AgentLoop(model, executor, skills=skills, config=AgentConfig(run_timeout=90))
    outcome = await agent.run(REVIEW_PROMPT)
    checks = {
        "final_response_received": outcome.termination == "final",
        "code_review_skill_loaded": any(
            result.name == "load_skill" and result.ok and isinstance(result.output, dict)
            and result.output.get("name") == "code_review" for result in outcome.tool_results
        ),
        "source_read": any(
            _read_target_succeeded(result, workspace, sample_path) for result in outcome.tool_results
        ),
        "all_tool_calls_succeeded": bool(outcome.tool_results) and all(result.ok for result in outcome.tool_results),
        "source_unchanged": sample_path.read_bytes() == source_before,
    }
    validate_history(agent.history)
    return {
        "demo": "live",
        "provider": type(model).__name__,
        "termination": outcome.termination,
        "iterations": outcome.iterations,
        "results": _tool_records(outcome.tool_results),
        "response": outcome.response,
        "checks": checks,
        "passed": all(checks.values()),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run observable acceptance demos; 'all' is entirely offline.")
    result.add_argument("demo", choices=("compression", "parallel", "code-review", "live", "all"))
    result.add_argument("--workspace", type=Path, default=Path("examples/workspace"))
    result.add_argument("--skills", type=Path, default=Path("skills"))
    result.add_argument("--delay", type=float, default=0.2, help="Fast-tool delay in seconds; slow tool takes 1.5 times as long.")
    result.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    result.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    result.add_argument("--token-limit-field", choices=("max_completion_tokens", "max_tokens"), default="max_completion_tokens")
    result.add_argument("--output", type=Path, help="Also save the acceptance report as UTF-8 JSON.")
    return result


async def run_demos(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.demo == "compression":
        return await compression_demo()
    if arguments.demo == "parallel":
        return await parallel_demo(arguments.delay)
    if arguments.demo == "code-review":
        return await code_review_demo(arguments.workspace, arguments.skills)
    if arguments.demo == "live":
        if not arguments.model:
            raise ValueError("Live demo requires --model or OPENAI_MODEL; no fake-model fallback is used")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and arguments.base_url.rstrip("/") == "https://api.openai.com/v1":
            raise ValueError("Live OpenAI demo requires OPENAI_API_KEY in the process environment")
        model = OpenAICompatibleProvider(
            model=arguments.model, api_key=api_key, base_url=arguments.base_url,
            token_limit_field=arguments.token_limit_field,
        )
        return await live_demo(model, arguments.workspace, arguments.skills)
    reports = [
        await compression_demo(),
        await parallel_demo(arguments.delay),
        await code_review_demo(arguments.workspace, arguments.skills),
    ]
    return {"demo": "all", "live_api_executed": False, "reports": reports, "passed": all(report["passed"] for report in reports)}


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        report = asyncio.run(run_demos(arguments))
        serialized = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False)
        if arguments.output:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return 0 if report["passed"] else 1
    except (ValueError, OSError) as error:
        print(f"Demo configuration error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
