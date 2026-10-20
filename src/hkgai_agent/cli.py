"""One-shot CLI with a deterministic offline demo by default."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from .agent import AgentConfig, AgentLoop
from .context import ContextConfig, ContextManager
from .model import FakeModelProvider, OpenAICompatibleProvider
from .safety import Workspace
from .skills import LoadSkillTool, SkillRegistry
from .tools import ExecutorConfig, ListFilesTool, ReadFileTool, ToolExecutor, ToolRegistry, WriteFileTool
from .types import ModelResponse, ToolCall


def demo_model() -> FakeModelProvider:
    return FakeModelProvider(
        [
            ModelResponse(tool_calls=(ToolCall("skill-1", "load_skill", '{"name":"workspace-review"}'),)),
            ModelResponse(
                tool_calls=(
                    ToolCall("read-1", "read_file", '{"path":"notes.txt"}'),
                    ToolCall("read-2", "read_file", '{"path":"constraints.txt"}'),
                )
            ),
            ModelResponse(
                "Scripted demo finished: one skill request and a batch of two file reads. "
                "The tool trace contains the actual file contents and any errors. "
                "Use --provider openai for model-generated analysis."
            ),
        ]
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run a bounded LLM agent or a credential-free scripted demo.")
    result.add_argument("prompt", nargs="?", default="Use workspace-review to inspect notes.txt and constraints.txt.")
    result.add_argument("--provider", choices=("fake", "openai"), default="fake")
    result.add_argument("--workspace", type=Path, default=Path("examples/workspace"))
    result.add_argument("--skills", type=Path, default=Path("skills"))
    result.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    result.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    result.add_argument("--token-limit-field", choices=("max_completion_tokens", "max_tokens"), default="max_completion_tokens")
    result.add_argument("--allow-write", action="store_true", help="Explicitly allow the write_file tool.")
    result.add_argument("--max-iterations", type=int, default=8)
    result.add_argument("--run-timeout", type=float, default=60)
    result.add_argument("--model-timeout", type=float, default=30)
    result.add_argument("--tool-timeout", type=float, default=5)
    result.add_argument("--concurrency", type=int, default=4)
    result.add_argument("--context-tokens", type=int, default=8192)
    result.add_argument("--compact-ratio", type=float, default=0.75)
    result.add_argument("--keep-recent", type=int, default=6)
    result.add_argument("--output-tokens", type=int, default=1024)
    return result


async def run_cli(arguments: argparse.Namespace) -> int:
    workspace = Workspace(arguments.workspace)
    skills = SkillRegistry(arguments.skills)
    skills.discover()
    tools = [ReadFileTool(workspace), ListFilesTool(workspace), LoadSkillTool(skills)]
    if arguments.allow_write:
        tools.append(WriteFileTool(workspace))
    executor = ToolExecutor(
        ToolRegistry(tools),
        allowed_tools=frozenset(tool.name for tool in tools),
        config=ExecutorConfig(concurrency=arguments.concurrency, tool_timeout=arguments.tool_timeout),
    )
    if arguments.output_tokens >= arguments.context_tokens:
        raise ValueError("--output-tokens must be smaller than --context-tokens")
    if arguments.provider == "fake":
        model = demo_model()
        print("Offline scripted demo (no API credentials or network calls).")
    else:
        if not arguments.model:
            raise ValueError("Set OPENAI_MODEL or pass --model for the openai provider")
        api_key = os.getenv("OPENAI_API_KEY")
        if urlsplit(arguments.base_url).hostname == "api.openai.com" and not (api_key and api_key.strip()):
            raise ValueError("Set OPENAI_API_KEY in the process environment before using the OpenAI API")
        model = OpenAICompatibleProvider(
            model=arguments.model,
            api_key=api_key,
            base_url=arguments.base_url,
            request_timeout=arguments.model_timeout,
            token_limit_field=arguments.token_limit_field,
        )
    agent = AgentLoop(
        model,
        executor,
        skills=skills,
        config=AgentConfig(
            max_iterations=arguments.max_iterations,
            run_timeout=arguments.run_timeout,
            model_timeout=arguments.model_timeout,
            max_output_tokens=arguments.output_tokens,
        ),
        context=ContextManager(
            ContextConfig(
                max_tokens=arguments.context_tokens,
                compact_at_ratio=arguments.compact_ratio,
                keep_recent_messages=arguments.keep_recent,
            )
        ),
    )
    outcome = await agent.run(arguments.prompt)
    for tool_result in outcome.tool_results:
        print(f"[tool {tool_result.tool_call_id}: {tool_result.name}] {tool_result.to_message().content}")
    print(outcome.response)
    print(json.dumps({"termination": outcome.termination, "iterations": outcome.iterations}))
    return 0 if outcome.termination == "final" else 1


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        return asyncio.run(run_cli(arguments))
    except (ValueError, OSError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
