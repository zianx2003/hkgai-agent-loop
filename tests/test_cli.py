import json
from pathlib import Path

import httpx
import pytest

from hkgai_agent.cli import main, parser, run_cli
from hkgai_agent.model import OpenAICompatibleProvider
from hkgai_agent.types import ToolCall


ROOT = Path(__file__).resolve().parents[1]


def test_credential_free_demo(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(["--workspace", str(ROOT / "examples/workspace"), "--skills", str(ROOT / "skills")])
    captured = capsys.readouterr()
    assert code == 0 and not captured.err
    assert "Offline scripted demo" in captured.out
    assert '"termination": "final", "iterations": 3' in captured.out
    assert "[tool skill-1: load_skill]" in captured.out
    assert "[tool read-1: read_file]" in captured.out
    assert "[tool read-2: read_file]" in captured.out
    assert "technical interview" in captured.out


def test_bad_workspace_is_a_configuration_error(tmp_path, capsys):
    assert main(["--workspace", str(tmp_path / "missing")]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_real_mode_requires_model_configuration(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert main(["--provider", "openai", "--workspace", str(ROOT / "examples/workspace")]) == 2
    assert "OPENAI_MODEL" in capsys.readouterr().err


def test_iteration_limit_exit_code(capsys):
    assert main(["--workspace", str(ROOT / "examples/workspace"), "--skills", str(ROOT / "skills"),
                 "--max-iterations", "1"]) == 1
    assert '"termination": "max_iterations"' in capsys.readouterr().out


@pytest.mark.parametrize("key", [None, "", "   "])
def test_openai_requires_key_before_constructing_provider(key, monkeypatch, capsys):
    if key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", key)

    def forbidden_provider(**kwargs):
        raise AssertionError("Missing OpenAI credentials must fail before provider construction")

    monkeypatch.setattr("hkgai_agent.cli.OpenAICompatibleProvider", forbidden_provider)
    code = main([
        "--provider", "openai", "--model", "gpt-4.1-mini", "--base-url", "https://api.openai.com/v1/",
        "--workspace", str(ROOT / "examples/workspace"), "--skills", str(ROOT / "skills"), "Read notes.txt."
    ])
    assert code == 2
    assert "Set OPENAI_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("base_url,key", [
    ("https://api.openai.com/v1", "test-only-key"), ("http://localhost:8000/v1", None),
])
async def test_real_cli_path_sends_file_observations_to_adapter_and_returns_final_answer(
    base_url, key, monkeypatch, capsys
):
    if key:
        monkeypatch.setenv("OPENAI_API_KEY", key)
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests = []
    prompt = "Read notes.txt and constraints.txt using read_file, then summarize both. Do not modify files."
    final_answer = "The project needs an asyncio agent loop, separate modules, and workspace restrictions."

    def handler(request):
        if key:
            assert request.headers["authorization"] == f"Bearer {key}"
        else:
            assert "authorization" not in request.headers
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            assert payload["messages"][-1] == {"role": "user", "content": prompt}
            message = {"role": "assistant", "content": None, "tool_calls": [
                ToolCall("notes-call", "read_file", '{"path":"notes.txt"}').to_wire(),
                ToolCall("constraints-call", "read_file", '{"path":"constraints.txt"}').to_wire(),
            ]}
            finish_reason = "tool_calls"
        else:
            observations = [message for message in payload["messages"] if message["role"] == "tool"]
            assert [message["tool_call_id"] for message in observations] == ["notes-call", "constraints-call"]
            assert "technical interview" in json.loads(observations[0]["content"])["output"]["text"]
            assert "workspace" in json.loads(observations[1]["content"])["output"]["text"]
            message = {"role": "assistant", "content": final_answer}
            finish_reason = "stop"
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish_reason}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        def provider_factory(**kwargs):
            return OpenAICompatibleProvider(**kwargs, client=client)

        monkeypatch.setattr("hkgai_agent.cli.OpenAICompatibleProvider", provider_factory)
        code = await run_cli(parser().parse_args([
            "--provider", "openai", "--model", "gpt-4.1-mini", "--base-url", base_url,
            "--workspace", str(ROOT / "examples/workspace"), "--skills", str(ROOT / "skills"), prompt,
        ]))
    captured = capsys.readouterr()
    assert code == 0 and len(requests) == 2
    assert all(request["model"] == "gpt-4.1-mini" for request in requests)
    assert "[tool notes-call: read_file]" in captured.out
    assert "[tool constraints-call: read_file]" in captured.out
    assert final_answer in captured.out
    assert '"termination": "final", "iterations": 2' in captured.out
    assert not captured.err
    if key:
        assert key not in captured.out
