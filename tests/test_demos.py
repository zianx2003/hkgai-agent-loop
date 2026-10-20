import json
from pathlib import Path

import httpx
import pytest

from hkgai_agent.demos import code_review_demo, compression_demo, live_demo, main, parallel_demo
from hkgai_agent.model import OpenAICompatibleProvider
from hkgai_agent.types import ToolCall


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "examples/workspace"
SKILLS = ROOT / "skills"


async def test_compression_demo_shows_message_structure_and_preserved_constraints():
    report = await compression_demo()
    before, after = report["before"], report["after"]
    assert report["passed"]
    assert after["estimated_tokens"] < before["estimated_tokens"]
    assert len(after["messages"]) < len(before["messages"])
    assert after["messages"][0] == before["messages"][0]
    assert after["messages"][-4:] == before["messages"][-4:]
    summary = next(message for message in after["messages"] if message["is_summary"])
    assert all(marker in summary["content"] for marker in (
        "review_sample.py", "budget <=10", "do not modify files", "pending validation"
    ))
    requested = [call["id"] for message in after["messages"] for call in message.get("tool_calls", [])]
    returned = [message["tool_call_id"] for message in after["messages"] if message["role"] == "tool"]
    assert requested == returned == ["recent-list"]


async def test_fixed_delay_tools_overlap_and_preserve_id_value_mapping():
    report = await parallel_demo(0.02)
    assert report["passed"]
    serial = report["serial"]["results"]
    parallel = report["parallel"]["results"]
    assert serial[1]["output"]["started_seconds"] >= serial[0]["output"]["finished_seconds"]
    latest_start = max(result["output"]["started_seconds"] for result in parallel)
    earliest_finish = min(result["output"]["finished_seconds"] for result in parallel)
    assert latest_start < earliest_finish
    assert parallel[1]["output"]["finished_seconds"] < parallel[0]["output"]["finished_seconds"]
    for results in (serial, parallel):
        assert [(result["id"], result["output"]["value"]) for result in results] == [
            ("slow-id", "notes"), ("fast-id", "constraints")
        ]
        for result in results:
            output = result["output"]
            assert output["finished_seconds"] > output["started_seconds"]


async def test_code_review_skill_loads_and_security_rejections_are_observable():
    sample = WORKSPACE / "review_sample.py"
    original = sample.read_bytes()
    report = await code_review_demo(WORKSPACE, SKILLS)
    assert report["passed"] and report["termination"] == "final"
    results = {result["id"]: result for result in report["results"]}
    assert results["review-skill"]["output"]["name"] == "code_review"
    assert "Do not execute" in results["review-skill"]["output"]["instructions"]
    assert "def average" in results["review-source"]["output"]["text"]
    assert results["outside-path"]["error"]["code"] == "unsafe_path"
    assert results["blocked-write"]["error"]["code"] == "not_allowed"
    assert results["blocked-shell"]["error"]["code"] == "not_allowed"
    assert sample.read_bytes() == original
    assert "Scripted" in report["response"]


@pytest.mark.parametrize("read_path", ["review_sample.py", "./review_sample.py", ".\\review_sample.py"])
async def test_live_acceptance_runner_uses_adapter_for_full_tool_calling_loop(read_path):
    requests = []
    responses = [
        {"role": "assistant", "content": None, "tool_calls": [
            ToolCall("live-skill", "load_skill", '{"name":"code_review"}').to_wire()
        ]},
        {"role": "assistant", "content": None, "tool_calls": [
            ToolCall("live-read", "read_file", json.dumps({"path": read_path})).to_wire()
        ]},
        {"role": "assistant", "content": "review_sample.py:5 divides by zero for an empty list."},
    ]

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        message = responses.pop(0)
        assert all(schema["function"]["name"] != "write_file" for schema in payload["tools"])
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop", "message": message,
        }]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleProvider(model="mock-http-only", client=client)
        report = await live_demo(model, WORKSPACE, SKILLS)
    assert report["passed"] and report["iterations"] == 3
    assert report["provider"] == "OpenAICompatibleProvider"
    assert len(requests) == 3
    assert requests[1]["messages"][-1]["tool_call_id"] == "live-skill"
    assert requests[2]["messages"][-1]["tool_call_id"] == "live-read"
    assert "def average" in requests[2]["messages"][-1]["content"]


async def test_live_acceptance_fails_if_model_answers_without_using_tools():
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "content": "An unsupported guess."},
        }]})
    )) as client:
        report = await live_demo(OpenAICompatibleProvider(model="mock-http-only", client=client), WORKSPACE, SKILLS)
    assert report["termination"] == "final"
    assert not report["passed"]
    assert not report["checks"]["code_review_skill_loaded"]
    assert not report["checks"]["source_read"]


def test_all_offline_demos_export_a_report_without_constructing_http_provider(tmp_path, monkeypatch, capsys):
    def forbidden_provider(*args, **kwargs):
        raise AssertionError("Offline demos must not construct an HTTP provider")

    monkeypatch.setattr("hkgai_agent.demos.OpenAICompatibleProvider", forbidden_provider)
    output = tmp_path / "reports" / "acceptance.json"
    code = main([
        "all", "--workspace", str(WORKSPACE), "--skills", str(SKILLS), "--delay", "0.02", "--output", str(output)
    ])
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == json.loads(capsys.readouterr().out)
    assert report["passed"] and not report["live_api_executed"]
    assert [item["demo"] for item in report["reports"]] == ["compression", "parallel", "code-review"]


@pytest.mark.parametrize("delay", ["0", "-1", "nan", "inf", "6"])
def test_invalid_demo_delays_are_configuration_errors(delay, capsys):
    assert main(["parallel", f"--delay={delay}"]) == 2
    assert "Demo configuration error" in capsys.readouterr().err


def test_live_demo_requires_model_configuration(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert main(["live"]) == 2
    assert "requires --model or OPENAI_MODEL" in capsys.readouterr().err


def test_live_openai_demo_requires_credentials_without_falling_back(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["live", "--model", "test", "--base-url", "https://api.openai.com/v1"]) == 2
    assert "requires OPENAI_API_KEY" in capsys.readouterr().err
