import json

import httpx
import pytest

from hkgai_agent.model import FakeModelProvider, ModelError, OpenAICompatibleProvider
from hkgai_agent.types import Message, ToolCall


def completion(content="Hello", *, tool_calls=None, finish_reason="stop"):
    return {
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                     "finish_reason": finish_reason}]
    }


async def test_http_request_serializes_tools_ids_and_token_limit():
    captured = []

    async def handler(request):
        captured.append(request)
        return httpx.Response(200, json=completion("Hello"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(model="test-model", api_key="test-only-key", client=client)
        response = await provider.complete(
            [Message("assistant", tool_calls=(ToolCall("original", "echo", "{}"),)),
             Message("tool", "{}", tool_call_id="original")],
            [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}],
            max_output_tokens=123,
        )
    assert response.content == "Hello"
    request = captured[0]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-only-key"
    body = json.loads(request.content)
    assert body["model"] == "test-model" and body["max_completion_tokens"] == 123
    assert body["messages"][1]["tool_call_id"] == "original"
    assert body["tools"][0]["function"]["name"] == "echo"
    assert body["tool_choice"] == "auto"


async def test_compatible_legacy_token_field_and_no_auth_header():
    async def handler(request):
        body = json.loads(request.content)
        assert body["max_tokens"] == 20 and "max_completion_tokens" not in body
        assert "authorization" not in request.headers
        assert "tools" not in body and "tool_choice" not in body
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            model="local", base_url="http://localhost:8000/v1/", token_limit_field="max_tokens", client=client
        )
        await provider.complete([Message("user", "Hi")], max_output_tokens=20)


async def test_parse_multiple_function_calls():
    calls = [ToolCall("a", "read_file", '{"path":"a.txt"}'), ToolCall("b", "read_file", '{"path":"b.txt"}')]
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=completion(
            None, tool_calls=[call.to_wire() for call in calls], finish_reason="tool_calls"
        ))
    )) as client:
        response = await OpenAICompatibleProvider(model="test", client=client).complete([Message("user", "Read")])
    assert response.tool_calls == tuple(calls)


@pytest.mark.parametrize("status", [301, 400, 401, 429, 500])
async def test_http_failures_do_not_expose_response_body_or_follow_redirects(status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, text="private-response-body", headers={"Location": "https://elsewhere.invalid"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        with pytest.raises(ModelError, match=str(status)) as caught:
            await OpenAICompatibleProvider(model="test", api_key="test-only-key", client=client).complete([])
    assert "private-response-body" not in str(caught.value)
    assert "test-only-key" not in str(caught.value)
    assert len(requests) == 1


@pytest.mark.parametrize("data", [
    {}, {"choices": []}, {"choices": [{"message": []}]}, completion("", finish_reason="stop"),
    completion("truncated", finish_reason="length"), completion("blocked", finish_reason="content_filter"),
    completion("Hello", tool_calls={"invalid": True}),
    completion(None, tool_calls=[{"type": "function", "id": "a", "function": {"name": "echo", "arguments": {}}}]),
    completion(None, tool_calls=[ToolCall("a", "echo", "{}").to_wire()] * 2, finish_reason="tool_calls"),
])
async def test_malformed_or_truncated_response_rejected(data):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=data)
    )) as client:
        with pytest.raises(ModelError):
            await OpenAICompatibleProvider(model="test", client=client).complete([])


async def test_network_timeout_is_wrapped():
    def handler(request):
        raise httpx.ReadTimeout("sensitive transport detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ModelError, match="timed out") as caught:
            await OpenAICompatibleProvider(model="test", client=client).complete([])
    assert "sensitive" not in str(caught.value)


@pytest.mark.parametrize("url", ["file:///tmp/socket", "https://user:pass@example.com", "https://example.com?key=secret"])
def test_invalid_endpoint_urls_rejected(url):
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(model="test", base_url=url)


def test_credentials_require_https_for_remote_endpoints():
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleProvider(model="test", base_url="http://remote.example/v1", api_key="test-only-key")
    OpenAICompatibleProvider(model="test", base_url="http://localhost:8000/v1", api_key="test-only-key")


async def test_fake_exhaustion_is_explicit():
    with pytest.raises(ModelError, match="scripted"):
        await FakeModelProvider([]).complete([])
