"""Abstract model interface, asynchronous HTTP adapter, and deterministic fake."""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Literal, Sequence

import httpx

from .types import Message, ModelResponse, ToolCall


class ModelError(Exception):
    """A model request failed or returned an unsupported response."""


class ModelProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        *,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        """Return text or tool calls without executing any requested action."""


class OpenAICompatibleProvider(ModelProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        request_timeout: float = 30,
        token_limit_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        endpoint = httpx.URL(base_url)
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.host
            or endpoint.userinfo
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) URL without credentials or query")
        if api_key and endpoint.scheme != "https" and endpoint.host not in {
            "localhost", "127.0.0.1", "::1"
        }:
            raise ValueError("An API key requires HTTPS except on loopback hosts")
        if not model.strip() or not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("A model name and positive request timeout are required")
        if token_limit_field not in {"max_completion_tokens", "max_tokens"}:
            raise ValueError("Unsupported token limit field")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.token_limit_field = token_limit_field
        self._client = client

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        *,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_wire() for message in messages],
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        if max_output_tokens is not None:
            payload[self.token_limit_field] = max_output_tokens
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            if self._client is None:
                async with httpx.AsyncClient(follow_redirects=False) as client:
                    response = await self._request(client, payload, headers)
            else:
                response = await self._request(self._client, payload, headers)
        except httpx.TimeoutException as error:
            raise ModelError("Model HTTP request timed out") from error
        except httpx.HTTPError as error:
            raise ModelError("Model HTTP request failed") from error
        if response.status_code != 200:
            raise ModelError(f"Model endpoint returned HTTP {response.status_code}")
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") not in {"stop", "tool_calls"}:
                raise ValueError("Response was truncated or rejected")
            if message.get("role") != "assistant":
                raise ValueError("Expected an assistant message")
            content = message.get("content")
            raw_calls = message.get("tool_calls")
            if raw_calls is None:
                raw_calls = []
            if content is not None and not isinstance(content, str):
                raise ValueError("Only text content is supported")
            if not isinstance(raw_calls, list):
                raise ValueError("tool_calls must be a list")
            calls = []
            for raw_call in raw_calls:
                function = raw_call["function"]
                if raw_call.get("type") != "function" or not all(
                    isinstance(value, str) and value
                    for value in (raw_call["id"], function["name"])
                ) or not isinstance(function["arguments"], str):
                    raise ValueError("Invalid function tool call")
                calls.append(ToolCall(raw_call["id"], function["name"], function["arguments"]))
            if not calls and not (content and content.strip()):
                raise ValueError("Empty model response")
            if len({call.id for call in calls}) != len(calls):
                raise ValueError("Duplicate tool-call IDs")
            return ModelResponse(content, tuple(calls))
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelError("Model endpoint returned an invalid or incomplete response") from error

    async def _request(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        return await client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.request_timeout,
            follow_redirects=False,
        )


@dataclass(frozen=True)
class RecordedRequest:
    messages: tuple[Message, ...]
    tools: tuple[dict[str, Any], ...]
    max_output_tokens: int | None


class FakeModelProvider(ModelProvider):
    """Consume scripted responses or exceptions; never contact a network."""

    def __init__(self, responses: Sequence[ModelResponse | Exception]) -> None:
        self.responses = deque(responses)
        self.requests: list[RecordedRequest] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        *,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        self.requests.append(RecordedRequest(tuple(messages), tuple(tools), max_output_tokens))
        if not self.responses:
            raise ModelError("Fake model has no scripted response remaining")
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response
