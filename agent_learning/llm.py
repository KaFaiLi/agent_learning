from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agent_learning.config import AzureOpenAISettings
from agent_learning.models import Message, ToolCall


@dataclass
class LLMReply:
    message: Message
    prompt_tokens: int
    completion_tokens: int
    model: str
    finish_reason: str | None = None


class LLMClient(Protocol):
    def chat(self, *, messages: list[Message], tools: list[dict[str, Any]] | None, model: str | None = None) -> LLMReply: ...


class AzureClient:
    def __init__(self, settings: AzureOpenAISettings) -> None:
        if not settings.is_configured:
            raise RuntimeError("Azure OpenAI is not configured (set AZURE_OPENAI_ENDPOINT, _API_KEY, _DEPLOYMENT).")
        from openai import AzureOpenAI  # local import keeps import cost off the cold path

        self.settings = settings
        self._client = AzureOpenAI(
            api_key=settings.api_key,
            api_version=settings.api_version,
            azure_endpoint=settings.endpoint,
        )

    def chat(self, *, messages: list[Message], tools: list[dict[str, Any]] | None, model: str | None = None) -> LLMReply:
        kwargs: dict[str, Any] = {
            "model": model or self.settings.deployment,
            "messages": [m.to_openai() for m in messages],
            "temperature": 0.2,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return LLMReply(
            message=Message(
                role="assistant",
                content=msg.content,
                tool_calls=tool_calls or None,
            ),
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
            model=response.model or (model or self.settings.deployment or ""),
            finish_reason=choice.finish_reason,
        )


@dataclass
class StubScript:
    """Canned LLM behaviour for tests: a queue of (tool_calls, content) pairs."""

    turns: list[tuple[list[ToolCall], str | None]]
    model: str = "stub-model"
    cursor: int = 0

    def chat(self, *, messages, tools=None, model=None) -> LLMReply:
        if self.cursor >= len(self.turns):
            return LLMReply(
                message=Message(role="assistant", content="(stub exhausted)"),
                prompt_tokens=10,
                completion_tokens=2,
                model=self.model,
                finish_reason="stop",
            )
        tcs, content = self.turns[self.cursor]
        self.cursor += 1
        return LLMReply(
            message=Message(role="assistant", content=content, tool_calls=tcs or None),
            prompt_tokens=20,
            completion_tokens=10,
            model=self.model,
            finish_reason="tool_calls" if tcs else "stop",
        )
