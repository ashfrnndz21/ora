"""LLM provider + embedder — all LLM interaction in one file.

Uses litellm as the universal routing layer. Supports:
  gpt-4o, claude-sonnet-4-6, bedrock/*, ollama/*, azure/*, vertex_ai/*
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    content: str = ""
    model: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    finish_reason: str = ""


@dataclass
class Message:
    role: str        # "system", "user", "assistant"
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse: ...

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]: ...


# ═══════════════════════════════════════════════════════════════════════════════
# LITELLM PROVIDER
# ═══════════════════════════════════════════════════════════════════════════════

class LiteLLMProvider:
    """Universal LLM provider via litellm.

    Routes to OpenAI, Anthropic, Bedrock, Ollama, vLLM, Azure, Vertex AI
    based on model name prefix.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0, max_tokens: int = 4096):
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        # Per-query token accumulators — reset at start of each query
        self._session_tokens_input: int = 0
        self._session_tokens_output: int = 0
        self._session_calls: int = 0

    def reset_session_tokens(self) -> None:
        """Call once at the start of each query to reset per-query token counters."""
        self._session_tokens_input = 0
        self._session_tokens_output = 0
        self._session_calls = 0

    async def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        import litellm

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        if json_mode:
            # Anthropic/Claude doesn't support response_format — use prompt instruction instead
            if "claude" in self.model.lower() or "anthropic" in self.model.lower():
                # Append JSON instruction to the last user message
                if kwargs["messages"]:
                    last = kwargs["messages"][-1]
                    if last["role"] == "user":
                        kwargs["messages"] = kwargs["messages"][:-1] + [
                            {"role": last["role"], "content": last["content"] + "\n\nRespond with valid JSON only. No markdown, no explanation, just the JSON object."}
                        ]
            else:
                kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await litellm.acompletion(**kwargs)

            content = response.choices[0].message.content or ""
            usage = response.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0

            try:
                cost = litellm.completion_cost(
                    model=self.model,
                    prompt=str(tokens_in),
                    completion=str(tokens_out),
                )
            except Exception as exc:
                logger.debug("llm.operation_failed", error=str(exc))
                cost = 0.0

            # Accumulate into per-query session counters
            self._session_tokens_input += tokens_in
            self._session_tokens_output += tokens_out
            self._session_calls += 1

            return LLMResponse(
                content=content,
                model=self.model,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                cost_usd=cost,
                finish_reason=response.choices[0].finish_reason or "",
            )

        except Exception as e:
            logger.error("llm.call_failed", model=self.model, error=str(e))
            from sqlagent.exceptions import LLMCallFailed
            raise LLMCallFailed(f"LLM call failed ({self.model}): {e}") from e

    async def stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        import litellm

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
            "stream": True,
        }

        response = await litellm.acompletion(**kwargs)

        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDER
# ═══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimensions(self) -> int: ...


class FastEmbedEmbedder:
    """Embedder using fastembed (local, no API key needed).

    Default model: BAAI/bge-small-en-v1.5 (384 dimensions, fast).
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self._model_name = model
        self._model = None
        self._dims = 384  # bge-small default

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self._model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        self._ensure_model()
        # fastembed is sync — wrap in thread
        embeddings = await asyncio.to_thread(
            lambda: list(self._model.embed(texts))
        )
        return [list(e) for e in embeddings]

    @property
    def dimensions(self) -> int:
        return self._dims
