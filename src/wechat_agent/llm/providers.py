"""Unified domestic LLM clients supporting OpenAI and Anthropic HTTP API schemas."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Any

import httpx
import structlog

from wechat_agent.config import settings

logger = structlog.get_logger(__name__)

DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 1024


class LLMProvider(ABC):
    """Abstract LLM provider."""

    name: str = "abstract"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        images: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: The text prompt.
            system: Optional system message.
            temperature: Sampling temperature.
            images: Optional list of base64-encoded image strings (without data URL prefix).
            max_tokens: Optional maximum number of tokens to generate.
        """


def _trim_prompt(prompt: str, max_chars: int = 120_000) -> str:
    """Trim prompt to avoid hitting token limits."""
    if len(prompt) <= max_chars:
        return prompt
    return "..." + prompt[-(max_chars - 3) :]


def _normalize_base_url(base_url: str) -> str:
    """Ensure base URL ends with a trailing slash."""
    if not base_url.endswith("/"):
        return base_url + "/"
    return base_url


class _HTTPProvider(LLMProvider):
    """Shared HTTP client scaffolding for OpenAI/Anthropic-style providers."""

    name = "abstract_http"
    api_type = "abstract"
    request_path = ""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._base_url = _normalize_base_url(base_url or settings.llm_base_url)
        self._api_key = api_key or settings.llm_api_key
        self._model = model or settings.llm_model
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=DEFAULT_TIMEOUT,
            headers=self._headers(),
        )

    @abstractmethod
    def _headers(self) -> dict[str, str]:
        """Return request headers for this API schema."""

    @abstractmethod
    def _build_payload(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        images: list[str] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build the request payload."""

    @abstractmethod
    def _parse_response(self, data: dict[str, Any]) -> str:
        """Extract the reply text from the response body."""

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        images: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self._api_key:
            raise RuntimeError(f"API key not configured for provider: {self.name}")
        if not self._model:
            raise RuntimeError(f"Model not configured for provider: {self.name}")

        effective_max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        payload = self._build_payload(
            _trim_prompt(prompt), system, temperature, images, effective_max_tokens
        )
        logger.debug(
            "llm_request",
            provider=self.name,
            model=self._model,
            api_type=self.api_type,
            has_images=bool(images),
            max_tokens=effective_max_tokens,
        )
        response = self._client.post(self.request_path, json=payload)
        response.raise_for_status()
        return self._parse_response(response.json())


class DomesticProvider(_HTTPProvider):
    """Generic domestic LLM provider using an OpenAI-compatible /chat/completions endpoint.

    Works with any vendor that exposes the standard chat completions schema:
    Volcano Ark, DeepSeek, Zhipu (GLM), Tongyi Qwen, Moonshot Kimi, Baichuan, etc.
    """

    name = "domestic"
    api_type = "openai"
    request_path = "chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _build_payload(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        images: list[str] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images:
            for image_b64 in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    }
                )
        messages.append({"role": "user", "content": content})

        return {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _parse_response(self, data: dict[str, Any]) -> str:
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response format from {self.name}: {data}") from exc


class AnthropicCompatibleProvider(_HTTPProvider):
    """Generic provider using Anthropic's /v1/messages API schema.

    Works with any vendor that exposes the Anthropic messages endpoint,
    including Volcano Ark's Anthropic-compatible route.
    """

    name = "anthropic_compatible"
    api_type = "anthropic"
    request_path = "v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _build_payload(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        images: list[str] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if images:
            for image_b64 in images:
                media_type = _guess_media_type(image_b64)
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    }
                )
        content.append({"type": "text", "text": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        return payload

    def _parse_response(self, data: dict[str, Any]) -> str:
        try:
            for block in data["content"]:
                if block.get("type") == "text":
                    return str(block["text"])
            raise RuntimeError("No text block found in Anthropic response")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response format from {self.name}: {data}") from exc


def get_provider(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    api_type: str | None = None,
) -> LLMProvider:
    """Return a domestic LLM provider from the unified configuration.

    Parameters can be overridden to create safety/fallback providers.
    api_type selects the HTTP schema: "openai" (default) or "anthropic".
    """
    resolved_api_type = (api_type or settings.llm_api_type).lower()
    if resolved_api_type == "anthropic":
        return AnthropicCompatibleProvider(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
    return DomesticProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


def _guess_media_type(image_b64: str) -> str:
    """Guess image media type from base64 prefix or default to PNG."""
    try:
        header = base64.b64decode(image_b64[:16])
        if header.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if header.startswith(b"\x89PNG"):
            return "image/png"
        if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            return "image/gif"
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return "image/webp"
    except Exception:
        pass
    return "image/png"
