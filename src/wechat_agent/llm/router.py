"""LLM router with primary provider and optional fallback."""

import structlog

from wechat_agent.config import settings
from wechat_agent.llm.providers import LLMProvider, get_provider

logger = structlog.get_logger(__name__)


class LLMRouter(LLMProvider):
    """Route to a primary domestic provider, falling back when configured."""

    name = "router"

    def __init__(
        self,
        primary: LLMProvider | None = None,
        fallback: LLMProvider | None = None,
    ) -> None:
        self._primary = primary or get_provider()
        self._fallback = fallback
        if self._fallback is None and _fallback_configured():
            self._fallback = get_provider(
                base_url=settings.fallback_llm_base_url,
                api_key=settings.fallback_llm_api_key,
                model=settings.fallback_llm_model,
                api_type=settings.fallback_llm_api_type,
            )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        images: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        last_error: Exception | None = None
        providers = [self._primary]
        if self._fallback:
            providers.append(self._fallback)

        for provider in providers:
            try:
                logger.debug("llm_attempt", provider=provider.name)
                return provider.generate(
                    prompt,
                    system=system,
                    temperature=temperature,
                    images=images,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("llm_provider_failed", provider=provider.name, error=str(exc))
                last_error = exc
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def _fallback_configured() -> bool:
    return bool(
        settings.fallback_llm_base_url
        and settings.fallback_llm_api_key
        and settings.fallback_llm_model,
    )


def get_default_router() -> LLMRouter:
    """Return the default LLM router from settings."""
    return LLMRouter()
