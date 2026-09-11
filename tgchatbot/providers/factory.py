from __future__ import annotations

from tgchatbot.config import AppConfig
from tgchatbot.providers.base import ModelProvider
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider


def build_provider(config: AppConfig, name: str) -> ModelProvider:
    """Construct one configured provider, including for offline maintenance jobs."""
    if name not in config.configured_provider_names():
        available = ', '.join(config.configured_provider_names()) or 'none'
        raise RuntimeError(f'Provider {name!r} is not configured. Available providers: {available}')
    if name == 'openai':
        return OpenAIResponsesProvider(config.openai)
    if name == 'gemini':
        return GeminiProvider(config.gemini)
    return ChatCompletionsProvider(config.provider_config(name))


def build_providers(config: AppConfig) -> dict[str, ModelProvider]:
    names = config.configured_provider_names()
    if not names:
        raise RuntimeError('Configure a provider API key: OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY or LLM_PROVIDERS_JSON')
    if config.default_provider not in names:
        raise RuntimeError(f'DEFAULT_PROVIDER={config.default_provider!r} is not configured. Available providers: {", ".join(names)}')
    return {name: build_provider(config, name) for name in names}
