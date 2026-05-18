"""Abstract LLM provider.

Hard invariant: code outside ``src/llm/`` must not import ``anthropic``
or ``ollama`` directly. Everything goes through :class:`LLMProvider`.
This is what lets the Ollama backup work and what lets tests use a
mock provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.config import load_config, load_prompts


class LLMProvider(ABC):
    """Abstract interface every LLM backend implements."""

    @abstractmethod
    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        """Return a completion for ``system`` + ``user``."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for ``text``."""

    # ----- shared helpers ------------------------------------------

    def render_and_complete(self, prompt_name: str, variables: dict, **kwargs: Any) -> str:
        """Render the named template and call :meth:`complete`."""
        prompts = load_prompts()
        if prompt_name not in prompts:
            raise KeyError(f"prompt '{prompt_name}' not found in prompts.yaml")
        template = prompts[prompt_name]
        system = template.get("system", "").format(**variables) if "system" in template else ""
        user = template.get("user", "").format(**variables) if "user" in template else ""
        if not user:
            raise ValueError(f"prompt '{prompt_name}' has no user template")
        return self.complete(system=system, user=user, **kwargs)


def get_provider(name: str | None = None) -> LLMProvider:
    """Factory keyed on ``config.llm.provider`` or the ``name`` argument.

    Wraps the chosen backend in :class:`CachedProvider` when
    ``config.llm.cache.enabled`` is true. The mock provider is never
    cached — tests rely on deterministic responses without any
    filesystem state.
    """
    cfg = load_config()
    name = name or cfg["llm"]["provider"]

    inner: LLMProvider
    if name == "anthropic":
        from src.llm.anthropic_provider import AnthropicProvider

        inner = AnthropicProvider()
    elif name == "ollama":
        from src.llm.ollama_provider import OllamaProvider

        inner = OllamaProvider()
    elif name == "mock":
        from src.llm.mock_provider import MockProvider

        # Mock is deterministic by design — never wrap it.
        return MockProvider()
    else:
        raise ValueError(f"unknown LLM provider '{name}'")

    cache_cfg = cfg["llm"].get("cache", {})
    if cache_cfg.get("enabled", False):
        from src.llm.cache import CachedProvider

        return CachedProvider(inner, cache_cfg["path"])
    return inner
