"""Adapter OpenRoutera dla modeli o otwartych wagach."""

from __future__ import annotations

import logging

import openai

from .base import RESPONSE_SCHEMA, LLMAnalyzer

logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"

MAX_TOKENS = 8192

MAX_RETRIES = 6

DEFAULT_PROVIDER = "deepinfra/bf16"


class OpenRouterAnalyzer(LLMAnalyzer):
    """Analizator oparty na modelu o otwartych wagach serwowanym przez OpenRouter."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        cache_control: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._client = openai.OpenAI(api_key=api_key, base_url=BASE_URL, max_retries=MAX_RETRIES)
        self._model = model
        self._provider = provider
        self._cache_control = cache_control

    def call_parameters(self) -> dict:
        return {
            **super().call_parameters(),
            "max_tokens": MAX_TOKENS,
            "max_retries": MAX_RETRIES,
            "base_url": BASE_URL,
            "provider": self._provider,
            "allow_fallbacks": False,
            "require_parameters": True,
            "cache_control": self._cache_control,
        }

    @property
    def model_id(self) -> str:
        return self._model

    def is_fatal_error(self, exc: Exception) -> str | None:
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return f"{type(exc).__name__}: {exc}"
        if isinstance(exc, openai.RateLimitError):
            message = str(exc).lower()
            if "insufficient" in message or "credits" in message or "quota" in message:
                return f"insufficient_credits: {exc}"
        if isinstance(exc, openai.NotFoundError):
            return f"{type(exc).__name__}: {exc}"
        return None

    def _system_content(self, system_message: str):
        if not self._cache_control:
            return system_message
        return [
            {
                "type": "text",
                "text": system_message,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def call_api(self, system_message: str, user_message: str) -> str:
        optional: dict = {}
        if self.temperature is not None:
            optional["temperature"] = self.temperature
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_content(system_message)},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
            **optional,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sast_verdict",
                    "schema": RESPONSE_SCHEMA,
                    "strict": True,
                },
            },
            extra_body={
                "provider": {
                    "order": [self._provider],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                "usage": {"include": True},
            },
        )

        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        message = response.choices[0].message
        self._last_meta = {
            "served_model": response.model,
            "served_provider": getattr(response, "provider", None),
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                "cached_input_tokens": getattr(prompt_details, "cached_tokens", None),
                "cost_usd": getattr(usage, "cost", None),
            },
            "stop_reason": response.choices[0].finish_reason,
            "thinking_chars": len(getattr(message, "reasoning", None) or ""),
        }

        return message.content or ""
