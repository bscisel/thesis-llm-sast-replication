"""Adapter OpenAI."""

from __future__ import annotations

import logging

import openai

from .base import RESPONSE_SCHEMA, LLMAnalyzer

logger = logging.getLogger(__name__)

MAX_COMPLETION_TOKENS = 8192

MAX_RETRIES = 6


class GPTAnalyzer(LLMAnalyzer):
    """Analizator oparty na OpenAI Chat Completions."""

    DEFAULT_MODEL = "gpt-5.6-terra"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = openai.OpenAI(api_key=api_key, max_retries=MAX_RETRIES)
        self._model = model
        if self.temperature is not None:
            logger.warning(
                "%s przyjmuje wyłącznie własną domyślną temperaturę; pominięto "
                "LLM_TEMPERATURE=%s.",
                model,
                self.temperature,
            )

    def call_parameters(self) -> dict:
        return {
            **super().call_parameters(),
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "max_retries": MAX_RETRIES,
            "reasoning_effort": "default",
        }

    @property
    def model_id(self) -> str:
        return self._model

    def is_fatal_error(self, exc: Exception) -> str | None:
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return f"{type(exc).__name__}: {exc}"
        if isinstance(exc, openai.RateLimitError):
            code = getattr(exc, "code", None) or ((getattr(exc, "body", None) or {}).get("error") or {}).get("code")
            if code == "insufficient_quota" or "no credits remaining" in str(exc).lower():
                return f"insufficient_quota: {exc}"
        return None

    def call_api(self, system_message: str, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sast_verdict",
                    "schema": RESPONSE_SCHEMA,
                    "strict": True,
                },
            },
        )

        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        self._last_meta = {
            "served_model": response.model,
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                "cached_input_tokens": getattr(prompt_details, "cached_tokens", None),
            },
            "stop_reason": response.choices[0].finish_reason,
        }

        return response.choices[0].message.content or ""
