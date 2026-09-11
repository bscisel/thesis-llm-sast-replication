"""Adapter Anthropic Claude."""

from __future__ import annotations

import logging

import anthropic

from .base import RESPONSE_SCHEMA, LLMAnalyzer

logger = logging.getLogger(__name__)

MAX_TOKENS = 8192

MAX_RETRIES = 6


class ClaudeAnalyzer(LLMAnalyzer):
    """Analizator oparty na modelach Anthropic Claude."""

    DEFAULT_MODEL = "claude-opus-5"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url
        client_kwargs: dict = {"api_key": api_key, "max_retries": MAX_RETRIES}
        if base_url:
            client_kwargs["base_url"] = base_url
            logger.info("Using non-default Anthropic endpoint: %s", base_url)
        self._client = anthropic.Anthropic(**client_kwargs)
        self._model = model
        if self.temperature is not None:
            logger.warning(
                "%s does not accept a temperature parameter; ignoring LLM_TEMPERATURE=%s "
                "and using the model default.",
                model,
                self.temperature,
            )

    def call_parameters(self) -> dict:
        return {
            **super().call_parameters(),
            "max_tokens": MAX_TOKENS,
            "max_retries": MAX_RETRIES,
            "base_url": self._base_url or "https://api.anthropic.com",
        }

    @property
    def model_id(self) -> str:
        return self._model

    def is_fatal_error(self, exc: Exception) -> str | None:
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return f"{type(exc).__name__}: {exc}"
        if isinstance(exc, anthropic.APIStatusError) and getattr(exc, "type", None) == "billing_error":
            return f"billing_error: {exc}"
        return None

    def call_api(self, system_message: str, user_message: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": system_message,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
            output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        )

        usage = response.usage
        self._last_meta = {
            "served_model": response.model,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            },
            "stop_reason": response.stop_reason,
        }

        return "".join(block.text for block in response.content if block.type == "text")
