"""Adapter Ollamy — modele o otwartych wagach uruchamiane lokalnie."""

from __future__ import annotations

import requests

from .base import RESPONSE_SCHEMA, LLMAnalyzer

NUM_CTX = 32768

REQUEST_TIMEOUT_S = 900


class OllamaAnalyzer(LLMAnalyzer):
    """Analizator oparty na modelu uruchamianym lokalnie przez Ollamę."""

    DEFAULT_MODEL = "qwen3.5:9b"

    def __init__(self, base_url: str, model: str = DEFAULT_MODEL, **kwargs) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._model = model

    def call_parameters(self) -> dict:
        return {
            **super().call_parameters(),
            "num_ctx": NUM_CTX,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "base_url": self._base_url,
        }

    @property
    def model_id(self) -> str:
        return self._model

    def call_api(self, system_message: str, user_message: str) -> str:
        options: dict = {"num_ctx": NUM_CTX}
        if self.temperature is not None:
            options["temperature"] = self.temperature

        response = requests.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": options,
                "format": RESPONSE_SCHEMA,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        body = response.json()

        message = body.get("message", {})
        self._last_meta = {
            "served_model": body.get("model"),
            "usage": {
                "input_tokens": body.get("prompt_eval_count"),
                "output_tokens": body.get("eval_count"),
            },
            "stop_reason": body.get("done_reason"),
            "thinking_chars": len(message.get("thinking") or ""),
        }
        return message.get("content", "")
