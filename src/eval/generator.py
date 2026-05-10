from __future__ import annotations

import json
import logging
import os

from loguru import logger
from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

_api_retry = retry(
    retry=retry_if_exception_type(
        (RateLimitError, APITimeoutError, APIConnectionError, json.JSONDecodeError)
    ),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class APIGenerator:
    """Generates responses via an OpenAI-compatible chat API with tenacity retry."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int = 512,
        thinking: bool | None = None,
        reasoning: str | None = None,
        thinking_token_budget: int | None = None,
    ):
        """Initialize an API generator.

        Args:
            model: Model identifier string.
            base_url: API base URL.
            api_key: API authentication key.
            max_tokens: Maximum output tokens.
            thinking: Enable/disable thinking mode (vLLM/Tinker only).
            reasoning: OpenAI Responses API reasoning effort level.
            thinking_token_budget: vLLM thinking token cap.

        Raises:
            ValueError: If incompatible options are specified for the provider.
        """
        from openai import OpenAI

        is_openrouter = OPENROUTER_BASE_URL in base_url
        is_openai = OPENAI_BASE_URL in base_url

        if is_openrouter:
            if thinking is not None:
                raise ValueError(
                    f"'thinking' is not supported for OpenRouter ({model}). "
                    "Use a thinking/instruct model variant instead."
                )
            if thinking_token_budget is not None:
                raise ValueError(
                    f"'thinking_token_budget' is not supported for OpenRouter ({model})."
                )
        if reasoning is not None and not is_openai:
            raise ValueError(
                f"'reasoning' (Responses API) is only supported for OpenAI ({model}), "
                f"not {base_url}"
            )
        if thinking_token_budget is not None and is_openai:
            raise ValueError(
                f"'thinking_token_budget' is a vLLM param, not supported for OpenAI ({model})."
            )

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking if thinking is not None else True
        self.reasoning = reasoning
        self.thinking_token_budget = thinking_token_budget
        self._is_openrouter = is_openrouter
        self._use_responses_api = reasoning is not None and is_openai

    def __call__(
        self, messages: list[dict], temperature: float = 0.7, top_p: float = 0.9
    ) -> str:
        """Generate a response from the model.

        Args:
            messages: Chat message history.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Model response text.
        """
        if self._use_responses_api:
            return self._call_responses_api(messages)
        return self._call_chat_completions(messages, temperature, top_p)

    @staticmethod
    def _sanitize_msgs(messages: list[dict]) -> list[dict]:
        ALLOWED = {"role", "content"}
        return [{k: v for k, v in m.items() if k in ALLOWED} for m in messages]

    @_api_retry
    def _call_responses_api(self, messages: list[dict]) -> str:
        """Call the OpenAI Responses API for reasoning models."""
        resp = self.client.responses.create(
            model=self.model,
            input=self._sanitize_msgs(messages),
            reasoning={"effort": self.reasoning, "summary": "auto"},
            max_output_tokens=self.max_tokens,
        )
        reasoning_text = ""
        content = ""
        for item in resp.output:
            if item.type == "reasoning" and item.summary:
                reasoning_text = "\n".join(s.text for s in item.summary)
            elif item.type == "message":
                for block in item.content:
                    if block.type == "output_text":
                        content = block.text.strip()
        if reasoning_text:
            return f"<think>{reasoning_text}</think>\n{content}"
        return content

    @_api_retry
    def _call_chat_completions(
        self, messages: list[dict], temperature: float, top_p: float
    ) -> str:
        """Call the Chat Completions API (vLLM, Tinker, OpenRouter, etc.)."""
        token_key = (
            "max_completion_tokens"
            if OPENAI_BASE_URL in self.base_url
            else "max_tokens"
        )
        kwargs: dict = dict(
            model=self.model,
            messages=self._sanitize_msgs(messages),
            temperature=temperature,
            top_p=top_p,
            **{token_key: self.max_tokens},
        )
        if not self._is_openrouter:
            extra: dict = {}
            if not self.thinking:
                extra["chat_template_kwargs"] = {"enable_thinking": False}
            if self.thinking_token_budget is not None:
                extra["thinking_token_budget"] = self.thinking_token_budget
            if extra:
                kwargs["extra_body"] = extra
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning", None) or ""
        if reasoning:
            return f"<think>{reasoning}</think>\n{content}"
        return content


def make_generator(
    agent_cfg: dict,
    default_base_url: str = OPENAI_BASE_URL,
    default_api_key_env: str = "OPENAI_API_KEY",
) -> APIGenerator:
    """Build an APIGenerator from a matchup agent config block.

    Args:
        agent_cfg: Dict with model, base_url, api_key_env, max_tokens, etc.
        default_base_url: Fallback base URL.
        default_api_key_env: Fallback env var name for API key.

    Returns:
        Configured APIGenerator instance.

    Raises:
        ValueError: If required API key env var is empty.
    """
    api_key_env = agent_cfg.get("api_key_env", default_api_key_env)
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ValueError(f"{api_key_env} env var required")
    base_url = agent_cfg.get("base_url", default_base_url)
    return APIGenerator(
        model=agent_cfg["model"],
        base_url=base_url,
        api_key=api_key,
        max_tokens=agent_cfg.get("max_tokens", 512),
        thinking=agent_cfg.get("thinking"),
        reasoning=agent_cfg.get("reasoning"),
        thinking_token_budget=agent_cfg.get("thinking_token_budget"),
    )
