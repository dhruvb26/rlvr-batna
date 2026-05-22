from __future__ import annotations

import json
import logging
import os

from loguru import logger
from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

_QUOTA_KEYWORDS = ("insufficient_quota", "billing", "exceeded your current quota")

_tinker_telemetry_filter_installed = False


def _install_tinker_telemetry_filter() -> None:
    """Suppress spurious warnings from nested @capture_exceptions in tinker>=0.21."""
    global _tinker_telemetry_filter_installed
    if _tinker_telemetry_filter_installed:
        return

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return (
                "@capture_exceptions used without TelemetryProvider"
                not in record.getMessage()
            )

    logging.getLogger("tinker.lib.telemetry").addFilter(_Filter())
    _tinker_telemetry_filter_installed = True


def _is_quota_error(exc: BaseException) -> bool:
    if not isinstance(exc, RateLimitError):
        return False
    msg = str(exc).lower()
    return any(kw in msg for kw in _QUOTA_KEYWORDS)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return not _is_quota_error(exc)
    return isinstance(exc, (APITimeoutError, APIConnectionError, json.JSONDecodeError))


_api_retry = retry(
    retry=retry_if_exception(_is_transient),
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
        api_keys: list[str],
        max_tokens: int = 512,
        thinking: bool | None = None,
        reasoning: str | None = None,
        thinking_token_budget: int | None = None,
    ):
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

        self._api_keys = api_keys
        self._key_idx = 0
        self.client = OpenAI(base_url=base_url, api_key=api_keys[0])
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking if thinking is not None else True
        self.reasoning = reasoning
        self.thinking_token_budget = thinking_token_budget
        self._is_openrouter = is_openrouter
        self._use_responses_api = reasoning is not None and is_openai

    def _rotate_key(self) -> bool:
        """Switch to next API key. Returns False if no keys left."""
        from openai import OpenAI

        if self._key_idx + 1 >= len(self._api_keys):
            return False
        self._key_idx += 1
        self.client = OpenAI(
            base_url=self.base_url, api_key=self._api_keys[self._key_idx]
        )
        logger.warning(
            f"Quota exhausted, rotated to API key "
            f"{self._key_idx + 1}/{len(self._api_keys)}"
        )
        return True

    def __call__(
        self, messages: list[dict], temperature: float = 0.7, top_p: float = 0.9
    ) -> str:
        while True:
            try:
                if self._use_responses_api:
                    return self._call_responses_api(messages)
                return self._call_chat_completions(messages, temperature, top_p)
            except RateLimitError as e:
                if _is_quota_error(e) and self._rotate_key():
                    continue
                raise

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


class TinkerGenerator:
    """Generates responses via the Tinker SDK SamplingClient."""

    def __init__(self, model_path: str, max_tokens: int = 512):
        import tinker
        from tinker_cookbook import model_info
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        _install_tinker_telemetry_filter()
        service = tinker.ServiceClient()
        self._sampler = service.create_sampling_client(model_path=model_path)

        base_model = self._sampler.get_base_model()
        renderer_name = model_info.get_recommended_renderer_name(base_model)
        tokenizer = get_tokenizer(base_model)
        self._renderer = get_renderer(renderer_name, tokenizer)
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._model_path = model_path
        logger.info(f"TinkerGenerator ready: {model_path} (base: {base_model})")

    def __call__(
        self, messages: list[dict], temperature: float = 0.7, top_p: float = 0.9
    ) -> str:
        import tinker
        from tinker_cookbook.renderers import get_text_content

        renderer_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        model_input = self._renderer.build_generation_prompt(renderer_msgs)
        stop = self._renderer.get_stop_sequences()

        result = self._sampler.sample(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=temperature,
                max_tokens=self._max_tokens,
                stop=stop,
            ),
        ).result()

        parsed, _ = self._renderer.parse_response(result.sequences[0].tokens)
        return get_text_content(parsed).strip()


# Global cache so multiple matchups sharing a checkpoint reuse the same session
_tinker_cache: dict[str, TinkerGenerator] = {}


def make_generator(
    agent_cfg: dict,
    default_base_url: str = OPENAI_BASE_URL,
    default_api_key_env: str = "OPENAI_API_KEY",
) -> APIGenerator | TinkerGenerator:
    """Build a generator from a matchup agent config block.

    Supports comma-separated API keys in the env var for automatic
    rotation when one key's quota is exhausted. Detects tinker:// model
    paths and uses the Tinker SDK directly.
    """
    model = agent_cfg["model"]
    if model.startswith("tinker://"):
        if model not in _tinker_cache:
            _tinker_cache[model] = TinkerGenerator(
                model_path=model,
                max_tokens=agent_cfg.get("max_tokens", 512),
            )
        return _tinker_cache[model]

    api_key_env = agent_cfg.get("api_key_env", default_api_key_env)
    raw = os.getenv(api_key_env, "")
    if not raw:
        raise ValueError(f"{api_key_env} env var required")
    api_keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not api_keys:
        raise ValueError(f"{api_key_env} env var has no valid keys")
    if len(api_keys) > 1:
        logger.info(f"{api_key_env}: loaded {len(api_keys)} API keys for rotation")
    base_url = agent_cfg.get("base_url", default_base_url)
    return APIGenerator(
        model=model,
        base_url=base_url,
        api_keys=api_keys,
        max_tokens=agent_cfg.get("max_tokens", 512),
        thinking=agent_cfg.get("thinking"),
        reasoning=agent_cfg.get("reasoning"),
        thinking_token_budget=agent_cfg.get("thinking_token_budget"),
    )
