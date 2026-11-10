"""The Azure OpenAI chat model used by the programming, benchmarking and reporting stages."""

from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI

from src.settings.config import LLMConfig

from .authentication import token_provider

CONNECT_TIMEOUT_SECONDS = 30.0
AZURE_BASE_URL = "https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/"


def build_chat_model(llm_config: LLMConfig) -> ChatOpenAI:
    # The Responses API, not chat completions: gpt-6.1-sol rejects function tools together
    # with a reasoning effort on /v1/chat/completions (HTTP 400), and the study needs both.
    return ChatOpenAI(
        use_responses_api=True,
        model=llm_config.model,
        temperature=llm_config.temperature,
        base_url=AZURE_BASE_URL,
        api_key=token_provider,
        # A hung connection should fail in seconds and be retried; only a reply that is genuinely
        # slow (a long reasoning step) may take up to llm_config.timeout.
        timeout=httpx.Timeout(llm_config.timeout, connect=CONNECT_TIMEOUT_SECONDS),
        max_retries=llm_config.max_retries,
        reasoning_effort=llm_config.reasoning_effort,
    )
