"""
OpenAICompatibleProvider — handles any provider that mirrors the OpenAI chat API.
This covers: OpenAI, DeepSeek, Qwen, Grok (they all use the same schema).
"""
import httpx
import time
from core.logger import get_logger
from router.providers.base import BaseProvider, ProviderRequest, ProviderResponse

log = get_logger(__name__)


class OpenAICompatibleProvider(BaseProvider):
    """
    Single adapter for all OpenAI-schema-compatible providers.
    DeepSeek, Qwen, and Grok intentionally mirror OpenAI's API —
    so one adapter handles all of them.
    """

    def __init__(self, api_url: str, api_key: str, timeout_s: int = 15):
        self.api_url = api_url
        self.api_key = api_key
        self.timeout_s = timeout_s

    async def call(self, request: ProviderRequest) -> ProviderResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [{"role": "system", "content": request.system}] + request.messages

        payload = {
            "model": request.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        # OpenAI supports native JSON mode — others use prompt engineering
        if request.json_mode and "openai.com" in self.api_url:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.time()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(self.api_url, headers=headers, json=payload)
        provider_latency_ms = int((time.time() - t0) * 1000)

        self._raise_for_status(resp)

        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        log.debug(f"{self.api_url.split('/')[2]} responded in {provider_latency_ms}ms")

        return ProviderResponse(
            content=choice["message"]["content"] or "",
            finish_reason=choice.get("finish_reason", "stop"),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            provider_latency_ms=provider_latency_ms,
        )

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 429:
            raise Exception("rate_limit_exceeded")
        if resp.status_code in (401, 403):
            raise Exception("invalid_api_key")
        if resp.status_code >= 500:
            raise Exception("server_error")
        if resp.status_code != 200:
            raise Exception(f"http_error_{resp.status_code}")
