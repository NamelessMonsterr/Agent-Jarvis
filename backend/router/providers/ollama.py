"""
OllamaProvider — local model via Ollama.
Different API schema from OpenAI — requires its own adapter.
"""
import httpx
import time
from router.providers.base import BaseProvider, ProviderRequest, ProviderResponse
from core.logger import get_logger

log = get_logger(__name__)


class OllamaProvider(BaseProvider):
    """Adapter for locally-running Ollama models."""

    def __init__(self, timeout_s: int = 30):
        self.api_url = "http://localhost:11434/api/chat"
        self.timeout_s = timeout_s

    async def call(self, request: ProviderRequest) -> ProviderResponse:
        messages = [{"role": "system", "content": request.system}] + request.messages

        payload = {
            "model": request.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": request.temperature},
        }

        t0 = time.time()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            try:
                resp = await client.post(self.api_url, json=payload)
            except httpx.ConnectError:
                raise Exception("server_error: Ollama not running. Install from ollama.ai and run: ollama pull gemma:7b")
        provider_latency_ms = int((time.time() - t0) * 1000)
        log.debug(f"Ollama responded in {provider_latency_ms}ms")

        data = resp.json()
        response_content = data.get("message", {}).get("content", "")

        return ProviderResponse(
            content=response_content,
            finish_reason="stop",
            input_tokens=0,
            output_tokens=0,
            provider_latency_ms=provider_latency_ms,
        )
