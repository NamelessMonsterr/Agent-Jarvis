"""
BaseProvider — all LLM provider adapters implement this interface.
The router calls call() and gets back a normalized dict.
No provider details leak into the router itself.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderRequest:
    model_name: str
    system: str
    messages: list[dict]
    max_tokens: int = 1500
    temperature: float = 0.7
    json_mode: bool = False


@dataclass
class ProviderResponse:
    content: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    provider_latency_ms: int = 0   # HTTP round-trip time for this specific provider call


class BaseProvider(ABC):
    """All provider adapters inherit this. Router only knows this interface."""

    @abstractmethod
    async def call(self, request: ProviderRequest) -> ProviderResponse:
        """Make the API call and return a normalized response."""
        ...
