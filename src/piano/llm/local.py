"""Local LLM provider for vLLM and Ollama.

Provides LLM providers that connect to local inference servers (vLLM or Ollama)
for Tier 2/3 modules, with automatic fallback to remote providers on failure.

Reference: docs/implementation/02-llm-integration.md (Section 1.3, 3.3)
"""

from __future__ import annotations

import time
from contextlib import suppress
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

import httpx
import structlog

from piano.core.types import LLMRequest, LLMResponse

logger = structlog.get_logger(__name__)

# --- Local Model Status ---


class LocalModelStatus(StrEnum):
    """Status of a local model."""

    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNKNOWN = "unknown"


# --- Local Model Info ---


class LocalModelInfo:
    """Information about a local model."""

    def __init__(
        self,
        model_name: str,
        status: LocalModelStatus,
        loaded_at: datetime | None = None,
        memory_mb: float = 0.0,
        parameters: str = "",
    ) -> None:
        self.model_name = model_name
        self.status = status
        self.loaded_at = loaded_at
        self.memory_mb = memory_mb
        self.parameters = parameters


# --- LLM Provider Protocol (for type checking) ---


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request and return the response."""
        ...


# --- Ollama Provider ---


class OllamaProvider:
    """LLM provider backed by Ollama.

    Connects to a local Ollama server via HTTP and supports automatic
    fallback to a remote provider if the local server is unavailable.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2",
        timeout_seconds: float = 30.0,
        fallback_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            base_url: Base URL of the Ollama server.
            model_name: Name of the model to use.
            timeout_seconds: Timeout for API requests in seconds.
            fallback_provider: Optional fallback provider if Ollama is unavailable.
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.fallback_provider = fallback_provider
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self._error_count = 0
        self._success_count = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to Ollama.

        Args:
            request: The LLM request to execute.

        Returns:
            An LLMResponse with content, model, and latency info.
        """
        model = request.model or self.model_name

        # Build prompt
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        if request.json_mode:
            payload["format"] = "json"

        try:
            start = time.monotonic()
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            latency_ms = (time.monotonic() - start) * 1000

            data = response.json()
            content = data.get("message", {}).get("content", "")

            # Extract token usage if available
            usage = {}
            if "prompt_eval_count" in data:
                usage["prompt_tokens"] = data.get("prompt_eval_count", 0)
            if "eval_count" in data:
                usage["completion_tokens"] = data.get("eval_count", 0)
            if usage:
                usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                    "completion_tokens", 0
                )

            self._success_count += 1
            logger.debug(
                "ollama_completion_success",
                model=model,
                latency_ms=latency_ms,
                usage=usage,
            )

            return LLMResponse(
                content=content,
                model=model,
                usage=usage,
                latency_ms=latency_ms,
                cost_usd=0.0,  # Local models have no API cost
            )

        except Exception as exc:
            self._error_count += 1
            logger.warning(
                "ollama_completion_failed",
                model=model,
                error=str(exc),
                fallback_available=self.fallback_provider is not None,
            )

            # Try fallback if available
            if self.fallback_provider is not None:
                logger.info("ollama_fallback_to_remote", model=model)
                return await self.fallback_provider.complete(request)

            # No fallback, re-raise
            raise

    async def check_health(self) -> bool:
        """Check if the Ollama server is healthy.

        Returns:
            True if the server is reachable, False otherwise.
        """
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def get_model_status(self) -> LocalModelInfo:
        """Get the status of the current model.

        Returns:
            LocalModelInfo with model status information.
        """
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            models = data.get("models", [])
            for model_data in models:
                if model_data.get("name", "").startswith(self.model_name):
                    # Model is loaded and ready
                    size_bytes = model_data.get("size", 0)
                    memory_mb = size_bytes / (1024 * 1024)

                    # Try to parse modified_at as datetime
                    loaded_at = None
                    if "modified_at" in model_data:
                        with suppress(Exception):
                            loaded_at = datetime.fromisoformat(
                                model_data["modified_at"].replace("Z", "+00:00")
                            )

                    return LocalModelInfo(
                        model_name=self.model_name,
                        status=LocalModelStatus.READY,
                        loaded_at=loaded_at,
                        memory_mb=memory_mb,
                        parameters=model_data.get("details", {}).get("parameter_size", ""),
                    )

            # Model not found in list
            return LocalModelInfo(
                model_name=self.model_name,
                status=LocalModelStatus.UNKNOWN,
            )

        except Exception as exc:
            logger.warning("ollama_model_status_failed", error=str(exc))
            return LocalModelInfo(
                model_name=self.model_name,
                status=LocalModelStatus.ERROR,
            )

    async def list_models(self) -> list[str]:
        """List all available models on the Ollama server.

        Returns:
            List of model names.
        """
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            models = data.get("models", [])
            return [model.get("name", "") for model in models if "name" in model]

        except Exception as exc:
            logger.warning("ollama_list_models_failed", error=str(exc))
            return []

    @property
    def error_count(self) -> int:
        """Return the total number of failed requests."""
        return self._error_count

    @property
    def success_count(self) -> int:
        """Return the total number of successful requests."""
        return self._success_count

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


# --- vLLM Provider ---


class VLLMProvider:
    """LLM provider backed by vLLM.

    Connects to a local vLLM server via OpenAI-compatible HTTP API and supports
    automatic fallback to a remote provider if the local server is unavailable.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000",
        model_name: str = "llama-3.3-70b",
        timeout_seconds: float = 30.0,
        fallback_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the vLLM provider.

        Args:
            base_url: Base URL of the vLLM server (OpenAI-compatible API).
            model_name: Name of the model to use.
            timeout_seconds: Timeout for API requests in seconds.
            fallback_provider: Optional fallback provider if vLLM is unavailable.
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.fallback_provider = fallback_provider
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self._error_count = 0
        self._success_count = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to vLLM.

        Args:
            request: The LLM request to execute.

        Returns:
            An LLMResponse with content, model, and latency info.
        """
        model = request.model or self.model_name

        # Build messages in OpenAI format
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            start = time.monotonic()
            response = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            latency_ms = (time.monotonic() - start) * 1000

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Extract token usage
            usage = {}
            if "usage" in data:
                usage_data = data["usage"]
                usage = {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                }

            self._success_count += 1
            logger.debug(
                "vllm_completion_success",
                model=model,
                latency_ms=latency_ms,
                usage=usage,
            )

            return LLMResponse(
                content=content,
                model=model,
                usage=usage,
                latency_ms=latency_ms,
                cost_usd=0.0,  # Local models have no API cost
            )

        except Exception as exc:
            self._error_count += 1
            logger.warning(
                "vllm_completion_failed",
                model=model,
                error=str(exc),
                fallback_available=self.fallback_provider is not None,
            )

            # Try fallback if available
            if self.fallback_provider is not None:
                logger.info("vllm_fallback_to_remote", model=model)
                return await self.fallback_provider.complete(request)

            # No fallback, re-raise
            raise

    async def check_health(self) -> bool:
        """Check if the vLLM server is healthy.

        Returns:
            True if the server is reachable, False otherwise.
        """
        try:
            response = await self._client.get(f"{self.base_url}/v1/models")
            return response.status_code == 200
        except Exception:
            return False

    async def get_model_status(self) -> LocalModelInfo:
        """Get the status of the current model.

        Returns:
            LocalModelInfo with model status information.
        """
        try:
            response = await self._client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            data = response.json()

            models = data.get("data", [])
            for model_data in models:
                if model_data.get("id", "") == self.model_name:
                    # Model is loaded and ready
                    created_timestamp = model_data.get("created")
                    loaded_at = None
                    if created_timestamp:
                        loaded_at = datetime.fromtimestamp(created_timestamp)

                    return LocalModelInfo(
                        model_name=self.model_name,
                        status=LocalModelStatus.READY,
                        loaded_at=loaded_at,
                        memory_mb=0.0,  # vLLM doesn't expose memory usage in API
                        parameters="",
                    )

            # Model not found in list
            return LocalModelInfo(
                model_name=self.model_name,
                status=LocalModelStatus.UNKNOWN,
            )

        except Exception as exc:
            logger.warning("vllm_model_status_failed", error=str(exc))
            return LocalModelInfo(
                model_name=self.model_name,
                status=LocalModelStatus.ERROR,
            )

    async def list_models(self) -> list[str]:
        """List all available models on the vLLM server.

        Returns:
            List of model names.
        """
        try:
            response = await self._client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            data = response.json()

            models = data.get("data", [])
            return [model.get("id", "") for model in models if "id" in model]

        except Exception as exc:
            logger.warning("vllm_list_models_failed", error=str(exc))
            return []

    @property
    def error_count(self) -> int:
        """Return the total number of failed requests."""
        return self._error_count

    @property
    def success_count(self) -> int:
        """Return the total number of successful requests."""
        return self._success_count

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


__all__ = [
    "LocalModelInfo",
    "LocalModelStatus",
    "OllamaProvider",
    "VLLMProvider",
]
