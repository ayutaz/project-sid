"""Tests for local LLM providers (Ollama and vLLM).

All tests mock httpx responses to avoid real server connections.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from piano.core.types import LLMRequest
from piano.llm.local import (
    LocalModelStatus,
    OllamaProvider,
    VLLMProvider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ollama_response(
    content: str = "Hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict:
    """Create a mock Ollama API response."""
    return {
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }


def _make_vllm_response(
    content: str = "Hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
) -> dict:
    """Create a mock vLLM (OpenAI-compatible) API response."""
    return {
        "id": "chat-123",
        "object": "chat.completion",
        "created": 1704067200,
        "model": "llama-3.3-70b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


# ---------------------------------------------------------------------------
# OllamaProvider Tests
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    @pytest.mark.asyncio
    async def test_successful_completion(self) -> None:
        """Should successfully complete a request via Ollama API."""
        provider = OllamaProvider(base_url="http://localhost:11434", model_name="llama3.2")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _make_ollama_response("world", 10, 5)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await provider.complete(LLMRequest(prompt="hello"))

        assert result.content == "world"
        assert result.model == "llama3.2"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15
        assert result.latency_ms > 0
        assert result.cost_usd == 0.0  # Local models have no cost

    @pytest.mark.asyncio
    async def test_uses_system_prompt(self) -> None:
        """Should include system prompt in messages."""
        provider = OllamaProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _make_ollama_response("ok")

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await provider.complete(LLMRequest(prompt="hello", system_prompt="you are helpful"))

        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]
        assert len(payload["messages"]) == 2
        assert payload["messages"][0] == {"role": "system", "content": "you are helpful"}
        assert payload["messages"][1] == {"role": "user", "content": "hello"}

    @pytest.mark.asyncio
    async def test_json_mode_sets_format(self) -> None:
        """Should set format to json when json_mode is True."""
        provider = OllamaProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _make_ollama_response('{"key": "value"}')

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await provider.complete(LLMRequest(prompt="json", json_mode=True))

        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]
        assert payload["format"] == "json"

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens_passed(self) -> None:
        """Should pass temperature and max_tokens to Ollama."""
        provider = OllamaProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _make_ollama_response()

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await provider.complete(LLMRequest(prompt="hi", temperature=0.7, max_tokens=512))

        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]
        assert payload["options"]["temperature"] == 0.7
        assert payload["options"]["num_predict"] == 512

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self) -> None:
        """Should fallback to remote provider on local failure."""
        mock_fallback = AsyncMock()
        mock_fallback.complete = AsyncMock(
            return_value=MagicMock(
                content="fallback response",
                model="gpt-4o",
                usage={},
                latency_ms=100,
                cost_usd=0.01,
            )
        )

        provider = OllamaProvider(fallback_provider=mock_fallback)

        with patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            result = await provider.complete(LLMRequest(prompt="hello"))

        assert result.content == "fallback response"
        assert result.model == "gpt-4o"
        mock_fallback.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_without_fallback(self) -> None:
        """Should raise exception if no fallback is available."""
        provider = OllamaProvider(fallback_provider=None)

        with (
            patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("connection refused"),
            ),
            pytest.raises(httpx.ConnectError),
        ):
            await provider.complete(LLMRequest(prompt="hello"))

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        """Should return True when server is healthy."""
        provider = OllamaProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200

        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await provider.check_health()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self) -> None:
        """Should return False when server is unreachable."""
        provider = OllamaProvider()

        with patch.object(
            provider._client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            result = await provider.check_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_model_status_ready(self) -> None:
        """Should return READY status when model is loaded."""
        provider = OllamaProvider(model_name="llama3.2")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3.2:latest",
                    "size": 2147483648,  # 2GB
                    "modified_at": "2024-01-01T00:00:00Z",
                    "details": {"parameter_size": "3B"},
                }
            ]
        }

        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            status = await provider.get_model_status()

        assert status.model_name == "llama3.2"
        assert status.status == LocalModelStatus.READY
        assert status.memory_mb == pytest.approx(2048.0)
        assert status.parameters == "3B"
        assert isinstance(status.loaded_at, datetime)

    @pytest.mark.asyncio
    async def test_get_model_status_unknown(self) -> None:
        """Should return UNKNOWN status when model is not found."""
        provider = OllamaProvider(model_name="nonexistent")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}

        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            status = await provider.get_model_status()

        assert status.model_name == "nonexistent"
        assert status.status == LocalModelStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_get_model_status_error(self) -> None:
        """Should return ERROR status on API failure."""
        provider = OllamaProvider()

        with patch.object(
            provider._client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            status = await provider.get_model_status()

        assert status.status == LocalModelStatus.ERROR

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        """Should list all available models."""
        provider = OllamaProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "gemma2:9b"},
                {"name": "mistral:7b"},
            ]
        }

        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            models = await provider.list_models()

        assert models == ["llama3.2:latest", "gemma2:9b", "mistral:7b"]

    @pytest.mark.asyncio
    async def test_list_models_on_error(self) -> None:
        """Should return empty list on API failure."""
        provider = OllamaProvider()

        with patch.object(
            provider._client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            models = await provider.list_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_error_count_tracking(self) -> None:
        """Should track error and success counts."""
        provider = OllamaProvider(fallback_provider=None)

        assert provider.error_count == 0
        assert provider.success_count == 0

        # Successful call
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _make_ollama_response()

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await provider.complete(LLMRequest(prompt="hello"))

        assert provider.error_count == 0
        assert provider.success_count == 1

        # Failed call
        with (
            patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("connection refused"),
            ),
            pytest.raises(httpx.ConnectError),
        ):
            await provider.complete(LLMRequest(prompt="hello"))

        assert provider.error_count == 1
        assert provider.success_count == 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        """Should handle timeout errors."""
        provider = OllamaProvider(timeout_seconds=1.0, fallback_provider=None)

        with (
            patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.TimeoutException("timeout"),
            ),
            pytest.raises(httpx.TimeoutException),
        ):
            await provider.complete(LLMRequest(prompt="hello"))

        assert provider.error_count == 1

    @pytest.mark.asyncio
    async def test_response_without_usage(self) -> None:
        """Should handle responses without token usage."""
        provider = OllamaProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "hello"},
            "done": True,
        }

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await provider.complete(LLMRequest(prompt="hello"))

        assert result.content == "hello"
        assert result.usage == {}


# ---------------------------------------------------------------------------
# VLLMProvider Tests
# ---------------------------------------------------------------------------


class TestVLLMProvider:
    """Tests for VLLMProvider."""

    @pytest.mark.asyncio
    async def test_successful_completion(self) -> None:
        """Should successfully complete a request via vLLM API."""
        provider = VLLMProvider(base_url="http://localhost:8000", model_name="llama-3.3-70b")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _make_vllm_response("world", 10, 5, 15)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await provider.complete(LLMRequest(prompt="hello"))

        assert result.content == "world"
        assert result.model == "llama-3.3-70b"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15
        assert result.latency_ms > 0
        assert result.cost_usd == 0.0  # Local models have no cost

    @pytest.mark.asyncio
    async def test_uses_system_prompt(self) -> None:
        """Should include system prompt in messages."""
        provider = VLLMProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _make_vllm_response("ok")

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await provider.complete(LLMRequest(prompt="hello", system_prompt="you are helpful"))

        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]
        assert len(payload["messages"]) == 2
        assert payload["messages"][0] == {"role": "system", "content": "you are helpful"}
        assert payload["messages"][1] == {"role": "user", "content": "hello"}

    @pytest.mark.asyncio
    async def test_json_mode_sets_response_format(self) -> None:
        """Should set response_format when json_mode is True."""
        provider = VLLMProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _make_vllm_response('{"key": "value"}')

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await provider.complete(LLMRequest(prompt="json", json_mode=True))

        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]
        assert payload["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens_passed(self) -> None:
        """Should pass temperature and max_tokens to vLLM."""
        provider = VLLMProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _make_vllm_response()

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await provider.complete(LLMRequest(prompt="hi", temperature=0.7, max_tokens=512))

        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self) -> None:
        """Should fallback to remote provider on local failure."""
        mock_fallback = AsyncMock()
        mock_fallback.complete = AsyncMock(
            return_value=MagicMock(
                content="fallback response",
                model="gpt-4o",
                usage={},
                latency_ms=100,
                cost_usd=0.01,
            )
        )

        provider = VLLMProvider(fallback_provider=mock_fallback)

        with patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            result = await provider.complete(LLMRequest(prompt="hello"))

        assert result.content == "fallback response"
        assert result.model == "gpt-4o"
        mock_fallback.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_without_fallback(self) -> None:
        """Should raise exception if no fallback is available."""
        provider = VLLMProvider(fallback_provider=None)

        with (
            patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("connection refused"),
            ),
            pytest.raises(httpx.ConnectError),
        ):
            await provider.complete(LLMRequest(prompt="hello"))

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        """Should return True when server is healthy."""
        provider = VLLMProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200

        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await provider.check_health()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self) -> None:
        """Should return False when server is unreachable."""
        provider = VLLMProvider()

        with patch.object(
            provider._client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            result = await provider.check_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_model_status_ready(self) -> None:
        """Should return READY status when model is loaded."""
        provider = VLLMProvider(model_name="llama-3.3-70b")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "llama-3.3-70b",
                    "object": "model",
                    "created": 1704067200,
                }
            ]
        }

        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            status = await provider.get_model_status()

        assert status.model_name == "llama-3.3-70b"
        assert status.status == LocalModelStatus.READY
        assert isinstance(status.loaded_at, datetime)

    @pytest.mark.asyncio
    async def test_get_model_status_unknown(self) -> None:
        """Should return UNKNOWN status when model is not found."""
        provider = VLLMProvider(model_name="nonexistent")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            status = await provider.get_model_status()

        assert status.model_name == "nonexistent"
        assert status.status == LocalModelStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_get_model_status_error(self) -> None:
        """Should return ERROR status on API failure."""
        provider = VLLMProvider()

        with patch.object(
            provider._client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            status = await provider.get_model_status()

        assert status.status == LocalModelStatus.ERROR

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        """Should list all available models."""
        provider = VLLMProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "llama-3.3-70b", "object": "model"},
                {"id": "mistral-7b", "object": "model"},
            ]
        }

        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            models = await provider.list_models()

        assert models == ["llama-3.3-70b", "mistral-7b"]

    @pytest.mark.asyncio
    async def test_list_models_on_error(self) -> None:
        """Should return empty list on API failure."""
        provider = VLLMProvider()

        with patch.object(
            provider._client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            models = await provider.list_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_error_count_tracking(self) -> None:
        """Should track error and success counts."""
        provider = VLLMProvider(fallback_provider=None)

        assert provider.error_count == 0
        assert provider.success_count == 0

        # Successful call
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _make_vllm_response()

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await provider.complete(LLMRequest(prompt="hello"))

        assert provider.error_count == 0
        assert provider.success_count == 1

        # Failed call
        with (
            patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("connection refused"),
            ),
            pytest.raises(httpx.ConnectError),
        ):
            await provider.complete(LLMRequest(prompt="hello"))

        assert provider.error_count == 1
        assert provider.success_count == 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        """Should handle timeout errors."""
        provider = VLLMProvider(timeout_seconds=1.0, fallback_provider=None)

        with (
            patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.TimeoutException("timeout"),
            ),
            pytest.raises(httpx.TimeoutException),
        ):
            await provider.complete(LLMRequest(prompt="hello"))

        assert provider.error_count == 1

    @pytest.mark.asyncio
    async def test_response_without_usage(self) -> None:
        """Should handle responses without token usage."""
        provider = VLLMProvider()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chat-123",
            "object": "chat.completion",
            "created": 1704067200,
            "model": "llama-3.3-70b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
        }

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await provider.complete(LLMRequest(prompt="hello"))

        assert result.content == "hello"
        assert result.usage == {}


# ---------------------------------------------------------------------------
# LLMProvider Protocol Conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Tests for LLMProvider Protocol conformance."""

    @pytest.mark.asyncio
    async def test_ollama_provider_is_llm_provider(self) -> None:
        """OllamaProvider should implement the LLMProvider protocol."""
        from piano.llm.provider import LLMProvider

        provider = OllamaProvider()
        assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_vllm_provider_is_llm_provider(self) -> None:
        """VLLMProvider should implement the LLMProvider protocol."""
        from piano.llm.provider import LLMProvider

        provider = VLLMProvider()
        assert isinstance(provider, LLMProvider)
