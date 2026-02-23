"""LLM abstraction layer.

Provider interface, mock, caching, gateway, local, tiering,
prompt cache, and multi-provider routing.
"""

from piano.llm.cache import CachedLLMProvider, LLMCache
from piano.llm.gateway import CircuitBreaker, LLMGateway, QueuedRequest, RequestPriority
from piano.llm.local import LocalModelInfo, LocalModelStatus, OllamaProvider, VLLMProvider
from piano.llm.mock import MockLLMProvider
from piano.llm.multi_provider import (
    AllProvidersFailedError,
    MultiProviderRouter,
    NoProvidersError,
    ProviderConfig,
    ProviderStats,
    RoutingStrategy,
)
from piano.llm.prompt_cache import (
    CacheStats,
    PrefixCacheOptimizer,
    PromptCacheManager,
    RedisCacheBackend,
    SemanticCache,
)
from piano.llm.provider import DEFAULT_MODELS, LiteLLMProvider, LLMProvider
from piano.llm.tiering import FallbackChain, ModelConfig, ModelRegistry, ModelRouter

__all__ = [
    "DEFAULT_MODELS",
    "AllProvidersFailedError",
    "CacheStats",
    "CachedLLMProvider",
    "CircuitBreaker",
    "FallbackChain",
    "LLMCache",
    "LLMGateway",
    "LLMProvider",
    "LiteLLMProvider",
    "LocalModelInfo",
    "LocalModelStatus",
    "MockLLMProvider",
    "ModelConfig",
    "ModelRegistry",
    "ModelRouter",
    "MultiProviderRouter",
    "NoProvidersError",
    "OllamaProvider",
    "PrefixCacheOptimizer",
    "PromptCacheManager",
    "ProviderConfig",
    "ProviderStats",
    "QueuedRequest",
    "RedisCacheBackend",
    "RequestPriority",
    "RoutingStrategy",
    "SemanticCache",
    "VLLMProvider",
]
