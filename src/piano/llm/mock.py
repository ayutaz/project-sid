"""Mock LLM provider for testing.

Provides a deterministic provider that returns pre-configured responses
based on prompt pattern matching, with call history tracking for assertions.
"""

from __future__ import annotations

from piano.core.types import LLMRequest, LLMResponse

_DEMO_RESPONSES: list[str] = [
    (
        '{"action": "explore", "action_params": '
        '{"direction": "north", "distance": 20}, '
        '"speaking": "", "reasoning": "Exploring the area"}'
    ),
    (
        '{"action": "explore", "action_params": '
        '{"direction": "east", "distance": 15}, '
        '"speaking": "Let me check over there", '
        '"reasoning": "Curious about surroundings"}'
    ),
    (
        '{"action": "chat", "action_params": '
        '{"message": "Hello everyone!"}, '
        '"speaking": "Hello everyone!", '
        '"reasoning": "Being social"}'
    ),
    (
        '{"action": "mine", "action_params": '
        '{"x": 10, "y": 62, "z": 10}, '
        '"speaking": "", "reasoning": "Mining nearby block"}'
    ),
    (
        '{"action": "look", "action_params": '
        '{"x": 0, "y": 64, "z": 0}, '
        '"speaking": "", "reasoning": "Looking around"}'
    ),
    (
        '{"action": "idle", "action_params": {}, '
        '"speaking": "", '
        '"reasoning": "Taking a moment to observe"}'
    ),
    (
        '{"action": "chat", "action_params": '
        '{"message": "What should we do?"}, '
        '"speaking": "What should we do?", '
        '"reasoning": "Engaging with others"}'
    ),
]


class MockLLMProvider:
    """A mock LLM provider for unit and integration tests.

    Supports pattern-based response matching, a default fallback response,
    and call history tracking.
    """

    def __init__(self) -> None:
        self._responses: list[tuple[str, str]] = []
        self._default_response: str = '{"action": "idle"}'
        self.call_history: list[LLMRequest] = []
        self._demo_mode: bool = False
        self._demo_responses: list[str] = _DEMO_RESPONSES
        self._demo_index: int = 0

    @classmethod
    def create_demo_provider(cls) -> MockLLMProvider:
        """Create a MockLLMProvider with diverse demo responses for E2E testing."""
        provider = cls()
        provider._demo_mode = True
        return provider

    def add_response(self, prompt_pattern: str, response: str) -> None:
        """Register a response for prompts containing ``prompt_pattern``.

        Args:
            prompt_pattern: Substring to match against the prompt text.
            response: The content to return when the pattern matches.
        """
        self._responses.append((prompt_pattern, response))

    def set_default_response(self, response: str) -> None:
        """Set the fallback response used when no pattern matches.

        Args:
            response: The content to return as default.
        """
        self._default_response = response

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a matching pre-configured response.

        In demo mode, cycles through diverse demo responses. Otherwise,
        patterns are checked in registration order; the first match wins.
        If no pattern matches, the default response is returned.

        Args:
            request: The LLM request.

        Returns:
            An LLMResponse with zero latency and cost.
        """
        self.call_history.append(request)

        if self._demo_mode:
            idx = self._demo_index % len(self._demo_responses)
            self._demo_index += 1
            return LLMResponse(
                content=self._demo_responses[idx],
                model="mock-demo",
                latency_ms=0.0,
                cost_usd=0.0,
            )

        for pattern, response_text in self._responses:
            if pattern in request.prompt:
                return LLMResponse(
                    content=response_text,
                    model="mock",
                    latency_ms=0.0,
                    cost_usd=0.0,
                )

        return LLMResponse(
            content=self._default_response,
            model="mock",
            latency_ms=0.0,
            cost_usd=0.0,
        )

    def assert_called_with(self, pattern: str) -> None:
        """Assert that at least one call contained ``pattern`` in its prompt.

        Args:
            pattern: Substring expected in at least one request prompt.

        Raises:
            AssertionError: If no matching call was found.
        """
        for req in self.call_history:
            if pattern in req.prompt:
                return
        prompts = [r.prompt[:80] for r in self.call_history]
        raise AssertionError(
            f"No call with pattern {pattern!r} found. "
            f"Call history ({len(self.call_history)} calls): {prompts}"
        )
