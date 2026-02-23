"""Mock LLM provider for testing.

Provides a deterministic provider that returns pre-configured responses
based on prompt pattern matching, with call history tracking for assertions.
"""

from __future__ import annotations

from piano.core.types import LLMRequest, LLMResponse


class MockLLMProvider:
    """A mock LLM provider for unit and integration tests.

    Supports pattern-based response matching, a default fallback response,
    and call history tracking.
    """

    def __init__(self) -> None:
        self._responses: list[tuple[str, str]] = []
        self._default_response: str = '{"action": "idle"}'
        self.call_history: list[LLMRequest] = []

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

        Patterns are checked in registration order; the first match wins.
        If no pattern matches, the default response is returned.

        Args:
            request: The LLM request.

        Returns:
            An LLMResponse with zero latency and cost.
        """
        self.call_history.append(request)

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
