"""Chaos engineering fault injection simulators.

Provides simulators for testing system resilience against:
- Redis connection failures
- ZMQ bridge disconnections
- LLM provider errors and timeouts
- Agent process crashes
"""
from __future__ import annotations

import asyncio
import random

import structlog

logger = structlog.get_logger(__name__)


class RedisFailureSimulator:
    """Simulates Redis connection failures.

    Wraps a Redis client or SAS to inject failures.
    """

    def __init__(self, failure_rate: float = 0.3, delay_ms: float = 0) -> None:
        self.failure_rate = failure_rate
        self.delay_ms = delay_ms
        self._failure_count = 0
        self._call_count = 0

    def should_fail(self) -> bool:
        """Determine if the next operation should fail."""
        self._call_count += 1
        if random.random() < self.failure_rate:
            self._failure_count += 1
            return True
        return False

    async def maybe_delay(self) -> None:
        """Inject optional delay."""
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000.0)

    @property
    def stats(self) -> dict[str, int]:
        return {"calls": self._call_count, "failures": self._failure_count}


class BridgeFailureSimulator:
    """Simulates ZMQ bridge disconnection.

    Can wrap a BridgeClient to inject connection failures.
    """

    def __init__(self, disconnect_after: int = 5) -> None:
        self.disconnect_after = disconnect_after
        self._command_count = 0
        self._disconnected = False

    def check_and_maybe_disconnect(self) -> bool:
        """Check if bridge should be disconnected.

        Returns True if the bridge was disconnected.
        """
        self._command_count += 1
        if self._command_count >= self.disconnect_after and not self._disconnected:
            self._disconnected = True
            logger.warning("bridge_fault_injected", after_commands=self._command_count)
            return True
        return False

    @property
    def is_disconnected(self) -> bool:
        return self._disconnected

    def reset(self) -> None:
        """Reset the simulator state."""
        self._command_count = 0
        self._disconnected = False


class LLMFailureSimulator:
    """Simulates LLM provider failures.

    Can inject random errors, timeouts, or malformed responses.
    """

    def __init__(
        self,
        error_rate: float = 0.2,
        timeout_rate: float = 0.1,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.error_rate = error_rate
        self.timeout_rate = timeout_rate
        self.timeout_seconds = timeout_seconds
        self._call_count = 0
        self._error_count = 0
        self._timeout_count = 0

    async def maybe_fail(self) -> None:
        """Check if this call should fail.

        Returns None if no failure, raises on failure.
        """
        self._call_count += 1
        roll = random.random()

        if roll < self.error_rate:
            self._error_count += 1
            raise RuntimeError("LLM provider error (injected fault)")

        if roll < self.error_rate + self.timeout_rate:
            self._timeout_count += 1
            raise TimeoutError("LLM request timed out (injected fault)")

    @property
    def stats(self) -> dict[str, int]:
        return {
            "calls": self._call_count,
            "errors": self._error_count,
            "timeouts": self._timeout_count,
        }


class AgentCrashSimulator:
    """Simulates agent process crashes.

    Provides mechanisms to simulate sudden agent termination
    for testing checkpoint/restore recovery.
    """

    def __init__(self, crash_at_tick: int = 5) -> None:
        self.crash_at_tick = crash_at_tick
        self._tick_count = 0
        self._crashed = False

    def tick(self) -> bool:
        """Process a tick. Returns True if agent should crash.

        Raises RuntimeError to simulate crash.
        """
        self._tick_count += 1
        if self._tick_count >= self.crash_at_tick and not self._crashed:
            self._crashed = True
            logger.warning("agent_crash_injected", at_tick=self._tick_count)
            return True
        return False

    @property
    def has_crashed(self) -> bool:
        return self._crashed

    def reset(self) -> None:
        """Reset the simulator."""
        self._tick_count = 0
        self._crashed = False
