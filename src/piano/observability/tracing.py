"""Lightweight distributed tracing system compatible with OpenTelemetry concepts.

Provides Span, Tracer, and SpanExporter abstractions for tracing agent
processing flows without depending on the opentelemetry library itself.

Typical trace structure for an agent tick::

    agent.tick
    +-- module.action_awareness.tick
    +-- module.goal_generation.tick
    |   +-- llm.complete
    +-- cc.compress
    +-- cc.decide
    |   +-- llm.complete
    +-- cc.broadcast

Reference: docs/implementation/08-infrastructure.md
"""

from __future__ import annotations

__all__ = [
    "InMemoryExporter",
    "LogExporter",
    "Span",
    "SpanEvent",
    "SpanExporter",
    "SpanStatus",
    "TraceContext",
    "Tracer",
]

import asyncio
import contextvars
import functools
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from enum import StrEnum
from typing import Any, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# Context variable for propagating the current span across async boundaries.
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_current_span", default=None
)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


class SpanStatus(StrEnum):
    """Status of a completed span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class TraceContext(BaseModel):
    """Propagation context identifying a span within a trace."""

    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    span_id: str = Field(default_factory=lambda: uuid4().hex)
    parent_span_id: str | None = None


class SpanEvent(BaseModel):
    """An event recorded within a span's lifetime."""

    name: str
    timestamp: float = Field(default_factory=time.time)
    attributes: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    """A single unit of work within a trace.

    Spans form a tree: each span may have a parent and children.
    Use :meth:`start` / :meth:`end` or the context-manager helpers on
    :class:`Tracer` to manage the span lifecycle.
    """

    name: str
    context: TraceContext = Field(default_factory=TraceContext)
    start_time: float = 0.0
    end_time: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: SpanStatus = SpanStatus.UNSET
    events: list[SpanEvent] = Field(default_factory=list)

    def start(self) -> Span:
        """Mark the span as started, recording the current monotonic time.

        Returns:
            self, for fluent chaining.
        """
        self.start_time = time.monotonic()
        return self

    def end(self, status: SpanStatus | None = None) -> Span:
        """Mark the span as ended, recording the current monotonic time.

        Args:
            status: Optional final status to set. If ``None`` and the
                current status is ``UNSET``, it is promoted to ``OK``.

        Returns:
            self, for fluent chaining.
        """
        self.end_time = time.monotonic()
        if status is not None:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
        return self

    @property
    def duration_ms(self) -> float | None:
        """Return the span duration in milliseconds, or ``None`` if incomplete."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a single attribute on the span."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Record an event within this span's lifetime."""
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))

    def set_status(self, status: SpanStatus, description: str | None = None) -> None:
        """Set the span status explicitly.

        Args:
            status: The new status.
            description: Optional human-readable description (attached
                as the ``status_description`` attribute when non-empty).
        """
        self.status = status
        if description:
            self.attributes["status_description"] = description


# ---------------------------------------------------------------------------
# Exporter protocol and built-in implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class SpanExporter(Protocol):
    """Protocol that all span exporters must satisfy."""

    def export(self, spans: list[Span]) -> None:
        """Export a batch of completed spans."""
        ...


class InMemoryExporter:
    """Test-friendly exporter that accumulates spans in a list.

    Example::

        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        with tracer.start_span("my_op") as span:
            ...
        assert len(exporter.spans) == 1
    """

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def export(self, spans: list[Span]) -> None:
        """Append *spans* to the internal list."""
        self.spans.extend(spans)

    def clear(self) -> None:
        """Remove all collected spans."""
        self.spans.clear()


class LogExporter:
    """Exporter that logs span information via structlog.

    Useful for development and debugging -- each completed span is
    emitted as a structured log entry at ``info`` level.
    """

    def __init__(self, log_level: str = "info") -> None:
        self._log_level = log_level

    def export(self, spans: list[Span]) -> None:
        """Log each span using structlog."""
        log_fn = getattr(logger, self._log_level, logger.info)
        for span in spans:
            log_fn(
                "span_completed",
                span_name=span.name,
                trace_id=span.context.trace_id,
                span_id=span.context.span_id,
                parent_span_id=span.context.parent_span_id,
                duration_ms=span.duration_ms,
                status=span.status.value,
                attributes=span.attributes,
                event_count=len(span.events),
            )


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class Tracer:
    """Factory for creating and managing spans.

    The tracer keeps track of the current active span via
    :mod:`contextvars` so that child spans automatically pick up
    their parent, even across ``await`` boundaries.

    Args:
        service_name: Logical service name attached to every span.
        exporter: An optional :class:`SpanExporter` that receives
            each span when it ends.
        enabled: When ``False``, all tracing operations become no-ops
            and :meth:`start_span` returns lightweight dummy spans.
    """

    def __init__(
        self,
        *,
        service_name: str = "piano",
        exporter: SpanExporter | None = None,
        enabled: bool = True,
    ) -> None:
        self.service_name = service_name
        self._exporter = exporter
        self._enabled = enabled

    # -- core API ----------------------------------------------------------

    def create_span(
        self,
        name: str,
        *,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Create and start a new span.

        If *parent* is ``None``, the tracer looks up the current span
        from the context variable.  If there is still no parent, a new
        root trace is created.

        Args:
            name: Human-readable span name (e.g. ``"cc.decide"``).
            parent: Explicit parent span.  When omitted, the current
                context span is used.
            attributes: Initial attributes to attach to the span.

        Returns:
            A started :class:`Span`.
        """
        if not self._enabled:
            return Span(name=name).start()

        # Resolve parent from explicit arg or context variable.
        effective_parent = parent or _current_span.get()

        if effective_parent is not None:
            ctx = TraceContext(
                trace_id=effective_parent.context.trace_id,
                parent_span_id=effective_parent.context.span_id,
            )
        else:
            ctx = TraceContext()

        span = Span(
            name=name,
            context=ctx,
            attributes={
                "service.name": self.service_name,
                **(attributes or {}),
            },
        )
        span.start()
        return span

    def _finish_span(self, span: Span, status: SpanStatus | None = None) -> None:
        """End *span* and export it if an exporter is configured."""
        span.end(status=status)
        if self._exporter is not None:
            self._exporter.export([span])

    # -- context-manager helpers -------------------------------------------

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Context manager that creates, activates, and auto-closes a span.

        The span is set as the current span for the duration of the
        ``with`` block and automatically ended on exit.

        Example::

            with tracer.start_span("cc.decide") as span:
                span.set_attribute("agent_id", "agent-001")
                ...
        """
        span = self.create_span(name, parent=parent, attributes=attributes)
        token = _current_span.set(span)
        try:
            yield span
        except Exception as exc:
            span.set_status(SpanStatus.ERROR, str(exc))
            raise
        finally:
            _current_span.reset(token)
            if span.status == SpanStatus.UNSET:
                self._finish_span(span, status=SpanStatus.OK)
            else:
                self._finish_span(span)

    @asynccontextmanager
    async def start_async_span(
        self,
        name: str,
        *,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        """Async context manager variant of :meth:`start_span`.

        Behaviour is identical, but can be used in ``async with`` blocks.
        """
        span = self.create_span(name, parent=parent, attributes=attributes)
        token = _current_span.set(span)
        try:
            yield span
        except Exception as exc:
            span.set_status(SpanStatus.ERROR, str(exc))
            raise
        finally:
            _current_span.reset(token)
            if span.status == SpanStatus.UNSET:
                self._finish_span(span, status=SpanStatus.OK)
            else:
                self._finish_span(span)

    # -- decorator ---------------------------------------------------------

    def trace(
        self,
        name: str | None = None,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Callable[[F], F]:
        """Decorator that wraps a function or coroutine in a span.

        Args:
            name: Span name.  Defaults to the qualified function name.
            attributes: Extra attributes to attach to every invocation.

        Example::

            @tracer.trace("llm.complete")
            async def complete(request):
                ...
        """

        def decorator(fn: F) -> F:
            span_name = name or fn.__qualname__

            if _is_coroutine_function(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    # Error handling is done by start_async_span context manager.
                    async with self.start_async_span(span_name, attributes=attributes):
                        return await fn(*args, **kwargs)

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Error handling is done by start_span context manager.
                with self.start_span(span_name, attributes=attributes):
                    return fn(*args, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

        return decorator

    # -- utility -----------------------------------------------------------

    @property
    def current_span(self) -> Span | None:
        """Return the currently active span, or ``None``."""
        return _current_span.get()

    @property
    def enabled(self) -> bool:
        """Whether tracing is active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_coroutine_function(fn: Any) -> bool:
    """Return ``True`` if *fn* is an async function (works with wrapped fns)."""
    return asyncio.iscoroutinefunction(fn)
