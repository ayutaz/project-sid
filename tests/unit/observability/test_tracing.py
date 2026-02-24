"""Tests for the lightweight distributed tracing system."""

from __future__ import annotations

import asyncio
import time

import pytest

from piano.observability.tracing import (
    InMemoryExporter,
    LogExporter,
    Span,
    SpanEvent,
    SpanExporter,
    SpanStatus,
    TraceContext,
    Tracer,
)

# ---------------------------------------------------------------------------
# TraceContext tests
# ---------------------------------------------------------------------------


class TestTraceContext:
    """Tests for TraceContext."""

    def test_default_ids_are_generated(self) -> None:
        ctx = TraceContext()
        assert ctx.trace_id
        assert ctx.span_id
        assert ctx.parent_span_id is None

    def test_ids_are_unique(self) -> None:
        ctx1 = TraceContext()
        ctx2 = TraceContext()
        assert ctx1.trace_id != ctx2.trace_id
        assert ctx1.span_id != ctx2.span_id

    def test_explicit_ids(self) -> None:
        ctx = TraceContext(
            trace_id="trace-abc",
            span_id="span-123",
            parent_span_id="span-000",
        )
        assert ctx.trace_id == "trace-abc"
        assert ctx.span_id == "span-123"
        assert ctx.parent_span_id == "span-000"


# ---------------------------------------------------------------------------
# SpanEvent tests
# ---------------------------------------------------------------------------


class TestSpanEvent:
    """Tests for SpanEvent."""

    def test_basic_event(self) -> None:
        event = SpanEvent(name="exception")
        assert event.name == "exception"
        assert event.timestamp > 0
        assert event.attributes == {}

    def test_event_with_attributes(self) -> None:
        event = SpanEvent(name="log", attributes={"message": "hello"})
        assert event.attributes["message"] == "hello"

    def test_timestamp_is_wall_clock(self) -> None:
        """SpanEvent.timestamp should be a wall-clock (epoch) time, not monotonic."""
        before = time.time()
        event = SpanEvent(name="test")
        after = time.time()
        # Wall-clock timestamps are large epoch values (> 1e9).
        # Monotonic values are typically small (seconds since boot).
        assert event.timestamp >= before
        assert event.timestamp <= after
        # Sanity: epoch timestamps are > 1 billion
        assert event.timestamp > 1_000_000_000


# ---------------------------------------------------------------------------
# Span tests
# ---------------------------------------------------------------------------


class TestSpan:
    """Tests for the Span model."""

    def test_default_span(self) -> None:
        span = Span(name="test")
        assert span.name == "test"
        assert span.status == SpanStatus.UNSET
        assert span.start_time == 0.0
        assert span.end_time is None
        assert span.events == []
        assert span.attributes == {}

    def test_start_sets_time(self) -> None:
        span = Span(name="test")
        before = time.monotonic()
        span.start()
        after = time.monotonic()
        assert before <= span.start_time <= after

    def test_start_returns_self(self) -> None:
        span = Span(name="test")
        result = span.start()
        assert result is span

    def test_end_sets_time_and_status(self) -> None:
        span = Span(name="test").start()
        time.sleep(0.001)
        span.end()
        assert span.end_time is not None
        assert span.end_time >= span.start_time
        assert span.status == SpanStatus.OK

    def test_end_returns_self(self) -> None:
        span = Span(name="test").start()
        result = span.end()
        assert result is span

    def test_end_explicit_status(self) -> None:
        span = Span(name="test").start()
        span.end(status=SpanStatus.ERROR)
        assert span.status == SpanStatus.ERROR

    def test_end_preserves_existing_error_status(self) -> None:
        span = Span(name="test").start()
        span.set_status(SpanStatus.ERROR, "something went wrong")
        span.end()
        assert span.status == SpanStatus.ERROR

    def test_duration_ms(self) -> None:
        span = Span(name="test").start()
        time.sleep(0.01)
        span.end()
        duration = span.duration_ms
        assert duration is not None
        assert duration >= 10.0  # at least 10ms

    def test_duration_ms_none_when_not_ended(self) -> None:
        span = Span(name="test").start()
        assert span.duration_ms is None

    def test_set_attribute(self) -> None:
        span = Span(name="test")
        span.set_attribute("agent_id", "agent-001")
        assert span.attributes["agent_id"] == "agent-001"

    def test_add_event(self) -> None:
        span = Span(name="test")
        span.add_event("checkpoint", {"step": 1})
        assert len(span.events) == 1
        assert span.events[0].name == "checkpoint"
        assert span.events[0].attributes["step"] == 1

    def test_add_multiple_events(self) -> None:
        span = Span(name="test")
        span.add_event("start")
        span.add_event("middle")
        span.add_event("end")
        assert len(span.events) == 3
        assert [e.name for e in span.events] == ["start", "middle", "end"]

    def test_set_status_with_description(self) -> None:
        span = Span(name="test")
        span.set_status(SpanStatus.ERROR, "timeout exceeded")
        assert span.status == SpanStatus.ERROR
        assert span.attributes["status_description"] == "timeout exceeded"

    def test_set_status_without_description(self) -> None:
        span = Span(name="test")
        span.set_status(SpanStatus.OK)
        assert span.status == SpanStatus.OK
        assert "status_description" not in span.attributes


# ---------------------------------------------------------------------------
# SpanStatus tests
# ---------------------------------------------------------------------------


class TestSpanStatus:
    """Tests for SpanStatus enum."""

    def test_values(self) -> None:
        assert SpanStatus.UNSET == "unset"
        assert SpanStatus.OK == "ok"
        assert SpanStatus.ERROR == "error"


# ---------------------------------------------------------------------------
# InMemoryExporter tests
# ---------------------------------------------------------------------------


class TestInMemoryExporter:
    """Tests for the InMemoryExporter."""

    def test_satisfies_protocol(self) -> None:
        assert isinstance(InMemoryExporter(), SpanExporter)

    def test_export_collects_spans(self) -> None:
        exporter = InMemoryExporter()
        span = Span(name="test").start()
        span.end()
        exporter.export([span])
        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "test"

    def test_export_accumulates(self) -> None:
        exporter = InMemoryExporter()
        exporter.export([Span(name="a")])
        exporter.export([Span(name="b")])
        assert len(exporter.spans) == 2

    def test_clear(self) -> None:
        exporter = InMemoryExporter()
        exporter.export([Span(name="a")])
        exporter.clear()
        assert len(exporter.spans) == 0


# ---------------------------------------------------------------------------
# LogExporter tests
# ---------------------------------------------------------------------------


class TestLogExporter:
    """Tests for the LogExporter."""

    def test_satisfies_protocol(self) -> None:
        assert isinstance(LogExporter(), SpanExporter)

    def test_export_does_not_raise(self) -> None:
        exporter = LogExporter()
        span = Span(name="test").start()
        span.end()
        # Should not raise
        exporter.export([span])

    def test_custom_log_level(self) -> None:
        exporter = LogExporter(log_level="debug")
        assert exporter._log_level == "debug"


# ---------------------------------------------------------------------------
# Tracer tests -- synchronous context manager
# ---------------------------------------------------------------------------


class TestTracerSync:
    """Tests for synchronous span creation and context management."""

    def test_start_span_context_manager(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        with tracer.start_span("test_op") as span:
            assert span.name == "test_op"
            assert span.start_time > 0
        assert len(exporter.spans) == 1
        assert exporter.spans[0].end_time is not None
        assert exporter.spans[0].status == SpanStatus.OK

    def test_span_records_error_on_exception(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        with pytest.raises(ValueError, match="boom"), tracer.start_span("failing"):
            raise ValueError("boom")
        assert len(exporter.spans) == 1
        assert exporter.spans[0].status == SpanStatus.ERROR

    def test_nested_spans_share_trace_id(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        with tracer.start_span("parent") as parent, tracer.start_span("child") as child:
            assert child.context.trace_id == parent.context.trace_id
            assert child.context.parent_span_id == parent.context.span_id
        assert len(exporter.spans) == 2

    def test_explicit_parent(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        parent = tracer.create_span("parent")
        with tracer.start_span("child", parent=parent) as child:
            assert child.context.trace_id == parent.context.trace_id
            assert child.context.parent_span_id == parent.context.span_id

    def test_service_name_attribute(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(service_name="my-service", exporter=exporter)
        with tracer.start_span("op") as span:
            assert span.attributes["service.name"] == "my-service"

    def test_initial_attributes(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        with tracer.start_span("op", attributes={"key": "value"}) as span:
            assert span.attributes["key"] == "value"

    def test_current_span_inside_context(self) -> None:
        tracer = Tracer()
        assert tracer.current_span is None
        with tracer.start_span("active"):
            assert tracer.current_span is not None
            assert tracer.current_span.name == "active"
        assert tracer.current_span is None

    def test_disabled_tracer_still_creates_spans(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter, enabled=False)
        with tracer.start_span("op") as span:
            assert span.name == "op"
        # Spans are still exported even when disabled
        assert len(exporter.spans) == 1

    def test_disabled_tracer_does_not_link_parent(self) -> None:
        tracer = Tracer(enabled=False)
        with tracer.start_span("parent"), tracer.start_span("child") as child:
            # Disabled tracer does not propagate context
            assert child.context.parent_span_id is None


# ---------------------------------------------------------------------------
# Tracer tests -- async context manager
# ---------------------------------------------------------------------------


class TestTracerAsync:
    """Tests for async span creation and context management."""

    async def test_async_context_manager(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        async with tracer.start_async_span("async_op") as span:
            assert span.name == "async_op"
        assert len(exporter.spans) == 1
        assert exporter.spans[0].status == SpanStatus.OK

    async def test_async_span_error(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        with pytest.raises(RuntimeError, match="async boom"):
            async with tracer.start_async_span("failing"):
                raise RuntimeError("async boom")
        assert exporter.spans[0].status == SpanStatus.ERROR

    async def test_async_nested_spans(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        async with tracer.start_async_span("parent") as parent:  # noqa: SIM117
            async with tracer.start_async_span("child") as child:
                assert child.context.trace_id == parent.context.trace_id
                assert child.context.parent_span_id == parent.context.span_id
        assert len(exporter.spans) == 2


# ---------------------------------------------------------------------------
# Tracer.trace decorator tests
# ---------------------------------------------------------------------------


class TestTraceDecorator:
    """Tests for the @tracer.trace() decorator."""

    def test_sync_decorator(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)

        @tracer.trace("my.operation")
        def do_work() -> str:
            return "done"

        result = do_work()
        assert result == "done"
        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "my.operation"
        assert exporter.spans[0].status == SpanStatus.OK

    async def test_async_decorator(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)

        @tracer.trace("async.operation")
        async def do_async_work() -> str:
            await asyncio.sleep(0.001)
            return "async done"

        result = await do_async_work()
        assert result == "async done"
        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "async.operation"

    def test_decorator_default_name(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)

        @tracer.trace()
        def my_function() -> None:
            pass

        my_function()
        # Should use the qualified name
        assert "my_function" in exporter.spans[0].name

    def test_decorator_captures_error(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)

        @tracer.trace("failing")
        def fail() -> None:
            raise TypeError("bad type")

        with pytest.raises(TypeError, match="bad type"):
            fail()
        assert exporter.spans[0].status == SpanStatus.ERROR

    async def test_async_decorator_captures_error(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)

        @tracer.trace("async_fail")
        async def async_fail() -> None:
            raise TypeError("async bad type")

        with pytest.raises(TypeError, match="async bad type"):
            await async_fail()
        assert exporter.spans[0].status == SpanStatus.ERROR

    def test_decorator_with_attributes(self) -> None:
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)

        @tracer.trace("op", attributes={"component": "cc"})
        def op() -> None:
            pass

        op()
        assert exporter.spans[0].attributes["component"] == "cc"

    def test_decorator_preserves_function_metadata(self) -> None:
        tracer = Tracer()

        @tracer.trace("op")
        def documented_fn() -> None:
            """This function has a docstring."""

        assert documented_fn.__name__ == "documented_fn"
        assert documented_fn.__doc__ == "This function has a docstring."


# ---------------------------------------------------------------------------
# Agent tick trace structure test
# ---------------------------------------------------------------------------


class TestAgentTickTraceStructure:
    """Test that the tracing system can represent the typical agent tick hierarchy."""

    def test_typical_agent_tick_trace(self) -> None:
        """Simulate the typical trace tree from an agent tick.

        agent.tick
        +-- module.action_awareness.tick
        +-- cc.compress
        +-- cc.decide
        |   +-- llm.complete
        +-- cc.broadcast
        """
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)

        with tracer.start_span("agent.tick") as root:
            root_trace_id = root.context.trace_id

            with tracer.start_span("module.action_awareness.tick") as aa:
                aa.set_attribute("module", "action_awareness")

            with tracer.start_span("cc.compress"):
                pass

            with tracer.start_span("cc.decide"):  # noqa: SIM117
                with tracer.start_span("llm.complete") as llm:
                    llm.set_attribute("model", "gpt-4o-mini")
                    llm.add_event("token_generated", {"count": 42})

            with tracer.start_span("cc.broadcast"):
                pass

        # All spans share the same trace ID (5 children + 1 root = 6)
        assert len(exporter.spans) == 6
        for span in exporter.spans:
            assert span.context.trace_id == root_trace_id
            assert span.status == SpanStatus.OK

        # Verify parent-child relationships
        names = {s.name: s for s in exporter.spans}
        root_span = names["agent.tick"]
        assert root_span.context.parent_span_id is None

        aa_span = names["module.action_awareness.tick"]
        assert aa_span.context.parent_span_id == root_span.context.span_id
        assert names["cc.compress"].context.parent_span_id == root_span.context.span_id
        assert names["cc.decide"].context.parent_span_id == root_span.context.span_id
        assert names["cc.broadcast"].context.parent_span_id == root_span.context.span_id

        decide_span = names["cc.decide"]
        assert names["llm.complete"].context.parent_span_id == decide_span.context.span_id

    def test_no_exporter_does_not_crash(self) -> None:
        tracer = Tracer(exporter=None)
        with tracer.start_span("op"):
            pass
        # No assertion needed -- just ensure no exception

    def test_enabled_toggle(self) -> None:
        tracer = Tracer()
        assert tracer.enabled is True
        tracer.enabled = False
        assert tracer.enabled is False
