"""Structured logging configuration with Grafana Loki compatibility.

Provides JSON structured logging via structlog with configurable
log levels, context injection (agent_id, worker_id, trace_id), and
log filtering by agent, module, or minimum level.

The ``configure_piano_logging`` entry point sets up the full structlog
processor chain: timestamping, level filtering, context injection, and
JSON/console output rendering.

Reference: docs/implementation/08-infrastructure.md
"""

from __future__ import annotations

__all__ = [
    "LogFilter",
    "LogFormat",
    "LoggingConfig",
    "configure_piano_logging",
]

import logging
import sys
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LogFormat enum
# ---------------------------------------------------------------------------


class LogFormat(StrEnum):
    """Supported log output formats."""

    JSON = "json"
    CONSOLE = "console"
    LOKI = "loki"


# ---------------------------------------------------------------------------
# LogFilter
# ---------------------------------------------------------------------------


class LogFilter:
    """Filter log events by agent, module, or minimum level.

    Multiple filters can be combined -- an event must satisfy **all**
    active filter criteria to pass through.
    """

    # Map level names to numeric values for comparison
    _LEVEL_MAP: ClassVar[dict[str, int]] = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def __init__(self) -> None:
        self._agent_ids: set[str] | None = None
        self._module_names: set[str] | None = None
        self._min_level: int | None = None

    def by_agent(self, agent_id: str) -> LogFilter:
        """Only pass log events whose ``agent_id`` matches *agent_id*.

        Can be called multiple times to allow several agents.
        """
        if self._agent_ids is None:
            self._agent_ids = set()
        self._agent_ids.add(agent_id)
        return self

    def by_module(self, module_name: str) -> LogFilter:
        """Only pass log events whose ``module`` matches *module_name*.

        Can be called multiple times to allow several modules.
        """
        if self._module_names is None:
            self._module_names = set()
        self._module_names.add(module_name)
        return self

    def by_level(self, min_level: str) -> LogFilter:
        """Only pass log events at or above *min_level*.

        Args:
            min_level: One of ``"debug"``, ``"info"``, ``"warning"``,
                ``"error"``, ``"critical"`` (case-insensitive).

        Raises:
            ValueError: If *min_level* is not a recognised level name.
        """
        key = min_level.lower()
        if key not in self._LEVEL_MAP:
            msg = f"Unknown log level: {min_level!r}"
            raise ValueError(msg)
        self._min_level = self._LEVEL_MAP[key]
        return self

    # -- structlog processor interface --

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """structlog processor: drop events that do not pass the filter."""
        # Agent filter
        if self._agent_ids is not None:
            event_agent = event_dict.get("agent_id")
            if event_agent not in self._agent_ids:
                raise structlog.DropEvent

        # Module filter
        if self._module_names is not None:
            event_module = event_dict.get("module")
            if event_module not in self._module_names:
                raise structlog.DropEvent

        # Level filter
        if self._min_level is not None:
            level_number = self._LEVEL_MAP.get(method_name.lower(), logging.DEBUG)
            if level_number < self._min_level:
                raise structlog.DropEvent

        return event_dict

    @property
    def active_agent_ids(self) -> set[str] | None:
        """Return the set of allowed agent IDs, or ``None`` if unset."""
        return self._agent_ids

    @property
    def active_module_names(self) -> set[str] | None:
        """Return the set of allowed module names, or ``None`` if unset."""
        return self._module_names

    @property
    def active_min_level(self) -> int | None:
        """Return the numeric minimum level, or ``None`` if unset."""
        return self._min_level


# ---------------------------------------------------------------------------
# LoggingConfig
# ---------------------------------------------------------------------------


class LoggingConfig(BaseModel):
    """Configuration container for the PIANO structured logging system.

    Attributes:
        level: Root log level (e.g. ``"INFO"``).
        format: Output format (:class:`LogFormat`).
        output: Output target -- ``"stdout"``, ``"stderr"``, or a file path.
        module_levels: Per-module overrides (module name -> level).
        context: Global key-value pairs injected into every log event.
        log_filter: Optional :class:`LogFilter` instance.
    """

    level: str = "INFO"
    format: LogFormat = LogFormat.JSON
    output: str = "stdout"
    module_levels: dict[str, str] = Field(default_factory=dict)
    context: dict[str, str] = Field(default_factory=dict)
    log_filter: LogFilter | None = None

    model_config = {"arbitrary_types_allowed": True}

    # -- Convenience mutators ------------------------------------------------

    def configure(
        self,
        level: str | None = None,
        fmt: LogFormat | str | None = None,
        output: str | None = None,
    ) -> LoggingConfig:
        """Set top-level logging parameters (builder-style).

        Args:
            level: Root log level string (e.g. ``"DEBUG"``).
            fmt: Log format (``LogFormat`` or string).
            output: Output target.

        Returns:
            Self for chaining.
        """
        if level is not None:
            self.level = level.upper()
        if fmt is not None:
            self.format = LogFormat(fmt) if isinstance(fmt, str) else fmt
        if output is not None:
            self.output = output
        return self

    def set_level(self, module: str, level: str) -> LoggingConfig:
        """Set a per-module log level override.

        Args:
            module: The module name (e.g. ``"goal_generation"``).
            level: Level string (e.g. ``"DEBUG"``).

        Returns:
            Self for chaining.
        """
        self.module_levels[module] = level.upper()
        return self

    def add_context(self, key: str, value: str) -> LoggingConfig:
        """Add a key-value pair to the global log context.

        Context entries are injected into **every** log event.

        Args:
            key: Context key (e.g. ``"agent_id"``).
            value: Context value (e.g. ``"agent-001"``).

        Returns:
            Self for chaining.
        """
        self.context[key] = value
        return self


# ---------------------------------------------------------------------------
# structlog processors
# ---------------------------------------------------------------------------


def _add_timestamp(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add ISO-8601 timestamp to the event dict."""
    event_dict["timestamp"] = datetime.now(tz=UTC).isoformat()
    return event_dict


def _add_log_level(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add ``level`` key derived from the method name."""
    event_dict["level"] = method_name
    return event_dict


def _make_context_injector(
    context: dict[str, str],
) -> structlog.types.Processor:
    """Return a processor that merges *context* into every event."""

    def _inject_context(
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        for key, value in context.items():
            event_dict.setdefault(key, value)
        return event_dict

    return _inject_context


def _make_module_level_filter(
    module_levels: dict[str, str],
) -> structlog.types.Processor:
    """Return a processor that filters events based on per-module levels."""
    level_map: dict[str, int] = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    parsed: dict[str, int] = {}
    for mod, lvl in module_levels.items():
        parsed[mod] = level_map.get(lvl.lower(), logging.DEBUG)

    def _filter(
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        mod = event_dict.get("module")
        if mod is not None and mod in parsed:
            event_level = level_map.get(method_name.lower(), logging.DEBUG)
            if event_level < parsed[mod]:
                raise structlog.DropEvent
        return event_dict

    return _filter


def _rename_event_key(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Rename structlog's default ``event`` key for Loki compatibility.

    structlog uses ``event`` as the key for the log message.  We keep
    it as ``event`` (which is Loki-friendly) but ensure it is a string.
    """
    if "event" in event_dict:
        event_dict["event"] = str(event_dict["event"])
    return event_dict


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def _build_processor_chain(config: LoggingConfig) -> list[structlog.types.Processor]:
    """Build the structlog processor chain from *config*."""
    processors: list[structlog.types.Processor] = [
        _add_timestamp,
        _add_log_level,
    ]

    # Context injection (agent_id, worker_id, trace_id, etc.)
    if config.context:
        processors.append(_make_context_injector(config.context))

    # Per-module level filtering
    if config.module_levels:
        processors.append(_make_module_level_filter(config.module_levels))

    # User-supplied LogFilter
    if config.log_filter is not None:
        processors.append(config.log_filter)

    # Event key normalisation
    processors.append(_rename_event_key)

    # Final renderer
    if config.format in (LogFormat.JSON, LogFormat.LOKI):
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    return processors


def configure_piano_logging(config: LoggingConfig | None = None) -> LoggingConfig:
    """Configure the PIANO structured logging system.

    Sets up the structlog processor chain according to *config*.  If
    *config* is ``None``, sensible defaults (JSON to stdout, INFO level)
    are used.

    Args:
        config: A :class:`LoggingConfig` instance, or ``None`` for defaults.

    Returns:
        The :class:`LoggingConfig` that was applied.
    """
    if config is None:
        config = LoggingConfig()

    processors = _build_processor_chain(config)

    # Map level string to stdlib numeric level
    root_level = getattr(logging, config.level.upper(), logging.INFO)

    # Configure stdlib logging as a fallback / integration layer
    handler = logging.StreamHandler(sys.stdout if config.output == "stdout" else sys.stderr)
    handler.setLevel(root_level)
    logging.basicConfig(
        format="%(message)s",
        handlers=[handler],
        level=root_level,
        force=True,
    )

    # Apply per-module levels to stdlib loggers
    for module_name, level_str in config.module_levels.items():
        stdlib_logger = logging.getLogger(module_name)
        stdlib_logger.setLevel(getattr(logging, level_str.upper(), logging.INFO))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(root_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    return config
