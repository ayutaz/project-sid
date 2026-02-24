"""PIANO simulation launcher.

Provides CLI entry point for running single or multi-agent simulations.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from typing import Any

import structlog

from piano.config.settings import PianoSettings
from piano.core.agent import Agent
from piano.core.scheduler import ModuleScheduler
from piano.llm.mock import MockLLMProvider

logger = structlog.get_logger()

_IS_WINDOWS = sys.platform == "win32"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    try:
        from importlib.metadata import version as _pkg_version

        _version = _pkg_version("piano")
    except Exception:
        _version = "0.0.0-dev"

    parser = argparse.ArgumentParser(
        prog="piano",
        description="PIANO simulation launcher",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_version}",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=1,
        help="Number of agents to run (default: 1)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Max ticks to run (default: infinite)",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use MockLLMProvider instead of real LLM",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (.env)",
    )
    parser.add_argument(
        "--no-bridge",
        action="store_true",
        help="Run without bridge connection (mock mode)",
    )
    parser.add_argument(
        "--sas-backend",
        type=str,
        default="auto",
        choices=["auto", "redis", "memory"],
        help="SAS backend: auto (redis if available, else memory), redis, memory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    """Configure structlog with the given level."""
    # On Windows, stderr defaults to the locale encoding (e.g. cp932 for Japanese).
    # LLM-generated text often contains Unicode characters (em dash, curly quotes, etc.)
    # that cp932 cannot encode, causing UnicodeEncodeError in log output which then
    # propagates as a module failure.  Reconfigure stderr to UTF-8 with replacement
    # fallback so logging never raises on unencodable characters.
    if _IS_WINDOWS and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level),
        stream=sys.stderr,
    )
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level),
        ),
    )


def _install_shutdown_handler(shutdown_event: asyncio.Event) -> None:
    """Install signal handlers that set *shutdown_event* on SIGINT/SIGTERM.

    On Windows, ``loop.add_signal_handler`` is not supported, so we fall back
    to ``signal.signal`` with a thread-safe ``call_soon_threadsafe``.
    """

    def _on_signal(*_args: object) -> None:
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    if _IS_WINDOWS:
        loop = asyncio.get_running_loop()

        def _win_handler(*_args: object) -> None:
            loop.call_soon_threadsafe(_on_signal)

        signal.signal(signal.SIGINT, _win_handler)
        with contextlib.suppress(OSError):
            signal.signal(signal.SIGTERM, _win_handler)
    else:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _on_signal)


def _register_demo_responses(mock: MockLLMProvider) -> None:
    """Register diverse rotating demo response patterns for MockLLM."""
    import json

    templates = [
        {"action": "explore", "action_params": {"direction": "north", "distance": 20},
         "speaking": "", "reasoning": "Exploring the area"},
        {"action": "explore", "action_params": {"direction": "east", "distance": 15},
         "speaking": "Let me check what is over there", "reasoning": "Curious about surroundings"},
        {"action": "chat", "action_params": {"message": "Hello everyone!"},
         "speaking": "Hello everyone!", "reasoning": "Being social"},
        {"action": "chat", "action_params": {"message": "Nice to meet you all"},
         "speaking": "Nice to meet you all", "reasoning": "Greeting others"},
        {"action": "mine", "action_params": {"x": 10, "y": 62, "z": 10},
         "speaking": "", "reasoning": "Mining nearby block"},
        {"action": "look", "action_params": {"x": 0, "y": 64, "z": 0},
         "speaking": "", "reasoning": "Looking around"},
        {"action": "idle", "action_params": {},
         "speaking": "", "reasoning": "Taking a moment to observe"},
    ]
    demo_responses = [json.dumps(t) for t in templates]
    call_count = [0]

    async def _rotating_complete(prompt: str, **kwargs: Any) -> str:
        idx = call_count[0] % len(demo_responses)
        call_count[0] += 1
        return demo_responses[idx]

    mock.complete = _rotating_complete  # type: ignore[assignment]


def _create_provider(args: argparse.Namespace, settings: PianoSettings | None = None) -> Any:
    """Create an LLM provider based on CLI flags.

    Args:
        args: Parsed CLI arguments.
        settings: Optional PianoSettings to configure the provider.
    """
    if args.mock_llm:
        provider = MockLLMProvider()
        _register_demo_responses(provider)
        return provider
    import os

    from piano.llm.provider import OpenAIProvider

    if settings is not None:
        llm_cfg = settings.llm
        # PIANO_LLM__API_KEY takes priority, then OPENAI_API_KEY env var
        api_key = llm_cfg.api_key or os.environ.get("OPENAI_API_KEY") or None
        return OpenAIProvider(
            api_key=api_key,
            cost_limit_usd=llm_cfg.cost_limit_usd,
            calls_per_minute_limit=llm_cfg.calls_per_minute_limit,
        )
    return OpenAIProvider()


def _resolve_sas_backend(args: argparse.Namespace) -> str:
    """Resolve effective SAS backend from CLI args.

    When ``--sas-backend`` is explicitly set to ``redis`` or ``memory``,
    that value is used.  When it is ``auto`` (the default) and
    ``--mock-llm`` is active, ``memory`` is returned so that demo runs
    do not require a Redis server.
    """
    backend = getattr(args, "sas_backend", "auto")
    if backend != "auto":
        return backend
    if getattr(args, "mock_llm", False):
        return "memory"
    return "auto"


def _create_sas(agent_id: str, *, sas_backend: str = "auto") -> Any:
    """Create a SAS instance for the given agent.

    Args:
        agent_id: The agent identifier.
        sas_backend: Backend selection - "auto", "redis", or "memory".
            "auto": Try Redis, fall back to in-memory.
            "redis": Use Redis, raise on failure.
            "memory": Always use in-memory.
    """
    if sas_backend == "memory":
        return _build_local_sas(agent_id)

    if sas_backend in ("redis", "auto"):
        try:
            from piano.core.sas_redis import RedisSAS

            redis_settings = PianoSettings().redis
            return RedisSAS.create_with_settings(redis_settings, agent_id)
        except Exception as e:
            if sas_backend == "redis":
                raise
            logger.warning("redis_sas_creation_failed", error=str(e), fallback="in-memory")

    return _build_local_sas(agent_id)


def _build_local_sas(agent_id: str) -> Any:
    """Build a lightweight in-memory SAS instance."""
    from piano.core.sas import SharedAgentState
    from piano.core.types import (
        ActionHistoryEntry,
        GoalData,
        MemoryEntry,
        PerceptData,
        PlanData,
        SelfReflectionData,
        SocialData,
    )

    class _LocalSAS(SharedAgentState):
        """Minimal in-memory SAS for launcher without Redis."""

        def __init__(self, aid: str) -> None:
            self._id = aid
            self._percepts = PerceptData()
            self._goals = GoalData()
            self._social = SocialData()
            self._plans = PlanData()
            self._actions: list[ActionHistoryEntry] = []
            self._wm: list[MemoryEntry] = []
            self._stm: list[MemoryEntry] = []
            self._reflection = SelfReflectionData()
            self._cc: dict[str, Any] | None = None
            self._sections: dict[str, dict[str, Any]] = {}

        @property
        def agent_id(self) -> str:
            return self._id

        async def get_percepts(self) -> PerceptData:
            return self._percepts

        async def update_percepts(self, p: PerceptData) -> None:
            self._percepts = p

        async def get_goals(self) -> GoalData:
            return self._goals

        async def update_goals(self, g: GoalData) -> None:
            self._goals = g

        async def get_social(self) -> SocialData:
            return self._social

        async def update_social(self, s: SocialData) -> None:
            self._social = s

        async def get_plans(self) -> PlanData:
            return self._plans

        async def update_plans(self, p: PlanData) -> None:
            self._plans = p

        async def get_action_history(self, limit: int = 50) -> list[ActionHistoryEntry]:
            return list(reversed(self._actions[-limit:]))

        async def add_action(self, entry: ActionHistoryEntry) -> None:
            self._actions.append(entry)
            self._actions = self._actions[-50:]

        async def get_working_memory(self) -> list[MemoryEntry]:
            return list(self._wm)

        async def set_working_memory(self, entries: list[MemoryEntry]) -> None:
            self._wm = list(entries)

        async def get_stm(self, limit: int = 100) -> list[MemoryEntry]:
            return list(reversed(self._stm[-limit:]))

        async def add_stm(self, entry: MemoryEntry) -> None:
            self._stm.append(entry)
            self._stm = self._stm[-100:]

        async def get_self_reflection(self) -> SelfReflectionData:
            return self._reflection

        async def update_self_reflection(self, r: SelfReflectionData) -> None:
            self._reflection = r

        async def get_last_cc_decision(self) -> dict[str, Any] | None:
            return self._cc

        async def set_cc_decision(self, d: dict[str, Any]) -> None:
            self._cc = d

        async def get_section(self, section: str) -> dict[str, Any]:
            return dict(self._sections.get(section, {}))

        async def update_section(self, section: str, data: dict[str, Any]) -> None:
            self._sections[section] = dict(data)

        async def snapshot(self) -> dict[str, Any]:
            return {
                "agent_id": self._id,
                "percepts": self._percepts.model_dump(),
                "goals": self._goals.model_dump(),
                "social": self._social.model_dump(),
                "plans": self._plans.model_dump(),
                "action_history": [e.model_dump() for e in self._actions],
                "working_memory": [e.model_dump() for e in self._wm],
                "stm": [e.model_dump() for e in self._stm],
                "self_reflection": self._reflection.model_dump(),
                "cc_decision": self._cc,
            }

        async def initialize(self) -> None:
            pass

        async def clear(self) -> None:
            self._percepts = PerceptData()
            self._goals = GoalData()
            self._social = SocialData()
            self._plans = PlanData()
            self._actions = []
            self._wm = []
            self._stm = []
            self._reflection = SelfReflectionData()
            self._cc = None
            self._sections = {}

    return _LocalSAS(agent_id)


async def _health_check_loop(
    health_monitor: Any,
    bridge_manager: Any,
    shutdown_event: asyncio.Event,
    interval_s: float = 10.0,
) -> None:
    """Periodic health check for bridge connections."""
    while not shutdown_event.is_set():
        try:
            results = await health_monitor.check_all(bridge_manager.bridges)
            summary = health_monitor.summary(results)
            logger.debug("bridge_health_check", summary=summary)
        except Exception as e:
            logger.debug("bridge_health_check_error", error=str(e))
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_s)
            return
        except TimeoutError:
            pass


def _register_modules(
    agent: Agent,
    provider: Any,
    bridge: Any | None = None,
    sas: Any | None = None,
) -> None:
    """Register standard PIANO modules on an agent.

    Args:
        agent: The agent to register modules on.
        provider: LLM provider instance.
        bridge: Optional bridge client for Mineflayer communication.
        sas: SAS instance. Always passed to TalkingModule for utterance storage.
            When both *bridge* and *sas* are provided, also registers bridge-connected
            modules: BridgePerceptionModule, SkillExecutor, and ChatBroadcaster.
    """
    from piano.awareness.action import ActionAwareness
    from piano.cc.controller import CognitiveController
    from piano.goals.generator import GoalGenerationModule
    from piano.planning.planner import PlanningModule
    from piano.reflection.module import SelfReflectionModule
    from piano.social.awareness import SocialAwarenessModule
    from piano.talking.module import TalkingModule

    agent.register_module(ActionAwareness())

    cc = CognitiveController(llm=provider)
    agent.register_module(cc)

    agent.register_module(GoalGenerationModule(llm_provider=provider))
    agent.register_module(PlanningModule(llm=provider))

    # Always pass sas to TalkingModule so it can store utterances
    talking = TalkingModule(llm_provider=provider, sas=sas)
    agent.register_module(talking)

    agent.register_module(SelfReflectionModule(llm_provider=provider))
    agent.register_module(SocialAwarenessModule(llm_provider=provider))

    if bridge is not None and sas is not None:
        from piano.bridge.chat_broadcaster import ChatBroadcaster
        from piano.bridge.perception import BridgePerceptionModule
        from piano.skills.action_mapper import create_full_registry
        from piano.skills.executor import SkillExecutor

        agent.register_module(BridgePerceptionModule(bridge))

        registry = create_full_registry()
        executor = SkillExecutor(registry=registry, bridge=bridge, sas=sas)
        agent.register_module(executor)

        bm = cc.broadcast_manager
        bm.register(executor)

        chat_broadcaster = ChatBroadcaster(bridge=bridge, sas=sas, llm=provider)
        agent.register_module(chat_broadcaster)
        bm.register(chat_broadcaster)
        bm.register(talking)


async def run(args: argparse.Namespace) -> None:
    """Run the PIANO simulation.

    Args:
        args: Parsed CLI arguments.
    """
    # Load .env so that OPENAI_API_KEY and other vars are available as env vars
    from dotenv import load_dotenv

    env_path = args.config if args.config else ".env"
    load_dotenv(env_path, override=False)

    settings_kwargs: dict[str, Any] = {}
    if args.config:
        import pathlib

        config_path = pathlib.Path(args.config)
        if not config_path.exists():
            logger.warning("config_file_not_found", path=args.config)
        settings_kwargs["_env_file"] = args.config
    settings = PianoSettings(**settings_kwargs)

    # CLI --log-level takes priority; fall back to settings.log.level
    log_level = args.log_level if args.log_level != "INFO" else settings.log.level
    _configure_logging(log_level)
    provider = _create_provider(args, settings=settings)

    tick_interval = settings.agent.tick_interval_ms / 1000.0

    if args.agents == 1:
        await _run_single(args, provider, tick_interval, settings=settings)
    else:
        await _run_multi(args, provider, settings, tick_interval)


async def _run_single(
    args: argparse.Namespace,
    provider: Any,
    tick_interval: float,
    settings: PianoSettings | None = None,
) -> None:
    """Run a single-agent simulation."""
    agent_id = "agent-001"
    use_bridge = not getattr(args, "no_bridge", False)
    sas_backend = _resolve_sas_backend(args)
    sas = _create_sas(agent_id, sas_backend=sas_backend)
    scheduler = ModuleScheduler(tick_interval=tick_interval)
    agent = Agent(agent_id=agent_id, sas=sas, scheduler=scheduler)

    bridge_client = None
    bridge_manager = None
    if use_bridge:
        try:
            from piano.bridge.manager import BridgeManager
            from piano.config.settings import BridgeSettings

            bridge_settings = settings.bridge if settings else BridgeSettings()
            bridge_manager = BridgeManager(
                host=bridge_settings.host,
                base_command_port=bridge_settings.base_command_port,
                base_event_port=bridge_settings.base_event_port,
                connect_timeout_s=bridge_settings.connect_timeout_s,
                connect_retry_count=bridge_settings.connect_retry_count,
            )
            bridge_client = bridge_manager.create_bridge(agent_id, 0)
            await bridge_manager.connect_all()
        except Exception:
            logger.warning("Bridge connection failed, running without bridge")
            bridge_client = None
            bridge_manager = None

    _register_modules(
        agent,
        provider,
        bridge=bridge_client,
        sas=sas,
    )

    await agent.initialize()
    logger.info("single_agent_starting", agent_id=agent_id, max_ticks=args.ticks)

    shutdown_event = asyncio.Event()
    _install_shutdown_handler(shutdown_event)

    # Start health monitor if bridge is connected
    health_task: asyncio.Task[None] | None = None
    if bridge_manager is not None and bridge_client is not None:
        from piano.bridge.health import BridgeHealthMonitor

        health_monitor = BridgeHealthMonitor()
        health_task = asyncio.create_task(
            _health_check_loop(health_monitor, bridge_manager, shutdown_event)
        )

    run_task = asyncio.create_task(agent.run(max_ticks=args.ticks))

    _done, _ = await asyncio.wait(
        [run_task, asyncio.create_task(shutdown_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )

    if shutdown_event.is_set() and not run_task.done():
        await agent.shutdown()
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task

    if health_task is not None and not health_task.done():
        health_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await health_task

    if bridge_manager is not None:
        await bridge_manager.disconnect_all()

    logger.info("single_agent_finished", agent_id=agent_id)


async def _run_multi(
    args: argparse.Namespace,
    provider: Any,
    settings: PianoSettings,
    tick_interval: float,
) -> None:
    """Run a multi-agent simulation."""
    from piano.core.orchestrator import (
        AgentConfig,
        AgentOrchestrator,
        OrchestratorConfig,
    )

    config = OrchestratorConfig(
        max_agents=args.agents,
        tick_interval=tick_interval,
    )
    sas_backend = _resolve_sas_backend(args)
    use_bridge = not getattr(args, "no_bridge", False)

    # Set up bridge manager if needed
    bridge_manager = None
    if use_bridge:
        try:
            from piano.bridge.manager import BridgeManager

            bridge_settings = settings.bridge
            bridge_manager = BridgeManager(
                host=bridge_settings.host,
                base_command_port=bridge_settings.base_command_port,
                base_event_port=bridge_settings.base_event_port,
                connect_timeout_s=bridge_settings.connect_timeout_s,
                connect_retry_count=bridge_settings.connect_retry_count,
            )
        except Exception:
            logger.warning("Bridge manager creation failed, running without bridge")
            bridge_manager = None

    # Track SAS instances per agent for bridge registration
    sas_map: dict[str, Any] = {}

    def _sas_factory(aid: str) -> Any:
        sas = _create_sas(aid, sas_backend=sas_backend)
        sas_map[aid] = sas
        return sas

    orchestrator = AgentOrchestrator(config, sas_factory=_sas_factory)

    # Create agents and optionally bridge clients
    agents_with_bridges: list[tuple[Agent, Any]] = []
    for i in range(args.agents):
        agent_id = f"agent-{i + 1:03d}"
        agent_config = AgentConfig(agent_id=agent_id)
        agent = await orchestrator.add_agent(agent_config)

        bridge_client = None
        if bridge_manager is not None:
            bridge_client = bridge_manager.create_bridge(agent_id, i)

        agents_with_bridges.append((agent, bridge_client))

    # Connect all bridges first, then register modules
    bridge_connected = False
    if bridge_manager is not None:
        try:
            status = await bridge_manager.connect_all()
            failed = [aid for aid, ok in status.items() if not ok]
            if failed:
                logger.warning("bridge_partial_connect_failure", failed_agents=failed)
            bridge_connected = any(status.values())
        except Exception:
            logger.warning("Bridge connections failed, continuing without bridge")

    # Register modules after bridge connection attempt
    for agent, bridge_client in agents_with_bridges:
        if bridge_connected and bridge_client is not None:
            _register_modules(
                agent,
                provider,
                bridge=bridge_client,
                sas=sas_map.get(agent.agent_id),
            )
        else:
            _register_modules(agent, provider, sas=sas_map.get(agent.agent_id))

    await orchestrator.start_all()
    logger.info("multi_agent_started", agent_count=args.agents)

    shutdown_event = asyncio.Event()
    _install_shutdown_handler(shutdown_event)

    # Start health monitor if bridge is connected
    health_task: asyncio.Task[None] | None = None
    if bridge_connected and bridge_manager is not None:
        from piano.bridge.health import BridgeHealthMonitor

        health_monitor = BridgeHealthMonitor()
        health_task = asyncio.create_task(
            _health_check_loop(health_monitor, bridge_manager, shutdown_event)
        )

    # If --ticks is set, monitor agent tick counts and stop when all reach the target
    monitor_task: asyncio.Task[None] | None = None
    if args.ticks is not None:

        async def _tick_monitor() -> None:
            max_ticks = args.ticks
            while not shutdown_event.is_set():
                stats = await orchestrator.get_all_stats()
                if all(s["tick_count"] >= max_ticks for s in stats.values()):
                    logger.info("all_agents_reached_max_ticks", max_ticks=max_ticks)
                    shutdown_event.set()
                    return
                await asyncio.sleep(0.5)

        monitor_task = asyncio.create_task(_tick_monitor())

    await shutdown_event.wait()
    await orchestrator.stop_all()

    if monitor_task is not None and not monitor_task.done():
        monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor_task

    if health_task is not None and not health_task.done():
        health_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await health_task

    if bridge_manager is not None:
        await bridge_manager.disconnect_all()

    logger.info("multi_agent_finished", agent_count=args.agents)


def cli() -> None:
    """CLI entry point for ``piano`` command."""
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    cli()
