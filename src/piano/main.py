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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="piano",
        description="PIANO simulation launcher",
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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    """Configure structlog with the given level."""
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


def _create_provider(args: argparse.Namespace) -> Any:
    """Create an LLM provider based on CLI flags."""
    if args.mock_llm:
        return MockLLMProvider()
    from piano.llm.provider import LiteLLMProvider

    return LiteLLMProvider()


def _create_sas(agent_id: str, *, mock_mode: bool = True) -> Any:
    """Create a SAS instance for the given agent.

    When *mock_mode* is ``False`` **and** Redis is reachable, returns a
    ``RedisSAS``.  Otherwise falls back to a lightweight in-memory SAS.
    """
    if not mock_mode:
        try:
            from piano.core.sas_redis import RedisSAS

            redis_settings = PianoSettings().redis
            return RedisSAS.create_with_settings(redis_settings, agent_id)
        except Exception:
            pass  # fall through to in-memory SAS

    if True:  # in-memory fallback (always used in mock mode)
        # Fallback: lightweight in-memory SAS for local/mock runs
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


def _register_modules(agent: Agent, provider: Any) -> None:
    """Register standard PIANO modules on an agent."""
    from piano.awareness.action import ActionAwareness
    from piano.cc.controller import CognitiveController
    from piano.goals.generator import GoalGenerationModule
    from piano.planning.planner import PlanningModule
    from piano.reflection.module import SelfReflectionModule
    from piano.social.awareness import SocialAwarenessModule
    from piano.talking.module import TalkingModule

    agent.register_module(ActionAwareness())
    agent.register_module(CognitiveController(llm=provider))
    agent.register_module(GoalGenerationModule(llm_provider=provider))
    agent.register_module(PlanningModule(llm=provider))
    agent.register_module(TalkingModule(llm_provider=provider))
    agent.register_module(SelfReflectionModule(llm_provider=provider))
    agent.register_module(SocialAwarenessModule(llm_provider=provider))


async def run(args: argparse.Namespace) -> None:
    """Run the PIANO simulation.

    Args:
        args: Parsed CLI arguments.
    """
    settings_kwargs: dict[str, Any] = {}
    if args.config:
        settings_kwargs["_env_file"] = args.config
    settings = PianoSettings(**settings_kwargs)

    _configure_logging(args.log_level)
    provider = _create_provider(args)

    tick_interval = settings.agent.tick_interval_ms / 1000.0

    if args.agents == 1:
        await _run_single(args, provider, tick_interval)
    else:
        await _run_multi(args, provider, settings, tick_interval)


async def _run_single(
    args: argparse.Namespace,
    provider: Any,
    tick_interval: float,
) -> None:
    """Run a single-agent simulation."""
    agent_id = "agent-001"
    sas = _create_sas(agent_id, mock_mode=args.mock_llm)
    scheduler = ModuleScheduler(tick_interval=tick_interval)
    agent = Agent(agent_id=agent_id, sas=sas, scheduler=scheduler)

    _register_modules(agent, provider)

    await agent.initialize()
    logger.info("single_agent_starting", agent_id=agent_id, max_ticks=args.ticks)

    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _signal_handler)

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
    mock = args.mock_llm
    orchestrator = AgentOrchestrator(
        config,
        sas_factory=lambda aid: _create_sas(aid, mock_mode=mock),
    )

    for i in range(args.agents):
        agent_config = AgentConfig(agent_id=f"agent-{i + 1:03d}")
        agent = await orchestrator.add_agent(agent_config)
        _register_modules(agent, provider)

    await orchestrator.start_all()
    logger.info("multi_agent_started", agent_count=args.agents)

    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _signal_handler)

    await shutdown_event.wait()
    await orchestrator.stop_all()
    logger.info("multi_agent_finished", agent_count=args.agents)


def cli() -> None:
    """CLI entry point for ``piano`` command."""
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    cli()
