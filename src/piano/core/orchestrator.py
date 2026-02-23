"""Multi-Agent Orchestrator for the PIANO architecture.

Manages multiple PIANO agents for 10+ agent simulations. Provides lifecycle
coordination, shared resource management, and inter-agent communication.

Reference: docs/implementation/01-system-architecture.md
"""

from __future__ import annotations

__all__ = ["AgentConfig", "AgentOrchestrator", "OrchestratorConfig"]

import asyncio
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

from piano.core.agent import Agent

if TYPE_CHECKING:
    from collections.abc import Callable

    from piano.core.sas import SharedAgentState

logger = structlog.get_logger()


class AgentConfig(BaseModel):
    """Configuration for creating a new agent.

    Attributes:
        agent_id: Unique identifier for the agent.
        personality: Big Five personality traits (default: neutral 0.5 for all).
        initial_position: Starting position in the world (default: origin).
        role_hint: Optional role description for the agent.
    """

    agent_id: str
    personality: dict[str, float] = Field(
        default_factory=lambda: {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        }
    )
    initial_position: dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
    )
    role_hint: str = ""


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator.

    Attributes:
        max_agents: Maximum number of agents that can be managed (default: 10).
        tick_interval: Tick interval for individual agents in seconds (default: 0.5).
        shared_tick_interval: Global tick interval for orchestrator operations (default: 1.0).
    """

    max_agents: int = 10
    tick_interval: float = 0.5
    shared_tick_interval: float = 1.0


class AgentOrchestrator:
    """Orchestrates multiple PIANO agents for multi-agent simulations.

    The orchestrator manages agent lifecycles, coordinates shared resources
    (LLM gateway, bridge connections), and provides inter-agent message routing.

    Usage::

        config = OrchestratorConfig(max_agents=10)
        orchestrator = AgentOrchestrator(config, sas_factory)

        # Add agents
        agent_config = AgentConfig(agent_id="agent-001")
        agent = await orchestrator.add_agent(agent_config)

        # Start all agents
        await orchestrator.start_all()

        # Broadcast message to all agents
        await orchestrator.broadcast_to_all("Server event: night approaching")

        # Get aggregate stats
        stats = await orchestrator.get_all_stats()

        # Clean shutdown
        await orchestrator.stop_all()
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        sas_factory: Callable[[str], SharedAgentState],
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: Orchestrator configuration.
            sas_factory: Factory function that creates a SharedAgentState
                instance for a given agent_id. Example:
                ``lambda agent_id: InMemorySAS(agent_id)``
        """
        self._config = config
        self._sas_factory = sas_factory
        self._agents: dict[str, Agent] = {}
        self._sas_instances: dict[str, SharedAgentState] = {}
        logger.info(
            "orchestrator_initialized",
            max_agents=config.max_agents,
            tick_interval=config.tick_interval,
        )

    @property
    def agent_count(self) -> int:
        """Number of currently managed agents."""
        return len(self._agents)

    def list_agents(self) -> list[str]:
        """Get list of all agent IDs.

        Returns:
            List of agent IDs currently managed by the orchestrator.
        """
        return list(self._agents.keys())

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            The Agent instance, or None if not found.
        """
        return self._agents.get(agent_id)

    async def add_agent(self, config: AgentConfig) -> Agent:
        """Create and register a new agent.

        Args:
            config: Configuration for the new agent.

        Returns:
            The newly created Agent instance.

        Raises:
            ValueError: If max_agents limit is reached or agent_id already exists.
        """
        if len(self._agents) >= self._config.max_agents:
            raise ValueError(
                f"Cannot add agent: max_agents limit ({self._config.max_agents}) reached"
            )

        if config.agent_id in self._agents:
            raise ValueError(f"Agent '{config.agent_id}' already exists")

        # Create SAS for this agent
        sas = self._sas_factory(config.agent_id)
        self._sas_instances[config.agent_id] = sas

        # Initialize SAS with agent config
        await sas.initialize()

        # Store personality and initial position in SAS
        await sas.update_section("personality", config.personality)
        await sas.update_section("initial_position", config.initial_position)
        await sas.update_section("role_hint", {"role": config.role_hint})

        # Create scheduler (note: scheduler is created here but modules would be
        # registered by the user after getting the agent back)
        from piano.core.scheduler import ModuleScheduler

        scheduler = ModuleScheduler(tick_interval=self._config.tick_interval)

        # Create agent
        agent = Agent(
            agent_id=config.agent_id,
            sas=sas,
            scheduler=scheduler,
            cc=None,  # CC can be set by user if needed
        )

        self._agents[config.agent_id] = agent

        logger.info(
            "agent_added",
            agent_id=config.agent_id,
            total_agents=len(self._agents),
        )

        return agent

    async def remove_agent(self, agent_id: str) -> None:
        """Remove and shut down an agent.

        Args:
            agent_id: The agent's unique identifier.

        Raises:
            ValueError: If agent_id does not exist.
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent '{agent_id}' does not exist")

        agent = self._agents[agent_id]

        # Gracefully shut down the agent
        await agent.shutdown()

        # Clean up SAS
        sas = self._sas_instances[agent_id]
        await sas.clear()

        # Remove from tracking
        del self._agents[agent_id]
        del self._sas_instances[agent_id]

        logger.info(
            "agent_removed",
            agent_id=agent_id,
            remaining_agents=len(self._agents),
        )

    async def start_all(self) -> None:
        """Start all agents concurrently.

        Initializes and starts the main loop for all registered agents.
        Agents run until stop_all() is called.
        """
        if not self._agents:
            logger.warning("start_all_called_with_no_agents")
            return

        logger.info("starting_all_agents", count=len(self._agents))

        # Initialize all agents concurrently
        init_tasks = [agent.initialize() for agent in self._agents.values()]
        await asyncio.gather(*init_tasks)

        # Start all agents concurrently (run in background)
        run_tasks = [asyncio.create_task(agent.run()) for agent in self._agents.values()]

        # Store tasks for cleanup in stop_all
        self._run_tasks = run_tasks

        logger.info("all_agents_started", count=len(self._agents))

    async def stop_all(self) -> None:
        """Gracefully stop all agents.

        Shuts down all agents and cleans up their resources.
        """
        if not self._agents:
            logger.warning("stop_all_called_with_no_agents")
            return

        logger.info("stopping_all_agents", count=len(self._agents))

        # Shutdown all agents concurrently
        shutdown_tasks = [agent.shutdown() for agent in self._agents.values()]
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        # Cancel background run tasks if they exist
        if hasattr(self, "_run_tasks"):
            for task in self._run_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._run_tasks, return_exceptions=True)
            del self._run_tasks

        logger.info("all_agents_stopped", count=len(self._agents))

    async def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Collect aggregate statistics from all agents.

        Returns:
            Dictionary mapping agent_id to their statistics. Each agent's stats
            include tick_count, module_count, running status, and scheduler state.
        """
        stats: dict[str, dict[str, Any]] = {}

        for agent_id, agent in self._agents.items():
            stats[agent_id] = {
                "tick_count": agent.scheduler.tick_count,
                "module_count": len(agent.modules),
                "running": agent.running,
                "scheduler_state": agent.scheduler.state.value,
            }

        return stats

    async def broadcast_to_all(self, message: str) -> None:
        """Send a message to all agents via their SAS.

        The message is stored in a special "broadcast" section in each agent's SAS.
        Modules can read from this section to respond to global events.

        Args:
            message: The message to broadcast to all agents.
        """
        if not self._agents:
            logger.warning("broadcast_to_all_called_with_no_agents")
            return

        logger.info("broadcasting_to_all_agents", message=message, count=len(self._agents))

        # Send message to all SAS instances concurrently
        timestamp = asyncio.get_event_loop().time()
        broadcast_tasks = [
            sas.update_section("broadcast", {"message": message, "timestamp": timestamp})
            for sas in self._sas_instances.values()
        ]
        await asyncio.gather(*broadcast_tasks)

        logger.debug("broadcast_complete", recipients=len(self._sas_instances))
