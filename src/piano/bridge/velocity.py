"""Velocity MC Proxy Bridge for multi-server management and dynamic load balancing.

Manages multiple Minecraft backend servers behind a Velocity proxy,
providing agent-to-server assignment with least-connections load balancing
and spatial sharding support.
"""

from __future__ import annotations

__all__ = [
    "LoadBalanceStrategy",
    "ServerConfig",
    "ServerLoad",
    "VelocityConfig",
    "VelocityProxyManager",
]

import logging
from enum import StrEnum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Enums ---


class LoadBalanceStrategy(StrEnum):
    """Load balancing strategy for agent assignment."""

    LEAST_CONNECTIONS = "least_connections"
    SPATIAL_SHARD = "spatial_shard"


# --- Configuration Models ---


class ServerConfig(BaseModel):
    """Configuration for a single backend Minecraft server."""

    name: str
    host: str
    port: int
    max_players: int = 200
    weight: float = 1.0


class VelocityConfig(BaseModel):
    """Configuration for the Velocity proxy."""

    proxy_host: str = "localhost"
    proxy_port: int = 25577
    max_players_per_server: int = 200
    servers: list[ServerConfig] = Field(default_factory=list)
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_CONNECTIONS


class ServerLoad(BaseModel):
    """Current load information for a backend server."""

    server_name: str
    current_players: int
    max_players: int
    load_factor: float
    is_healthy: bool


# --- Manager ---


class VelocityProxyManager:
    """Manages agent-to-server assignment across multiple Minecraft backends.

    Tracks server registrations, agent assignments, per-server health,
    and performs load-based rebalancing.
    """

    def __init__(self, config: VelocityConfig) -> None:
        self._config = config
        # server_name -> ServerConfig
        self._servers: dict[str, ServerConfig] = {}
        # agent_id -> server_name
        self._agent_assignments: dict[str, str] = {}
        # server_name -> set of agent_ids
        self._server_agents: dict[str, set[str]] = {}
        # server_name -> healthy flag
        self._server_health: dict[str, bool] = {}
        # shard_id -> server_name (for spatial sharding)
        self._shard_map: dict[int, str] = {}

        # Register initial servers from config
        for server in config.servers:
            self._add_server(server)

    # --- Properties ---

    @property
    def config(self) -> VelocityConfig:
        """Return the current proxy configuration."""
        return self._config

    @property
    def server_count(self) -> int:
        """Return the number of registered servers."""
        return len(self._servers)

    @property
    def total_agents(self) -> int:
        """Return the total number of assigned agents across all servers."""
        return len(self._agent_assignments)

    # --- Internal helpers ---

    def _add_server(self, server: ServerConfig) -> None:
        """Register a server internally without logging."""
        self._servers[server.name] = server
        self._server_agents[server.name] = set()
        self._server_health[server.name] = True

    # --- Server registration ---

    def register_server(self, server_name: str, host: str, port: int) -> None:
        """Register a new backend server.

        If a server with the same name already exists, its configuration
        is updated in place.

        Args:
            server_name: Unique name for the server.
            host: Server hostname or IP address.
            port: Server port number.
        """
        if server_name in self._servers:
            logger.info(
                "Updating existing server %s host=%s port=%d",
                server_name,
                host,
                port,
            )
            self._servers[server_name] = ServerConfig(
                name=server_name,
                host=host,
                port=port,
                max_players=self._servers[server_name].max_players,
                weight=self._servers[server_name].weight,
            )
            return

        config = ServerConfig(name=server_name, host=host, port=port)
        self._add_server(config)
        logger.info("Server registered: %s host=%s port=%d", server_name, host, port)

    def unregister_server(self, server_name: str) -> None:
        """Unregister a backend server and reassign its agents.

        Args:
            server_name: Name of the server to remove.

        Raises:
            KeyError: If the server is not registered.
        """
        if server_name not in self._servers:
            raise KeyError(f"Server '{server_name}' is not registered")

        # Collect agents that need reassignment
        orphaned_agents = list(self._server_agents.get(server_name, set()))

        # Remove server
        del self._servers[server_name]
        del self._server_agents[server_name]
        del self._server_health[server_name]

        # Remove from shard map
        shards_to_remove = [
            shard_id for shard_id, srv in self._shard_map.items() if srv == server_name
        ]
        for shard_id in shards_to_remove:
            del self._shard_map[shard_id]

        # Reassign orphaned agents
        for agent_id in orphaned_agents:
            del self._agent_assignments[agent_id]
            if self._servers:
                new_server = self._pick_least_loaded_server()
                if new_server is not None:
                    self._assign_agent(agent_id, new_server)

        logger.info(
            "Server unregistered: %s orphaned_agents=%d",
            server_name,
            len(orphaned_agents),
        )

    # --- Agent assignment ---

    def get_server_for_agent(self, agent_id: str) -> str:
        """Get or assign a server for the given agent.

        If the agent is already assigned to a healthy server, returns
        that server. Otherwise, assigns the agent using the configured
        load balancing strategy.

        Args:
            agent_id: Unique identifier of the agent.

        Returns:
            Name of the server the agent is assigned to.

        Raises:
            RuntimeError: If no healthy servers are available.
        """
        # Return existing assignment if server is still healthy
        if agent_id in self._agent_assignments:
            assigned = self._agent_assignments[agent_id]
            if assigned in self._servers and self._server_health.get(assigned, False):
                return assigned
            # Server gone or unhealthy — remove stale assignment
            self._remove_agent_assignment(agent_id)

        # Assign using strategy
        server_name = self._select_server(agent_id)
        self._assign_agent(agent_id, server_name)
        logger.debug(
            "Agent assigned: %s -> %s",
            agent_id,
            server_name,
        )
        return server_name

    def _select_server(self, agent_id: str) -> str:
        """Select a server for an agent based on the configured strategy.

        Args:
            agent_id: The agent's identifier (used for spatial sharding hash).

        Returns:
            Name of the selected server.

        Raises:
            RuntimeError: If no healthy servers are available.
        """
        if self._config.strategy == LoadBalanceStrategy.SPATIAL_SHARD:
            return self._select_by_shard(agent_id)
        return self._select_least_connections()

    def _select_least_connections(self) -> str:
        """Pick the healthy server with the fewest current connections.

        Returns:
            Name of the least loaded healthy server.

        Raises:
            RuntimeError: If no healthy servers are available.
        """
        server = self._pick_least_loaded_server()
        if server is None:
            raise RuntimeError("No healthy servers available for assignment")
        return server

    def _pick_least_loaded_server(self) -> str | None:
        """Return the healthy server name with fewest agents, or None."""
        candidates = [
            name
            for name, healthy in self._server_health.items()
            if healthy and name in self._servers
        ]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda name: (
                len(self._server_agents.get(name, set())) / self._servers[name].weight,
                name,  # deterministic tie-break
            ),
        )

    def _select_by_shard(self, agent_id: str) -> str:
        """Assign agent to a server based on spatial shard ID.

        Uses a hash of the agent_id to derive a shard, then maps
        the shard to a server. Falls back to least-connections if
        the shard's server is unhealthy or no shard mapping exists.

        Args:
            agent_id: The agent's identifier.

        Returns:
            Name of the selected server.
        """
        healthy_servers = [
            name
            for name, healthy in self._server_health.items()
            if healthy and name in self._servers
        ]
        if not healthy_servers:
            raise RuntimeError("No healthy servers available for assignment")

        shard_id = hash(agent_id) % max(len(healthy_servers), 1)

        # Check if shard is mapped to a healthy server
        if shard_id in self._shard_map:
            mapped = self._shard_map[shard_id]
            if mapped in self._servers and self._server_health.get(mapped, False):
                return mapped

        # Assign shard to a healthy server round-robin style
        server_name = healthy_servers[shard_id % len(healthy_servers)]
        self._shard_map[shard_id] = server_name
        return server_name

    def _assign_agent(self, agent_id: str, server_name: str) -> None:
        """Record agent -> server assignment."""
        self._agent_assignments[agent_id] = server_name
        self._server_agents.setdefault(server_name, set()).add(agent_id)

    def _remove_agent_assignment(self, agent_id: str) -> None:
        """Remove an agent's server assignment."""
        if agent_id in self._agent_assignments:
            old_server = self._agent_assignments.pop(agent_id)
            self._server_agents.get(old_server, set()).discard(agent_id)

    # --- Load monitoring ---

    def get_server_load(self) -> dict[str, ServerLoad]:
        """Get current load information for all registered servers.

        Returns:
            Mapping of server name to its load information.
        """
        loads: dict[str, ServerLoad] = {}
        for name, config in self._servers.items():
            current = len(self._server_agents.get(name, set()))
            max_p = min(config.max_players, self._config.max_players_per_server)
            factor = current / max_p if max_p > 0 else 0.0
            loads[name] = ServerLoad(
                server_name=name,
                current_players=current,
                max_players=max_p,
                load_factor=round(factor, 4),
                is_healthy=self._server_health.get(name, False),
            )
        return loads

    # --- Health management ---

    def set_server_health(self, server_name: str, is_healthy: bool) -> None:
        """Update the health status of a server.

        Args:
            server_name: Name of the server.
            is_healthy: Whether the server is healthy.

        Raises:
            KeyError: If the server is not registered.
        """
        if server_name not in self._servers:
            raise KeyError(f"Server '{server_name}' is not registered")
        self._server_health[server_name] = is_healthy
        logger.info(
            "Server health updated: %s is_healthy=%s",
            server_name,
            is_healthy,
        )

    # --- Rebalancing ---

    def rebalance_agents(self) -> dict[str, str]:
        """Rebalance agents across healthy servers to even out load.

        Moves agents from overloaded servers to underloaded servers.
        Only considers healthy servers for rebalancing targets.

        Returns:
            Mapping of agent_id -> new_server_name for agents that were moved.
        """
        healthy_servers = [name for name in self._servers if self._server_health.get(name, False)]

        if not healthy_servers:
            logger.warning("Rebalance skipped: no healthy servers")
            return {}

        total_agents = self.total_agents
        if total_agents == 0:
            return {}

        # Calculate ideal distribution (weighted)
        total_weight = sum(self._servers[s].weight for s in healthy_servers)
        if total_weight <= 0:
            return {}

        ideal: dict[str, int] = {}
        assigned_so_far = 0
        for i, name in enumerate(healthy_servers):
            weight_frac = self._servers[name].weight / total_weight
            if i == len(healthy_servers) - 1:
                # Last server gets remainder to avoid rounding issues
                ideal[name] = total_agents - assigned_so_far
            else:
                count = round(total_agents * weight_frac)
                ideal[name] = count
                assigned_so_far += count

        # Identify over- and under-loaded servers
        excess_agents: list[str] = []  # agent IDs to move
        for name in healthy_servers:
            current_agents = list(self._server_agents.get(name, set()))
            target = ideal.get(name, 0)
            if len(current_agents) > target:
                # Take excess agents (from the end for determinism with sorted)
                sorted_agents = sorted(current_agents)
                to_remove = sorted_agents[target:]
                excess_agents.extend(to_remove)

        # Also collect agents from unhealthy servers
        for name in list(self._servers):
            if not self._server_health.get(name, False):
                orphans = list(self._server_agents.get(name, set()))
                for agent_id in orphans:
                    if agent_id not in excess_agents:
                        excess_agents.append(agent_id)

        # Reassign excess agents to underloaded servers
        # Remove old assignment and immediately reassign to avoid orphaned agents
        moves: dict[str, str] = {}
        for agent_id in excess_agents:
            self._remove_agent_assignment(agent_id)
            target_server = self._pick_least_loaded_server()
            if target_server is None:
                break
            self._assign_agent(agent_id, target_server)
            moves[agent_id] = target_server

        logger.info("Rebalance complete: %d moves", len(moves))
        return moves

    # --- Shard management ---

    def assign_shard(self, shard_id: int, server_name: str) -> None:
        """Explicitly map a spatial shard to a server.

        Args:
            shard_id: The shard identifier.
            server_name: Name of the server to handle the shard.

        Raises:
            KeyError: If the server is not registered.
        """
        if server_name not in self._servers:
            raise KeyError(f"Server '{server_name}' is not registered")
        self._shard_map[shard_id] = server_name
        logger.debug("Shard assigned: %d -> %s", shard_id, server_name)

    def get_shard_map(self) -> dict[int, str]:
        """Return a copy of the current shard-to-server mapping."""
        return dict(self._shard_map)

    # --- Utility ---

    def get_agent_assignment(self, agent_id: str) -> str | None:
        """Return the server an agent is currently assigned to, or None."""
        return self._agent_assignments.get(agent_id)

    def get_agents_on_server(self, server_name: str) -> set[str]:
        """Return the set of agent IDs assigned to a server.

        Args:
            server_name: Name of the server.

        Returns:
            Set of agent IDs (copy).

        Raises:
            KeyError: If the server is not registered.
        """
        if server_name not in self._servers:
            raise KeyError(f"Server '{server_name}' is not registered")
        return set(self._server_agents.get(server_name, set()))
