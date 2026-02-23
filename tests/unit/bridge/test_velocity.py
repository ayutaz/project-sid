"""Tests for the Velocity Proxy Manager.

Covers server registration/unregistration, agent assignment with
least-connections and spatial sharding strategies, load monitoring,
health management, and rebalancing.
"""

from __future__ import annotations

import pytest

from piano.bridge.velocity import (
    LoadBalanceStrategy,
    ServerConfig,
    ServerLoad,
    VelocityConfig,
    VelocityProxyManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(
    servers: list[ServerConfig] | None = None,
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_CONNECTIONS,
    max_players_per_server: int = 200,
) -> VelocityConfig:
    """Build a VelocityConfig with sensible defaults."""
    return VelocityConfig(
        proxy_host="localhost",
        proxy_port=25577,
        max_players_per_server=max_players_per_server,
        servers=servers or [],
        strategy=strategy,
    )


def _two_server_config() -> VelocityConfig:
    """Config with two equally weighted servers."""
    return _make_config(
        servers=[
            ServerConfig(name="mc-1", host="10.0.0.1", port=25565, max_players=200),
            ServerConfig(name="mc-2", host="10.0.0.2", port=25565, max_players=200),
        ],
    )


@pytest.fixture
def manager() -> VelocityProxyManager:
    """Return a manager with two servers."""
    return VelocityProxyManager(_two_server_config())


# ---------------------------------------------------------------------------
# Tests: Configuration models
# ---------------------------------------------------------------------------


class TestConfigModels:
    """Tests for Pydantic configuration models."""

    def test_server_config_defaults(self) -> None:
        """ServerConfig has correct defaults for max_players and weight."""
        sc = ServerConfig(name="s1", host="localhost", port=25565)
        assert sc.max_players == 200
        assert sc.weight == 1.0

    def test_velocity_config_defaults(self) -> None:
        """VelocityConfig has sensible defaults."""
        vc = VelocityConfig()
        assert vc.proxy_host == "localhost"
        assert vc.proxy_port == 25577
        assert vc.max_players_per_server == 200
        assert vc.servers == []
        assert vc.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS

    def test_server_load_model(self) -> None:
        """ServerLoad can be constructed and has correct fields."""
        load = ServerLoad(
            server_name="mc-1",
            current_players=50,
            max_players=200,
            load_factor=0.25,
            is_healthy=True,
        )
        assert load.server_name == "mc-1"
        assert load.current_players == 50
        assert load.load_factor == 0.25


# ---------------------------------------------------------------------------
# Tests: Server registration
# ---------------------------------------------------------------------------


class TestServerRegistration:
    """Tests for register_server / unregister_server."""

    def test_initial_servers_from_config(self, manager: VelocityProxyManager) -> None:
        """Servers passed in config are registered at construction."""
        assert manager.server_count == 2

    def test_register_new_server(self, manager: VelocityProxyManager) -> None:
        """register_server adds a new server."""
        manager.register_server("mc-3", "10.0.0.3", 25565)
        assert manager.server_count == 3

    def test_register_existing_server_updates(self, manager: VelocityProxyManager) -> None:
        """Registering an already-known server updates its config."""
        manager.register_server("mc-1", "10.0.0.99", 25566)
        assert manager.server_count == 2  # no new server added
        # Agent assignment still works
        server = manager.get_server_for_agent("agent-001")
        assert server in ("mc-1", "mc-2")

    def test_unregister_server(self, manager: VelocityProxyManager) -> None:
        """unregister_server removes the server."""
        manager.unregister_server("mc-2")
        assert manager.server_count == 1

    def test_unregister_unknown_server_raises(self, manager: VelocityProxyManager) -> None:
        """Unregistering an unknown server raises KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            manager.unregister_server("mc-999")

    def test_unregister_reassigns_agents(self, manager: VelocityProxyManager) -> None:
        """Agents on an unregistered server are reassigned to remaining servers."""
        # Assign agents to mc-1
        manager.get_server_for_agent("agent-001")
        manager.get_server_for_agent("agent-002")

        # Force both onto mc-1 for determinism
        manager._agent_assignments["agent-001"] = "mc-1"
        manager._agent_assignments["agent-002"] = "mc-1"
        manager._server_agents["mc-1"] = {"agent-001", "agent-002"}
        manager._server_agents["mc-2"] = set()

        manager.unregister_server("mc-1")

        # Both agents should now be on mc-2
        assert manager.get_agent_assignment("agent-001") == "mc-2"
        assert manager.get_agent_assignment("agent-002") == "mc-2"
        assert manager.server_count == 1


# ---------------------------------------------------------------------------
# Tests: Agent assignment (least-connections)
# ---------------------------------------------------------------------------


class TestLeastConnectionsAssignment:
    """Tests for least-connections load balancing."""

    def test_first_agent_gets_assigned(self, manager: VelocityProxyManager) -> None:
        """First agent is assigned to one of the servers."""
        server = manager.get_server_for_agent("agent-001")
        assert server in ("mc-1", "mc-2")

    def test_agents_spread_evenly(self, manager: VelocityProxyManager) -> None:
        """Agents are distributed across servers."""
        for i in range(10):
            manager.get_server_for_agent(f"agent-{i:03d}")

        load = manager.get_server_load()
        counts = [load[s].current_players for s in ("mc-1", "mc-2")]
        assert abs(counts[0] - counts[1]) <= 1  # balanced

    def test_same_agent_returns_same_server(self, manager: VelocityProxyManager) -> None:
        """Repeated calls for the same agent return the same server."""
        server1 = manager.get_server_for_agent("agent-001")
        server2 = manager.get_server_for_agent("agent-001")
        assert server1 == server2

    def test_no_servers_raises(self) -> None:
        """RuntimeError raised when no servers are available."""
        mgr = VelocityProxyManager(_make_config(servers=[]))
        with pytest.raises(RuntimeError, match="No healthy servers"):
            mgr.get_server_for_agent("agent-001")

    def test_weighted_distribution(self) -> None:
        """Higher-weight servers receive proportionally more agents."""
        config = _make_config(
            servers=[
                ServerConfig(name="heavy", host="h", port=1, weight=3.0),
                ServerConfig(name="light", host="l", port=2, weight=1.0),
            ],
        )
        mgr = VelocityProxyManager(config)

        for i in range(40):
            mgr.get_server_for_agent(f"agent-{i:03d}")

        load = mgr.get_server_load()
        # heavy should have roughly 3x the agents of light
        heavy_count = load["heavy"].current_players
        light_count = load["light"].current_players
        assert heavy_count > light_count


# ---------------------------------------------------------------------------
# Tests: Agent assignment (spatial sharding)
# ---------------------------------------------------------------------------


class TestSpatialShardAssignment:
    """Tests for spatial shard strategy."""

    def test_shard_assignment(self) -> None:
        """Agents are assigned via shard mapping."""
        config = _make_config(
            servers=[
                ServerConfig(name="mc-1", host="10.0.0.1", port=25565),
                ServerConfig(name="mc-2", host="10.0.0.2", port=25565),
            ],
            strategy=LoadBalanceStrategy.SPATIAL_SHARD,
        )
        mgr = VelocityProxyManager(config)
        server = mgr.get_server_for_agent("agent-shard-test")
        assert server in ("mc-1", "mc-2")

    def test_explicit_shard_map(self) -> None:
        """Explicitly mapped shards direct agents correctly."""
        config = _make_config(
            servers=[
                ServerConfig(name="mc-1", host="10.0.0.1", port=25565),
                ServerConfig(name="mc-2", host="10.0.0.2", port=25565),
            ],
            strategy=LoadBalanceStrategy.SPATIAL_SHARD,
        )
        mgr = VelocityProxyManager(config)

        # Map all shards to mc-2
        for shard_id in range(10):
            mgr.assign_shard(shard_id, "mc-2")

        shard_map = mgr.get_shard_map()
        assert all(v == "mc-2" for v in shard_map.values())

    def test_assign_shard_to_unknown_server_raises(self, manager: VelocityProxyManager) -> None:
        """Assigning a shard to an unknown server raises KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            manager.assign_shard(0, "mc-999")


# ---------------------------------------------------------------------------
# Tests: Load monitoring
# ---------------------------------------------------------------------------


class TestLoadMonitoring:
    """Tests for get_server_load."""

    def test_empty_load(self, manager: VelocityProxyManager) -> None:
        """Load is zero when no agents are assigned."""
        load = manager.get_server_load()
        assert len(load) == 2
        for sl in load.values():
            assert sl.current_players == 0
            assert sl.load_factor == 0.0
            assert sl.is_healthy is True

    def test_load_factor_calculation(self, manager: VelocityProxyManager) -> None:
        """Load factor is current_players / max_players."""
        for i in range(10):
            manager.get_server_for_agent(f"agent-{i:03d}")

        load = manager.get_server_load()
        for sl in load.values():
            expected_factor = sl.current_players / sl.max_players
            assert abs(sl.load_factor - round(expected_factor, 4)) < 0.001

    def test_load_respects_config_max_players_per_server(self) -> None:
        """max_players in load uses min(server max, config max)."""
        config = _make_config(
            servers=[
                ServerConfig(name="mc-1", host="h", port=1, max_players=500),
            ],
            max_players_per_server=100,
        )
        mgr = VelocityProxyManager(config)
        load = mgr.get_server_load()
        assert load["mc-1"].max_players == 100


# ---------------------------------------------------------------------------
# Tests: Health management
# ---------------------------------------------------------------------------


class TestHealthManagement:
    """Tests for set_server_health and its effect on assignment."""

    def test_mark_unhealthy(self, manager: VelocityProxyManager) -> None:
        """Marking a server unhealthy excludes it from load output."""
        manager.set_server_health("mc-1", False)
        load = manager.get_server_load()
        assert load["mc-1"].is_healthy is False
        assert load["mc-2"].is_healthy is True

    def test_unhealthy_server_excluded_from_assignment(self, manager: VelocityProxyManager) -> None:
        """Agents are never assigned to unhealthy servers."""
        manager.set_server_health("mc-1", False)
        for i in range(10):
            server = manager.get_server_for_agent(f"agent-{i:03d}")
            assert server == "mc-2"

    def test_health_unknown_server_raises(self, manager: VelocityProxyManager) -> None:
        """Setting health on an unknown server raises KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            manager.set_server_health("mc-999", True)

    def test_agent_reassigned_when_server_becomes_unhealthy(
        self, manager: VelocityProxyManager
    ) -> None:
        """If an agent's server becomes unhealthy, next lookup reassigns it."""
        server = manager.get_server_for_agent("agent-001")
        manager.set_server_health(server, False)

        new_server = manager.get_server_for_agent("agent-001")
        assert new_server != server


# ---------------------------------------------------------------------------
# Tests: Rebalancing
# ---------------------------------------------------------------------------


class TestRebalancing:
    """Tests for rebalance_agents."""

    def test_rebalance_evens_load(self) -> None:
        """After rebalancing, agents are more evenly distributed."""
        config = _two_server_config()
        mgr = VelocityProxyManager(config)

        # Manually create imbalance: all 10 agents on mc-1
        for i in range(10):
            aid = f"agent-{i:03d}"
            mgr._agent_assignments[aid] = "mc-1"
            mgr._server_agents["mc-1"].add(aid)

        moves = mgr.rebalance_agents()
        assert len(moves) > 0

        load = mgr.get_server_load()
        counts = [load[s].current_players for s in ("mc-1", "mc-2")]
        assert abs(counts[0] - counts[1]) <= 1

    def test_rebalance_no_agents(self, manager: VelocityProxyManager) -> None:
        """Rebalancing with no agents returns empty moves."""
        moves = manager.rebalance_agents()
        assert moves == {}

    def test_rebalance_no_healthy_servers(self) -> None:
        """Rebalancing with no healthy servers returns empty moves."""
        config = _two_server_config()
        mgr = VelocityProxyManager(config)
        mgr.set_server_health("mc-1", False)
        mgr.set_server_health("mc-2", False)
        moves = mgr.rebalance_agents()
        assert moves == {}

    def test_rebalance_moves_agents_from_unhealthy(self) -> None:
        """Agents on unhealthy servers are moved during rebalancing."""
        config = _two_server_config()
        mgr = VelocityProxyManager(config)

        # Place agents on mc-1
        for i in range(4):
            aid = f"agent-{i:03d}"
            mgr._agent_assignments[aid] = "mc-1"
            mgr._server_agents["mc-1"].add(aid)

        # Mark mc-1 unhealthy
        mgr.set_server_health("mc-1", False)

        moves = mgr.rebalance_agents()
        # All agents should have been moved to mc-2
        assert len(moves) == 4
        assert all(v == "mc-2" for v in moves.values())


# ---------------------------------------------------------------------------
# Tests: Properties and utilities
# ---------------------------------------------------------------------------


class TestPropertiesAndUtilities:
    """Tests for properties and utility methods."""

    def test_config_property(self, manager: VelocityProxyManager) -> None:
        """config property returns the VelocityConfig."""
        assert manager.config.proxy_port == 25577

    def test_server_count(self, manager: VelocityProxyManager) -> None:
        """server_count reflects registered servers."""
        assert manager.server_count == 2
        manager.register_server("mc-3", "10.0.0.3", 25565)
        assert manager.server_count == 3

    def test_total_agents(self, manager: VelocityProxyManager) -> None:
        """total_agents reflects number of assigned agents."""
        assert manager.total_agents == 0
        manager.get_server_for_agent("agent-001")
        assert manager.total_agents == 1

    def test_get_agent_assignment_none(self, manager: VelocityProxyManager) -> None:
        """get_agent_assignment returns None for unassigned agent."""
        assert manager.get_agent_assignment("unknown") is None

    def test_get_agents_on_server(self, manager: VelocityProxyManager) -> None:
        """get_agents_on_server returns the correct set."""
        manager.get_server_for_agent("agent-001")
        assigned_server = manager.get_agent_assignment("agent-001")
        assert assigned_server is not None
        agents = manager.get_agents_on_server(assigned_server)
        assert "agent-001" in agents

    def test_get_agents_on_unknown_server_raises(self, manager: VelocityProxyManager) -> None:
        """get_agents_on_server raises KeyError for unknown server."""
        with pytest.raises(KeyError, match="not registered"):
            manager.get_agents_on_server("mc-999")

    def test_load_balance_strategy_enum(self) -> None:
        """LoadBalanceStrategy enum has expected values."""
        assert LoadBalanceStrategy.LEAST_CONNECTIONS == "least_connections"
        assert LoadBalanceStrategy.SPATIAL_SHARD == "spatial_shard"
