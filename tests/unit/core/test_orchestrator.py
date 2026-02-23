"""Tests for the Multi-Agent Orchestrator."""

import asyncio

import pytest

from piano.core.orchestrator import AgentConfig, AgentOrchestrator, OrchestratorConfig
from piano.core.types import ModuleTier
from tests.helpers import DummyModule, InMemorySAS


@pytest.fixture
def sas_factory():
    """Factory function that creates InMemorySAS instances."""
    return lambda agent_id: InMemorySAS(agent_id)


@pytest.fixture
def orchestrator_config():
    """Default orchestrator configuration."""
    return OrchestratorConfig(max_agents=5, tick_interval=0.1)


@pytest.fixture
def orchestrator(orchestrator_config, sas_factory):
    """Create an orchestrator instance."""
    return AgentOrchestrator(orchestrator_config, sas_factory)


@pytest.fixture
def agent_config():
    """Default agent configuration."""
    return AgentConfig(agent_id="agent-001")


# --- Agent Management Tests ---


async def test_add_agent_creates_agent_with_sas(orchestrator, agent_config):
    """Test that add_agent creates an agent with its own SAS."""
    agent = await orchestrator.add_agent(agent_config)

    assert agent is not None
    assert agent.agent_id == "agent-001"
    assert agent.sas is not None
    assert agent.sas.agent_id == "agent-001"
    assert orchestrator.agent_count == 1


async def test_add_agent_stores_personality_and_position(orchestrator):
    """Test that agent config is stored in SAS."""
    config = AgentConfig(
        agent_id="agent-002",
        personality={"openness": 0.8, "conscientiousness": 0.6},
        initial_position={"x": 100.0, "y": 64.0, "z": 200.0},
        role_hint="builder",
    )
    agent = await orchestrator.add_agent(config)

    personality = await agent.sas.get_section("personality")
    assert personality["openness"] == 0.8
    assert personality["conscientiousness"] == 0.6

    position = await agent.sas.get_section("initial_position")
    assert position["x"] == 100.0
    assert position["y"] == 64.0
    assert position["z"] == 200.0

    role = await agent.sas.get_section("role_hint")
    assert role["role"] == "builder"


async def test_add_agent_enforces_max_agents_limit(orchestrator):
    """Test that max_agents limit is enforced."""
    # Add 5 agents (max_agents=5 in fixture)
    for i in range(5):
        config = AgentConfig(agent_id=f"agent-{i:03d}")
        await orchestrator.add_agent(config)

    # 6th agent should raise ValueError
    config = AgentConfig(agent_id="agent-999")
    with pytest.raises(ValueError, match="max_agents limit"):
        await orchestrator.add_agent(config)


async def test_add_agent_duplicate_id_raises_error(orchestrator, agent_config):
    """Test that adding an agent with duplicate ID raises ValueError."""
    await orchestrator.add_agent(agent_config)

    # Try to add again with same ID
    with pytest.raises(ValueError, match="already exists"):
        await orchestrator.add_agent(agent_config)


async def test_remove_agent_shuts_down_agent(orchestrator, agent_config):
    """Test that remove_agent properly shuts down the agent."""
    await orchestrator.add_agent(agent_config)
    assert orchestrator.agent_count == 1

    await orchestrator.remove_agent("agent-001")

    assert orchestrator.agent_count == 0
    assert orchestrator.get_agent("agent-001") is None


async def test_remove_agent_clears_sas(orchestrator, agent_config):
    """Test that remove_agent clears the agent's SAS."""
    agent = await orchestrator.add_agent(agent_config)

    # Add some data to SAS
    await agent.sas.update_section("test", {"data": "value"})

    # Store reference to SAS before removal
    sas = agent.sas

    await orchestrator.remove_agent("agent-001")

    # SAS should be cleared
    section = await sas.get_section("test")
    assert section == {}


async def test_remove_nonexistent_agent_raises_error(orchestrator):
    """Test that removing a non-existent agent raises ValueError."""
    with pytest.raises(ValueError, match="does not exist"):
        await orchestrator.remove_agent("nonexistent-agent")


# --- Agent Retrieval Tests ---


async def test_get_agent_returns_correct_agent(orchestrator):
    """Test that get_agent returns the correct agent."""
    config1 = AgentConfig(agent_id="agent-001")
    config2 = AgentConfig(agent_id="agent-002")

    agent1 = await orchestrator.add_agent(config1)
    agent2 = await orchestrator.add_agent(config2)

    assert orchestrator.get_agent("agent-001") is agent1
    assert orchestrator.get_agent("agent-002") is agent2
    assert orchestrator.get_agent("agent-999") is None


async def test_list_agents_returns_all_ids(orchestrator):
    """Test that list_agents returns all agent IDs."""
    for i in range(3):
        config = AgentConfig(agent_id=f"agent-{i:03d}")
        await orchestrator.add_agent(config)

    agent_ids = orchestrator.list_agents()

    assert len(agent_ids) == 3
    assert "agent-000" in agent_ids
    assert "agent-001" in agent_ids
    assert "agent-002" in agent_ids


async def test_agent_count_property(orchestrator):
    """Test that agent_count property is accurate."""
    assert orchestrator.agent_count == 0

    config1 = AgentConfig(agent_id="agent-001")
    await orchestrator.add_agent(config1)
    assert orchestrator.agent_count == 1

    config2 = AgentConfig(agent_id="agent-002")
    await orchestrator.add_agent(config2)
    assert orchestrator.agent_count == 2

    await orchestrator.remove_agent("agent-001")
    assert orchestrator.agent_count == 1


# --- Lifecycle Management Tests ---


async def test_start_all_starts_all_agents(orchestrator):
    """Test that start_all initializes and starts all agents."""
    # Add 3 agents
    for i in range(3):
        config = AgentConfig(agent_id=f"agent-{i:03d}")
        agent = await orchestrator.add_agent(config)

        # Register a dummy module
        module = DummyModule(f"module-{i}", ModuleTier.FAST)
        agent.register_module(module)

    # Start all agents
    await orchestrator.start_all()

    # Give agents time to start
    await asyncio.sleep(0.2)

    # All agents should be running
    for agent_id in orchestrator.list_agents():
        agent = orchestrator.get_agent(agent_id)
        assert agent is not None
        assert agent.running

    # Clean up
    await orchestrator.stop_all()


async def test_start_all_with_no_agents(orchestrator):
    """Test that start_all handles empty agent list gracefully."""
    # Should not raise an error
    await orchestrator.start_all()


async def test_stop_all_stops_all_agents(orchestrator):
    """Test that stop_all gracefully shuts down all agents."""
    # Add and start agents
    for i in range(3):
        config = AgentConfig(agent_id=f"agent-{i:03d}")
        agent = await orchestrator.add_agent(config)

        # Register a dummy module
        module = DummyModule(f"module-{i}", ModuleTier.FAST)
        agent.register_module(module)

    await orchestrator.start_all()
    await asyncio.sleep(0.2)

    # Stop all agents
    await orchestrator.stop_all()

    # All agents should be stopped
    for agent_id in orchestrator.list_agents():
        agent = orchestrator.get_agent(agent_id)
        assert agent is not None
        assert not agent.running


async def test_stop_all_with_no_agents(orchestrator):
    """Test that stop_all handles empty agent list gracefully."""
    # Should not raise an error
    await orchestrator.stop_all()


# --- Statistics Collection Tests ---


async def test_get_all_stats_returns_agent_stats(orchestrator):
    """Test that get_all_stats returns statistics for all agents."""
    # Add agents with modules
    for i in range(2):
        config = AgentConfig(agent_id=f"agent-{i:03d}")
        agent = await orchestrator.add_agent(config)

        # Register modules
        module1 = DummyModule(f"module-{i}-a", ModuleTier.FAST)
        module2 = DummyModule(f"module-{i}-b", ModuleTier.MID)
        agent.register_module(module1)
        agent.register_module(module2)

    stats = await orchestrator.get_all_stats()

    assert len(stats) == 2
    assert "agent-000" in stats
    assert "agent-001" in stats

    # Check stats structure
    for _agent_id, agent_stats in stats.items():
        assert "tick_count" in agent_stats
        assert "module_count" in agent_stats
        assert "running" in agent_stats
        assert "scheduler_state" in agent_stats

        assert agent_stats["module_count"] == 2
        assert agent_stats["running"] is False
        assert agent_stats["scheduler_state"] == "idle"


async def test_get_all_stats_reflects_running_state(orchestrator):
    """Test that get_all_stats shows correct running state."""
    config = AgentConfig(agent_id="agent-001")
    agent = await orchestrator.add_agent(config)

    # Register module
    module = DummyModule("test-module", ModuleTier.FAST)
    agent.register_module(module)

    # Before starting
    stats = await orchestrator.get_all_stats()
    assert stats["agent-001"]["running"] is False

    # After starting
    await orchestrator.start_all()
    await asyncio.sleep(0.1)

    stats = await orchestrator.get_all_stats()
    assert stats["agent-001"]["running"] is True

    # Clean up
    await orchestrator.stop_all()


# --- Broadcasting Tests ---


async def test_broadcast_to_all_sends_to_all_sas(orchestrator):
    """Test that broadcast_to_all sends message to all agents."""
    # Add 3 agents
    for i in range(3):
        config = AgentConfig(agent_id=f"agent-{i:03d}")
        await orchestrator.add_agent(config)

    # Broadcast message
    await orchestrator.broadcast_to_all("Server announcement: event starting")

    # Check all agents received the message
    for agent_id in orchestrator.list_agents():
        agent = orchestrator.get_agent(agent_id)
        assert agent is not None

        broadcast = await agent.sas.get_section("broadcast")
        assert broadcast["message"] == "Server announcement: event starting"
        assert "timestamp" in broadcast


async def test_broadcast_to_all_with_no_agents(orchestrator):
    """Test that broadcast_to_all handles empty agent list gracefully."""
    # Should not raise an error
    await orchestrator.broadcast_to_all("Test message")


async def test_broadcast_updates_all_agents_independently(orchestrator):
    """Test that each agent's SAS receives the broadcast independently."""
    # Add 2 agents
    config1 = AgentConfig(agent_id="agent-001")
    config2 = AgentConfig(agent_id="agent-002")

    agent1 = await orchestrator.add_agent(config1)
    agent2 = await orchestrator.add_agent(config2)

    # First broadcast
    await orchestrator.broadcast_to_all("Message 1")

    broadcast1 = await agent1.sas.get_section("broadcast")
    broadcast2 = await agent2.sas.get_section("broadcast")

    assert broadcast1["message"] == "Message 1"
    assert broadcast2["message"] == "Message 1"

    # Second broadcast
    await orchestrator.broadcast_to_all("Message 2")

    broadcast1 = await agent1.sas.get_section("broadcast")
    broadcast2 = await agent2.sas.get_section("broadcast")

    assert broadcast1["message"] == "Message 2"
    assert broadcast2["message"] == "Message 2"


# --- Integration Tests ---


async def test_full_lifecycle_with_multiple_agents(orchestrator):
    """Test complete lifecycle: add, start, run, stop, remove."""
    # Add 3 agents
    agents = []
    for i in range(3):
        config = AgentConfig(agent_id=f"agent-{i:03d}")
        agent = await orchestrator.add_agent(config)

        # Register a module
        module = DummyModule(f"module-{i}", ModuleTier.FAST)
        agent.register_module(module)
        agents.append(agent)

    assert orchestrator.agent_count == 3

    # Start all
    await orchestrator.start_all()
    await asyncio.sleep(0.3)  # Let agents run for a bit

    # All agents should be running
    stats = await orchestrator.get_all_stats()
    for agent_id in orchestrator.list_agents():
        assert stats[agent_id]["running"] is True
        assert stats[agent_id]["tick_count"] > 0

    # Broadcast message
    await orchestrator.broadcast_to_all("Test broadcast")

    # Stop all
    await orchestrator.stop_all()

    # All agents should be stopped
    for agent_id in orchestrator.list_agents():
        agent = orchestrator.get_agent(agent_id)
        assert agent is not None
        assert not agent.running

    # Remove agents
    await orchestrator.remove_agent("agent-000")
    await orchestrator.remove_agent("agent-001")
    await orchestrator.remove_agent("agent-002")

    assert orchestrator.agent_count == 0


async def test_orchestrator_config_defaults():
    """Test that OrchestratorConfig has correct defaults."""
    config = OrchestratorConfig()

    assert config.max_agents == 10
    assert config.tick_interval == 0.5
    assert config.shared_tick_interval == 1.0


async def test_agent_config_defaults():
    """Test that AgentConfig has correct defaults."""
    config = AgentConfig(agent_id="test-agent")

    assert config.agent_id == "test-agent"
    assert config.personality["openness"] == 0.5
    assert config.personality["conscientiousness"] == 0.5
    assert config.personality["extraversion"] == 0.5
    assert config.personality["agreeableness"] == 0.5
    assert config.personality["neuroticism"] == 0.5
    assert config.initial_position["x"] == 0.0
    assert config.initial_position["y"] == 0.0
    assert config.initial_position["z"] == 0.0
    assert config.role_hint == ""
