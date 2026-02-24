"""Integration tests for the full simulation flow."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from piano.bridge.chat_broadcaster import ChatBroadcaster
from piano.bridge.perception import BridgePerceptionModule
from piano.core.agent import Agent
from piano.core.scheduler import ModuleScheduler
from piano.core.types import BridgeEvent, CCDecision
from piano.llm.mock import MockLLMProvider
from piano.main import _create_sas, _register_modules
from piano.skills.action_mapper import create_full_registry
from piano.skills.executor import SkillExecutor


@pytest.fixture
def sas():
    return _create_sas("test-agent", sas_backend="memory")


@pytest.fixture
def mock_bridge():
    bridge = AsyncMock()
    bridge.start_event_listener = AsyncMock()
    bridge.send_command = AsyncMock(return_value={"success": True})
    bridge.chat = AsyncMock(return_value={"success": True})
    bridge.connect = AsyncMock()
    bridge.disconnect = AsyncMock()
    bridge.ping = AsyncMock(return_value=True)
    return bridge


class TestSimulationFlow:
    async def test_perception_to_sas_pipeline(self, sas, mock_bridge):
        """Perception events flow through module to SAS."""
        module = BridgePerceptionModule(mock_bridge)
        event = BridgeEvent(
            event_type="perception",
            data={"position": {"x": 1.0, "y": 2.0, "z": 3.0}, "health": 20.0, "food": 20.0},
        )
        await module._on_event(event)
        await module.tick(sas)

        percepts = await sas.get_percepts()
        assert percepts.position == {"x": 1.0, "y": 2.0, "z": 3.0}

    async def test_chat_broadcaster_sends_utterance(self, sas, mock_bridge):
        """ChatBroadcaster sends utterance when speaking directive present."""
        await sas.update_section(
            "talking",
            {
                "latest_utterance": {"content": "Hello world!", "tone": "friendly"},
            },
        )

        broadcaster = ChatBroadcaster(bridge=mock_bridge, sas=sas)
        decision = CCDecision(speaking="say hello", action="chat")
        await broadcaster.on_broadcast(decision)

        mock_bridge.chat.assert_called_once_with("Hello world!")

    async def test_skill_executor_dispatches_action(self, sas, mock_bridge):
        """SkillExecutor dispatches action from CC decision."""
        registry = create_full_registry()
        executor = SkillExecutor(registry=registry, bridge=mock_bridge, sas=sas)

        decision = CCDecision(action="move", action_params={"x": 5.0, "y": 64.0, "z": 10.0})
        await executor.on_broadcast(decision)

        # Poll for background task completion with timeout
        for _ in range(50):
            if executor._current_task is None or executor._current_task.done():
                break
            await asyncio.sleep(0.02)

        assert mock_bridge.send_command.called

    async def test_register_modules_with_bridge(self, sas, mock_bridge):
        """_register_modules with bridge registers perception, executor, broadcaster."""
        scheduler = ModuleScheduler(tick_interval=0.5)
        agent = Agent(agent_id="test-agent", sas=sas, scheduler=scheduler)
        provider = MockLLMProvider()

        _register_modules(agent, provider, bridge=mock_bridge, sas=sas)

        module_names = [m.name for m in agent._modules]
        assert "bridge_perception" in module_names
        assert "skill_executor" in module_names
        assert "chat_broadcaster" in module_names

    async def test_register_modules_without_bridge(self, sas):
        """_register_modules without bridge has core modules only."""
        scheduler = ModuleScheduler(tick_interval=0.5)
        agent = Agent(agent_id="test-agent", sas=sas, scheduler=scheduler)
        provider = MockLLMProvider()

        _register_modules(agent, provider)

        module_names = [m.name for m in agent._modules]
        assert "bridge_perception" not in module_names
        assert "skill_executor" not in module_names
        assert "chat_broadcaster" not in module_names
        # Core modules should be present
        assert "cognitive_controller" in module_names

    async def test_five_tick_loop(self, sas, mock_bridge):
        """Run 5 ticks of a fully-wired agent."""
        scheduler = ModuleScheduler(tick_interval=0.01)
        agent = Agent(agent_id="test-agent", sas=sas, scheduler=scheduler)
        provider = MockLLMProvider()

        _register_modules(agent, provider, bridge=mock_bridge, sas=sas)

        # Inject a perception event
        for module in agent._modules:
            if module.name == "bridge_perception":
                event = BridgeEvent(
                    event_type="perception",
                    data={
                        "position": {"x": 0.0, "y": 64.0, "z": 0.0},
                        "health": 20.0,
                        "food": 20.0,
                    },
                )
                await module._on_event(event)
                break

        await agent.initialize()
        await agent.run(max_ticks=5)

        # Verify perception was processed
        percepts = await sas.get_percepts()
        assert percepts.position == {"x": 0.0, "y": 64.0, "z": 0.0}

    async def test_three_agents_parallel(self, mock_bridge):
        """Run 3 agents in parallel for 3 ticks each."""
        agents = []
        for i in range(3):
            agent_id = f"agent-{i + 1:03d}"
            agent_sas = _create_sas(agent_id, sas_backend="memory")
            scheduler = ModuleScheduler(tick_interval=0.01)
            agent = Agent(agent_id=agent_id, sas=agent_sas, scheduler=scheduler)
            provider = MockLLMProvider()
            _register_modules(agent, provider)
            await agent.initialize()
            agents.append(agent)

        await asyncio.gather(*(a.run(max_ticks=3) for a in agents))

        # All agents should have completed without error
        assert len(agents) == 3

    async def test_no_bridge_backward_compat(self):
        """Agents work without bridge (no-bridge mode)."""
        sas = _create_sas("test-agent", sas_backend="memory")
        scheduler = ModuleScheduler(tick_interval=0.01)
        agent = Agent(agent_id="test-agent", sas=sas, scheduler=scheduler)
        provider = MockLLMProvider()

        _register_modules(agent, provider)  # no bridge
        await agent.initialize()
        await agent.run(max_ticks=3)
        # Should complete without error
