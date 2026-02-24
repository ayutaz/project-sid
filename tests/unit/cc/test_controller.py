"""Tests for the CognitiveController and BroadcastManager."""

from __future__ import annotations

import json
from typing import Any

import pytest

from piano.cc.broadcast import BroadcastManager, BroadcastResult
from piano.cc.controller import CC_SYSTEM_PROMPT, CognitiveController
from piano.core.types import CCDecision, LLMRequest, LLMResponse, ModuleResult, ModuleTier

# -- Helpers / Fixtures -------------------------------------------------------


class MockLLM:
    """Mock LLM provider that returns a configurable JSON response."""

    def __init__(self, response_data: dict[str, Any] | None = None):
        self._response_data = response_data or {
            "action": "mine",
            "action_params": {"block": "oak_log"},
            "speaking": "Time to chop some wood!",
            "reasoning": "Need wood to build house.",
            "salience_scores": {"goals": 0.9, "percepts": 0.6},
        }
        self.last_request: LLMRequest | None = None
        self.call_count = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        self.call_count += 1
        return LLMResponse(
            content=json.dumps(self._response_data),
            model="mock-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            latency_ms=10.0,
        )


class FailingLLM:
    """Mock LLM provider that always raises."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise RuntimeError("LLM unavailable")


class BadJsonLLM:
    """Mock LLM provider that returns invalid JSON."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="not json at all", model="mock-model")


class MockSAS:
    """Minimal mock SAS for controller tests."""

    def __init__(self, agent_id: str = "test-agent-001"):
        self._agent_id = agent_id
        self._cc_decision: dict[str, Any] | None = None

    @property
    def agent_id(self) -> str:
        return self._agent_id

    async def snapshot(self) -> dict[str, Any]:
        return {
            "percepts": {
                "position": {"x": 10, "y": 64, "z": -20},
                "health": 20.0,
                "hunger": 18.0,
                "inventory": {"wood": 10},
                "nearby_players": ["Alice"],
                "weather": "clear",
                "time_of_day": 6000,
                "chat_messages": [],
            },
            "goals": {"current_goal": "Explore", "goal_stack": [], "sub_goals": []},
            "social": {"relationships": {}, "emotions": {}, "recent_interactions": []},
            "plans": {"current_plan": [], "plan_status": "idle", "current_step": 0},
            "action_history": [],
            "working_memory": [],
            "stm": [],
        }

    async def set_cc_decision(self, decision: dict[str, Any]) -> None:
        self._cc_decision = decision

    async def get_last_cc_decision(self) -> dict[str, Any] | None:
        return self._cc_decision


class MockModule:
    """A simple mock output module for broadcast tests."""

    def __init__(self, name: str = "mock_output", fail: bool = False):
        self._name = name
        self._fail = fail
        self.received: list[CCDecision] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.FAST

    async def on_broadcast(self, decision: CCDecision) -> None:
        if self._fail:
            raise RuntimeError(f"{self._name} failed")
        self.received.append(decision)


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def mock_sas() -> MockSAS:
    return MockSAS()


@pytest.fixture
def controller(mock_llm: MockLLM) -> CognitiveController:
    return CognitiveController(llm=mock_llm)


# -- CognitiveController Tests ------------------------------------------------


class TestCognitiveControllerProperties:
    """Basic property / identity tests."""

    def test_name(self, controller: CognitiveController):
        assert controller.name == "cognitive_controller"

    def test_tier(self, controller: CognitiveController):
        assert controller.tier == ModuleTier.MID

    def test_initial_cycle_count(self, controller: CognitiveController):
        assert controller.cycle_count == 0

    def test_initial_last_decision(self, controller: CognitiveController):
        assert controller.last_decision is None


class TestCognitiveControllerTick:
    """Tests for the main tick lifecycle."""

    async def test_tick_success(self, controller: CognitiveController, mock_sas: MockSAS):
        result = await controller.tick(mock_sas)

        assert isinstance(result, ModuleResult)
        assert result.success
        assert result.data["action"] == "mine"
        assert result.data["speaking"] == "Time to chop some wood!"
        assert controller.cycle_count == 1

    async def test_tick_stores_decision_in_sas(
        self, controller: CognitiveController, mock_sas: MockSAS
    ):
        await controller.tick(mock_sas)

        stored = await mock_sas.get_last_cc_decision()
        assert stored is not None
        assert stored["action"] == "mine"

    async def test_tick_updates_last_decision(
        self, controller: CognitiveController, mock_sas: MockSAS
    ):
        await controller.tick(mock_sas)

        decision = controller.last_decision
        assert decision is not None
        assert decision.action == "mine"

    async def test_tick_token_estimate_in_result(
        self, controller: CognitiveController, mock_sas: MockSAS
    ):
        result = await controller.tick(mock_sas)

        assert "token_estimate" in result.data
        assert result.data["token_estimate"] > 0

    async def test_tick_multiple_cycles(self, controller: CognitiveController, mock_sas: MockSAS):
        await controller.tick(mock_sas)
        await controller.tick(mock_sas)

        assert controller.cycle_count == 2

    async def test_llm_receives_system_prompt(
        self, mock_llm: MockLLM, controller: CognitiveController, mock_sas: MockSAS
    ):
        await controller.tick(mock_sas)

        assert mock_llm.last_request is not None
        assert mock_llm.last_request.system_prompt == CC_SYSTEM_PROMPT
        assert mock_llm.last_request.json_mode is True


class TestCognitiveControllerFallback:
    """Tests for error handling and fallback behaviour."""

    async def test_llm_failure_returns_error_on_first_call(self, mock_sas: MockSAS):
        cc = CognitiveController(llm=FailingLLM())
        result = await cc.tick(mock_sas)

        assert not result.success
        assert "LLM unavailable" in (result.error or "")

    async def test_llm_failure_reuses_previous_decision(self, mock_sas: MockSAS):
        good_llm = MockLLM()
        cc = CognitiveController(llm=good_llm)

        # First tick succeeds
        await cc.tick(mock_sas)
        assert cc.last_decision is not None

        # Swap to failing LLM
        cc._llm = FailingLLM()
        result = await cc.tick(mock_sas)

        # Should return fallback with previous decision data
        assert result.success  # fallback is not an error
        assert result.data.get("fallback") is True
        assert result.data["action"] == "mine"

    async def test_bad_json_falls_back(self, mock_sas: MockSAS):
        cc = CognitiveController(llm=BadJsonLLM())
        result = await cc.tick(mock_sas)

        assert not result.success
        assert "parse error" in (result.error or "")


# -- BroadcastManager Tests ---------------------------------------------------


class TestBroadcastManager:
    """Tests for BroadcastManager."""

    def test_register_and_list(self):
        bm = BroadcastManager()
        m1 = MockModule("talking")
        m2 = MockModule("skill_exec")
        bm.register(m1)
        bm.register(m2)

        assert sorted(bm.listener_names) == ["skill_exec", "talking"]

    def test_unregister(self):
        bm = BroadcastManager()
        m = MockModule("talking")
        bm.register(m)
        bm.unregister("talking")

        assert bm.listener_names == []

    def test_unregister_nonexistent(self):
        bm = BroadcastManager()
        bm.unregister("does_not_exist")  # should not raise

    async def test_broadcast_delivers_to_all(self):
        bm = BroadcastManager()
        m1 = MockModule("mod_a")
        m2 = MockModule("mod_b")
        bm.register(m1)
        bm.register(m2)

        decision = CCDecision(action="mine", reasoning="test")
        result = await bm.broadcast(decision)

        assert isinstance(result, BroadcastResult)
        assert result.total_listeners == 2
        assert result.success_count == 2
        assert result.failure_count == 0
        assert len(m1.received) == 1
        assert len(m2.received) == 1

    async def test_broadcast_handles_listener_failure(self):
        bm = BroadcastManager()
        good = MockModule("good")
        bad = MockModule("bad", fail=True)
        bm.register(good)
        bm.register(bad)

        decision = CCDecision(action="talk")
        result = await bm.broadcast(decision)

        assert result.success_count == 1
        assert result.failure_count == 1
        assert "bad" in result.errors

    async def test_broadcast_empty_listeners(self):
        bm = BroadcastManager()
        decision = CCDecision(action="idle")
        result = await bm.broadcast(decision)

        assert result.total_listeners == 0
        assert result.success_count == 0

    async def test_latest_decision_updated(self):
        bm = BroadcastManager()
        assert bm.latest_decision is None

        decision = CCDecision(action="explore")
        await bm.broadcast(decision)

        assert bm.latest_decision is decision


class TestControllerWithBroadcast:
    """Integration-style tests: CC tick with broadcast listeners."""

    async def test_tick_broadcasts_to_listeners(self, mock_sas: MockSAS):
        bm = BroadcastManager()
        listener = MockModule("output")
        bm.register(listener)

        cc = CognitiveController(llm=MockLLM(), broadcast_manager=bm)
        result = await cc.tick(mock_sas)

        assert result.success
        assert result.data["broadcast_success"] == 1
        assert len(listener.received) == 1
        assert listener.received[0].action == "mine"
