"""Tests for the CognitiveController and BroadcastManager."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from piano.cc.broadcast import BroadcastManager, BroadcastResult
from piano.cc.controller import CC_SYSTEM_PROMPT, CC_TIMEOUT, VALID_ACTIONS, CognitiveController
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


class SlowLLM:
    """Mock LLM provider that takes too long to respond."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(120)  # way longer than timeout
        return LLMResponse(content='{"action": "idle"}', model="mock-model")


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


class TestValidActionsMatchActionMapper:
    """Test that VALID_ACTIONS includes all ACTION_TO_SKILL keys."""

    def test_all_action_mapper_keys_in_valid_actions(self):
        from piano.skills.action_mapper import ACTION_TO_SKILL

        missing = set(ACTION_TO_SKILL.keys()) - VALID_ACTIONS
        assert not missing, f"Actions in ACTION_TO_SKILL but not VALID_ACTIONS: {missing}"

    def test_valid_actions_superset(self):
        from piano.skills.action_mapper import ACTION_TO_SKILL

        # VALID_ACTIONS may contain extra actions (e.g. "defend", "vote")
        # but must cover all ACTION_TO_SKILL keys
        for action in ACTION_TO_SKILL:
            assert action in VALID_ACTIONS


class TestFallbackResultBroadcasts:
    """Test that _fallback_result broadcasts the decision to listeners."""

    async def test_fallback_broadcasts_previous_decision(self, mock_sas: MockSAS):
        bm = BroadcastManager()
        listener = MockModule("output")
        bm.register(listener)

        cc = CognitiveController(llm=MockLLM(), broadcast_manager=bm)

        # First tick succeeds and broadcasts
        await cc.tick(mock_sas)
        assert len(listener.received) == 1

        # Swap to failing LLM - fallback should still broadcast
        cc._llm = FailingLLM()
        result = await cc.tick(mock_sas)

        assert result.success
        assert result.data.get("fallback") is True
        # Listener should have received a second broadcast from the fallback
        assert len(listener.received) == 2

    async def test_fallback_no_broadcast_when_no_previous_decision(self, mock_sas: MockSAS):
        bm = BroadcastManager()
        listener = MockModule("output")
        bm.register(listener)

        cc = CognitiveController(llm=FailingLLM(), broadcast_manager=bm)
        result = await cc.tick(mock_sas)

        assert not result.success
        # No previous decision, so no broadcast
        assert len(listener.received) == 0


class TestLLMTimeout:
    """Test that LLM calls are subject to a timeout."""

    async def test_slow_llm_triggers_timeout(self, mock_sas: MockSAS):
        cc = CognitiveController(llm=SlowLLM())

        async def fast_timeout_call_llm(compression):
            request = LLMRequest(
                prompt=compression.text,
                system_prompt=CC_SYSTEM_PROMPT,
                tier=ModuleTier.MID,
                temperature=cc._temperature,
                max_tokens=cc._max_tokens,
                json_mode=True,
            )
            return await asyncio.wait_for(
                cc._llm.complete(request), timeout=0.05
            )

        cc._call_llm = fast_timeout_call_llm
        result = await cc.tick(mock_sas)

        # Should fall back due to timeout
        assert not result.success
        assert result.error is not None


class TestInvalidActionFallbackToIdle:
    """Test that invalid actions consistently fall back to idle."""

    async def test_invalid_action_becomes_idle_no_previous(self, mock_sas: MockSAS):
        llm = MockLLM({"action": "INVALID_ACTION", "reasoning": "test"})
        cc = CognitiveController(llm=llm)
        result = await cc.tick(mock_sas)

        assert result.success
        assert result.data["action"] == "idle"

    async def test_invalid_action_becomes_idle_with_previous(self, mock_sas: MockSAS):
        """Even when a previous decision exists, invalid action falls back to idle, not raises."""
        good_llm = MockLLM({"action": "mine", "reasoning": "first"})
        cc = CognitiveController(llm=good_llm)
        await cc.tick(mock_sas)

        # Second tick with invalid action
        cc._llm = MockLLM({"action": "BOGUS_ACTION", "reasoning": "second"})
        result = await cc.tick(mock_sas)

        assert result.success
        assert result.data["action"] == "idle"


class TestBroadcastConcurrentModificationSafety:
    """Test that broadcast is safe against concurrent listener modifications."""

    async def test_register_during_broadcast(self):
        """Registering a new listener during broadcast should not cause RuntimeError."""
        bm = BroadcastManager()

        class ModifyingModule:
            """Module that registers a new listener during on_broadcast."""

            def __init__(self, bm: BroadcastManager):
                self._bm = bm

            @property
            def name(self) -> str:
                return "modifier"

            @property
            def tier(self) -> ModuleTier:
                return ModuleTier.FAST

            async def on_broadcast(self, decision: CCDecision) -> None:
                # Register a new module during broadcast
                new = MockModule("late_joiner")
                self._bm.register(new)

        modifier = ModifyingModule(bm)
        bm.register(modifier)

        decision = CCDecision(action="mine", reasoning="test")
        # Should not raise RuntimeError
        result = await bm.broadcast(decision)

        assert result.success_count == 1
        # The late_joiner was added during broadcast, now registered
        assert "late_joiner" in bm.listener_names


class TestCCSystemPromptActionParams:
    """Test that CC_SYSTEM_PROMPT includes action parameter specifications."""

    def test_prompt_includes_action_params(self):
        assert "Available actions and their required parameters:" in CC_SYSTEM_PROMPT

    def test_prompt_includes_move_params(self):
        assert '"x": number, "y": number, "z": number' in CC_SYSTEM_PROMPT

    def test_prompt_includes_craft_params(self):
        assert '"item": string, "count": number' in CC_SYSTEM_PROMPT

    def test_prompt_includes_explore_params(self):
        assert '"direction": string' in CC_SYSTEM_PROMPT
        assert '"distance": number' in CC_SYSTEM_PROMPT

    def test_prompt_includes_farm_params(self):
        assert '"plant"|"harvest"' in CC_SYSTEM_PROMPT

    def test_prompt_includes_chat_params(self):
        assert '- chat: {"message": string}' in CC_SYSTEM_PROMPT

    def test_prompt_includes_deposit_withdraw(self):
        assert "- deposit:" in CC_SYSTEM_PROMPT
        assert "- withdraw:" in CC_SYSTEM_PROMPT

    def test_prompt_includes_send_message(self):
        assert "- send_message:" in CC_SYSTEM_PROMPT


class TestExtendedValidActions:
    """Test that VALID_ACTIONS covers all necessary actions."""

    def test_farm_actions_in_valid_actions(self):
        assert "farm" in VALID_ACTIONS
        assert "plant" in VALID_ACTIONS
        assert "harvest" in VALID_ACTIONS

    def test_social_actions_in_valid_actions(self):
        assert "send_message" in VALID_ACTIONS
        assert "unfollow" in VALID_ACTIONS
        assert "request_help" in VALID_ACTIONS
        assert "form_group" in VALID_ACTIONS
        assert "leave_group" in VALID_ACTIONS

    def test_chest_actions_in_valid_actions(self):
        assert "deposit" in VALID_ACTIONS
        assert "withdraw" in VALID_ACTIONS

    def test_smelt_in_valid_actions(self):
        assert "smelt" in VALID_ACTIONS


class TestCCTimeout:
    """Test CC_TIMEOUT constant."""

    def test_cc_timeout_value(self):
        assert CC_TIMEOUT == 45.0

    def test_cc_timeout_greater_than_provider_default(self):
        # Provider default is 30s, CC_TIMEOUT should be > 30s
        assert CC_TIMEOUT > 30.0


class TestFallbackEscalation:
    """Test that consecutive fallbacks escalate to forced idle."""

    async def test_escalation_after_max_consecutive_fallbacks(self, mock_sas: MockSAS):
        cc = CognitiveController(llm=FailingLLM())

        # First tick: error (no last_decision)
        result1 = await cc.tick(mock_sas)
        assert not result1.success
        assert cc._consecutive_fallbacks == 1

        # Set a last_decision manually so fallback returns success
        cc._last_decision = CCDecision(action="mine", reasoning="prev")

        # 2nd fallback
        result2 = await cc.tick(mock_sas)
        assert result2.success
        assert result2.data.get("fallback") is True
        assert cc._consecutive_fallbacks == 2

        # 3rd fallback
        result3 = await cc.tick(mock_sas)
        assert result3.success
        assert cc._consecutive_fallbacks == 3

        # 4th fallback -> escalation (> max_consecutive_fallbacks=3)
        result4 = await cc.tick(mock_sas)
        assert result4.success
        assert result4.data.get("escalated") is True
        assert result4.data["action"] == "idle"
        # Counter should be reset after escalation
        assert cc._consecutive_fallbacks == 0

    async def test_success_resets_consecutive_fallbacks(self, mock_sas: MockSAS):
        good_llm = MockLLM()
        cc = CognitiveController(llm=good_llm)

        # Simulate some fallbacks
        cc._consecutive_fallbacks = 2

        # Successful tick should reset counter
        await cc.tick(mock_sas)
        assert cc._consecutive_fallbacks == 0
