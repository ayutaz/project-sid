"""Tests for LLM-based role inference pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from piano.eval.role_inference import (
    AgentRole,
    RoleHistory,
    RoleInferencePipeline,
    RoleInferenceRequest,
    RoleInferenceResult,
)
from piano.llm.mock import MockLLMProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_role_response(
    role: str = "farmer",
    confidence: float = 0.9,
    reasoning: str = "Test reasoning",
) -> str:
    """Build a mock JSON response for role inference."""
    return json.dumps({"role": role, "confidence": confidence, "reasoning": reasoning})


def _make_request(
    agent_id: str = "agent-001",
    goals: list[str] | None = None,
) -> RoleInferenceRequest:
    """Create a RoleInferenceRequest with defaults."""
    return RoleInferenceRequest(
        agent_id=agent_id,
        recent_goals=goals or ["gather wheat", "plant seeds"],
    )


def _result(
    agent_id: str = "a1",
    role: AgentRole = AgentRole.FARMER,
    confidence: float = 0.9,
    reasoning: str = "",
    timestamp: datetime | None = None,
) -> RoleInferenceResult:
    """Create a RoleInferenceResult with defaults."""
    kwargs: dict = {
        "agent_id": agent_id,
        "inferred_role": role,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    if timestamp is not None:
        kwargs["timestamp"] = timestamp
    return RoleInferenceResult(**kwargs)


# ---------------------------------------------------------------------------
# AgentRole enum
# ---------------------------------------------------------------------------


class TestAgentRole:
    """Tests for AgentRole enum."""

    def test_all_roles_defined(self):
        """All expected roles exist in the enum."""
        expected = {
            "farmer",
            "miner",
            "engineer",
            "guard",
            "explorer",
            "blacksmith",
            "scout",
            "strategist",
            "curator",
            "collector",
            "other",
        }
        actual = {r.value for r in AgentRole}
        assert actual == expected

    def test_role_is_str_enum(self):
        """AgentRole values are strings usable directly."""
        assert AgentRole.FARMER == "farmer"
        assert str(AgentRole.MINER) == "miner"


# ---------------------------------------------------------------------------
# RoleInferenceRequest
# ---------------------------------------------------------------------------


class TestRoleInferenceRequest:
    """Tests for RoleInferenceRequest model."""

    def test_creation_with_defaults(self):
        req = RoleInferenceRequest(agent_id="a1", recent_goals=["mine iron"])
        assert req.agent_id == "a1"
        assert req.recent_goals == ["mine iron"]
        assert req.timestamp is not None

    def test_max_five_goals(self):
        """recent_goals allows up to 5 items."""
        goals = [f"goal-{i}" for i in range(5)]
        req = RoleInferenceRequest(agent_id="a1", recent_goals=goals)
        assert len(req.recent_goals) == 5

    def test_rejects_more_than_five_goals(self):
        """recent_goals rejects more than 5 items."""
        goals = [f"goal-{i}" for i in range(6)]
        with pytest.raises(ValueError):
            RoleInferenceRequest(agent_id="a1", recent_goals=goals)

    def test_empty_goals(self):
        req = RoleInferenceRequest(agent_id="a1", recent_goals=[])
        assert req.recent_goals == []


# ---------------------------------------------------------------------------
# RoleInferenceResult
# ---------------------------------------------------------------------------


class TestRoleInferenceResult:
    """Tests for RoleInferenceResult model."""

    def test_creation(self):
        result = RoleInferenceResult(
            agent_id="a1",
            inferred_role=AgentRole.MINER,
            confidence=0.85,
            reasoning="Agent focuses on mining ores",
        )
        assert result.agent_id == "a1"
        assert result.inferred_role == AgentRole.MINER
        assert result.confidence == 0.85

    def test_confidence_bounds(self):
        """Confidence must be in [0, 1]."""
        RoleInferenceResult(
            agent_id="a1",
            inferred_role=AgentRole.OTHER,
            confidence=0.0,
            reasoning="low",
        )
        RoleInferenceResult(
            agent_id="a1",
            inferred_role=AgentRole.OTHER,
            confidence=1.0,
            reasoning="high",
        )
        with pytest.raises(ValueError):
            RoleInferenceResult(
                agent_id="a1",
                inferred_role=AgentRole.OTHER,
                confidence=-0.1,
                reasoning="bad",
            )
        with pytest.raises(ValueError):
            RoleInferenceResult(
                agent_id="a1",
                inferred_role=AgentRole.OTHER,
                confidence=1.1,
                reasoning="bad",
            )


# ---------------------------------------------------------------------------
# RoleInferencePipeline
# ---------------------------------------------------------------------------


class TestRoleInferencePipeline:
    """Tests for the RoleInferencePipeline."""

    @pytest.fixture()
    def mock_llm(self) -> MockLLMProvider:
        llm = MockLLMProvider()
        llm.set_default_response(_mock_role_response("farmer", 0.9, "Goals focus on agriculture"))
        return llm

    @pytest.fixture()
    def pipeline(self, mock_llm: MockLLMProvider) -> RoleInferencePipeline:
        return RoleInferencePipeline(mock_llm)

    async def test_infer_role_basic(
        self,
        pipeline: RoleInferencePipeline,
        mock_llm: MockLLMProvider,
    ):
        """Basic single-agent inference returns correct role."""
        request = _make_request(goals=["gather wheat", "plant seeds", "harvest crops"])
        result = await pipeline.infer_role(request)

        assert result.agent_id == "agent-001"
        assert result.inferred_role == AgentRole.FARMER
        assert result.confidence == 0.9
        assert result.reasoning == "Goals focus on agriculture"
        assert len(mock_llm.call_history) == 1

    async def test_infer_role_miner(
        self,
        pipeline: RoleInferencePipeline,
        mock_llm: MockLLMProvider,
    ):
        """Inference correctly picks up miner role."""
        mock_llm.set_default_response(_mock_role_response("miner", 0.95, "Agent mines ores"))
        request = _make_request(goals=["mine iron ore", "dig tunnel"])
        result = await pipeline.infer_role(request)

        assert result.inferred_role == AgentRole.MINER
        assert result.confidence == 0.95

    async def test_infer_role_empty_goals(self, pipeline: RoleInferencePipeline):
        """Pipeline handles empty goals gracefully."""
        request = _make_request(goals=[])
        result = await pipeline.infer_role(request)
        # Should still return a valid result
        assert result.agent_id == "agent-001"
        assert isinstance(result.inferred_role, AgentRole)

    async def test_infer_role_json_parse_error(
        self,
        pipeline: RoleInferencePipeline,
        mock_llm: MockLLMProvider,
    ):
        """Falls back to OTHER when LLM returns invalid JSON."""
        mock_llm.set_default_response("not valid json at all!")
        request = _make_request()
        result = await pipeline.infer_role(request)

        assert result.inferred_role == AgentRole.OTHER
        assert result.confidence == 0.0
        assert "Failed to parse" in result.reasoning

    async def test_infer_role_unknown_role_fallback(
        self,
        pipeline: RoleInferencePipeline,
        mock_llm: MockLLMProvider,
    ):
        """Falls back to OTHER for unrecognized role strings."""
        mock_llm.set_default_response(_mock_role_response("wizard", 0.8, "Magical role"))
        request = _make_request()
        result = await pipeline.infer_role(request)

        assert result.inferred_role == AgentRole.OTHER

    async def test_infer_role_confidence_clamped(
        self,
        pipeline: RoleInferencePipeline,
        mock_llm: MockLLMProvider,
    ):
        """Confidence clamped to [0, 1] if LLM returns out-of-range."""
        mock_llm.set_default_response(
            json.dumps(
                {
                    "role": "guard",
                    "confidence": 1.5,
                    "reasoning": "Very confident",
                }
            )
        )
        request = _make_request()
        result = await pipeline.infer_role(request)

        assert result.confidence == 1.0

    async def test_infer_role_confidence_negative_clamped(
        self,
        pipeline: RoleInferencePipeline,
        mock_llm: MockLLMProvider,
    ):
        """Negative confidence is clamped to 0."""
        mock_llm.set_default_response(
            json.dumps(
                {
                    "role": "guard",
                    "confidence": -0.5,
                    "reasoning": "Unsure",
                }
            )
        )
        request = _make_request()
        result = await pipeline.infer_role(request)

        assert result.confidence == 0.0

    async def test_infer_role_llm_error(self, mock_llm: MockLLMProvider):
        """Returns fallback result when LLM raises an exception."""

        class FailingLLM:
            async def complete(self, request):
                raise RuntimeError("LLM is down")

        pipeline = RoleInferencePipeline(FailingLLM())  # type: ignore[arg-type]
        request = _make_request()
        result = await pipeline.infer_role(request)

        assert result.inferred_role == AgentRole.OTHER
        assert result.confidence == 0.0
        assert "LLM call failed" in result.reasoning

    async def test_infer_role_prompt_contains_goals(
        self,
        pipeline: RoleInferencePipeline,
        mock_llm: MockLLMProvider,
    ):
        """The prompt sent to LLM includes agent goals."""
        request = _make_request(
            agent_id="agent-007",
            goals=["build castle", "craft stone bricks"],
        )
        await pipeline.infer_role(request)

        assert len(mock_llm.call_history) == 1
        prompt = mock_llm.call_history[0].prompt
        assert "agent-007" in prompt
        assert "build castle" in prompt
        assert "craft stone bricks" in prompt

    async def test_infer_role_uses_json_mode(
        self,
        pipeline: RoleInferencePipeline,
        mock_llm: MockLLMProvider,
    ):
        """LLM request sets json_mode=True."""
        request = _make_request()
        await pipeline.infer_role(request)
        assert mock_llm.call_history[0].json_mode is True

    async def test_infer_roles_batch(
        self,
        pipeline: RoleInferencePipeline,
        mock_llm: MockLLMProvider,
    ):
        """Batch inference returns results in correct order."""
        requests = [
            _make_request(agent_id="a1", goals=["farm wheat"]),
            _make_request(agent_id="a2", goals=["mine diamonds"]),
            _make_request(agent_id="a3", goals=["guard gate"]),
        ]
        results = await pipeline.infer_roles_batch(requests)

        assert len(results) == 3
        assert results[0].agent_id == "a1"
        assert results[1].agent_id == "a2"
        assert results[2].agent_id == "a3"
        assert len(mock_llm.call_history) == 3

    async def test_infer_roles_batch_empty(self, pipeline: RoleInferencePipeline):
        """Batch with empty list returns empty list."""
        results = await pipeline.infer_roles_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# get_role_distribution (static method)
# ---------------------------------------------------------------------------


class TestGetRoleDistribution:
    """Tests for RoleInferencePipeline.get_role_distribution."""

    def test_distribution_basic(self):
        results = [
            _result("a1", AgentRole.FARMER, 0.9),
            _result("a2", AgentRole.FARMER, 0.8),
            _result("a3", AgentRole.MINER, 0.85),
            _result("a4", AgentRole.GUARD, 0.7),
        ]
        dist = RoleInferencePipeline.get_role_distribution(results)

        assert dist["farmer"] == 0.5
        assert dist["miner"] == 0.25
        assert dist["guard"] == 0.25
        assert abs(sum(dist.values()) - 1.0) < 1e-10

    def test_distribution_empty(self):
        assert RoleInferencePipeline.get_role_distribution([]) == {}

    def test_distribution_single_role(self):
        results = [_result(f"a{i}", AgentRole.EXPLORER, 0.9) for i in range(5)]
        dist = RoleInferencePipeline.get_role_distribution(results)
        assert dist == {"explorer": 1.0}


# ---------------------------------------------------------------------------
# RoleHistory
# ---------------------------------------------------------------------------


class TestRoleHistory:
    """Tests for RoleHistory tracking."""

    @pytest.fixture()
    def history(self) -> RoleHistory:
        return RoleHistory()

    def test_add_and_retrieve(self, history: RoleHistory):
        history.add_result(_result("a1", AgentRole.FARMER, 0.9, "farming"))
        assert len(history.results) == 1
        assert history.results[0].agent_id == "a1"

    def test_results_returns_copy(self, history: RoleHistory):
        """results property returns a copy, not the internal list."""
        history.add_result(_result("a1", AgentRole.FARMER, 0.9))

        results_copy = history.results
        results_copy.clear()
        # Internal list should be unaffected
        assert len(history.results) == 1

    def test_get_role_transitions(self, history: RoleHistory):
        t1 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        t2 = datetime(2024, 1, 1, 1, 0, tzinfo=UTC)
        t3 = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)

        history.add_result(_result("a1", AgentRole.FARMER, 0.9, timestamp=t1))
        history.add_result(_result("a1", AgentRole.MINER, 0.8, timestamp=t2))
        history.add_result(_result("a1", AgentRole.MINER, 0.85, timestamp=t3))
        # Different agent - should not appear
        history.add_result(_result("a2", AgentRole.GUARD, 0.7, timestamp=t2))

        transitions = history.get_role_transitions("a1")
        assert len(transitions) == 3
        assert transitions[0] == (t1, AgentRole.FARMER)
        assert transitions[1] == (t2, AgentRole.MINER)
        assert transitions[2] == (t3, AgentRole.MINER)

    def test_get_role_transitions_empty(self, history: RoleHistory):
        assert history.get_role_transitions("nonexistent") == []

    def test_get_role_transitions_sorted(self, history: RoleHistory):
        """Results are sorted by timestamp even if added out of order."""
        t1 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        t2 = datetime(2024, 1, 1, 1, 0, tzinfo=UTC)

        # Add in reverse order
        history.add_result(_result("a1", AgentRole.GUARD, 0.9, timestamp=t2))
        history.add_result(_result("a1", AgentRole.FARMER, 0.9, timestamp=t1))

        transitions = history.get_role_transitions("a1")
        assert transitions[0][0] < transitions[1][0]
        assert transitions[0][1] == AgentRole.FARMER
        assert transitions[1][1] == AgentRole.GUARD

    def test_role_persistence_perfect(self, history: RoleHistory):
        """Persistence is 1.0 when role never changes."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(5):
            history.add_result(
                _result(
                    "a1",
                    AgentRole.FARMER,
                    0.9,
                    timestamp=base + timedelta(hours=i),
                )
            )

        assert history.get_role_persistence("a1") == 1.0

    def test_role_persistence_zero(self, history: RoleHistory):
        """Persistence is 0.0 when role changes every step."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        roles = [
            AgentRole.FARMER,
            AgentRole.MINER,
            AgentRole.GUARD,
            AgentRole.EXPLORER,
        ]
        for i, role in enumerate(roles):
            history.add_result(
                _result(
                    "a1",
                    role,
                    0.9,
                    timestamp=base + timedelta(hours=i),
                )
            )

        assert history.get_role_persistence("a1") == 0.0

    def test_role_persistence_partial(self, history: RoleHistory):
        """Persistence reflects partial stability."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        # farmer, farmer, miner (2 transitions: 1 maintained, 1 changed)
        history.add_result(
            _result(
                "a1",
                AgentRole.FARMER,
                0.9,
                timestamp=base,
            )
        )
        history.add_result(
            _result(
                "a1",
                AgentRole.FARMER,
                0.9,
                timestamp=base + timedelta(hours=1),
            )
        )
        history.add_result(
            _result(
                "a1",
                AgentRole.MINER,
                0.9,
                timestamp=base + timedelta(hours=2),
            )
        )

        # 1 maintained out of 2 transitions = 0.5
        assert history.get_role_persistence("a1") == 0.5

    def test_role_persistence_single_observation(self, history: RoleHistory):
        """Single observation has perfect persistence."""
        history.add_result(_result("a1", AgentRole.FARMER, 0.9))
        assert history.get_role_persistence("a1") == 1.0

    def test_role_persistence_no_observations(self, history: RoleHistory):
        """No observations means perfect persistence (trivially)."""
        assert history.get_role_persistence("unknown-agent") == 1.0
