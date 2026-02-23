"""Tests for the TemplateCompressor."""

from __future__ import annotations

import pytest

from piano.cc.compression import CompressionResult, TemplateCompressor


@pytest.fixture
def compressor() -> TemplateCompressor:
    return TemplateCompressor()


@pytest.fixture
def full_snapshot() -> dict:
    """A realistic SAS snapshot with all sections populated."""
    return {
        "percepts": {
            "position": {"x": 100.5, "y": 64.0, "z": -200.3},
            "health": 18.0,
            "hunger": 15.0,
            "inventory": {"diamond": 3, "wood": 64, "stone": 32, "iron": 10, "food": 5},
            "nearby_players": ["Alice", "Bob"],
            "weather": "rain",
            "time_of_day": 6000,
            "chat_messages": [
                {"sender": "Alice", "message": "Want to trade?"},
                {"sender": "Bob", "message": "I found a cave!"},
            ],
        },
        "goals": {
            "current_goal": "Build a house",
            "goal_stack": ["Gather wood", "Find location", "Build"],
            "sub_goals": ["Collect 20 wood planks"],
        },
        "social": {
            "relationships": {"Alice": 0.8, "Bob": 0.5, "Charlie": -0.3},
            "emotions": {"happy": 6.0, "curious": 3.0, "fear": 1.0},
            "recent_interactions": [
                {"partner": "Alice", "action": "traded items"},
            ],
        },
        "plans": {
            "current_plan": ["go to forest", "chop trees", "return to base", "craft planks"],
            "plan_status": "executing",
            "current_step": 1,
        },
        "action_history": [
            {"action": "move_to forest", "success": True, "actual_result": "arrived"},
            {"action": "mine_tree", "success": True, "actual_result": "got 4 wood"},
            {"action": "mine_tree", "success": False, "actual_result": "interrupted by mob"},
        ],
        "working_memory": [
            {"content": "Alice wants to trade diamonds", "importance": 0.9},
        ],
        "stm": [
            {"content": "Saw a village to the north", "importance": 0.7},
            {"content": "It started raining", "importance": 0.3},
            {"content": "Bob mentioned a cave nearby", "importance": 0.6},
        ],
    }


@pytest.fixture
def empty_snapshot() -> dict:
    return {}


class TestTemplateCompressor:
    """Tests for TemplateCompressor.compress."""

    def test_compress_full_snapshot(self, compressor: TemplateCompressor, full_snapshot: dict):
        result = compressor.compress(full_snapshot)

        assert isinstance(result, CompressionResult)
        assert result.text
        assert result.token_estimate > 0
        assert len(result.sections) == 6

    def test_compress_empty_snapshot(self, compressor: TemplateCompressor, empty_snapshot: dict):
        result = compressor.compress(empty_snapshot)

        assert isinstance(result, CompressionResult)
        assert result.text  # still produces template text
        assert result.token_estimate > 0

    def test_percepts_section_contains_position(
        self, compressor: TemplateCompressor, full_snapshot: dict
    ):
        result = compressor.compress(full_snapshot)
        percepts = result.sections["percepts"]

        assert "100" in percepts  # x
        assert "64" in percepts  # y
        assert "18" in percepts  # health

    def test_percepts_section_contains_chat(
        self, compressor: TemplateCompressor, full_snapshot: dict
    ):
        result = compressor.compress(full_snapshot)
        percepts = result.sections["percepts"]

        assert "Alice" in percepts
        assert "trade" in percepts.lower()

    def test_goals_section_primary(self, compressor: TemplateCompressor, full_snapshot: dict):
        result = compressor.compress(full_snapshot)
        goals = result.sections["goals"]

        assert "Build a house" in goals

    def test_social_section_relationships(
        self, compressor: TemplateCompressor, full_snapshot: dict
    ):
        result = compressor.compress(full_snapshot)
        social = result.sections["social"]

        assert "Alice" in social

    def test_plans_section_current_step(self, compressor: TemplateCompressor, full_snapshot: dict):
        result = compressor.compress(full_snapshot)
        plans = result.sections["plans"]

        assert "chop trees" in plans
        assert ">>>" in plans  # current step marker

    def test_action_history_shows_failure(
        self, compressor: TemplateCompressor, full_snapshot: dict
    ):
        result = compressor.compress(full_snapshot)
        history = result.sections["action_history"]

        assert "FAIL" in history
        assert "interrupted" in history

    def test_memory_section_includes_wm(self, compressor: TemplateCompressor, full_snapshot: dict):
        result = compressor.compress(full_snapshot)
        memory = result.sections["memory"]

        assert "WM" in memory
        assert "trade diamonds" in memory

    def test_retention_score_high_for_full_snapshot(
        self, compressor: TemplateCompressor, full_snapshot: dict
    ):
        result = compressor.compress(full_snapshot)

        # Full snapshot should have high retention
        assert result.retention_score > 0.8

    def test_retention_score_for_empty(self, compressor: TemplateCompressor, empty_snapshot: dict):
        result = compressor.compress(empty_snapshot)

        # No data to lose => retention is 1.0
        assert result.retention_score == 1.0


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_empty_text(self, compressor: TemplateCompressor):
        assert compressor.estimate_tokens("") == 0

    def test_short_text(self, compressor: TemplateCompressor):
        tokens = compressor.estimate_tokens("Hello world")
        assert tokens >= 1

    def test_scales_with_length(self, compressor: TemplateCompressor):
        short = compressor.estimate_tokens("a" * 10)
        long = compressor.estimate_tokens("a" * 100)
        assert long > short


class TestSectionCompressors:
    """Unit tests for individual section compressors."""

    def test_compress_percepts_empty(self, compressor: TemplateCompressor):
        assert compressor._compress_percepts({}) == "No perception data."

    def test_compress_goals_empty(self, compressor: TemplateCompressor):
        assert compressor._compress_goals({}) == "No goals."

    def test_compress_social_empty(self, compressor: TemplateCompressor):
        assert compressor._compress_social({}) == "No social data."

    def test_compress_plans_idle(self, compressor: TemplateCompressor):
        result = compressor._compress_plans({"plan_status": "idle"})
        assert "No active plan" in result

    def test_compress_action_history_empty(self, compressor: TemplateCompressor):
        result = compressor._compress_action_history([])
        assert "No recent actions" in result

    def test_compress_memory_empty(self, compressor: TemplateCompressor):
        result = compressor._compress_memory([], [])
        assert "No relevant memories" in result
