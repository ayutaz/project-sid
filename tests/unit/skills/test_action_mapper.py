"""Tests for action mapper module."""

from __future__ import annotations

from unittest.mock import MagicMock

from piano.skills.action_mapper import (
    ACTION_TO_SKILL,
    create_full_registry,
    map_action,
)


class TestMapAction:
    def test_map_move_action(self) -> None:
        assert map_action("move") == "move_to"

    def test_map_mine_action(self) -> None:
        assert map_action("mine") == "mine_block"

    def test_map_craft_action(self) -> None:
        assert map_action("craft") == "craft_item"

    def test_map_chat_action(self) -> None:
        assert map_action("chat") == "chat"

    def test_map_idle_returns_none(self) -> None:
        assert map_action("idle") is None

    def test_map_wait_returns_none(self) -> None:
        assert map_action("wait") is None

    def test_map_unknown_returns_none(self) -> None:
        assert map_action("fly") is None

    def test_map_unknown_logs_warning(self, monkeypatch) -> None:
        """Unmapped actions log a structlog warning."""
        import piano.skills.action_mapper as am

        mock_logger = MagicMock()
        monkeypatch.setattr(am, "logger", mock_logger)

        map_action("unknown_action_xyz")

        mock_logger.warning.assert_called_once()
        call_kwargs = mock_logger.warning.call_args
        assert "unknown_action_xyz" in str(call_kwargs)

    def test_map_gather_aliases_mine(self) -> None:
        assert map_action("gather") == "mine_block"
        assert map_action("dig") == "mine_block"

    def test_map_follow_action(self) -> None:
        assert map_action("follow") == "follow_agent"

    def test_map_trade_action(self) -> None:
        assert map_action("trade") == "trade_items"

    def test_map_explore_aliases_move(self) -> None:
        assert map_action("explore") == "move_to"

    def test_action_to_skill_dict_has_none_for_noop_actions(self) -> None:
        noop_actions = ["idle", "wait", "think", "observe"]
        for action in noop_actions:
            assert ACTION_TO_SKILL[action] is None


class TestCreateFullRegistry:
    def test_create_full_registry_has_basic_skills(self) -> None:
        registry = create_full_registry()
        basic_skills = ["move_to", "mine_block", "craft_item", "chat", "look_at"]
        for skill_name in basic_skills:
            assert skill_name in registry

    def test_create_full_registry_has_social_skills(self) -> None:
        registry = create_full_registry()
        social_skills = [
            "trade_items",
            "follow_agent",
            "gift_item",
            "vote",
            "send_message",
        ]
        for skill_name in social_skills:
            assert skill_name in registry

    def test_create_full_registry_total_count(self) -> None:
        registry = create_full_registry()
        # 7 basic + 9 social = 16
        assert len(registry) == 16
