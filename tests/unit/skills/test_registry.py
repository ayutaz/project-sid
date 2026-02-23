"""Tests for SkillRegistry."""

from __future__ import annotations

import pytest

from piano.skills.registry import Skill, SkillRegistry


async def _dummy_skill(**kwargs: object) -> dict[str, object]:
    return {"success": True}


class TestSkill:
    def test_skill_is_frozen(self) -> None:
        skill = Skill(name="test", execute_fn=_dummy_skill)
        with pytest.raises(AttributeError):
            skill.name = "changed"  # type: ignore[misc]

    def test_skill_defaults(self) -> None:
        skill = Skill(name="test", execute_fn=_dummy_skill)
        assert skill.params_schema == {}
        assert skill.description == ""


class TestSkillRegistry:
    def test_register_and_get(self) -> None:
        registry = SkillRegistry()
        registry.register("move", _dummy_skill, description="Move somewhere")
        skill = registry.get("move")
        assert skill.name == "move"
        assert skill.execute_fn is _dummy_skill
        assert skill.description == "Move somewhere"

    def test_register_duplicate_raises(self) -> None:
        registry = SkillRegistry()
        registry.register("move", _dummy_skill)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("move", _dummy_skill)

    def test_get_missing_raises(self) -> None:
        registry = SkillRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_skills_sorted(self) -> None:
        registry = SkillRegistry()
        registry.register("chat", _dummy_skill)
        registry.register("move", _dummy_skill)
        registry.register("attack", _dummy_skill)
        assert registry.list_skills() == ["attack", "chat", "move"]

    def test_contains(self) -> None:
        registry = SkillRegistry()
        registry.register("move", _dummy_skill)
        assert "move" in registry
        assert "chat" not in registry

    def test_len(self) -> None:
        registry = SkillRegistry()
        assert len(registry) == 0
        registry.register("move", _dummy_skill)
        assert len(registry) == 1
