"""Skill registry for managing and retrieving executable skills.

Skills are registered with a name, execution function, and optional
parameter schema. The SkillExecutor uses the registry to look up
skills by name when processing CC decisions.

Reference: docs/implementation/05-minecraft-platform.md Section 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass(frozen=True)
class Skill:
    """A registered skill that can be executed by the SkillExecutor.

    Attributes:
        name: Unique skill identifier (e.g., "move_to", "mine_block").
        execute_fn: Async function that performs the skill action.
        params_schema: JSON-schema-like dict describing expected parameters.
        description: Human-readable description of what the skill does.
    """

    name: str
    execute_fn: Callable[..., Awaitable[dict[str, Any]]]
    params_schema: dict[str, Any] = field(default_factory=dict)
    description: str = ""


class SkillRegistry:
    """Registry for skill lookup and management.

    Provides registration, retrieval, and listing of skills that
    the SkillExecutor can dispatch to.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(
        self,
        name: str,
        skill_fn: Callable[..., Awaitable[dict[str, Any]]],
        params_schema: dict[str, Any] | None = None,
        description: str = "",
    ) -> None:
        """Register a skill.

        Args:
            name: Unique skill name.
            skill_fn: Async callable that executes the skill.
            params_schema: Optional JSON-schema-like parameter description.
            description: Human-readable description.

        Raises:
            ValueError: If a skill with the same name is already registered.
        """
        if name in self._skills:
            raise ValueError(f"Skill '{name}' is already registered")
        self._skills[name] = Skill(
            name=name,
            execute_fn=skill_fn,
            params_schema=params_schema or {},
            description=description,
        )

    def get(self, name: str) -> Skill:
        """Get a skill by name.

        Args:
            name: The skill name to look up.

        Returns:
            The matching Skill.

        Raises:
            KeyError: If the skill is not found.
        """
        try:
            return self._skills[name]
        except KeyError:
            raise KeyError(f"Skill '{name}' not found in registry") from None

    def list_skills(self) -> list[str]:
        """List all registered skill names.

        Returns:
            Sorted list of skill names.
        """
        return sorted(self._skills.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)
