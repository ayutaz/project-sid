"""Skill execution module for Minecraft actions."""

from piano.skills.basic import create_default_registry
from piano.skills.executor import SkillExecutor
from piano.skills.registry import Skill, SkillRegistry

__all__ = [
    "Skill",
    "SkillExecutor",
    "SkillRegistry",
    "create_default_registry",
]
