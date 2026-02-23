"""Template-based compression for SAS snapshots.

Compresses the full SAS state into a concise text prompt suitable for
the Cognitive Controller's LLM call, reducing token usage while
preserving decision-relevant information.

Reference: docs/implementation/03-cognitive-controller.md Section 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompressionResult:
    """Result of compressing a SAS snapshot."""

    text: str
    token_estimate: int
    sections: dict[str, str] = field(default_factory=dict)
    retention_score: float = 1.0  # 0.0-1.0, information retention estimate


class TemplateCompressor:
    """Compress SAS snapshots into concise text using section templates.

    Each SAS section (percepts, goals, social, plans, action_history,
    memory) is summarised via a dedicated template method, then assembled
    into a single prompt string.
    """

    # Rough chars-per-token ratio for estimation (GPT-family average).
    _CHARS_PER_TOKEN = 4.0

    def compress(self, snapshot: dict[str, Any]) -> CompressionResult:
        """Compress a full SAS snapshot into a text prompt.

        Args:
            snapshot: Raw SAS snapshot dict (from ``SharedAgentState.snapshot``).

        Returns:
            CompressionResult with the compressed text and metrics.
        """
        sections: dict[str, str] = {}

        sections["percepts"] = self._compress_percepts(snapshot.get("percepts", {}))
        sections["goals"] = self._compress_goals(snapshot.get("goals", {}))
        sections["social"] = self._compress_social(snapshot.get("social", {}))
        sections["plans"] = self._compress_plans(snapshot.get("plans", {}))
        sections["action_history"] = self._compress_action_history(
            snapshot.get("action_history", [])
        )
        sections["memory"] = self._compress_memory(
            snapshot.get("working_memory", []),
            snapshot.get("stm", []),
        )

        text = self._assemble(sections)
        token_estimate = self.estimate_tokens(text)
        retention = self._estimate_retention(snapshot, sections)

        return CompressionResult(
            text=text,
            token_estimate=token_estimate,
            sections=sections,
            retention_score=retention,
        )

    # --- Section templates ---------------------------------------------------

    def _compress_percepts(self, percepts: dict[str, Any]) -> str:
        """Compress perception data into a situation summary."""
        if not percepts:
            return "No perception data."

        pos = percepts.get("position", {})
        pos_str = f"({pos.get('x', 0):.0f}, {pos.get('y', 0):.0f}, {pos.get('z', 0):.0f})"

        health = percepts.get("health", 20.0)
        hunger = percepts.get("hunger", 20.0)

        # Inventory: top 5 items by count
        inv = percepts.get("inventory", {})
        top_items = sorted(inv.items(), key=lambda kv: kv[1], reverse=True)[:5]
        inv_str = ", ".join(f"{k}x{v}" for k, v in top_items) if top_items else "empty"

        nearby = percepts.get("nearby_players", [])
        nearby_str = ", ".join(nearby[:5]) if nearby else "none"

        weather = percepts.get("weather", "clear")
        tod = percepts.get("time_of_day", 0)

        # Recent chat (last 3)
        chats = percepts.get("chat_messages", [])
        chat_lines: list[str] = []
        for msg in chats[-3:]:
            sender = msg.get("sender", "?")
            text = msg.get("message", msg.get("text", ""))
            chat_lines.append(f"  {sender}: {text}")
        chat_str = "\n".join(chat_lines) if chat_lines else "  (none)"

        return (
            f"Location: {pos_str} | HP: {health:.0f} | Hunger: {hunger:.0f}"
            f" | Weather: {weather} | Time: {tod}\n"
            f"Inventory: {inv_str}\n"
            f"Nearby players: {nearby_str}\n"
            f"Recent chat:\n{chat_str}"
        )

    def _compress_goals(self, goals: dict[str, Any]) -> str:
        """Compress goal data."""
        if not goals:
            return "No goals."

        current = goals.get("current_goal", "")
        stack = goals.get("goal_stack", [])
        sub = goals.get("sub_goals", [])

        lines = []
        if current:
            lines.append(f"Primary: {current}")
        for g in stack[:3]:
            lines.append(f"- {g}")
        if sub:
            lines.append(f"Sub-goals: {', '.join(sub[:3])}")
        return "\n".join(lines) if lines else "No goals."

    def _compress_social(self, social: dict[str, Any]) -> str:
        """Compress social awareness data."""
        if not social:
            return "No social data."

        lines: list[str] = []

        # Relationships: top 3 by absolute affinity
        rels = social.get("relationships", {})
        top_rels = sorted(rels.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        if top_rels:
            lines.append(
                "Relationships: "
                + ", ".join(f"{name}({score:+.1f})" for name, score in top_rels)
            )

        # Emotions
        emotions = social.get("emotions", {})
        if emotions:
            emo_parts = [f"{e}={v:.0f}" for e, v in emotions.items() if v > 2.0]
            if emo_parts:
                lines.append("Emotions: " + ", ".join(emo_parts))

        # Recent interactions (last 2)
        interactions = social.get("recent_interactions", [])
        for inter in interactions[-2:]:
            partner = inter.get("partner", inter.get("agent", "?"))
            action = inter.get("action", inter.get("type", "?"))
            lines.append(f"- Interaction with {partner}: {action}")

        return "\n".join(lines) if lines else "No notable social context."

    def _compress_plans(self, plans: dict[str, Any]) -> str:
        """Compress plan data."""
        if not plans:
            return "No active plan."

        status = plans.get("plan_status", "idle")
        steps = plans.get("current_plan", [])
        current_step = plans.get("current_step", 0)

        if status == "idle" or not steps:
            return "No active plan."

        # Show a window around current step
        total = len(steps)
        window_start = max(0, current_step - 1)
        window_end = min(total, current_step + 3)

        lines = [f"Plan ({status}): step {current_step + 1}/{total}"]
        for i in range(window_start, window_end):
            marker = ">>>" if i == current_step else "   "
            lines.append(f"{marker} {i + 1}. {steps[i]}")

        return "\n".join(lines)

    def _compress_action_history(self, history: list[dict[str, Any]]) -> str:
        """Compress action history (last 3 actions)."""
        if not history:
            return "No recent actions."

        lines: list[str] = []
        for entry in history[-3:]:
            action = entry.get("action", "?")
            success = entry.get("success", True)
            result_icon = "OK" if success else "FAIL"
            actual = entry.get("actual_result", "")
            line = f"- [{result_icon}] {action}"
            if actual:
                line += f" -> {actual}"
            lines.append(line)

        return "\n".join(lines)

    def _compress_memory(
        self,
        working_memory: list[dict[str, Any]],
        stm: list[dict[str, Any]],
    ) -> str:
        """Compress memory entries (WM + top STM by importance)."""
        lines: list[str] = []

        # Working memory (all, should be small)
        for entry in working_memory[:5]:
            content = entry.get("content", "")
            if content:
                lines.append(f"- [WM] {content}")

        # STM: top 3 by importance
        sorted_stm = sorted(stm, key=lambda e: e.get("importance", 0.0), reverse=True)
        for entry in sorted_stm[:3]:
            content = entry.get("content", "")
            if content:
                lines.append(f"- [STM] {content}")

        return "\n".join(lines) if lines else "No relevant memories."

    # --- Assembly & metrics ---------------------------------------------------

    def _assemble(self, sections: dict[str, str]) -> str:
        """Assemble compressed sections into the CC prompt."""
        header_map = {
            "percepts": "## Current Situation",
            "goals": "## Goals",
            "social": "## Social Context",
            "plans": "## Plan Status",
            "action_history": "## Recent Actions",
            "memory": "## Relevant Memories",
        }

        parts: list[str] = []
        for key in ("percepts", "goals", "social", "plans", "action_history", "memory"):
            header = header_map.get(key, f"## {key}")
            body = sections.get(key, "")
            if body:
                parts.append(f"{header}\n{body}")

        return "\n\n".join(parts)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length.

        Uses a simple chars-per-token heuristic.
        """
        if not text:
            return 0
        return max(1, int(len(text) / self._CHARS_PER_TOKEN))

    def _estimate_retention(
        self,
        snapshot: dict[str, Any],
        sections: dict[str, str],
    ) -> float:
        """Estimate information retention score (0.0-1.0).

        Heuristic: ratio of non-empty compressed sections to total
        available snapshot sections, weighted by section importance.
        """
        weights = {
            "percepts": 0.25,
            "goals": 0.20,
            "social": 0.15,
            "plans": 0.15,
            "action_history": 0.15,
            "memory": 0.10,
        }

        # Map snapshot keys to section keys
        snapshot_key_map = {
            "percepts": "percepts",
            "goals": "goals",
            "social": "social",
            "plans": "plans",
            "action_history": "action_history",
            "working_memory": "memory",
            "stm": "memory",
        }

        available_weight = 0.0
        retained_weight = 0.0

        seen_sections: set[str] = set()
        for snap_key, sec_key in snapshot_key_map.items():
            if sec_key in seen_sections:
                continue
            seen_sections.add(sec_key)

            snap_data = snapshot.get(snap_key)
            weight = weights.get(sec_key, 0.1)

            has_data = bool(snap_data)
            if isinstance(snap_data, dict):
                has_data = any(
                    v for v in snap_data.values()
                    if v and v != 0 and v != 0.0
                )
            elif isinstance(snap_data, list):
                has_data = len(snap_data) > 0

            if has_data:
                available_weight += weight
                # Check compressed section has meaningful content
                compressed = sections.get(sec_key, "")
                placeholder_phrases = {"no ", "none", "empty", "(none)"}
                is_placeholder = any(
                    compressed.lower().startswith(p) or compressed.lower() == p
                    for p in placeholder_phrases
                )
                if compressed and not is_placeholder:
                    retained_weight += weight

        if available_weight == 0.0:
            return 1.0  # No data to lose
        return min(1.0, retained_weight / available_weight)
