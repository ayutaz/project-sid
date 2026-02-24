"""Item Collection Benchmark for PIANO architecture.

Measures single-agent Minecraft item collection performance over time,
comparing against dependency tree depth and saturation metrics.
"""

from __future__ import annotations

__all__ = [
    "DependencyNode",
    "ItemCollectionBenchmark",
    "ItemSnapshot",
    "MinecraftDependencyTree",
]

from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class ItemSnapshot(BaseModel):
    """A timestamped snapshot of an agent's inventory."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    inventory: dict[str, int] = Field(default_factory=dict)
    unique_count: int = Field(
        default=0,
        description="Cumulative count of unique items seen up to this snapshot",
    )


class DependencyNode(BaseModel):
    """A node in the Minecraft item dependency tree."""

    item_name: str
    depth: int
    requires: list[str] = Field(default_factory=list)


class MinecraftDependencyTree:
    """Static Minecraft crafting dependency tree data.

    Contains ~30 common Minecraft items with crafting depths and dependencies.
    Depth 0 = items obtainable without tools (e.g., oak_log)
    Depth 1+ = items requiring progressively more advanced tools/items
    """

    TREE: ClassVar[dict[str, DependencyNode]] = {
        # Depth 0 - Raw materials (no tools needed)
        "oak_log": DependencyNode(item_name="oak_log", depth=0, requires=[]),
        "dirt": DependencyNode(item_name="dirt", depth=0, requires=[]),
        "sand": DependencyNode(item_name="sand", depth=0, requires=[]),
        "gravel": DependencyNode(item_name="gravel", depth=0, requires=[]),
        "apple": DependencyNode(item_name="apple", depth=0, requires=[]),
        # Depth 1 - Basic crafting (log → planks)
        "oak_planks": DependencyNode(item_name="oak_planks", depth=1, requires=["oak_log"]),
        "wood_planks": DependencyNode(item_name="wood_planks", depth=1, requires=["oak_log"]),
        # Depth 2 - Crafted basic items
        "stick": DependencyNode(item_name="stick", depth=2, requires=["oak_planks"]),
        "crafting_table": DependencyNode(
            item_name="crafting_table", depth=2, requires=["oak_planks"]
        ),
        # Depth 3 - Wooden tools
        "wooden_pickaxe": DependencyNode(
            item_name="wooden_pickaxe", depth=3, requires=["oak_planks", "stick", "crafting_table"]
        ),
        "wooden_axe": DependencyNode(
            item_name="wooden_axe", depth=3, requires=["oak_planks", "stick", "crafting_table"]
        ),
        "wooden_shovel": DependencyNode(
            item_name="wooden_shovel", depth=3, requires=["oak_planks", "stick", "crafting_table"]
        ),
        "wooden_sword": DependencyNode(
            item_name="wooden_sword", depth=3, requires=["oak_planks", "stick", "crafting_table"]
        ),
        # Depth 4 - Stone (requires wooden pickaxe)
        "cobblestone": DependencyNode(
            item_name="cobblestone", depth=4, requires=["wooden_pickaxe"]
        ),
        "coal": DependencyNode(item_name="coal", depth=4, requires=["wooden_pickaxe"]),
        # Depth 5 - Stone tools and furnace
        "stone_pickaxe": DependencyNode(
            item_name="stone_pickaxe", depth=5, requires=["cobblestone", "stick", "crafting_table"]
        ),
        "stone_axe": DependencyNode(
            item_name="stone_axe", depth=5, requires=["cobblestone", "stick", "crafting_table"]
        ),
        "furnace": DependencyNode(item_name="furnace", depth=5, requires=["cobblestone"]),
        # Depth 6 - Iron ore (requires stone pickaxe)
        "raw_iron": DependencyNode(item_name="raw_iron", depth=6, requires=["stone_pickaxe"]),
        "iron_ore": DependencyNode(item_name="iron_ore", depth=6, requires=["stone_pickaxe"]),
        "gold_ore": DependencyNode(item_name="gold_ore", depth=6, requires=["stone_pickaxe"]),
        # Depth 7 - Smelted items
        "iron_ingot": DependencyNode(
            item_name="iron_ingot", depth=7, requires=["raw_iron", "furnace", "coal"]
        ),
        "gold_ingot": DependencyNode(
            item_name="gold_ingot", depth=7, requires=["gold_ore", "furnace", "coal"]
        ),
        # Depth 8 - Iron tools
        "iron_pickaxe": DependencyNode(
            item_name="iron_pickaxe", depth=8, requires=["iron_ingot", "stick", "crafting_table"]
        ),
        "iron_axe": DependencyNode(
            item_name="iron_axe", depth=8, requires=["iron_ingot", "stick", "crafting_table"]
        ),
        "iron_sword": DependencyNode(
            item_name="iron_sword", depth=8, requires=["iron_ingot", "stick", "crafting_table"]
        ),
        # Depth 9 - Diamond ore (requires iron pickaxe)
        "diamond_ore": DependencyNode(item_name="diamond_ore", depth=9, requires=["iron_pickaxe"]),
        "redstone": DependencyNode(item_name="redstone", depth=9, requires=["iron_pickaxe"]),
        "lapis_lazuli": DependencyNode(
            item_name="lapis_lazuli", depth=9, requires=["iron_pickaxe"]
        ),
        # Depth 10 - Diamond
        "diamond": DependencyNode(item_name="diamond", depth=10, requires=["diamond_ore"]),
        # Depth 11 - Diamond tools (highest tech level in this tree)
        "diamond_pickaxe": DependencyNode(
            item_name="diamond_pickaxe", depth=11, requires=["diamond", "stick", "crafting_table"]
        ),
    }

    @classmethod
    def get_depth(cls, item_name: str) -> int:
        """Return the crafting depth of an item.

        Args:
            item_name: Name of the Minecraft item

        Returns:
            Depth in dependency tree, or 0 if item not found
        """
        node = cls.TREE.get(item_name)
        return node.depth if node else 0

    @classmethod
    def get_max_depth(cls) -> int:
        """Return the maximum depth in the dependency tree."""
        if not cls.TREE:
            return 0
        return max(node.depth for node in cls.TREE.values())


class ItemCollectionBenchmark:
    """Benchmark for measuring item collection performance.

    Tracks unique items collected over time, compares against Minecraft
    dependency tree progression, and calculates saturation metrics.

    Paper baseline: 15-20 unique items in 30 minutes (single agent).
    """

    def __init__(self) -> None:
        """Initialize the benchmark."""
        self._snapshots: list[ItemSnapshot] = []
        self._start_time: datetime | None = None
        self._all_items_seen: set[str] = set()

    def record_snapshot(self, inventory: dict[str, int]) -> None:
        """Record a timestamped inventory snapshot.

        Args:
            inventory: Current inventory state {item_name: count}
        """
        timestamp = datetime.now(UTC)
        if self._start_time is None:
            self._start_time = timestamp

        # Update all items seen
        self._all_items_seen.update(inventory.keys())

        snapshot = ItemSnapshot(
            timestamp=timestamp, inventory=inventory.copy(), unique_count=len(self._all_items_seen)
        )
        self._snapshots.append(snapshot)

    def get_unique_items(self) -> set[str]:
        """Get all unique items ever collected.

        Returns:
            Set of unique item names
        """
        return self._all_items_seen.copy()

    def get_unique_count(self) -> int:
        """Get count of unique items ever collected.

        Returns:
            Number of unique items
        """
        return len(self._all_items_seen)

    def get_progression_curve(self) -> list[tuple[float, int]]:
        """Get progression curve of unique items over time.

        Returns:
            List of (elapsed_seconds, unique_count) tuples
        """
        if not self._snapshots or self._start_time is None:
            return []

        curve = []
        for snapshot in self._snapshots:
            elapsed = (snapshot.timestamp - self._start_time).total_seconds()
            curve.append((elapsed, snapshot.unique_count))

        return curve

    def get_max_depth_reached(self) -> int:
        """Get the deepest item in the dependency tree that has been collected.

        Returns:
            Maximum depth reached, or 0 if no items collected
        """
        if not self._all_items_seen:
            return 0

        max_depth = 0
        for item in self._all_items_seen:
            depth = MinecraftDependencyTree.get_depth(item)
            max_depth = max(max_depth, depth)

        return max_depth

    def get_saturation_time(
        self, window_seconds: float = 600.0, new_item_threshold: int = 2
    ) -> float | None:
        """Detect when item collection rate plateaus.

        Saturation is detected when a time window contains fewer new items
        than the threshold. Uses O(N) precomputed cumulative item sets.

        Args:
            window_seconds: Time window to check (default: 600s = 10 minutes)
            new_item_threshold: Max new items to consider saturated (default: 2)

        Returns:
            Elapsed seconds when saturation occurred, or None if not saturated
        """
        if len(self._snapshots) < 2 or self._start_time is None:
            return None

        total_duration = (self._snapshots[-1].timestamp - self._start_time).total_seconds()
        if total_duration < window_seconds:
            return None

        # Precompute cumulative item sets in O(N)
        cumulative: list[set[str]] = []
        running: set[str] = set()
        for snapshot in self._snapshots:
            running = running | set(snapshot.inventory.keys())
            cumulative.append(running.copy())

        # Use a sliding window start pointer
        window_start_idx = 0

        for i in range(len(self._snapshots)):
            current_snapshot = self._snapshots[i]
            elapsed_at_snapshot = (current_snapshot.timestamp - self._start_time).total_seconds()

            if elapsed_at_snapshot < window_seconds:
                continue

            window_start_time = current_snapshot.timestamp.timestamp() - window_seconds

            # Advance window_start_idx to the first snapshot within the window
            while window_start_idx < i:
                if self._snapshots[window_start_idx].timestamp.timestamp() >= window_start_time:
                    break
                window_start_idx += 1

            # Items seen before window = cumulative set just before window_start_idx
            if window_start_idx > 0:
                items_before_window = cumulative[window_start_idx - 1]
            else:
                items_before_window = set()

            # Items in window = cumulative at i minus items before window
            items_at_i = cumulative[i]
            new_items = items_at_i - items_before_window

            if len(new_items) <= new_item_threshold:
                return elapsed_at_snapshot

        return None

    def meets_paper_baseline(self, duration_minutes: float = 30.0, min_items: int = 15) -> bool:
        """Check if performance meets paper's baseline.

        Paper baseline: 15-20 unique items in 30 minutes.

        Args:
            duration_minutes: Expected duration (default: 30 minutes)
            min_items: Minimum unique items required (default: 15)

        Returns:
            True if baseline is met
        """
        if not self._snapshots or self._start_time is None:
            return False

        # Find snapshots within duration window
        duration_seconds = duration_minutes * 60
        end_time = self._start_time.timestamp() + duration_seconds

        items_at_duration: set[str] = set()
        for snapshot in self._snapshots:
            if snapshot.timestamp.timestamp() <= end_time:
                items_at_duration.update(snapshot.inventory.keys())

        return len(items_at_duration) >= min_items

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive benchmark statistics.

        Returns:
            Dictionary containing all benchmark metrics
        """
        if not self._snapshots or self._start_time is None:
            return {
                "unique_count": 0,
                "max_depth_reached": 0,
                "saturation_time_seconds": None,
                "total_duration_seconds": 0.0,
                "snapshot_count": 0,
                "meets_baseline": False,
                "progression_curve": [],
            }

        last_snapshot = self._snapshots[-1]
        total_duration = (last_snapshot.timestamp - self._start_time).total_seconds()

        return {
            "unique_count": self.get_unique_count(),
            "max_depth_reached": self.get_max_depth_reached(),
            "saturation_time_seconds": self.get_saturation_time(),
            "total_duration_seconds": total_duration,
            "snapshot_count": len(self._snapshots),
            "meets_baseline": self.meets_paper_baseline(),
            "progression_curve": self.get_progression_curve(),
        }

    def reset(self) -> None:
        """Reset the benchmark to initial state."""
        self._snapshots.clear()
        self._start_time = None
        self._all_items_seen.clear()
