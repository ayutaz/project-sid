"""Tests for Item Collection Benchmark."""

from __future__ import annotations

import time
from datetime import datetime

from piano.eval.items import (
    DependencyNode,
    ItemCollectionBenchmark,
    ItemSnapshot,
    MinecraftDependencyTree,
)


class TestItemSnapshot:
    """Tests for ItemSnapshot model."""

    def test_creation_with_defaults(self) -> None:
        """Test snapshot creation with default values."""
        snapshot = ItemSnapshot()
        assert snapshot.inventory == {}
        assert snapshot.unique_count == 0
        assert isinstance(snapshot.timestamp, datetime)

    def test_creation_with_inventory(self) -> None:
        """Test snapshot creation with inventory data."""
        inventory = {"oak_log": 5, "dirt": 10}
        snapshot = ItemSnapshot(inventory=inventory, unique_count=2)
        assert snapshot.inventory == inventory
        assert snapshot.unique_count == 2


class TestDependencyNode:
    """Tests for DependencyNode model."""

    def test_creation_minimal(self) -> None:
        """Test node creation with minimal parameters."""
        node = DependencyNode(item_name="dirt", depth=0)
        assert node.item_name == "dirt"
        assert node.depth == 0
        assert node.requires == []

    def test_creation_with_dependencies(self) -> None:
        """Test node creation with dependencies."""
        node = DependencyNode(item_name="stick", depth=2, requires=["oak_planks"])
        assert node.item_name == "stick"
        assert node.depth == 2
        assert node.requires == ["oak_planks"]


class TestMinecraftDependencyTree:
    """Tests for MinecraftDependencyTree."""

    def test_tree_contains_items(self) -> None:
        """Test that dependency tree contains expected items."""
        assert "oak_log" in MinecraftDependencyTree.TREE
        assert "wooden_pickaxe" in MinecraftDependencyTree.TREE
        assert "iron_ingot" in MinecraftDependencyTree.TREE
        assert "diamond_pickaxe" in MinecraftDependencyTree.TREE

    def test_get_depth_basic_item(self) -> None:
        """Test depth retrieval for basic items."""
        assert MinecraftDependencyTree.get_depth("oak_log") == 0
        assert MinecraftDependencyTree.get_depth("dirt") == 0

    def test_get_depth_crafted_item(self) -> None:
        """Test depth retrieval for crafted items."""
        assert MinecraftDependencyTree.get_depth("oak_planks") == 1
        assert MinecraftDependencyTree.get_depth("stick") == 2
        assert MinecraftDependencyTree.get_depth("wooden_pickaxe") == 3

    def test_get_depth_unknown_item(self) -> None:
        """Test depth for unknown items returns 0."""
        assert MinecraftDependencyTree.get_depth("unknown_item") == 0
        assert MinecraftDependencyTree.get_depth("fake_block") == 0

    def test_dependency_chain_correctness(self) -> None:
        """Test that dependency depths increase correctly."""
        # oak_log (0) → oak_planks (1) → stick (2) → wooden_pickaxe (3)
        assert MinecraftDependencyTree.get_depth("oak_log") < MinecraftDependencyTree.get_depth(
            "oak_planks"
        )
        assert MinecraftDependencyTree.get_depth("oak_planks") < MinecraftDependencyTree.get_depth(
            "stick"
        )
        assert MinecraftDependencyTree.get_depth("stick") < MinecraftDependencyTree.get_depth(
            "wooden_pickaxe"
        )

    def test_tool_progression_depths(self) -> None:
        """Test that tool progression has increasing depths."""
        # wooden_pickaxe (3) < stone_pickaxe (5) < iron_pickaxe (8)
        wooden_depth = MinecraftDependencyTree.get_depth("wooden_pickaxe")
        stone_depth = MinecraftDependencyTree.get_depth("stone_pickaxe")
        iron_depth = MinecraftDependencyTree.get_depth("iron_pickaxe")

        assert wooden_depth < stone_depth < iron_depth

    def test_iron_ingot_deeper_than_basics(self) -> None:
        """Test that iron_ingot has appropriate depth."""
        # iron_ingot should be deeper than wooden_pickaxe but shallower than diamond
        iron_ingot_depth = MinecraftDependencyTree.get_depth("iron_ingot")
        wooden_depth = MinecraftDependencyTree.get_depth("wooden_pickaxe")
        diamond_depth = MinecraftDependencyTree.get_depth("diamond")

        assert wooden_depth < iron_ingot_depth < diamond_depth

    def test_get_max_depth(self) -> None:
        """Test maximum depth calculation."""
        max_depth = MinecraftDependencyTree.get_max_depth()
        assert max_depth > 0
        # diamond_pickaxe should be the deepest at depth 11
        assert max_depth == 11

    def test_dependency_requirements(self) -> None:
        """Test that nodes have correct dependency requirements."""
        iron_pickaxe = MinecraftDependencyTree.TREE["iron_pickaxe"]
        assert "iron_ingot" in iron_pickaxe.requires
        assert "stick" in iron_pickaxe.requires
        assert "crafting_table" in iron_pickaxe.requires


class TestItemCollectionBenchmark:
    """Tests for ItemCollectionBenchmark."""

    def test_initialization(self) -> None:
        """Test benchmark initialization."""
        benchmark = ItemCollectionBenchmark()
        assert benchmark.get_unique_count() == 0
        assert benchmark.get_unique_items() == set()
        assert benchmark.get_progression_curve() == []

    def test_record_snapshot_single(self) -> None:
        """Test recording a single snapshot."""
        benchmark = ItemCollectionBenchmark()
        inventory = {"oak_log": 5, "dirt": 3}

        benchmark.record_snapshot(inventory)

        assert benchmark.get_unique_count() == 2
        assert benchmark.get_unique_items() == {"oak_log", "dirt"}

    def test_record_snapshot_multiple(self) -> None:
        """Test recording multiple snapshots."""
        benchmark = ItemCollectionBenchmark()

        benchmark.record_snapshot({"oak_log": 5})
        time.sleep(0.01)  # Small delay to ensure different timestamps
        benchmark.record_snapshot({"oak_log": 6, "dirt": 2})
        time.sleep(0.01)
        benchmark.record_snapshot({"oak_log": 7, "dirt": 3, "sand": 1})

        assert benchmark.get_unique_count() == 3
        assert benchmark.get_unique_items() == {"oak_log", "dirt", "sand"}

    def test_duplicate_items_dont_increase_count(self) -> None:
        """Test that duplicate items don't increase unique count."""
        benchmark = ItemCollectionBenchmark()

        benchmark.record_snapshot({"oak_log": 5})
        initial_count = benchmark.get_unique_count()

        benchmark.record_snapshot({"oak_log": 10})
        assert benchmark.get_unique_count() == initial_count

    def test_progression_curve_increases(self) -> None:
        """Test that progression curve shows increasing unique count."""
        benchmark = ItemCollectionBenchmark()

        benchmark.record_snapshot({"oak_log": 1})
        time.sleep(0.01)
        benchmark.record_snapshot({"oak_log": 1, "dirt": 1})
        time.sleep(0.01)
        benchmark.record_snapshot({"oak_log": 1, "dirt": 1, "sand": 1})

        curve = benchmark.get_progression_curve()
        assert len(curve) == 3

        # Check that times are increasing
        times = [t for t, _ in curve]
        assert times == sorted(times)

        # Check that unique counts are non-decreasing
        counts = [c for _, c in curve]
        assert counts == sorted(counts)
        assert counts == [1, 2, 3]

    def test_max_depth_reached_empty(self) -> None:
        """Test max depth with no items."""
        benchmark = ItemCollectionBenchmark()
        assert benchmark.get_max_depth_reached() == 0

    def test_max_depth_reached_basic_items(self) -> None:
        """Test max depth with basic items."""
        benchmark = ItemCollectionBenchmark()
        benchmark.record_snapshot({"oak_log": 5, "dirt": 3})

        assert benchmark.get_max_depth_reached() == 0  # Both are depth 0

    def test_max_depth_reached_progression(self) -> None:
        """Test max depth increases with tool progression."""
        benchmark = ItemCollectionBenchmark()

        # Start with basic items
        benchmark.record_snapshot({"oak_log": 1})
        assert benchmark.get_max_depth_reached() == 0

        # Add planks (depth 1)
        benchmark.record_snapshot({"oak_log": 1, "oak_planks": 4})
        assert benchmark.get_max_depth_reached() == 1

        # Add wooden pickaxe (depth 3)
        benchmark.record_snapshot({"oak_log": 1, "oak_planks": 4, "wooden_pickaxe": 1})
        assert benchmark.get_max_depth_reached() == 3

        # Add iron ingot (depth 7)
        benchmark.record_snapshot(
            {"oak_log": 1, "oak_planks": 4, "wooden_pickaxe": 1, "iron_ingot": 2}
        )
        assert benchmark.get_max_depth_reached() == 7

    def test_saturation_detection_no_saturation(self) -> None:
        """Test saturation detection when not saturated."""
        benchmark = ItemCollectionBenchmark()

        # Continuously add new items (no saturation)
        for i in range(10):
            benchmark.record_snapshot({f"item_{i}": 1})
            time.sleep(0.01)

        saturation_time = benchmark.get_saturation_time(window_seconds=0.5, new_item_threshold=2)
        # Should not detect saturation when continuously adding items
        assert saturation_time is None

    def test_saturation_detection_with_saturation(self) -> None:
        """Test saturation detection when collection plateaus."""
        benchmark = ItemCollectionBenchmark()

        # Add items rapidly
        benchmark.record_snapshot({"item_1": 1})
        time.sleep(0.05)
        benchmark.record_snapshot({"item_1": 1, "item_2": 1})
        time.sleep(0.05)
        benchmark.record_snapshot({"item_1": 1, "item_2": 1, "item_3": 1})

        # Pause and then only add one more item (saturation)
        time.sleep(0.7)
        benchmark.record_snapshot({"item_1": 1, "item_2": 1, "item_3": 1, "item_4": 1})

        saturation_time = benchmark.get_saturation_time(window_seconds=0.5, new_item_threshold=1)
        # Should detect saturation
        assert saturation_time is not None
        assert saturation_time > 0

    def test_saturation_detection_insufficient_data(self) -> None:
        """Test saturation with insufficient snapshots."""
        benchmark = ItemCollectionBenchmark()
        benchmark.record_snapshot({"item_1": 1})

        saturation_time = benchmark.get_saturation_time()
        assert saturation_time is None

    def test_meets_paper_baseline_success(self) -> None:
        """Test baseline check when requirements are met."""
        benchmark = ItemCollectionBenchmark()

        # Simulate collecting 17 unique items in 30 minutes
        items = {f"item_{i}": 1 for i in range(17)}
        benchmark.record_snapshot(items)

        assert benchmark.meets_paper_baseline(duration_minutes=30, min_items=15) is True

    def test_meets_paper_baseline_failure(self) -> None:
        """Test baseline check when requirements are not met."""
        benchmark = ItemCollectionBenchmark()

        # Only collect 10 items (below threshold of 15)
        items = {f"item_{i}": 1 for i in range(10)}
        benchmark.record_snapshot(items)

        assert benchmark.meets_paper_baseline(duration_minutes=30, min_items=15) is False

    def test_meets_paper_baseline_empty(self) -> None:
        """Test baseline check with no data."""
        benchmark = ItemCollectionBenchmark()
        assert benchmark.meets_paper_baseline() is False

    def test_summary_empty_benchmark(self) -> None:
        """Test summary with empty benchmark returns sensible defaults."""
        benchmark = ItemCollectionBenchmark()
        summary = benchmark.get_summary()

        assert summary["unique_count"] == 0
        assert summary["max_depth_reached"] == 0
        assert summary["saturation_time_seconds"] is None
        assert summary["total_duration_seconds"] == 0.0
        assert summary["snapshot_count"] == 0
        assert summary["meets_baseline"] is False
        assert summary["progression_curve"] == []

    def test_summary_with_data(self) -> None:
        """Test summary includes all expected keys and values."""
        benchmark = ItemCollectionBenchmark()

        benchmark.record_snapshot({"oak_log": 1})
        time.sleep(0.01)
        benchmark.record_snapshot({"oak_log": 1, "oak_planks": 4})
        time.sleep(0.01)
        benchmark.record_snapshot({"oak_log": 1, "oak_planks": 4, "stick": 2})

        summary = benchmark.get_summary()

        # Check all expected keys exist
        assert "unique_count" in summary
        assert "max_depth_reached" in summary
        assert "saturation_time_seconds" in summary
        assert "total_duration_seconds" in summary
        assert "snapshot_count" in summary
        assert "meets_baseline" in summary
        assert "progression_curve" in summary

        # Check values make sense
        assert summary["unique_count"] == 3
        assert summary["max_depth_reached"] == 2  # stick has depth 2
        assert summary["snapshot_count"] == 3
        assert summary["total_duration_seconds"] > 0
        assert len(summary["progression_curve"]) == 3

    def test_reset(self) -> None:
        """Test reset clears all data."""
        benchmark = ItemCollectionBenchmark()

        # Add some data
        benchmark.record_snapshot({"oak_log": 5, "dirt": 3, "sand": 2})
        time.sleep(0.01)
        benchmark.record_snapshot({"oak_log": 6, "dirt": 4, "sand": 3, "gravel": 1})

        assert benchmark.get_unique_count() > 0

        # Reset
        benchmark.reset()

        # Verify everything is cleared
        assert benchmark.get_unique_count() == 0
        assert benchmark.get_unique_items() == set()
        assert benchmark.get_progression_curve() == []
        assert benchmark.get_max_depth_reached() == 0

        summary = benchmark.get_summary()
        assert summary["unique_count"] == 0
        assert summary["snapshot_count"] == 0


class TestSaturationTimePerformance:
    """Tests for the O(N) saturation time optimization."""

    def test_saturation_time_cumulative_correctness(self) -> None:
        """Verify the O(N) implementation produces correct results."""
        benchmark = ItemCollectionBenchmark()

        # Add items rapidly
        benchmark.record_snapshot({"item_1": 1})
        time.sleep(0.05)
        benchmark.record_snapshot({"item_1": 1, "item_2": 1})
        time.sleep(0.05)
        benchmark.record_snapshot({"item_1": 1, "item_2": 1, "item_3": 1})

        # Pause then add one more (saturation)
        time.sleep(0.7)
        benchmark.record_snapshot({"item_1": 1, "item_2": 1, "item_3": 1, "item_4": 1})

        saturation = benchmark.get_saturation_time(window_seconds=0.5, new_item_threshold=1)
        assert saturation is not None
        assert saturation > 0

    def test_saturation_time_many_snapshots(self) -> None:
        """Test saturation detection with many snapshots (O(N) check)."""
        benchmark = ItemCollectionBenchmark()

        # Add unique items continuously for first 50 snapshots
        for i in range(50):
            benchmark.record_snapshot({f"item_{i}": 1})
            time.sleep(0.01)

        # Then stop adding new items for another 50 snapshots
        time.sleep(0.7)
        for _ in range(50):
            benchmark.record_snapshot({f"item_{j}": 1 for j in range(50)})
            time.sleep(0.01)

        saturation = benchmark.get_saturation_time(window_seconds=0.5, new_item_threshold=2)
        assert saturation is not None


class TestItemCollectionIntegration:
    """Integration tests for complete workflows."""

    def test_realistic_collection_sequence(self) -> None:
        """Test a realistic item collection sequence."""
        benchmark = ItemCollectionBenchmark()

        # Stage 1: Gather basic resources
        benchmark.record_snapshot({"oak_log": 10, "dirt": 5})
        time.sleep(0.01)

        # Stage 2: Craft basic items
        benchmark.record_snapshot({"oak_log": 8, "dirt": 5, "oak_planks": 8, "crafting_table": 1})
        time.sleep(0.01)

        # Stage 3: Create tools
        benchmark.record_snapshot(
            {
                "oak_log": 8,
                "dirt": 5,
                "oak_planks": 4,
                "crafting_table": 1,
                "stick": 4,
                "wooden_pickaxe": 1,
            }
        )
        time.sleep(0.01)

        # Stage 4: Mine stone
        benchmark.record_snapshot(
            {
                "oak_log": 8,
                "dirt": 5,
                "oak_planks": 4,
                "crafting_table": 1,
                "stick": 4,
                "wooden_pickaxe": 1,
                "cobblestone": 20,
                "coal": 5,
            }
        )

        # Verify progression
        assert benchmark.get_unique_count() == 8
        assert benchmark.get_max_depth_reached() == 4  # cobblestone/coal at depth 4

        # Check progression curve is smooth
        curve = benchmark.get_progression_curve()
        assert len(curve) == 4
        counts = [c for _, c in curve]
        assert counts == [2, 4, 6, 8]  # Increasing count

    def test_paper_baseline_scenario(self) -> None:
        """Test scenario matching paper's baseline experiment."""
        benchmark = ItemCollectionBenchmark()

        # Simulate 30 minutes of collection (17 unique items)
        items_collected = [
            "oak_log",
            "dirt",
            "sand",
            "gravel",
            "oak_planks",
            "stick",
            "crafting_table",
            "wooden_pickaxe",
            "cobblestone",
            "coal",
            "furnace",
            "stone_pickaxe",
            "raw_iron",
            "iron_ore",
            "iron_ingot",
            "iron_pickaxe",
            "diamond_ore",
        ]

        inventory = {item: 1 for item in items_collected}
        benchmark.record_snapshot(inventory)

        # Should meet baseline
        assert benchmark.meets_paper_baseline(duration_minutes=30, min_items=15)

        # Should have progressed to depth 9 (diamond_ore)
        assert benchmark.get_max_depth_reached() == 9

        # Should have 17 unique items
        assert benchmark.get_unique_count() == 17
