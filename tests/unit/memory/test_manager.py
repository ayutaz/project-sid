"""Unit tests for MemoryManager."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from piano.core.types import ActionHistoryEntry, MemoryEntry, ModuleTier, PerceptData
from piano.memory.manager import MemoryManager, _percepts_summary
from piano.memory.stm import ShortTermMemory
from piano.memory.working import WorkingMemory

from .conftest import InMemorySAS

# --- helpers ---


def _entry(
    content: str = "test",
    category: str = "perception",
    importance: float = 0.5,
    age_minutes: float = 0.0,
) -> MemoryEntry:
    ts = datetime.now(UTC) - timedelta(minutes=age_minutes)
    return MemoryEntry(
        content=content,
        category=category,
        importance=importance,
        timestamp=ts,
        source_module="test",
    )


def _action_entry(
    action: str = "mine",
    success: bool = True,
    age_minutes: float = 0.0,
) -> ActionHistoryEntry:
    ts = datetime.now(UTC) - timedelta(minutes=age_minutes)
    return ActionHistoryEntry(
        timestamp=ts,
        action=action,
        expected_result="ok",
        actual_result="ok" if success else "failed",
        success=success,
    )


# --- module properties ---


class TestMemoryManagerProperties:
    def test_name(self) -> None:
        manager = MemoryManager()
        assert manager.name == "memory_manager"

    def test_tier(self) -> None:
        manager = MemoryManager()
        assert manager.tier == ModuleTier.MID

    def test_wm_accessor(self) -> None:
        wm = WorkingMemory()
        manager = MemoryManager(wm=wm)
        assert manager.wm is wm

    def test_stm_accessor(self) -> None:
        stm = ShortTermMemory()
        manager = MemoryManager(stm=stm)
        assert manager.stm is stm

    def test_default_wm_and_stm_created(self) -> None:
        manager = MemoryManager()
        assert isinstance(manager.wm, WorkingMemory)
        assert isinstance(manager.stm, ShortTermMemory)


# --- tick() basic functionality ---


class TestMemoryManagerTick:
    @pytest.mark.asyncio
    async def test_tick_adds_percepts_to_wm(self) -> None:
        sas = InMemorySAS()
        await sas.update_percepts(
            PerceptData(
                position={"x": 10, "y": 64, "z": 20},
                health=18.0,
                nearby_players=["alice", "bob"],
            )
        )
        manager = MemoryManager()
        result = await manager.tick(sas)

        assert result.success
        assert manager.wm.size == 1
        entries = manager.wm.get_by_category("perception")
        assert len(entries) == 1
        assert "pos=(10.0,64.0,20.0)" in entries[0].content
        assert "nearby=['alice', 'bob']" in entries[0].content

    @pytest.mark.asyncio
    async def test_tick_adds_action_to_stm(self) -> None:
        sas = InMemorySAS()
        action = _action_entry(action="mine_block", success=True)
        await sas.add_action(action)
        manager = MemoryManager()
        result = await manager.tick(sas)

        assert result.success
        assert manager.stm.size == 1
        entries = manager.stm.search_by_category("action")
        assert len(entries) == 1
        assert "mine_block" in entries[0].content

    @pytest.mark.asyncio
    async def test_tick_syncs_wm_to_sas(self) -> None:
        sas = InMemorySAS()
        manager = MemoryManager()
        await manager.tick(sas)

        # WM should be synced to SAS
        stored = await sas.get_working_memory()
        assert len(stored) > 0

    @pytest.mark.asyncio
    async def test_tick_returns_correct_module_result(self) -> None:
        sas = InMemorySAS()
        manager = MemoryManager()
        result = await manager.tick(sas)

        assert result.module_name == "memory_manager"
        assert result.tier == ModuleTier.MID
        assert result.success
        assert "added" in result.data

    @pytest.mark.asyncio
    async def test_tick_returns_error_on_exception(self) -> None:
        # Use a SAS that will raise an exception
        class FailingSAS(InMemorySAS):
            async def get_percepts(self) -> PerceptData:
                raise RuntimeError("Test failure")

        sas = FailingSAS()
        manager = MemoryManager()
        result = await manager.tick(sas)

        assert not result.success
        assert result.error is not None
        assert "Test failure" in result.error

    @pytest.mark.asyncio
    async def test_tick_handles_empty_sas(self) -> None:
        """Tick should work even when SAS has no actions."""
        sas = InMemorySAS()
        manager = MemoryManager()
        result = await manager.tick(sas)

        assert result.success
        # Only percept entry added, no action
        assert result.data["added"] == 1
        assert manager.wm.size == 1
        assert manager.stm.size == 0

    @pytest.mark.asyncio
    async def test_tick_with_failed_action_has_higher_importance(self) -> None:
        """Failed actions should have importance 0.5 vs 0.3 for success."""
        sas = InMemorySAS()
        failed_action = _action_entry(action="craft_fail", success=False)
        await sas.add_action(failed_action)
        manager = MemoryManager()
        await manager.tick(sas)

        entries = manager.stm.search_by_category("action")
        assert len(entries) == 1
        assert entries[0].importance == 0.5

    @pytest.mark.asyncio
    async def test_tick_with_successful_action_has_lower_importance(self) -> None:
        """Successful actions should have importance 0.3."""
        sas = InMemorySAS()
        success_action = _action_entry(action="craft_success", success=True)
        await sas.add_action(success_action)
        manager = MemoryManager()
        await manager.tick(sas)

        entries = manager.stm.search_by_category("action")
        assert len(entries) == 1
        assert entries[0].importance == 0.3


# --- multiple ticks ---


class TestMemoryManagerMultipleTicks:
    @pytest.mark.asyncio
    async def test_multiple_ticks_accumulate(self) -> None:
        sas = InMemorySAS()
        manager = MemoryManager()

        # Tick 1
        await sas.update_percepts(PerceptData(position={"x": 0, "y": 0, "z": 0}))
        await manager.tick(sas)

        # Tick 2
        await sas.update_percepts(PerceptData(position={"x": 1, "y": 0, "z": 0}))
        await manager.tick(sas)

        # Tick 3
        await sas.update_percepts(PerceptData(position={"x": 2, "y": 0, "z": 0}))
        await manager.tick(sas)

        assert manager.wm.size == 3

    @pytest.mark.asyncio
    async def test_wm_eviction_on_capacity_exceeded(self) -> None:
        """WM should evict when capacity is exceeded."""
        wm = WorkingMemory(capacity=2)
        manager = MemoryManager(wm=wm)
        sas = InMemorySAS()

        # Tick 1
        await manager.tick(sas)
        assert manager.wm.size == 1

        # Tick 2
        await manager.tick(sas)
        assert manager.wm.size == 2

        # Tick 3 - should trigger eviction
        await manager.tick(sas)
        assert manager.wm.size == 2  # capacity maintained


# --- store() method ---


class TestMemoryManagerStore:
    def test_store_to_stm_default(self) -> None:
        manager = MemoryManager()
        entry = _entry("test content", category="social")
        manager.store(entry)

        assert manager.stm.size == 1
        assert manager.wm.size == 0

    def test_store_to_wm_explicit(self) -> None:
        manager = MemoryManager()
        entry = _entry("test content", category="perception")
        manager.store(entry, to_wm=True)

        assert manager.wm.size == 1
        assert manager.stm.size == 0

    def test_store_multiple_to_stm(self) -> None:
        manager = MemoryManager()
        manager.store(_entry("a"))
        manager.store(_entry("b"))
        manager.store(_entry("c"))

        assert manager.stm.size == 3

    def test_store_multiple_to_wm(self) -> None:
        manager = MemoryManager()
        manager.store(_entry("a"), to_wm=True)
        manager.store(_entry("b"), to_wm=True)

        assert manager.wm.size == 2


# --- lifecycle ---


class TestMemoryManagerLifecycle:
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """initialize() should not raise (default is no-op)."""
        manager = MemoryManager()
        await manager.initialize()  # Should succeed

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """shutdown() should not raise (default is no-op)."""
        manager = MemoryManager()
        await manager.shutdown()  # Should succeed


# --- integration with SAS ---


class TestMemoryManagerSASIntegration:
    @pytest.mark.asyncio
    async def test_full_integration_percepts_and_actions(self) -> None:
        """Test a complete flow: percepts + actions from SAS."""
        sas = InMemorySAS()
        await sas.update_percepts(
            PerceptData(
                position={"x": 100, "y": 64, "z": 200},
                health=15.0,
                nearby_players=["charlie"],
            )
        )
        await sas.add_action(_action_entry(action="dig", success=True))

        manager = MemoryManager()
        result = await manager.tick(sas)

        assert result.success
        assert result.data["added"] == 2

        # Check WM
        wm_entries = manager.wm.get_all()
        assert len(wm_entries) == 1
        assert "pos=(100.0,64.0,200.0)" in wm_entries[0].content

        # Check STM
        stm_entries = manager.stm.get_recent()
        assert len(stm_entries) == 1
        assert "dig" in stm_entries[0].content

        # Check SAS was updated
        stored_wm = await sas.get_working_memory()
        assert len(stored_wm) == 1

    @pytest.mark.asyncio
    async def test_percepts_summary_with_empty_position(self) -> None:
        """Test percepts summary when position is empty."""
        sas = InMemorySAS()
        await sas.update_percepts(PerceptData(health=20.0))

        manager = MemoryManager()
        await manager.tick(sas)

        entries = manager.wm.get_all()
        assert len(entries) == 1
        # Should at least have hp
        assert "hp=20" in entries[0].content


# --- _percepts_summary helper ---


class TestPerceptsSummary:
    def test_percepts_summary_full(self) -> None:
        percepts = PerceptData(
            position={"x": 10, "y": 20, "z": 30},
            nearby_players=["alice", "bob"],
            health=18.5,
        )
        summary = _percepts_summary(percepts)
        assert "pos=(10.0,20.0,30.0)" in summary
        assert "nearby=['alice', 'bob']" in summary
        assert "hp=18.5" in summary

    def test_percepts_summary_minimal(self) -> None:
        percepts = PerceptData(health=20.0)
        summary = _percepts_summary(percepts)
        assert "hp=20" in summary

    def test_percepts_summary_invalid_input(self) -> None:
        """Invalid input should return string representation with fallback."""
        summary = _percepts_summary("invalid")
        # The function has a fallback that tries to extract health, returns "hp=?"
        assert "hp=" in summary or summary == "invalid"

    def test_percepts_summary_none_nearby_players(self) -> None:
        percepts = PerceptData(
            position={"x": 5, "y": 10, "z": 15},
            health=10.0,
        )
        summary = _percepts_summary(percepts)
        assert "pos=(5.0,10.0,15.0)" in summary
        assert "hp=10" in summary
        assert "nearby" not in summary


# --- repr ---


class TestMemoryManagerRepr:
    def test_repr(self) -> None:
        manager = MemoryManager()
        r = repr(manager)
        assert "MemoryManager" in r
        assert "memory_manager" in r
        assert "mid" in r
