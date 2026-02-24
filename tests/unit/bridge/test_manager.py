"""Tests for BridgeManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from piano.bridge.manager import BridgeManager


@pytest.fixture()
def manager() -> BridgeManager:
    return BridgeManager(
        host="localhost",
        base_command_port=5555,
        base_event_port=5556,
        connect_timeout_s=5.0,
        connect_retry_count=3,
    )


class TestCreateBridge:
    def test_port_assignment(self, manager: BridgeManager) -> None:
        """Port is base + index * 2."""
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            manager.create_bridge("agent-0", 0)
            mock_cls.assert_called_once_with(
                command_url="tcp://localhost:5555",
                event_url="tcp://localhost:5556",
                tls_config=None,
            )

    def test_port_assignment_index_3(self, manager: BridgeManager) -> None:
        """Index 3 -> base + 6."""
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            manager.create_bridge("agent-3", 3)
            mock_cls.assert_called_once_with(
                command_url="tcp://localhost:5561",
                event_url="tcp://localhost:5562",
                tls_config=None,
            )

    def test_create_multiple_bridges(self, manager: BridgeManager) -> None:
        """Create 3 bridges with unique ports."""
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            manager.create_bridge("a0", 0)
            manager.create_bridge("a1", 1)
            manager.create_bridge("a2", 2)

            calls = mock_cls.call_args_list
            ports = [(c.kwargs["command_url"], c.kwargs["event_url"]) for c in calls]
            assert len(set(ports)) == 3
            assert ports[0] == ("tcp://localhost:5555", "tcp://localhost:5556")
            assert ports[1] == ("tcp://localhost:5557", "tcp://localhost:5558")
            assert ports[2] == ("tcp://localhost:5559", "tcp://localhost:5560")

    def test_duplicate_agent_raises_valueerror(self, manager: BridgeManager) -> None:
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            manager.create_bridge("agent-0", 0)
            with pytest.raises(ValueError, match="already exists"):
                manager.create_bridge("agent-0", 1)


class TestGetBridge:
    def test_returns_correct_client(self, manager: BridgeManager) -> None:
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            sentinel = MagicMock()
            mock_cls.return_value = sentinel
            manager.create_bridge("agent-x", 0)
            assert manager.get_bridge("agent-x") is sentinel

    def test_unknown_raises_keyerror(self, manager: BridgeManager) -> None:
        with pytest.raises(KeyError, match="No bridge"):
            manager.get_bridge("nonexistent")

    def test_bridges_property_returns_copy(self, manager: BridgeManager) -> None:
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            manager.create_bridge("a0", 0)
            bridges = manager.bridges
            assert "a0" in bridges
            bridges["a0"] = None  # type: ignore[assignment]
            assert manager.get_bridge("a0") is not None


class TestConnectAll:
    async def test_calls_connect_on_all(self, manager: BridgeManager) -> None:
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            b1, b2 = AsyncMock(), AsyncMock()
            mock_cls.side_effect = [b1, b2]
            manager.create_bridge("a0", 0)
            manager.create_bridge("a1", 1)

            results = await manager.connect_all()

            b1.connect.assert_awaited_once()
            b2.connect.assert_awaited_once()
            assert results == {"a0": True, "a1": True}

    async def test_retry_on_failure_then_success(self) -> None:
        mgr = BridgeManager(
            connect_timeout_s=1.0,
            connect_retry_count=3,
        )
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            bridge_mock = AsyncMock()
            bridge_mock.connect.side_effect = [ConnectionError("fail"), None]
            mock_cls.return_value = bridge_mock
            mgr.create_bridge("a0", 0)

            with patch("piano.bridge.manager.asyncio.sleep", new_callable=AsyncMock):
                results = await mgr.connect_all()

            assert results["a0"] is True
            assert bridge_mock.connect.await_count == 2

    async def test_all_retries_exhausted(self) -> None:
        mgr = BridgeManager(
            connect_timeout_s=1.0,
            connect_retry_count=2,
        )
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            bridge_mock = AsyncMock()
            bridge_mock.connect.side_effect = ConnectionError("fail")
            mock_cls.return_value = bridge_mock
            mgr.create_bridge("a0", 0)

            with patch("piano.bridge.manager.asyncio.sleep", new_callable=AsyncMock):
                results = await mgr.connect_all()

            assert results["a0"] is False
            assert bridge_mock.connect.await_count == 2


class TestDisconnectAll:
    async def test_disconnects_all(self, manager: BridgeManager) -> None:
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            b1, b2 = AsyncMock(), AsyncMock()
            mock_cls.side_effect = [b1, b2]
            manager.create_bridge("a0", 0)
            manager.create_bridge("a1", 1)

            await manager.disconnect_all()

            b1.disconnect.assert_awaited_once()
            b2.disconnect.assert_awaited_once()

    async def test_disconnect_continues_on_error(self, manager: BridgeManager) -> None:
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            b1 = AsyncMock()
            b1.disconnect.side_effect = RuntimeError("oops")
            b2 = AsyncMock()
            mock_cls.side_effect = [b1, b2]
            manager.create_bridge("a0", 0)
            manager.create_bridge("a1", 1)

            await manager.disconnect_all()

            b1.disconnect.assert_awaited_once()
            b2.disconnect.assert_awaited_once()


class TestHealthCheck:
    async def test_all_healthy(self, manager: BridgeManager) -> None:
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            b1, b2 = AsyncMock(), AsyncMock()
            b1.ping.return_value = True
            b2.ping.return_value = True
            mock_cls.side_effect = [b1, b2]
            manager.create_bridge("a0", 0)
            manager.create_bridge("a1", 1)

            result = await manager.health_check()
            assert result == {"a0": True, "a1": True}

    async def test_mixed_status(self, manager: BridgeManager) -> None:
        with patch("piano.bridge.manager.BridgeClient") as mock_cls:
            b1, b2 = AsyncMock(), AsyncMock()
            b1.ping.return_value = True
            b2.ping.side_effect = ConnectionError("down")
            mock_cls.side_effect = [b1, b2]
            manager.create_bridge("a0", 0)
            manager.create_bridge("a1", 1)

            result = await manager.health_check()
            assert result == {"a0": True, "a1": False}
