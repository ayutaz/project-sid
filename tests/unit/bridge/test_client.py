"""Tests for the BridgeClient ZMQ communication layer.

Uses unittest.mock to patch ZMQ sockets for reliable, fast unit tests.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import zmq

from piano.bridge.client import BridgeClient
from piano.bridge.types import BridgeStatus
from piano.core.types import BridgeCommand, BridgeEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_context() -> MagicMock:
    """Create a mock ZMQ async context with socket factories."""
    ctx = MagicMock()
    ctx.destroy = MagicMock()

    def _make_socket(sock_type: int) -> MagicMock:
        sock = MagicMock()
        sock.setsockopt = MagicMock()
        sock.connect = MagicMock()
        sock.subscribe = MagicMock()
        sock.close = MagicMock()
        sock.send_string = AsyncMock()
        sock.recv_json = AsyncMock(
            return_value={
                "id": "test",
                "success": True,
                "data": {},
                "error": None,
            }
        )
        sock.send_json = AsyncMock()
        return sock

    ctx.socket = MagicMock(side_effect=_make_socket)
    return ctx


@pytest.fixture
def mock_ctx() -> MagicMock:
    """Return a mock ZMQ context."""
    return _make_mock_context()


@pytest.fixture
async def connected_client(mock_ctx: MagicMock) -> tuple[BridgeClient, MagicMock, MagicMock]:
    """Return (client, cmd_socket, sub_socket) with client already connected."""
    with patch("piano.bridge.client.zmq.asyncio.Context", return_value=mock_ctx):
        client = BridgeClient(
            command_url="tcp://127.0.0.1:5555",
            event_url="tcp://127.0.0.1:5556",
            timeout_ms=2000,
        )
        await client.connect()

        # Access the actual mock objects from the client's internals
        cmd_mock = client._cmd_socket
        sub_mock = client._sub_socket

        yield client, cmd_mock, sub_mock

        await client.disconnect()


# ---------------------------------------------------------------------------
# Tests: Connection lifecycle
# ---------------------------------------------------------------------------


class TestBridgeClientConnection:
    """Tests for connect / disconnect lifecycle."""

    async def test_initial_status_is_disconnected(self) -> None:
        """New client starts disconnected."""
        client = BridgeClient()
        assert client.status == BridgeStatus.DISCONNECTED

    async def test_connect_changes_status(self, mock_ctx: MagicMock) -> None:
        """Connecting sets status to CONNECTED."""
        with patch("piano.bridge.client.zmq.asyncio.Context", return_value=mock_ctx):
            client = BridgeClient()
            await client.connect()
            assert client.status == BridgeStatus.CONNECTED
            await client.disconnect()

    async def test_disconnect_changes_status(self, connected_client) -> None:
        """Disconnecting sets status to DISCONNECTED."""
        client, _, _ = connected_client
        await client.disconnect()
        assert client.status == BridgeStatus.DISCONNECTED

    async def test_double_connect_is_idempotent(self, connected_client) -> None:
        """Calling connect when already connected is a no-op."""
        client, _, _ = connected_client
        await client.connect()
        assert client.status == BridgeStatus.CONNECTED


class TestBridgeClientSendCommand:
    """Tests for send_command and message serialization."""

    async def test_send_command_roundtrip(self, connected_client) -> None:
        """Command is sent and response received."""
        client, cmd_mock, _ = connected_client
        cmd_mock.recv_json = AsyncMock(
            return_value={
                "id": "test-id",
                "success": True,
                "data": {"moved": True},
                "error": None,
            }
        )

        cmd = BridgeCommand(action="move", params={"x": 1, "y": 2, "z": 3})
        resp = await client.send_command(cmd)

        assert resp["success"] is True
        assert resp["data"]["moved"] is True
        cmd_mock.send_string.assert_awaited_once()

    async def test_command_is_json_serialized(self, connected_client) -> None:
        """Sent payload is valid JSON matching the command."""
        client, cmd_mock, _ = connected_client

        cmd = BridgeCommand(action="chat", params={"message": "hello"})
        await client.send_command(cmd)

        payload_str = cmd_mock.send_string.call_args[0][0]
        payload = json.loads(payload_str)
        assert payload["action"] == "chat"
        assert payload["params"]["message"] == "hello"

    async def test_command_preserves_id(self, connected_client) -> None:
        """Response ID matches the command ID."""
        client, cmd_mock, _ = connected_client

        cmd = BridgeCommand(action="ping", params={})
        cmd_mock.recv_json = AsyncMock(
            return_value={
                "id": str(cmd.id),
                "success": True,
                "data": {},
                "error": None,
            }
        )

        resp = await client.send_command(cmd)
        assert resp["id"] == str(cmd.id)

    async def test_send_command_when_disconnected_raises(self) -> None:
        """Sending a command on a disconnected client raises ConnectionError."""
        client = BridgeClient()
        cmd = BridgeCommand(action="ping", params={})
        with pytest.raises(ConnectionError):
            await client.send_command(cmd)


class TestBridgeClientTimeout:
    """Tests for timeout and retry behaviour."""

    async def test_timeout_raises_after_retries(self, connected_client) -> None:
        """Client raises TimeoutError after MAX_RETRIES zmq.Again errors."""
        client, cmd_mock, _ = connected_client

        # Make the current socket and all reconnect-created sockets fail
        cmd_mock.send_string = AsyncMock(side_effect=zmq.Again("timeout"))

        # Override _reconnect_cmd_socket to just reset the same failing mock
        async def fake_reconnect() -> None:
            client._cmd_socket = cmd_mock
            client._status = BridgeStatus.CONNECTED

        client._reconnect_cmd_socket = fake_reconnect

        cmd = BridgeCommand(action="ping", params={})
        with pytest.raises(TimeoutError, match="timed out after"):
            await client.send_command(cmd)

    async def test_timeout_triggers_reconnect(self, connected_client) -> None:
        """On timeout, client transitions through RECONNECTING status."""
        client, cmd_mock, _ = connected_client

        cmd_mock.send_string = AsyncMock(side_effect=zmq.Again("timeout"))
        statuses: list[BridgeStatus] = []

        async def tracking_reconnect() -> None:
            statuses.append(client.status)
            client._cmd_socket = cmd_mock
            client._status = BridgeStatus.CONNECTED

        client._reconnect_cmd_socket = tracking_reconnect

        cmd = BridgeCommand(action="ping", params={})
        with pytest.raises(TimeoutError):
            await client.send_command(cmd)

        assert BridgeStatus.RECONNECTING in statuses


class TestPing:
    """Tests for the ping health check."""

    async def test_ping_success(self, connected_client) -> None:
        """Ping returns True when bridge responds with success=True."""
        client, cmd_mock, _ = connected_client
        cmd_mock.recv_json = AsyncMock(
            return_value={
                "id": "x",
                "success": True,
                "data": {"pong": True},
                "error": None,
            }
        )
        assert await client.ping() is True

    async def test_ping_returns_false_on_failure(self, connected_client) -> None:
        """Ping returns False when bridge responds with success=False."""
        client, cmd_mock, _ = connected_client
        cmd_mock.recv_json = AsyncMock(
            return_value={
                "id": "x",
                "success": False,
                "data": {},
                "error": "unhealthy",
            }
        )
        assert await client.ping() is False

    async def test_ping_returns_false_on_timeout(self, connected_client) -> None:
        """Ping returns False on TimeoutError."""
        client, cmd_mock, _ = connected_client

        cmd_mock.send_string = AsyncMock(side_effect=zmq.Again("timeout"))

        async def fake_reconnect() -> None:
            client._cmd_socket = cmd_mock
            client._status = BridgeStatus.CONNECTED

        client._reconnect_cmd_socket = fake_reconnect

        assert await client.ping() is False


class TestConvenienceMethods:
    """Tests for move_to, mine_block, craft_item, chat, get_position, get_inventory."""

    async def test_move_to(self, connected_client) -> None:
        """move_to sends a 'move' action with x/y/z params."""
        client, cmd_mock, _ = connected_client
        cmd_mock.recv_json = AsyncMock(
            return_value={
                "id": "x",
                "success": True,
                "data": {"x": 10, "y": 64, "z": -5},
                "error": None,
            }
        )
        resp = await client.move_to(10, 64, -5)
        assert resp["data"]["x"] == 10

        payload = json.loads(cmd_mock.send_string.call_args[0][0])
        assert payload["action"] == "move"
        assert payload["params"] == {"x": 10, "y": 64, "z": -5}

    async def test_mine_block(self, connected_client) -> None:
        """mine_block sends a 'mine' action."""
        client, cmd_mock, _ = connected_client
        await client.mine_block(5, 60, 5)

        payload = json.loads(cmd_mock.send_string.call_args[0][0])
        assert payload["action"] == "mine"
        assert payload["params"] == {"x": 5, "y": 60, "z": 5}

    async def test_craft_item(self, connected_client) -> None:
        """craft_item sends a 'craft' action with item and count."""
        client, cmd_mock, _ = connected_client
        await client.craft_item("planks", 4)

        payload = json.loads(cmd_mock.send_string.call_args[0][0])
        assert payload["action"] == "craft"
        assert payload["params"]["item"] == "planks"
        assert payload["params"]["count"] == 4

    async def test_chat(self, connected_client) -> None:
        """chat sends a 'chat' action with the message."""
        client, cmd_mock, _ = connected_client
        await client.chat("hello world")

        payload = json.loads(cmd_mock.send_string.call_args[0][0])
        assert payload["action"] == "chat"
        assert payload["params"]["message"] == "hello world"

    async def test_get_position(self, connected_client) -> None:
        """get_position sends a 'get_position' action."""
        client, cmd_mock, _ = connected_client
        cmd_mock.recv_json = AsyncMock(
            return_value={
                "id": "x",
                "success": True,
                "data": {"x": 1.5, "y": 65.0, "z": -2.3},
                "error": None,
            }
        )
        resp = await client.get_position()
        assert resp["data"]["y"] == 65.0

        payload = json.loads(cmd_mock.send_string.call_args[0][0])
        assert payload["action"] == "get_position"

    async def test_get_inventory(self, connected_client) -> None:
        """get_inventory sends a 'get_inventory' action."""
        client, cmd_mock, _ = connected_client
        cmd_mock.recv_json = AsyncMock(
            return_value={
                "id": "x",
                "success": True,
                "data": {"items": [{"name": "dirt", "count": 64}]},
                "error": None,
            }
        )
        resp = await client.get_inventory()
        assert resp["data"]["items"][0]["name"] == "dirt"

        payload = json.loads(cmd_mock.send_string.call_args[0][0])
        assert payload["action"] == "get_inventory"


class TestEventListener:
    """Tests for the PUB/SUB event listener."""

    async def test_event_listener_receives_events(self, connected_client) -> None:
        """Listener calls callback with parsed BridgeEvent."""
        client, _, sub_mock = connected_client

        event_data = {
            "event_type": "perception",
            "data": {"position": {"x": 0, "y": 64, "z": 0}},
            "timestamp": "2026-01-01T00:00:00Z",
        }

        # Make recv_json return the event once then block forever
        call_count = 0

        async def recv_once() -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return event_data
            # Block until cancelled
            await asyncio.sleep(999)
            return {}  # pragma: no cover

        sub_mock.recv_json = recv_once

        received: list[BridgeEvent] = []

        async def on_event(event: BridgeEvent) -> None:
            received.append(event)

        await client.start_event_listener(on_event)
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].event_type == "perception"
        assert received[0].data["position"]["x"] == 0

    async def test_event_listener_requires_connection(self) -> None:
        """Starting event listener on disconnected client raises ConnectionError."""
        client = BridgeClient()

        async def on_event(event: BridgeEvent) -> None:
            pass  # pragma: no cover

        with pytest.raises(ConnectionError):
            await client.start_event_listener(on_event)


class TestBridgeTypes:
    """Tests for bridge type definitions."""

    def test_bridge_status_values(self) -> None:
        """BridgeStatus enum has expected string values."""
        assert BridgeStatus.CONNECTED == "connected"
        assert BridgeStatus.DISCONNECTED == "disconnected"
        assert BridgeStatus.RECONNECTING == "reconnecting"

    def test_bridge_command_defaults(self) -> None:
        """BridgeCommand has sensible defaults."""
        cmd = BridgeCommand(action="ping")
        assert cmd.action == "ping"
        assert cmd.params == {}
        assert cmd.timeout_ms == 5000
        assert cmd.id is not None

    def test_bridge_command_serialization(self) -> None:
        """BridgeCommand serializes to valid JSON."""
        cmd = BridgeCommand(action="move", params={"x": 1, "y": 2, "z": 3})
        data = json.loads(cmd.model_dump_json())
        assert data["action"] == "move"
        assert data["params"]["x"] == 1

    def test_bridge_event_creation(self) -> None:
        """BridgeEvent can be created with event_type and data."""
        event = BridgeEvent(event_type="chat", data={"msg": "hi"})
        assert event.event_type == "chat"
        assert event.data["msg"] == "hi"
        assert event.timestamp is not None

    def test_bridge_event_validation(self) -> None:
        """BridgeEvent can be created from a raw dict."""
        raw = {
            "event_type": "death",
            "data": {},
            "timestamp": "2026-01-01T00:00:00Z",
        }
        event = BridgeEvent.model_validate(raw)
        assert event.event_type == "death"


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestBridgeClientEdgeCases:
    """Edge case tests for reconnection, errors, and edge conditions."""

    async def test_reconnect_cmd_socket_creates_new_socket(self, connected_client) -> None:
        """Reconnecting command socket creates a new socket object."""
        client, old_cmd_socket, _ = connected_client
        old_socket_id = id(old_cmd_socket)

        # Manually trigger reconnect
        await client._reconnect_cmd_socket()

        new_socket_id = id(client._cmd_socket)
        assert new_socket_id != old_socket_id
        assert client.status == BridgeStatus.CONNECTED
        old_cmd_socket.close.assert_called_once()

    async def test_disconnect_when_already_disconnected(self, mock_ctx: MagicMock) -> None:
        """Disconnecting an already disconnected client is idempotent."""
        with patch("piano.bridge.client.zmq.asyncio.Context", return_value=mock_ctx):
            client = BridgeClient()
            await client.disconnect()  # First disconnect (no-op)
            assert client.status == BridgeStatus.DISCONNECTED

            await client.disconnect()  # Second disconnect (should be safe)
            assert client.status == BridgeStatus.DISCONNECTED

    async def test_send_command_with_empty_params(self, connected_client) -> None:
        """Commands with empty params dict are handled correctly."""
        client, cmd_mock, _ = connected_client
        cmd_mock.recv_json = AsyncMock(
            return_value={
                "id": "test",
                "success": True,
                "data": {},
                "error": None,
            }
        )

        cmd = BridgeCommand(action="ping", params={})
        resp = await client.send_command(cmd)

        assert resp["success"] is True
        payload_str = cmd_mock.send_string.call_args[0][0]
        payload = json.loads(payload_str)
        assert payload["params"] == {}

    async def test_event_listener_error_continues_listening(self, connected_client) -> None:
        """Error in callback doesn't stop event listener."""
        client, _, sub_mock = connected_client

        event1 = {
            "event_type": "chat",
            "data": {"message": "first"},
            "timestamp": "2026-01-01T00:00:00Z",
        }
        event2 = {
            "event_type": "chat",
            "data": {"message": "second"},
            "timestamp": "2026-01-01T00:00:01Z",
        }

        call_count = 0

        async def recv_two_events() -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return event1
            elif call_count == 2:
                return event2
            # Block forever after two events
            await asyncio.sleep(999)
            return {}  # pragma: no cover

        sub_mock.recv_json = recv_two_events

        received: list[BridgeEvent] = []
        error_count = 0

        async def on_event_with_error(event: BridgeEvent) -> None:
            nonlocal error_count
            if event.data["message"] == "first":
                error_count += 1
                raise ValueError("Callback error")
            received.append(event)

        await client.start_event_listener(on_event_with_error)
        await asyncio.sleep(0.2)

        # First event should have raised an error, second should succeed
        assert error_count == 1
        assert len(received) == 1
        assert received[0].data["message"] == "second"

    async def test_event_listener_cancelled_stops_cleanly(self, connected_client) -> None:
        """Cancelling event listener task stops it cleanly."""
        client, _, sub_mock = connected_client

        async def recv_forever() -> dict[str, Any]:
            await asyncio.sleep(999)
            return {}  # pragma: no cover

        sub_mock.recv_json = recv_forever

        received: list[BridgeEvent] = []

        async def on_event(event: BridgeEvent) -> None:
            received.append(event)

        await client.start_event_listener(on_event)
        assert client._event_task is not None

        # Cancel the task
        client._event_task.cancel()
        await asyncio.sleep(0.1)

        assert client._event_task.done()
        assert len(received) == 0

    async def test_convenience_methods_propagate_timeout(self, connected_client) -> None:
        """Timeout in convenience methods propagates TimeoutError."""
        client, cmd_mock, _ = connected_client

        # Make send_command always timeout
        cmd_mock.send_string = AsyncMock(side_effect=zmq.Again("timeout"))

        # Override _reconnect_cmd_socket to reset the same failing mock
        async def fake_reconnect() -> None:
            client._cmd_socket = cmd_mock
            # Don't set status here - let send_command handle it

        client._reconnect_cmd_socket = fake_reconnect

        # Test move_to
        with pytest.raises(TimeoutError):
            await client.move_to(1, 2, 3)

        # Reset status for next test
        client._status = BridgeStatus.CONNECTED

        # Test mine_block
        with pytest.raises(TimeoutError):
            await client.mine_block(5, 60, 5)

        client._status = BridgeStatus.CONNECTED

        # Test craft_item
        with pytest.raises(TimeoutError):
            await client.craft_item("planks", 4)

        client._status = BridgeStatus.CONNECTED

        # Test chat
        with pytest.raises(TimeoutError):
            await client.chat("hello")

        client._status = BridgeStatus.CONNECTED

        # Test get_position
        with pytest.raises(TimeoutError):
            await client.get_position()

        client._status = BridgeStatus.CONNECTED

        # Test get_inventory
        with pytest.raises(TimeoutError):
            await client.get_inventory()

    async def test_reconnect_without_context_raises_error(self, connected_client) -> None:
        """Reconnecting when context is None raises ConnectionError."""
        client, _, _ = connected_client
        client._ctx = None

        with pytest.raises(ConnectionError, match="ZMQ context is not available"):
            await client._reconnect_cmd_socket()

    async def test_status_transitions_on_timeout(self, connected_client) -> None:
        """Status correctly transitions CONNECTED->RECONNECTING->DISCONNECTED on timeouts."""
        client, cmd_mock, _ = connected_client

        cmd_mock.send_string = AsyncMock(side_effect=zmq.Again("timeout"))

        async def fake_reconnect() -> None:
            client._status = BridgeStatus.CONNECTED

        client._reconnect_cmd_socket = fake_reconnect

        assert client.status == BridgeStatus.CONNECTED

        cmd = BridgeCommand(action="ping", params={})
        with pytest.raises(TimeoutError):
            await client.send_command(cmd)

        # After MAX_RETRIES, status should be DISCONNECTED
        assert client.status == BridgeStatus.DISCONNECTED

    async def test_disconnect_cancels_event_task(self, connected_client) -> None:
        """Disconnecting cancels the event listener task."""
        client, _, sub_mock = connected_client

        async def recv_forever() -> dict[str, Any]:
            await asyncio.sleep(999)
            return {}  # pragma: no cover

        sub_mock.recv_json = recv_forever

        async def on_event(event: BridgeEvent) -> None:
            pass  # pragma: no cover

        await client.start_event_listener(on_event)
        assert client._event_task is not None
        task = client._event_task

        await client.disconnect()

        assert task.done()
        assert client._event_task is None

    async def test_send_command_with_zero_timeout(self, mock_ctx: MagicMock) -> None:
        """Client with zero timeout immediately times out."""
        with patch("piano.bridge.client.zmq.asyncio.Context", return_value=mock_ctx):
            client = BridgeClient(timeout_ms=0)
            await client.connect()

            cmd_mock = client._cmd_socket
            cmd_mock.send_string = AsyncMock(side_effect=zmq.Again("timeout"))

            async def fake_reconnect() -> None:
                client._cmd_socket = cmd_mock
                client._status = BridgeStatus.CONNECTED

            client._reconnect_cmd_socket = fake_reconnect

            cmd = BridgeCommand(action="ping", params={})
            with pytest.raises(TimeoutError):
                await client.send_command(cmd)

            await client.disconnect()

    async def test_stop_event_listener_cancels_task(self, connected_client) -> None:
        """stop_event_listener() cancels the running event listener task."""
        client, _, sub_mock = connected_client

        async def recv_forever() -> dict[str, Any]:
            await asyncio.sleep(999)
            return {}  # pragma: no cover

        sub_mock.recv_json = recv_forever

        async def on_event(event: BridgeEvent) -> None:
            pass  # pragma: no cover

        await client.start_event_listener(on_event)
        assert client._event_task is not None
        task = client._event_task

        await client.stop_event_listener()

        assert task.done()
        assert client._event_task is None

    async def test_stop_event_listener_when_no_task(self, connected_client) -> None:
        """stop_event_listener() is safe to call when no task is running."""
        client, _, _ = connected_client
        assert client._event_task is None
        await client.stop_event_listener()  # Should not raise
        assert client._event_task is None

    async def test_req_socket_recovery_after_recv_timeout(self, connected_client) -> None:
        """After zmq.Again on recv, always reconnect the cmd socket."""
        client, cmd_mock, _ = connected_client

        reconnect_count = 0

        async def tracking_reconnect() -> None:
            nonlocal reconnect_count
            reconnect_count += 1
            client._cmd_socket = cmd_mock
            client._status = BridgeStatus.CONNECTED

        # Fail on recv (after successful send) to simulate partial REQ/REP
        cmd_mock.send_string = AsyncMock()
        cmd_mock.recv_json = AsyncMock(side_effect=zmq.Again("timeout"))
        client._reconnect_cmd_socket = tracking_reconnect

        cmd = BridgeCommand(action="ping", params={})
        with pytest.raises(TimeoutError):
            await client.send_command(cmd)

        # Should have reconnected on every attempt (including the last one)
        from piano.bridge.client import MAX_RETRIES
        assert reconnect_count == MAX_RETRIES

    async def test_event_listener_backoff_on_repeated_errors(self, connected_client) -> None:
        """Event listener sleeps after 5 consecutive errors."""
        client, _, sub_mock = connected_client

        call_count = 0

        async def error_then_block() -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count <= 6:
                raise RuntimeError("parse error")
            # Block forever after errors
            await asyncio.sleep(999)
            return {}  # pragma: no cover

        sub_mock.recv_json = error_then_block

        async def on_event(event: BridgeEvent) -> None:
            pass  # pragma: no cover

        await client.start_event_listener(on_event)
        # Wait for errors to accumulate and backoff to trigger
        await asyncio.sleep(1.5)

        # Verify error counter incremented
        assert client._evt_error_count >= 5

        # Clean up
        await client.stop_event_listener()

    async def test_repr_shows_urls(self, connected_client) -> None:
        """__repr__ includes command and event URLs."""
        client, _, _ = connected_client
        r = repr(client)
        assert "tcp://127.0.0.1:5555" in r
        assert "tcp://127.0.0.1:5556" in r

    async def test_connect_reuses_existing_context(self, mock_ctx: MagicMock) -> None:
        """If a context already exists, connect() reuses it instead of creating a new one."""
        with patch("piano.bridge.client.zmq.asyncio.Context", return_value=mock_ctx):
            client = BridgeClient()
            # Pre-set context (simulating re-connect after disconnect that didn't clear it)
            client._ctx = mock_ctx
            await client.connect()
            assert client._ctx is mock_ctx
            assert client.status == BridgeStatus.CONNECTED
            await client.disconnect()
