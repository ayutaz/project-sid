"""Tests for the BridgeClient ZMQ communication layer."""

from __future__ import annotations

import asyncio
import json

import pytest
import zmq
import zmq.asyncio

from piano.bridge.client import BridgeClient, DEFAULT_TIMEOUT_MS, MAX_RETRIES
from piano.bridge.types import BridgeStatus
from piano.core.types import BridgeCommand, BridgeEvent


# ---------------------------------------------------------------------------
# Helpers: in-process ZMQ mock server
# ---------------------------------------------------------------------------


class MockBridgeServer:
    """Minimal ZMQ REP + PUB server running in-process for tests."""

    def __init__(self, cmd_url: str, evt_url: str) -> None:
        self._cmd_url = cmd_url
        self._evt_url = evt_url
        self._ctx: zmq.asyncio.Context | None = None
        self._rep: zmq.asyncio.Socket | None = None
        self._pub: zmq.asyncio.Socket | None = None
        self._task: asyncio.Task[None] | None = None
        self.received_commands: list[dict] = []
        self._response_fn: callable | None = None

    async def start(self) -> None:
        self._ctx = zmq.asyncio.Context()
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(self._cmd_url)
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(self._evt_url)
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._rep:
            self._rep.close()
        if self._pub:
            self._pub.close()
        if self._ctx:
            self._ctx.term()

    def set_response(self, fn: callable) -> None:
        """Set a callable(cmd_dict) -> response_dict for the mock server."""
        self._response_fn = fn

    async def publish_event(self, event: dict) -> None:
        """Publish a single event on the PUB socket."""
        assert self._pub is not None
        await self._pub.send_json(event)

    async def _loop(self) -> None:
        assert self._rep is not None
        while True:
            try:
                raw = await self._rep.recv_string()
                cmd = json.loads(raw)
                self.received_commands.append(cmd)

                if self._response_fn:
                    resp = self._response_fn(cmd)
                else:
                    resp = {
                        "id": cmd.get("id", ""),
                        "success": True,
                        "data": {},
                        "error": None,
                    }
                await self._rep.send_json(resp)
            except asyncio.CancelledError:
                break
            except Exception:
                break


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Use unique inproc addresses per-test to avoid address reuse issues.
# We use tcp on random ports instead since inproc needs shared context.
_PORT_BASE = 15600


def _ports(request: pytest.FixtureRequest) -> tuple[int, int]:
    """Return unique port pair for a test."""
    # Use the test node id hash for a quasi-unique offset
    offset = abs(hash(request.node.nodeid)) % 500
    cmd_port = _PORT_BASE + offset * 2
    evt_port = _PORT_BASE + offset * 2 + 1
    return cmd_port, evt_port


@pytest.fixture
async def bridge_env(request: pytest.FixtureRequest):
    """Yield a (server, client) pair wired together via TCP."""
    cmd_port, evt_port = _ports(request)
    cmd_url = f"tcp://127.0.0.1:{cmd_port}"
    evt_url = f"tcp://127.0.0.1:{evt_port}"

    server = MockBridgeServer(cmd_url, evt_url)
    await server.start()
    # Small sleep to let ZMQ sockets bind
    await asyncio.sleep(0.05)

    client = BridgeClient(
        command_url=cmd_url,
        event_url=evt_url,
        timeout_ms=2000,
    )
    await client.connect()

    yield server, client

    await client.disconnect()
    await server.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBridgeClientConnection:
    """Tests for connect / disconnect lifecycle."""

    async def test_initial_status_is_disconnected(self) -> None:
        client = BridgeClient()
        assert client.status == BridgeStatus.DISCONNECTED

    async def test_connect_changes_status(self, bridge_env) -> None:
        _, client = bridge_env
        assert client.status == BridgeStatus.CONNECTED

    async def test_disconnect_changes_status(self, bridge_env) -> None:
        _, client = bridge_env
        await client.disconnect()
        assert client.status == BridgeStatus.DISCONNECTED

    async def test_double_connect_is_idempotent(self, bridge_env) -> None:
        _, client = bridge_env
        await client.connect()  # second call
        assert client.status == BridgeStatus.CONNECTED


class TestBridgeClientSendCommand:
    """Tests for send_command and message serialization."""

    async def test_send_command_roundtrip(self, bridge_env) -> None:
        server, client = bridge_env
        server.set_response(lambda cmd: {
            "id": cmd["id"],
            "success": True,
            "data": {"moved": True},
            "error": None,
        })

        cmd = BridgeCommand(action="move", params={"x": 1, "y": 2, "z": 3})
        resp = await client.send_command(cmd)

        assert resp["success"] is True
        assert resp["data"]["moved"] is True

    async def test_command_is_json_serialized(self, bridge_env) -> None:
        server, client = bridge_env
        cmd = BridgeCommand(action="chat", params={"message": "hello"})
        await client.send_command(cmd)

        assert len(server.received_commands) == 1
        received = server.received_commands[0]
        assert received["action"] == "chat"
        assert received["params"]["message"] == "hello"

    async def test_command_preserves_id(self, bridge_env) -> None:
        server, client = bridge_env
        server.set_response(lambda cmd: {
            "id": cmd["id"],
            "success": True,
            "data": {},
            "error": None,
        })

        cmd = BridgeCommand(action="ping", params={})
        resp = await client.send_command(cmd)
        assert resp["id"] == str(cmd.id)

    async def test_send_command_when_disconnected_raises(self) -> None:
        client = BridgeClient()
        cmd = BridgeCommand(action="ping", params={})
        with pytest.raises(ConnectionError):
            await client.send_command(cmd)


class TestBridgeClientTimeout:
    """Tests for timeout and retry behaviour."""

    async def test_timeout_raises_after_retries(self) -> None:
        """Client should raise TimeoutError when server never responds."""
        # Connect to an address where nothing is listening
        client = BridgeClient(
            command_url="tcp://127.0.0.1:19999",
            event_url="tcp://127.0.0.1:19998",
            timeout_ms=200,
        )
        ctx = zmq.asyncio.Context()
        # We need a dummy REP that never responds -> just don't create one
        # The REQ socket will timeout waiting for a reply.
        # But first we need to connect
        await client.connect()

        cmd = BridgeCommand(action="ping", params={})
        with pytest.raises(TimeoutError):
            await client.send_command(cmd)

        await client.disconnect()


class TestPing:
    """Tests for the ping health check."""

    async def test_ping_success(self, bridge_env) -> None:
        server, client = bridge_env
        server.set_response(lambda cmd: {
            "id": cmd["id"],
            "success": True,
            "data": {"pong": True},
            "error": None,
        })
        result = await client.ping()
        assert result is True

    async def test_ping_returns_false_on_failure(self, bridge_env) -> None:
        server, client = bridge_env
        server.set_response(lambda cmd: {
            "id": cmd["id"],
            "success": False,
            "data": {},
            "error": "unhealthy",
        })
        result = await client.ping()
        assert result is False


class TestConvenienceMethods:
    """Tests for move_to, mine_block, craft_item, chat, get_position, get_inventory."""

    async def test_move_to(self, bridge_env) -> None:
        server, client = bridge_env
        server.set_response(lambda cmd: {
            "id": cmd["id"],
            "success": True,
            "data": {"x": 10, "y": 64, "z": -5},
            "error": None,
        })
        resp = await client.move_to(10, 64, -5)
        assert resp["data"]["x"] == 10
        assert server.received_commands[-1]["action"] == "move"
        assert server.received_commands[-1]["params"]["x"] == 10

    async def test_mine_block(self, bridge_env) -> None:
        server, client = bridge_env
        server.set_response(lambda cmd: {
            "id": cmd["id"],
            "success": True,
            "data": {"mined": "stone"},
            "error": None,
        })
        resp = await client.mine_block(5, 60, 5)
        assert server.received_commands[-1]["action"] == "mine"

    async def test_craft_item(self, bridge_env) -> None:
        server, client = bridge_env
        server.set_response(lambda cmd: {
            "id": cmd["id"],
            "success": True,
            "data": {"crafted": "planks"},
            "error": None,
        })
        resp = await client.craft_item("planks", 4)
        assert server.received_commands[-1]["action"] == "craft"
        assert server.received_commands[-1]["params"]["item"] == "planks"
        assert server.received_commands[-1]["params"]["count"] == 4

    async def test_chat(self, bridge_env) -> None:
        server, client = bridge_env
        server.set_response(lambda cmd: {
            "id": cmd["id"],
            "success": True,
            "data": {"sent": "hello"},
            "error": None,
        })
        resp = await client.chat("hello")
        assert server.received_commands[-1]["action"] == "chat"
        assert server.received_commands[-1]["params"]["message"] == "hello"

    async def test_get_position(self, bridge_env) -> None:
        server, client = bridge_env
        server.set_response(lambda cmd: {
            "id": cmd["id"],
            "success": True,
            "data": {"x": 1.5, "y": 65.0, "z": -2.3},
            "error": None,
        })
        resp = await client.get_position()
        assert resp["data"]["y"] == 65.0
        assert server.received_commands[-1]["action"] == "get_position"

    async def test_get_inventory(self, bridge_env) -> None:
        server, client = bridge_env
        server.set_response(lambda cmd: {
            "id": cmd["id"],
            "success": True,
            "data": {"items": [{"name": "dirt", "count": 64}]},
            "error": None,
        })
        resp = await client.get_inventory()
        assert resp["data"]["items"][0]["name"] == "dirt"
        assert server.received_commands[-1]["action"] == "get_inventory"


class TestEventListener:
    """Tests for the PUB/SUB event listener."""

    async def test_event_listener_receives_events(self, bridge_env) -> None:
        server, client = bridge_env

        received: list[BridgeEvent] = []

        async def on_event(event: BridgeEvent) -> None:
            received.append(event)

        await client.start_event_listener(on_event)

        # Give the SUB socket a moment to subscribe
        await asyncio.sleep(0.1)

        event_data = {
            "event_type": "perception",
            "data": {"position": {"x": 0, "y": 64, "z": 0}},
            "timestamp": "2026-01-01T00:00:00Z",
        }
        await server.publish_event(event_data)

        # Wait for the event to arrive
        await asyncio.sleep(0.2)

        assert len(received) >= 1
        assert received[0].event_type == "perception"
        assert received[0].data["position"]["x"] == 0

    async def test_event_listener_requires_connection(self) -> None:
        client = BridgeClient()

        async def on_event(event: BridgeEvent) -> None:
            pass  # pragma: no cover

        with pytest.raises(ConnectionError):
            await client.start_event_listener(on_event)


class TestBridgeTypes:
    """Tests for bridge type definitions."""

    def test_bridge_status_values(self) -> None:
        assert BridgeStatus.CONNECTED == "connected"
        assert BridgeStatus.DISCONNECTED == "disconnected"
        assert BridgeStatus.RECONNECTING == "reconnecting"

    def test_bridge_command_defaults(self) -> None:
        cmd = BridgeCommand(action="ping")
        assert cmd.action == "ping"
        assert cmd.params == {}
        assert cmd.timeout_ms == 5000
        assert cmd.id is not None

    def test_bridge_event_creation(self) -> None:
        event = BridgeEvent(event_type="chat", data={"msg": "hi"})
        assert event.event_type == "chat"
        assert event.data["msg"] == "hi"
        assert event.timestamp is not None
