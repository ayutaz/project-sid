"""ZMQ-based bridge client for communicating with the Mineflayer bot process."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

import zmq
import zmq.asyncio

from piano.bridge.types import BridgeStatus
from piano.core.types import BridgeCommand, BridgeEvent

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_TIMEOUT_MS = 5000
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 0.5


class BridgeClient:
    """Async ZMQ client for sending commands to and receiving events from the Mineflayer bridge.

    Uses a REQ socket for request-reply commands and a SUB socket for event streaming.
    """

    def __init__(
        self,
        command_url: str = "tcp://127.0.0.1:5555",
        event_url: str = "tcp://127.0.0.1:5556",
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        """Initialise the bridge client.

        Args:
            command_url: ZMQ endpoint for command REQ/REP channel.
            event_url: ZMQ endpoint for event PUB/SUB channel.
            timeout_ms: Default timeout for command responses in milliseconds.
        """
        self._command_url = command_url
        self._event_url = event_url
        self._timeout_ms = timeout_ms
        self._status = BridgeStatus.DISCONNECTED
        self._ctx: zmq.asyncio.Context | None = None
        self._cmd_socket: zmq.asyncio.Socket | None = None
        self._sub_socket: zmq.asyncio.Socket | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    # --- Properties ---------------------------------------------------------

    @property
    def status(self) -> BridgeStatus:
        """Return the current connection status."""
        return self._status

    # --- Connection lifecycle -----------------------------------------------

    async def connect(self) -> None:
        """Connect to the bridge process.

        Creates ZMQ REQ and SUB sockets and connects them to the configured endpoints.
        """
        if self._status == BridgeStatus.CONNECTED:
            return

        self._ctx = zmq.asyncio.Context()

        # REQ socket for commands
        self._cmd_socket = self._ctx.socket(zmq.REQ)
        self._cmd_socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._cmd_socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._cmd_socket.setsockopt(zmq.LINGER, 0)
        self._cmd_socket.connect(self._command_url)

        # SUB socket for events
        self._sub_socket = self._ctx.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.LINGER, 0)
        self._sub_socket.subscribe(b"")
        self._sub_socket.connect(self._event_url)

        self._status = BridgeStatus.CONNECTED
        logger.info("Bridge client connected (cmd=%s, evt=%s)", self._command_url, self._event_url)

    async def disconnect(self) -> None:
        """Disconnect from the bridge process and clean up resources."""
        if self._event_task and not self._event_task.done():
            self._event_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._event_task
            self._event_task = None

        if self._cmd_socket:
            self._cmd_socket.close(linger=0)
            self._cmd_socket = None
        if self._sub_socket:
            self._sub_socket.close(linger=0)
            self._sub_socket = None
        if self._ctx:
            self._ctx.destroy(linger=0)
            self._ctx = None

        self._status = BridgeStatus.DISCONNECTED
        logger.info("Bridge client disconnected")

    # --- Command send / receive ---------------------------------------------

    async def send_command(self, cmd: BridgeCommand) -> dict[str, Any]:
        """Send a command and wait for the response.

        Retries up to MAX_RETRIES times with exponential backoff on timeout.

        Args:
            cmd: The bridge command to send.

        Returns:
            The response dictionary from the bridge.

        Raises:
            ConnectionError: If not connected or the bridge is unreachable after retries.
            TimeoutError: If no response is received within the timeout.
        """
        if self._status != BridgeStatus.CONNECTED or self._cmd_socket is None:
            raise ConnectionError("Bridge client is not connected")

        payload = cmd.model_dump_json()
        backoff = INITIAL_BACKOFF_S

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await self._cmd_socket.send_string(payload)
                raw = await self._cmd_socket.recv_json()
                return dict(raw)
            except zmq.Again:
                logger.warning(
                    "Bridge command timeout (attempt %d/%d, action=%s)",
                    attempt,
                    MAX_RETRIES,
                    cmd.action,
                )
                if attempt < MAX_RETRIES:
                    self._status = BridgeStatus.RECONNECTING
                    await self._reconnect_cmd_socket()
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    self._status = BridgeStatus.DISCONNECTED
                    raise TimeoutError(
                        f"Bridge command '{cmd.action}' timed out after {MAX_RETRIES} retries"
                    ) from None

        # Unreachable but makes type checkers happy
        raise TimeoutError("Bridge command failed")  # pragma: no cover

    async def _reconnect_cmd_socket(self) -> None:
        """Tear down and recreate the command socket."""
        if self._cmd_socket:
            self._cmd_socket.close(linger=0)
        if self._ctx is None:
            raise ConnectionError("ZMQ context is not available")
        self._cmd_socket = self._ctx.socket(zmq.REQ)
        self._cmd_socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._cmd_socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._cmd_socket.setsockopt(zmq.LINGER, 0)
        self._cmd_socket.connect(self._command_url)
        self._status = BridgeStatus.CONNECTED

    # --- Event listener -----------------------------------------------------

    async def start_event_listener(
        self,
        callback: Callable[[BridgeEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Start listening for events from the bridge in a background task.

        Args:
            callback: Async function called for each received BridgeEvent.
        """
        if self._sub_socket is None:
            raise ConnectionError("Bridge client is not connected")

        async def _listen() -> None:
            assert self._sub_socket is not None
            while True:
                try:
                    raw = await self._sub_socket.recv_json()
                    event = BridgeEvent.model_validate(raw)
                    await callback(event)
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.exception("Error processing bridge event")

        self._event_task = asyncio.create_task(_listen())

    # --- Health check -------------------------------------------------------

    async def ping(self) -> bool:
        """Send a ping command to verify the bridge is responsive.

        Returns:
            True if the bridge responded successfully, False otherwise.
        """
        try:
            cmd = BridgeCommand(action="ping", params={})
            resp = await self.send_command(cmd)
            return bool(resp.get("success", False))
        except (TimeoutError, ConnectionError):
            return False

    # --- Convenience methods ------------------------------------------------

    async def move_to(self, x: float, y: float, z: float) -> dict[str, Any]:
        """Move the bot to the specified coordinates.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            z: Target Z coordinate.

        Returns:
            Response from the bridge.
        """
        cmd = BridgeCommand(action="move", params={"x": x, "y": y, "z": z})
        return await self.send_command(cmd)

    async def mine_block(self, x: float, y: float, z: float) -> dict[str, Any]:
        """Mine a block at the specified coordinates.

        Args:
            x: Block X coordinate.
            y: Block Y coordinate.
            z: Block Z coordinate.

        Returns:
            Response from the bridge.
        """
        cmd = BridgeCommand(action="mine", params={"x": x, "y": y, "z": z})
        return await self.send_command(cmd)

    async def craft_item(self, item_name: str, count: int = 1) -> dict[str, Any]:
        """Craft an item.

        Args:
            item_name: The name of the item to craft.
            count: Number of items to craft.

        Returns:
            Response from the bridge.
        """
        cmd = BridgeCommand(action="craft", params={"item": item_name, "count": count})
        return await self.send_command(cmd)

    async def chat(self, message: str) -> dict[str, Any]:
        """Send a chat message in-game.

        Args:
            message: The message to send.

        Returns:
            Response from the bridge.
        """
        cmd = BridgeCommand(action="chat", params={"message": message})
        return await self.send_command(cmd)

    async def get_position(self) -> dict[str, Any]:
        """Get the bot's current position.

        Returns:
            Dictionary with x, y, z coordinates.
        """
        cmd = BridgeCommand(action="get_position", params={})
        return await self.send_command(cmd)

    async def get_inventory(self) -> dict[str, Any]:
        """Get the bot's current inventory.

        Returns:
            Dictionary with inventory data.
        """
        cmd = BridgeCommand(action="get_inventory", params={})
        return await self.send_command(cmd)
