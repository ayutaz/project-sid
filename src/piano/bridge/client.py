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
        tls_config: dict[str, str] | None = None,
    ) -> None:
        """Initialise the bridge client.

        Args:
            command_url: ZMQ endpoint for command REQ/REP channel.
            event_url: ZMQ endpoint for event PUB/SUB channel.
            timeout_ms: Default timeout for command responses in milliseconds.
            tls_config: Optional CurveZMQ TLS configuration with keys
                ``enabled``, ``public_key``, ``secret_key``, ``server_key``.
        """
        self._command_url = command_url
        self._event_url = event_url
        self._timeout_ms = timeout_ms
        self._tls_config = tls_config
        self._status = BridgeStatus.DISCONNECTED
        self._ctx: zmq.asyncio.Context | None = None
        self._cmd_socket: zmq.asyncio.Socket | None = None
        self._sub_socket: zmq.asyncio.Socket | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._evt_error_count = 0
        self._max_retries = MAX_RETRIES

    # --- Properties ---------------------------------------------------------

    @property
    def status(self) -> BridgeStatus:
        """Return the current connection status."""
        return self._status

    # --- Connection lifecycle -----------------------------------------------

    def __repr__(self) -> str:
        return f"BridgeClient(cmd={self._command_url!r}, evt={self._event_url!r})"

    async def connect(self) -> None:
        """Connect to the bridge process.

        Creates ZMQ REQ and SUB sockets and connects them to the configured endpoints.
        """
        if self._status == BridgeStatus.CONNECTED:
            return

        if self._ctx is None:
            self._ctx = zmq.asyncio.Context()

        # REQ socket for commands
        self._cmd_socket = self._ctx.socket(zmq.REQ)
        self._cmd_socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._cmd_socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._cmd_socket.setsockopt(zmq.LINGER, 0)
        self._apply_curve_config(self._cmd_socket)
        self._cmd_socket.connect(self._command_url)

        # SUB socket for events
        self._sub_socket = self._ctx.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.LINGER, 0)
        self._apply_curve_config(self._sub_socket)
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
        Uses the command's ``timeout_ms`` field for per-command timeout.

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

        # Apply per-command timeout
        effective_timeout = cmd.timeout_ms or self._timeout_ms
        self._cmd_socket.setsockopt(zmq.RCVTIMEO, effective_timeout)
        self._cmd_socket.setsockopt(zmq.SNDTIMEO, effective_timeout)

        payload = cmd.model_dump_json()
        backoff = INITIAL_BACKOFF_S

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await self._cmd_socket.send_string(payload)
                raw = await self._cmd_socket.recv_json()
                return dict(raw)
            except zmq.Again:
                logger.warning(
                    "Bridge command timeout (attempt %d/%d, action=%s, timeout=%dms)",
                    attempt,
                    MAX_RETRIES,
                    cmd.action,
                    effective_timeout,
                )
                # Always reconnect after zmq.Again to reset REQ socket state
                self._status = BridgeStatus.RECONNECTING
                await self._reconnect_cmd_socket()
                # Reapply per-command timeout to the new socket
                if self._cmd_socket:
                    self._cmd_socket.setsockopt(zmq.RCVTIMEO, effective_timeout)
                    self._cmd_socket.setsockopt(zmq.SNDTIMEO, effective_timeout)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    self._status = BridgeStatus.DISCONNECTED
                    # Attempt auto-reconnect before giving up
                    if await self.reconnect():
                        return await self._send_once(cmd)
                    raise TimeoutError(
                        f"Bridge command '{cmd.action}' timed out after {MAX_RETRIES} retries"
                    ) from None

        # Unreachable but makes type checkers happy
        raise TimeoutError("Bridge command failed")  # pragma: no cover

    async def _send_once(self, cmd: BridgeCommand) -> dict[str, Any]:
        """Send a command once without retries (used after reconnect)."""
        if self._cmd_socket is None:
            raise ConnectionError("Bridge client is not connected")
        effective_timeout = cmd.timeout_ms or self._timeout_ms
        self._cmd_socket.setsockopt(zmq.RCVTIMEO, effective_timeout)
        self._cmd_socket.setsockopt(zmq.SNDTIMEO, effective_timeout)
        payload = cmd.model_dump_json()
        try:
            await self._cmd_socket.send_string(payload)
            raw = await self._cmd_socket.recv_json()
            return dict(raw)
        except zmq.Again:
            raise TimeoutError(
                f"Bridge command '{cmd.action}' timed out after reconnect"
            ) from None

    def _apply_curve_config(self, socket: zmq.asyncio.Socket) -> None:
        """Apply CurveZMQ configuration to a socket if TLS is enabled."""
        if self._tls_config and self._tls_config.get("enabled"):
            socket.curve_publickey = self._tls_config["public_key"].encode()
            socket.curve_secretkey = self._tls_config["secret_key"].encode()
            socket.curve_serverkey = self._tls_config["server_key"].encode()

    async def _reconnect_cmd_socket(self) -> None:
        """Tear down and recreate the command socket."""
        if self._cmd_socket:
            self._cmd_socket.close(linger=0)
            self._cmd_socket = None
        if self._ctx is None:
            self._status = BridgeStatus.DISCONNECTED
            raise ConnectionError("ZMQ context is not available")
        self._cmd_socket = self._ctx.socket(zmq.REQ)
        self._cmd_socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._cmd_socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._cmd_socket.setsockopt(zmq.LINGER, 0)
        self._apply_curve_config(self._cmd_socket)
        self._cmd_socket.connect(self._command_url)
        self._status = BridgeStatus.CONNECTED

    async def reconnect(self) -> bool:
        """Attempt full reconnection after DISCONNECTED state.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        if self._status != BridgeStatus.DISCONNECTED:
            return False
        try:
            self._cleanup_sockets()
            await self.connect()
            return True
        except Exception:
            logger.warning("Reconnect failed", exc_info=True)
            return False

    def _cleanup_sockets(self) -> None:
        """Close existing sockets without destroying the context."""
        if self._cmd_socket:
            self._cmd_socket.close(linger=0)
            self._cmd_socket = None
        if self._sub_socket:
            self._sub_socket.close(linger=0)
            self._sub_socket = None

    async def _reconnect_evt_socket(self) -> None:
        """Reconnect the SUB event socket."""
        if self._ctx is None:
            return
        try:
            if self._sub_socket:
                self._sub_socket.close(linger=0)
        except Exception:
            pass
        self._sub_socket = self._ctx.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.LINGER, 0)
        self._apply_curve_config(self._sub_socket)
        self._sub_socket.subscribe(b"")
        self._sub_socket.connect(self._event_url)
        logger.info("SUB socket reconnected to %s", self._event_url)

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

        _max_consecutive_errors = 5
        _backoff_sleep_s = 1.0

        async def _listen() -> None:
            assert self._sub_socket is not None
            while True:
                try:
                    raw = await self._sub_socket.recv_json()
                    event = BridgeEvent.model_validate(raw)
                    await callback(event)
                    self._evt_error_count = 0
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.exception("Error processing bridge event")
                    self._evt_error_count += 1
                    if self._evt_error_count >= _max_consecutive_errors:
                        logger.warning(
                            "Event listener hit %d consecutive errors, reconnecting SUB socket",
                            self._evt_error_count,
                        )
                        await self._reconnect_evt_socket()
                        self._evt_error_count = 0
                        await asyncio.sleep(_backoff_sleep_s)

        self._event_task = asyncio.create_task(_listen())

    async def stop_event_listener(self) -> None:
        """Cancel the event listener background task if running."""
        if self._event_task and not self._event_task.done():
            self._event_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._event_task
        self._event_task = None

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
