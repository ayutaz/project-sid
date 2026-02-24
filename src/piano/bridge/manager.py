"""Bridge manager for multi-bot connections."""

from __future__ import annotations

import asyncio
import logging

from piano.bridge.client import BridgeClient

logger = logging.getLogger(__name__)

_DEFAULT_BASE_CMD_PORT = 5555
_DEFAULT_BASE_EVT_PORT = 5556
_CONNECT_RETRY_DELAY = 2.0


class BridgeManager:
    """Manages multiple BridgeClient instances for multi-agent simulations.

    Each agent gets its own bridge with a unique port pair:
    agent_index 0: cmd=base_cmd_port, evt=base_evt_port
    agent_index 1: cmd=base_cmd_port+2, evt=base_evt_port+2
    ...
    """

    def __init__(
        self,
        host: str = "localhost",
        base_command_port: int = _DEFAULT_BASE_CMD_PORT,
        base_event_port: int = _DEFAULT_BASE_EVT_PORT,
        connect_timeout_s: float = 30.0,
        connect_retry_count: int = 5,
        tls_config: dict[str, str] | None = None,
    ) -> None:
        self._host = host
        self._base_cmd_port = base_command_port
        self._base_evt_port = base_event_port
        self._connect_timeout_s = connect_timeout_s
        self._connect_retry_count = connect_retry_count
        self._tls_config = tls_config
        self._bridges: dict[str, BridgeClient] = {}
        self._agent_indices: dict[str, int] = {}

    def create_bridge(self, agent_id: str, agent_index: int) -> BridgeClient:
        """Create a bridge for a specific agent."""
        if agent_id in self._bridges:
            raise ValueError(f"Bridge already exists for agent '{agent_id}'")

        cmd_port = self._base_cmd_port + agent_index * 2
        evt_port = self._base_evt_port + agent_index * 2

        bridge = BridgeClient(
            command_url=f"tcp://{self._host}:{cmd_port}",
            event_url=f"tcp://{self._host}:{evt_port}",
            tls_config=self._tls_config,
        )
        self._bridges[agent_id] = bridge
        self._agent_indices[agent_id] = agent_index
        return bridge

    def get_bridge(self, agent_id: str) -> BridgeClient:
        """Get the bridge for a specific agent."""
        if agent_id not in self._bridges:
            raise KeyError(f"No bridge for agent '{agent_id}'")
        return self._bridges[agent_id]

    @property
    def bridges(self) -> dict[str, BridgeClient]:
        """All bridges indexed by agent_id."""
        return dict(self._bridges)

    async def connect_all(self) -> dict[str, bool]:
        """Connect all bridges with retry logic in parallel. Returns status per agent."""

        async def _connect_one(agent_id: str, bridge: BridgeClient) -> tuple[str, bool]:
            for attempt in range(1, self._connect_retry_count + 1):
                try:
                    await asyncio.wait_for(
                        bridge.connect(),
                        timeout=self._connect_timeout_s,
                    )
                    logger.info(
                        "Bridge connected agent_id=%s attempt=%d",
                        agent_id,
                        attempt,
                    )
                    return agent_id, True
                except Exception:
                    logger.warning(
                        "Bridge connect failed agent_id=%s attempt=%d/%d",
                        agent_id,
                        attempt,
                        self._connect_retry_count,
                    )
                    if attempt < self._connect_retry_count:
                        await asyncio.sleep(_CONNECT_RETRY_DELAY)
            return agent_id, False

        pairs = await asyncio.gather(*[_connect_one(aid, br) for aid, br in self._bridges.items()])
        return dict(pairs)

    async def disconnect_all(self) -> None:
        """Disconnect all bridges in parallel."""

        async def _disconnect_one(agent_id: str, bridge: BridgeClient) -> None:
            try:
                await bridge.disconnect()
            except Exception:
                logger.warning(
                    "Error disconnecting bridge agent_id=%s",
                    agent_id,
                    exc_info=True,
                )

        await asyncio.gather(*[_disconnect_one(aid, br) for aid, br in self._bridges.items()])

    async def health_check(self) -> dict[str, bool]:
        """Ping all bridges and return status."""
        results: dict[str, bool] = {}
        for agent_id, bridge in self._bridges.items():
            try:
                results[agent_id] = await bridge.ping()
            except Exception:
                results[agent_id] = False
        return results
