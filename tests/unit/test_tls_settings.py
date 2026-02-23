"""Tests for TLS/SSL settings across PIANO configuration and connection code."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from piano.bridge.client import BridgeClient
from piano.config.settings import (
    BridgeSettings,
    PianoSettings,
    QdrantSettings,
    RedisSettings,
)
from piano.core.sas_redis import RedisSAS
from piano.memory.ltm import QdrantLTMStore

if TYPE_CHECKING:
    import pytest


# --- RedisSettings TLS fields ---


class TestRedisSettingsTLS:
    def test_default_ssl_disabled(self) -> None:
        settings = RedisSettings()
        assert settings.ssl_enabled is False
        assert settings.ssl_ca_certs is None
        assert settings.ssl_certfile is None
        assert settings.ssl_keyfile is None

    def test_ssl_enabled_with_certs(self) -> None:
        settings = RedisSettings(
            ssl_enabled=True,
            ssl_ca_certs="/path/to/ca.pem",
            ssl_certfile="/path/to/cert.pem",
            ssl_keyfile="/path/to/key.pem",
        )
        assert settings.ssl_enabled is True
        assert settings.ssl_ca_certs == "/path/to/ca.pem"
        assert settings.ssl_certfile == "/path/to/cert.pem"
        assert settings.ssl_keyfile == "/path/to/key.pem"

    def test_ssl_enabled_without_certs(self) -> None:
        settings = RedisSettings(ssl_enabled=True)
        assert settings.ssl_enabled is True
        assert settings.ssl_ca_certs is None


# --- BridgeSettings TLS fields ---


class TestBridgeSettingsTLS:
    def test_default_tls_disabled(self) -> None:
        settings = BridgeSettings()
        assert settings.tls_enabled is False
        assert settings.curve_public_key is None
        assert settings.curve_secret_key is None
        assert settings.curve_server_key is None

    def test_tls_enabled_with_keys(self) -> None:
        settings = BridgeSettings(
            tls_enabled=True,
            curve_public_key="pub-key-xxx",
            curve_secret_key="sec-key-xxx",
            curve_server_key="srv-key-xxx",
        )
        assert settings.tls_enabled is True
        assert settings.curve_public_key == "pub-key-xxx"
        assert settings.curve_secret_key == "sec-key-xxx"
        assert settings.curve_server_key == "srv-key-xxx"


# --- QdrantSettings TLS fields ---


class TestQdrantSettingsTLS:
    def test_default_https_disabled(self) -> None:
        settings = QdrantSettings()
        assert settings.use_https is False
        assert settings.api_key is None

    def test_https_enabled_with_api_key(self) -> None:
        settings = QdrantSettings(
            use_https=True,
            api_key="qdrant-api-key-xxx",
        )
        assert settings.use_https is True
        assert settings.api_key == "qdrant-api-key-xxx"


# --- PianoSettings env loading ---


class TestPianoSettingsTLSEnv:
    def test_tls_fields_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PIANO_REDIS__SSL_ENABLED", "true")
        monkeypatch.setenv("PIANO_REDIS__SSL_CA_CERTS", "/ca.pem")
        monkeypatch.setenv("PIANO_BRIDGE__TLS_ENABLED", "true")
        monkeypatch.setenv("PIANO_BRIDGE__CURVE_PUBLIC_KEY", "pub")
        monkeypatch.setenv("PIANO_BRIDGE__CURVE_SECRET_KEY", "sec")
        monkeypatch.setenv("PIANO_BRIDGE__CURVE_SERVER_KEY", "srv")
        monkeypatch.setenv("PIANO_QDRANT__USE_HTTPS", "true")
        monkeypatch.setenv("PIANO_QDRANT__API_KEY", "qkey")

        settings = PianoSettings()
        assert settings.redis.ssl_enabled is True
        assert settings.redis.ssl_ca_certs == "/ca.pem"
        assert settings.bridge.tls_enabled is True
        assert settings.bridge.curve_public_key == "pub"
        assert settings.bridge.curve_secret_key == "sec"
        assert settings.bridge.curve_server_key == "srv"
        assert settings.qdrant.use_https is True
        assert settings.qdrant.api_key == "qkey"


# --- BridgeClient TLS config ---


class TestBridgeClientTLS:
    def test_accepts_tls_config_none(self) -> None:
        client = BridgeClient(tls_config=None)
        assert client._tls_config is None

    def test_accepts_tls_config_dict(self) -> None:
        config = {
            "enabled": "true",
            "public_key": "pub",
            "secret_key": "sec",
            "server_key": "srv",
        }
        client = BridgeClient(tls_config=config)
        assert client._tls_config == config

    def test_apply_curve_config_noop_when_disabled(self) -> None:
        client = BridgeClient(tls_config=None)
        mock_socket = MagicMock(spec=[])
        client._apply_curve_config(mock_socket)
        # No curve attributes should be set on a strict mock
        assert not hasattr(mock_socket, "curve_publickey")

    def test_apply_curve_config_sets_keys(self) -> None:
        config = {
            "enabled": "true",
            "public_key": "pub-key",
            "secret_key": "sec-key",
            "server_key": "srv-key",
        }
        client = BridgeClient(tls_config=config)
        mock_socket = MagicMock()
        client._apply_curve_config(mock_socket)
        assert mock_socket.curve_publickey == b"pub-key"
        assert mock_socket.curve_secretkey == b"sec-key"
        assert mock_socket.curve_serverkey == b"srv-key"


# --- RedisSAS.create_with_settings ---


class TestRedisSASCreateWithSettings:
    def test_create_without_ssl(self) -> None:
        settings = RedisSettings(host="redis-host", port=6380, db=2)
        with patch("redis.asyncio.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = MagicMock()
            sas = RedisSAS.create_with_settings(settings, agent_id="agent-1")

        mock_redis_cls.assert_called_once_with(
            host="redis-host",
            port=6380,
            db=2,
            password=None,
        )
        assert sas.agent_id == "agent-1"

    def test_create_with_ssl(self) -> None:
        settings = RedisSettings(
            host="redis-host",
            port=6380,
            ssl_enabled=True,
            ssl_ca_certs="/ca.pem",
            ssl_certfile="/cert.pem",
            ssl_keyfile="/key.pem",
        )
        with patch("redis.asyncio.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = MagicMock()
            sas = RedisSAS.create_with_settings(settings, agent_id="agent-2")

        mock_redis_cls.assert_called_once_with(
            host="redis-host",
            port=6380,
            db=0,
            password=None,
            ssl=True,
            ssl_ca_certs="/ca.pem",
            ssl_certfile="/cert.pem",
            ssl_keyfile="/key.pem",
        )
        assert sas.agent_id == "agent-2"

    def test_create_with_ssl_no_certs(self) -> None:
        settings = RedisSettings(ssl_enabled=True)
        with patch("redis.asyncio.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = MagicMock()
            sas = RedisSAS.create_with_settings(settings, agent_id="agent-3")

        call_kwargs = mock_redis_cls.call_args[1]
        assert call_kwargs["ssl"] is True
        assert "ssl_ca_certs" not in call_kwargs
        assert sas.agent_id == "agent-3"


# --- QdrantLTMStore TLS params ---


class TestQdrantLTMStoreTLS:
    def test_default_no_https(self) -> None:
        store = QdrantLTMStore()
        assert store.use_https is False
        assert store.api_key is None

    def test_https_and_api_key(self) -> None:
        store = QdrantLTMStore(
            url="https://qdrant.example.com:6333",
            use_https=True,
            api_key="my-api-key",
        )
        assert store.use_https is True
        assert store.api_key == "my-api-key"

    async def test_initialize_passes_https_and_api_key(self) -> None:
        store = QdrantLTMStore(
            url="https://qdrant.example.com:6333",
            use_https=True,
            api_key="my-api-key",
        )
        with patch("piano.memory.ltm.QdrantClient", create=True) as mock_cls:
            # Patch the import inside initialize
            import sys

            mock_module = MagicMock()
            mock_module.QdrantClient = mock_cls
            sys.modules["qdrant_client"] = mock_module
            try:
                await store.initialize()
                mock_cls.assert_called_once_with(
                    url="https://qdrant.example.com:6333",
                    https=True,
                    api_key="my-api-key",
                )
            finally:
                del sys.modules["qdrant_client"]

    async def test_initialize_without_https(self) -> None:
        store = QdrantLTMStore(url="http://localhost:6333")
        with patch("piano.memory.ltm.QdrantClient", create=True) as mock_cls:
            import sys

            mock_module = MagicMock()
            mock_module.QdrantClient = mock_cls
            sys.modules["qdrant_client"] = mock_module
            try:
                await store.initialize()
                mock_cls.assert_called_once_with(url="http://localhost:6333")
            finally:
                del sys.modules["qdrant_client"]
