"""Tests for the PIANO simulation launcher."""

from __future__ import annotations

import asyncio

import pytest

from piano.config.settings import PianoSettings
from piano.llm.mock import MockLLMProvider
from piano.main import _run_multi, _run_single, cli, parse_args, run


class TestParseArgs:
    """Tests for argument parsing."""

    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.agents == 1
        assert args.ticks is None
        assert args.mock_llm is False
        assert args.config is None
        assert args.log_level == "INFO"

    def test_agents_flag(self) -> None:
        args = parse_args(["--agents", "5"])
        assert args.agents == 5

    def test_ticks_flag(self) -> None:
        args = parse_args(["--ticks", "100"])
        assert args.ticks == 100

    def test_mock_llm_flag(self) -> None:
        args = parse_args(["--mock-llm"])
        assert args.mock_llm is True

    def test_config_flag(self) -> None:
        args = parse_args(["--config", "/path/to/.env"])
        assert args.config == "/path/to/.env"

    def test_log_level_flag(self) -> None:
        args = parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_all_flags_combined(self) -> None:
        args = parse_args([
            "--agents", "3",
            "--ticks", "50",
            "--mock-llm",
            "--config", "test.env",
            "--log-level", "WARNING",
        ])
        assert args.agents == 3
        assert args.ticks == 50
        assert args.mock_llm is True
        assert args.config == "test.env"
        assert args.log_level == "WARNING"

    def test_rejects_invalid_log_level(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--log-level", "INVALID"])


class TestCli:
    """Tests for the CLI entry point."""

    def test_cli_is_callable(self) -> None:
        assert callable(cli)


class TestRun:
    """Tests for the run function."""

    async def test_single_agent_mock_llm(self) -> None:
        args = parse_args(["--mock-llm", "--ticks", "3"])
        await run(args)

    async def test_single_agent_with_log_level(self) -> None:
        args = parse_args([
            "--mock-llm", "--ticks", "2", "--log-level", "DEBUG",
        ])
        await run(args)

    async def test_run_single_directly(self) -> None:
        provider = MockLLMProvider()
        args = parse_args(["--mock-llm", "--ticks", "2"])
        await _run_single(args, provider, tick_interval=0.1)

    async def test_run_multi_directly(self) -> None:
        provider = MockLLMProvider()
        args = parse_args(["--mock-llm", "--agents", "2"])
        settings = PianoSettings()

        async def _stop_after_delay() -> None:
            await asyncio.sleep(0.5)
            # Import to get the function's internal event - not possible,
            # so we just cancel the task.

        task = asyncio.create_task(
            _run_multi(args, provider, settings, tick_interval=0.1),
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestSignalHandling:
    """Tests for graceful shutdown signal handling."""

    async def test_shutdown_via_ticks_single_agent(self) -> None:
        args = parse_args(["--mock-llm", "--ticks", "2"])
        await run(args)

    async def test_single_agent_finite_run_completes(self) -> None:
        provider = MockLLMProvider()
        args = parse_args(["--mock-llm", "--ticks", "1"])
        await _run_single(args, provider, tick_interval=0.1)
