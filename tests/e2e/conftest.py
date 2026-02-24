"""E2E test configuration and fixtures.

E2E tests are skipped by default. Run with --run-e2e flag:
    uv run pytest tests/e2e/ --run-e2e
"""

from __future__ import annotations

import pytest

from piano.core.scheduler import ModuleScheduler
from piano.core.types import ModuleTier
from piano.llm.mock import MockLLMProvider
from tests.helpers import DummyModule, InMemorySAS


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --run-e2e command line option."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip e2e tests unless --run-e2e is provided."""
    if config.getoption("--run-e2e"):
        return
    skip_e2e = pytest.mark.skip(reason="Need --run-e2e option to run")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


@pytest.fixture
def e2e_sas() -> InMemorySAS:
    """InMemorySAS for Docker-free E2E tests."""
    return InMemorySAS(agent_id="e2e-agent-001")


@pytest.fixture
def e2e_mock_llm() -> MockLLMProvider:
    """MockLLMProvider for fast E2E tests."""
    return MockLLMProvider()


@pytest.fixture
def e2e_scheduler() -> ModuleScheduler:
    """Fast scheduler for E2E tests."""
    return ModuleScheduler(tick_interval=0.02)


@pytest.fixture
def e2e_fast_module() -> DummyModule:
    """FAST tier dummy module for E2E tests."""
    return DummyModule(module_name="e2e_fast", tier=ModuleTier.FAST)


@pytest.fixture
def e2e_mid_module() -> DummyModule:
    """MID tier dummy module for E2E tests."""
    return DummyModule(module_name="e2e_mid", tier=ModuleTier.MID)
