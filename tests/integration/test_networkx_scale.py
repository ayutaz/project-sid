"""NetworkX scalability integration tests.

Verifies that SocialGraph operations complete within acceptable time limits
for the expected agent counts.
"""

from __future__ import annotations

import random
import time

import pytest

from piano.social.graph import SocialGraph


def _create_test_graph(n: int) -> SocialGraph:
    """Create a test graph with n agents and random edges."""
    graph = SocialGraph()
    agents = [f"agent-{i:04d}" for i in range(n)]

    for agent_id in agents:
        graph.add_agent(agent_id)

    # Add ~10% density edges
    num_edges = int(n * (n - 1) * 0.1)
    for _ in range(num_edges):
        src = random.choice(agents)
        tgt = random.choice(agents)
        if src != tgt:
            graph.update_relationship(
                src,
                tgt,
                affinity_delta=random.uniform(-0.5, 0.5),
            )

    return graph


@pytest.mark.integration
class TestNetworkXScale:
    """Test SocialGraph operations at 100-agent scale."""

    def test_100_agent_graph_creation(self) -> None:
        """Creating a 100-agent graph should complete quickly."""
        start = time.perf_counter()
        graph = _create_test_graph(100)
        elapsed = time.perf_counter() - start

        assert graph.agent_count == 100
        assert elapsed < 1.0, f"Graph creation took {elapsed:.3f}s (limit: 1.0s)"

    def test_100_agent_get_friends(self) -> None:
        """get_friends on 100-agent graph should be fast."""
        graph = _create_test_graph(100)
        agents = graph.get_all_agents()

        start = time.perf_counter()
        for agent in agents[:20]:  # Test 20 agents
            graph.get_friends(agent)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"get_friends took {elapsed:.3f}s (limit: 1.0s)"

    def test_100_agent_get_relationship(self) -> None:
        """get_relationship on 100-agent graph should be fast."""
        graph = _create_test_graph(100)
        agents = graph.get_all_agents()

        start = time.perf_counter()
        for i in range(100):
            graph.get_relationship(
                agents[i % len(agents)],
                agents[(i + 1) % len(agents)],
            )
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f"get_relationship took {elapsed:.3f}s (limit: 1.0s)"
        )

    def test_100_agent_pagerank(self) -> None:
        """PageRank on 100-agent graph should complete within 1 second."""
        graph = _create_test_graph(100)
        agents = graph.get_all_agents()

        start = time.perf_counter()
        score = graph.get_influence_score(agents[0])
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"PageRank took {elapsed:.3f}s (limit: 1.0s)"
        assert 0.0 <= score <= 1.0

    def test_100_agent_communities(self) -> None:
        """Community detection on 100-agent graph should be fast."""
        graph = _create_test_graph(100)

        start = time.perf_counter()
        communities = graph.get_communities()
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f"Community detection took {elapsed:.3f}s (limit: 1.0s)"
        )
        assert len(communities) >= 1
