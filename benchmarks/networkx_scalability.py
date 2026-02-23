"""NetworkX scalability benchmark for SocialGraph.

Tests graph operations at 10, 50, 100, and 500 agent scales.
Measures execution time and memory usage.

Usage:
    python benchmarks/networkx_scalability.py
    python benchmarks/networkx_scalability.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import tracemalloc
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from piano.social.graph import SocialGraph


def create_graph(num_agents: int, edge_density: float = 0.1) -> SocialGraph:
    """Create a SocialGraph with random edges."""
    graph = SocialGraph()
    agents = [f"agent-{i:04d}" for i in range(num_agents)]

    for agent_id in agents:
        graph.add_agent(agent_id)

    # Add random edges
    num_edges = int(num_agents * (num_agents - 1) * edge_density)
    for _ in range(num_edges):
        src = random.choice(agents)
        tgt = random.choice(agents)
        if src != tgt:
            graph.update_relationship(
                src,
                tgt,
                affinity_delta=random.uniform(-0.5, 0.5),
                trust_delta=random.uniform(-0.2, 0.2),
            )

    return graph


def benchmark_operation(
    name: str,
    func: object,
    iterations: int = 100,
) -> dict:
    """Benchmark a single operation."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()  # type: ignore[operator]
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times.sort()
    return {
        "operation": name,
        "iterations": iterations,
        "mean_ms": sum(times) / len(times) * 1000,
        "p50_ms": times[len(times) // 2] * 1000,
        "p95_ms": times[int(len(times) * 0.95)] * 1000,
        "p99_ms": times[int(len(times) * 0.99)] * 1000,
        "min_ms": times[0] * 1000,
        "max_ms": times[-1] * 1000,
    }


def run_benchmark(num_agents: int) -> dict:
    """Run full benchmark suite for a given agent count."""
    print(f"\n--- Benchmarking {num_agents} agents ---")

    tracemalloc.start()
    graph = create_graph(num_agents)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    agents = graph.get_all_agents()
    results: dict = {
        "num_agents": num_agents,
        "num_edges": graph.edge_count,
        "memory_current_kb": current / 1024,
        "memory_peak_kb": peak / 1024,
        "operations": [],
    }

    # Benchmark operations
    agent_a = agents[0]
    agent_b = agents[min(1, len(agents) - 1)]

    results["operations"].append(
        benchmark_operation(
            "add_agent",
            lambda: graph.add_agent(f"temp-{random.randint(0, 99999)}"),
        )
    )

    results["operations"].append(
        benchmark_operation(
            "get_friends",
            lambda: graph.get_friends(agent_a),
        )
    )

    results["operations"].append(
        benchmark_operation(
            "get_relationship",
            lambda: graph.get_relationship(agent_a, agent_b),
        )
    )

    results["operations"].append(
        benchmark_operation(
            "update_relationship",
            lambda: graph.update_relationship(
                random.choice(agents),
                random.choice(agents),
                affinity_delta=0.1,
                trust_delta=0.05,
            ),
        )
    )

    results["operations"].append(
        benchmark_operation(
            "get_influence_score (PageRank)",
            lambda: graph.get_influence_score(agent_a),
            iterations=10,
        )
    )

    results["operations"].append(
        benchmark_operation(
            "get_communities",
            lambda: graph.get_communities(),
            iterations=10,
        )
    )

    # Print results
    for op in results["operations"]:
        flag = "  SLOW" if op["p99_ms"] > 10 else ""
        print(
            f"  {op['operation']:40s}"
            f"  p50={op['p50_ms']:8.3f}ms"
            f"  p99={op['p99_ms']:8.3f}ms{flag}"
        )

    print(
        f"  Memory: current={results['memory_current_kb']:.1f}KB"
        f"  peak={results['memory_peak_kb']:.1f}KB"
    )

    return results


def main() -> None:
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(
        description="NetworkX scalability benchmark",
    )
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument(
        "--scales",
        type=str,
        default="10,50,100,500",
        help="Comma-separated agent counts",
    )
    args = parser.parse_args()

    scales = [int(x) for x in args.scales.split(",")]

    print("=== NetworkX Scalability Benchmark ===")
    all_results = []

    for num_agents in scales:
        result = run_benchmark(num_agents)
        all_results.append(result)

    # Summary
    print("\n=== Summary ===")
    print("igraph migration recommended if P99 > 10ms for core operations")

    needs_migration = False
    for result in all_results:
        for op in result["operations"]:
            if op["p99_ms"] > 10 and "PageRank" not in op["operation"]:
                needs_migration = True
                print(
                    f"  WARNING: {result['num_agents']} agents: "
                    f"{op['operation']} P99={op['p99_ms']:.3f}ms > 10ms"
                )

    if not needs_migration:
        print("  All core operations within threshold. NetworkX is adequate.")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
