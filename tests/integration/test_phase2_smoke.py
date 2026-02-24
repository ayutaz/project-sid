"""Phase 2 smoke tests -- import, instantiation, and basic integration checks.

Verifies that all Phase 2 modules can be imported and that key classes can
be instantiated with default or minimal arguments.  Modules that have not
yet been implemented are gracefully skipped via ``pytest.importorskip``.
"""

from __future__ import annotations

import pytest

# =====================================================================
# 1. Import tests -- all Phase 2 modules
# =====================================================================


class TestPhase2Imports:
    """Verify that Phase 2 modules are importable."""

    # -- scaling --

    @pytest.mark.integration
    def test_import_scaling_worker(self):
        pytest.importorskip("piano.scaling.worker", reason="scaling.worker not yet implemented")
        from piano.scaling.worker import AgentWorkerProcess  # noqa: F401

    @pytest.mark.integration
    def test_import_scaling_supervisor(self):
        pytest.importorskip(
            "piano.scaling.supervisor", reason="scaling.supervisor not yet implemented"
        )
        from piano.scaling.supervisor import AgentSupervisor  # noqa: F401

    @pytest.mark.integration
    def test_import_scaling_sharding(self):
        from piano.scaling.sharding import ShardManager  # noqa: F401

    @pytest.mark.integration
    def test_import_scaling_resource_limiter(self):
        pytest.importorskip(
            "piano.scaling.resource_limiter",
            reason="scaling.resource_limiter not yet implemented",
        )
        from piano.scaling.resource_limiter import ResourceLimiter  # noqa: F401

    # -- llm --

    @pytest.mark.integration
    def test_import_llm_prompt_cache(self):
        from piano.llm.prompt_cache import PromptCacheManager  # noqa: F401

    # -- bridge --

    @pytest.mark.integration
    def test_import_bridge_velocity(self):
        from piano.bridge.velocity import VelocityProxyManager  # noqa: F401

    # -- observability --

    @pytest.mark.integration
    def test_import_observability_metrics(self):
        from piano.observability.metrics import MetricsRegistry, PianoMetrics  # noqa: F401

    @pytest.mark.integration
    def test_import_observability_tracing(self):
        pytest.importorskip(
            "piano.observability.tracing", reason="observability.tracing not yet implemented"
        )
        from piano.observability.tracing import Span, Tracer  # noqa: F401

    @pytest.mark.integration
    def test_import_observability_logging_config(self):
        pytest.importorskip(
            "piano.observability.logging_config",
            reason="observability.logging_config not yet implemented",
        )
        from piano.observability.logging_config import LoggingConfig  # noqa: F401

    # -- social --

    @pytest.mark.integration
    def test_import_social_collective(self):
        from piano.social.collective import CollectiveIntelligence  # noqa: F401

    @pytest.mark.integration
    def test_import_social_influencer(self):
        from piano.social.influencer import InfluencerModel  # noqa: F401

    # -- eval --

    @pytest.mark.integration
    def test_import_eval_role_inference(self):
        pytest.importorskip(
            "piano.eval.role_inference", reason="eval.role_inference not yet implemented"
        )
        from piano.eval.role_inference import RoleInferencePipeline  # noqa: F401

    @pytest.mark.integration
    def test_import_eval_governance(self):
        pytest.importorskip("piano.eval.governance", reason="eval.governance not yet implemented")
        from piano.eval.governance import TaxComplianceMetrics, VotingMetrics  # noqa: F401

    @pytest.mark.integration
    def test_import_eval_memes(self):
        pytest.importorskip("piano.eval.memes", reason="eval.memes not yet implemented")
        from piano.eval.memes import MemeTracker  # noqa: F401

    @pytest.mark.integration
    def test_import_eval_performance(self):
        from piano.eval.performance import PerformanceBenchmark  # noqa: F401

    # -- core --

    @pytest.mark.integration
    def test_import_core_distributed_checkpoint(self):
        from piano.core.distributed_checkpoint import DistributedCheckpointManager  # noqa: F401


# =====================================================================
# 2. Instantiation tests -- key classes with defaults
# =====================================================================


class TestPhase2Instantiation:
    """Verify that Phase 2 classes can be instantiated with default/minimal args."""

    @pytest.mark.integration
    def test_shard_manager_default(self):
        from piano.scaling.sharding import ShardManager

        mgr = ShardManager()
        assert mgr.num_shards == 4
        assert mgr.agent_count == 0

    @pytest.mark.integration
    def test_prompt_cache_manager_default(self):
        from piano.llm.prompt_cache import PromptCacheManager

        cache = PromptCacheManager()
        assert cache.l1_size == 0
        stats = cache.get_stats()
        assert stats.total_requests == 0

    @pytest.mark.integration
    def test_velocity_proxy_manager_default(self):
        from piano.bridge.velocity import VelocityConfig, VelocityProxyManager

        config = VelocityConfig()
        mgr = VelocityProxyManager(config)
        assert mgr.server_count == 0
        assert mgr.total_agents == 0

    @pytest.mark.integration
    def test_metrics_registry_default(self):
        from piano.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        assert registry.export() == ""

    @pytest.mark.integration
    def test_piano_metrics_default(self):
        from piano.observability.metrics import PianoMetrics

        metrics = PianoMetrics()
        assert metrics.registry is not None
        assert metrics.agent_tick_total is not None

    @pytest.mark.integration
    def test_collective_intelligence_default(self):
        from piano.social.collective import AggregationMethod, CollectiveIntelligence

        ci = CollectiveIntelligence()
        assert ci.default_method == AggregationMethod.SIMPLE_MEAN

    @pytest.mark.integration
    def test_influencer_model_default(self):
        from piano.social.influencer import InfluencerModel

        model = InfluencerModel()
        assert model.config is not None
        assert model.config.decay_factor == 0.5

    @pytest.mark.integration
    def test_performance_benchmark_default(self):
        from piano.eval.performance import PerformanceBenchmark

        bench = PerformanceBenchmark()
        # Should not be running yet
        with pytest.raises(RuntimeError):
            bench.stop()

    @pytest.mark.integration
    def test_distributed_checkpoint_manager_default(self, tmp_path):
        from piano.core.distributed_checkpoint import DistributedCheckpointManager

        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        assert mgr.list_checkpoints(0) == []


# =====================================================================
# 3. Scaling integration -- Worker + Supervisor + Shard basics
# =====================================================================


class TestScalingIntegration:
    """Basic integration tests for scaling components that exist."""

    @pytest.mark.integration
    def test_shard_assignment_and_retrieval(self):
        """ShardManager assigns agents to shards deterministically."""
        from piano.scaling.sharding import ShardManager

        mgr = ShardManager()
        shard_a = mgr.assign_shard("agent-001")
        shard_b = mgr.assign_shard("agent-002")

        # Assignment is deterministic (same agent -> same shard)
        assert mgr.assign_shard("agent-001") == shard_a
        assert mgr.assign_shard("agent-002") == shard_b
        assert mgr.agent_count == 2

        # Can look up agent's shard
        assert mgr.get_shard_for_agent("agent-001") == shard_a

    @pytest.mark.integration
    def test_shard_rebalance(self):
        """ShardManager rebalances when shard count changes."""
        from piano.scaling.sharding import ShardManager

        mgr = ShardManager()
        for i in range(10):
            mgr.assign_shard(f"agent-{i:03d}")
        assert mgr.agent_count == 10

        moves = mgr.rebalance(new_num_shards=8)
        assert mgr.num_shards == 8
        assert mgr.agent_count == 10
        # Some agents may have moved
        assert isinstance(moves, dict)

    @pytest.mark.integration
    def test_shard_stats(self):
        """ShardManager produces valid statistics."""
        from piano.scaling.sharding import ShardManager

        mgr = ShardManager()
        for i in range(8):
            mgr.assign_shard(f"agent-{i:03d}")

        stats = mgr.get_shard_stats()
        assert len(stats) == mgr.num_shards
        total_from_stats = sum(s.agent_count for s in stats.values())
        assert total_from_stats == 8

    @pytest.mark.integration
    def test_velocity_server_registration(self):
        """VelocityProxyManager registers servers and assigns agents."""
        from piano.bridge.velocity import VelocityConfig, VelocityProxyManager

        config = VelocityConfig()
        mgr = VelocityProxyManager(config)

        mgr.register_server("mc-1", "localhost", 25565)
        mgr.register_server("mc-2", "localhost", 25566)
        assert mgr.server_count == 2

        server = mgr.get_server_for_agent("agent-001")
        assert server in ("mc-1", "mc-2")
        assert mgr.total_agents == 1

    @pytest.mark.integration
    def test_shard_manager_with_velocity_proxy(self):
        """ShardManager and VelocityProxyManager can coordinate agent placement."""
        from piano.bridge.velocity import (
            ServerConfig,
            VelocityConfig,
            VelocityProxyManager,
        )
        from piano.scaling.sharding import ShardConfig, ShardManager

        # Set up sharding
        shard_config = ShardConfig(num_shards=2, strategy="consistent_hash")
        shard_mgr = ShardManager(shard_config)

        # Set up velocity proxy
        servers = [
            ServerConfig(name="mc-0", host="localhost", port=25565),
            ServerConfig(name="mc-1", host="localhost", port=25566),
        ]
        velocity_config = VelocityConfig(servers=servers)
        proxy_mgr = VelocityProxyManager(velocity_config)

        # Map shards to servers
        proxy_mgr.assign_shard(0, "mc-0")
        proxy_mgr.assign_shard(1, "mc-1")

        # Assign agents through both systems
        agents = [f"agent-{i:03d}" for i in range(6)]
        for agent_id in agents:
            shard_id = shard_mgr.assign_shard(agent_id)
            assert 0 <= shard_id < 2

            server = proxy_mgr.get_server_for_agent(agent_id)
            assert server in ("mc-0", "mc-1")

        assert shard_mgr.agent_count == 6
        assert proxy_mgr.total_agents == 6

    @pytest.mark.integration
    async def test_distributed_checkpoint_save_restore(self, tmp_path):
        """DistributedCheckpointManager can save and restore shard states."""
        from piano.core.distributed_checkpoint import DistributedCheckpointManager

        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path / "ckpt")

        agent_states = {
            "agent-001": {"health": 20, "pos": {"x": 10, "y": 64, "z": -30}},
            "agent-002": {"health": 18, "pos": {"x": 15, "y": 64, "z": -25}},
        }
        info = await mgr.save_shard(shard_id=0, agent_states=agent_states)
        assert info.shard_id == 0
        assert info.agent_count == 2
        assert info.size_bytes > 0

        restored = await mgr.restore_shard(shard_id=0, checkpoint_id=info.checkpoint_id)
        assert restored["agent-001"]["health"] == 20
        assert restored["agent-002"]["pos"]["x"] == 15

    @pytest.mark.integration
    def test_metrics_with_shard_operations(self):
        """PianoMetrics can record metrics alongside shard operations."""
        from piano.observability.metrics import PianoMetrics
        from piano.scaling.sharding import ShardManager

        metrics = PianoMetrics()
        mgr = ShardManager()

        for i in range(5):
            agent_id = f"agent-{i:03d}"
            mgr.assign_shard(agent_id)
            metrics.agent_tick_total.inc(labels={"agent_id": agent_id})

        assert mgr.agent_count == 5
        for i in range(5):
            assert metrics.agent_tick_total.get(labels={"agent_id": f"agent-{i:03d}"}) == 1.0

    @pytest.mark.integration
    def test_performance_benchmark_lifecycle(self):
        """PerformanceBenchmark start/record/stop lifecycle works."""
        from piano.eval.performance import PerformanceBenchmark

        bench = PerformanceBenchmark()
        bench.start()

        bench.record_tick("agent-001", duration_ms=10.0)
        bench.record_tick("agent-001", duration_ms=12.0)
        bench.record_tick("agent-002", duration_ms=8.0)
        bench.record_llm_call("openai", latency_ms=200.0, cost_usd=0.002, tokens=300)
        bench.record_bridge_command("move", latency_ms=30.0)

        result = bench.stop()
        assert result.agent_count == 2
        assert result.total_llm_calls == 1
        assert result.total_bridge_commands == 1
        assert result.tps > 0
        assert result.total_cost_usd == pytest.approx(0.002)
