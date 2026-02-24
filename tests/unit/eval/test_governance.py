"""Tests for governance evaluation metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from piano.eval.governance import (
    ConstitutionMetrics,
    GovernanceReport,
    TaxComplianceMetrics,
    VotingMetrics,
    generate_governance_report,
)

# =========================================================================
# TaxComplianceMetrics
# =========================================================================


class TestTaxComplianceMetrics:
    """Tests for TaxComplianceMetrics."""

    def test_record_and_compliance_rate_all_paid(self) -> None:
        """Compliance is 1.0 when every agent pays."""
        tax = TaxComplianceMetrics()
        for i in range(5):
            tax.record_tax_event(f"agent_{i}", amount=100.0, paid=True)
        assert tax.get_compliance_rate() == 1.0

    def test_compliance_rate_none_paid(self) -> None:
        """Compliance is 0.0 when nobody pays."""
        tax = TaxComplianceMetrics()
        for i in range(5):
            tax.record_tax_event(f"agent_{i}", amount=100.0, paid=False)
        assert tax.get_compliance_rate() == 0.0

    def test_compliance_rate_partial(self) -> None:
        """Compliance is correct for mixed payment outcomes."""
        tax = TaxComplianceMetrics()
        tax.record_tax_event("a1", amount=100.0, paid=True)
        tax.record_tax_event("a2", amount=100.0, paid=False)
        tax.record_tax_event("a3", amount=100.0, paid=True)
        tax.record_tax_event("a4", amount=100.0, paid=True)
        assert tax.get_compliance_rate() == pytest.approx(0.75)

    def test_compliance_rate_empty(self) -> None:
        """Compliance is 0.0 when no events recorded."""
        tax = TaxComplianceMetrics()
        assert tax.get_compliance_rate() == 0.0

    def test_compliance_rate_filters_by_condition(self) -> None:
        """Events from other conditions are excluded."""
        tax = TaxComplianceMetrics()
        tax.record_tax_event("a1", amount=100.0, paid=True, condition="normal")
        tax.record_tax_event("a2", amount=100.0, paid=False, condition="anti_tax")
        tax.record_tax_event("a3", amount=100.0, paid=True, condition="normal")

        assert tax.get_compliance_rate("normal") == 1.0
        assert tax.get_compliance_rate("anti_tax") == 0.0

    def test_compare_conditions(self) -> None:
        """Compare conditions returns rate per condition."""
        tax = TaxComplianceMetrics()
        # Normal: 4 paid out of 5
        for i in range(4):
            tax.record_tax_event(f"a{i}", amount=100.0, paid=True, condition="normal")
        tax.record_tax_event("a4", amount=100.0, paid=False, condition="normal")

        # Anti-tax: 2 paid out of 5
        for i in range(2):
            tax.record_tax_event(f"b{i}", amount=100.0, paid=True, condition="anti_tax")
        for i in range(3):
            tax.record_tax_event(f"c{i}", amount=100.0, paid=False, condition="anti_tax")

        result = tax.compare_conditions(["normal", "anti_tax"])
        assert result["normal"] == pytest.approx(0.8)
        assert result["anti_tax"] == pytest.approx(0.4)

    def test_compare_conditions_paper_scenario(self) -> None:
        """Paper reports ~15% compliance drop under anti-tax condition."""
        tax = TaxComplianceMetrics()
        # Normal: 85% compliance
        for i in range(85):
            tax.record_tax_event(f"n{i}", amount=10.0, paid=True, condition="normal")
        for i in range(15):
            tax.record_tax_event(f"nd{i}", amount=10.0, paid=False, condition="normal")

        # Anti-tax: 70% compliance (~15% drop)
        for i in range(70):
            tax.record_tax_event(f"a{i}", amount=10.0, paid=True, condition="anti_tax")
        for i in range(30):
            tax.record_tax_event(f"ad{i}", amount=10.0, paid=False, condition="anti_tax")

        result = tax.compare_conditions(["normal", "anti_tax"])
        drop = result["normal"] - result["anti_tax"]
        assert drop == pytest.approx(0.15)

    def test_compare_conditions_unknown_condition(self) -> None:
        """Unknown condition returns 0.0 compliance."""
        tax = TaxComplianceMetrics()
        tax.record_tax_event("a1", amount=100.0, paid=True, condition="normal")

        result = tax.compare_conditions(["normal", "nonexistent"])
        assert result["normal"] == 1.0
        assert result["nonexistent"] == 0.0

    def test_total_events_count(self) -> None:
        """Total events tracks all recorded events."""
        tax = TaxComplianceMetrics()
        assert tax.total_events == 0
        tax.record_tax_event("a1", amount=50.0, paid=True)
        tax.record_tax_event("a2", amount=50.0, paid=False)
        assert tax.total_events == 2

    def test_get_agent_compliance(self) -> None:
        """Per-agent compliance is computed correctly."""
        tax = TaxComplianceMetrics()
        tax.record_tax_event("a1", amount=100.0, paid=True)
        tax.record_tax_event("a1", amount=100.0, paid=True)
        tax.record_tax_event("a1", amount=100.0, paid=False)

        assert tax.get_agent_compliance("a1") == pytest.approx(2.0 / 3.0)
        assert tax.get_agent_compliance("unknown") == 0.0


# =========================================================================
# VotingMetrics
# =========================================================================


class TestVotingMetrics:
    """Tests for VotingMetrics."""

    def test_participation_rate_with_total_agents(self) -> None:
        """Participation is unique_voters / total_agents."""
        voting = VotingMetrics(total_agents=10)
        voting.record_vote("a1", "prop1", vote=True)
        voting.record_vote("a2", "prop1", vote=False)
        voting.record_vote("a3", "prop1", vote=True)

        assert voting.get_participation_rate() == pytest.approx(0.3)

    def test_participation_rate_without_total_agents(self) -> None:
        """Without total_agents, participation is always 1.0."""
        voting = VotingMetrics()
        voting.record_vote("a1", "prop1", vote=True)
        voting.record_vote("a2", "prop1", vote=False)

        assert voting.get_participation_rate() == 1.0

    def test_participation_rate_empty(self) -> None:
        """Participation is 0.0 with no votes."""
        voting = VotingMetrics(total_agents=10)
        assert voting.get_participation_rate() == 0.0

    def test_participation_rate_duplicate_voters(self) -> None:
        """Same agent voting twice still counts as one participant."""
        voting = VotingMetrics(total_agents=5)
        voting.record_vote("a1", "prop1", vote=True)
        voting.record_vote("a1", "prop2", vote=False)

        assert voting.get_participation_rate() == pytest.approx(0.2)

    def test_consensus_rate_unanimous_yes(self) -> None:
        """Consensus is 1.0 when all vote yes."""
        voting = VotingMetrics()
        for i in range(5):
            voting.record_vote(f"a{i}", "prop1", vote=True)
        assert voting.get_consensus_rate("prop1") == 1.0

    def test_consensus_rate_unanimous_no(self) -> None:
        """Consensus is 0.0 when all vote no."""
        voting = VotingMetrics()
        for i in range(5):
            voting.record_vote(f"a{i}", "prop1", vote=False)
        assert voting.get_consensus_rate("prop1") == 0.0

    def test_consensus_rate_mixed(self) -> None:
        """Consensus reflects fraction of yes votes."""
        voting = VotingMetrics()
        voting.record_vote("a1", "prop1", vote=True)
        voting.record_vote("a2", "prop1", vote=True)
        voting.record_vote("a3", "prop1", vote=False)

        assert voting.get_consensus_rate("prop1") == pytest.approx(2.0 / 3.0)

    def test_consensus_rate_unknown_proposal(self) -> None:
        """Unknown proposal returns 0.0."""
        voting = VotingMetrics()
        voting.record_vote("a1", "prop1", vote=True)
        assert voting.get_consensus_rate("unknown") == 0.0

    def test_consensus_rate_multiple_proposals(self) -> None:
        """Consensus is computed per proposal."""
        voting = VotingMetrics()
        voting.record_vote("a1", "prop1", vote=True)
        voting.record_vote("a2", "prop1", vote=True)
        voting.record_vote("a1", "prop2", vote=False)
        voting.record_vote("a2", "prop2", vote=False)

        assert voting.get_consensus_rate("prop1") == 1.0
        assert voting.get_consensus_rate("prop2") == 0.0

    def test_influence_correlation_positive(self) -> None:
        """Positive correlation when influenced agents tend to vote yes."""
        voting = VotingMetrics()
        # Influenced agents vote yes
        for i in range(10):
            voting.record_vote(f"a{i}", "prop1", vote=True, influenced_by=["leader"])
        # Uninfluenced agents vote no
        for i in range(10, 20):
            voting.record_vote(f"a{i}", "prop1", vote=False, influenced_by=[])

        r = voting.get_influence_correlation()
        assert r > 0.9

    def test_influence_correlation_no_variance(self) -> None:
        """Returns 0.0 when all votes or all influences are the same."""
        voting = VotingMetrics()
        # Everyone votes yes regardless of influence -> no variance in vote
        for i in range(5):
            voting.record_vote(
                f"a{i}", "prop1", vote=True, influenced_by=["leader"] if i % 2 == 0 else []
            )
        # At least vote has no variance for all-yes: r should still handle it
        # Actually vote column is all True here, so variance of vote = 0 -> r = 0
        r = voting.get_influence_correlation()
        assert r == 0.0

    def test_influence_correlation_insufficient_data(self) -> None:
        """Returns 0.0 when fewer than 2 votes."""
        voting = VotingMetrics()
        voting.record_vote("a1", "prop1", vote=True)
        assert voting.get_influence_correlation() == 0.0

    def test_influence_correlation_empty(self) -> None:
        """Returns 0.0 when no votes recorded."""
        voting = VotingMetrics()
        assert voting.get_influence_correlation() == 0.0

    def test_total_votes(self) -> None:
        """Total votes counts all recorded votes."""
        voting = VotingMetrics()
        assert voting.total_votes == 0
        voting.record_vote("a1", "prop1", vote=True)
        voting.record_vote("a2", "prop1", vote=False)
        assert voting.total_votes == 2

    def test_get_proposal_ids(self) -> None:
        """Unique proposal IDs are returned sorted."""
        voting = VotingMetrics()
        voting.record_vote("a1", "prop_b", vote=True)
        voting.record_vote("a2", "prop_a", vote=False)
        voting.record_vote("a3", "prop_b", vote=True)

        assert voting.get_proposal_ids() == ["prop_a", "prop_b"]


# =========================================================================
# ConstitutionMetrics
# =========================================================================


class TestConstitutionMetrics:
    """Tests for ConstitutionMetrics."""

    def test_record_rule_and_behavior(self) -> None:
        """Basic recording of rules and behaviours."""
        cm = ConstitutionMetrics()
        cm.record_rule("r1", "No stealing", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))
        cm.record_behavior("a1", "r1", compliant=True)
        cm.record_behavior("a2", "r1", compliant=False)

        assert cm.total_rules == 1
        assert cm.total_behaviors == 2

    def test_rule_compliance_all_compliant(self) -> None:
        """Compliance is 1.0 when all obey the rule."""
        cm = ConstitutionMetrics()
        cm.record_rule("r1", "No stealing", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))
        for i in range(5):
            cm.record_behavior(f"a{i}", "r1", compliant=True)
        assert cm.get_rule_compliance("r1") == 1.0

    def test_rule_compliance_none_compliant(self) -> None:
        """Compliance is 0.0 when nobody obeys."""
        cm = ConstitutionMetrics()
        cm.record_rule("r1", "No stealing", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))
        for i in range(5):
            cm.record_behavior(f"a{i}", "r1", compliant=False)
        assert cm.get_rule_compliance("r1") == 0.0

    def test_rule_compliance_partial(self) -> None:
        """Compliance reflects the fraction of compliant behaviours."""
        cm = ConstitutionMetrics()
        cm.record_rule("r1", "No stealing", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))
        cm.record_behavior("a1", "r1", compliant=True)
        cm.record_behavior("a2", "r1", compliant=True)
        cm.record_behavior("a3", "r1", compliant=False)

        assert cm.get_rule_compliance("r1") == pytest.approx(2.0 / 3.0)

    def test_rule_compliance_unknown_rule(self) -> None:
        """Unknown rule returns 0.0."""
        cm = ConstitutionMetrics()
        assert cm.get_rule_compliance("nonexistent") == 0.0

    def test_rule_compliance_multiple_rules(self) -> None:
        """Each rule's compliance is tracked independently."""
        cm = ConstitutionMetrics()
        cm.record_rule("r1", "No stealing", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))
        cm.record_rule("r2", "Share resources", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))

        cm.record_behavior("a1", "r1", compliant=True)
        cm.record_behavior("a2", "r1", compliant=True)
        cm.record_behavior("a1", "r2", compliant=False)
        cm.record_behavior("a2", "r2", compliant=False)

        assert cm.get_rule_compliance("r1") == 1.0
        assert cm.get_rule_compliance("r2") == 0.0

    def test_amendment_effect_positive_change(self) -> None:
        """Positive delta when compliance improves after amendment."""
        cm = ConstitutionMetrics()
        cm.record_rule("r1", "No stealing", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))

        base_time = datetime(2025, 6, 1, tzinfo=UTC)

        # Before window: 50% compliance
        for i in range(5):
            b = BehaviorRecordHelper(cm, "r1", base_time + timedelta(hours=i))
            b.record(f"a{i}", compliant=(i % 2 == 0))

        # After window: 100% compliance
        after_base = base_time + timedelta(days=1)
        for i in range(5):
            b = BehaviorRecordHelper(cm, "r1", after_base + timedelta(hours=i))
            b.record(f"a{i}", compliant=True)

        delta = cm.get_amendment_effect(
            rule_id="r1",
            before_window=(base_time, base_time + timedelta(hours=5)),
            after_window=(after_base, after_base + timedelta(hours=5)),
        )
        # 3/5 compliant before (i=0,2,4) -> 0.6, 5/5 after -> 1.0, delta = 0.4
        assert delta == pytest.approx(0.4)

    def test_amendment_effect_negative_change(self) -> None:
        """Negative delta when compliance worsens after amendment."""
        cm = ConstitutionMetrics()
        cm.record_rule("r1", "Share food", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))

        base_time = datetime(2025, 6, 1, tzinfo=UTC)

        # Before: all compliant
        for i in range(4):
            b = BehaviorRecordHelper(cm, "r1", base_time + timedelta(hours=i))
            b.record(f"a{i}", compliant=True)

        # After: none compliant
        after_base = base_time + timedelta(days=1)
        for i in range(4):
            b = BehaviorRecordHelper(cm, "r1", after_base + timedelta(hours=i))
            b.record(f"a{i}", compliant=False)

        delta = cm.get_amendment_effect(
            rule_id="r1",
            before_window=(base_time, base_time + timedelta(hours=4)),
            after_window=(after_base, after_base + timedelta(hours=4)),
        )
        assert delta == pytest.approx(-1.0)

    def test_amendment_effect_empty_windows(self) -> None:
        """Returns 0.0 when either window has no behaviours."""
        cm = ConstitutionMetrics()
        cm.record_rule("r1", "No stealing", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))

        base_time = datetime(2025, 6, 1, tzinfo=UTC)
        delta = cm.get_amendment_effect(
            rule_id="r1",
            before_window=(base_time, base_time + timedelta(hours=1)),
            after_window=(base_time + timedelta(days=1), base_time + timedelta(days=1, hours=1)),
        )
        assert delta == 0.0

    def test_get_all_compliance(self) -> None:
        """get_all_compliance returns a dict for every recorded rule."""
        cm = ConstitutionMetrics()
        cm.record_rule("r1", "Rule 1", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))
        cm.record_rule("r2", "Rule 2", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))

        cm.record_behavior("a1", "r1", compliant=True)
        cm.record_behavior("a1", "r2", compliant=False)

        result = cm.get_all_compliance()
        assert result == {"r1": 1.0, "r2": 0.0}


# =========================================================================
# GovernanceReport
# =========================================================================


class TestGovernanceReport:
    """Tests for GovernanceReport model."""

    def test_creation(self) -> None:
        """Report can be created with valid data."""
        report = GovernanceReport(
            tax_compliance_rate=0.85,
            voting_participation=0.7,
            rule_compliance={"r1": 0.9, "r2": 0.6},
            timestamp=datetime.now(UTC),
        )
        assert report.tax_compliance_rate == 0.85
        assert report.voting_participation == 0.7
        assert report.rule_compliance["r1"] == 0.9

    def test_bounds_validation(self) -> None:
        """Fields reject out-of-range values."""
        with pytest.raises(ValueError):
            GovernanceReport(
                tax_compliance_rate=-0.1,
                voting_participation=0.5,
                timestamp=datetime.now(UTC),
            )
        with pytest.raises(ValueError):
            GovernanceReport(
                tax_compliance_rate=1.1,
                voting_participation=0.5,
                timestamp=datetime.now(UTC),
            )
        with pytest.raises(ValueError):
            GovernanceReport(
                tax_compliance_rate=0.5,
                voting_participation=-0.1,
                timestamp=datetime.now(UTC),
            )
        with pytest.raises(ValueError):
            GovernanceReport(
                tax_compliance_rate=0.5,
                voting_participation=1.1,
                timestamp=datetime.now(UTC),
            )

    def test_default_metadata(self) -> None:
        """Metadata defaults to empty dict."""
        report = GovernanceReport(
            tax_compliance_rate=0.5,
            voting_participation=0.5,
            timestamp=datetime.now(UTC),
        )
        assert report.metadata == {}
        assert report.rule_compliance == {}


# =========================================================================
# generate_governance_report
# =========================================================================


class TestGenerateGovernanceReport:
    """Tests for generate_governance_report function."""

    def test_basic_report_generation(self) -> None:
        """Generate a report from populated metrics objects."""
        tax = TaxComplianceMetrics()
        tax.record_tax_event("a1", amount=100.0, paid=True)
        tax.record_tax_event("a2", amount=100.0, paid=False)

        voting = VotingMetrics(total_agents=5)
        voting.record_vote("a1", "prop1", vote=True)
        voting.record_vote("a2", "prop1", vote=False)

        constitution = ConstitutionMetrics()
        constitution.record_rule("r1", "No griefing", enacted_at=datetime(2025, 1, 1, tzinfo=UTC))
        constitution.record_behavior("a1", "r1", compliant=True)

        report = generate_governance_report(tax, voting, constitution)

        assert isinstance(report, GovernanceReport)
        assert report.tax_compliance_rate == pytest.approx(0.5)
        assert report.voting_participation == pytest.approx(0.4)
        assert report.rule_compliance == {"r1": 1.0}
        assert report.metadata["total_tax_events"] == 2
        assert report.metadata["total_votes"] == 2
        assert report.metadata["total_rules"] == 1
        assert report.metadata["total_behaviors"] == 1

    def test_report_with_custom_metadata(self) -> None:
        """Custom metadata is merged into the report."""
        tax = TaxComplianceMetrics()
        voting = VotingMetrics()
        constitution = ConstitutionMetrics()

        custom = {"experiment_id": "exp42", "duration_hours": 4}
        report = generate_governance_report(tax, voting, constitution, metadata=custom)

        assert report.metadata["experiment_id"] == "exp42"
        assert report.metadata["duration_hours"] == 4
        assert "total_tax_events" in report.metadata

    def test_report_empty_metrics(self) -> None:
        """Report from empty metrics has sensible defaults."""
        tax = TaxComplianceMetrics()
        voting = VotingMetrics()
        constitution = ConstitutionMetrics()

        report = generate_governance_report(tax, voting, constitution)

        assert report.tax_compliance_rate == 0.0
        assert report.voting_participation == 0.0
        assert report.rule_compliance == {}
        assert report.metadata["total_tax_events"] == 0

    def test_report_timestamp(self) -> None:
        """Report includes a recent timestamp."""
        tax = TaxComplianceMetrics()
        voting = VotingMetrics()
        constitution = ConstitutionMetrics()

        before = datetime.now(UTC)
        report = generate_governance_report(tax, voting, constitution)
        after = datetime.now(UTC)

        assert before <= report.timestamp <= after

    def test_report_includes_proposal_consensus(self) -> None:
        """Proposal-level consensus rates are in metadata."""
        tax = TaxComplianceMetrics()
        voting = VotingMetrics()
        voting.record_vote("a1", "prop1", vote=True)
        voting.record_vote("a2", "prop1", vote=True)
        voting.record_vote("a3", "prop1", vote=False)
        constitution = ConstitutionMetrics()

        report = generate_governance_report(tax, voting, constitution)

        assert "proposal_consensus" in report.metadata
        assert report.metadata["proposal_consensus"]["prop1"] == pytest.approx(2.0 / 3.0)


# =========================================================================
# Helper
# =========================================================================


class BehaviorRecordHelper:
    """Helper to record behaviours with explicit timestamps for window tests."""

    def __init__(self, cm: ConstitutionMetrics, rule_id: str, timestamp: datetime) -> None:
        self._cm = cm
        self._rule_id = rule_id
        self._timestamp = timestamp

    def record(self, agent_id: str, compliant: bool) -> None:
        from piano.eval.governance import BehaviorRecord

        record = BehaviorRecord(
            agent_id=agent_id,
            rule_id=self._rule_id,
            compliant=compliant,
            timestamp=self._timestamp,
        )
        self._cm._behaviors.append(record)
