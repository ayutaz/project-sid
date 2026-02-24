"""Governance evaluation metrics for PIANO architecture.

Implements institutional governance benchmarks from paper Section 5:
1. Tax compliance rates (with condition-based comparison)
2. Voting behavior (participation, consensus, influence correlation)
3. Constitutional rule compliance and amendment effects

Reference:
    Paper reports ~15% drop in tax compliance under anti-tax conditions,
    and measures how constitutional amendments affect agent behavior.
"""

from __future__ import annotations

__all__ = [
    "ConstitutionMetrics",
    "GovernanceReport",
    "TaxComplianceMetrics",
    "VotingMetrics",
    "generate_governance_report",
]

from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

from piano.eval.social_metrics import _manual_pearson

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TaxEvent(BaseModel):
    """A single tax event for an agent."""

    agent_id: str
    amount: float = Field(ge=0.0, description="Tax amount due")
    paid: bool = Field(description="Whether the agent paid the tax")
    condition: str = Field(default="normal", description="Experimental condition label")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VoteRecord(BaseModel):
    """A single vote cast by an agent."""

    agent_id: str
    proposal_id: str
    vote: bool = Field(description="True = in favour, False = against")
    influenced_by: list[str] = Field(
        default_factory=list, description="Agent IDs that influenced this vote"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RuleRecord(BaseModel):
    """A constitutional rule."""

    rule_id: str
    rule_text: str
    enacted_at: datetime


class BehaviorRecord(BaseModel):
    """An observed behaviour relative to a rule."""

    agent_id: str
    rule_id: str
    compliant: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Tax compliance
# ---------------------------------------------------------------------------


class TaxComplianceMetrics:
    """Metrics for tax compliance evaluation.

    Tracks tax events per agent and condition, computing compliance rates
    and cross-condition comparisons (paper: ~15% drop under anti-tax).
    """

    def __init__(self) -> None:
        self._events: list[TaxEvent] = []

    # -- recording ----------------------------------------------------------

    def record_tax_event(
        self,
        agent_id: str,
        amount: float,
        paid: bool,
        condition: str = "normal",
    ) -> None:
        """Record a tax event.

        Args:
            agent_id: The agent that owes the tax.
            amount: Tax amount due.
            paid: Whether the agent paid.
            condition: Experimental condition (e.g. "normal", "anti_tax").
        """
        event = TaxEvent(
            agent_id=agent_id,
            amount=amount,
            paid=paid,
            condition=condition,
        )
        self._events.append(event)
        logger.debug(
            "tax_event_recorded",
            agent_id=agent_id,
            amount=amount,
            paid=paid,
            condition=condition,
        )

    # -- queries ------------------------------------------------------------

    def get_compliance_rate(self, condition: str = "normal") -> float:
        """Return the fraction of paid taxes for a given condition.

        Args:
            condition: Experimental condition to filter by.

        Returns:
            Compliance rate in [0.0, 1.0].  Returns 0.0 when no events match.
        """
        filtered = [e for e in self._events if e.condition == condition]
        if not filtered:
            return 0.0
        paid_count = sum(1 for e in filtered if e.paid)
        return paid_count / len(filtered)

    def compare_conditions(self, conditions: list[str]) -> dict[str, float]:
        """Compare compliance rates across experimental conditions.

        Args:
            conditions: List of condition labels to compare.

        Returns:
            Dict mapping each condition to its compliance rate.

        Note:
            Paper reports ~15% lower compliance under "anti_tax" vs "normal".
        """
        result: dict[str, float] = {}
        for cond in conditions:
            result[cond] = self.get_compliance_rate(cond)
        logger.info("tax_conditions_compared", conditions=conditions, rates=result)
        return result

    # -- helpers ------------------------------------------------------------

    @property
    def total_events(self) -> int:
        """Total number of recorded tax events."""
        return len(self._events)

    def get_agent_compliance(self, agent_id: str) -> float:
        """Return compliance rate for a specific agent.

        Args:
            agent_id: The agent to query.

        Returns:
            Compliance rate in [0.0, 1.0].  Returns 0.0 when no events match.
        """
        agent_events = [e for e in self._events if e.agent_id == agent_id]
        if not agent_events:
            return 0.0
        return sum(1 for e in agent_events if e.paid) / len(agent_events)


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------


class VotingMetrics:
    """Metrics for voting behaviour evaluation.

    Tracks votes per proposal, computes participation, consensus,
    and the correlation between influence and voting direction.
    """

    def __init__(self, total_agents: int = 0) -> None:
        """Initialise voting metrics.

        Args:
            total_agents: Total number of agents eligible to vote.
                          Used for participation-rate denominator.
                          When 0, participation rate is computed from
                          unique voters in the recorded data.
        """
        self._votes: list[VoteRecord] = []
        self._total_agents = total_agents

    # -- recording ----------------------------------------------------------

    def record_vote(
        self,
        agent_id: str,
        proposal_id: str,
        vote: bool,
        influenced_by: list[str] | None = None,
    ) -> None:
        """Record a vote.

        Args:
            agent_id: Voting agent.
            proposal_id: The proposal being voted on.
            vote: True = in favour, False = against.
            influenced_by: Agent IDs that influenced this vote.
        """
        record = VoteRecord(
            agent_id=agent_id,
            proposal_id=proposal_id,
            vote=vote,
            influenced_by=influenced_by or [],
        )
        self._votes.append(record)
        logger.debug(
            "vote_recorded",
            agent_id=agent_id,
            proposal_id=proposal_id,
            vote=vote,
        )

    # -- queries ------------------------------------------------------------

    def get_participation_rate(self) -> float:
        """Return the fraction of agents that participated in any vote.

        When *total_agents* was set at construction, the rate is
        ``unique_voters / total_agents``.  Otherwise it is always 1.0
        (every recorded voter counts as a participant).

        Returns:
            Participation rate in [0.0, 1.0].  Returns 0.0 when no votes.
        """
        if not self._votes:
            return 0.0

        unique_voters = len({v.agent_id for v in self._votes})
        if self._total_agents > 0:
            return min(unique_voters / self._total_agents, 1.0)
        return 1.0

    def get_consensus_rate(self, proposal_id: str) -> float:
        """Return the fraction of votes in favour of a proposal.

        Args:
            proposal_id: The proposal to query.

        Returns:
            Consensus rate in [0.0, 1.0].  Returns 0.0 when no votes match.
        """
        proposal_votes = [v for v in self._votes if v.proposal_id == proposal_id]
        if not proposal_votes:
            return 0.0
        in_favour = sum(1 for v in proposal_votes if v.vote)
        return in_favour / len(proposal_votes)

    def get_influence_correlation(self) -> float:
        """Return the correlation between being influenced and voting 'yes'.

        Computes the Pearson r between a binary "was influenced" variable
        (1 if influenced_by is non-empty, else 0) and the vote direction
        (1 for yes, 0 for no).

        Returns:
            Pearson r in [-1.0, 1.0].  Returns 0.0 when insufficient data
            or zero-variance in either variable.
        """
        if len(self._votes) < 2:
            return 0.0

        influenced_vals = [1.0 if v.influenced_by else 0.0 for v in self._votes]
        vote_vals = [1.0 if v.vote else 0.0 for v in self._votes]

        return _manual_pearson(influenced_vals, vote_vals)

    # -- helpers ------------------------------------------------------------

    @property
    def total_votes(self) -> int:
        """Total number of recorded votes."""
        return len(self._votes)

    def get_proposal_ids(self) -> list[str]:
        """Return unique proposal IDs that have received votes."""
        return sorted({v.proposal_id for v in self._votes})


# ---------------------------------------------------------------------------
# Constitution
# ---------------------------------------------------------------------------


class ConstitutionMetrics:
    """Metrics for constitutional rule compliance and amendment effects.

    Tracks rules and observed behaviours, computing per-rule compliance
    and before/after behavioural shifts following rule amendments.
    """

    def __init__(self) -> None:
        self._rules: dict[str, RuleRecord] = {}
        self._behaviors: list[BehaviorRecord] = []

    # -- recording ----------------------------------------------------------

    def record_rule(
        self,
        rule_id: str,
        rule_text: str,
        enacted_at: datetime,
    ) -> None:
        """Record a constitutional rule.

        Args:
            rule_id: Unique identifier for the rule.
            rule_text: Human-readable description of the rule.
            enacted_at: When the rule was enacted.
        """
        record = RuleRecord(rule_id=rule_id, rule_text=rule_text, enacted_at=enacted_at)
        self._rules[rule_id] = record
        logger.debug("rule_recorded", rule_id=rule_id, enacted_at=enacted_at)

    def record_behavior(
        self,
        agent_id: str,
        rule_id: str,
        compliant: bool,
    ) -> None:
        """Record an observed behaviour relative to a rule.

        Args:
            agent_id: The agent whose behaviour was observed.
            rule_id: The rule the behaviour relates to.
            compliant: Whether the behaviour was rule-compliant.
        """
        record = BehaviorRecord(
            agent_id=agent_id,
            rule_id=rule_id,
            compliant=compliant,
        )
        self._behaviors.append(record)
        logger.debug(
            "behavior_recorded",
            agent_id=agent_id,
            rule_id=rule_id,
            compliant=compliant,
        )

    # -- queries ------------------------------------------------------------

    def get_rule_compliance(self, rule_id: str) -> float:
        """Return the overall compliance rate for a specific rule.

        Args:
            rule_id: The rule to query.

        Returns:
            Compliance rate in [0.0, 1.0].  Returns 0.0 when no behaviours match.
        """
        rule_behaviors = [b for b in self._behaviors if b.rule_id == rule_id]
        if not rule_behaviors:
            return 0.0
        compliant_count = sum(1 for b in rule_behaviors if b.compliant)
        return compliant_count / len(rule_behaviors)

    def get_amendment_effect(
        self,
        rule_id: str,
        before_window: tuple[datetime, datetime],
        after_window: tuple[datetime, datetime],
    ) -> float | None:
        """Measure the behavioural change after a rule amendment.

        Computes ``compliance_after - compliance_before`` so that a
        positive value indicates *improved* compliance after the amendment.

        Args:
            rule_id: The rule whose amendment to evaluate.
            before_window: (start, end) datetime window *before* the amendment.
            after_window: (start, end) datetime window *after* the amendment.

        Returns:
            Compliance delta (after - before).  Range is [-1.0, 1.0].
            Returns ``None`` when either window contains no matching behaviours.
        """
        before_behaviors = [
            b
            for b in self._behaviors
            if b.rule_id == rule_id
            and before_window[0] <= b.timestamp <= before_window[1]
        ]
        after_behaviors = [
            b
            for b in self._behaviors
            if b.rule_id == rule_id
            and after_window[0] <= b.timestamp <= after_window[1]
        ]

        if not before_behaviors or not after_behaviors:
            return None

        compliance_before = sum(1 for b in before_behaviors if b.compliant) / len(before_behaviors)
        compliance_after = sum(1 for b in after_behaviors if b.compliant) / len(after_behaviors)

        delta = compliance_after - compliance_before
        logger.info(
            "amendment_effect_measured",
            rule_id=rule_id,
            compliance_before=compliance_before,
            compliance_after=compliance_after,
            delta=delta,
        )
        return delta

    # -- helpers ------------------------------------------------------------

    @property
    def total_rules(self) -> int:
        """Total number of recorded rules."""
        return len(self._rules)

    @property
    def total_behaviors(self) -> int:
        """Total number of recorded behaviours."""
        return len(self._behaviors)

    def get_all_compliance(self) -> dict[str, float]:
        """Return compliance rates for every recorded rule.

        Returns:
            Dict mapping rule_id to compliance rate.
        """
        result: dict[str, float] = {}
        for rule_id in self._rules:
            result[rule_id] = self.get_rule_compliance(rule_id)
        return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class GovernanceReport(BaseModel):
    """Comprehensive governance evaluation report."""

    tax_compliance_rate: float = Field(
        ge=0.0, le=1.0, description="Overall tax compliance rate"
    )
    voting_participation: float = Field(
        ge=0.0, le=1.0, description="Voting participation rate"
    )
    rule_compliance: dict[str, float] = Field(
        default_factory=dict, description="Per-rule compliance rates"
    )
    timestamp: datetime = Field(description="When the report was generated")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context and metrics"
    )


def generate_governance_report(
    tax: TaxComplianceMetrics,
    voting: VotingMetrics,
    constitution: ConstitutionMetrics,
    metadata: dict[str, Any] | None = None,
) -> GovernanceReport:
    """Generate a comprehensive governance evaluation report.

    Args:
        tax: Tax compliance metrics instance with recorded events.
        voting: Voting metrics instance with recorded votes.
        constitution: Constitution metrics instance with rules and behaviours.
        metadata: Optional additional metadata to include.

    Returns:
        GovernanceReport summarising all governance evaluation dimensions.
    """
    tax_rate = tax.get_compliance_rate("normal")
    participation = voting.get_participation_rate()
    rule_compliance = constitution.get_all_compliance()

    extended_metadata = metadata.copy() if metadata else {}

    # Consensus rates per proposal
    proposal_consensus: dict[str, float] = {}
    for proposal_id in voting.get_proposal_ids():
        proposal_consensus[proposal_id] = voting.get_consensus_rate(proposal_id)

    extended_metadata.update(
        {
            "total_tax_events": tax.total_events,
            "total_votes": voting.total_votes,
            "total_rules": constitution.total_rules,
            "total_behaviors": constitution.total_behaviors,
            "influence_correlation": voting.get_influence_correlation(),
            "proposal_consensus": proposal_consensus,
        }
    )

    report = GovernanceReport(
        tax_compliance_rate=tax_rate,
        voting_participation=participation,
        rule_compliance=rule_compliance,
        timestamp=datetime.now(UTC),
        metadata=extended_metadata,
    )
    logger.info("governance_report_generated", tax_rate=tax_rate, participation=participation)
    return report


