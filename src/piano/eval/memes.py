"""Cultural meme detection and tracking for PIANO architecture.

Implements meme (belief, custom, story, slang, ritual) detection, transmission
tracking, and spread analysis based on epidemiological SIR model fitting.
Supports inter-town cultural diversity measurement via Jaccard similarity.

Reference: Paper Section 5 — Cultural Propagation Benchmark
"""

from __future__ import annotations

__all__ = [
    "Meme",
    "MemeAnalyzer",
    "MemeCategory",
    "MemeSpread",
    "MemeTracker",
    "SIRParams",
    "TransmissionRecord",
    "fit_sir_model",
]

import math
from datetime import UTC, datetime
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums & Models
# ---------------------------------------------------------------------------


class MemeCategory(StrEnum):
    """Category of a cultural meme."""

    BELIEF = "belief"
    CUSTOM = "custom"
    STORY = "story"
    SLANG = "slang"
    RITUAL = "ritual"


class Meme(BaseModel):
    """A cultural meme that can propagate between agents."""

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    content: str = Field(description="Textual content / description of the meme")
    origin_agent: str = Field(description="Agent ID that first produced this meme")
    origin_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    category: MemeCategory = Field(default=MemeCategory.BELIEF)
    carriers: set[str] = Field(
        default_factory=set, description="Set of agent IDs currently carrying this meme"
    )


class TransmissionRecord(BaseModel):
    """Record of a single meme transmission event."""

    meme_id: str
    from_agent: str
    to_agent: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemeSpread(BaseModel):
    """Snapshot of how far a meme has spread."""

    meme_id: str
    carrier_count: int = 0
    total_transmissions: int = 0
    spread_rate: float = 0.0
    geographic_range: float = 0.0


class SIRParams(BaseModel):
    """Parameters of the SIR epidemiological model fitted to meme spread."""

    beta: float = Field(description="Infection (transmission) rate")
    gamma: float = Field(description="Recovery (forgetting) rate")
    r0: float = Field(description="Basic reproduction number (beta / gamma)")


# ---------------------------------------------------------------------------
# MemeTracker
# ---------------------------------------------------------------------------


class MemeTracker:
    """Tracks cultural memes across an agent population.

    Provides meme detection from utterances, transmission recording, and
    spread analysis including geographic range computation.
    """

    def __init__(self, similarity_threshold: float = 0.6) -> None:
        """Initialise the tracker.

        Args:
            similarity_threshold: Minimum string similarity (0-1) for an
                utterance to be considered a match to an existing meme.
        """
        self._memes: dict[str, Meme] = {}
        self._transmissions: list[TransmissionRecord] = []
        self._similarity_threshold = similarity_threshold
        self._similarity_cache: dict[tuple[str, str], float] = {}

    # -- properties ----------------------------------------------------------

    @property
    def memes(self) -> dict[str, Meme]:
        """Return internal meme registry (read-only view)."""
        return dict(self._memes)

    @property
    def transmissions(self) -> list[TransmissionRecord]:
        """Return full transmission log."""
        return list(self._transmissions)

    # -- core API ------------------------------------------------------------

    def register_meme(self, meme: Meme) -> Meme:
        """Explicitly register a new meme.

        The origin agent is automatically added to the carriers set.

        Args:
            meme: The meme to register.

        Returns:
            The registered meme (with updated carriers).
        """
        meme.carriers.add(meme.origin_agent)
        self._memes[meme.id] = meme
        logger.info("meme_registered", meme_id=meme.id, category=meme.category)
        return meme

    def detect_meme(self, agent_id: str, utterance: str) -> list[Meme]:
        """Detect known memes present in an agent's utterance.

        Uses fuzzy string matching (SequenceMatcher) to find existing memes
        whose content appears in the utterance.  If a match is found the
        agent is added to the meme's carrier set.

        Args:
            agent_id: ID of the speaking agent.
            utterance: Raw utterance text.

        Returns:
            List of matched ``Meme`` objects (possibly empty).
        """
        if not utterance:
            return []

        matched: list[Meme] = []
        utterance_lower = utterance.lower()

        for meme in self._memes.values():
            meme_lower = meme.content.lower()

            # Early exit: SequenceMatcher ratio cannot exceed
            # 2 * min(len_a, len_b) / (len_a + len_b), so skip when
            # the length difference makes it impossible to reach threshold.
            len_a = len(meme_lower)
            len_b = len(utterance_lower)
            total_len = len_a + len_b
            if total_len > 0:
                max_possible = 2.0 * min(len_a, len_b) / total_len
                if max_possible < self._similarity_threshold:
                    continue

            cache_key = (meme_lower, utterance_lower)
            cached = self._similarity_cache.get(cache_key)
            if cached is not None:
                ratio = cached
            else:
                ratio = SequenceMatcher(None, meme_lower, utterance_lower).ratio()
                self._similarity_cache[cache_key] = ratio

            if ratio >= self._similarity_threshold:
                meme.carriers.add(agent_id)
                matched.append(meme)
                logger.debug(
                    "meme_detected",
                    meme_id=meme.id,
                    agent_id=agent_id,
                    similarity=round(ratio, 3),
                )

        return matched

    def record_transmission(self, meme_id: str, from_agent: str, to_agent: str) -> None:
        """Record a meme transmission from one agent to another.

        The receiving agent is added to the meme's carrier set.

        Args:
            meme_id: ID of the meme being transmitted.
            from_agent: Sender agent ID.
            to_agent: Receiver agent ID.

        Raises:
            KeyError: If the meme_id is not registered.
        """
        if meme_id not in self._memes:
            msg = f"Unknown meme_id: {meme_id}"
            raise KeyError(msg)

        meme = self._memes[meme_id]
        meme.carriers.add(to_agent)
        record = TransmissionRecord(
            meme_id=meme_id,
            from_agent=from_agent,
            to_agent=to_agent,
        )
        self._transmissions.append(record)
        logger.debug(
            "meme_transmitted",
            meme_id=meme_id,
            from_agent=from_agent,
            to_agent=to_agent,
        )

    def get_meme_spread(self, meme_id: str) -> MemeSpread:
        """Compute spread statistics for a single meme.

        Args:
            meme_id: The meme to query.

        Returns:
            A ``MemeSpread`` snapshot.

        Raises:
            KeyError: If the meme_id is not registered.
        """
        if meme_id not in self._memes:
            msg = f"Unknown meme_id: {meme_id}"
            raise KeyError(msg)

        meme = self._memes[meme_id]
        meme_transmissions = [t for t in self._transmissions if t.meme_id == meme_id]
        total_tx = len(meme_transmissions)
        carrier_count = len(meme.carriers)

        # spread_rate: transmissions per carrier (average efficiency)
        spread_rate = total_tx / carrier_count if carrier_count > 0 else 0.0

        return MemeSpread(
            meme_id=meme_id,
            carrier_count=carrier_count,
            total_transmissions=total_tx,
            spread_rate=spread_rate,
        )

    def get_geographic_spread(
        self, meme_id: str, positions: dict[str, tuple[float, float]]
    ) -> float:
        """Compute the geographic spread (max pairwise distance) of a meme.

        Uses the maximum Euclidean distance between any two carrier positions
        as the geographic range metric.

        Args:
            meme_id: Meme to query.
            positions: Mapping of agent_id -> (x, z) position.

        Returns:
            Maximum pairwise Euclidean distance between carriers, or 0.0 if
            fewer than two carriers have known positions.

        Raises:
            KeyError: If the meme_id is not registered.
        """
        if meme_id not in self._memes:
            msg = f"Unknown meme_id: {meme_id}"
            raise KeyError(msg)

        meme = self._memes[meme_id]
        carrier_positions = [positions[a] for a in meme.carriers if a in positions]

        if len(carrier_positions) < 2:
            return 0.0

        max_dist = 0.0
        for i in range(len(carrier_positions)):
            for j in range(i + 1, len(carrier_positions)):
                dx = carrier_positions[i][0] - carrier_positions[j][0]
                dz = carrier_positions[i][1] - carrier_positions[j][1]
                dist = math.sqrt(dx * dx + dz * dz)
                max_dist = max(max_dist, dist)

        return max_dist

    def get_active_memes(self) -> list[Meme]:
        """Return all registered memes that have at least one carrier.

        Returns:
            List of active ``Meme`` objects sorted by carrier count (desc).
        """
        active = [m for m in self._memes.values() if len(m.carriers) > 0]
        active.sort(key=lambda m: len(m.carriers), reverse=True)
        return active

    def get_meme(self, meme_id: str) -> Meme:
        """Retrieve a meme by ID.

        Args:
            meme_id: The meme to retrieve.

        Returns:
            The requested ``Meme``.

        Raises:
            KeyError: If the meme_id is not registered.
        """
        if meme_id not in self._memes:
            msg = f"Unknown meme_id: {meme_id}"
            raise KeyError(msg)
        return self._memes[meme_id]


# ---------------------------------------------------------------------------
# MemeAnalyzer — inter-town cultural comparison
# ---------------------------------------------------------------------------


class MemeAnalyzer:
    """Analyses cultural profiles and diversity across towns.

    Provides Jaccard similarity between meme sets and a population-level
    cultural diversity index.
    """

    @staticmethod
    def jaccard_similarity(town_a_memes: set[str], town_b_memes: set[str]) -> float:
        """Compute the Jaccard index between two meme sets.

        J(A, B) = |A & B| / |A | B| (union)

        Args:
            town_a_memes: Set of meme IDs held by town A.
            town_b_memes: Set of meme IDs held by town B.

        Returns:
            Jaccard similarity in [0.0, 1.0].  Returns 0.0 if both sets empty.
        """
        if not town_a_memes and not town_b_memes:
            return 0.0

        intersection = town_a_memes & town_b_memes
        union = town_a_memes | town_b_memes

        return len(intersection) / len(union)

    @staticmethod
    def get_town_meme_profiles(
        town_assignments: dict[str, str], memes: list[Meme]
    ) -> dict[str, set[str]]:
        """Build per-town meme profiles.

        For each town, collect the set of meme IDs whose carriers include at
        least one agent assigned to that town.

        Args:
            town_assignments: Mapping of agent_id -> town_name.
            memes: List of all memes to consider.

        Returns:
            Dict mapping town_name -> set of meme IDs present in that town.
        """
        profiles: dict[str, set[str]] = {}
        # Pre-build town -> agents mapping
        town_agents: dict[str, set[str]] = {}
        for agent_id, town in town_assignments.items():
            town_agents.setdefault(town, set()).add(agent_id)

        for town, agents in town_agents.items():
            town_meme_ids: set[str] = set()
            for meme in memes:
                if meme.carriers & agents:
                    town_meme_ids.add(meme.id)
            profiles[town] = town_meme_ids

        return profiles

    @staticmethod
    def cultural_diversity_index(town_profiles: dict[str, set[str]]) -> float:
        """Compute cultural diversity across towns.

        The index is 1.0 - mean(Jaccard similarity) over all town pairs.
        A value of 1.0 means completely distinct cultures; 0.0 means
        identical meme sets across all towns.

        Args:
            town_profiles: Dict mapping town -> set of meme IDs.

        Returns:
            Diversity index in [0.0, 1.0].  Returns 0.0 when fewer than
            two towns are present.
        """
        towns = list(town_profiles.keys())
        if len(towns) < 2:
            return 0.0

        pair_count = 0
        total_jaccard = 0.0
        for i in range(len(towns)):
            for j in range(i + 1, len(towns)):
                sim = MemeAnalyzer.jaccard_similarity(
                    town_profiles[towns[i]], town_profiles[towns[j]]
                )
                total_jaccard += sim
                pair_count += 1

        mean_sim = total_jaccard / pair_count if pair_count > 0 else 0.0
        return 1.0 - mean_sim


# ---------------------------------------------------------------------------
# SIR model fitting
# ---------------------------------------------------------------------------


def fit_sir_model(meme_spread_timeseries: list[dict[str, Any]]) -> SIRParams:
    """Fit SIR epidemiological parameters to meme spread data.

    Uses a simple discrete-time estimation approach:
    - beta is estimated from the growth rate of new infections
    - gamma is estimated from the recovery (carrier loss) rate
    - R0 = beta / gamma

    Each element of *meme_spread_timeseries* should have at least:
    - ``susceptible`` (int): agents who do not yet carry the meme
    - ``infected`` (int): agents currently carrying the meme
    - ``recovered`` (int): agents who previously carried but no longer do

    When there are fewer than two data points or the data is degenerate,
    sensible defaults are returned.

    Args:
        meme_spread_timeseries: Time-ordered list of SIR state dicts.

    Returns:
        Fitted ``SIRParams``.
    """
    if len(meme_spread_timeseries) < 2:
        return SIRParams(beta=0.0, gamma=0.0, r0=0.0)

    beta_estimates: list[float] = []
    gamma_estimates: list[float] = []

    for k in range(len(meme_spread_timeseries) - 1):
        cur = meme_spread_timeseries[k]
        nxt = meme_spread_timeseries[k + 1]

        s_cur = cur.get("susceptible", 0)
        i_cur = cur.get("infected", 0)
        r_nxt = nxt.get("recovered", 0)
        r_cur = cur.get("recovered", 0)
        s_nxt = nxt.get("susceptible", 0)

        n = s_cur + i_cur + r_cur
        if n == 0:
            continue

        # New infections: delta_S (susceptible lost)
        new_infections = s_cur - s_nxt
        # New recoveries
        new_recoveries = r_nxt - r_cur

        # beta = new_infections / (S * I / N)
        denom_beta = (s_cur * i_cur) / n if n > 0 else 0
        if denom_beta > 0 and new_infections >= 0:
            beta_estimates.append(new_infections / denom_beta)

        # gamma = new_recoveries / I
        if i_cur > 0 and new_recoveries >= 0:
            gamma_estimates.append(new_recoveries / i_cur)

    beta = sum(beta_estimates) / len(beta_estimates) if beta_estimates else 0.0
    gamma = sum(gamma_estimates) / len(gamma_estimates) if gamma_estimates else 0.0
    r0 = beta / gamma if gamma > 0 else 0.0

    logger.info("sir_model_fitted", beta=round(beta, 4), gamma=round(gamma, 4), r0=round(r0, 4))

    return SIRParams(beta=beta, gamma=gamma, r0=r0)
