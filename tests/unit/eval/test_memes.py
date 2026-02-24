"""Tests for cultural meme detection and tracking."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from piano.eval.memes import (
    Meme,
    MemeAnalyzer,
    MemeCategory,
    MemeSpread,
    MemeTracker,
    SIRParams,
    TransmissionRecord,
    fit_sir_model,
)

# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestMemeModel:
    """Tests for the Meme Pydantic model."""

    def test_meme_creation_defaults(self) -> None:
        meme = Meme(content="the sun is sacred", origin_agent="agent-001")
        assert meme.content == "the sun is sacred"
        assert meme.origin_agent == "agent-001"
        assert meme.category == MemeCategory.BELIEF
        assert isinstance(meme.id, str)
        assert len(meme.id) > 0
        assert isinstance(meme.origin_time, datetime)
        assert meme.carriers == set()

    def test_meme_creation_full(self) -> None:
        now = datetime.now(UTC)
        meme = Meme(
            id="meme-abc",
            content="always greet newcomers",
            origin_agent="agent-002",
            origin_time=now,
            category=MemeCategory.CUSTOM,
            carriers={"agent-002", "agent-003"},
        )
        assert meme.id == "meme-abc"
        assert meme.category == MemeCategory.CUSTOM
        assert meme.carriers == {"agent-002", "agent-003"}
        assert meme.origin_time == now

    def test_meme_category_values(self) -> None:
        assert MemeCategory.BELIEF == "belief"
        assert MemeCategory.CUSTOM == "custom"
        assert MemeCategory.STORY == "story"
        assert MemeCategory.SLANG == "slang"
        assert MemeCategory.RITUAL == "ritual"


class TestTransmissionRecord:
    """Tests for TransmissionRecord model."""

    def test_creation(self) -> None:
        rec = TransmissionRecord(meme_id="m1", from_agent="a1", to_agent="a2")
        assert rec.meme_id == "m1"
        assert rec.from_agent == "a1"
        assert rec.to_agent == "a2"
        assert isinstance(rec.timestamp, datetime)


class TestMemeSpread:
    """Tests for MemeSpread model."""

    def test_creation_defaults(self) -> None:
        spread = MemeSpread(meme_id="m1")
        assert spread.carrier_count == 0
        assert spread.total_transmissions == 0
        assert spread.spread_rate == 0.0
        assert spread.geographic_range == 0.0

    def test_creation_with_values(self) -> None:
        spread = MemeSpread(
            meme_id="m1",
            carrier_count=5,
            total_transmissions=12,
            spread_rate=2.4,
            geographic_range=150.0,
        )
        assert spread.carrier_count == 5
        assert spread.spread_rate == 2.4


class TestSIRParams:
    """Tests for SIRParams model."""

    def test_creation(self) -> None:
        params = SIRParams(beta=0.3, gamma=0.1, r0=3.0)
        assert params.beta == 0.3
        assert params.gamma == 0.1
        assert params.r0 == 3.0


# ---------------------------------------------------------------------------
# MemeTracker tests
# ---------------------------------------------------------------------------


class TestMemeTracker:
    """Tests for MemeTracker core functionality."""

    def _make_tracker_with_meme(self) -> tuple[MemeTracker, Meme]:
        """Helper: create a tracker with one registered meme."""
        tracker = MemeTracker(similarity_threshold=0.6)
        meme = Meme(
            id="sun-meme",
            content="the sun is sacred",
            origin_agent="agent-001",
            category=MemeCategory.BELIEF,
        )
        tracker.register_meme(meme)
        return tracker, meme

    def test_register_meme_adds_origin_as_carrier(self) -> None:
        tracker = MemeTracker()
        meme = Meme(content="greet newcomers", origin_agent="a1")
        registered = tracker.register_meme(meme)
        assert "a1" in registered.carriers

    def test_register_meme_accessible(self) -> None:
        tracker = MemeTracker()
        meme = Meme(id="m1", content="test", origin_agent="a1")
        tracker.register_meme(meme)
        assert "m1" in tracker.memes
        assert tracker.get_meme("m1").content == "test"

    def test_detect_meme_matching_utterance(self) -> None:
        tracker, meme = self._make_tracker_with_meme()
        matches = tracker.detect_meme("agent-002", "I believe the sun is sacred")
        assert len(matches) >= 1
        assert meme.id in {m.id for m in matches}
        assert "agent-002" in tracker.get_meme("sun-meme").carriers

    def test_detect_meme_no_match(self) -> None:
        tracker, _ = self._make_tracker_with_meme()
        matches = tracker.detect_meme("agent-002", "the weather is nice today")
        assert len(matches) == 0

    def test_detect_meme_empty_utterance(self) -> None:
        tracker, _ = self._make_tracker_with_meme()
        assert tracker.detect_meme("agent-002", "") == []

    def test_detect_meme_exact_match(self) -> None:
        tracker, meme = self._make_tracker_with_meme()
        matches = tracker.detect_meme("agent-005", "the sun is sacred")
        assert len(matches) == 1
        assert matches[0].id == meme.id

    def test_record_transmission_success(self) -> None:
        tracker, _meme = self._make_tracker_with_meme()
        tracker.record_transmission("sun-meme", "agent-001", "agent-002")

        assert "agent-002" in tracker.get_meme("sun-meme").carriers
        assert len(tracker.transmissions) == 1
        assert tracker.transmissions[0].from_agent == "agent-001"
        assert tracker.transmissions[0].to_agent == "agent-002"

    def test_record_transmission_unknown_meme_raises(self) -> None:
        tracker = MemeTracker()
        with pytest.raises(KeyError, match="Unknown meme_id"):
            tracker.record_transmission("nonexistent", "a1", "a2")

    def test_get_meme_spread(self) -> None:
        tracker, _meme = self._make_tracker_with_meme()
        tracker.record_transmission("sun-meme", "agent-001", "agent-002")
        tracker.record_transmission("sun-meme", "agent-001", "agent-003")

        spread = tracker.get_meme_spread("sun-meme")
        assert isinstance(spread, MemeSpread)
        assert spread.carrier_count == 3  # origin + 2 receivers
        assert spread.total_transmissions == 2
        assert spread.spread_rate == pytest.approx(2 / 3)

    def test_get_meme_spread_unknown_raises(self) -> None:
        tracker = MemeTracker()
        with pytest.raises(KeyError):
            tracker.get_meme_spread("nonexistent")

    def test_get_geographic_spread_basic(self) -> None:
        tracker, _ = self._make_tracker_with_meme()
        tracker.record_transmission("sun-meme", "agent-001", "agent-002")

        positions = {
            "agent-001": (0.0, 0.0),
            "agent-002": (3.0, 4.0),
        }
        geo = tracker.get_geographic_spread("sun-meme", positions)
        assert geo == pytest.approx(5.0)

    def test_get_geographic_spread_single_carrier(self) -> None:
        tracker, _ = self._make_tracker_with_meme()
        positions = {"agent-001": (10.0, 20.0)}
        geo = tracker.get_geographic_spread("sun-meme", positions)
        assert geo == 0.0

    def test_get_geographic_spread_unknown_raises(self) -> None:
        tracker = MemeTracker()
        with pytest.raises(KeyError):
            tracker.get_geographic_spread("nonexistent", {})

    def test_get_geographic_spread_multiple_carriers(self) -> None:
        tracker, _ = self._make_tracker_with_meme()
        tracker.record_transmission("sun-meme", "agent-001", "agent-002")
        tracker.record_transmission("sun-meme", "agent-001", "agent-003")

        positions = {
            "agent-001": (0.0, 0.0),
            "agent-002": (3.0, 4.0),
            "agent-003": (6.0, 8.0),
        }
        geo = tracker.get_geographic_spread("sun-meme", positions)
        # max distance is between (0,0) and (6,8) = 10.0
        assert geo == pytest.approx(10.0)

    def test_get_active_memes_sorted_by_carriers(self) -> None:
        tracker = MemeTracker()
        m1 = Meme(id="m1", content="meme one", origin_agent="a1")
        m2 = Meme(id="m2", content="meme two", origin_agent="a2")
        tracker.register_meme(m1)
        tracker.register_meme(m2)

        # m2 gets more carriers
        tracker.record_transmission("m2", "a2", "a3")
        tracker.record_transmission("m2", "a2", "a4")

        active = tracker.get_active_memes()
        assert len(active) == 2
        assert active[0].id == "m2"  # most carriers first

    def test_get_active_memes_excludes_empty(self) -> None:
        tracker = MemeTracker()
        meme = Meme(id="m1", content="orphan", origin_agent="a1", carriers=set())
        # Manually insert without going through register_meme (which adds origin)
        tracker._memes["m1"] = meme
        active = tracker.get_active_memes()
        assert len(active) == 0

    def test_get_meme_unknown_raises(self) -> None:
        tracker = MemeTracker()
        with pytest.raises(KeyError):
            tracker.get_meme("nonexistent")

    def test_similarity_threshold_respected(self) -> None:
        """High threshold should reject partial matches."""
        tracker = MemeTracker(similarity_threshold=0.95)
        meme = Meme(id="strict", content="the sun is sacred", origin_agent="a1")
        tracker.register_meme(meme)
        # Very different utterance should not match at high threshold
        matches = tracker.detect_meme("a2", "sacred stuff maybe")
        assert len(matches) == 0

    def test_detect_meme_early_exit_length_difference(self) -> None:
        """Memes with extreme length difference are skipped early."""
        tracker = MemeTracker(similarity_threshold=0.6)
        meme = Meme(id="long-meme", content="a" * 100, origin_agent="a1")
        tracker.register_meme(meme)
        # Very short utterance can't match (max_possible ratio too low)
        matches = tracker.detect_meme("a2", "ab")
        assert len(matches) == 0

    def test_detect_meme_cache_hit(self) -> None:
        """Repeated detection calls use the similarity cache."""
        tracker = MemeTracker(similarity_threshold=0.6)
        meme = Meme(id="cached", content="the sun is sacred", origin_agent="a1")
        tracker.register_meme(meme)

        # First call populates cache
        tracker.detect_meme("a2", "the sun is sacred")
        assert len(tracker._similarity_cache) > 0

        # Second call with same text should use cache
        tracker.detect_meme("a3", "the sun is sacred")
        # Agent a3 should also be a carrier
        assert "a3" in tracker.get_meme("cached").carriers


# ---------------------------------------------------------------------------
# MemeAnalyzer tests
# ---------------------------------------------------------------------------


class TestMemeAnalyzer:
    """Tests for inter-town cultural analysis."""

    def test_jaccard_identical_sets(self) -> None:
        s = {"m1", "m2", "m3"}
        assert MemeAnalyzer.jaccard_similarity(s, s) == 1.0

    def test_jaccard_disjoint_sets(self) -> None:
        a = {"m1", "m2"}
        b = {"m3", "m4"}
        assert MemeAnalyzer.jaccard_similarity(a, b) == 0.0

    def test_jaccard_partial_overlap(self) -> None:
        a = {"m1", "m2", "m3"}
        b = {"m2", "m3", "m4"}
        # intersection = {m2, m3} = 2, union = {m1,m2,m3,m4} = 4
        assert MemeAnalyzer.jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_jaccard_both_empty(self) -> None:
        assert MemeAnalyzer.jaccard_similarity(set(), set()) == 0.0

    def test_jaccard_one_empty(self) -> None:
        assert MemeAnalyzer.jaccard_similarity({"m1"}, set()) == 0.0

    def test_get_town_meme_profiles_basic(self) -> None:
        town_assignments = {"a1": "town-A", "a2": "town-A", "a3": "town-B"}
        memes = [
            Meme(id="m1", content="x", origin_agent="a1", carriers={"a1", "a2"}),
            Meme(id="m2", content="y", origin_agent="a3", carriers={"a3"}),
            Meme(id="m3", content="z", origin_agent="a1", carriers={"a1", "a3"}),
        ]
        profiles = MemeAnalyzer.get_town_meme_profiles(town_assignments, memes)

        assert "town-A" in profiles
        assert "town-B" in profiles
        assert profiles["town-A"] == {"m1", "m3"}  # a1 carries m1 & m3
        assert profiles["town-B"] == {"m2", "m3"}  # a3 carries m2 & m3

    def test_get_town_meme_profiles_empty(self) -> None:
        profiles = MemeAnalyzer.get_town_meme_profiles({}, [])
        assert profiles == {}

    def test_cultural_diversity_index_identical_towns(self) -> None:
        profiles = {
            "town-A": {"m1", "m2"},
            "town-B": {"m1", "m2"},
        }
        # Jaccard = 1.0, diversity = 0.0
        assert MemeAnalyzer.cultural_diversity_index(profiles) == pytest.approx(0.0)

    def test_cultural_diversity_index_completely_distinct(self) -> None:
        profiles = {
            "town-A": {"m1", "m2"},
            "town-B": {"m3", "m4"},
        }
        # Jaccard = 0.0, diversity = 1.0
        assert MemeAnalyzer.cultural_diversity_index(profiles) == pytest.approx(1.0)

    def test_cultural_diversity_index_partial(self) -> None:
        profiles = {
            "town-A": {"m1", "m2", "m3"},
            "town-B": {"m2", "m3", "m4"},
        }
        # Jaccard = 0.5, diversity = 0.5
        assert MemeAnalyzer.cultural_diversity_index(profiles) == pytest.approx(0.5)

    def test_cultural_diversity_index_single_town(self) -> None:
        profiles = {"town-A": {"m1", "m2"}}
        assert MemeAnalyzer.cultural_diversity_index(profiles) == 0.0

    def test_cultural_diversity_index_three_towns(self) -> None:
        profiles = {
            "town-A": {"m1"},
            "town-B": {"m2"},
            "town-C": {"m3"},
        }
        # All pairs disjoint => mean Jaccard = 0, diversity = 1.0
        assert MemeAnalyzer.cultural_diversity_index(profiles) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SIR model fitting tests
# ---------------------------------------------------------------------------


class TestFitSIRModel:
    """Tests for SIR model parameter estimation."""

    def test_insufficient_data(self) -> None:
        result = fit_sir_model([])
        assert result.beta == 0.0
        assert result.gamma == 0.0
        assert result.r0 == 0.0

    def test_single_data_point(self) -> None:
        result = fit_sir_model([{"susceptible": 90, "infected": 10, "recovered": 0}])
        assert result.beta == 0.0

    def test_basic_sir_spread(self) -> None:
        """Verify that a simple spreading scenario yields positive beta."""
        ts = [
            {"susceptible": 90, "infected": 10, "recovered": 0},
            {"susceptible": 80, "infected": 18, "recovered": 2},
            {"susceptible": 65, "infected": 30, "recovered": 5},
        ]
        result = fit_sir_model(ts)
        assert result.beta > 0
        assert result.gamma >= 0

    def test_no_spread(self) -> None:
        """When nothing changes, beta and gamma should be zero."""
        ts = [
            {"susceptible": 90, "infected": 10, "recovered": 0},
            {"susceptible": 90, "infected": 10, "recovered": 0},
        ]
        result = fit_sir_model(ts)
        assert result.beta == 0.0
        assert result.gamma == 0.0
        assert result.r0 == 0.0

    def test_r0_calculation(self) -> None:
        """R0 should equal beta / gamma when gamma > 0."""
        ts = [
            {"susceptible": 100, "infected": 10, "recovered": 0},
            {"susceptible": 80, "infected": 25, "recovered": 5},
        ]
        result = fit_sir_model(ts)
        if result.gamma > 0:
            assert result.r0 == pytest.approx(result.beta / result.gamma)

    def test_pure_recovery(self) -> None:
        """All infected recover, no new infections."""
        ts = [
            {"susceptible": 50, "infected": 50, "recovered": 0},
            {"susceptible": 50, "infected": 25, "recovered": 25},
        ]
        result = fit_sir_model(ts)
        assert result.beta == 0.0
        assert result.gamma > 0
        # R0 should be 0 when beta is 0
        assert result.r0 == 0.0

    def test_degenerate_zero_population(self) -> None:
        """When population is zero, should not crash."""
        ts = [
            {"susceptible": 0, "infected": 0, "recovered": 0},
            {"susceptible": 0, "infected": 0, "recovered": 0},
        ]
        result = fit_sir_model(ts)
        assert isinstance(result, SIRParams)
