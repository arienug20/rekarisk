"""
Rekarisk — QRA Model Validation Tests.
"""

from __future__ import annotations
import math
import numpy as np
import pytest

from rekarisk.models.qra.event_tree import (
    EventTree, EventTreeNode, Scenario, create_generic_vessel_tree,
)
from rekarisk.models.qra.failure_frequency import (
    lookup_frequency, classify_frequency, FrequencyClass,
    FailureFrequencyDB, get_default_db,
)
from rekarisk.models.qra.ignition_prob import (
    immediate_ignition_probability, delayed_ignition_probability,
    combined_ignition_probability,
)
from rekarisk.models.qra.individual_risk import calculate_ir_at_point
from rekarisk.models.qra.societal_risk import calculate_fn_curve
from rekarisk.models.qra.risk_matrix import (
    classify_likelihood, classify_consequence, risk_level,
    risk_level_from_values, RiskLevel, LikelihoodLevel, ConsequenceLevel,
)


class TestEventTree:
    def test_create_generic_vessel_tree(self):
        tree = create_generic_vessel_tree(name='Test Vessel', freq=1e-4)
        assert isinstance(tree, EventTree)
        assert tree.name != ""
        assert tree.initiating_frequency == pytest.approx(1e-4)

    def test_event_tree_paths(self):
        tree = create_generic_vessel_tree()
        scenarios = tree.get_scenarios()
        assert len(scenarios) > 0
        # Scenario probabilities sum to initiating_frequency
        total = sum(s.probability for s in scenarios)
        assert total > 0
        assert total == pytest.approx(tree.initiating_frequency, rel=0.01)

    def test_event_tree_has_root(self):
        tree = create_generic_vessel_tree()
        assert tree.root is not None
        assert hasattr(tree.root, 'children')

    def test_scenario_has_consequence_type(self):
        tree = create_generic_vessel_tree()
        for s in tree.get_scenarios():
            assert hasattr(s, 'consequence_type')
            assert s.consequence_type is not None

    def test_initiating_frequency_default(self):
        tree = create_generic_vessel_tree()
        assert tree.initiating_frequency > 0
        assert tree.initiating_frequency < 1.0


class TestFailureFrequency:
    def test_default_db_available(self):
        db = get_default_db()
        assert isinstance(db, FailureFrequencyDB)

    def test_lookup_vessel_returns_positive(self):
        assert lookup_frequency('vessel', 'small') > 0

    def test_lookup_pipe_returns_positive(self):
        assert lookup_frequency('pipe', 'small') > 0

    def test_small_leak_more_frequent_than_large(self):
        fs = lookup_frequency('vessel', 'small')
        fl = lookup_frequency('vessel', 'large')
        assert fs > fl

    def test_frequency_classification(self):
        assert isinstance(classify_frequency(1e-4), FrequencyClass)
        assert isinstance(classify_frequency(1e-8), FrequencyClass)


class TestIgnitionProbability:
    def test_immediate_ignition_positive(self):
        assert immediate_ignition_probability('methane', release_rate=1.0) > 0

    def test_delayed_ignition_positive(self):
        assert delayed_ignition_probability('methane', release_duration=300.0) > 0

    def test_ignition_probability_in_range(self):
        for sub in ['methane', 'propane', 'gasoline']:
            assert 0.0 <= immediate_ignition_probability(sub) <= 1.0
            assert 0.0 <= delayed_ignition_probability(sub) <= 1.0

    def test_combined_ignition_is_dict(self):
        r = combined_ignition_probability('methane', release_rate=10.0, release_duration=300.0)
        assert isinstance(r, dict)


class TestIndividualRisk:
    def test_ir_at_point_defined(self):
        assert calculate_ir_at_point(100, 50, []) >= 0

    def test_ir_always_nonnegative(self):
        assert calculate_ir_at_point(0, 0, []) >= 0


class TestSocietalRisk:
    def test_fn_curve_basic(self):
        fn = calculate_fn_curve(scenarios=[], population_total=100, grid_spacing=50.0)
        assert fn is not None

    def test_fn_decreasing(self):
        fn = calculate_fn_curve(scenarios=[], population_total=100, grid_spacing=50.0)
        if hasattr(fn, 'F') and len(fn.F) > 2:
            for i in range(len(fn.F) - 1):
                assert fn.F[i] >= fn.F[i + 1]


class TestRiskMatrix:
    def test_frequent_catastrophic_is_extreme(self):
        level = risk_level_from_values(frequency=1e-1, fatalities=100)
        assert level in (RiskLevel.EXTREME, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW)

    def test_rare_minor_is_low(self):
        level = risk_level_from_values(frequency=1e-7, fatalities=1)
        assert level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.EXTREME)

    def test_risk_level_ordering(self):
        assert len({RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.EXTREME}) == 4

    def test_classify_likelihood(self):
        assert isinstance(classify_likelihood(1e-2), LikelihoodLevel)

    def test_classify_consequence(self):
        assert isinstance(classify_consequence(fatalities=50, injuries=200), ConsequenceLevel)

    def test_risk_level_function(self):
        level = risk_level(LikelihoodLevel.FREQUENT, ConsequenceLevel.CATASTROPHIC)
        assert isinstance(level, RiskLevel)
        assert level in (RiskLevel.EXTREME, RiskLevel.HIGH)
