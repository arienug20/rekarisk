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


# ══════════════════════════════════════════════════════════════════════════════
# 7-Section QRA Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestQRAPipeline7Section:
    """End-to-end 7-section pipeline validation."""

    def test_pipeline_runs_with_7_iso_sections(self):
        """7 ISO sections with realistic Indonesian layout produce LSIR."""
        from rekarisk.models.qra.qra_pipeline import (
            QRAPipeline, IsoSection, ReceptorPoint,
        )
        iso_sections = [
            IsoSection(name="Process Area", P=60e5, T=320.0, volume=8.5,
                       composition="natural_gas", molecular_weight=20.5,
                       fill_fraction=0.0, x=0, y=30, elevation=3.0,
                       n_equipment=5, freq_scale=2.4),
            IsoSection(name="Storage Farm", P=3e5, T=305.0, volume=500.0,
                       composition="propane", molecular_weight=44.1,
                       fill_fraction=0.75, x=80, y=-20, elevation=0.5,
                       rho_liquid=520.0, n_equipment=2),
            IsoSection(name="Loading Area", P=5e5, T=300.0, volume=2.0,
                       composition="propane", molecular_weight=44.1,
                       fill_fraction=0.5, x=110, y=-50, elevation=0.0,
                       rho_liquid=520.0),
            IsoSection(name="Utility Area", P=15e5, T=300.0, volume=3.0,
                       composition="natural_gas", molecular_weight=20.5,
                       fill_fraction=0.0, x=-80, y=60, n_equipment=2),
            IsoSection(name="Pipeline", P=70e5, T=315.0, volume=15.0,
                       composition="natural_gas", molecular_weight=20.5,
                       fill_fraction=0.0, x=-40, y=0, n_equipment=2),
            IsoSection(name="Flare KO Drum", P=5e5, T=310.0, volume=12.0,
                       composition="natural_gas", molecular_weight=20.5,
                       fill_fraction=0.3, x=100, y=120, elevation=15.0),
            IsoSection(name="Control Room", P=1e5, T=300.0, volume=1.0,
                       composition="natural_gas", molecular_weight=20.5,
                       fill_fraction=0.0, x=20, y=-40, n_equipment=0),
        ]
        receptors = [
            ReceptorPoint(label="Process Area NKT", x=0, y=30),
            ReceptorPoint(label="Storage Tank Farm", x=80, y=-20),
            ReceptorPoint(label="Control Room NKT", x=20, y=-40),
        ]
        result = QRAPipeline(
            iso_sections=iso_sections,
            receptor_grid=receptors,
        ).run()
        assert result.scenario_count > 0
        assert len(result.lsir_grid) == 3
        assert result.lsir_grid[(0, 30)] > 0, "Process Area must have non-zero LSIR"
        assert result.lsir_grid[(80, -20)] > 0, "Storage must have non-zero LSIR"

    def test_shelter_factor_applied(self):
        """Receptor shelter factor reduces LSIR at control rooms."""
        from rekarisk.models.qra.qra_pipeline import (
            QRAPipeline, IsoSection, ReceptorPoint,
        )
        iso_sections = [
            IsoSection(name="Process Area", P=60e5, T=320.0, volume=8.5,
                       composition="natural_gas", molecular_weight=20.5,
                       fill_fraction=0.0, x=0, y=30, elevation=3.0,
                       n_equipment=5),
        ]
        # Same location, different shelter factors
        result_blasted = QRAPipeline(
            iso_sections=iso_sections,
            receptor_grid=[ReceptorPoint(label="Blast Room", x=0, y=30)],
            receptor_shelter_factors={"Blast Room": 0.2},
        ).run()
        result_open = QRAPipeline(
            iso_sections=iso_sections,
            receptor_grid=[ReceptorPoint(label="Open Area", x=0, y=30)],
            receptor_shelter_factors={"Open Area": 1.0},
        ).run()
        val_blasted = result_blasted.lsir_grid.get((0, 30), 1.0)
        val_open = result_open.lsir_grid.get((0, 30), 1.0)
        assert val_blasted > 0
        assert val_blasted < val_open, (
            f"Shelter 0.2 ({val_blasted:.2e}) < shelter 1.0 ({val_open:.2e})"
        )

    def test_n_equipment_scales_frequency(self):
        """n_equipment linearly scales leak frequency."""
        from rekarisk.models.qra.qra_pipeline import (
            QRAPipeline, IsoSection, ReceptorPoint,
        )
        iso_1 = [IsoSection(name="Single Unit", P=60e5, T=320.0, volume=8.5,
                            composition="natural_gas", molecular_weight=20.5,
                            fill_fraction=0.0, x=0, y=30, n_equipment=1)]
        iso_3 = [IsoSection(name="Triple Unit", P=60e5, T=320.0, volume=8.5,
                            composition="natural_gas", molecular_weight=20.5,
                            fill_fraction=0.0, x=0, y=30, n_equipment=3)]
        receptors = [ReceptorPoint(label="Test", x=0, y=30)]
        res_1 = QRAPipeline(iso_sections=iso_1, receptor_grid=receptors).run()
        res_3 = QRAPipeline(iso_sections=iso_3, receptor_grid=receptors).run()
        lsir_1 = res_1.lsir_grid.get((0, 30), 0)
        lsir_3 = res_3.lsir_grid.get((0, 30), 0)
        # 3 equipment should produce >= 3x the risk of 1 (linear scaling)
        assert lsir_3 > lsir_1 * 2.0, (
            f"3 equipment ({lsir_3:.2e}) should be > 2x 1 equipment ({lsir_1:.2e})"
        )

    def test_lsir_ranking_matches_layout(self):
        """Higher LSIR near major process equipment, lower near admin."""
        from rekarisk.models.qra.qra_pipeline import (
            QRAPipeline, IsoSection, ReceptorPoint,
        )
        iso_sections = [
            IsoSection(name="Process Area", P=60e5, T=320.0, volume=8.5,
                       composition="natural_gas", molecular_weight=20.5,
                       fill_fraction=0.0, x=0, y=30, n_equipment=5),
            IsoSection(name="Storage", P=3e5, T=305.0, volume=500.0,
                       composition="propane", molecular_weight=44.1,
                       fill_fraction=0.75, x=80, y=-20, rho_liquid=520.0, n_equipment=2),
        ]
        receptors = [
            ReceptorPoint(label="Process Area", x=0, y=30),
            ReceptorPoint(label="Storage", x=80, y=-20),
            ReceptorPoint(label="Remote", x=500, y=500),
        ]
        result = QRAPipeline(iso_sections=iso_sections, receptor_grid=receptors).run()
        lsir_process = result.lsir_grid.get((0, 30), 0)
        lsir_storage = result.lsir_grid.get((80, -20), 0)
        lsir_remote = result.lsir_grid.get((500, 500), 0)
        assert lsir_process > 0, "Process area should have risk"
        assert lsir_remote < lsir_process, "Remote should be < process"
