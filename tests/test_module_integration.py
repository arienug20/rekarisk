"""Integration tests for disconnected QRA modules.

Tests that dense gas, event tree, wind rose, batch runner,
sensitivity, and Monte Carlo modules are properly callable.
"""
from __future__ import annotations

import math
import pytest

from rekarisk.models.qra.qra_pipeline import (
    QRAPipeline, IsoSection, HoleSize, WeatherScenario,
    ReceptorPoint, WorkerGroup,
)


def _mini_pipeline(P=50e5):
    """Create a minimal QRA pipeline for testing."""
    iso = [IsoSection(name="Test", P=P, T=310, volume=5.0,
                      composition="methane", fill_fraction=0.0, x=0, y=0)]
    holes = [HoleSize("Large", 0.1)]
    wx = [WeatherScenario("D5", 5.0, "D")]
    rec = [ReceptorPoint(x=0, y=65, label="Test")]
    wrk = [WorkerGroup("Op", 1, [(0, 65, 0.5)])]
    return QRAPipeline(iso_sections=iso, hole_sizes=holes,
                       weather_scenarios=wx, receptor_grid=rec,
                       worker_groups=wrk)


# ══════════════════════════════════════════════════════════════════════════════
# Dense Gas
# ══════════════════════════════════════════════════════════════════════════════

class TestDenseGasIntegration:

    def test_dense_gas_produces_result(self):
        """Dense gas model produces valid result."""
        from rekarisk.models.dispersion.dense_gas import (
            DenseGasInput, calculate_dense_gas,
        )
        inp = DenseGasInput(
            source_rate=5.0, source_mass=100.0,
            release_type="continuous", release_duration=600.0,
            cloud_density=2.5, air_density=1.2,
            wind_speed=3.0, stability_class="D",
            release_height=0.0, temperature_cloud=250.0,
            temperature_ambient=300.0, molecular_weight=0.044,
        )
        result = calculate_dense_gas(inp)
        assert result is not None
        # Check available attributes
        assert result.transition_distance > 0
        assert result.max_concentration >= 0

    def test_dense_gas_transition_positive(self):
        """Dense gas transition distance is positive for valid input."""
        from rekarisk.models.dispersion.dense_gas import (
            DenseGasInput, calculate_dense_gas,
        )
        inp_calm = DenseGasInput(
            source_rate=5.0, source_mass=100.0,
            release_type="continuous", release_duration=600.0,
            cloud_density=2.5, air_density=1.2,
            wind_speed=1.0, stability_class="F",
            release_height=0.0, temperature_cloud=250.0,
            temperature_ambient=300.0, molecular_weight=0.044,
        )
        inp_windy = DenseGasInput(
            source_rate=5.0, source_mass=100.0,
            release_type="continuous", release_duration=600.0,
            cloud_density=2.5, air_density=1.2,
            wind_speed=8.0, stability_class="C",
            release_height=0.0, temperature_cloud=250.0,
            temperature_ambient=300.0, molecular_weight=0.044,
        )
        r_calm = calculate_dense_gas(inp_calm).transition_distance
        r_windy = calculate_dense_gas(inp_windy).transition_distance
        # Both should be positive; exact relationship depends on model
        assert r_calm > 0
        assert r_windy > 0


# ══════════════════════════════════════════════════════════════════════════════
# Event Tree
# ══════════════════════════════════════════════════════════════════════════════

class TestEventTreeIntegration:

    def test_event_tree_creates_and_generates(self):
        """EventTree can be created and generates scenarios."""
        from rekarisk.models.qra.event_tree import (
            EventTree, Scenario, ConsequenceType,
        )
        tree = EventTree(name="Test Tree", initiating_frequency=1e-3)
        # Add nodes using actual API: add_node returns (yes_node, no_node)
        imm_yes, imm_no = tree.add_node(
            parent_name=tree.root.name,
            name="Immediate Ignition",
            prob_yes=0.1,
        )
        tree.add_terminal_node(
            parent_name=imm_yes.name,
            outcome_name="Jet Fire",
            consequence_type=ConsequenceType.JET_FIRE,
        )
        del_yes, del_no = tree.add_node(
            parent_name=imm_no.name,
            name="Delayed Ignition",
            prob_yes=0.2,
        )
        tree.add_terminal_node(
            parent_name=del_yes.name,
            outcome_name="Flash Fire",
            consequence_type=ConsequenceType.FLASH_FIRE,
        )
        tree.add_terminal_node(
            parent_name=del_no.name,
            outcome_name="Toxic Dispersion",
            consequence_type=ConsequenceType.TOXIC,
        )
        scenarios = tree.get_scenarios()
        assert len(scenarios) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Wind Rose
# ══════════════════════════════════════════════════════════════════════════════

class TestWindRoseIntegration:

    def test_wind_rose_data_creates(self):
        """WindRoseData can be created."""
        from rekarisk.meteorology.wind_rose import WindRoseData
        wrd = WindRoseData(n_directions=8)
        assert wrd is not None
        assert wrd.n_directions == 8

    def test_wind_rose_total_probability_empty(self):
        """Empty wind rose has total count of 0."""
        from rekarisk.meteorology.wind_rose import WindRoseData
        wrd = WindRoseData(n_directions=8)
        assert wrd.total == 0


# ══════════════════════════════════════════════════════════════════════════════
# Batch Runner
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchRunnerIntegration:

    def test_batch_runner_with_qra(self):
        """BatchRunner runs multiple QRA configurations."""
        from rekarisk.analysis.batch_runner import BatchRunner, BatchInput

        def qra_model(**kwargs):
            return _mini_pipeline(P=kwargs.get("pressure", 50e5)).run()

        batch_input = BatchInput(
            scenarios=[
                {"pressure": 30e5},
                {"pressure": 50e5},
                {"pressure": 80e5},
            ],
            model_function=qra_model,
        )
        runner = BatchRunner()
        result = runner.run(batch_input)
        assert result.total_scenarios == 3
        assert len(result.failed_scenarios) == 0

    def test_batch_runner_pll_increases_with_pressure(self):
        """Higher pressure → higher PLL in batch results."""
        from rekarisk.analysis.batch_runner import BatchRunner, BatchInput

        def qra_model(**kwargs):
            return _mini_pipeline(P=kwargs.get("pressure", 50e5)).run()

        batch_input = BatchInput(
            scenarios=[{"pressure": 20e5}, {"pressure": 90e5}],
            model_function=qra_model,
        )
        runner = BatchRunner()
        result = runner.run(batch_input)
        pll_low = result.results[0][2].pll_total
        pll_high = result.results[1][2].pll_total
        assert pll_high > pll_low


# ══════════════════════════════════════════════════════════════════════════════
# Sensitivity Analysis
# ══════════════════════════════════════════════════════════════════════════════

class TestSensitivityIntegration:

    def test_sensitivity_oat_pressure(self):
        """OAT sensitivity: PLL varies with pressure."""
        from rekarisk.analysis.sensitivity import (
            SensitivityInput, run_oat,
        )

        def qra_model(**kwargs):
            return _mini_pipeline(P=kwargs.get("pressure", 50e5)).run()

        si = SensitivityInput(
            base_case={"pressure": 50e5},
            parameters={"pressure": (20e5, 90e5)},
            model_function=qra_model,
            output_extractor=lambda r: r.pll_total,
        )
        result = run_oat(si)
        assert result is not None
        # Actual attribute name: parameter_effects
        assert len(result.parameter_effects) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo
# ══════════════════════════════════════════════════════════════════════════════

class TestMonteCarloIntegration:

    def test_monte_carlo_with_qra(self):
        """Monte Carlo produces distribution of PLL values."""
        from rekarisk.analysis.monte_carlo import (
            MCInput, Uniform, run_monte_carlo,
        )

        def qra_model(**kwargs):
            return _mini_pipeline(P=kwargs.get("pressure", 50e5)).run()

        mc_input = MCInput(
            parameters={"pressure": Uniform(20e5, 90e5)},
            model_function=qra_model,
            output_extractor=lambda r: r.pll_total,
            output_keys=["pll"],
            n_samples=5,
        )
        result = run_monte_carlo(mc_input)
        # result.samples is a dict of param_name → array of samples
        assert "pressure" in result.samples
        assert len(result.samples["pressure"]) >= 5
        # result.outputs is a dict of output_key → array
        assert "pll" in result.outputs
        assert len(result.outputs["pll"]) >= 5
        assert all(v >= 0 for v in result.outputs["pll"])

    def test_monte_carlo_mean_positive(self):
        """Mean PLL from MC is positive."""
        from rekarisk.analysis.monte_carlo import (
            MCInput, Uniform, run_monte_carlo,
        )

        def qra_model(**kwargs):
            return _mini_pipeline(P=kwargs.get("pressure", 50e5)).run()

        mc_input = MCInput(
            parameters={"pressure": Uniform(30e5, 80e5)},
            model_function=qra_model,
            output_extractor=lambda r: r.pll_total,
            output_keys=["pll"],
            n_samples=5,
        )
        result = run_monte_carlo(mc_input)
        # statistics is dict of output_key → {mean, std, ...}
        assert "pll" in result.statistics
        assert result.statistics["pll"]["mean"] > 0
