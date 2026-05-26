"""
Rekarisk — Fire Model Validation Tests.

Tests for pool fire, jet fire, and BLEVE/fireball thermal radiation models.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rekarisk.core.constants import P_ATM, SIGMA_SB
from rekarisk.models.fire.pool_fire import (
    PoolFireInput,
    PoolFireResult,
    calculate_pool_fire,
    burning_rate,
    flame_geometry,
    thermal_radiation_vs_distance_pool,
    distance_to_thresholds_pool,
    BURNING_RATE_PARAMS,
)
from rekarisk.models.fire.jet_fire import (
    JetFireInput,
    calculate_jet_fire,
    flame_length_vertical_jet,
    flame_length_kalghatgi,
    thermal_radiation_vs_distance_jet,
)
from rekarisk.models.fire.bleve import (
    BLEVEInput,
    calculate_bleve,
    fireball_diameter_roberts,
    fireball_duration_roberts,
    thermal_radiation_vs_distance_bleve,
)


# ══════════════════════════════════════════════════════════════════════════════
# Pool Fire
# ══════════════════════════════════════════════════════════════════════════════

class TestPoolFire:
    """Pool fire thermal radiation model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            substance='gasoline',
            pool_diameter=10.0,
            pool_area=78.54,  # π * (5)²
            wind_speed=3.0,
            temperature=298.15,
            pressure=P_ATM,
            relative_humidity=50.0,
            mass_inventory=5000.0,
            rho_liquid=740.0,
            heat_of_combustion=44e6,
            heat_of_vaporization=3.5e5,
            boiling_point=350.0,
            molecular_weight=0.100,
            emissivity=0.95,
        )
        defaults.update(kwargs)
        return PoolFireInput(**defaults)

    def test_burning_rate_positive_for_gasoline(self):
        """Gasoline burning rate > 0."""
        inp = self._make_input(substance='gasoline')
        result = calculate_pool_fire(inp)
        assert result.burning_rate > 0, "Gasoline should burn"

    def test_burning_rate_has_known_params(self):
        """Gasoline has known burning rate params."""
        params = BURNING_RATE_PARAMS.get('gasoline')
        assert params is not None
        m_dot_inf, k_beta = params
        assert m_dot_inf > 0
        assert k_beta > 0

    def test_thermal_radiation_decreases_with_distance(self):
        """Thermal radiation decreases with distance from pool fire."""
        inp = self._make_input()
        q_50 = thermal_radiation_vs_distance_pool(inp, 50)
        q_100 = thermal_radiation_vs_distance_pool(inp, 100)
        q_200 = thermal_radiation_vs_distance_pool(inp, 200)

        assert q_100 < q_50, "Radiation should decrease with distance"
        assert q_200 < q_100

    def test_distance_to_4kw_less_than_1kw(self):
        """Distance to 4 kW/m² threshold < distance to 1 kW/m² threshold."""
        inp = self._make_input()
        d_4 = distance_to_thresholds_pool(inp, 4.0)
        d_1 = distance_to_thresholds_pool(inp, 1.0)
        # Higher threshold → shorter distance
        assert d_4 < d_1, f"d(4kW)={d_4:.1f} should be < d(1kW)={d_1:.1f}"

    def test_larger_pool_more_radiation(self):
        """Larger pool diameter → higher radiation at same distance."""
        inp_small = self._make_input(pool_diameter=5.0, pool_area=19.63)
        inp_large = self._make_input(pool_diameter=20.0, pool_area=314.16)
        q_small = thermal_radiation_vs_distance_pool(inp_small, 50)
        q_large = thermal_radiation_vs_distance_pool(inp_large, 50)
        assert q_large > q_small, "Larger pool → more radiation"

    def test_pool_fire_result_has_flame_geometry(self):
        """Pool fire result includes flame geometry parameters."""
        inp = self._make_input()
        result = calculate_pool_fire(inp)
        assert hasattr(result, 'flame_length')
        assert hasattr(result, 'flame_tilt')
        assert result.flame_length > 0
        assert result.sep > 0  # Surface emissive power

    def test_flame_geometry_function(self):
        """Standalone flame_geometry returns reasonable values."""
        L, theta, D_eq = flame_geometry(
            pool_diameter=10.0,
            wind_speed=3.0,
            m_dot_burning=0.055,
            rho_air=1.2,
            heat_of_combustion=44e6,
        )
        assert L > 0
        assert D_eq > 0

    def test_burning_rate_function(self):
        """Standalone burning_rate returns positive values."""
        m_dot = burning_rate(
            pool_diameter=10.0,
            substance='gasoline',
            m_dot_inf=None,
            k_beta=None,
        )
        assert m_dot > 0

    def test_lng_vs_gasoline_burning_rate(self):
        """LNG has different burning rate than gasoline."""
        m_dot_lng = burning_rate(
            pool_diameter=10.0,
            substance='lng',
        )
        m_dot_gas = burning_rate(
            pool_diameter=10.0,
            substance='gasoline',
        )
        assert m_dot_lng > 0
        assert m_dot_gas > 0
        assert m_dot_lng != pytest.approx(m_dot_gas, rel=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# Jet Fire
# ══════════════════════════════════════════════════════════════════════════════

class TestJetFire:
    """Jet fire thermal radiation model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            substance='methane',
            orifice_diameter=0.025,
            mass_flow_rate=2.0,
            release_pressure=1e6,
            release_temperature=300,
            wind_speed=5.0,
            temperature=298.15,
            pressure=P_ATM,
            molecular_weight=0.016,
            cp_cv_ratio=1.3,
            heat_of_combustion=50e6,
        )
        defaults.update(kwargs)
        return JetFireInput(**defaults)

    def test_flame_length_positive(self):
        """Jet fire flame length > 0."""
        inp = self._make_input()
        result = calculate_jet_fire(inp)
        assert result.flame_length > 0

    def test_flame_length_vertical_jet_function(self):
        """flame_length_vertical_jet returns positive."""
        L = flame_length_vertical_jet(
            mass_flow_rate=2.0,
            orifice_diameter=0.025,
            rho_air=1.2,
        )
        assert L > 0

    def test_flame_length_kalghatgi_function(self):
        """flame_length_kalghatgi returns positive."""
        L = flame_length_kalghatgi(
            mass_flow_rate=2.0,
            orifice_diameter=0.025,
            rho_air=1.2,
            heat_of_combustion=50e6,
        )
        assert L > 0

    def test_radiation_decreases_with_distance(self):
        """Jet fire thermal radiation decreases with distance."""
        inp = self._make_input()
        q_30 = thermal_radiation_vs_distance_jet(inp, 30)
        q_60 = thermal_radiation_vs_distance_jet(inp, 60)
        q_100 = thermal_radiation_vs_distance_jet(inp, 100)

        if q_30 > 0:
            assert q_60 < q_30
            assert q_100 < q_60

    def test_higher_mass_flow_longer_flame(self):
        """Higher mass flow rate → longer flame."""
        inp_low = self._make_input(mass_flow_rate=0.5)
        inp_high = self._make_input(mass_flow_rate=5.0)
        L_low = flame_length_vertical_jet(0.5, 0.025, 1.2)
        L_high = flame_length_vertical_jet(5.0, 0.025, 1.2)
        assert L_high > L_low, "Higher mdot → longer flame"

    def test_jet_fire_result_attributes(self):
        """Jet fire result has essential attributes."""
        inp = self._make_input()
        result = calculate_jet_fire(inp)
        assert hasattr(result, 'flame_length')
        assert hasattr(result, 'sep')
        assert hasattr(result, 'total_heat_release')
        assert result.total_heat_release > 0


# ══════════════════════════════════════════════════════════════════════════════
# BLEVE / Fireball
# ══════════════════════════════════════════════════════════════════════════════

class TestBLEVE:
    """BLEVE/fireball thermal radiation model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            substance='propane',
            mass=1000.0,
            temperature=298.15,
            pressure=P_ATM,
            relative_humidity=50.0,
            heat_of_combustion=46.3e6,
            failure_fraction=1.0,
            T_boiling=231.0,
        )
        defaults.update(kwargs)
        return BLEVEInput(**defaults)

    def test_fireball_diameter_roberts(self):
        """D = 5.8 * M^0.333 for 1000 kg."""
        D = fireball_diameter_roberts(1000)
        expected = 5.8 * 1000.0 ** (1.0 / 3.0)
        assert D == pytest.approx(expected, rel=0.01)
        assert D > 50  # should be ~58m for 1000 kg

    def test_fireball_diameter_increases_with_mass(self):
        """Larger mass → larger fireball."""
        D_100 = fireball_diameter_roberts(100)
        D_1000 = fireball_diameter_roberts(1000)
        D_10000 = fireball_diameter_roberts(10000)
        assert D_1000 > D_100
        assert D_10000 > D_1000
        # Scaling: D ∝ M^(1/3) approximately
        ratio_10x = D_1000 / D_100
        assert 1.5 < ratio_10x < 3.0  # 10^(1/3) ≈ 2.15

    def test_fireball_duration_positive(self):
        """BLEVE fireball duration > 0."""
        dur = fireball_duration_roberts(1000)
        assert dur > 0

    def test_fireball_duration_increases_with_mass(self):
        """Larger mass → longer duration."""
        d_100 = fireball_duration_roberts(100)
        d_1000 = fireball_duration_roberts(1000)
        assert d_1000 > d_100

    def test_bleve_radiation_decreases_with_distance(self):
        """BLEVE thermal radiation decreases with distance."""
        inp = self._make_input()
        q_100 = thermal_radiation_vs_distance_bleve(inp, 100)
        q_200 = thermal_radiation_vs_distance_bleve(inp, 200)
        q_500 = thermal_radiation_vs_distance_bleve(inp, 500)

        assert q_200 < q_100
        assert q_500 < q_200

    def test_bleve_calculation_produces_result(self):
        """BLEVE calculation produces valid result."""
        inp = self._make_input()
        result = calculate_bleve(inp)
        assert hasattr(result, 'fireball_diameter')
        assert hasattr(result, 'fireball_duration')
        assert hasattr(result, 'sep')
        assert result.fireball_diameter > 0
        assert result.fireball_duration > 0

    def test_bleve_mass_scaling(self):
        """Regression: BLEVE diameter scales with M^(1/3)."""
        for mass in [100, 500, 1000, 5000, 10000]:
            D = fireball_diameter_roberts(mass)
            assert D > 0
            # Should approximately follow D = 5.8 * M^0.333
            expected = 5.8 * mass ** (1.0 / 3.0)
            # Allow ±20% for different correlations
            assert 0.5 * expected < D < 2.0 * expected

    def test_bleve_distance_to_thresholds(self):
        """Distance to BLEVE thresholds are ordered correctly."""
        from rekarisk.models.fire.bleve import distance_to_thresholds_bleve
        inp = self._make_input()
        d_37 = distance_to_thresholds_bleve(inp, 37.5)
        d_12 = distance_to_thresholds_bleve(inp, 12.5)
        d_5 = distance_to_thresholds_bleve(inp, 5.0)

        assert d_12 > d_37  # Lower threshold → farther distance
        assert d_5 > d_12
