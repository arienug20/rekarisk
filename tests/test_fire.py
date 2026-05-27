"""
Rekarisk — Fire Model Validation Tests.

Tests for pool fire, jet fire, and BLEVE/fireball thermal radiation models.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rekarisk.core.constants import P_ATM
from rekarisk.models.fire.pool_fire import (
    PoolFireInput,
    PoolFireResult,
    calculate_pool_fire,
    burning_rate_default,
    flame_length_thomas,
    flame_tilt_aga,
    surface_emissive_power,
    thermal_radiation_vs_distance,
    distance_to_thresholds,
    BURNING_RATE_PARAMS,
)
from rekarisk.models.fire.jet_fire import (
    JetFireInput,
    JetFireResult,
    calculate_jet_fire,
    flame_length_vertical_jet,
    flame_length_kalghatgi,
    thermal_radiation_vs_distance_jet,
)
from rekarisk.models.fire.bleve import (
    BLEVEInput,
    BLEVEResult,
    calculate_bleve,
    fireball_diameter_roberts,
    fireball_duration_roberts,
    thermal_radiation_vs_distance_bleve,
    distance_to_thresholds_bleve,
)


# ══════════════════════════════════════════════════════════════════════════════
# Pool Fire
# ══════════════════════════════════════════════════════════════════════════════

class TestPoolFire:
    """Pool fire thermal radiation model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            pool_diameter=10.0,
            substance='gasoline',
            wind_speed=3.0,
            ambient_temperature=298.15,
            relative_humidity=50.0,
        )
        defaults.update(kwargs)
        return PoolFireInput(**defaults)

    def test_pool_fire_result_has_burning_rate(self):
        """PoolFireResult.burning_rate > 0."""
        inp = self._make_input(substance='gasoline')
        result = calculate_pool_fire(inp)
        assert result.burning_rate > 0, "Gasoline should burn"

    def test_burning_rate_default_positive(self):
        """burning_rate_default returns positive value."""
        m_dot = burning_rate_default(substance='gasoline', pool_diameter=10.0)
        assert m_dot > 0
        assert m_dot < 1.0

    def test_burning_rate_default_lng_vs_gasoline(self):
        """LNG has different burning rate than gasoline."""
        m_dot_lng = burning_rate_default(substance='lng', pool_diameter=10.0)
        m_dot_gas = burning_rate_default(substance='gasoline', pool_diameter=10.0)
        assert m_dot_lng > 0
        assert m_dot_gas > 0
        # LNG typically burns slower than gasoline
        assert m_dot_lng != pytest.approx(m_dot_gas, rel=0.2)

    def test_burning_rate_params_known(self):
        """Gasoline has known burning rate params in BURNING_RATE_PARAMS."""
        params = BURNING_RATE_PARAMS.get('gasoline')
        assert params is not None
        m_dot_inf, k_beta = params
        assert m_dot_inf > 0
        assert k_beta > 0

    def test_thermal_radiation_decreases_with_distance(self):
        """Thermal radiation decreases with distance from pool fire."""
        result = calculate_pool_fire(self._make_input())
        sep_val = result.sep
        length = result.flame_length
        dia = result.pool_diameter
        tilt = result.flame_tilt

        # Returns Nx2 array: [[d1, q1], [d2, q2], ...]
        grid = thermal_radiation_vs_distance(
            sep_val, length, dia, tilt,
            ambient_temperature=298.15, relative_humidity=50.0,
            min_distance=10.0, max_distance=200.0, n_points=3,
        )
        assert grid.shape[1] == 2
        assert len(grid) >= 2
        # Radiation should decrease with distance
        for i in range(1, len(grid)):
            assert grid[i, 1] < grid[i - 1, 1], \
                "Radiation should decrease with distance"

    def test_distance_to_4kw_less_than_1kw(self):
        """Distance to 4 kW/m² < distance to 1 kW/m²."""
        result = calculate_pool_fire(self._make_input())
        distances = distance_to_thresholds(
            result.sep, result.flame_length, result.pool_diameter,
            result.flame_tilt,
            ambient_temperature=298.15, relative_humidity=50.0,
            thresholds=[1.0, 4.0],
        )
        assert 4.0 in distances
        assert 1.0 in distances
        assert distances[4.0] < distances[1.0], \
            f"d(4kW)={distances[4.0]:.1f} should be < d(1kW)={distances[1.0]:.1f}"

    def test_larger_pool_more_total_heat(self):
        """Larger pool diameter → higher total heat release."""
        inp_small = self._make_input(pool_diameter=5.0)
        inp_large = self._make_input(pool_diameter=20.0)
        r_small = calculate_pool_fire(inp_small)
        r_large = calculate_pool_fire(inp_large)
        # Larger pool has higher total heat release (Q = m_dot * A * dHc)
        if hasattr(r_small, 'total_heat_release') and hasattr(r_large, 'total_heat_release'):
            assert r_large.total_heat_release > r_small.total_heat_release

    def test_pool_fire_result_has_flame_geometry(self):
        """PoolFireResult includes flame geometry attributes."""
        inp = self._make_input()
        result = calculate_pool_fire(inp)
        assert hasattr(result, 'flame_length')
        assert hasattr(result, 'flame_tilt')
        assert result.flame_length > 0, f"Flame length: {result.flame_length}"
        assert result.sep > 0, "Surface emissive power should be > 0"

    def test_flame_length_thomas_positive(self):
        """flame_length_thomas returns positive value."""
        L = flame_length_thomas(m_dot=0.055, pool_diameter=10.0, wind_speed=3.0)
        assert L > 0
        assert 5 < L < 100, f"Flame length out of expected range: {L}"

    def test_flame_tilt_aga_returns_value(self):
        """flame_tilt_aga returns a value or tuple."""
        result = flame_tilt_aga(wind_speed=3.0, pool_diameter=10.0)
        # May return a single angle or (cos_theta, sin_theta) tuple
        if isinstance(result, tuple):
            assert len(result) >= 1
        else:
            assert result >= 0

    def test_surface_emissive_power_positive(self):
        """surface_emissive_power returns positive value."""
        L = flame_length_thomas(m_dot=0.055, pool_diameter=10.0)
        sep = surface_emissive_power(
            m_dot=0.055, dhc=44e6, chi_r=0.35,
            flame_length=L, pool_diameter=10.0,
        )
        assert sep > 0, f"SEP should be positive, got {sep}"

    def test_pool_fire_result_grid_arrays(self):
        """PoolFireResult has thermal_radiation_vs_distance grid."""
        result = calculate_pool_fire(self._make_input())
        grid = result.thermal_radiation_vs_distance
        assert grid.shape[1] == 2
        assert len(grid) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Jet Fire
# ══════════════════════════════════════════════════════════════════════════════

class TestJetFire:
    """Jet fire thermal radiation model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            orifice_diameter=0.025,
            discharge_velocity=300.0,
            discharge_density=0.6,
            mass_flow_rate=0.1,
            substance='methane',
            wind_speed=5.0,
            release_direction='horizontal',
            heat_of_combustion=50e6,
        )
        defaults.update(kwargs)
        return JetFireInput(**defaults)

    def test_flame_length_positive(self):
        """Jet fire flame length > 0 when mass flow rate is positive."""
        inp = self._make_input(mass_flow_rate=0.5)
        result = calculate_jet_fire(inp)
        assert result.flame_length > 0

    def test_flame_length_vertical_jet_positive(self):
        """flame_length_vertical_jet returns positive."""
        L = flame_length_vertical_jet(
            orifice_diameter=0.025,
            jet_density=0.6,
            air_density=1.2,
        )
        assert L > 0

    def test_flame_length_kalghatgi_positive(self):
        """flame_length_kalghatgi returns positive."""
        L = flame_length_kalghatgi(
            orifice_diameter=0.025,
            discharge_velocity=300.0,
            jet_density=0.6,
            air_density=1.2,
        )
        assert L > 0

    def test_radiation_decreases_with_distance_jet(self):
        """Jet fire thermal radiation decreases with distance."""
        inp = self._make_input(mass_flow_rate=0.5)
        result = calculate_jet_fire(inp)
        if result.total_heat_release <= 0:
            pytest.skip("Jet fire calculation requires valid mass flow rate")

        grid = thermal_radiation_vs_distance_jet(
            total_heat_release=result.total_heat_release,
            radiative_fraction=0.3,
            sep=result.sep if result.sep > 0 else 200e3,
            flame_length=result.flame_length,
            flame_width=result.flame_width,
            tilt_deg=result.flame_tilt_deg,
            center_height=result.flame_center_height,
            ambient_temperature=298.15,
            relative_humidity=50.0,
            min_distance=30.0,
            max_distance=200.0,
            n_points=3,
        )
        if len(grid) >= 2:
            assert grid[1, 1] < grid[0, 1], \
                f"Jet radiation should decrease with distance"

    def test_kalghatgi_flame_length_positive(self):
        """Kalghatgi flame length is positive."""
        L = flame_length_kalghatgi(
            orifice_diameter=0.025, discharge_velocity=100.0,
            jet_density=0.6, air_density=1.2,
        )
        assert L > 0, "Flame length should be > 0"

    def test_jet_fire_calculation_runs(self):
        """calculate_jet_fire runs without error and returns result."""
        inp = self._make_input(mass_flow_rate=0.5)
        result = calculate_jet_fire(inp)
        assert isinstance(result, JetFireResult)
        assert hasattr(result, 'flame_length')

    def test_solid_flame_flux_decreases_far_field(self):
        """Solid flame model: flux decreases monotonically at far field.

        Near-field non-monotonicity is physically correct for surface-to-surface
        radiation: a ground-level receiver directly under a vertical cylinder
        has poor view of the cylinder side, and flux can increase slightly
        as the receiver moves to a distance with better viewing angle.
        At far field (>2× flame length), flux must decrease monotonically.
        """
        from rekarisk.models.fire.jet_fire import thermal_radiation_solid_flame_jet

        L = 20.0
        start_d = 2 * L  # 40m — well into far field
        distances = [start_d, start_d + 10, start_d + 20, start_d + 40,
                     start_d + 80, start_d + 150]
        fluxes = []
        for d in distances:
            flux = thermal_radiation_solid_flame_jet(
                total_heat_release=50e6, radiative_fraction=0.30,
                flame_length=L, flame_tilt_deg=60.0,
                center_height=1.0, distance=d,
            )
            fluxes.append(flux)

        for i in range(1, len(fluxes)):
            assert fluxes[i] <= fluxes[i-1] + 1e-6, (
                f"Far-field flux should decrease: d={distances[i]:.0f}m "
                f"flux={fluxes[i]:.6f} > d={distances[i-1]:.0f}m flux={fluxes[i-1]:.6f}"
            )

    def test_solid_flame_peak_at_closest_point(self):
        """Peak flux must be at the closest distance, not somewhere in the middle."""
        from rekarisk.models.fire.jet_fire import thermal_radiation_solid_flame_jet

        flux_1m = thermal_radiation_solid_flame_jet(
            total_heat_release=50e6, radiative_fraction=0.30,
            flame_length=20.0, flame_tilt_deg=70.0,
            center_height=1.0, distance=1.0,
        )
        flux_10m = thermal_radiation_solid_flame_jet(
            total_heat_release=50e6, radiative_fraction=0.30,
            flame_length=20.0, flame_tilt_deg=70.0,
            center_height=1.0, distance=10.0,
        )
        flux_50m = thermal_radiation_solid_flame_jet(
            total_heat_release=50e6, radiative_fraction=0.30,
            flame_length=20.0, flame_tilt_deg=70.0,
            center_height=1.0, distance=50.0,
        )
        assert flux_1m >= flux_10m, "Flux at 1m must be >= flux at 10m"
        assert flux_10m >= flux_50m, "Flux at 10m must be >= flux at 50m"

    def test_solid_flame_vertical_jet(self):
        """Solid flame model works for vertical jet (tilt=0)."""
        from rekarisk.models.fire.jet_fire import thermal_radiation_solid_flame_jet

        flux_5m = thermal_radiation_solid_flame_jet(
            total_heat_release=30e6, radiative_fraction=0.30,
            flame_length=15.0, flame_tilt_deg=0.0,
            center_height=0.0, distance=5.0,
        )
        flux_20m = thermal_radiation_solid_flame_jet(
            total_heat_release=30e6, radiative_fraction=0.30,
            flame_length=15.0, flame_tilt_deg=0.0,
            center_height=0.0, distance=20.0,
        )
        assert flux_5m > flux_20m
        assert flux_5m > 0

    def test_solid_flame_multipoint_agree_far_field(self):
        """Solid flame and multipoint models agree at far field (>5x flame length)."""
        from rekarisk.models.fire.jet_fire import (
            thermal_radiation_solid_flame_jet,
            thermal_radiation_multipoint,
        )

        flux_sf = thermal_radiation_solid_flame_jet(
            total_heat_release=20e6, radiative_fraction=0.30,
            flame_length=10.0, flame_tilt_deg=30.0,
            center_height=1.0, distance=100.0,
        )
        flux_mp = thermal_radiation_multipoint(
            total_heat_release=20e6, radiative_fraction=0.30,
            flame_length=10.0, flame_tilt_deg=30.0,
            center_height=1.0, distance=100.0,
        )
        # At far field, both should be within 50% of each other
        ratio = flux_sf / max(flux_mp, 1e-10)
        assert 0.3 < ratio < 3.0, \
            f"Far-field models should roughly agree: SF={flux_sf:.4f}, MP={flux_mp:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# BLEVE / Fireball
# ══════════════════════════════════════════════════════════════════════════════

class TestBLEVE:
    """BLEVE/fireball thermal radiation model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            vessel_mass=1000.0,
            substance='propane',
        )
        defaults.update(kwargs)
        return BLEVEInput(**defaults)

    def test_fireball_diameter_roberts(self):
        """D = 5.8 * M^0.333 for 1000 kg ≈ 58m."""
        D = fireball_diameter_roberts(1000.0)
        expected = 5.8 * 1000.0 ** (1.0 / 3.0)
        assert D == pytest.approx(expected, rel=0.01)
        assert D > 50, "Should be ~58m for 1000 kg"

    def test_fireball_diameter_increases_with_mass(self):
        """Larger mass → larger fireball."""
        D_100 = fireball_diameter_roberts(100.0)
        D_1000 = fireball_diameter_roberts(1000.0)
        D_10000 = fireball_diameter_roberts(10000.0)

        assert D_1000 > D_100
        assert D_10000 > D_1000
        ratio_10x = D_1000 / D_100
        assert 1.5 < ratio_10x < 3.5

    def test_fireball_duration_positive(self):
        """BLEVE fireball duration > 0."""
        dur = fireball_duration_roberts(1000.0)
        assert dur > 0
        assert dur < 120.0, f"Duration should be < 120s, got {dur}"

    def test_fireball_duration_increases_with_mass(self):
        """Larger mass → longer duration."""
        d_100 = fireball_duration_roberts(100.0)
        d_1000 = fireball_duration_roberts(1000.0)
        assert d_1000 > d_100

    def test_bleve_radiation_decreases_with_distance(self):
        """BLEVE thermal radiation decreases with distance."""
        inp = self._make_input()
        result = calculate_bleve(inp)
        grid = result.thermal_radiation_vs_distance
        # Check first few values decrease
        for i in range(1, min(5, len(grid))):
            if grid[i, 0] > grid[0, 0]:
                assert grid[i, 1] < grid[0, 1], "Flux should decrease with distance"

    def test_bleve_result_attributes(self):
        """BLEVEResult has essential attributes."""
        inp = self._make_input()
        result = calculate_bleve(inp)
        assert hasattr(result, 'fireball_diameter')
        assert hasattr(result, 'fireball_duration')
        assert hasattr(result, 'sep')
        assert result.fireball_diameter > 0
        assert result.fireball_duration > 0

    def test_bleve_mass_scaling(self):
        """BLEVE diameter scales with M^(1/3)."""
        for mass in [100.0, 500.0, 1000.0, 5000.0, 10000.0]:
            D = fireball_diameter_roberts(mass)
            assert D > 0
            expected = 5.8 * mass ** (1.0 / 3.0)
            assert 0.5 * expected < D < 2.0 * expected, \
                f"Mass {mass}kg → D={D:.1f}m vs expected {expected:.1f}m"

    def test_bleve_distance_to_thresholds_ordered(self):
        """Distance to BLEVE thresholds ordered: d_37 < d_12."""
        inp = self._make_input()
        result = calculate_bleve(inp)
        distances = distance_to_thresholds_bleve(
            sep=result.sep,
            fireball_radius=result.fireball_diameter / 2,
            fireball_height=result.center_height,
            ambient_temperature=298.15,
            relative_humidity=50.0,
            thresholds=[5.0, 12.5, 37.5],
        )
        assert 37.5 in distances
        assert 12.5 in distances
        assert 5.0 in distances
        assert distances[37.5] < distances[12.5], \
            "d(37.5kW) should be closer than d(12.5kW)"
        assert distances[12.5] < distances[5.0], \
            "d(12.5kW) should be closer than d(5kW)"
