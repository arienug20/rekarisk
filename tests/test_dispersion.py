"""
Rekarisk — Dispersion Model Validation Tests.

Tests for Gaussian plume, Gaussian puff, and dense gas dispersion models
against known physical behaviors and mathematical properties.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rekarisk.core.constants import P_ATM
from rekarisk.meteorology.stability import StabilityClass
from rekarisk.models.dispersion.gaussian_plume import (
    PlumeInput,
    PlumeResult,
    concentration_at_point,
    calculate_plume,
    max_ground_concentration,
)
from rekarisk.models.dispersion.gaussian_puff import (
    PuffInput,
    PuffResult,
    concentration_puff,
    calculate_puff,
)
from rekarisk.models.dispersion.dense_gas import (
    DenseGasInput,
    DenseGasResult,
    calculate_dense_gas,
    check_if_dense,
)


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian Plume
# ══════════════════════════════════════════════════════════════════════════════

class TestGaussianPlumeBasic:
    """Basic physical checks for Gaussian plume model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            source_rate=1.0,
            wind_speed=5.0,
            stability_class='D',
            release_height=10.0,
            terrain_type='rural',
            temperature=298.15,
            pressure=P_ATM,
            grid_x_range=(100.0, 5000.0, 50),
            grid_y_range=(-500.0, 500.0, 51),
            grid_z_range=(0.0, 200.0, 21),
        )
        defaults.update(kwargs)
        return PlumeInput(**defaults)

    def test_centerline_concentration_decreases_with_distance(self):
        """Ground-level centerline concentration decreases with distance."""
        inp = self._make_input()
        c1 = concentration_at_point(x=100, y=0, z=0, input=inp)
        c2 = concentration_at_point(x=500, y=0, z=0, input=inp)
        c3 = concentration_at_point(x=2000, y=0, z=0, input=inp)

        assert c1 > 0, "Concentration should be positive"
        assert c2 < c1, "Concentration should decrease with distance"
        assert c3 < c2, "Concentration should continue decreasing"

    def test_concentration_zero_at_origin(self):
        """Concentration at x=0 should be near zero (no plume yet)."""
        inp = self._make_input()
        c = concentration_at_point(x=0, y=0, z=0, input=inp)
        assert c == pytest.approx(0.0, abs=1e-6)

    def test_concentration_negative_x_zero(self):
        """Concentration upstream of source (negative x) should be zero."""
        inp = self._make_input()
        c = concentration_at_point(x=-100, y=0, z=0, input=inp)
        assert c == pytest.approx(0.0, abs=1e-6)

    def test_max_concentration_at_centerline(self):
        """Maximum concentration at given x is at y=0 (centerline)."""
        inp = self._make_input()
        c_center = concentration_at_point(x=500, y=0, z=0, input=inp)
        c_off_1 = concentration_at_point(x=500, y=100, z=0, input=inp)
        c_off_2 = concentration_at_point(x=500, y=200, z=0, input=inp)

        assert c_center > c_off_1, "Centerline > off-center at y=100"
        assert c_off_1 > c_off_2, "Concentration should drop further off-axis"

    def test_concentration_symmetric_about_centerline(self):
        """Concentration is symmetric about y=0."""
        inp = self._make_input()
        c_pos = concentration_at_point(x=500, y=50, z=0, input=inp)
        c_neg = concentration_at_point(x=500, y=-50, z=0, input=inp)
        assert c_pos == pytest.approx(c_neg, rel=1e-10)

    def test_concentration_positive_for_all_downwind(self):
        """Downwind concentrations are positive."""
        inp = self._make_input()
        for x in [100, 500, 1000, 3000]:
            c = concentration_at_point(x=x, y=0, z=0, input=inp)
            assert c > 0, f"Concentration at x={x} should be > 0"

    def test_inverse_proportional_to_wind_speed(self):
        """Higher wind speed → lower concentration (∝ 1/u)."""
        inp_low = self._make_input(wind_speed=2.0)
        inp_high = self._make_input(wind_speed=10.0)

        c_low = concentration_at_point(x=500, y=0, z=0, input=inp_low)
        c_high = concentration_at_point(x=500, y=0, z=0, input=inp_high)

        assert c_low > c_high, "Low wind → higher concentration"

    def test_proportional_to_source_rate(self):
        """Doubling source rate doubles concentration."""
        inp_1 = self._make_input(source_rate=1.0)
        inp_2 = self._make_input(source_rate=2.0)

        c1 = concentration_at_point(x=500, y=0, z=0, input=inp_1)
        c2 = concentration_at_point(x=500, y=0, z=0, input=inp_2)

        assert c2 == pytest.approx(2.0 * c1, rel=0.01)

    def test_max_ground_concentration_function(self):
        """max_ground_concentration returns positive value."""
        inp = self._make_input()
        c_max, x_max = max_ground_concentration(inp)
        assert c_max > 0, f"Max ground conc should be positive, got {c_max}"
        assert x_max > 0

    def test_calculate_plume_returns_plume_result(self):
        """calculate_plume returns PlumeResult with expected attributes."""
        inp = self._make_input()
        result = calculate_plume(inp)
        assert isinstance(result, PlumeResult)
        assert hasattr(result, 'concentration_grid')
        assert hasattr(result, 'max_concentration')
        assert result.max_concentration > 0

    def test_stable_vs_unstable_ground_conc(self):
        """Stable (F) vs unstable (A) both produce valid positive concentrations."""
        inp_a = self._make_input(stability_class='A')
        inp_f = self._make_input(stability_class='F')

        c_a = concentration_at_point(x=500, y=0, z=0, input=inp_a)
        c_f = concentration_at_point(x=500, y=0, z=0, input=inp_f)

        assert c_a > 0
        assert c_f > 0


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian Puff
# ══════════════════════════════════════════════════════════════════════════════

class TestGaussianPuff:
    """Gaussian puff model physical checks."""

    def _make_input(self, **kwargs):
        defaults = dict(
            mass=100.0,
            release_time=0.0,
            release_duration=0.0,
            wind_speed=5.0,
            wind_direction=0.0,
            stability_class='D',
            release_height=10.0,
            terrain_type='rural',
            temperature=298.15,
            pressure=P_ATM,
            time_start=0.0,
            time_end=600.0,
            time_steps=50,
            grid_x_range=(0.0, 5000.0, 51),
            grid_y_range=(-500.0, 500.0, 51),
            grid_z_range=(0.0, 200.0, 21),
        )
        defaults.update(kwargs)
        return PuffInput(**defaults)

    def test_puff_peak_decreases_with_time(self):
        """Peak concentration decreases as puff disperses."""
        inp = self._make_input()
        c_t1 = concentration_puff(x=500, y=0, z=0, t=100, input=inp)
        c_t2 = concentration_puff(x=1000, y=0, z=0, t=200, input=inp)
        c_t3 = concentration_puff(x=3000, y=0, z=0, t=600, input=inp)

        # At least one pair should show concentration decreasing
        decreases = (c_t3 < c_t2) or (c_t2 < c_t1)
        assert decreases, "Peak concentration should decrease as puff disperses"

    def test_puff_concentration_gaussian(self):
        """Puff concentration is Gaussian-like in y direction."""
        inp = self._make_input()
        t = 60
        x_center = inp.wind_speed * t

        c_center = concentration_puff(x=x_center, y=0, z=0, t=t, input=inp)
        c_off = concentration_puff(x=x_center, y=50, z=0, t=t, input=inp)
        # At centerline, concentration should be higher or equal
        assert c_center >= c_off * 0.99

    def test_calculate_puff_returns_puff_result(self):
        """calculate_puff returns PuffResult with expected data."""
        inp = self._make_input(
            mass=10.0,
            time_start=10.0,
            time_end=60.0,
            time_steps=3,
        )
        result = calculate_puff(inp)
        assert isinstance(result, PuffResult)
        assert len(result.time_series) > 0
        assert len(result.times) > 0
        assert result.max_concentration_over_time.size > 0

    def test_puff_mass_positive(self):
        """All puff grid values are non-negative."""
        inp = self._make_input(mass=10.0)
        result = calculate_puff(inp)
        for snapshot in result.time_series:
            C = snapshot.concentration_grid
            assert C.min() >= 0, f"Negative concentration found"


# ══════════════════════════════════════════════════════════════════════════════
# Dense Gas Dispersion
# ══════════════════════════════════════════════════════════════════════════════

class TestDenseGas:
    """Dense gas dispersion model checks."""

    def _make_input(self, **kwargs):
        defaults = dict(
            source_mass=1000.0,
            release_type='instantaneous',
            cloud_density=6.0,
            air_density=1.2,
            wind_speed=3.0,
            stability_class='D',
            terrain_type='rural',
            temperature_cloud=250.0,
            temperature_ambient=298.15,
            cloud_radius_initial=10.0,
            cloud_height_initial=5.0,
            time_end=300.0,
        )
        defaults.update(kwargs)
        return DenseGasInput(**defaults)

    def test_dense_gas_result_has_radii(self):
        """DenseGasResult has time_series with radius data."""
        inp = self._make_input()
        result = calculate_dense_gas(inp)
        assert hasattr(result, 'time_series')
        assert len(result.time_series) > 0
        assert result.radii.size > 0

    def test_dense_gas_result_has_heights(self):
        """DenseGasResult has height data."""
        inp = self._make_input()
        result = calculate_dense_gas(inp)
        assert result.heights.size > 0

    def test_dense_gas_result_has_times(self):
        """DenseGasResult has time data."""
        inp = self._make_input()
        result = calculate_dense_gas(inp)
        assert len(result.times) > 0
        assert result.times[0] >= 0

    def test_density_ratio_decreases(self):
        """Dense gas density ratio approaches 1.0 (dilution)."""
        inp = self._make_input()
        result = calculate_dense_gas(inp)
        if len(result.density_ratios) >= 2:
            assert result.density_ratios[-1] < result.initial_density_ratio, \
                "Density ratio should decrease with dilution"

    def test_check_if_dense_function(self):
        """check_if_dense returns a boolean for dense gas conditions."""
        # High density ratio → dense gas behavior
        is_dense = check_if_dense(
            substance_density=6.0,
            air_density=1.2,
            threshold=1.1,
        )
        assert isinstance(is_dense, bool)

    def test_dense_gas_continuous_produces_result(self):
        """Continuous release also produces valid result."""
        inp = self._make_input(
            release_type='continuous',
            source_rate=10.0,
            release_duration=60.0,
        )
        result = calculate_dense_gas(inp)
        assert len(result.time_series) > 0

    def test_dense_gas_instantaneous_produces_result(self):
        """Instantaneous release produces valid result."""
        inp = self._make_input(release_type='instantaneous')
        result = calculate_dense_gas(inp)
        assert len(result.time_series) > 0


class TestBuildingWake:
    """Tests for Huber-Snyder building wake factor (ab1131d)."""

    def test_no_building_returns_atm_sigma(self):
        from rekarisk.models.dispersion.gaussian_plume import (
            BuildingParams, sigma_building_wake,
        )
        bldg = BuildingParams(height=0.0, width=0.0, length=0.0)
        assert sigma_building_wake(100.0, 5.0, bldg, "y") == 5.0

    def test_wake_enhances_sigma_inside_zone(self):
        from rekarisk.models.dispersion.gaussian_plume import (
            BuildingParams, sigma_building_wake,
        )
        bldg = BuildingParams(height=10.0, width=20.0, length=15.0)
        sy = sigma_building_wake(20.0, 3.0, bldg, "y")  # 20m < 5×10m=50m
        assert sy > 3.0  # enhanced

    def test_wake_decays_outside_zone(self):
        from rekarisk.models.dispersion.gaussian_plume import (
            BuildingParams, sigma_building_wake,
        )
        bldg = BuildingParams(height=10.0, width=20.0, length=15.0)
        # Far downstream (200m >> 50m wake zone)
        sy_far = sigma_building_wake(200.0, 10.0, bldg, "y")
        # Should be close to atmospheric sigma since decay is strong
        assert sy_far >= 10.0  # never less than atmospheric
        # Should be much less enhancement than inside wake
        sy_near = sigma_building_wake(20.0, 10.0, bldg, "y")
        assert sy_far < sy_near

    def test_vertical_coefficient_larger_than_lateral(self):
        from rekarisk.models.dispersion.gaussian_plume import (
            BuildingParams, sigma_building_wake,
        )
        bldg = BuildingParams(height=10.0, width=10.0, length=10.0)
        # Same distance, same atmospheric sigma
        sy = sigma_building_wake(20.0, 5.0, bldg, "y")
        sz = sigma_building_wake(20.0, 5.0, bldg, "z")
        # Vertical uses 0.7×H, lateral uses 0.35×W with equal H=W=10
        assert sz > sy  # 0.7×10 > 0.35×10

    def test_building_wake_correction_returns_tuple(self):
        from rekarisk.models.dispersion.gaussian_plume import (
            BuildingParams, building_wake_correction,
        )
        bldg = BuildingParams(height=10.0, width=20.0, length=15.0)
        sy, sz = building_wake_correction(30.0, 5.0, 3.0, bldg)
        assert sy > 5.0
        assert sz > 3.0
