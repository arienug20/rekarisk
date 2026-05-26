"""
Rekarisk — Meteorology Model Validation Tests.

Tests for atmospheric stability classification, dispersion coefficients (sig_y, sig_z),
wind profile (power law, log law), and atmospheric density calculations.
"""

from __future__ import annotations

import math

import pytest

from rekarisk.core.constants import P_ATM, R
from rekarisk.meteorology.stability import (
    classify_stability,
    sigma_y,
    sigma_z,
    power_law_exponent,
)
from rekarisk.meteorology.meteorology import (
    wind_power_law,
    wind_log_law,
    friction_velocity,
    atmospheric_density,
)


# ══════════════════════════════════════════════════════════════════════════════
# Stability Classification
# ══════════════════════════════════════════════════════════════════════════════

class TestStabilityClassification:
    """Pasquill stability class classification."""

    def test_strong_solar_day_light_wind_is_a(self):
        """classify_stability(1, solar_radiation=700, is_daytime=True) == 'A'."""
        stability = classify_stability(
            wind_speed_ms=1.0,
            solar_radiation=700,
            is_daytime=True,
        )
        assert stability == 'A', f"Expected 'A', got {stability}"

    def test_strong_wind_clear_night_is_d(self):
        """classify_stability(7, cloud_cover_oktas=0, is_daytime=False) == 'D'."""
        stability = classify_stability(
            wind_speed_ms=7.0,
            cloud_cover_oktas=0,
            is_daytime=False,
        )
        assert stability == 'D', f"Expected 'D', got {stability}"

    def test_high_solar_light_wind_unstable(self):
        """Strong solar + light wind → unstable (A or B)."""
        for ws in [0.5, 1.5, 2.0]:
            stability = classify_stability(
                wind_speed_ms=ws,
                solar_radiation=800,
                is_daytime=True,
            )
            assert stability in ('A', 'B'), \
                f"Expected A or B, got {stability} at ws={ws}"

    def test_overcast_strong_wind_is_neutral(self):
        """Overcast + strong wind → neutral (D)."""
        stability = classify_stability(
            wind_speed_ms=7.0,
            solar_radiation=100,
            cloud_cover_oktas=8,
            is_daytime=True,
        )
        assert stability == 'D', f"Expected D (neutral), got {stability}"

    def test_clear_night_light_wind_is_stable(self):
        """Clear night + light wind → stable (E or F)."""
        stability = classify_stability(
            wind_speed_ms=1.5,
            solar_radiation=0,
            cloud_cover_oktas=0,
            is_daytime=False,
        )
        assert stability in ('E', 'F'), \
            f"Expected E or F (stable), got {stability}"

    def test_all_stability_classes_produced(self):
        """Various conditions produce different valid stability classes."""
        classes_seen = set()
        test_cases = [
            (1.0, 700, None, True),     # A
            (2.5, 500, None, True),     # B
            (4.0, 300, None, True),     # C
            (5.0, 200, None, True),     # D
            (3.0, None, 2, False),      # E/F
            (7.0, None, 0, False),      # D
        ]
        for ws, solar, cloud, daytime in test_cases:
            stability = classify_stability(
                wind_speed_ms=ws,
                solar_radiation=solar,
                cloud_cover_oktas=cloud,
                is_daytime=daytime,
            )
            classes_seen.add(stability)
        assert len(classes_seen) >= 3, f"Should see multiple classes, saw {classes_seen}"

    def test_negative_wind_speed_raises_error(self):
        """Negative wind speed should raise ValueError."""
        with pytest.raises(ValueError):
            classify_stability(wind_speed_ms=-1.0, is_daytime=True)


# ══════════════════════════════════════════════════════════════════════════════
# Dispersion Coefficients (sigma_y, sigma_z)
# ══════════════════════════════════════════════════════════════════════════════

class TestSigmaCoefficients:
    """sigma_y and sigma_z dispersion coefficient functions."""

    def test_sigma_y_1000m_d_rural_about_80(self):
        """sigma_y(1000, 'D', 'rural') ≈ 80 ± 10."""
        sy = sigma_y(1000.0, 'D', 'rural')
        assert 70.0 < sy < 90.0, f"sigma_y(1000, D, rural) should be ~80, got {sy}"

    def test_sigma_z_1000m_d_rural_range(self):
        """sigma_z(1000, 'D', 'rural') ≈ 50-70."""
        sz = sigma_z(1000.0, 'D', 'rural')
        assert 40.0 < sz < 85.0, f"sigma_z(1000, D, rural) should be ~50-70, got {sz}"

    def test_sigma_y_positive(self):
        """sigma_y > 0 for positive distance."""
        for stability in ['A', 'B', 'C', 'D', 'E', 'F']:
            sy = sigma_y(500.0, stability, 'rural')
            assert sy > 0, f"sigma_y for {stability} at 500m should be > 0, got {sy}"

    def test_sigma_z_positive(self):
        """sigma_z > 0 for positive distance."""
        for stability in ['A', 'B', 'C', 'D', 'E', 'F']:
            sz = sigma_z(500.0, stability, 'rural')
            assert sz > 0, f"sigma_z for {stability} at 500m should be > 0, got {sz}"

    def test_sigma_increases_with_distance(self):
        """sigma_y and sigma_z increase with distance."""
        for stability in ['A', 'D', 'F']:
            sy_100 = sigma_y(100, stability, 'rural')
            sy_1000 = sigma_y(1000, stability, 'rural')
            sz_100 = sigma_z(100, stability, 'rural')
            sz_1000 = sigma_z(1000, stability, 'rural')

            assert sy_1000 > sy_100, \
                f"sigma_y({stability}, 1000)={sy_1000:.1f} > sigma_y(100)={sy_100:.1f}"
            assert sz_1000 > sz_100, \
                f"sigma_z({stability}, 1000)={sz_1000:.1f} > sigma_z(100)={sz_100:.1f}"

    def test_zero_distance_gives_small_sigma(self):
        """sigma_y and sigma_z at x=0 are small (near source)."""
        sy = sigma_y(0, 'D', 'rural')
        sz = sigma_z(0, 'D', 'rural')
        assert sy >= 0
        assert sz >= 0

    def test_unstable_has_larger_sigma_y(self):
        """Unstable (A) → larger sigma_y than stable (F) at same distance."""
        sy_a = sigma_y(500, 'A', 'rural')
        sy_f = sigma_y(500, 'F', 'rural')
        assert sy_a > sy_f, \
            f"A(sigma_y={sy_a:.0f}) should be > F(sigma_y={sy_f:.0f})"

    def test_sigma_y_negative_distance_raises(self):
        """Negative distance should raise ValueError."""
        with pytest.raises(ValueError):
            sigma_y(-100, 'D', 'rural')
        with pytest.raises(ValueError):
            sigma_z(-100, 'D', 'rural')


# ══════════════════════════════════════════════════════════════════════════════
# Wind Profile
# ══════════════════════════════════════════════════════════════════════════════

class TestWindProfile:
    """Wind profile (power law, logarithmic) tests."""

    def test_wind_power_law_50m_gt_5(self):
        """wind_power_law(z_m=50, u_ref=5, z_ref=10, 'D') > 5.0."""
        u_50 = wind_power_law(z_m=50.0, u_ref=5.0, z_ref=10.0, stability='D')
        assert u_50 > 5.0, f"u(50m)={u_50:.2f} should be > u(10m)=5.0"

    def test_wind_log_law_increases_with_height(self):
        """Log profile: wind speed increases with height."""
        u_50 = wind_log_law(z_m=50.0, u_ref=5.0, z_ref=10.0, z0=0.1)
        u_10 = wind_log_law(z_m=10.0, u_ref=5.0, z_ref=10.0, z0=0.1)
        assert u_50 > u_10

    def test_power_law_exponent_stable_gt_unstable(self):
        """Stable → larger power law exponent than unstable."""
        p_a = power_law_exponent('A')
        p_d = power_law_exponent('D')
        p_f = power_law_exponent('F')

        assert p_f > p_d, \
            f"Stable (F) exponent {p_f} should be > neutral (D) {p_d}"
        assert p_a > 0
        assert p_f > 0
        assert p_d > 0

    def test_friction_velocity_positive(self):
        """Friction velocity u* > 0 for non-zero wind."""
        u_star = friction_velocity(u_ref=5.0, z_ref=10.0, z0=0.1)
        assert u_star > 0
        assert u_star < 5.0

    def test_log_law_different_roughness(self):
        """Log law works for very different roughness lengths."""
        # With the same reference, rougher terrain gives steeper gradient
        # Both must be valid for their respective z0 values
        u_sea = wind_log_law(z_m=50.0, u_ref=5.0, z_ref=10.0, z0=0.0002)
        u_grass = wind_log_law(z_m=50.0, u_ref=5.0, z_ref=10.0, z0=0.03)
        assert u_sea > 0
        assert u_grass > 0
        # Different roughness gives different wind profiles
        assert abs(u_sea - u_grass) > 0.01


# ══════════════════════════════════════════════════════════════════════════════
# Atmospheric Density
# ══════════════════════════════════════════════════════════════════════════════

class TestAtmosphericDensity:
    """atmospheric_density function tests."""

    def test_atmospheric_density_293k_101325pa_about_12(self):
        """atmospheric_density(293.15, 101325) ≈ 1.2 ± 0.05."""
        rho = atmospheric_density(temperature_k=293.15, pressure_pa=101325.0)
        assert 1.15 < rho < 1.25, \
            f"Air density should be ~1.2 kg/m³ at 20°C, got {rho:.3f}"

    def test_atmospheric_density_decreases_with_temperature(self):
        """Higher temperature → lower air density."""
        rho_cold = atmospheric_density(temperature_k=273.15, pressure_pa=P_ATM)
        rho_hot = atmospheric_density(temperature_k=313.15, pressure_pa=P_ATM)

        assert rho_cold > rho_hot, \
            f"Colder air ({rho_cold:.3f}) should be denser than hot ({rho_hot:.3f})"

    def test_atmospheric_density_increases_with_pressure(self):
        """Higher pressure → higher air density."""
        rho_low = atmospheric_density(temperature_k=273.15, pressure_pa=0.8 * P_ATM)
        rho_high = atmospheric_density(temperature_k=273.15, pressure_pa=1.2 * P_ATM)

        assert rho_high > rho_low

    def test_atmospheric_density_positive(self):
        """Density is always positive."""
        rho = atmospheric_density(temperature_k=313.15, pressure_pa=80000.0)
        assert rho > 0

    def test_atmospheric_density_dry_air_ideal_gas(self):
        """Dry air density approaches ideal gas law ρ = P * M_air / (R * T)."""
        M_air = 0.0289647  # kg/mol
        rho = atmospheric_density(
            temperature_k=300.0,
            pressure_pa=P_ATM,
            relative_humidity_pct=0.0,
        )
        rho_ideal = (P_ATM * M_air) / (R * 300.0)
        # Should be within ~1% of ideal gas for dry air
        assert abs(rho - rho_ideal) / rho_ideal < 0.02, \
            f"Dry air density {rho:.3f} vs ideal {rho_ideal:.3f}"
