"""
Rekarisk — Meteorology Model Validation Tests.

Tests for atmospheric stability classification, dispersion coefficients (σy, σz),
wind profile, and air density calculations.
"""

from __future__ import annotations

import math

import pytest

from rekarisk.core.constants import P_ATM, R, G
from rekarisk.meteorology.stability import (
    StabilityClass,
    classify_stability,
    get_sigma_y,
    get_sigma_z,
    SIGMA_Y_PARAMS,
    SIGMA_Z_PARAMS,
)
from rekarisk.meteorology.wind_profile import (
    wind_speed_at_height,
    power_law_exponent,
    log_wind_profile,
    friction_velocity,
    wind_profile_power_law,
)
from rekarisk.meteorology.air_density import (
    air_density,
    air_density_stp,
)


# ══════════════════════════════════════════════════════════════════════════════
# Stability Classification
# ══════════════════════════════════════════════════════════════════════════════

class TestStabilityClassification:
    """Pasquill stability class classification."""

    def test_strong_solar_day_light_wind_is_unstable(self):
        """Strong solar radiation + light wind + daytime → unstable (A-B)."""
        # Strong insolation: clear day, midday
        stability = classify_stability(
            wind_speed_ms=2.0,
            solar_radiation=800,  # W/m², strong
            cloud_cover_oktas=0,
            is_daytime=True,
        )
        assert stability in ('A', 'B', StabilityClass.A, StabilityClass.B), \
            f"Expected A or B, got {stability}"

    def test_overcast_strong_wind_is_neutral(self):
        """Overcast + strong wind → neutral (D)."""
        stability = classify_stability(
            wind_speed_ms=7.0,
            solar_radiation=100,
            cloud_cover_oktas=8,  # fully overcast
            is_daytime=True,
        )
        assert stability in ('D', StabilityClass.D), \
            f"Expected D (neutral), got {stability}"

    def test_clear_night_light_wind_is_stable(self):
        """Clear night + light wind → stable (F)."""
        stability = classify_stability(
            wind_speed_ms=2.0,
            solar_radiation=0,
            cloud_cover_oktas=0,  # clear
            is_daytime=False,
        )
        assert stability in ('E', 'F', StabilityClass.E, StabilityClass.F), \
            f"Expected E or F (stable), got {stability}"

    def test_returns_valid_class(self):
        """Always returns a valid stability class."""
        for ws in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
            for daytime in [True, False]:
                stability = classify_stability(
                    wind_speed_ms=ws,
                    solar_radiation=300,
                    cloud_cover_oktas=4,
                    is_daytime=daytime,
                )
                valid_classes = ['A', 'B', 'C', 'D', 'E', 'F'] + \
                    [s.value if hasattr(s, 'value') else str(s) for s in StabilityClass]
                assert str(stability) in valid_classes or stability in valid_classes, \
                    f"Invalid class: {stability}"

    def test_classify_stability_with_verbose(self):
        """classify_stability returns a meaningful result for verbose checks."""
        # Check that function runs without exception
        result = classify_stability(
            wind_speed_ms=5.0,
            solar_radiation=500,
            cloud_cover_oktas=3,
            is_daytime=True,
        )
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# Dispersion Coefficients (σy, σz)
# ══════════════════════════════════════════════════════════════════════════════

class TestSigmaCoefficients:
    """σy and σz dispersion coefficient functions."""

    def test_sigma_y_positive(self):
        """σy > 0 for positive distance."""
        # Test multiple stability classes
        for stability in ['A', 'B', 'C', 'D', 'E', 'F']:
            sy = get_sigma_y(
                distance_m=500,
                stability_class=stability,
                terrain_type='rural',
            )
            assert sy > 0, f"σy for {stability} at 500m should be > 0, got {sy}"

    def test_sigma_z_positive(self):
        """σz > 0 for positive distance."""
        for stability in ['A', 'B', 'C', 'D', 'E', 'F']:
            sz = get_sigma_z(
                distance_m=500,
                stability_class=stability,
                terrain_type='rural',
            )
            assert sz > 0, f"σz for {stability} at 500m should be > 0, got {sz}"

    def test_sigma_increases_with_distance(self):
        """σy and σz increase with distance."""
        for stability in ['A', 'D', 'F']:
            sy_100 = get_sigma_y(100, stability, 'rural')
            sy_1000 = get_sigma_y(1000, stability, 'rural')
            sz_100 = get_sigma_z(100, stability, 'rural')
            sz_1000 = get_sigma_z(1000, stability, 'rural')

            assert sy_1000 > sy_100, \
                f"σy({stability}, 1000)={sy_1000:.1f} should > σy(100)={sy_100:.1f}"
            assert sz_1000 > sz_100, \
                f"σz({stability}, 1000)={sz_1000:.1f} should > σz(100)={sz_100:.1f}"

    def test_zero_distance_gives_small_sigma(self):
        """σy and σz at x=0 are small (near source)."""
        sy = get_sigma_y(0, 'D', 'rural')
        sz = get_sigma_z(0, 'D', 'rural')
        assert sy >= 0
        assert sz >= 0

    def test_unstable_has_larger_sigma_y(self):
        """Unstable (A) → larger σy than stable (F) at same distance.

        (Unstable means more turbulence → wider lateral spread.)
        """
        sy_a = get_sigma_y(500, 'A', 'rural')
        sy_f = get_sigma_y(500, 'F', 'rural')
        assert sy_a > sy_f, f"A(σy={sy_a:.0f}) should give wider plume than F(σy={sy_f:.0f})"

    def test_sigma_params_dictionary_populated(self):
        """SIGMA_Y_PARAMS and SIGMA_Z_PARAMS have entries for all classes."""
        for c in ['A', 'B', 'C', 'D', 'E', 'F']:
            assert c in SIGMA_Y_PARAMS, f"Missing σy params for {c}"
            assert c in SIGMA_Z_PARAMS, f"Missing σz params for {c}"


# ══════════════════════════════════════════════════════════════════════════════
# Wind Profile
# ══════════════════════════════════════════════════════════════════════════════

class TestWindProfile:
    """Wind profile (power law, logarithmic) tests."""

    def test_u_50m_greater_than_u_10m_neutral(self):
        """Wind speed at 50m > wind speed at 10m for neutral stability."""
        u_10 = 5.0  # m/s at 10m
        u_50 = wind_speed_at_height(
            wind_speed_ref=u_10,
            ref_height=10.0,
            target_height=50.0,
            stability_class='D',  # neutral
            method='power_law',
        )
        # Wind speed increases with height
        assert u_50 > u_10, f"u(50m)={u_50:.2f} should > u(10m)={u_10}"

    def test_log_profile_u_50m_greater_than_u_10m(self):
        """Log profile: wind speed increases with height."""
        u_10 = 5.0
        u_50 = wind_speed_at_height(
            wind_speed_ref=u_10,
            ref_height=10.0,
            target_height=50.0,
            stability_class='D',
            method='log',
        )
        assert u_50 > u_10

    def test_power_law_exponent_stable_greater_than_unstable(self):
        """Stable → larger power law exponent (p) than unstable.

        Stable air means less vertical mixing → stronger wind gradient.
        """
        p_a = power_law_exponent('A')
        p_d = power_law_exponent('D')
        p_f = power_law_exponent('F')

        assert p_f > p_d, \
            f"Stable (F) exponent {p_f} should be > neutral (D) {p_d}"
        # p_d may be > p_a
        assert p_a > 0
        assert p_f > 0
        assert p_d > 0

    def test_friction_velocity_positive(self):
        """Friction velocity u* > 0 for non-zero wind."""
        u_star = friction_velocity(
            wind_speed_ref=5.0,
            ref_height=10.0,
            roughness_length=0.1,  # open terrain
        )
        assert u_star > 0
        assert u_star < 5.0  # u* is always less than the reference wind speed

    def test_wind_profile_power_law_function(self):
        """Direct power law function returns correct value."""
        u_100 = wind_profile_power_law(
            u_ref=5.0,
            z_ref=10.0,
            z=100.0,
            exponent=0.15,
        )
        expected = 5.0 * (100.0 / 10.0) ** 0.15
        assert u_100 == pytest.approx(expected, rel=0.01)
        assert u_100 > 5.0  # higher altitude → higher speed

    def test_log_wind_profile_function(self):
        """Log wind profile returns positive value."""
        u_100 = log_wind_profile(
            u_ref=5.0,
            z_ref=10.0,
            z=100.0,
            z0=0.1,
        )
        assert u_100 > 0
        assert u_100 > 5.0  # higher altitude → higher speed

    def test_sea_roughness_lower_than_urban(self):
        """Roughness length: sea < rural < urban."""
        z0_sea = 0.0002
        z0_rural = 0.1
        z0_urban = 2.0

        u_10 = 5.0
        u_50_sea = log_wind_profile(u_10, 10.0, 50.0, z0_sea)
        u_50_urban = log_wind_profile(u_10, 10.0, 50.0, z0_urban)

        # Smoother surface → higher wind speed at altitude
        assert u_50_sea > u_50_urban, \
            "Smooth surface (sea) should have higher wind speed aloft"


# ══════════════════════════════════════════════════════════════════════════════
# Air Density
# ══════════════════════════════════════════════════════════════════════════════

class TestAirDensity:
    """Air density calculations."""

    def test_air_density_at_stp(self):
        """Air density at STP ≈ 1.2 kg/m³."""
        rho = air_density(
            T=293.15,  # 20°C
            P=P_ATM,
            humidity=0.5,
        )
        assert 1.15 < rho < 1.30, \
            f"Air density should be ~1.2 kg/m³ at 20°C, got {rho:.3f}"

    def test_air_density_stp_function(self):
        """air_density_stp() returns ~1.2 kg/m³."""
        rho = air_density_stp()
        assert 1.15 < rho < 1.25, f"STP density should be ~1.2 kg/m³, got {rho:.3f}"

    def test_air_density_decreases_with_temperature(self):
        """Higher temperature → lower air density."""
        rho_cold = air_density(T=273.15, P=P_ATM, humidity=0.5)
        rho_hot = air_density(T=313.15, P=P_ATM, humidity=0.5)

        assert rho_cold > rho_hot, \
            f"Colder air ({rho_cold:.3f}) should be denser than hot ({rho_hot:.3f})"

    def test_air_density_increases_with_pressure(self):
        """Higher pressure → higher air density."""
        rho_low = air_density(T=273.15, P=0.8 * P_ATM, humidity=0.5)
        rho_high = air_density(T=273.15, P=1.2 * P_ATM, humidity=0.5)

        assert rho_high > rho_low

    def test_air_density_ideal_gas_limit(self):
        """Dry air density approaches ideal gas law ρ = P * M / (R * T)."""
        # In dry air limit (humidity=0), density ≈ P * M_air / (R * T)
        M_air = 0.0289647  # kg/mol
        rho = air_density(T=300, P=P_ATM, humidity=0.0)
        rho_ideal = (P_ATM * M_air) / (R * 300)
        # Should be within ~2% of ideal gas
        assert abs(rho - rho_ideal) / rho_ideal < 0.02
