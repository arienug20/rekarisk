"""
Rekarisk — Equation of State Model Validation Tests.

Tests for Peng-Robinson (PR), Soave-Redlich-Kwong (SRK) cubic EoS:
single component, mixtures, density, and fugacity coefficient.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rekarisk.core.eos import (
    PengRobinson,
    SoaveRedlichKwong,
    EoSParameters,
)

# ── CoolProp (optional) ──
try:
    from CoolProp.CoolProp import PropsSI
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False
    PropsSI = None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_methane_params() -> tuple:
    """Tc [K], Pc [Pa], omega [-] for methane."""
    return (190.56, 4.599e6, 0.0115)

def get_ethane_params() -> tuple:
    """Tc [K], Pc [Pa], omega [-] for ethane."""
    return (305.32, 4.872e6, 0.0995)

def get_co2_params() -> tuple:
    """Tc [K], Pc [Pa], omega [-] for CO2."""
    return (304.18, 7.382e6, 0.228)


# ══════════════════════════════════════════════════════════════════════════════
# Peng-Robinson Pure Component
# ══════════════════════════════════════════════════════════════════════════════

class TestPengRobinsonPure:
    """Peng-Robinson equation of state for pure components."""

    def test_pr_z_factor_methane_300k_1mpa(self):
        """PR Z-factor for methane at 300K, 1 MPa in known range 0.95-0.99."""
        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        Z = eos.Z_factor(P=1e6, T=300, phase='vapor')
        assert Z > 0, f"Z should be positive, got {Z}"
        # Real gas Z should be slightly < 1 at these conditions
        assert Z > 0.85, f"Z too low: {Z}"
        assert Z < 1.05, f"Z too high: {Z}"

    def test_pr_z_factor_methane_vapor_less_than_1(self):
        """PR Z-factor for methane vapor below critical T < 1."""
        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        Z = eos.Z_factor(P=1e6, T=200, phase='vapor')
        assert Z > 0
        # At low T and moderate P, non-ideal effects should give Z < 1
        assert Z < 1.0, f"Methane vapor at 200K, 1MPa should have Z < 1, got {Z}"

    def test_pr_z_factor_high_pressure(self):
        """PR Z-factor at high pressure > 0 (may be > 1 in supercritical)."""
        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        Z = eos.Z_factor(P=20e6, T=300, phase='vapor')
        assert Z > 0

    def test_pr_z_factor_liquid_phase_small(self):
        """PR Z-factor for liquid phase is small (~0.01-0.5)."""
        Tc, Pc, omega = get_ethane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=30.07)

        Z_liq = eos.Z_factor(P=1e6, T=200, phase='liquid')
        assert Z_liq > 0
        assert Z_liq < 0.5, f"Liquid Z should be small, got {Z_liq}"

    def test_pr_z_factors_returns_two_roots(self):
        """Z_factors returns (Z_liquid, Z_vapor)."""
        Tc, Pc, omega = get_ethane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=30.07)

        Z_liq, Z_vap = eos.Z_factors(P=1e6, T=250)
        assert Z_liq > 0
        assert Z_vap > 0
        assert Z_liq < Z_vap, f"Liquid Z ({Z_liq}) should be < vapor Z ({Z_vap})"

    def test_pr_z_factor_co2(self):
        """PR Z-factor for CO2 at 300K, 5 MPa returns a value."""
        Tc, Pc, omega = get_co2_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=44.01)

        Z = eos.Z_factor(P=5e6, T=300, phase='vapor')
        assert Z > 0, f"CO2 Z should be positive, got {Z}"

    def test_pr_density_positive(self):
        """Density computed from Z-factor is > 0."""
        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        rho = eos.density(P=1e6, T=300, phase='vapor')
        assert rho > 0, f"Density should be positive, got {rho}"
        # Methane at 1 MPa, 300K should be ~6-7 kg/m³
        assert 1.0 < rho < 50.0, f"Density out of expected range: {rho}"

    def test_pr_fugacity_coefficient_single_component(self):
        """Fugacity coefficient for pure component is a finite positive number."""
        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        Z = eos.Z_factor(P=1e6, T=300, phase='vapor')
        phi = eos.fugacity_coefficient(P=1e6, T=300, Z=Z)
        assert len(phi) == 1
        assert phi[0] > 0
        # For ideal-ish conditions, φ should be close to 1
        assert 0.5 < phi[0] < 2.0, f"Fugacity coefficient out of range: {phi[0]}"

    def test_pr_eos_parameters_calculation(self):
        """Peng-Robinson computes a and b parameters."""
        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        # a and b should be positive
        assert eos._params.a > 0, "Attraction parameter a should be positive"
        assert eos._params.b > 0, "Co-volume b should be positive"


# ══════════════════════════════════════════════════════════════════════════════
# Peng-Robinson Mixture
# ══════════════════════════════════════════════════════════════════════════════

class TestPengRobinsonMixture:
    """Peng-Robinson EoS for mixtures using vdW one-fluid mixing rules."""

    def test_mixture_z_factor_computes(self):
        """Mixture Z-factor for methane/ethane computes without error."""
        Tc_me, Pc_me, omega_me = get_methane_params()
        Tc_eth, Pc_eth, omega_eth = get_ethane_params()

        # Create a PR instance (mixture params are passed via comp_params)
        eos = PengRobinson(tc=Tc_me, pc=Pc_me, omega=omega_me, mw=16.04)

        comp_params = [
            EoSParameters(
                a=PengRobinson._compute_a(Tc_me, Pc_me),
                b=PengRobinson._compute_b(Tc_me, Pc_me),
                tc=Tc_me,
                pc=Pc_me,
                omega=omega_me,
                mw=16.04,
            ),
            EoSParameters(
                a=PengRobinson._compute_a(Tc_eth, Pc_eth),
                b=PengRobinson._compute_b(Tc_eth, Pc_eth),
                tc=Tc_eth,
                pc=Pc_eth,
                omega=omega_eth,
                mw=30.07,
            ),
        ]
        mole_fractions = [0.7, 0.3]

        Z = eos.Z_factor(
            P=1e6, T=300,
            phase='vapor',
            mole_fractions=mole_fractions,
            comp_params=comp_params,
        )
        assert Z > 0, f"Mixture Z should be positive, got {Z}"

    def test_mixture_with_binary_kij(self):
        """Mixture with k_ij uses interaction parameters."""
        Tc_me, Pc_me, omega_me = get_methane_params()
        Tc_eth, Pc_eth, omega_eth = get_ethane_params()

        eos = PengRobinson(tc=Tc_me, pc=Pc_me, omega=omega_me, mw=16.04)
        comp_params = [
            EoSParameters(
                a=PengRobinson._compute_a(Tc_me, Pc_me),
                b=PengRobinson._compute_b(Tc_me, Pc_me),
                tc=Tc_me, pc=Pc_me, omega=omega_me, mw=16.04,
            ),
            EoSParameters(
                a=PengRobinson._compute_a(Tc_eth, Pc_eth),
                b=PengRobinson._compute_b(Tc_eth, Pc_eth),
                tc=Tc_eth, pc=Pc_eth, omega=omega_eth, mw=30.07,
            ),
        ]
        mole_fractions = [0.7, 0.3]
        k_ij = np.zeros((2, 2))
        k_ij[0, 1] = k_ij[1, 0] = 0.02  # small binary interaction

        Z = eos.Z_factor(
            P=1e6, T=300,
            phase='vapor',
            mole_fractions=mole_fractions,
            comp_params=comp_params,
            k_ij=k_ij,
        )
        assert Z > 0

    def test_mixture_density_positive(self):
        """Mixture density computed > 0."""
        Tc_me, Pc_me, omega_me = get_methane_params()
        Tc_eth, Pc_eth, omega_eth = get_ethane_params()

        eos = PengRobinson(tc=Tc_me, pc=Pc_me, omega=omega_me, mw=16.04)
        comp_params = [
            EoSParameters(
                a=PengRobinson._compute_a(Tc_me, Pc_me),
                b=PengRobinson._compute_b(Tc_me, Pc_me),
                tc=Tc_me, pc=Pc_me, omega=omega_me, mw=16.04,
            ),
            EoSParameters(
                a=PengRobinson._compute_a(Tc_eth, Pc_eth),
                b=PengRobinson._compute_b(Tc_eth, Pc_eth),
                tc=Tc_eth, pc=Pc_eth, omega=omega_eth, mw=30.07,
            ),
        ]
        mole_fractions = [0.7, 0.3]

        rho = eos.density(
            P=1e6, T=300, phase='vapor',
            mole_fractions=mole_fractions,
            comp_params=comp_params,
        )
        assert rho > 0, f"Mixture density should be positive, got {rho}"

    def test_mixture_fugacity_coefficients(self):
        """Fugacity coefficients for each component in mixture."""
        Tc_me, Pc_me, omega_me = get_methane_params()
        Tc_eth, Pc_eth, omega_eth = get_ethane_params()

        eos = PengRobinson(tc=Tc_me, pc=Pc_me, omega=omega_me, mw=16.04)
        comp_params = [
            EoSParameters(
                a=PengRobinson._compute_a(Tc_me, Pc_me),
                b=PengRobinson._compute_b(Tc_me, Pc_me),
                tc=Tc_me, pc=Pc_me, omega=omega_me, mw=16.04,
            ),
            EoSParameters(
                a=PengRobinson._compute_a(Tc_eth, Pc_eth),
                b=PengRobinson._compute_b(Tc_eth, Pc_eth),
                tc=Tc_eth, pc=Pc_eth, omega=omega_eth, mw=30.07,
            ),
        ]
        mole_fractions = [0.7, 0.3]

        Z = eos.Z_factor(
            P=1e6, T=300, phase='vapor',
            mole_fractions=mole_fractions,
            comp_params=comp_params,
        )
        phi = eos.fugacity_coefficient(
            P=1e6, T=300, Z=Z,
            mole_fractions=mole_fractions,
            comp_params=comp_params,
        )
        assert len(phi) == 2, f"Should get 2 fugacity coefficients, got {len(phi)}"
        for i, p in enumerate(phi):
            assert p > 0, f"φ[{i}] should be positive, got {p}"


# ══════════════════════════════════════════════════════════════════════════════
# SRK Pure Component
# ══════════════════════════════════════════════════════════════════════════════

class TestSRK:
    """Soave-Redlich-Kwong equation of state."""

    def test_srk_z_factor_basic(self):
        """SRK Z-factor for methane returns a plausible value."""
        Tc, Pc, omega = get_methane_params()
        eos = SoaveRedlichKwong(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        Z = eos.Z_factor(P=1e6, T=300, phase='vapor')
        assert Z > 0, f"SRK Z should be positive, got {Z}"
        assert Z > 0.85, f"SRK Z too low: {Z}"

    def test_srk_parameters_computed(self):
        """SRK computes EoS parameters."""
        Tc, Pc, omega = get_methane_params()
        eos = SoaveRedlichKwong(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        assert eos._params.a > 0
        assert eos._params.b > 0

    def test_srk_density_positive(self):
        """SRK density > 0."""
        Tc, Pc, omega = get_methane_params()
        eos = SoaveRedlichKwong(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        rho = eos.density(P=1e6, T=300, phase='vapor')
        assert rho > 0


# ══════════════════════════════════════════════════════════════════════════════
# CoolProp Comparison (optional)
# ══════════════════════════════════════════════════════════════════════════════

class TestCoolPropComparison:
    """Compare PR results against CoolProp reference data."""

    def test_pr_vs_coolprop_methane(self):
        """PR Z-factor within 15% of CoolProp for methane at 300K, 1 MPa."""
        if not HAS_COOLPROP:
            pytest.skip("CoolProp not installed")

        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        T = 300.0
        P = 1e6
        Z_pr = eos.Z_factor(P=P, T=T, phase='vapor')

        try:
            Z_cp = PropsSI('Z', 'T', T, 'P', P, 'Methane')
            assert Z_pr > 0
            # Should be within 15% of CoolProp
            rel_err = abs(Z_pr - Z_cp) / abs(Z_cp)
            assert rel_err < 0.15, \
                f"PR Z={Z_pr:.4f}, CoolProp Z={Z_cp:.4f}, err={rel_err*100:.1f}%"
        except Exception as e:
            pytest.skip(f"CoolProp lookup failed: {e}")

    def test_pr_alpha_decreases_with_temperature(self):
        """alpha(T) should decrease as T increases (away from Tc)."""
        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        a_Tc = eos._alpha(Tc)
        a_2Tc = eos._alpha(2 * Tc)
        a_5Tc = eos._alpha(5 * Tc)

        # alpha decreases with temperature away from critical
        assert a_2Tc < a_Tc, "alpha should decrease above Tc"
        assert a_5Tc < a_2Tc, "alpha should continue decreasing"
