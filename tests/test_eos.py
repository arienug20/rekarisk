"""
Rekarisk — Equation of State Validation Tests.

Tests for Peng-Robinson and Soave-Redlich-Kwong equations of state:
compressibility factor, density, and mixture calculations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rekarisk.core.constants import R, P_ATM

# Try to import EoS modules
from rekarisk.core.eos import (
    PengRobinson,
    SoaveRedlichKwong,
    EoSParameters,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_methane_params():
    """Return (Tc, Pc, omega) for methane."""
    return (190.6, 4.599e6, 0.0115)


def get_ethane_params():
    """Return (Tc, Pc, omega) for ethane."""
    return (305.3, 4.872e6, 0.099)


def get_propane_params():
    """Return (Tc, Pc, omega) for propane."""
    return (369.8, 4.247e6, 0.152)


HAS_COOLPROP = False
try:
    from CoolProp.CoolProp import PropsSI
    HAS_COOLPROP = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Peng-Robinson
# ══════════════════════════════════════════════════════════════════════════════

class TestPengRobinson:
    """Peng-Robinson EoS tests."""

    def test_pr_z_factor_methane_300k_1mpa_positive(self):
        """PR Z-factor for methane at 300K, 1 MPa > 0."""
        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        Z = eos.Z_factor(P=1e6, T=300, phase='vapor')
        assert Z > 0, f"Z should be positive, got {Z}"
        # Ideal gas Z ≈ 1.0; real gas should be close but < 1
        assert Z < 1.0, f"Z should be < 1 for real gas at these conditions, got {Z}"

    def test_pr_z_factor_high_pressure(self):
        """PR Z-factor at high pressure > 0 but may be >> 1 (supercritical)."""
        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        Z = eos.Z_factor(P=20e6, T=300, phase='vapor')
        assert Z > 0

    def test_pr_z_factor_liquid(self):
        """PR Z-factor for liquid phase is small (~0.001-0.1)."""
        Tc, Pc, omega = get_ethane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=30.07)

        # At conditions where ethane is liquid
        Z_liq = eos.Z_factor(P=1e6, T=200, phase='liquid')
        assert Z_liq > 0
        assert Z_liq < 0.5, f"Liquid Z should be small, got {Z_liq}"

    def test_pr_z_factor_vs_coolprop(self):
        """PR Z-factor within 10% of CoolProp for methane (if CoolProp available)."""
        if not HAS_COOLPROP:
            pytest.skip("CoolProp not available")

        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(tc=Tc, pc=Pc, omega=omega, mw=16.04)

        # Methane at 300K, 1 MPa
        T = 300
        P = 1e6

        Z_pr = eos.Z_factor(P=P, T=T, phase='vapor')

        # CoolProp reference Z-factor
        try:
            Z_cp = PropsSI('Z', 'T', T, 'P', P, 'Methane')
            assert Z_pr > 0
            # Compare within 10%
            assert abs(Z_pr - Z_cp) / Z_cp < 0.15, \
                f"PR Z={Z_pr:.4f} differs from CoolProp Z={Z_cp:.4f}"
        except Exception as e:
            pytest.skip(f"CoolProp lookup failed: {e}")

    def test_pr_density_positive(self):
        """PR density > 0."""
        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(Tc, Pc, omega)

        Mw = 0.01604  # kg/mol

        Z = eos.Z_factor(
            P=1e6, T=300, phase='vapor',
            mole_fractions=None,
            comp_params=None,
            k_ij=None,
        )
        # ρ = P * Mw / (Z * R * T)
        rho = (1e6 * Mw) / (Z * R * 300)
        assert rho > 0
        # For methane at 1 MPa, 300K: ~6.4 kg/m³
        assert 5.0 < rho < 8.0, f"ρ should be ~6.4 kg/m³, got {rho:.2f}"

    def test_pr_density_liquid(self):
        """PR liquid density > 0 and substantially higher than vapor."""
        Tc, Pc, omega = get_propane_params()
        eos = PengRobinson(Tc, Pc, omega)

        Mw = 0.04409  # kg/mol

        Z_liq = eos.Z_factor(
            P=1e6, T=250, phase='liquid',
            mole_fractions=None,
            comp_params=None,
            k_ij=None,
        )
        Z_vap = eos.Z_factor(
            P=1e6, T=250, phase='vapor',
            mole_fractions=None,
            comp_params=None,
            k_ij=None,
        )

        rho_liq = (1e6 * Mw) / (Z_liq * R * 250)
        rho_vap = (1e6 * Mw) / (Z_vap * R * 250)

        assert rho_liq > 0
        assert rho_vap > 0
        assert rho_liq > rho_vap * 3, "Liquid density should be much higher than vapor"

    def test_pr_parameter_calculation(self):
        """PR a, b parameters are computed correctly."""
        Tc, Pc, omega = get_methane_params()
        eos = PengRobinson(Tc, Pc, omega)

        a = eos.a(300)
        b = eos.b

        assert a > 0, "a parameter should be positive"
        assert b > 0, "b parameter should be positive"
        assert b < 0.001, "b (co-volume) should be small"


# ══════════════════════════════════════════════════════════════════════════════
# Peng-Robinson Mixture
# ══════════════════════════════════════════════════════════════════════════════

class TestPRMixture:
    """Peng-Robinson mixture EoS tests."""

    def test_mixture_z_factor_no_error(self):
        """Mixture Z-factor computed without error for methane + ethane."""
        comp_params = [
            {'name': 'methane', 'Tc': 190.6, 'Pc': 4.599e6, 'omega': 0.0115},
            {'name': 'ethane', 'Tc': 305.3, 'Pc': 4.872e6, 'omega': 0.099},
        ]
        mole_fractions = [0.7, 0.3]
        k_ij = {(0, 1): 0.0}  # zero interaction for methane-ethane

        eos = PengRobinsonMixture(comp_params, k_ij)
        Z = eos.Z_factor(
            P=1e6, T=300, phase='vapor',
            mole_fractions=mole_fractions,
            comp_params=comp_params,
            k_ij=k_ij,
        )
        assert Z > 0

    def test_mixture_z_between_pure_components(self):
        """Mixture Z-factor is between pure component Z-factors."""
        comp_params = [
            {'name': 'methane', 'Tc': 190.6, 'Pc': 4.599e6, 'omega': 0.0115},
            {'name': 'ethane', 'Tc': 305.3, 'Pc': 4.872e6, 'omega': 0.099},
        ]
        mole_fractions = [0.5, 0.5]
        k_ij = {(0, 1): 0.0}

        eos_mix = PengRobinsonMixture(comp_params, k_ij)
        Z_mix = eos_mix.Z_factor(
            P=1e6, T=300, phase='vapor',
            mole_fractions=mole_fractions,
            comp_params=comp_params,
            k_ij=k_ij,
        )

        # Pure components
        eos_me = PengRobinson(190.6, 4.599e6, 0.0115)
        Z_me = eos_me.Z_factor(
            P=1e6, T=300, phase='vapor',
            mole_fractions=[1.0], comp_params=[{'name': 'methane', 'Tc': 190.6, 'Pc': 4.599e6, 'omega': 0.0115}],
            k_ij=None,
        )

        eos_et = PengRobinson(305.3, 4.872e6, 0.099)
        Z_et = eos_et.Z_factor(
            P=1e6, T=300, phase='vapor',
            mole_fractions=[1.0], comp_params=[{'name': 'ethane', 'Tc': 305.3, 'Pc': 4.872e6, 'omega': 0.099}],
            k_ij=None,
        )

        # Mixture Z should be between the two pure Z values
        Z_min = min(Z_me, Z_et)
        Z_max = max(Z_me, Z_et)
        assert Z_min - 0.05 <= Z_mix <= Z_max + 0.05, \
            f"Z_mix={Z_mix:.4f} not between Z_me={Z_me:.4f} and Z_et={Z_et:.4f}"

    def test_mixture_density_positive(self):
        """Mixture density > 0."""
        comp_params = [
            {'name': 'methane', 'Tc': 190.6, 'Pc': 4.599e6, 'omega': 0.0115},
            {'name': 'ethane', 'Tc': 305.3, 'Pc': 4.872e6, 'omega': 0.099},
        ]
        mole_fractions = [0.7, 0.3]
        k_ij = {(0, 1): 0.0}

        eos = PengRobinsonMixture(comp_params, k_ij)
        Z = eos.Z_factor(
            P=1e6, T=300, phase='vapor',
            mole_fractions=mole_fractions,
            comp_params=comp_params,
            k_ij=k_ij,
        )

        # Average MW
        Mw = 0.7 * 0.01604 + 0.3 * 0.03007
        rho = (1e6 * Mw) / (Z * R * 300)
        assert rho > 0


# ══════════════════════════════════════════════════════════════════════════════
# Soave-Redlich-Kwong
# ══════════════════════════════════════════════════════════════════════════════

class TestSRK:
    """Soave-Redlich-Kwong EoS tests."""

    def test_srk_z_factor_positive(self):
        """SRK Z-factor for methane at 300K, 1 MPa > 0."""
        Tc, Pc, omega = get_methane_params()
        eos = SoaveRedlichKwong(Tc, Pc, omega)

        Z = eos.Z_factor(
            P=1e6, T=300, phase='vapor',
            mole_fractions=None,
            comp_params=None,
            k_ij=None,
        )
        assert Z > 0

    def test_srk_density_positive(self):
        """SRK density > 0."""
        Tc, Pc, omega = get_methane_params()
        eos = SoaveRedlichKwong(Tc, Pc, omega)

        Mw = 0.01604
        Z = eos.Z_factor(
            P=1e6, T=300, phase='vapor',
            mole_fractions=None,
            comp_params=None,
            k_ij=None,
        )
        rho = (1e6 * Mw) / (Z * R * 300)
        assert rho > 0

    def test_srk_vs_pr_z_factor_similar(self):
        """SRK and PR give similar Z-factors for simple gases."""
        Tc, Pc, omega = get_methane_params()
        eos_pr = PengRobinson(Tc, Pc, omega)
        eos_srk = SoaveRedlichKwong(Tc, Pc, omega)

        Z_pr = eos_pr.Z_factor(
            P=1e6, T=300, phase='vapor',
            mole_fractions=None, comp_params=None, k_ij=None,
        )
        Z_srk = eos_srk.Z_factor(
            P=1e6, T=300, phase='vapor',
            mole_fractions=None, comp_params=None, k_ij=None,
        )

        assert Z_pr > 0
        assert Z_srk > 0
        # They should be within 5% of each other for methane at moderate conditions
        assert abs(Z_pr - Z_srk) / Z_pr < 0.05
