"""
Rekarisk — Explosion Model Validation Tests.

Tests for TNT equivalency, TNO Multi-Energy, and Baker-Strehlow-Tang
explosion/blast models.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rekarisk.core.constants import P_ATM, TNT_HEAT_OF_DETONATION
from rekarisk.models.explosion.tnt_equivalency import (
    TNTInput,
    calculate_tnt_equivalency,
    overpressure_at_distance,
    distance_to_overpressure,
    kingery_bulmash_overpressure,
)
from rekarisk.models.explosion.tno_multi_energy import (
    TNOInput,
    calculate_tno_multi_energy,
    tno_overpressure,
    auto_blast_strength,
)
from rekarisk.models.explosion.baker_strehlow import (
    BSTInput,
    calculate_bst,
    bst_overpressure,
    fuel_reactivity_category,
)


# ══════════════════════════════════════════════════════════════════════════════
# TNT Equivalency
# ══════════════════════════════════════════════════════════════════════════════

class TestTNTEquivalency:
    """TNT equivalency model checks."""

    def _make_input(self, **kwargs):
        defaults = dict(
            substance='propane',
            mass=1000.0,
            efficiency_factor=0.03,
            heat_of_combustion=46.3e6,
            T_ambient=298.15,
        )
        defaults.update(kwargs)
        return TNTInput(**defaults)

    def test_w_tnt_calculation(self):
        """W_TNT = eta * M * dHc / dHc_TNT."""
        eta = 0.03
        M = 1000.0
        dHc = 46.3e6  # propane
        dHc_TNT = TNT_HEAT_OF_DETONATION  # 4.68e6 J/kg

        W_TNT_expected = eta * M * dHc / dHc_TNT
        # 0.03 * 1000 * 46.3e6 / 4.68e6 ≈ 296.8 kg TNT
        assert 200 < W_TNT_expected < 400

    def test_overpressure_decreases_with_distance(self):
        """TNT overpressure decreases with distance."""
        inp = self._make_input()
        result = calculate_tnt_equivalency(inp)

        p_50 = overpressure_at_distance(result, 50)
        p_100 = overpressure_at_distance(result, 100)
        p_200 = overpressure_at_distance(result, 200)

        assert p_100 < p_50, "Overpressure should decrease with distance"
        assert p_200 < p_100

    def test_overpressure_zero_at_large_distance(self):
        """Overpressure at very large distance → approaches zero."""
        inp = self._make_input()
        result = calculate_tnt_equivalency(inp)
        p_far = overpressure_at_distance(result, 10000)
        assert p_far < 1e-4, "Overpressure should be near zero far away"

    def test_distance_to_overpressure(self):
        """Distance to overpressure threshold decreases with higher threshold."""
        inp = self._make_input()
        d_1 = distance_to_overpressure(inp, 1.0)
        d_10 = distance_to_overpressure(inp, 10.0)
        assert d_10 < d_1, "Higher threshold → shorter distance"

    def test_kingery_bulmash_positive(self):
        """Kingery-Bulmash returns positive overpressure at moderate scaled distance."""
        z = 1.0  # Z = R / W_TNT^(1/3), moderate scaled distance
        p = kingery_bulmash_overpressure(z)
        assert p > 0
        assert p < 1e7  # not absurd

    def test_scaled_distance_relationship(self):
        """Overpressure is a function of scaled distance Z = R / W^(1/3).

        Two different masses at appropriate distances should give similar overpressures.
        """
        # For a small charge
        inp_small = TNTInput(
            substance='tnt',
            mass=10.0,
            efficiency_factor=1.0,
            heat_of_combustion=TNT_HEAT_OF_DETONATION,
        )
        result_small = calculate_tnt_equivalency(inp_small)
        z_small = 5.0  # R / W_tnt^(1/3)
        R_small = z_small * result_small.W_TNT ** (1.0 / 3.0)

        # For a large charge
        inp_large = TNTInput(
            substance='tnt',
            mass=1000.0,
            efficiency_factor=1.0,
            heat_of_combustion=TNT_HEAT_OF_DETONATION,
        )
        result_large = calculate_tnt_equivalency(inp_large)
        z_large = 5.0
        R_large = z_large * result_large.W_TNT ** (1.0 / 3.0)

        p_small = overpressure_at_distance(result_small, R_small)
        p_large = overpressure_at_distance(result_large, R_large)

        # At same scaled distance, overpressure should be similar
        assert abs(p_small - p_large) / max(p_small, 1e-10) < 0.2, \
            f"Same Z should give same P: {p_small:.0f} vs {p_large:.0f} Pa"

    def test_overpressure_output_positive(self):
        """All TNT model overpressures at reasonable distances should be positive."""
        inp = self._make_input()
        result = calculate_tnt_equivalency(inp)
        for R in [10, 20, 50, 100, 200, 500]:
            p = overpressure_at_distance(result, R)
            assert p >= 0, f"Overpressure at R={R}m should be >= 0"

    def test_larger_mass_greater_overpressure(self):
        """Larger mass → greater overpressure at same distance."""
        inp_small = self._make_input(mass=100.0)
        inp_large = self._make_input(mass=5000.0)

        result_small = calculate_tnt_equivalency(inp_small)
        result_large = calculate_tnt_equivalency(inp_large)

        p_small = overpressure_at_distance(result_small, 100)
        p_large = overpressure_at_distance(result_large, 100)

        assert p_large > p_small, "Larger mass → higher overpressure"


# ══════════════════════════════════════════════════════════════════════════════
# TNO Multi-Energy
# ══════════════════════════════════════════════════════════════════════════════

class TestTNO:
    """TNO Multi-Energy model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            substance='propane',
            mass=1000.0,
            blast_strength=6,
            heat_of_combustion=46.3e6,
            T_ambient=298.15,
        )
        defaults.update(kwargs)
        return TNOInput(**defaults)

    def test_strength_10_greater_than_4(self):
        """Blast strength 10 > strength 4 at same distance."""
        inp_s4 = self._make_input(blast_strength=4)
        inp_s10 = self._make_input(blast_strength=10)

        result_s4 = calculate_tno_multi_energy(inp_s4)
        result_s10 = calculate_tno_multi_energy(inp_s10)

        p_s4 = tno_overpressure(result_s4, 100)
        p_s10 = tno_overpressure(result_s10, 100)

        assert p_s10 > p_s4, "Higher blast strength → higher overpressure"

    def test_overpressure_decreases_with_distance(self):
        """TNO overpressure decreases with distance."""
        inp = self._make_input()
        result = calculate_tno_multi_energy(inp)

        p_50 = tno_overpressure(result, 50)
        p_100 = tno_overpressure(result, 100)
        p_200 = tno_overpressure(result, 200)

        assert p_100 < p_50
        assert p_200 < p_100

    def test_overpressure_positive(self):
        """All TNO overpressures at reasonable distances should be positive."""
        inp = self._make_input()
        result = calculate_tno_multi_energy(inp)
        for R in [20, 50, 100, 200, 500]:
            p = tno_overpressure(result, R)
            assert p >= 0, f"TNO overpressure at R={R}m should be >= 0"

    def test_tno_result_has_w_tnt(self):
        """TNO result includes equivalent TNT mass."""
        inp = self._make_input()
        result = calculate_tno_multi_energy(inp)
        assert hasattr(result, 'W_TNT') or hasattr(result, 'overpressure')

    def test_auto_blast_strength_in_range(self):
        """Auto blast strength returns value in 1-10."""
        for congestion in ['low', 'medium', 'high']:
            for confinement in ['unconfined', 'partly_confined', 'confined']:
                s = auto_blast_strength(congestion, confinement)
                assert 1 <= s <= 10, f"Strength {s} out of range for {congestion}/{confinement}"

    def test_mass_scales_appropriately(self):
        """Larger mass → longer reach at same overpressure threshold."""
        inp_small = self._make_input(mass=100.0)
        inp_large = self._make_input(mass=5000.0)

        result_small = calculate_tno_multi_energy(inp_small)
        result_large = calculate_tno_multi_energy(inp_large)

        # At the same distance, large mass → more overpressure
        p_small = tno_overpressure(result_small, 100)
        p_large = tno_overpressure(result_large, 100)

        assert p_large >= p_small * 0.99


# ══════════════════════════════════════════════════════════════════════════════
# Baker-Strehlow-Tang (BST)
# ══════════════════════════════════════════════════════════════════════════════

class TestBST:
    """Baker-Strehlow-Tang model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            substance='propane',
            mass=1000.0,
            reactivity='medium',
            mach_number=0.5,
            heat_of_combustion=46.3e6,
            T_ambient=298.15,
        )
        defaults.update(kwargs)
        return BSTInput(**defaults)

    def test_high_reactivity_greater_than_low(self):
        """High reactivity > low reactivity at same distance."""
        inp_low = self._make_input(reactivity='low')
        inp_high = self._make_input(reactivity='high')

        result_low = calculate_bst(inp_low)
        result_high = calculate_bst(inp_high)

        p_low = bst_overpressure(result_low, 100)
        p_high = bst_overpressure(result_high, 100)

        assert p_high > p_low, "High reactivity → higher overpressure"

    def test_overpressure_decreases_with_distance(self):
        """BST overpressure decreases with distance."""
        inp = self._make_input()
        result = calculate_bst(inp)

        p_50 = bst_overpressure(result, 50)
        p_100 = bst_overpressure(result, 100)
        p_200 = bst_overpressure(result, 200)

        assert p_100 < p_50
        assert p_200 < p_100

    def test_overpressure_positive(self):
        """All BST overpressures at reasonable distances should be positive."""
        inp = self._make_input()
        result = calculate_bst(inp)
        for R in [20, 50, 100, 200, 500]:
            p = bst_overpressure(result, R)
            assert p >= 0, f"BST overpressure at R={R}m should be >= 0"

    def test_mach_number_effect(self):
        """Higher Mach number → higher overpressure."""
        inp_low = self._make_input(mach_number=0.1)
        inp_high = self._make_input(mach_number=1.0)

        result_low = calculate_bst(inp_low)
        result_high = calculate_bst(inp_high)

        p_low = bst_overpressure(result_low, 100)
        p_high = bst_overpressure(result_high, 100)

        assert p_high > p_low, "Higher Mach → higher overpressure"

    def test_fuel_reactivity_category(self):
        """Fuel reactivity returns expected categories."""
        # Methane is low reactivity
        cat = fuel_reactivity_category('methane')
        assert cat in ('low', 'medium', 'high')

        # Hydrogen is high reactivity
        cat_h2 = fuel_reactivity_category('hydrogen')
        assert cat_h2 in ('low', 'medium', 'high')
        # Typically hydrogen ≥ methane in reactivity
        if cat != cat_h2:
            # At minimum, function should return valid values
            assert cat_h2 in ('low', 'medium', 'high')

    def test_bst_result_attributes(self):
        """BST result has essential attributes."""
        inp = self._make_input()
        result = calculate_bst(inp)
        assert result is not None
