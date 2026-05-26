"""Rekarisk — Explosion Model Validation Tests."""

from __future__ import annotations
import math
import pytest
from rekarisk.core.constants import P_ATM
from rekarisk.models.explosion.tnt_equivalency import (
    TNTInput, calculate_tnt_equivalency, overpressure_at_distance,
)
from rekarisk.models.explosion.tno_multi_energy import (
    TNOInput, calculate_tno_multi_energy, auto_blast_strength,
)
from rekarisk.models.explosion.baker_strehlow import (
    BSTInput, calculate_bst, fuel_reactivity_category,
)


class TestTNTEquivalency:
    """TNT equivalency model checks."""

    def _make_input(self, **kwargs):
        defaults = dict(
            mass_flammable=1000.0,
            heat_of_combustion=46.3e6,
            explosion_efficiency=0.03,
            distances=[10, 20, 50, 100, 200, 500],
            substance_name='propane',
            ambient_pressure=P_ATM,
        )
        defaults.update(kwargs)
        return TNTInput(**defaults)

    def test_tnt_equivalent_mass_positive(self):
        inp = self._make_input()
        r = calculate_tnt_equivalency(inp)
        assert r.tnt_equivalent_mass > 200
        assert r.tnt_equivalent_mass < 400

    def test_overpressure_decreases_with_distance(self):
        inp = self._make_input()
        r = calculate_tnt_equivalency(inp)
        for i in range(len(r.distances) - 1):
            if r.distances[i] >= 50:
                assert r.overpressure[i] > r.overpressure[i + 1]

    def test_overpressure_positive(self):
        inp = self._make_input()
        r = calculate_tnt_equivalency(inp)
        for i, d in enumerate(r.distances):
            if d > 0:
                assert r.overpressure[i] > 0

    def test_larger_mass_greater_overpressure(self):
        inp_s = self._make_input(mass_flammable=100.0)
        inp_l = self._make_input(mass_flammable=5000.0)
        rs = calculate_tnt_equivalency(inp_s)
        rl = calculate_tnt_equivalency(inp_l)
        for i, d in enumerate(rs.distances):
            if d >= 50:
                assert rl.overpressure[i] > rs.overpressure[i]
                break

    def test_tnt_mass_scales_linearly(self):
        inp1 = self._make_input(mass_flammable=100.0)
        inp2 = self._make_input(mass_flammable=200.0)
        r1 = calculate_tnt_equivalency(inp1)
        r2 = calculate_tnt_equivalency(inp2)
        assert r2.tnt_equivalent_mass == pytest.approx(2.0 * r1.tnt_equivalent_mass, rel=0.01)

    def test_efficiency_affects_tnt_mass(self):
        inp_low = self._make_input(explosion_efficiency=0.01)
        inp_high = self._make_input(explosion_efficiency=0.10)
        r_low = calculate_tnt_equivalency(inp_low)
        r_high = calculate_tnt_equivalency(inp_high)
        assert r_high.tnt_equivalent_mass > r_low.tnt_equivalent_mass

    def test_overpressure_at_distance(self):
        p = overpressure_at_distance(1000, 46.3e6, 30, 0.03)
        assert p > 0
        p_far = overpressure_at_distance(1000, 46.3e6, 500, 0.03)
        assert p_far < p


class TestTNO:
    """TNO Multi-Energy model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            mass_flammable=1000.0,
            heat_of_combustion=46.3e6,
            explosion_efficiency=1.0,
            confinement_class='2D',
            congestion_level='high',
            blast_strength=6,
            distances=[10, 20, 50, 100, 200, 500],
            substance_name='propane',
            ambient_pressure=P_ATM,
        )
        defaults.update(kwargs)
        return TNOInput(**defaults)

    def test_strength_10_greater_than_4(self):
        inp4 = self._make_input(blast_strength=4)
        inp10 = self._make_input(blast_strength=10)
        r4 = calculate_tno_multi_energy(inp4)
        r10 = calculate_tno_multi_energy(inp10)
        for i, d in enumerate(r4.distances):
            if d >= 50:
                assert r10.overpressure[i] > r4.overpressure[i]
                break

    def test_overpressure_decreases_with_distance(self):
        inp = self._make_input()
        r = calculate_tno_multi_energy(inp)
        for i in range(len(r.distances) - 1):
            if r.distances[i] >= 50:
                assert r.overpressure[i] > r.overpressure[i + 1]

    def test_overpressure_positive(self):
        inp = self._make_input()
        r = calculate_tno_multi_energy(inp)
        for i, d in enumerate(r.distances):
            if d > 0:
                assert r.overpressure[i] > 0

    def test_auto_blast_strength_valid(self):
        for conf in ['none', '1D', '2D', '3D']:
            for cong in ['low', 'medium', 'high']:
                s = auto_blast_strength(conf, cong)
                assert 1 <= s <= 10


class TestBST:
    """Baker-Strehlow-Tang model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            mass_flammable=1000.0,
            heat_of_combustion=46.3e6,
            fuel_reactivity='medium',
            confinement_class='2D',
            congestion_level='medium',
            flame_mach=0.5,
            distances=[10, 20, 50, 100, 200, 500],
            substance_name='propane',
            ambient_pressure=P_ATM,
        )
        defaults.update(kwargs)
        return BSTInput(**defaults)

    def test_high_reactivity_ge_low(self):
        inp_low = self._make_input(fuel_reactivity='low')
        inp_high = self._make_input(fuel_reactivity='high')
        r_low = calculate_bst(inp_low)
        r_high = calculate_bst(inp_high)
        for i, d in enumerate(r_low.distances):
            if d >= 50:
                assert r_high.overpressure[i] >= r_low.overpressure[i]
                break

    def test_overpressure_decreases_with_distance(self):
        inp = self._make_input()
        r = calculate_bst(inp)
        for i in range(len(r.distances) - 1):
            if r.distances[i] >= 50:
                assert r.overpressure[i] > r.overpressure[i + 1]

    def test_overpressure_positive(self):
        inp = self._make_input()
        r = calculate_bst(inp)
        for i, d in enumerate(r.distances):
            if d > 0:
                assert r.overpressure[i] > 0

    def test_higher_mach_greater_overpressure(self):
        inp_low = self._make_input(flame_mach=0.1)
        inp_high = self._make_input(flame_mach=1.0)
        r_low = calculate_bst(inp_low)
        r_high = calculate_bst(inp_high)
        for i, d in enumerate(r_low.distances):
            if d >= 50:
                assert r_high.overpressure[i] > r_low.overpressure[i]
                break

    def test_fuel_reactivity_category(self):
        for fuel in ['methane', 'propane', 'hydrogen']:
            cat = fuel_reactivity_category(fuel)
            assert cat in ('low', 'medium', 'high')
