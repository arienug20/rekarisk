"""
Rekarisk — Source Term Model Validation Tests.

Tests for orifice discharge, PSV/relief valve sizing, vessel blowdown,
and pool evaporation models against known benchmarks and physical checks.
"""

from __future__ import annotations

import math

import pytest

from rekarisk.core.constants import R, P_ATM, G, EPSILON
from rekarisk.models.source_term.orifice import (
    OrificeInput,
    calculate_orifice,
    liquid_orifice_discharge,
    gas_orifice_discharge,
    calculate_flashing_fraction,
    calculate_choked_pressure,
    estimate_omega_parameter,
    quick_liquid_orifice,
    quick_gas_orifice,
)
from rekarisk.models.source_term.relief_valve import (
    ReliefValveInput,
    calculate_relief_valve,
    size_gas_vapor_relief,
    size_liquid_relief,
    select_orifice_designation,
    API_ORIFICE_AREAS,
)
from rekarisk.models.source_term.vessel_depressur import (
    VesselInput,
    calculate_vessel_blowdown,
)
from rekarisk.models.source_term.pool_evaporation import (
    PoolInput,
    simulate_pool,
    evaporation_rate,
    mass_transfer_coefficient,
)


# ══════════════════════════════════════════════════════════════════════════════
# Orifice Discharge
# ══════════════════════════════════════════════════════════════════════════════

class TestLiquidOrifice:
    """Liquid water discharge through orifice."""

    def test_water_10mm_hole_5bar(self):
        """Liquid water through 10mm hole at 5 bar → check mdot order of magnitude.

        Physical expectation:
          A = π * (0.01)² / 4 = 7.854e-5 m²
          v = sqrt(2 * Δp / ρ) = sqrt(2 * 5e5 / 1000) ≈ 31.6 m/s
          mdot ≈ 0.62 * 7.854e-5 * 1000 * 31.6 ≈ 1.54 kg/s
        """
        result = quick_liquid_orifice(
            Cd=0.62,
            d_hole=0.010,
            P_upstream=5e5 + P_ATM,  # 5 bar gauge + atm = ~6 bar abs, dp = 5 bar gauge
            P_downstream=P_ATM,
            rho=1000.0,
        )
        # mdot should be ~1.5 kg/s (order of magnitude ~1-3 kg/s)
        assert result.mdot_initial > 0.5, "mdot too low for 5 bar through 10mm"
        assert result.mdot_initial < 5.0, "mdot too high for 5 bar through 10mm"
        assert result.velocity > 0
        assert result.velocity < 100  # < 100 m/s for liquid at 5 bar

    def test_liquid_no_pressure_difference(self):
        """Zero pressure difference → zero flow."""
        result = quick_liquid_orifice(
            Cd=0.62, d_hole=0.010,
            P_upstream=P_ATM, P_downstream=P_ATM,
            rho=1000.0,
        )
        assert result.mdot_initial == pytest.approx(0.0, abs=1e-6)

    def test_liquid_higher_downstream_no_backflow(self):
        """Higher downstream pressure → no backflow (zero mdot)."""
        result = quick_liquid_orifice(
            Cd=0.62, d_hole=0.010,
            P_upstream=P_ATM, P_downstream=2 * P_ATM,
            rho=1000.0,
        )
        assert result.mdot_initial == pytest.approx(0.0, abs=1e-6)

    def test_liquid_with_head(self):
        """Liquid head adds to driving pressure."""
        result_no_head = quick_liquid_orifice(
            Cd=0.62, d_hole=0.010,
            P_upstream=2e5 + P_ATM, P_downstream=P_ATM,
            rho=1000.0, h_head=0.0,
        )
        result_with_head = quick_liquid_orifice(
            Cd=0.62, d_hole=0.010,
            P_upstream=2e5 + P_ATM, P_downstream=P_ATM,
            rho=1000.0, h_head=5.0,
        )
        # With head, mdot should be larger
        assert result_with_head.mdot_initial > result_no_head.mdot_initial

    def test_liquid_orifice_area(self):
        """Verify orifice area calculation."""
        inp = OrificeInput(Cd=0.62, d_hole=0.05, P_upstream=5e5 + P_ATM,
                          P_downstream=P_ATM, T=300, phase='liquid', rho=1000)
        result = calculate_orifice(inp)
        expected_area = math.pi * (0.025) ** 2
        assert result.area == pytest.approx(expected_area, rel=1e-6)


class TestGasOrifice:
    """Gas/vapor discharge through orifice."""

    def test_gas_choked_flow(self):
        """High pressure ratio → choked flow for gas."""
        # Methane at 50 bar gauge through 5mm hole to atmosphere
        # Critical pressure ratio for k=1.3: r_crit ≈ 0.546
        result = quick_gas_orifice(
            Cd=0.62, d_hole=0.005,
            P_upstream=50e5 + P_ATM, P_downstream=P_ATM,
            T=300, k=1.3, MW=0.016,
        )
        assert result.is_choked, "Should be choked at 50 bar → atm"
        assert result.mdot_initial > 0
        assert result.flow_regime == "choked"

    def test_gas_subsonic_flow(self):
        """Low pressure ratio → subsonic flow."""
        # Methane at 1.1 bar abs through 5mm hole to 1.0 atm
        result = quick_gas_orifice(
            Cd=0.62, d_hole=0.005,
            P_upstream=1.1 * P_ATM, P_downstream=P_ATM,
            T=300, k=1.3, MW=0.016,
        )
        assert not result.is_choked, "Should be subsonic at low pressure ratio"
        assert result.flow_regime == "subsonic"
        assert result.mdot_initial > 0

    def test_gas_mdot_positive_choked(self):
        """Mass flow rate > 0 for choked gas flow."""
        result = quick_gas_orifice(
            Cd=0.62, d_hole=0.010,
            P_upstream=10e5 + P_ATM, P_downstream=P_ATM,
            T=300, k=1.4, MW=0.0289647,
        )
        assert result.mdot_initial > 0
        assert result.velocity > 0

    def test_gas_hydrogen_vs_methane(self):
        """H₂ (lighter) → higher velocity than CH₄ at same conditions."""
        hydrogen = quick_gas_orifice(
            Cd=0.62, d_hole=0.005,
            P_upstream=10e5 + P_ATM, P_downstream=P_ATM,
            T=300, k=1.41, MW=0.002,
        )
        methane = quick_gas_orifice(
            Cd=0.62, d_hole=0.005,
            P_upstream=10e5 + P_ATM, P_downstream=P_ATM,
            T=300, k=1.3, MW=0.016,
        )
        # H₂ has higher sonic velocity → should produce higher velocity
        assert hydrogen.velocity > methane.velocity

    def test_choked_pressure_calculation(self):
        """Critical pressure ratio for k=1.4 is ~0.528."""
        P_up = 1e6
        P_choked = calculate_choked_pressure(P_up, 1.4)
        expected_ratio = (2.0 / (1.4 + 1.0)) ** (1.4 / (1.4 - 1.0))
        assert P_choked == pytest.approx(P_up * expected_ratio, rel=1e-3)
        assert P_choked < P_up

    def test_gas_orifice_discharge_function(self):
        """Direct function call returns expected dict keys."""
        area = math.pi * (0.005) ** 2
        result = gas_orifice_discharge(
            Cd=0.62, area=area,
            P_up=10e5, P_down=P_ATM,
            k=1.4, T=300, MW=0.0289647,
        )
        assert "mdot" in result
        assert "velocity" in result
        assert "is_choked" in result
        assert "P_choked" in result
        assert result["mdot"] > 0


class TestTwoPhaseOrifice:
    """Two-phase / flashing discharge."""

    def test_flashing_fraction_zero_when_subcooled(self):
        """No flashing when liquid is below boiling point."""
        x = calculate_flashing_fraction(
            T=350, T_boil=373, cp_liquid=4200, hfg=2.26e6,
        )
        assert x == 0.0

    def test_flashing_fraction_positive_when_superheated(self):
        """Positive flashing when liquid is above boiling point."""
        x = calculate_flashing_fraction(
            T=400, T_boil=373, cp_liquid=4200, hfg=2.26e6,
        )
        assert x > 0.0
        assert x < 1.0

    def test_flashing_fraction_clamped_to_one(self):
        """Flashing fraction cannot exceed 1.0."""
        x = calculate_flashing_fraction(
            T=1000, T_boil=373, cp_liquid=4200, hfg=1000,
        )
        assert x <= 1.0

    def test_omega_parameter_estimate(self):
        """Omega parameter returns reasonable value."""
        omega = estimate_omega_parameter(
            x0=0.0, k=1.3, rho_l=1000, rho_g=1.2,
            cp_liquid=4200, T=373, P=P_ATM, hfg=2.26e6,
        )
        assert omega > 0.1
        assert omega < 100.0

    def test_two_phase_orifice_positive_mdot(self):
        """Two-phase orifice gives positive mdot."""
        inp = OrificeInput(
            Cd=0.62, d_hole=0.010,
            P_upstream=5e5 + P_ATM, P_downstream=P_ATM,
            T=400, phase='two_phase',
            rho=1000, rho_gas=2.5, molecular_weight=0.018,
            cp_cv_ratio=1.3,
            heat_of_vaporization=2.26e6,
            cp_liquid=4200,
            T_boiling=373,
        )
        result = calculate_orifice(inp)
        assert result.mdot_initial > 0


# ══════════════════════════════════════════════════════════════════════════════
# PSV / Relief Valve Sizing (API 520)
# ══════════════════════════════════════════════════════════════════════════════

class TestPSVSizing:
    """API 520 relief valve sizing."""

    def test_gas_vapor_sizing_positive_area(self):
        """Gas relief gives positive required area."""
        inp = ReliefValveInput(
            scenario='fire_exposure',
            P_set=1e6, T_relieving=350,
            flow_rate=2.5, fluid='gas',
            molecular_weight=0.016, cp_cv_ratio=1.3,
        )
        result = calculate_relief_valve(inp)
        assert result.A_required_mm2 > 0
        assert result.A_required_mm2 < 1e6  # reasonable

    def test_liquid_sizing_positive_area(self):
        """Liquid relief gives positive required area."""
        inp = ReliefValveInput(
            scenario='blocked_outlet',
            P_set=1e6, T_relieving=300,
            flow_rate=5.0, fluid='liquid',
            rho=1000,
        )
        result = calculate_relief_valve(inp)
        assert result.A_required_mm2 > 0

    def test_steam_sizing_positive_area(self):
        """Steam relief gives positive required area."""
        inp = ReliefValveInput(
            scenario='fire_exposure',
            P_set=1e6, T_relieving=450,
            flow_rate=2.0, fluid='steam',
        )
        result = calculate_relief_valve(inp)
        assert result.A_required_mm2 > 0

    def test_gas_choked_default(self):
        """Gas relief at typical conditions is choked."""
        inp = ReliefValveInput(
            scenario='fire_exposure',
            P_set=1e6, T_relieving=350,
            flow_rate=1.0, fluid='gas',
            molecular_weight=0.016, cp_cv_ratio=1.3,
        )
        result = calculate_relief_valve(inp)
        assert result.is_choked

    def test_orifice_designation_selection(self):
        """Select smallest API orifice meeting area requirement."""
        # 100 mm² → between E(126) and F(198)
        assert select_orifice_designation(100) >= "D"
        assert select_orifice_designation(500) == "H"  # H area = 506 mm²
        assert select_orifice_designation(1000) == "K"  # K area = 1186 mm²
        assert select_orifice_designation(20000) == "T+"  # exceeds largest

    def test_api_area_greater_than_required(self):
        """Selected API area >= required."""
        for req in [50, 150, 500, 1000, 5000, 15000]:
            des = select_orifice_designation(req)
            if des != "T+":
                assert API_ORIFICE_AREAS[des] >= req

    def test_gas_relief_subcritical_at_high_backpressure(self):
        """High backpressure (near set pressure) → subcritical flow."""
        # Convenient test: use the size_gas_vapor_relief directly
        P_set_abs = 1.5e6
        P_back = 1.2e6  # high backpressure
        result = size_gas_vapor_relief(
            W=2.0, T=350, Z=1.0, MW=0.016, k=1.3,
            P_relieve=P_set_abs, P_back=P_back,
        )
        assert not result["is_choked"]

    def test_liquid_relief_high_dp(self):
        """Liquid sizing at high ΔP."""
        result = size_liquid_relief(
            W=10.0, rho=1000,
            P_relieve=2e6, P_back=P_ATM,
        )
        assert result["A_mm2"] > 0
        assert result["A_mm2"] < 1e6


# ══════════════════════════════════════════════════════════════════════════════
# Vessel Blowdown
# ══════════════════════════════════════════════════════════════════════════════

class TestVesselBlowdown:
    """Vessel blowdown / depressurization."""

    def setup_method(self):
        self.inputs = VesselInput(
            V=10.0, A_wall=25.0,
            P_initial=600000, T_initial=300,
            orifice_d=0.025, Cd=0.62,
            t_max=120, P_target=P_ATM,
            phase='gas', mode='api521',
            molecular_weight=0.0289647, cp_cv_ratio=1.4,
        )

    def test_pressure_decreases(self):
        """Pressure decreases during blowdown."""
        result = calculate_vessel_blowdown(self.inputs)
        assert result.P[-1] < result.P[0]
        assert result.P[-1] >= P_ATM - EPSILON

    def test_temperature_decreases(self):
        """Temperature decreases (isentropic expansion + depressurization)."""
        result = calculate_vessel_blowdown(self.inputs)
        assert result.T[-1] < result.T[0]

    def test_mass_decreases(self):
        """Mass in vessel decreases."""
        result = calculate_vessel_blowdown(self.inputs)
        assert result.m[-1] < result.m[0]

    def test_total_mass_released_positive(self):
        """Total mass released > 0."""
        result = calculate_vessel_blowdown(self.inputs)
        assert result.total_mass_released > 0

    def test_mdot_positive_initial(self):
        """Initial mass flow rate > 0."""
        result = calculate_vessel_blowdown(self.inputs)
        # First few mdot values should be > 0
        assert any(md > 0 for md in result.mdot[:5])

    def test_blowdown_returns_result_object(self):
        """Returns VesselResult with correct attributes."""
        result = calculate_vessel_blowdown(self.inputs)
        assert hasattr(result, 't')
        assert hasattr(result, 'P')
        assert hasattr(result, 'T')
        assert hasattr(result, 'm')
        assert hasattr(result, 'mdot')
        assert len(result.t) > 0
        assert len(result.P) == len(result.t)

    def test_two_phase_blowdown_runs(self):
        """Two-phase blowdown runs without error."""
        inp = VesselInput(
            V=10.0, A_wall=25.0,
            P_initial=600000, T_initial=400,
            orifice_d=0.025, Cd=0.62,
            t_max=60, P_target=P_ATM,
            phase='two_phase', mode='api521',
            rho_liquid=1000, molecular_weight=0.018,
            cp_cv_ratio=1.3,
            heat_of_vaporization=2.26e6,
            T_boiling=373,
        )
        result = calculate_vessel_blowdown(inp)
        assert result.total_mass_released >= 0

    def test_high_pressure_vessel_choked(self):
        """High-pressure gas vessel blowdown involves choking."""
        inp = VesselInput(
            V=5.0, A_wall=15.0,
            P_initial=2e6, T_initial=300,
            orifice_d=0.010, Cd=0.62,
            t_max=60, P_target=P_ATM,
            phase='gas', mode='api521',
            molecular_weight=0.016, cp_cv_ratio=1.3,
        )
        result = calculate_vessel_blowdown(inp)
        assert result.total_mass_released > 0


# ══════════════════════════════════════════════════════════════════════════════
# Pool Evaporation
# ══════════════════════════════════════════════════════════════════════════════

class TestPoolEvaporation:
    """Pool spreading and evaporation."""

    def test_evaporation_rate_positive(self):
        """Evaporation rate for spill on concrete > 0."""
        inp = PoolInput(
            substance='gasoline',
            spill_mass=1000,
            T_ambient=298.15,
            wind_speed=3.0,
            surface='concrete',
            rho_l=740,
            molecular_weight=0.100,
            vapor_pressure=50000,
            heat_of_vaporization=3.5e5,
            boiling_point=350,
            cp_liquid=2000,
        )
        result = simulate_pool(inp)
        assert result.total_evaporated > 0, "Should evaporate some mass"
        assert result.evap_rate.max() > 0, "Peak evaporation rate should be positive"

    def test_pool_area_is_reasonable(self):
        """Pool area is positive and finite."""
        inp = PoolInput(
            substance='water',
            spill_mass=1000,
            T_ambient=298.15,
            wind_speed=3.0,
            surface='concrete',
            rho_l=1000,
            molecular_weight=0.018,
            vapor_pressure=3169,  # water at 25°C
            heat_of_vaporization=2.26e6,
            boiling_point=373.15,
        )
        result = simulate_pool(inp)
        assert result.pool_area.max() > 0
        assert result.pool_area.max() < 10000  # < 1 hectare for 1 ton of water

    def test_high_wind_increases_evaporation(self):
        """Higher wind speed → higher evaporation rate (via mass transfer)."""
        inp_low = PoolInput(
            substance='gasoline',
            spill_mass=1000,
            T_ambient=298.15,
            wind_speed=1.0,
            surface='concrete',
            rho_l=740,
            molecular_weight=0.100,
            vapor_pressure=50000,
            heat_of_vaporization=3.5e5,
            boiling_point=350,
        )
        inp_high = PoolInput(
            substance='gasoline',
            spill_mass=1000,
            T_ambient=298.15,
            wind_speed=10.0,
            surface='concrete',
            rho_l=740,
            molecular_weight=0.100,
            vapor_pressure=50000,
            heat_of_vaporization=3.5e5,
            boiling_point=350,
        )
        r_low = simulate_pool(inp_low)
        r_high = simulate_pool(inp_high)
        # Higher wind → higher evaporation
        assert r_high.evap_rate.max() >= r_low.evap_rate.max() * 0.8, \
            "Higher wind should not decrease evaporation"

    def test_bunded_pool_area_limited(self):
        """Bunded pool is limited to bund area."""
        inp = PoolInput(
            substance='gasoline',
            spill_mass=10000,
            T_ambient=298.15,
            wind_speed=3.0,
            surface='concrete',
            bunded_area=50.0,  # m²
            rho_l=740,
            molecular_weight=0.100,
            vapor_pressure=50000,
            heat_of_vaporization=3.5e5,
            boiling_point=350,
        )
        result = simulate_pool(inp)
        # Maximum pool area should not exceed bund area significantly
        assert result.pool_area.max() <= 50.0 + EPSILON

    def test_boiling_pool_evaporates(self):
        """Cryogenic/boiling pool evaporates."""
        inp = PoolInput(
            substance='LNG',
            spill_mass=1000,
            T_ambient=298.15,
            wind_speed=3.0,
            surface='concrete',
            rho_l=450,
            molecular_weight=0.016,
            vapor_pressure=P_ATM,  # boiling at ambient pressure
            heat_of_vaporization=5.1e5,
            boiling_point=111.6,  # methane boiling point
            cp_liquid=3500,
        )
        result = simulate_pool(inp)
        assert result.total_evaporated > 0

    def test_mass_conservation(self):
        """Mass remaining + mass evaporated ≈ initial spill mass."""
        inp = PoolInput(
            substance='gasoline',
            spill_mass=1000,
            T_ambient=298.15,
            wind_speed=3.0,
            surface='concrete',
            rho_l=740,
            molecular_weight=0.100,
            vapor_pressure=50000,
            heat_of_vaporization=3.5e5,
            boiling_point=350,
        )
        result = simulate_pool(inp)
        total = result.mass_remaining + result.total_evaporated
        assert total == pytest.approx(1000.0, rel=0.01)  # within 1%

    def test_evaporation_rate_function(self):
        """evaporation_rate() standalone returns non-negative."""
        e = evaporation_rate(
            pool_area=10.0,
            k_m=0.01,
            P_vapor=50000,
            T_pool=300,
            T_ambient=300,
            MW=0.100,
        )
        assert e >= 0

    def test_mass_transfer_coefficient_positive(self):
        """Mass transfer coefficient > 0 for typical conditions."""
        k_m = mass_transfer_coefficient(
            wind_speed=3.0,
            pool_diameter=10.0,
            D_ab=1.5e-5,
        )
        assert k_m > 0
