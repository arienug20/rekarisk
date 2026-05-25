"""
Rekarisk — Orifice Discharge Models.

Calculates mass flow rate through an orifice (hole, leak, rupture)
for liquid, gas/vapor, and two-phase releases.

References:
  - CCPS Guidelines for Consequence Analysis of Chemical Releases (1999)
  - API 520 Part I, Annex C — Omega Method for Two-Phase Flow
  - TNO Yellow Book (CPR 14E), Chapter 2
  - Leung, J.C. (1986) — A Generalized Correlation for One-Component
    Homogeneous Equilibrium Flashing Choked Flow, AIChE J, 32(10)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ...core.constants import R, P_ATM, G, EPSILON, DISCHARGE_COEFFICIENT


# ══════════════════════════════════════════════════════════════════════════════
# Enums & Constants
# ══════════════════════════════════════════════════════════════════════════════

class ReleasePhase(str, Enum):
    """Phase of the released material."""
    LIQUID = "liquid"
    GAS = "gas"
    VAPOR = "vapor"
    TWO_PHASE = "two_phase"
    FLASHING = "flashing"


class FlowRegime(str, Enum):
    """Flow regime classification."""
    SUBSONIC = "subsonic"
    SONIC = "sonic"
    CHOKED = "choked"


# ══════════════════════════════════════════════════════════════════════════════
# Input/Output Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrificeInput:
    """Input parameters for orifice discharge calculation.

    Attributes:
        Cd: Discharge coefficient [dimensionless] (0.6-1.0).
        d_hole: Orifice/hole diameter [m].
        P_upstream: Upstream (stagnation) pressure [Pa abs].
        P_downstream: Downstream (back) pressure [Pa abs].
        T: Upstream temperature [K].
        phase: Expected release phase ('liquid', 'gas', 'two_phase', 'auto').
            Use 'auto' to determine from fluid state.
        rho: Liquid density [kg/m³] (required for liquid/two-phase).
        rho_gas: Gas density [kg/m³] at upstream conditions (required for gas).
        molecular_weight: Molecular weight [kg/mol] or None (use 0.029 for air-like).
        cp_cv_ratio: Specific heat ratio k = Cp/Cv (required for gas).
        h_liquid_head: Liquid head height above orifice [m] (default 0).
        heat_of_vaporization: Latent heat [J/kg] (for two-phase flashing).
        cp_liquid: Liquid specific heat [J/(kg·K)] (for flashing fraction).
        T_boiling: Boiling point at P_downstream [K] (for flashing fraction).
        omega: Two-phase compressibility parameter [-] (optional, auto-computed).
    """
    Cd: float
    d_hole: float
    P_upstream: float
    P_downstream: float
    T: float
    phase: str = "auto"

    # Fluid properties
    rho: float | None = None           # liquid density [kg/m³]
    rho_gas: float | None = None       # gas density at upstream [kg/m³]
    molecular_weight: float | None = None  # [kg/mol]
    cp_cv_ratio: float | None = None   # k = Cp/Cv [-]

    # Additional for liquid
    h_liquid_head: float = 0.0          # liquid head above orifice [m]

    # Additional for two-phase / flashing
    heat_of_vaporization: float | None = None  # [J/kg]
    cp_liquid: float | None = None      # [J/(kg·K)]
    T_boiling: float | None = None      # boiling point at downstream P [K]
    omega: float | None = None          # compressibility parameter


@dataclass
class OrificeResult:
    """Results from orifice discharge calculation.

    Attributes:
        mdot_initial: Initial mass flow rate [kg/s].
        phase: Release phase determined/used.
        velocity: Exit velocity [m/s].
        P_choked: Critical (choked) pressure [Pa abs] (None for liquid).
        P_exit: Actual exit plane pressure [Pa abs].
        is_choked: Whether flow is choked (sonic).
        flow_regime: Subsonic, sonic, or choked.
        total_mass: For given duration — None unless duration provided.
        flashing_fraction: Mass fraction that flashes (two-phase only).
        area: Orifice cross-sectional area [m²].
        G: Mass flux [kg/(m²·s)].
        messages: List of warning/info strings.
    """
    mdot_initial: float
    phase: str
    velocity: float
    P_choked: float | None
    P_exit: float
    is_choked: bool
    flow_regime: str
    total_mass: float | None = None
    flashing_fraction: float = 0.0
    area: float = 0.0
    G: float = 0.0
    messages: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Core Calculation Functions
# ══════════════════════════════════════════════════════════════════════════════

def _hole_area(d: float) -> float:
    """Calculate circular orifice area [m²]."""
    return math.pi * (d / 2.0) ** 2


def calculate_choked_pressure(P_up: float, k: float) -> float:
    """Calculate critical (choked) pressure ratio for gas flow.

    P_choked = P_upstream * (2/(k+1))^(k/(k-1))

    Args:
        P_up: Upstream pressure [Pa abs].
        k: Specific heat ratio Cp/Cv [-].

    Returns:
        Critical downstream pressure for choking [Pa abs].
    """
    if k <= 1.0:
        return P_up * 0.5  # fallback
    ratio = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
    return P_up * ratio


def liquid_orifice_discharge(
    Cd: float,
    area: float,
    rho: float,
    dp: float,
    h_head: float = 0.0,
) -> tuple[float, float]:
    """Calculate liquid mass flow through an orifice (Bernoulli).

    Args:
        Cd: Discharge coefficient [-].
        area: Orifice cross-sectional area [m²].
        rho: Liquid density [kg/m³].
        dp: Pressure difference P1 - P2 [Pa].
        h_head: Liquid head height above orifice [m].

    Returns:
        Tuple of (mass_flow_rate [kg/s], exit_velocity [m/s]).
    """
    # Bernoulli: velocity from pressure + gravity head
    head_term = 2.0 * G * h_head if h_head > EPSILON else 0.0
    v_sq = 2.0 * dp / rho + head_term
    if v_sq <= EPSILON:
        return 0.0, 0.0
    u = math.sqrt(v_sq)
    mdot = Cd * area * rho * u
    return mdot, u


def gas_orifice_discharge(
    Cd: float,
    area: float,
    P_up: float,
    P_down: float,
    k: float,
    T: float,
    MW: float,
) -> dict:
    """Calculate gas/vapor mass flow through an orifice.

    Handles choked (sonic) and subsonic flow regimes.

    For choked flow (P_down / P_up <= r_crit):
        mdot = Cd * A * P_up * sqrt( k*MW / (R*T) * (2/(k+1))^((k+1)/(k-1)) )

    For subsonic flow:
        mdot = Cd * A * P_up * sqrt( (2*k/(k-1)) * (MW/(R*T)) *
               ((P_down/P_up)^(2/k) - (P_down/P_up)^((k+1)/k)) )

    Args:
        Cd: Discharge coefficient [-].
        area: Orifice area [m²].
        P_up: Upstream stagnation pressure [Pa abs].
        P_down: Downstream back pressure [Pa abs].
        k: Specific heat ratio Cp/Cv [-].
        T: Upstream temperature [K].
        MW: Molecular weight [kg/mol].

    Returns:
        Dict with keys: mdot [kg/s], velocity [m/s], is_choked, P_choked [Pa],
        P_exit [Pa], flow_regime.
    """
    # Critical pressure ratio
    if k <= 1.0 + EPSILON:
        k = 1.001  # avoid singularity
    r_crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
    P_choked = P_up * r_crit

    # Ensure downstream pressure is not below absolute zero
    P_down_eff = max(P_down, 0.0)

    is_choked = P_down_eff < P_choked
    P_exit = P_choked if is_choked else P_down_eff

    if is_choked:
        # Choked (sonic) flow — mass flux independent of downstream pressure
        exponent = (k + 1.0) / (k - 1.0)
        rhs = k * (MW / (R * T)) * (2.0 / (k + 1.0)) ** exponent
        if rhs <= EPSILON:
            return {"mdot": 0.0, "velocity": 0.0, "is_choked": True,
                    "P_choked": P_choked, "P_exit": P_exit, "flow_regime": FlowRegime.CHOKED.value}
        G_flux = P_up * math.sqrt(rhs)  # mass flux through orifice [kg/(m²·s)]
        mdot = Cd * area * G_flux

        # Sonic velocity at throat
        R_specific = R / MW  # J/(kg·K)
        T_throat = T * (2.0 / (k + 1.0))
        vel = math.sqrt(k * R_specific * T_throat)
    else:
        # Subsonic flow
        pr = P_down_eff / P_up
        if pr <= EPSILON or pr >= 1.0 - EPSILON:
            return {"mdot": 0.0, "velocity": 0.0, "is_choked": False,
                    "P_choked": P_choked, "P_exit": P_exit, "flow_regime": FlowRegime.SUBSONIC.value}

        ratio_term = pr ** (2.0 / k) - pr ** ((k + 1.0) / k)
        if ratio_term <= EPSILON:
            return {"mdot": 0.0, "velocity": 0.0, "is_choked": False,
                    "P_choked": P_choked, "P_exit": P_exit, "flow_regime": FlowRegime.SUBSONIC.value}

        factor = (2.0 * k / (k - 1.0)) * (MW / (R * T))
        G_flux = P_up * math.sqrt(factor * ratio_term)
        mdot = Cd * area * G_flux

        # Exit velocity from energy balance
        R_specific = R / MW
        vel = math.sqrt(2.0 * k * R_specific * T / (k - 1.0) *
                        (1.0 - pr ** ((k - 1.0) / k)))

    return {
        "mdot": mdot,
        "velocity": vel,
        "is_choked": is_choked,
        "P_choked": P_choked,
        "P_exit": P_exit,
        "flow_regime": FlowRegime.CHOKED.value if is_choked else FlowRegime.SUBSONIC.value,
    }


def calculate_flashing_fraction(
    T: float,
    T_boil: float,
    cp_liquid: float,
    hfg: float,
) -> float:
    """Calculate the mass fraction that flashes upon depressurization.

    Uses energy balance: fraction of liquid vaporized to cool remaining
    liquid to its boiling point at the new pressure.

    x = cp_liquid * (T - T_boil) / hfg

    Args:
        T: Initial liquid temperature [K].
        T_boil: Boiling point at downstream pressure [K].
        cp_liquid: Liquid specific heat [J/(kg·K)].
        hfg: Latent heat of vaporization [J/kg].

    Returns:
        Flashing mass fraction [0-1].
    """
    if T <= T_boil:
        return 0.0
    if hfg <= EPSILON:
        return 0.0
    x = cp_liquid * (T - T_boil) / hfg
    return max(0.0, min(1.0, x))


def estimate_omega_parameter(
    x0: float,
    k: float,
    rho_l: float,
    rho_g: float,
    cp_liquid: float | None = None,
    T: float | None = None,
    P: float | None = None,
    hfg: float | None = None,
) -> float:
    """Estimate the omega (compressibility) parameter for two-phase flow.

    For saturated liquid entry (x0=0):
        ω = Cp_l * T * P_s * (ν_fg / h_fg)² / ν_l * (ρ_l / ρ)

    Simplified correlation from API 520 Annex C / Leung (1986):
        ω = α0/k + (1-α0) * ω_s
    where ω_s is for saturated conditions.

    For practical applications, direct estimate:
        ω = x0 * ν_g0 / (κ * ν_0) + cp_l * T * P * ν_0 * (ν_fg / h_fg)²

    Args:
        x0: Inlet mass quality (vapor mass fraction) [-].
        k: Specific heat ratio Cp/Cv [-].
        rho_l: Liquid density [kg/m³] at stagnation.
        rho_g: Gas density [kg/m³] at stagnation.
        cp_liquid: Liquid specific heat [J/(kg·K)].
        T: Temperature [K].
        P: Pressure [Pa].
        hfg: Latent heat of vaporization [J/kg].

    Returns:
        Omega parameter [-].
    """
    if rho_l <= EPSILON or rho_g <= EPSILON:
        return 1.0  # default

    # Specific volumes
    v_l = 1.0 / rho_l
    v_g = 1.0 / rho_g
    v_fg = v_g - v_l  # specific volume change on vaporization

    # Mixture specific volume at stagnation
    v_0 = x0 * v_g + (1.0 - x0) * v_l

    # Omega for subcooled / saturated conditions
    omega_s = 0.0
    if cp_liquid and hfg and T and P and hfg > EPSILON and T > 0 and P > 0:
        # ω_s term from Leung (1986), equation for saturated conditions:
        # ω_s = cp_l * T * P * (v_fg / h_fg)² / v_0
        term = (v_fg / hfg) ** 2
        omega_s = cp_liquid * T * P * term / v_0

    # Combined omega (accounting for inlet quality)
    # For pure liquid at saturation: x0 = 0 → ω ≈ ω_s
    # For pure gas: x0 = 1 → ω ≈ 1/k
    if x0 > EPSILON:
        # Weighted by phase fractions
        alpha_0 = x0 * v_g / v_0  # inlet void fraction
        omega = alpha_0 / k + (1.0 - alpha_0) * omega_s
    else:
        omega = omega_s if omega_s > EPSILON else 1.0

    # Clamp to physically reasonable range
    return max(0.1, min(100.0, omega))


def two_phase_orifice_discharge(
    Cd: float,
    area: float,
    P_up: float,
    P_down: float,
    rho_0: float,
    omega: float,
) -> dict:
    """Calculate two-phase mass flow through an orifice using the Omega method.

    Implements the HEM (Homogeneous Equilibrium Model) via Leung's Omega method
    as described in API 520 Part I Annex C.

    For two-phase choked flow, the critical pressure ratio is:
        η_c = ω / (ω + 1)
    and the dimensionless mass flux is:
        G* = sqrt( -2 * (ω*ln(η_c) + (ω-1)*(1-η_c)) ) /
             ( ω * (1/η_c - 1) + 1 )

    Args:
        Cd: Discharge coefficient [-].
        area: Orifice area [m²].
        P_up: Upstream stagnation pressure [Pa abs].
        P_down: Downstream back pressure [Pa abs].
        rho_0: Stagnation density (two-phase mixture) [kg/m³].
        omega: Two-phase compressibility parameter [-].

    Returns:
        Dict with keys: mdot [kg/s], velocity [m/s], is_choked, P_choked [Pa],
        P_exit [Pa], flow_regime.
    """
    if omega <= EPSILON or P_up <= EPSILON:
        return {"mdot": 0.0, "velocity": 0.0, "is_choked": False,
                "P_choked": P_down, "P_exit": P_down,
                "flow_regime": FlowRegime.SUBSONIC.value}

    # Critical pressure ratio for omega method
    if omega > EPSILON:
        eta_c = omega / (omega + 1.0)
    else:
        eta_c = 0.5  # fallback

    P_choked = P_up * eta_c
    P_down_eff = max(P_down, 0.0)
    is_choked = P_down_eff < P_choked
    P_exit = P_choked if is_choked else P_down_eff
    eta = P_exit / P_up

    if eta <= EPSILON or eta >= 1.0 - EPSILON:
        return {"mdot": 0.0, "velocity": 0.0, "is_choked": is_choked,
                "P_choked": P_choked, "P_exit": P_exit,
                "flow_regime": FlowRegime.CHOKED.value if is_choked else FlowRegime.SUBSONIC.value}

    # Compute dimensionless mass flux G* using the omega correlation
    # G* = (G / sqrt(P_up * rho_0)) / sqrt(2)
    # The full omega method equation:
    # For omega != 1:
    if abs(omega - 1.0) > 0.001:
        # Chi = 1 - eta
        # G* = sqrt( -2/(omega^2) * [omega*ln(eta) + (omega-1)*(1-eta)] /
        #           ( (1/eta - 1) + 1/omega ) )
        # Better: use standard formulation from CCPS:
        term1 = -2.0 * (omega * math.log(eta) + (omega - 1.0) * (1.0 - eta))
        denom = omega * (1.0 / eta - 1.0) + 1.0
        if term1 <= 0.0 or denom <= 0.0:
            return {"mdot": 0.0, "velocity": 0.0, "is_choked": is_choked,
                    "P_choked": P_choked, "P_exit": P_exit,
                    "flow_regime": FlowRegime.CHOKED.value if is_choked else FlowRegime.SUBSONIC.value}
        G_star = math.sqrt(term1) / denom
    else:
        # For omega = 1: G* = sqrt(2 * (1 - eta - eta*ln(eta))) / (1/eta - 1 + 1)
        term1 = 2.0 * (1.0 - eta - eta * math.log(eta))
        denom = (1.0 / eta)
        if term1 <= 0.0:
            return {"mdot": 0.0, "velocity": 0.0, "is_choked": is_choked,
                    "P_choked": P_choked, "P_exit": P_exit,
                    "flow_regime": FlowRegime.CHOKED.value if is_choked else FlowRegime.SUBSONIC.value}
        G_star = math.sqrt(term1) / denom

    # Mass flux
    G_flux = G_star * math.sqrt(P_up * rho_0)  # kg/(m²·s)
    mdot = Cd * area * G_flux

    # Exit velocity from continuity: u = G / rho_exit
    # For homogeneous flow: rho_exit ≈ rho_0 / (x_exit * v_g/v_l + ...)
    # Simplified: use G_flux / rho_0 as approximate
    vel = G_flux / rho_0 if rho_0 > EPSILON else 0.0

    return {
        "mdot": mdot,
        "velocity": vel,
        "is_choked": is_choked,
        "P_choked": P_choked,
        "P_exit": P_exit,
        "flow_regime": FlowRegime.CHOKED.value if is_choked else FlowRegime.SUBSONIC.value,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def calculate_orifice(inputs: OrificeInput,
                      duration: float | None = None) -> OrificeResult:
    """Calculate mass flow rate through an orifice for any phase.

    This is the main entry point for orifice discharge calculations.
    Dispatches to the appropriate sub-model based on phase.

    Args:
        inputs: OrificeInput dataclass with all required parameters.
        duration: Release duration [s]. If provided, computes total_mass.

    Returns:
        OrificeResult with mass flow rate, velocity, phase, etc.

    Raises:
        ValueError: If required parameters are missing for the selected phase.

    Example:
        >>> inp = OrificeInput(Cd=0.62, d_hole=0.05, P_up=500000, P_down=101325,
        ...                    T=300, phase='gas', rho_gas=5.8, cp_cv_ratio=1.3,
        ...                    molecular_weight=0.016)
        >>> result = calculate_orifice(inp)
        >>> print(f"Mass flow: {result.mdot_initial:.3f} kg/s")
    """
    messages = []
    area = _hole_area(inputs.d_hole)
    dp = inputs.P_upstream - max(inputs.P_downstream, 0.0)
    phase = inputs.phase.lower()

    # Auto-detect phase if requested
    if phase == "auto":
        if inputs.rho_gas and inputs.cp_cv_ratio and inputs.molecular_weight:
            phase = "gas"
        elif inputs.rho:
            phase = "liquid"
        else:
            raise ValueError("Cannot auto-detect phase — provide rho (liquid) "
                             "or rho_gas + cp_cv_ratio + molecular_weight (gas).")

    # ── Liquid Discharge ──
    if phase in ("liquid",):
        if inputs.rho is None:
            raise ValueError("Liquid density (rho) is required for liquid discharge.")
        mdot, vel = liquid_orifice_discharge(
            inputs.Cd, area, inputs.rho, dp, inputs.h_liquid_head
        )
        return OrificeResult(
            mdot_initial=mdot,
            phase=ReleasePhase.LIQUID.value,
            velocity=vel,
            P_choked=None,
            P_exit=inputs.P_downstream,
            is_choked=False,
            flow_regime=FlowRegime.SUBSONIC.value,
            total_mass=mdot * duration if duration is not None else None,
            area=area,
            G=mdot / area if area > EPSILON else 0.0,
            messages=messages,
        )

    # ── Gas/Vapor Discharge ──
    if phase in ("gas", "vapor"):
        if inputs.cp_cv_ratio is None:
            raise ValueError("cp_cv_ratio (k = Cp/Cv) is required for gas/vapor discharge.")
        if inputs.molecular_weight is None:
            raise ValueError("molecular_weight [kg/mol] is required for gas/vapor discharge.")

        k = inputs.cp_cv_ratio
        MW = inputs.molecular_weight

        # If rho_gas not given, estimate from ideal gas law
        rho_gas = inputs.rho_gas
        if rho_gas is None:
            rho_gas = (inputs.P_upstream * MW) / (R * inputs.T)

        result = gas_orifice_discharge(
            Cd=inputs.Cd,
            area=area,
            P_up=inputs.P_upstream,
            P_down=inputs.P_downstream,
            k=k,
            T=inputs.T,
            MW=MW,
        )
        return OrificeResult(
            mdot_initial=result["mdot"],
            phase=ReleasePhase.GAS.value,
            velocity=result["velocity"],
            P_choked=result["P_choked"],
            P_exit=result["P_exit"],
            is_choked=result["is_choked"],
            flow_regime=result["flow_regime"],
            total_mass=result["mdot"] * duration if duration is not None else None,
            area=area,
            G=result["mdot"] / area if area > EPSILON else 0.0,
            messages=messages,
        )

    # ── Two-Phase Discharge ──
    if phase in ("two_phase", "flashing"):
        if inputs.rho is None:
            raise ValueError("Liquid density (rho) is required for two-phase flow.")

        # Calculate flashing fraction
        x_flash = 0.0
        if inputs.T_boiling and inputs.cp_liquid and inputs.heat_of_vaporization:
            x_flash = calculate_flashing_fraction(
                inputs.T, inputs.T_boiling, inputs.cp_liquid,
                inputs.heat_of_vaporization
            )

        # Calculate omega if not provided
        omega = inputs.omega
        if omega is None:
            # Need gas density at stagnation for omega
            rho_g = inputs.rho_gas
            if rho_g is None and inputs.molecular_weight:
                rho_g = (inputs.P_upstream * inputs.molecular_weight) / (R * inputs.T)
            elif rho_g is None:
                rho_g = 1.0  # fallback

            k_default = inputs.cp_cv_ratio if inputs.cp_cv_ratio else 1.3
            omega = estimate_omega_parameter(
                x0=x_flash,
                k=k_default,
                rho_l=inputs.rho,
                rho_g=rho_g,
                cp_liquid=inputs.cp_liquid,
                T=inputs.T,
                P=inputs.P_upstream,
                hfg=inputs.heat_of_vaporization,
            )

        # Stagnation density (two-phase mixture)
        if inputs.rho_gas and inputs.rho_gas > EPSILON:
            v_g = 1.0 / inputs.rho_gas
            v_l = 1.0 / inputs.rho
            v_mix = x_flash * v_g + (1.0 - x_flash) * v_l
            rho_mix = 1.0 / v_mix if v_mix > EPSILON else inputs.rho
        else:
            rho_mix = inputs.rho

        result = two_phase_orifice_discharge(
            Cd=inputs.Cd,
            area=area,
            P_up=inputs.P_upstream,
            P_down=inputs.P_downstream,
            rho_0=rho_mix,
            omega=omega,
        )

        # Estimate exit velocity better for two-phase
        if result["mdot"] > EPSILON and area > EPSILON:
            vel = result["mdot"] / (area * rho_mix)
        else:
            vel = result["velocity"]

        return OrificeResult(
            mdot_initial=result["mdot"],
            phase=(ReleasePhase.FLASHING.value if x_flash > 0.01
                   else ReleasePhase.TWO_PHASE.value),
            velocity=vel,
            P_choked=result["P_choked"],
            P_exit=result["P_exit"],
            is_choked=result["is_choked"],
            flow_regime=result["flow_regime"],
            total_mass=result["mdot"] * duration if duration is not None else None,
            flashing_fraction=x_flash,
            area=area,
            G=result["mdot"] / area if area > EPSILON else 0.0,
            messages=messages,
        )

    raise ValueError(f"Unknown phase '{phase}'. Use 'liquid', 'gas', 'two_phase', or 'auto'.")


# ══════════════════════════════════════════════════════════════════════════════
# Quick Convenience Functions
# ══════════════════════════════════════════════════════════════════════════════

def quick_liquid_orifice(
    Cd: float,
    d_hole: float,
    P_upstream: float,
    P_downstream: float,
    rho: float,
    h_head: float = 0.0,
    duration: float | None = None,
) -> OrificeResult:
    """Quick liquid orifice calculation with minimal arguments.

    Args:
        Cd: Discharge coefficient [-].
        d_hole: Orifice diameter [m].
        P_upstream: Upstream pressure [Pa abs].
        P_downstream: Downstream pressure [Pa abs].
        rho: Liquid density [kg/m³].
        h_head: Liquid head above orifice [m].
        duration: Release duration [s].

    Returns:
        OrificeResult.
    """
    inp = OrificeInput(
        Cd=Cd, d_hole=d_hole,
        P_upstream=P_upstream, P_downstream=P_downstream,
        T=298.15, phase="liquid", rho=rho,
        h_liquid_head=h_head,
    )
    return calculate_orifice(inp, duration)


def quick_gas_orifice(
    Cd: float,
    d_hole: float,
    P_upstream: float,
    P_downstream: float,
    T: float,
    k: float,
    MW: float,
    duration: float | None = None,
) -> OrificeResult:
    """Quick gas orifice calculation with minimal arguments.

    Args:
        Cd: Discharge coefficient [-].
        d_hole: Orifice diameter [m].
        P_upstream: Upstream pressure [Pa abs].
        P_downstream: Downstream pressure [Pa abs].
        T: Temperature [K].
        k: Specific heat ratio Cp/Cv [-].
        MW: Molecular weight [kg/mol].
        duration: Release duration [s].

    Returns:
        OrificeResult.
    """
    inp = OrificeInput(
        Cd=Cd, d_hole=d_hole,
        P_upstream=P_upstream, P_downstream=P_downstream,
        T=T, phase="gas", cp_cv_ratio=k,
        molecular_weight=MW,
    )
    return calculate_orifice(inp, duration)
