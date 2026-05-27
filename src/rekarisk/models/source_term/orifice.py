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

from ...core.constants import R, P_ATM, G, EPSILON, DISCHARGE_COEFFICIENT, WATER_BOILING_POINT


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
    d_pipe: float | None = None         # pipe diameter [m] — caps d_hole to this


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

    # Cap hole diameter to pipe diameter if provided (PHAST convention)
    d_eff = inputs.d_hole
    if inputs.d_pipe is not None and inputs.d_pipe > 0:
        if d_eff > inputs.d_pipe:
            messages.append(f"Hole diameter ({d_eff*1000:.0f}mm) capped to pipe diameter ({inputs.d_pipe*1000:.0f}mm)")
            d_eff = inputs.d_pipe
    area = _hole_area(d_eff)
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


# ══════════════════════════════════════════════════════════════════════════════
# Extended Liquid & Two-Phase Release Models
# ══════════════════════════════════════════════════════════════════════════════

def estimate_boiling_point(
    T_boil_ref: float,
    P: float,
    P_ref: float = P_ATM,
    h_fg: float | None = None,
    MW: float | None = None,
) -> float:
    """Estimate boiling point at a given pressure using Clausius-Clapeyron.

    Uses the integrated Clausius-Clapeyron equation for a single-component
    liquid-vapor equilibrium, assuming constant latent heat:

        ln(P/P_ref) = -(h_fg / R_specific) * (1/T_boil - 1/T_boil_ref)

    Solving for T_boil:

        T_boil = T_boil_ref / (1 - (R_specific * T_boil_ref / h_fg) * ln(P/P_ref))

    where R_specific = R / MW is the specific gas constant [J/(kg·K)].

    This is a simplified estimation — for accurate results use measured
    boiling curves or Antoine-equation parameters.

    Args:
        T_boil_ref: Reference boiling point [K] at P_ref (e.g., 373.15 K for water).
        P: Target pressure [Pa abs] at which to estimate boiling point.
        P_ref: Reference pressure [Pa abs]. Default 101325 Pa (1 atm).
        h_fg: Latent heat of vaporization [J/kg]. If None, returns T_boil_ref.
        MW: Molecular weight [kg/mol]. If provided, uses R/MW as specific
            gas constant. If None, assumes h_fg is in J/mol and uses R directly.

    Returns:
        Estimated boiling point [K] at pressure P.

    Example:
        >>> # Water boiling point at 2 atm (~121°C)
        >>> T = estimate_boiling_point(373.15, 202650, h_fg=2.26e6, MW=0.018)
        >>> round(T, 1)
        394.3
    """
    if h_fg is None or h_fg <= EPSILON:
        return T_boil_ref
    if P <= EPSILON or P_ref <= EPSILON:
        return T_boil_ref

    # Specific gas constant [J/(kg·K)]
    R_specific = R / MW if MW is not None else R

    ln_term = math.log(P / P_ref)
    # Denominator: 1 - (R_specific * T_boil_ref * ln(P/P_ref)) / h_fg
    denom = 1.0 - (R_specific * T_boil_ref * ln_term) / h_fg

    if abs(denom) < EPSILON:
        return T_boil_ref  # degenerate — pressure ratio too extreme

    T_boil = T_boil_ref / denom
    return max(1.0, T_boil)  # force physically meaningful > 0 K


def calculate_liquid_release(
    P_up: float,
    P_down: float,
    rho_l: float,
    d_hole: float,
    Cd: float = 0.61,
) -> float:
    """Calculate liquid mass flow rate through an orifice (Bernoulli equation).

    Standard incompressible-flow orifice equation:

        mdot = Cd * A * sqrt(2 * rho_l * (P_up - P_down))

    This applies when the liquid is subcooled (below its boiling point at the
    downstream pressure) — i.e., no flashing occurs across the orifice.

    The default Cd = 0.61 is the standard value for a sharp-edged orifice.

    Args:
        P_up: Upstream (stagnation) pressure [Pa abs].
        P_down: Downstream (back) pressure [Pa abs].
        rho_l: Liquid density [kg/m³].
        d_hole: Orifice diameter [m].
        Cd: Discharge coefficient [-]. Default 0.61 (sharp-edged orifice).

    Returns:
        Mass flow rate [kg/s]. Returns 0.0 if dp <= 0.

    Example:
        >>> mdot = calculate_liquid_release(
        ...     P_up=5e5, P_down=1e5, rho_l=1000, d_hole=0.01, Cd=0.61
        ... )
        >>> round(mdot, 4)
        0.1355
    """
    dp = P_up - P_down
    if dp <= EPSILON:
        return 0.0

    area = _hole_area(d_hole)
    mdot = Cd * area * math.sqrt(2.0 * rho_l * dp)
    return mdot


def calculate_flashing_release(
    P_up: float,
    P_down: float,
    T: float,
    d_hole: float,
    rho_l: float,
    rho_g: float,
    cp_l: float,
    h_fg: float,
    MW: float,
    gamma: float = 1.3,
    Cd: float = 0.61,
    T_boil_ref: float | None = None,
) -> float:
    """Calculate two-phase flashing mass flow rate using the Omega method.

    Implements the Fauske / API 520 Part I Annex C Omega method for
    Homogeneous Equilibrium Model (HEM) two-phase flashing flow through
    an orifice. This applies when a pressurized liquid is above its
    boiling point at the downstream pressure, causing partial vaporization
    (flashing) across the orifice.

    Process:
      1. Estimate boiling point at downstream pressure (Clausius-Clapeyron).
      2. Calculate flashing quality x = cp_l * (T - T_boil) / h_fg.
      3. Estimate omega compressibility parameter from fluid properties.
      4. Determine critical pressure ratio η_c = ω / (ω + 1).
      5. Check for choked flow.
      6. Calculate dimensionless mass flux G* via omega correlation.
      7. Return mdot = Cd * A * G.

    Reference:
        Leung, J.C. (1986) AIChE J, 32(10) — Generalized Correlation for
        One-Component Homogeneous Equilibrium Flashing Choked Flow.

    Args:
        P_up: Upstream stagnation pressure [Pa abs].
        P_down: Downstream back pressure [Pa abs].
        T: Upstream temperature [K].
        d_hole: Orifice diameter [m].
        rho_l: Liquid density at stagnation conditions [kg/m³].
        rho_g: Gas density at stagnation conditions [kg/m³].
        cp_l: Liquid specific heat capacity [J/(kg·K)].
        h_fg: Latent heat of vaporization [J/kg].
        MW: Molecular weight [kg/mol].
        gamma: Specific heat ratio Cp/Cv [-]. Default 1.3.
        Cd: Discharge coefficient [-]. Default 0.61.
        T_boil_ref: Reference boiling point [K] at P_ref=101325 Pa.
            If None, defaults to WATER_BOILING_POINT (373.15 K).

    Returns:
        Mass flow rate [kg/s]. Returns 0.0 when upstream pressure does not
        exceed downstream pressure, or when temperature is below boiling point.
    """
    area = _hole_area(d_hole)

    if P_up <= P_down or rho_l <= EPSILON or rho_g <= EPSILON:
        return 0.0

    # ── Estimate boiling point at downstream pressure ──
    t_ref = T_boil_ref if T_boil_ref is not None else WATER_BOILING_POINT
    T_boil = estimate_boiling_point(
        T_boil_ref=t_ref,
        P=P_down,
        P_ref=P_ATM,
        h_fg=h_fg,
        MW=MW,
    )
    if T_boil <= EPSILON:
        return 0.0

    # If liquid is below its boiling point at downstream pressure,
    # no flashing occurs — caller should use liquid-only model.
    if T <= T_boil:
        return 0.0

    # ── Calculate flashing quality (mass fraction that vaporizes) ──
    x0 = cp_l * (T - T_boil) / h_fg
    x0 = max(0.0, min(1.0, x0))

    # ── Specific volumes ──
    v_l = 1.0 / rho_l
    v_g = 1.0 / rho_g
    v_fg = v_g - v_l        # volume change on vaporization [m³/kg]
    v_0 = x0 * v_g + (1.0 - x0) * v_l   # mixture specific volume at stagnation

    # ── Estimate omega (compressibility) parameter ──
    # ω = α₀/γ + (1 - α₀) * ω_s
    # where ω_s = cp_l * T * P_up * (v_fg / h_fg)² / v_0
    omega: float = 1.0  # default fallback

    if h_fg > EPSILON and T > 0 and P_up > 0:
        term = (v_fg / h_fg) ** 2
        omega_s = cp_l * T * P_up * term / v_0

        if x0 > EPSILON:
            # Weighted by inlet void fraction
            alpha_0 = x0 * v_g / v_0
            omega = alpha_0 / gamma + (1.0 - alpha_0) * omega_s
        else:
            omega = omega_s if omega_s > EPSILON else 1.0

    omega = max(0.1, min(100.0, omega))

    # ── Critical pressure ratio and choking check ──
    eta_c = omega / (omega + 1.0)
    P_choked = P_up * eta_c
    is_choked = P_down < P_choked
    P_exit = P_choked if is_choked else max(P_down, 0.0)
    eta = P_exit / P_up

    if eta <= EPSILON or eta >= 1.0 - EPSILON:
        return 0.0

    # ── Stagnation density (two-phase mixture) ──
    rho_0 = 1.0 / v_0 if v_0 > EPSILON else rho_l

    # ── Dimensionless mass flux G* ──
    # Leung (1986) equation:
    #   For ω ≠ 1:  G* = sqrt(-2·[ω·ln(η) + (ω-1)·(1-η)]) / [ω·(1/η - 1) + 1]
    #   For ω = 1:  G* = sqrt(2·[1 - η - η·ln(η)]) / (1/η)
    if abs(omega - 1.0) > 0.001:
        arg = -2.0 * (omega * math.log(eta) + (omega - 1.0) * (1.0 - eta))
        denom = omega * (1.0 / eta - 1.0) + 1.0
        if arg <= 0.0 or denom <= 0.0:
            return 0.0
        G_star = math.sqrt(arg) / denom
    else:
        arg = 2.0 * (1.0 - eta - eta * math.log(eta))
        denom = 1.0 / eta
        if arg <= 0.0:
            return 0.0
        G_star = math.sqrt(arg) / denom

    # ── Mass flux and mass flow rate ──
    G_flux = G_star * math.sqrt(P_up * rho_0)   # kg/(m²·s)
    mdot = Cd * area * G_flux                    # kg/s
    return mdot


def calculate_release_rate_auto(
    P_up: float,
    P_down: float,
    T: float,
    d_hole: float,
    phase: str,
    rho_l: float | None = None,
    rho_g: float | None = None,
    cp_l: float | None = None,
    h_fg: float | None = None,
    MW: float | None = None,
    gamma: float | None = None,
    Cd: float = 0.61,
    T_boil_ref: float | None = None,
) -> tuple[float, str]:
    """Smart dispatcher — selects the correct release model based on phase.

    Routing logic:

    +--------------+-----------------------------------+--------------------+
    | phase        | Condition                         | Model used         |
    +==============+===================================+====================+
    | ``"gas"``    | —                                 | Gas orifice        |
    |              |                                   | (choked/subsonic)  |
    +--------------+-----------------------------------+--------------------+
    | ``"liquid"`` | T ≤ T_boil(P_down)                | Liquid Bernoulli   |
    +--------------+-----------------------------------+--------------------+
    | ``"liquid"`` | T > T_boil(P_down)                | Two-phase flashing |
    |              |                                   | (Omega method)     |
    +--------------+-----------------------------------+--------------------+

    Args:
        P_up: Upstream stagnation pressure [Pa abs].
        P_down: Downstream back pressure [Pa abs].
        T: Upstream temperature [K].
        d_hole: Orifice diameter [m].
        phase: Release phase: ``"gas"`` or ``"liquid"``.
        rho_l: Liquid density [kg/m³] (required for liquid phase).
        rho_g: Gas density at upstream [kg/m³] (required for gas phase;
            also used for two-phase flashing).
        cp_l: Liquid specific heat [J/(kg·K)] (for two-phase flashing).
        h_fg: Latent heat of vaporization [J/kg] (for two-phase flashing).
        MW: Molecular weight [kg/mol] (required for gas; for two-phase).
        gamma: Specific heat ratio Cp/Cv [-] (required for gas).
        Cd: Discharge coefficient [-]. Default 0.61.
        T_boil_ref: Reference boiling point [K] at P_ref=101325 Pa.
            If None, estimated via Clausius-Clapeyron using water reference.

    Returns:
        Tuple of ``(mass_flow_rate [kg/s], flow_type)`` where
        ``flow_type`` is one of:

        - ``"gas_choked"`` — gas flow at sonic velocity
        - ``"gas_subsonic"`` — gas flow below sonic velocity
        - ``"liquid"`` — incompressible liquid Bernoulli flow
        - ``"two_phase"`` — two-phase flashing flow (Omega method)

    Raises:
        ValueError: If required fluid properties are missing for the
            selected phase.

    Example:
        >>> # Subcooled water release:
        >>> mdot, ftype = calculate_release_rate_auto(
        ...     P_up=5e5, P_down=1e5, T=320, d_hole=0.01,
        ...     phase="liquid", rho_l=988, rho_g=0.6,
        ...     cp_l=4180, h_fg=2.26e6, MW=0.018, gamma=1.33,
        ...     T_boil_ref=373.15
        ... )
        >>> ftype
        'liquid'
    """
    area = _hole_area(d_hole)

    # ── Gas phase ──
    if phase == "gas":
        if rho_g is None or gamma is None or MW is None:
            raise ValueError(
                "Gas phase requires rho_g, gamma, and MW parameters."
            )
        k = gamma
        result = gas_orifice_discharge(
            Cd=Cd,
            area=area,
            P_up=P_up,
            P_down=P_down,
            k=k,
            T=T,
            MW=MW,
        )
        flow_type = "gas_choked" if result["is_choked"] else "gas_subsonic"
        return result["mdot"], flow_type

    # ── Liquid phase ──
    if phase == "liquid":
        if rho_l is None:
            raise ValueError("Liquid phase requires rho_l parameter.")

        # Determine boiling point at downstream pressure.
        # If T_boil_ref not given and we have the properties, estimate it.
        T_boil: float
        if T_boil_ref is not None and h_fg is not None and MW is not None:
            T_boil = estimate_boiling_point(
                T_boil_ref=T_boil_ref,
                P=P_down,
                P_ref=P_ATM,
                h_fg=h_fg,
                MW=MW,
            )
        elif h_fg is not None and MW is not None:
            # No explicit T_boil_ref — use water as reference
            T_boil = estimate_boiling_point(
                T_boil_ref=WATER_BOILING_POINT,
                P=P_down,
                P_ref=P_ATM,
                h_fg=h_fg,
                MW=MW,
            )
        else:
            # Cannot estimate boiling point without h_fg and MW.
            # Assume non-flashing; use pure liquid Bernoulli.
            mdot = calculate_liquid_release(
                P_up=P_up,
                P_down=P_down,
                rho_l=rho_l,
                d_hole=d_hole,
                Cd=Cd,
            )
            return mdot, "liquid"

        # ── Decision: liquid-only vs two-phase flashing ──
        if T <= T_boil:
            # Subcooled — no flashing occurs
            mdot = calculate_liquid_release(
                P_up=P_up,
                P_down=P_down,
                rho_l=rho_l,
                d_hole=d_hole,
                Cd=Cd,
            )
            return mdot, "liquid"
        else:
            # Superheated — two-phase flashing flow
            if rho_g is None or cp_l is None or h_fg is None or MW is None:
                # Missing flashing parameters — degrade gracefully
                mdot = calculate_liquid_release(
                    P_up=P_up,
                    P_down=P_down,
                    rho_l=rho_l,
                    d_hole=d_hole,
                    Cd=Cd,
                )
                return mdot, "liquid"

            mdot = calculate_flashing_release(
                P_up=P_up,
                P_down=P_down,
                T=T,
                d_hole=d_hole,
                rho_l=rho_l,
                rho_g=rho_g,
                cp_l=cp_l,
                h_fg=h_fg,
                MW=MW,
                gamma=gamma if gamma is not None else 1.3,
                Cd=Cd,
                T_boil_ref=T_boil_ref,
            )
            return mdot, "two_phase"

    raise ValueError(
        f"Unknown phase '{phase}'. Use 'gas' or 'liquid'."
    )
