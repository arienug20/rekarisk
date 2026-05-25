"""
Rekarisk — Two-Phase Flow Models.

Implements models for two-phase flow through restrictions (orifices, nozzles,
pipes) used in source term calculations and relief valve sizing.

Models:
  - HEM (Homogeneous Equilibrium Model): equal phase velocities, thermal equilibrium
  - Fauske slip model: non-equilibrium with empirical slip ratio
  - Moody slip model: annular flow slip correlation
  - Omega method (API 520 Part I Annex C): two-phase mass flux
  - Flash fraction: mass fraction that vaporizes across a pressure drop

References:
  - API 520 Part I, Annex C — Sizing for Two-Phase Liquid/Vapor Relief
  - Leung, J.C. (1986), AIChE J, 32(10), 1743-1746
  - Fauske, H.K. (1965), Reactor & Fuel-Processing Tech 1965
  - Moody, F.J. (1965), J. Heat Transfer, 87(1), 134-142
  - CCPS Guidelines for Pressure Relief and Effluent Handling (1998)
  - TNO Yellow Book (CPR 14E), Chapter 2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from ...core.constants import R, P_ATM, EPSILON, G as GRAVITY


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class TwoPhaseModel(str, Enum):
    """Two-phase flow model type."""
    HEM = "hem"               # Homogeneous Equilibrium Model
    FAUSKE = "fauske"         # Fauske slip model
    MOODY = "moody"           # Moody slip model
    OMEGA = "omega"           # Omega method (API 520 Annex C)


class TwoPhaseRegime(str, Enum):
    """Two-phase flow regime."""
    BUBBLY = "bubbly"
    SLUG = "slug"
    CHURN = "churn"
    ANNULAR = "annular"
    MIST = "mist"
    FROZEN = "frozen"         # no phase change (frozen flow)
    EQUILIBRIUM = "equilibrium"


# ══════════════════════════════════════════════════════════════════════════════
# Input/Output Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TwoPhaseInput:
    """Input parameters for two-phase flow calculation.

    Attributes:
        P_stagnation: Stagnation (vessel) pressure [Pa abs].
        T: Stagnation temperature [K].
        composition: Fluid identifier (for reference).
        P_backpressure: Downstream (back) pressure [Pa abs].
        L_over_D: Pipe length-to-diameter ratio [-] (0 for orifice/nozzle).
        rho_l: Liquid density [kg/m³] at stagnation conditions.
        rho_g: Gas/vapor density [kg/m³] at stagnation conditions.
        cp_l: Liquid specific heat [J/(kg·K)].
        cp_g: Gas specific heat [J/(kg·K)].
        cp_cv_ratio: Specific heat ratio k = Cp/Cv [-] (gas phase).
        heat_of_vaporization: Latent heat [J/kg].
        surface_tension: Liquid surface tension [N/m] (optional).
        mu_l: Liquid dynamic viscosity [Pa·s] (optional).
        mu_g: Gas dynamic viscosity [Pa·s] (optional).
        x0: Inlet mass quality (vapor mass fraction) [-] (0 for sat liquid).
        model: Two-phase model to use ('hem', 'fauske', 'moody', 'omega').
    """
    P_stagnation: float
    T: float
    composition: str = "water"
    P_backpressure: float = 101325.0
    L_over_D: float = 0.0

    # Fluid properties
    rho_l: float = 1000.0
    rho_g: float = 1.0
    cp_l: float = 4184.0
    cp_g: float = 2000.0
    cp_cv_ratio: float = 1.3
    heat_of_vaporization: float = 2.26e6
    surface_tension: float | None = None
    mu_l: float | None = None
    mu_g: float | None = None
    x0: float = 0.0

    # Model selection
    model: str = "hem"


@dataclass
class TwoPhaseResult:
    """Results from two-phase flow calculation.

    Attributes:
        G_critical: Critical (maximum) mass flux [kg/(m²·s)].
        G_actual: Actual mass flux at backpressure [kg/(m²·s)].
        x_mass: Exit mass quality (vapor fraction) [-].
        eta_critical: Critical pressure ratio P_choked/P_stag [-].
        regime: Flow regime description.
        omega: Compressibility parameter [-] (omega method).
        slip_ratio: Slip ratio u_g/u_l [-] (for slip models).
        void_fraction: Exit void fraction [-].
        is_choked: Whether flow is choked.
        messages: Info/warning strings.
    """
    G_critical: float
    G_actual: float
    x_mass: float
    eta_critical: float
    regime: str
    omega: float = 1.0
    slip_ratio: float = 1.0
    void_fraction: float = 0.0
    is_choked: bool = False
    messages: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Flash Fraction
# ══════════════════════════════════════════════════════════════════════════════

def calculate_flash_fraction(
    T_stag: float,
    T_sat_at_backpressure: float,
    cp_liquid: float,
    h_fg: float,
) -> float:
    """Calculate the mass fraction that flashes when pressure drops.

    Energy balance: sensible heat of liquid above boiling point at new
    pressure provides latent heat for vaporization.

    x = cp_l * (T_stag - T_sat) / h_fg

    Args:
        T_stag: Stagnation (initial) temperature [K].
        T_sat_at_backpressure: Saturation temperature at backpressure [K].
        cp_liquid: Liquid specific heat [J/(kg·K)].
        h_fg: Latent heat of vaporization at backpressure [J/kg].

    Returns:
        Flashing mass fraction [0-1].
    """
    if T_stag <= T_sat_at_backpressure or h_fg <= EPSILON:
        return 0.0
    x = cp_liquid * (T_stag - T_sat_at_backpressure) / h_fg
    return max(0.0, min(1.0, x))


def calculate_saturated_flash_fraction(
    P_stag: float,
    P_back: float,
    cp_liquid: float,
    h_fg: float,
    T_stag: float,
    omega: float | None = None,
) -> float:
    """Calculate flash fraction for saturated liquid undergoing depressurization.

    For saturated liquid, the omega parameter determines the flashing behavior.
    For low omega (<< 1), there is little flashing. For high omega (>> 1),
    flashing is substantial.

    Simplified approach: use the fluid's omega parameter.

    Args:
        P_stag: Stagnation pressure [Pa].
        P_back: Back pressure [Pa].
        cp_liquid: Liquid specific heat [J/(kg·K)].
        h_fg: Latent heat of vaporization [J/kg].
        T_stag: Stagnation temperature [K].
        omega: Two-phase compressibility parameter (auto-computed if None).

    Returns:
        Flashing fraction [-].
    """
    if P_stag <= P_back or h_fg <= EPSILON:
        return 0.0

    eta = P_back / P_stag
    if omega is None:
        # Rough estimate
        omega_est = cp_liquid * T_stag * P_back / (h_fg * h_fg) * (1.0 / 1.0 - 1.0 / 0.001)
        omega_est = abs(omega_est)  # crude
        omega_use = max(0.1, min(10.0, omega_est))
    else:
        omega_use = omega

    # For saturated liquid (x0=0), flash fraction:
    # x = cp * T_stag * P_stag * (v_fg/h_fg)^2 * omega * ...
    # Simplified: x ≈ (1 - eta) * omega / (1 + omega)
    x_est = (1.0 - eta) * omega_use / (1.0 + omega_use)
    return max(0.0, min(1.0, x_est))


# ══════════════════════════════════════════════════════════════════════════════
# Omega Parameter
# ══════════════════════════════════════════════════════════════════════════════

def calculate_omega_parameter(
    p0: float,
    T0: float,
    rho_l0: float,
    rho_g0: float,
    cp_l: float,
    h_fg: float,
    x0: float = 0.0,
    k: float = 1.3,
) -> float:
    """Calculate the two-phase compressibility (omega) parameter.

    The omega parameter characterizes a two-phase fluid's compressibility and
    is used in the API 520 Annex C method for two-phase relief sizing.

    For a saturated liquid (x0=0):
        ω = cp_l * T0 * P0 / v_0 * (v_fg / h_fg)²

    For two-phase mixture (x0 > 0):
        ω = α_0 / k + (1 - α_0) * ω_s

    where α_0 = inlet void fraction, ω_s = saturated omega.

    Args:
        p0: Stagnation pressure [Pa].
        T0: Stagnation temperature [K].
        rho_l0: Stagnation liquid density [kg/m³].
        rho_g0: Stagnation vapor density [kg/m³].
        cp_l: Liquid specific heat [J/(kg·K)].
        h_fg: Latent heat of vaporization [J/kg].
        x0: Inlet mass quality [-].
        k: Specific heat ratio Cp/Cv for vapor [-].

    Returns:
        Omega parameter [-] (clamped to [0.05, 100.0]).
    """
    if p0 <= EPSILON or T0 <= EPSILON:
        return 1.0

    v_l0 = 1.0 / rho_l0 if rho_l0 > EPSILON else 1e-6
    v_g0 = 1.0 / rho_g0 if rho_g0 > EPSILON else 1e-6

    # Specific volume change upon vaporization
    v_fg0 = max(v_g0 - v_l0, EPSILON)

    # Mixture specific volume at stagnation
    v_0 = x0 * v_g0 + (1.0 - x0) * v_l0
    if v_0 <= EPSILON:
        return 1.0

    # Omega for saturated conditions (x=0 at vessel P, subcooled or saturated)
    # From Leung (1986):
    omega_s = 0.0
    if h_fg > EPSILON:
        omega_s = cp_l * T0 * p0 / v_0 * (v_fg0 / h_fg) ** 2

    # Inlet void fraction
    alpha_0 = x0 * v_g0 / v_0 if v_0 > EPSILON else 0.0

    # Combined omega
    if x0 <= EPSILON:
        omega_val = omega_s if omega_s > EPSILON else 1.0
    else:
        omega_val = alpha_0 / k + (1.0 - alpha_0) * max(omega_s, EPSILON)

    # Clamp to physically reasonable range per API 520
    return max(0.05, min(100.0, omega_val))


# ══════════════════════════════════════════════════════════════════════════════
# HEM — Homogeneous Equilibrium Model
# ══════════════════════════════════════════════════════════════════════════════

def _hem_critical_mass_flux(
    p0: float,
    rho_0: float,
    omega: float,
) -> Tuple[float, float, bool]:
    """Compute critical mass flux using HEM (Omega method).

    The dimensionless mass flux G* and critical pressure ratio η_c
    come from solving the isentropic nozzle flow equations with the
    omega parameter.

    Args:
        p0: Stagnation pressure [Pa].
        rho_0: Stagnation mixture density [kg/m³].
        omega: Compressibility parameter [-].

    Returns:
        (G_critical [kg/(m²·s)], eta_c [-], is_choked).
    """
    if p0 <= EPSILON or rho_0 <= EPSILON or omega <= EPSILON:
        return 0.0, 0.0, False

    # Critical pressure ratio for HEM
    # For omega → 0 (incompressible): η_c → 0
    # For omega → ∞ (highly compressible): η_c → 1
    if omega > 0.0:
        eta_c = omega / (omega + 1.0)
    else:
        eta_c = 0.0

    # Dimensionless critical mass flux
    # G*_c from Leung (1986) for choked flow at η = η_c
    eta = eta_c
    if eta <= EPSILON or eta >= 1.0 - EPSILON:
        return 0.0, eta_c, False

    # Compute G* using the general omega expression
    if abs(omega - 1.0) > 0.001:
        # General omega ≠ 1
        term = -2.0 * (omega * math.log(eta) + (omega - 1.0) * (1.0 - eta))
        denom = omega * (1.0 / eta - 1.0) + 1.0
    else:
        # omega = 1 special case
        term = 2.0 * (1.0 - eta - eta * math.log(eta))
        denom = 1.0 / eta

    if term <= 0.0 or denom <= 0.0:
        return 0.0, eta_c, False

    G_star_c = math.sqrt(term) / denom

    # Dimensionful mass flux
    G_critical = G_star_c * math.sqrt(2.0 * p0 * rho_0)

    return G_critical, eta_c, True


def hem_mass_flux(
    p0: float,
    p_back: float,
    rho_0: float,
    omega: float,
) -> dict:
    """Calculate mass flux using the HEM/Omega method.

    Determines whether flow is choked and computes the appropriate mass flux.

    Args:
        p0: Stagnation pressure [Pa].
        p_back: Back pressure [Pa].
        rho_0: Stagnation mixture density [kg/m³].
        omega: Compressibility parameter [-].

    Returns:
        Dict with G_flux [kg/(m²·s)], eta [-], is_choked, eta_c.
    """
    if p0 <= EPSILON:
        return {"G_flux": 0.0, "eta": 0.0, "is_choked": False, "eta_c": 0.0}

    p_back_eff = max(p_back, 0.0)

    # Critical pressure ratio
    eta_c = omega / (omega + 1.0) if omega > EPSILON else 0.0
    eta_actual = p_back_eff / p0
    is_choked = eta_actual < eta_c
    eta_use = eta_c if is_choked else eta_actual

    if eta_use <= EPSILON or eta_use >= 1.0 - EPSILON:
        return {"G_flux": 0.0, "eta": eta_use, "is_choked": is_choked, "eta_c": eta_c}

    # Compute dimensionless mass flux at operating eta
    if abs(omega - 1.0) > 0.001:
        term = -2.0 * (omega * math.log(eta_use) + (omega - 1.0) * (1.0 - eta_use))
        denom = omega * (1.0 / eta_use - 1.0) + 1.0
    else:
        term = 2.0 * (1.0 - eta_use - eta_use * math.log(eta_use))
        denom = 1.0 / eta_use

    if term <= 0.0 or denom <= 0.0:
        return {"G_flux": 0.0, "eta": eta_use, "is_choked": is_choked, "eta_c": eta_c}

    G_star = math.sqrt(term) / denom
    G_flux = G_star * math.sqrt(2.0 * p0 * rho_0)

    return {"G_flux": G_flux, "eta": eta_use, "is_choked": is_choked, "eta_c": eta_c}


# ══════════════════════════════════════════════════════════════════════════════
# Slip Models — Fauske & Moody
# ══════════════════════════════════════════════════════════════════════════════

def fauske_slip_ratio(
    rho_l: float,
    rho_g: float,
    x: float,
) -> float:
    """Fauske slip ratio correlation for two-phase flow.

    The Fauske model is commonly used for non-flashing, non-equilibrium
    two-phase flow where the phases travel at different velocities.

    S = u_g / u_l = sqrt(ρ_l / ρ_g)

    Args:
        rho_l: Liquid density [kg/m³].
        rho_g: Gas density [kg/m³].
        x: Mass quality (vapor fraction) [-] (for reference).

    Returns:
        Slip ratio S = u_g/u_l [-].
    """
    if rho_g <= EPSILON or rho_l <= EPSILON:
        return 1.0
    return math.sqrt(rho_l / rho_g)


def moody_slip_ratio(
    rho_l: float,
    rho_g: float,
    x: float,
) -> float:
    """Moody slip ratio correlation for annular two-phase flow.

    For annular flow, the Moody model gives:
        S = (ρ_l / ρ_g)^(1/3)

    This predicts less slip than Fauske but is more appropriate
    for well-mixed annular flow patterns.

    Args:
        rho_l: Liquid density [kg/m³].
        rho_g: Gas density [kg/m³].
        x: Mass quality [-].

    Returns:
        Slip ratio S = u_g/u_l [-].
    """
    if rho_g <= EPSILON or rho_l <= EPSILON:
        return 1.0
    return (rho_l / rho_g) ** (1.0 / 3.0)


def slip_model_mass_flux(
    p0: float,
    p_back: float,
    rho_l: float,
    rho_g: float,
    x: float,
    model: str = "fauske",
    L_over_D: float = 0.0,
    Cd: float = 1.0,
) -> dict:
    """Calculate critical mass flux using a slip-flow model.

    For orifice flow (L/D = 0), use Fauske or Moody slip ratios.
    For pipe flow (L/D > 0), incorporate friction effects.

    The slip-based critical mass flux is given by:
        G_c^2 = -1 / d/dP (1/G^2) evaluated at critical conditions

    Simplified form from Fauske (1965):
        G_c = h_fg / (v_fg * sqrt(T * cp_l))

    Better simplified slip model:
        G_c ≈ sqrt( (h_fg * rho_g) / (v_fg * T * cp_l) )

    Args:
        p0: Stagnation pressure [Pa].
        p_back: Back pressure [Pa].
        rho_l: Liquid density [kg/m³].
        rho_g: Vapor density [kg/m³] at stagnation.
        x: Mass quality [-].
        model: 'fauske' or 'moody'.
        L_over_D: Length/diameter ratio for pipe effects.
        Cd: Discharge coefficient [-].

    Returns:
        Dict with G_flux, slip_ratio, void_fraction, is_choked.
    """
    # Slip ratio
    if model == "fauske":
        S = fauske_slip_ratio(rho_l, rho_g, x)
    elif model == "moody":
        S = moody_slip_ratio(rho_l, rho_g, x)
    else:
        S = 1.0

    # Void fraction based on mass quality and slip ratio
    if rho_g > EPSILON and rho_l > EPSILON:
        alpha = 1.0 / (1.0 + S * (1.0 - x) / x * rho_g / rho_l) if x > EPSILON else 0.0
    else:
        alpha = 0.0

    # Mixture density using slip model
    rho_mix = alpha * rho_g + (1.0 - alpha) * rho_l

    # Two-phase critical mass flux approximation
    # Based on momentum balance across nozzle
    dp = p0 - p_back
    if dp <= EPSILON:
        return {"G_flux": 0.0, "slip_ratio": S, "void_fraction": alpha, "is_choked": False}

    # Simplified slip-based mass flux
    # G ≈ sqrt(2 * rho_mix * dp / (1 + resistance))
    # For orifice: resistance = 0 (just acceleration)
    # For pipe: resistance includes friction

    # Check if choked (for two-phase, choked at much lower pressure ratios)
    eta_c = 0.55  # typical for saturated water
    eta = p_back / p0
    is_choked = eta < eta_c

    # Mass flux estimate
    if is_choked:
        # Use simplified critical flow correlation
        # G_c proportional to h_fg / (v_fg * sqrt(T * cp))
        # This is a rough engineering estimate
        G_flux = math.sqrt(2.0 * p0 * rho_mix) * 0.5  # conservative
    else:
        G_flux = Cd * math.sqrt(2.0 * rho_mix * dp)

    return {"G_flux": G_flux, "slip_ratio": S, "void_fraction": alpha, "is_choked": is_choked}


# ══════════════════════════════════════════════════════════════════════════════
# Main Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def calculate_two_phase_flow(inputs: TwoPhaseInput) -> TwoPhaseResult:
    """Calculate two-phase flow characteristics.

    Main entry point that selects the appropriate model (HEM, Fauske, Moody,
    Omega) and computes mass flux, quality, regime, etc.

    Args:
        inputs: TwoPhaseInput with fluid properties and model selection.

    Returns:
        TwoPhaseResult with mass flux, quality, etc.

    Raises:
        ValueError: If invalid model or missing parameters.

    Example:
        >>> inp = TwoPhaseInput(
        ...     P_stagnation=5e5, T=420, P_backpressure=1e5,
        ...     rho_l=900, rho_g=2.5, cp_l=2500,
        ...     heat_of_vaporization=3.5e5, model='hem'
        ... )
        >>> result = calculate_two_phase_flow(inp)
        >>> print(f"G_critical: {result.G_critical:.1f} kg/m²s")
    """
    messages = []
    model = inputs.model.lower()

    # Compute omega parameter
    omega = calculate_omega_parameter(
        p0=inputs.P_stagnation,
        T0=inputs.T,
        rho_l0=inputs.rho_l,
        rho_g0=inputs.rho_g,
        cp_l=inputs.cp_l,
        h_fg=inputs.heat_of_vaporization,
        x0=inputs.x0,
        k=inputs.cp_cv_ratio,
    )

    # Flash fraction
    x_flash = calculate_flash_fraction(
        T_stag=inputs.T,
        T_sat_at_backpressure=inputs.T * (inputs.P_backpressure / inputs.P_stagnation) ** 0.1,
        cp_liquid=inputs.cp_l,
        h_fg=inputs.heat_of_vaporization,
    )
    x_total = max(inputs.x0, x_flash)
    x_total = max(0.0, min(1.0, x_total))

    # Estimate mixture density
    if inputs.rho_g > EPSILON and inputs.rho_l > EPSILON:
        v_g = 1.0 / inputs.rho_g
        v_l = 1.0 / inputs.rho_l
        v_mix = x_total * v_g + (1.0 - x_total) * v_l
        rho_mix = 1.0 / v_mix if v_mix > EPSILON else inputs.rho_l
    else:
        rho_mix = inputs.rho_l

    # Run selected model
    if model in ("hem", "omega", ""):
        # Use HEM/Omega method
        result_hem = hem_mass_flux(
            p0=inputs.P_stagnation,
            p_back=inputs.P_backpressure,
            rho_0=rho_mix,
            omega=omega,
        )
        G_critical, eta_c, is_choked = _hem_critical_mass_flux(
            p0=inputs.P_stagnation,
            rho_0=rho_mix,
            omega=omega,
        )
        G_actual = result_hem["G_flux"]
        slip = 1.0
        # Void fraction for homogeneous flow
        if inputs.rho_g > EPSILON and x_total > EPSILON:
            void_frac = 1.0 / (1.0 + (1.0 - x_total) / x_total * inputs.rho_g / inputs.rho_l)
        else:
            void_frac = 0.0

    elif model == "fauske" or model == "moody":
        result_slip = slip_model_mass_flux(
            p0=inputs.P_stagnation,
            p_back=inputs.P_backpressure,
            rho_l=inputs.rho_l,
            rho_g=inputs.rho_g,
            x=x_total,
            model=model,
            L_over_D=inputs.L_over_D,
        )
        G_critical = result_slip["G_flux"]
        G_actual = result_slip["G_flux"]
        eta_c = omega / (omega + 1.0) if omega > EPSILON else 0.5
        is_choked = result_slip["is_choked"]
        slip = result_slip["slip_ratio"]
        void_frac = result_slip["void_fraction"]

    else:
        raise ValueError(f"Unknown two-phase model: '{model}'. "
                         f"Use 'hem', 'fauske', 'moody', or 'omega'.")

    # Determine regime
    if x_total < 0.01:
        regime = TwoPhaseRegime.BUBBLY.value
    elif x_total < 0.1:
        regime = TwoPhaseRegime.SLUG.value
    elif x_total < 0.8:
        regime = TwoPhaseRegime.ANNULAR.value
    else:
        regime = TwoPhaseRegime.MIST.value

    # Adjust for pipe vs orifice
    if inputs.L_over_D > 0 and is_choked:
        regime = f"{regime}_pipe"

    return TwoPhaseResult(
        G_critical=max(G_critical, 0.0),
        G_actual=max(G_actual, 0.0),
        x_mass=x_total,
        eta_critical=eta_c,
        regime=regime,
        omega=omega,
        slip_ratio=slip,
        void_fraction=void_frac,
        is_choked=is_choked,
        messages=messages,
    )
