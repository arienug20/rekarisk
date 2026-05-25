"""
Rekarisk — Pipe Flow & Pipeline Rupture Models.

Calculates flow through pipes including full-bore rupture (guillotine break),
hole-in-pipe leaks, long pipeline pressure drop + discharge, and two-phase
pipe flow using Lockhart-Martinelli correlation.

Models:
  - Darcy-Weisbach with Colebrook (implicit) friction factor
  - Full-bore rupture: upstream pressure drives flow at both ends
  - Hole-in-pipe: orifice + pipe friction in series
  - Long pipeline: pressure profile along pipe with discharge at end
  - Two-phase: Lockhart-Martinelli multiplier for pressure drop

References:
  - Crane Technical Paper No. 410 — Flow of Fluids
  - CCPS Guidelines for Consequence Analysis (1999), Chapter 2
  - TNO Yellow Book (CPR 14E), Chapter 2
  - Lockhart & Martinelli (1949), Chem. Eng. Prog. 45(1), 39-48
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from ...core.constants import R, P_ATM, G as GRAVITY, EPSILON


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class RuptureType(str, Enum):
    """Pipe failure mode."""
    FULL_BORE = "full_bore"      # guillotine / complete severance
    HOLE_IN_PIPE = "hole_in_pipe"  # leak from a hole in pipe wall
    LONG_PIPELINE = "long_pipeline"  # pressure drop along pipe + end discharge


class FlowType(str, Enum):
    """Fluid flow type."""
    LIQUID = "liquid"
    GAS = "gas"
    TWO_PHASE = "two_phase"


class FlowRegime(str, Enum):
    """Pipe flow regime."""
    LAMINAR = "laminar"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"
    CHOKED = "choked"


# ══════════════════════════════════════════════════════════════════════════════
# Input/Output Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipeInput:
    """Input parameters for pipe flow / rupture calculation.

    Attributes:
        L: Pipe length [m].
        D: Pipe internal diameter [m].
        roughness: Pipe absolute roughness [m] (e.g., 0.0457e-3 for commercial steel).
        P_up: Upstream pressure [Pa abs].
        P_down: Downstream pressure [Pa abs] (ambient for rupture to atmosphere).
        T: Fluid temperature [K].
        fluid: Fluid type ('liquid', 'gas', 'two_phase').
        rupture_type: Failure mode ('full_bore', 'hole_in_pipe', 'long_pipeline').
        rho: Fluid density [kg/m³] (liquid or gas at upstream).
        mu: Dynamic viscosity [Pa·s].
        molecular_weight: [kg/mol] (for gas).
        cp_cv_ratio: Specific heat ratio k = Cp/Cv [-] (for gas).
        Cd: Discharge coefficient [-] (for hole/orifice).
        d_hole: Hole diameter [m] (for hole_in_pipe).
        x_mass: Mass quality [-] (for two-phase).
        rho_g: Gas density [kg/m³] (for two-phase).
        n_segments: Number of pipe segments for profile calculation.
    """
    L: float
    D: float
    roughness: float = 0.0457e-3  # commercial steel [m]
    P_up: float = 5e5
    P_down: float = 101325.0
    T: float = 300.0

    # Fluid
    fluid: str = "gas"
    rupture_type: str = "full_bore"
    rho: float | None = None
    mu: float = 1.8e-5
    molecular_weight: float = 0.0289647
    cp_cv_ratio: float = 1.4

    # Hole-in-pipe
    Cd: float = 0.62
    d_hole: float | None = None

    # Two-phase
    x_mass: float = 0.0
    rho_g: float | None = None

    # Numerical
    n_segments: int = 50


@dataclass
class PipeResult:
    """Results from pipe flow / rupture calculation.

    Attributes:
        mdot: Mass flow rate [kg/s] (discharge from break/end).
        P_profile: Pressure along pipe [Pa] (len = n_segments + 1).
        x_profile: Distance along pipe [m].
        delta_P: Total pressure drop [Pa].
        flow_regime: Flow regime in the pipe.
        friction_factor: Darcy friction factor [-].
        velocity: Exit velocity [m/s].
        is_choked: Whether flow is choked at exit/burst.
        Re: Reynolds number [-].
        messages: Info/warning strings.
    """
    mdot: float
    P_profile: np.ndarray
    x_profile: np.ndarray
    delta_P: float
    flow_regime: str
    friction_factor: float
    velocity: float
    is_choked: bool = False
    Re: float = 0.0
    messages: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Friction Factor — Colebrook Equation
# ══════════════════════════════════════════════════════════════════════════════

def colebrook_friction_factor(
    Re: float,
    relative_roughness: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Compute Darcy-Weisbach friction factor using Colebrook equation.

    The implicit Colebrook equation:
        1/√f = -2.0 * log10( ε/(3.7*D) + 2.51/(Re*√f) )

    Solved via iterative substitution (fast convergence).

    For laminar flow (Re < 2100): f = 64/Re

    Args:
        Re: Reynolds number [-] (> 0).
        relative_roughness: ε/D, absolute roughness/diameter [-].
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        Darcy friction factor f [-].
    """
    if Re <= EPSILON:
        return 0.0
    if Re < 2100.0:
        # Laminar
        return 64.0 / Re
    if Re > 4000.0 and relative_roughness <= EPSILON:
        # Smooth pipe — Blasius correlation for fully turbulent
        return 0.0791 / (Re ** 0.25)

    # Colebrook (iterative, turbulent)
    # Initial guess from Swamee-Jain explicit approximation
    eps_D = relative_roughness
    if eps_D > EPSILON:
        f_guess = 0.25 / (math.log10(eps_D / 3.7 + 5.74 / Re ** 0.9)) ** 2
    else:
        f_guess = 0.0791 / Re ** 0.25

    f = f_guess
    for _ in range(max_iter):
        if f <= EPSILON:
            break
        inv_sqrt_f = 1.0 / math.sqrt(f)
        # Colebrook:
        # 1/√f + 2*log10(ε/(3.7*D) + 2.51/(Re*√f)) = 0
        term = eps_D / 3.7 + 2.51 / (Re * inv_sqrt_f) if Re > 0 else eps_D / 3.7
        if term <= EPSILON:
            break
        new_inv_sqrt_f = -2.0 * math.log10(term)
        f_new = 1.0 / (new_inv_sqrt_f * new_inv_sqrt_f)

        if abs(f_new - f) < tol:
            return f_new
        f = f_new

    return max(f, 0.001)  # minimum reasonable friction factor


def swamee_jain_friction_factor(
    Re: float,
    relative_roughness: float,
) -> float:
    """Explicit friction factor using Swamee-Jain approximation.

    Valid for 1e-6 ≤ ε/D ≤ 1e-2 and 5000 ≤ Re ≤ 1e8.

    f = 0.25 / (log10(ε/(3.7*D) + 5.74/Re^0.9))²

    Args:
        Re: Reynolds number [-].
        relative_roughness: ε/D [-].

    Returns:
        Darcy friction factor f [-].
    """
    if Re <= EPSILON:
        return 0.0
    if Re < 2100.0:
        return 64.0 / Re

    eps_D = max(relative_roughness, 1e-8)
    denom = math.log10(eps_D / 3.7 + 5.74 / Re ** 0.9)
    if abs(denom) < EPSILON:
        return 0.1
    return 0.25 / (denom * denom)


def reynolds_number(
    rho: float,
    vel: float,
    D: float,
    mu: float,
) -> float:
    """Calculate Reynolds number.

    Re = ρ * v * D / μ

    Args:
        rho: Fluid density [kg/m³].
        vel: Flow velocity [m/s].
        D: Pipe diameter [m].
        mu: Dynamic viscosity [Pa·s].

    Returns:
        Reynolds number [-].
    """
    if mu <= EPSILON:
        return 1e9  # infinite Re (inviscid)
    return rho * vel * D / mu


# ══════════════════════════════════════════════════════════════════════════════
# Darcy-Weisbach Pressure Drop
# ══════════════════════════════════════════════════════════════════════════════

def darcy_weisbach_pressure_drop(
    f: float,
    L: float,
    D: float,
    rho: float,
    vel: float,
) -> float:
    """Calculate pressure drop via Darcy-Weisbach equation.

    ΔP = f * (L/D) * (ρ * v² / 2)

    Args:
        f: Darcy friction factor [-].
        L: Pipe length [m].
        D: Pipe diameter [m].
        rho: Fluid density [kg/m³].
        vel: Mean flow velocity [m/s].

    Returns:
        Pressure drop [Pa].
    """
    return f * (L / D) * (rho * vel * vel / 2.0)


# ══════════════════════════════════════════════════════════════════════════════
# Pipe Flow Calculations
# ══════════════════════════════════════════════════════════════════════════════

def _single_phase_pipe_flow(
    inputs: PipeInput,
) -> dict:
    """Calculate single-phase (liquid or gas) pipe flow.

    Uses iterative velocity-pressure drop coupling to find the
    equilibrium flow rate.

    Args:
        inputs: PipeInput dataclass.

    Returns:
        Dict with mdot, P_profile, velocity, etc.
    """
    D = inputs.D
    A = math.pi * (D / 2.0) ** 2
    L = inputs.L
    eps = inputs.roughness
    eps_D = eps / D if D > EPSILON else 0.0

    # Determine density
    if inputs.rho is not None:
        rho = inputs.rho
    elif inputs.fluid == "gas":
        rho = (inputs.P_up * inputs.molecular_weight) / (R * inputs.T)
    else:
        rho = 1000.0  # default water

    mu = inputs.mu

    # Simple solution approach:
    # For liquid or incompressible gas (low ΔP/P):
    #   1. Guess velocity
    #   2. Compute Re, f
    #   3. Balance: P_up - P_down = f * (L/D) * (rho * v²/2)
    #   4. Solve for v

    dp_total = inputs.P_up - inputs.P_down
    if dp_total <= EPSILON:
        return {
            "mdot": 0.0,
            "P_profile": np.array([inputs.P_up, inputs.P_down]),
            "x_profile": np.array([0.0, L]),
            "velocity": 0.0,
            "delta_P": 0.0,
            "flow_regime": FlowRegime.LAMINAR.value,
            "friction_factor": 0.0,
            "Re": 0.0,
            "is_choked": False,
        }

    # Compute pressure profile along pipe
    n_seg = inputs.n_segments
    x_prof = np.linspace(0.0, L, n_seg + 1)
    P_prof = np.zeros(n_seg + 1)
    P_prof[0] = inputs.P_up

    # For gas, need to account for density change with pressure
    # Simplified: use average density
    if inputs.fluid == "gas" and dp_total / inputs.P_up > 0.1:
        rho_avg = (rho + (inputs.P_down * inputs.molecular_weight) / (R * inputs.T)) / 2.0
    else:
        rho_avg = rho

    # Iterative solution: balance pressure drop
    # ΔP = f * (L/D) * (ρ * v²/2)
    # v = mdot / (ρ * A)

    # Start with guess using fully turbulent f
    f_turb = 0.02  # typical for turbulent flow
    v_guess = math.sqrt(2.0 * dp_total * D / (f_turb * L * rho_avg))
    v_guess = max(v_guess, 0.001)

    Re_final = 0.0
    f_final = 0.0
    vel_final = v_guess

    for iteration in range(30):
        vel = vel_final
        Re_val = reynolds_number(rho_avg, vel, D, mu)
        f = colebrook_friction_factor(Re_val, eps_D)

        # Solve: v = sqrt(2 * dp * D / (f * L * rho))
        # But dp = P_up - P_down - rho * v²/2 (exit loss)
        # Including minor losses: exit K=1
        K_exit = 1.0
        K_total = f * (L / D) + K_exit
        if K_total <= EPSILON:
            dp_actual = dp_total
            vel_final = 1000.0  # near-infinite flow, cap
        else:
            vel_final = math.sqrt(2.0 * dp_total / (rho_avg * K_total))

        Re_final = Re_val
        f_final = f

        if abs(vel_final - vel) / max(vel, 0.001) < 1e-4:
            break

    # Mass flow rate
    mdot = rho_avg * A * vel_final

    # Check for choked flow (gas only)
    is_choked = False
    if inputs.fluid == "gas":
        k = inputs.cp_cv_ratio
        r_crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
        P_crit = inputs.P_up * r_crit
        if inputs.P_down < P_crit:
            is_choked = True

    # Generate pressure profile
    # Linear pressure drop for liquid; more complex for gas
    if inputs.fluid == "gas" and len(x_prof) > 1:
        # Exponential-like for compressible flow
        for i in range(len(x_prof)):
            P_prof[i] = inputs.P_up * (1.0 - x_prof[i] / L * (1.0 - inputs.P_down / inputs.P_up))
    else:
        P_prof = np.linspace(inputs.P_up, inputs.P_down, n_seg + 1)

    # Determine flow regime
    if Re_final < 2100.0:
        regime = FlowRegime.LAMINAR.value
    elif Re_final < 4000.0:
        regime = FlowRegime.TRANSITIONAL.value
    else:
        regime = FlowRegime.TURBULENT.value
    if is_choked:
        regime = FlowRegime.CHOKED.value

    return {
        "mdot": mdot,
        "P_profile": P_prof,
        "x_profile": x_prof,
        "velocity": vel_final,
        "delta_P": dp_total,
        "flow_regime": regime,
        "friction_factor": f_final,
        "Re": Re_final,
        "is_choked": is_choked,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Full-Bore Rupture (Guillotine Break)
# ══════════════════════════════════════════════════════════════════════════════

def _full_bore_rupture(inputs: PipeInput) -> dict:
    """Calculate full-bore pipe rupture (guillotine break).

    At the rupture plane, flow is driven by the pressure difference between
    pipe internal pressure (at break location) and ambient. The pipe
    friction upstream limits the flow rate.

    Simplification: treat as pipe end discharge with upstream reservoir.
    The break point sees P_internal (after friction loss) discharging to P_ambient.

    Args:
        inputs: PipeInput dataclass.

    Returns:
        Dict with mdot, P_profile, etc.
    """
    D = inputs.D
    A = math.pi * (D / 2.0) ** 2
    L = inputs.L

    # For a guillotine break, there are TWO discharge points:
    # 1. From upstream reservoir to break (L_upstream)
    # 2. From break to downstream (L_downstream)
    # Both discharge to atmosphere.

    # Simplified: treat as pipe of length L/2 (assume break at midpoint)
    # discharging at both ends. The flow from each side is roughly equal.
    # More conservative: use full L for single-sided flow.

    half_L = L / 2.0  # assume break at pipe midpoint

    # Fluid properties
    if inputs.rho is not None:
        rho = inputs.rho
    elif inputs.fluid == "gas":
        rho = (inputs.P_up * inputs.molecular_weight) / (R * inputs.T)
    else:
        rho = 1000.0

    mu = inputs.mu
    eps_D = inputs.roughness / D if D > EPSILON else 0.0

    # Pressure at break plane is not P_down; it's somewhere between
    # P_up and P_down. For full-bore, approximate as:
    # P_break ~ P_up - friction_loss(half_L)

    dp_avail = inputs.P_up - inputs.P_down
    if dp_avail <= EPSILON:
        return {
            "mdot": 0.0, "P_profile": np.array([inputs.P_up]), "x_profile": np.array([0.0]),
            "velocity": 0.0, "delta_P": 0.0, "flow_regime": FlowRegime.LAMINAR.value,
            "friction_factor": 0.0, "Re": 0.0, "is_choked": False,
        }

    # Iterative: solve for flow such that friction loss matches available pressure
    # velocity from: v = sqrt(2*dp_avail / (rho * (f*L/D + K_exit)))
    f_est = 0.015
    n_iter = 30

    vel = math.sqrt(2.0 * dp_avail * D / (f_est * half_L * rho))
    vel = max(vel, 0.01)

    for _ in range(n_iter):
        Re_val = reynolds_number(rho, vel, D, mu)
        f = colebrook_friction_factor(Re_val, eps_D)
        K_total = f * (half_L / D) + 1.0  # exit loss
        vel_new = math.sqrt(2.0 * dp_avail / (rho * K_total))
        vel_new = max(vel_new, 0.001)
        if abs(vel_new - vel) / max(vel, 0.01) < 1e-4:
            vel = vel_new
            break
        vel = vel_new

    mdot_single_side = rho * A * vel
    # Both sides discharge
    mdot_total = 2.0 * mdot_single_side

    # Pressure profile: linear from P_up to P_down
    # P at break ≈ P_down (atmospheric)
    P_prof = np.linspace(inputs.P_up, inputs.P_down, inputs.n_segments + 1)
    x_prof = np.linspace(0.0, L, inputs.n_segments + 1)

    # Check choking (gas)
    is_choked = False
    if inputs.fluid == "gas":
        k = inputs.cp_cv_ratio
        r_crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
        P_crit = inputs.P_up * r_crit
        if inputs.P_down < P_crit:
            is_choked = True

    regime = FlowRegime.TURBULENT.value if Re_val > 4000 else FlowRegime.LAMINAR.value

    return {
        "mdot": mdot_total,
        "P_profile": P_prof,
        "x_profile": x_prof,
        "velocity": vel,
        "delta_P": dp_avail,
        "flow_regime": regime,
        "friction_factor": f,
        "Re": Re_val,
        "is_choked": is_choked,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Hole-in-Pipe
# ══════════════════════════════════════════════════════════════════════════════

def _hole_in_pipe(inputs: PipeInput) -> dict:
    """Calculate flow from a hole in a pressurized pipe.

    The flow is limited by two resistances in series:
    1. Pipe friction from upstream to hole location
    2. Orifice (hole) discharge

    Args:
        inputs: PipeInput dataclass.

    Returns:
        Dict with mdot, etc.
    """
    if inputs.d_hole is None:
        raise ValueError("d_hole is required for hole_in_pipe rupture type.")

    D_pipe = inputs.D
    A_pipe = math.pi * (D_pipe / 2.0) ** 2
    A_hole = math.pi * (inputs.d_hole / 2.0) ** 2
    L = inputs.L
    Cd = inputs.Cd

    # Fluid properties
    if inputs.rho is not None:
        rho = inputs.rho
    elif inputs.fluid == "gas":
        rho = (inputs.P_up * inputs.molecular_weight) / (R * inputs.T)
    else:
        rho = 1000.0

    mu = inputs.mu
    eps_D = inputs.roughness / D_pipe if D_pipe > EPSILON else 0.0

    dp_avail = inputs.P_up - inputs.P_down
    if dp_avail <= EPSILON:
        return {
            "mdot": 0.0, "P_profile": np.array([inputs.P_up]), "x_profile": np.array([0.0]),
            "velocity": 0.0, "delta_P": 0.0,
            "flow_regime": FlowRegime.LAMINAR.value,
            "friction_factor": 0.0, "Re": 0.0, "is_choked": False,
        }

    # Iterative: the flow through the hole determines pipe velocity
    # Orifice equation: mdot = Cd * A_hole * rho * sqrt(2 * (P_at_hole - P_down) / rho)
    # Pipe friction: P_up - P_at_hole = f * (L_hole/D) * (rho * v_pipe²/2)
    # Continuity: mdot = rho * A_pipe * v_pipe

    # Solve iteratively
    mdot_guess = Cd * A_hole * math.sqrt(2.0 * rho * dp_avail)
    vel_pipe = mdot_guess / (rho * A_pipe) if rho * A_pipe > EPSILON else 0.0

    for _ in range(30):
        Re_pipe = reynolds_number(rho, vel_pipe, D_pipe, mu)
        f = colebrook_friction_factor(Re_pipe, eps_D)

        # Pressure drop along pipe section to hole (assume hole at midpoint)
        dp_pipe = f * (L / 2.0) / D_pipe * (rho * vel_pipe * vel_pipe / 2.0)
        P_at_hole = inputs.P_up - dp_pipe
        dp_hole = max(P_at_hole - inputs.P_down, 0.0)

        # Orifice flow through hole
        mdot_new = Cd * A_hole * math.sqrt(2.0 * rho * dp_hole)
        vel_new = mdot_new / (rho * A_pipe) if rho * A_pipe > EPSILON else 0.0

        if abs(mdot_new - mdot_guess) / max(mdot_guess, 1e-9) < 1e-4:
            mdot_guess = mdot_new
            vel_pipe = vel_new
            break

        mdot_guess = mdot_new
        vel_pipe = vel_new

    # Pressure profile
    x_prof = np.linspace(0.0, L, inputs.n_segments + 1)
    P_prof = np.zeros_like(x_prof)
    P_prof[0] = inputs.P_up
    x_hole = L / 2.0
    for i, x in enumerate(x_prof):
        if x <= x_hole:
            P_prof[i] = inputs.P_up - dp_pipe * (x / x_hole)
        else:
            P_prof[i] = P_at_hole  # downstream of hole, pressure ~ ambient

    is_choked = False
    if inputs.fluid == "gas":
        k = inputs.cp_cv_ratio
        r_crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
        if P_at_hole > dp_hole * r_crit + inputs.P_down:
            is_choked = True

    return {
        "mdot": mdot_guess,
        "P_profile": P_prof,
        "x_profile": x_prof,
        "velocity": math.sqrt(2.0 * dp_hole / rho) if dp_hole > 0 and rho > 0 else 0.0,
        "delta_P": dp_avail,
        "flow_regime": FlowRegime.TURBULENT.value if Re_pipe > 4000 else FlowRegime.LAMINAR.value,
        "friction_factor": f,
        "Re": Re_pipe,
        "is_choked": is_choked,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Lockhart-Martinelli Two-Phase Multiplier
# ══════════════════════════════════════════════════════════════════════════════

def lockhart_martinelli_multiplier(
    x: float,
    rho_l: float,
    rho_g: float,
    mu_l: float,
    mu_g: float,
) -> float:
    """Calculate Lockhart-Martinelli two-phase pressure drop multiplier.

    The multiplier Φ²_L relates two-phase pressure drop to liquid-only:
        (dP/dz)_tp = Φ²_L * (dP/dz)_L

    Lockhart-Martinelli parameter X:
        X² = (dP/dz)_L / (dP/dz)_G

    where both are computed at the respective phase flow rates.

    Simplified for turbulent-turbulent flow (both phases Re > 2000):
        X_tt = ((1-x)/x)^0.9 * (ρ_g/ρ_l)^0.5 * (μ_l/μ_g)^0.1

        Φ²_L = 1 + C/X + 1/X²

    where C = 20 for turbulent-turbulent, 12 for other combinations.

    Args:
        x: Mass quality (vapor mass fraction) [-].
        rho_l: Liquid density [kg/m³].
        rho_g: Gas density [kg/m³].
        mu_l: Liquid dynamic viscosity [Pa·s].
        mu_g: Gas dynamic viscosity [Pa·s].

    Returns:
        Tuple of (X_tt, Phi_L_squared) — Lockhart-Martinelli parameter X
        and two-phase multiplier for liquid.
    """
    if x <= EPSILON:
        return 0.0, 1.0
    if x >= 1.0 - EPSILON:
        return 1e9, 1e9  # gas-only

    # Lockhart-Martinelli parameter for turbulent-turbulent
    if mu_g <= EPSILON or mu_l <= EPSILON:
        mu_ratio = 1.0
    else:
        mu_ratio = (mu_l / mu_g)

    X_tt = ((1.0 - x) / x) ** 0.9 * (rho_g / rho_l) ** 0.5 * mu_ratio ** 0.1

    # Chisholm correlation for two-phase multiplier
    if X_tt <= EPSILON:
        return X_tt, 1.0

    # For turbulent-turbulent: C = 20
    C = 20.0

    Phi_L_sq = 1.0 + C / X_tt + 1.0 / (X_tt * X_tt)

    return X_tt, max(Phi_L_sq, 1.0)


def two_phase_pipe_pressure_drop(
    L: float,
    D: float,
    mdot_total: float,
    x: float,
    rho_l: float,
    rho_g: float,
    mu_l: float,
    mu_g: float,
    roughness: float = 0.0457e-3,
) -> float:
    """Calculate two-phase pressure drop in a pipe.

    Uses Lockhart-Martinelli multiplier applied to liquid-only pressure drop.

    Args:
        L: Pipe length [m].
        D: Pipe diameter [m].
        mdot_total: Total mass flow rate [kg/s].
        x: Mass quality [-].
        rho_l: Liquid density [kg/m³].
        rho_g: Gas density [kg/m³].
        mu_l: Liquid viscosity [Pa·s].
        mu_g: Gas viscosity [Pa·s].
        roughness: Pipe roughness [m].

    Returns:
        Two-phase pressure drop [Pa].
    """
    A = math.pi * (D / 2.0) ** 2
    if A <= EPSILON or rho_l <= EPSILON:
        return 0.0

    # Liquid-only flow rate
    mdot_l = mdot_total * (1.0 - x)

    # Liquid-only velocity
    vel_l = mdot_l / (rho_l * A) if rho_l * A > EPSILON else 0.0

    # Liquid-only Re and friction factor
    if vel_l > EPSILON and mu_l > EPSILON:
        Re_l = rho_l * vel_l * D / mu_l
    else:
        Re_l = 0.0

    eps_D = roughness / D if D > EPSILON else 1e-6
    f_l = colebrook_friction_factor(Re_l, eps_D)

    # Liquid-only pressure drop
    dP_L = f_l * (L / D) * (rho_l * vel_l * vel_l / 2.0)

    # Two-phase multiplier
    _, Phi_L_sq = lockhart_martinelli_multiplier(x, rho_l, rho_g, mu_l, mu_g)

    return dP_L * Phi_L_sq


# ══════════════════════════════════════════════════════════════════════════════
# Main Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def calculate_pipe_flow(inputs: PipeInput) -> PipeResult:
    """Calculate pipe flow or pipe rupture discharge.

    Main entry point that routes to the appropriate model based on
    fluid type and rupture type.

    Args:
        inputs: PipeInput with all required parameters.

    Returns:
        PipeResult with mdot, P_profile, ΔP, regime, etc.

    Raises:
        ValueError: If invalid rupture type or missing parameters.

    Example:
        >>> inp = PipeInput(
        ...     L=100, D=0.1, P_up=1e6, P_down=101325,
        ...     T=300, fluid='gas', rupture_type='full_bore',
        ...     rho=None, molecular_weight=0.016, cp_cv_ratio=1.3
        ... )
        >>> result = calculate_pipe_flow(inp)
        >>> print(f"Release rate: {result.mdot:.3f} kg/s")
    """
    rupture_type = inputs.rupture_type.lower()

    if rupture_type == RuptureType.FULL_BORE.value:
        data = _full_bore_rupture(inputs)
    elif rupture_type == RuptureType.HOLE_IN_PIPE.value:
        data = _hole_in_pipe(inputs)
    elif rupture_type == RuptureType.LONG_PIPELINE.value:
        data = _single_phase_pipe_flow(inputs)
    else:
        raise ValueError(f"Unknown rupture type '{inputs.rupture_type}'. "
                         f"Use 'full_bore', 'hole_in_pipe', or 'long_pipeline'.")

    return PipeResult(
        mdot=data["mdot"],
        P_profile=data["P_profile"],
        x_profile=data["x_profile"],
        delta_P=data["delta_P"],
        flow_regime=data["flow_regime"],
        friction_factor=data["friction_factor"],
        velocity=data["velocity"],
        is_choked=data.get("is_choked", False),
        Re=data.get("Re", 0.0),
        messages=[],
    )
