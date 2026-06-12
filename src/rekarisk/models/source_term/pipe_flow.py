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
from typing import List, Optional, Tuple, Dict

import numpy as np

from ...core.constants import R, P_ATM, G as GRAVITY, EPSILON

# Lazy import of CoolProp (may not be installed)
_COOLPROP_AVAILABLE = False
try:
    from CoolProp.CoolProp import AbstractState as CP_AbstractState
    import CoolProp
    _COOLPROP_AVAILABLE = True
except ImportError:
    CP_AbstractState = None
    CoolProp = None


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class RuptureType(str, Enum):
    """Pipe failure mode."""
    FULL_BORE = "full_bore"      # guillotine / complete severance
    HOLE_IN_PIPE = "hole_in_pipe"  # leak from a hole in pipe wall
    LONG_PIPELINE = "long_pipeline"  # pressure drop along pipe + end discharge
    PIPELINE_BLOWDOWN = "pipeline_blowdown"  # multi-segment transient depressurization


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
        dynamic_props: Use CoolProp for dynamic Z/k (default False).
        mole_fractions: Dict of component -> mole fraction for CoolProp.
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

    # Dynamic EOS (Feature 3)
    dynamic_props: bool = False
    mole_fractions: Dict[str, float] | None = None


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
    Z: float = 1.0
    k: float = 1.4


# ══════════════════════════════════════════════════════════════════════════════
# Feature 1: Multi-Segment Pipeline Blowdown
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineSegment:
    """Single pipeline segment for multi-segment blowdown.

    Attributes:
        L: Segment length [m].
        D: Segment internal diameter [m].
        roughness: Absolute roughness [m].
        wall_thickness: Wall thickness [m].
        wall_density: Wall material density [kg/m³] (default 7850 for steel).
        wall_cp: Wall specific heat [J/(kg·K)] (default 500 for steel).
    """
    L: float
    D: float
    roughness: float = 0.0457e-3
    wall_thickness: float = 0.01
    wall_density: float = 7850.0
    wall_cp: float = 500.0


@dataclass
class PipelineBlowdownInput:
    """Input for multi-segment pipeline blowdown (PDE-like model).

    Attributes:
        segments: List of PipelineSegment objects.
        P_initial: Initial pressure [Pa].
        T_initial: Initial temperature [K].
        P_ambient: Ambient pressure [Pa].
        t_max: Maximum simulation time [s].
        n_time_steps: Number of output time points.
        molecular_weight: Gas molecular weight [kg/mol].
        cp_cv_ratio: Specific heat ratio k = Cp/Cv [-].
        Z: Compressibility factor [-] (initial, overridden if dynamic_props=True).
        Cd: Discharge coefficient [-].
        orifice_d: Orifice diameter [m].
        mole_fractions: Dict of component -> mole fraction for CoolProp.
        dynamic_props: Use CoolProp for dynamic Z/k.
        wall_htc: Wall-to-gas heat transfer coefficient [W/(m²·K)].
    """
    segments: List[PipelineSegment]
    P_initial: float
    T_initial: float
    P_ambient: float = 101325.0
    t_max: float = 600.0
    n_time_steps: int = 200
    molecular_weight: float = 0.0289647
    cp_cv_ratio: float = 1.4
    Z: float = 1.0
    Cd: float = 0.62
    orifice_d: float = 0.05
    mole_fractions: Dict[str, float] | None = None
    dynamic_props: bool = False
    wall_htc: float = 0.0
    wall_cp: float = 500.0


@dataclass
class PipelineBlowdownResult:
    """Results from multi-segment pipeline blowdown.

    Attributes:
        t: Time array [s].
        P_segments: Pressure per segment over time [Pa] (shape: n_time × n_segments).
        T_segments: Temperature per segment over time [K].
        T_wall_segments: Wall temperature per segment over time [K].
        mdot: Discharge mass flow rate over time [kg/s].
        total_released: Total mass released [kg].
        t_final: Final time [s].
        messages: Info/warning strings.
    """
    t: np.ndarray
    P_segments: np.ndarray
    T_segments: np.ndarray
    T_wall_segments: np.ndarray
    mdot: np.ndarray
    total_released: float
    t_final: float
    messages: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Feature 2: Two-Phase Flashing Flow
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FlashingPipeInput:
    """Input for two-phase flashing flow with phase tracking.

    Attributes:
        L: Pipe length [m].
        D: Pipe internal diameter [m].
        roughness: Pipe absolute roughness [m].
        P_up: Upstream pressure [Pa].
        P_down: Downstream pressure [Pa].
        T: Fluid temperature [K].
        molecular_weight: [kg/mol].
        cp_cv_ratio: Specific heat ratio [-].
        Cd: Discharge coefficient [-].
        x_mass: Inlet mass quality [-].
        rho_l: Liquid density [kg/m³].
        rho_g: Gas density [kg/m³] (if not provided, calculated).
        mu_l: Liquid viscosity [Pa·s].
        mu_g: Gas viscosity [Pa·s].
        mole_fractions: Dict of component -> mole fraction for CoolProp.
        dynamic_props: Use CoolProp for flash calculations.
        flash_to_ambient: Flash all the way to ambient pressure.
        n_points: Number of points along pipe for profile.
    """
    L: float
    D: float
    roughness: float = 0.0457e-3
    P_up: float = 5e5
    P_down: float = 101325.0
    T: float = 300.0
    molecular_weight: float = 0.0289647
    cp_cv_ratio: float = 1.4
    Cd: float = 0.62
    x_mass: float = 0.0
    rho_l: float = 800.0
    rho_g: float | None = None
    mu_l: float = 1e-3
    mu_g: float = 1.8e-5
    mole_fractions: Dict[str, float] | None = None
    dynamic_props: bool = False
    flash_to_ambient: bool = True
    n_points: int = 50


@dataclass
class FlashingPipeResult:
    """Results from two-phase flashing pipe flow.

    Attributes:
        mdot: Mass flow rate [kg/s].
        P_profile: Pressure profile along pipe [Pa].
        T_profile: Temperature profile along pipe [K].
        quality_profile: Vapor quality profile along pipe [-].
        phase_profile: Phase at each point ('gas', 'liquid', 'two_phase').
        flashing_fraction: Fraction of flow that flashes to vapor [-].
        is_choked: Whether flow is choked.
        messages: Info/warning strings.
    """
    mdot: float
    P_profile: np.ndarray
    T_profile: np.ndarray
    quality_profile: np.ndarray
    phase_profile: list
    flashing_fraction: float
    is_choked: bool
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
                         f"Use 'full_bore', 'hole_in_pipe', 'long_pipeline', or 'pipeline_blowdown'.")

    # Dynamic EOS: compute Z and k from CoolProp if enabled (Feature 3)
    Z_final = 1.0
    k_final = 1.4
    if inputs.dynamic_props and inputs.mole_fractions and _COOLPROP_AVAILABLE:
        Z_final, k_final, _ = _coolprop_flash(
            inputs.mole_fractions, inputs.P_up, inputs.T, fallback_Z=inputs.Z, fallback_k=inputs.cp_cv_ratio
        )

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
        Z=Z_final,
        k=k_final,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Feature 3: CoolProp Full Flash Integration
# ══════════════════════════════════════════════════════════════════════════════

def _coolprop_flash(
    mole_fractions: Dict[str, float],
    P: float,
    T: float,
    fallback_Z: float = 1.0,
    fallback_k: float = 1.4,
) -> Tuple[float, float, Dict[str, any]]:
    """Perform CoolProp PT flash to get fluid properties.

    Args:
        mole_fractions: Dict of component name -> mole fraction.
        P: Pressure [Pa].
        T: Temperature [K].
        fallback_Z: Default Z if CoolProp fails.
        fallback_k: Default k if CoolProp fails.

    Returns:
        Tuple of (Z, k, properties_dict) where properties_dict contains:
        - phase: 'gas', 'liquid', or 'two_phase'
        - rho: density [kg/m³]
        - mu: viscosity [Pa·s]
        - k: thermal conductivity [W/(m·K)]
        - cp, cv: specific heats [J/(kg·K)]
        - quality: vapor quality (-1 for single-phase)
        - T_dew, T_bubble: dew/bubble points [K]
    """
    result = {
        "phase": "unknown",
        "rho": 0.0,
        "mu": 0.0,
        "k": 0.0,
        "cp": 0.0,
        "cv": 0.0,
        "quality": -1.0,
        "T_dew": None,
        "T_bubble": None,
    }

    if not _COOLPROP_AVAILABLE:
        return fallback_Z, fallback_k, result

    try:
        comp_names = []
        fracs = []
        for comp_name, frac in sorted(mole_fractions.items()):
            if frac > 0:
                comp_names.append(comp_name)
                fracs.append(frac)

        if len(comp_names) == 0:
            return fallback_Z, fallback_k, result

        # Create AbstractState
        if len(comp_names) == 1:
            AS = CP_AbstractState("HEOS", comp_names[0])
        else:
            AS = CP_AbstractState("HEOS", "&".join(comp_names))
            AS.set_mole_fractions(fracs)

        # PT flash
        AS.update(CoolProp.PT_INPUTS, P, T)

        # Get properties
        result["phase"] = AS.phase() if hasattr(AS, 'phase') else ("gas" if AS.rhomolar() < 100 else "liquid")
        result["rho"] = AS.rhomolar() * AS.molar_mass()  # kg/m³
        result["mu"] = AS.viscosity()  # Pa·s
        result["k"] = AS.conductivity()  # W/(m·K)
        result["cp"] = AS.cpmolar() / AS.molar_mass()  # J/(kg·K)
        result["cv"] = AS.cvmolar() / AS.molar_mass()  # J/(kg·K)
        result["quality"] = AS.quality() if hasattr(AS, 'quality') else -1.0

        # Get Z and k
        Z_val = AS.compressibility_factor()
        cp_val = result["cp"]
        cv_val = result["cv"]
        k_val = cp_val / cv_val if cv_val > EPSILON else fallback_k

        # Validate
        if not (0.01 < Z_val < 10.0):
            Z_val = fallback_Z
        if not (1.001 < k_val < 5.0):
            k_val = fallback_k

        return Z_val, k_val, result

    except Exception as exc:
        # Graceful fallback on any CoolProp error
        return fallback_Z, fallback_k, result


# ══════════════════════════════════════════════════════════════════════════════
# Feature 1: Multi-Segment Pipeline Blowdown Implementation
# ══════════════════════════════════════════════════════════════════════════════

def _pipeline_blowdown_ode(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    """ODE for multi-segment pipeline blowdown.

    State vector y = [m1, U1, Tw1, m2, U2, Tw2, ..., mN, UN, TwN]
    where N = number of segments.
    For each segment i:
        mi = mass of gas in segment [kg]
        Ui = internal energy of gas in segment [J]
        Twi = wall internal energy in segment [J]

    ODEs:
        dmi/dt = mdot_in - mdot_out (mass balance)
        dUi/dt = mdot_in*hin - mdot_out*hout + Q_wall (energy balance)
        dTwi/dt = Q_amb - Q_wall (wall energy balance)

    Flow between segments via orifice equation (choked/subsonic).
    Wall thermal inertia from wall mass and specific heat.
    """
    N = params["N"]
    n_states = N * 3

    # Reshape state: [N, 3] -> [[m, U, Tw], ...]
    state_reshaped = y.reshape((N, 3))

    m = state_reshaped[:, 0]  # masses
    U = state_reshaped[:, 1]  # gas internal energies
    U_w = state_reshaped[:, 2]  # wall internal energies

    # Extract parameters
    segment_vols = params["segment_vols"]  # m³
    segment_areas = params["segment_areas"]  # m² (wall surface area)
    MW = params["MW"]
    k = params["k"]
    Z = params["Z"]
    cv = params["cv"]
    A_orifice = params["A_orifice"]
    Cd = params["Cd"]
    wall_cp = params["wall_cp"]
    wall_masses = params["wall_masses"]  # kg
    wall_htc = params["wall_htc"]
    T_amb = params["T_amb"]
    P_amb = params["P_amb"]
    use_dynamic = params["use_dynamic"]
    mole_fracs = params.get("mole_fractions")

    # Initialize derivatives
    dydt = np.zeros(n_states)
    dmdt = np.zeros(N)
    dUdt = np.zeros(N)
    dUwdt = np.zeros(N)

    # Calculate temperatures and pressures
    T_gas = np.where(
        m > EPSILON,
        U / (m * cv),
        T_amb
    )
    T_wall = np.where(
        wall_masses > EPSILON,
        U_w / (wall_masses * wall_cp),
        T_amb
    )

    # Pressure from real gas law: P = (m * Z * R * T) / (V * MW)
    P = np.array([
        (m[i] * Z * R * T_gas[i]) / (segment_vols[i] * MW) if segment_vols[i] > EPSILON and m[i] > EPSILON else P_amb
        for i in range(N)
    ])

    # Dynamic EOS update
    if use_dynamic and mole_fracs and _COOLPROP_AVAILABLE:
        for i in range(N):
            T_clamped = max(T_gas[i], 150.0)
            Z_dyn, k_dyn, _ = _coolprop_flash(mole_fracs, P[i], T_clamped, Z, k)
            if 0.1 < Z_dyn < 5.0 and 1.001 < k_dyn < 3.0:
                P[i] = (m[i] * Z_dyn * R * T_gas[i]) / (segment_vols[i] * MW)

    # Calculate flow rates between segments and discharge
    # mdot[i] = flow from segment i to segment i+1 (or to atmosphere for last segment)
    mdot_inter = np.zeros(N)

    for i in range(N):
        if i < N - 1:
            # Inter-segment flow
            P_down_segment = P[i + 1]
        else:
            # Last segment discharges to atmosphere
            P_down_segment = P_amb

        if P[i] <= P_down_segment + EPSILON or m[i] <= EPSILON:
            mdot_inter[i] = 0.0
            continue

        # Choked check
        r_crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
        P_choked = P[i] * r_crit
        is_choked = P_down_segment < P_choked

        if is_choked:
            # Choked flow
            exponent = (k + 1.0) / (k - 1.0)
            term = k * (MW / (Z * R * T_gas[i])) * (2.0 / (k + 1.0)) ** exponent
            if term > EPSILON:
                G_flux = P[i] * math.sqrt(term)
            else:
                G_flux = 0.0
        else:
            # Subsonic flow
            pr = P_down_segment / P[i]
            ratio_term = pr ** (2.0 / k) - pr ** ((k + 1.0) / k)
            if ratio_term > EPSILON:
                factor = (2.0 * k / (k - 1.0)) * (MW / (Z * R * T_gas[i]))
                G_flux = P[i] * math.sqrt(factor * ratio_term)
            else:
                G_flux = 0.0

        mdot_inter[i] = Cd * A_orifice * G_flux

    # Mass balance: dmi/dt = mdot_in - mdot_out
    # First segment: only outflow
    dmdt[0] = -mdot_inter[0]

    # Middle segments: inflow - outflow
    for i in range(1, N):
        dmdt[i] = mdot_inter[i - 1] - mdot_inter[i]

    # Energy balance: dUi/dt = mdot_in*hin - mdot_out*hout + Q_wall
    cp = cv * k

    for i in range(N):
        h_out = cp * T_gas[i]
        h_in = cp * T_gas[i - 1] if i > 0 else h_out

        # Heat transfer from wall to gas
        if wall_htc > EPSILON and segment_areas[i] > EPSILON:
            Q_wg = wall_htc * segment_areas[i] * (T_wall[i] - T_gas[i])
        else:
            Q_wg = 0.0

        dUdt[i] = (mdot_inter[i - 1] * h_in if i > 0 else 0.0) - mdot_inter[i] * h_out + Q_wg

        # Wall energy balance: Q_amb_to_wall - Q_wall_to_gas
        if segment_areas[i] > EPSILON:
            Q_aw = 0.0  # No ambient fire in this simple model
            dUwdt[i] = Q_aw - Q_wg
        else:
            dUwdt[i] = 0.0

    # Pack derivatives
    for i in range(N):
        dydt[i * 3] = dmdt[i]
        dydt[i * 3 + 1] = dUdt[i]
        dydt[i * 3 + 2] = dUwdt[i]

    return dydt


def calculate_pipeline_blowdown(inputs: PipelineBlowdownInput) -> PipelineBlowdownResult:
    """Calculate multi-segment pipeline blowdown (Feature 1).

    Args:
        inputs: PipelineBlowdownInput dataclass.

    Returns:
        PipelineBlowdownResult with time histories.

    Raises:
        ImportError: If scipy is not available.
    """
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        raise ImportError("scipy.integrate.solve_ivp is required for pipeline_blowdown.")

    N = len(inputs.segments)
    if N == 0:
        raise ValueError("At least one segment is required.")

    # Compute segment properties
    segment_vols = np.array([
        math.pi * (seg.D / 2.0) ** 2 * seg.L
        for seg in inputs.segments
    ])

    segment_areas = np.array([
        math.pi * seg.D * seg.L  # Wall surface area (simplified)
        for seg in inputs.segments
    ])

    # Wall masses per segment
    wall_masses = np.array([
        math.pi * ((seg.D + 2 * seg.wall_thickness) ** 2 - seg.D ** 2) / 4.0 * seg.L * seg.wall_density
        for seg in inputs.segments
    ])

    # Specific heats
    R_specific = R / inputs.molecular_weight
    cv = inputs.Z * R_specific / (inputs.cp_cv_ratio - 1.0) if inputs.cp_cv_ratio > 1.001 else R_specific / 0.4

    # Initial conditions
    m0 = np.array([
        (inputs.P_initial * segment_vols[i] * inputs.molecular_weight) / (inputs.Z * R * inputs.T_initial)
        for i in range(N)
    ])

    U0 = m0 * cv * inputs.T_initial
    Uw0 = wall_masses * inputs.wall_cp * inputs.T_initial

    # Pack initial state: [m1, U1, Tw1, m2, U2, Tw2, ...]
    y0 = np.zeros(N * 3)
    for i in range(N):
        y0[i * 3] = m0[i]
        y0[i * 3 + 1] = U0[i]
        y0[i * 3 + 2] = Uw0[i]

    # ODE parameters
    A_orifice = math.pi * (inputs.orifice_d / 2.0) ** 2

    params = {
        "N": N,
        "segment_vols": segment_vols,
        "segment_areas": segment_areas,
        "MW": inputs.molecular_weight,
        "k": inputs.cp_cv_ratio,
        "Z": inputs.Z,
        "cv": cv,
        "A_orifice": A_orifice,
        "Cd": inputs.Cd,
        "wall_cp": inputs.wall_cp,
        "wall_masses": wall_masses,
        "wall_htc": inputs.wall_htc,
        "T_amb": inputs.T_initial,  # Simplified: ambient = initial T
        "P_amb": inputs.P_ambient,
        "use_dynamic": inputs.dynamic_props and inputs.mole_fractions is not None,
        "mole_fractions": inputs.mole_fractions,
    }

    # Solve ODE
    sol = solve_ivp(
        _pipeline_blowdown_ode,
        t_span=(0.0, inputs.t_max),
        y0=y0,
        args=(params,),
        method="RK45",
        max_step=inputs.t_max / inputs.n_time_steps,
        rtol=1e-6,
        atol=1e-9,
        dense_output=True,
    )

    # Interpolate onto uniform time grid
    t_eval = np.linspace(0.0, sol.t[-1], inputs.n_time_steps)
    y_interp = sol.sol(t_eval)

    # Extract results
    P_segments = np.zeros((inputs.n_time_steps, N))
    T_segments = np.zeros((inputs.n_time_steps, N))
    T_wall_segments = np.zeros((inputs.n_time_steps, N))
    mdot = np.zeros(inputs.n_time_steps)

    for i in range(inputs.n_time_steps):
        for j in range(N):
            m_ij = y_interp[j * 3, i]
            U_ij = y_interp[j * 3 + 1, i]
            Uw_ij = y_interp[j * 3 + 2, i]

            T_ij = U_ij / (m_ij * cv) if m_ij > EPSILON else inputs.T_initial
            Tw_ij = Uw_ij / (wall_masses[j] * inputs.wall_cp) if wall_masses[j] > EPSILON else inputs.T_initial

            T_segments[i, j] = T_ij
            T_wall_segments[i, j] = Tw_ij

            # Pressure
            if segment_vols[j] > EPSILON and m_ij > EPSILON:
                P_segments[i, j] = (m_ij * inputs.Z * R * T_ij) / (segment_vols[j] * inputs.molecular_weight)
            else:
                P_segments[i, j] = inputs.P_ambient

        # Discharge from last segment
        P_last = P_segments[i, -1]
        T_last = T_segments[i, -1]
        if P_last > inputs.P_ambient + EPSILON:
            r_crit = (2.0 / (inputs.cp_cv_ratio + 1.0)) ** (inputs.cp_cv_ratio / (inputs.cp_cv_ratio - 1.0))
            P_choked = P_last * r_crit
            is_choked = inputs.P_ambient < P_choked

            if is_choked:
                exponent = (inputs.cp_cv_ratio + 1.0) / (inputs.cp_cv_ratio - 1.0)
                term = inputs.cp_cv_ratio * (inputs.molecular_weight / (inputs.Z * R * T_last)) * (2.0 / (inputs.cp_cv_ratio + 1.0)) ** exponent
                if term > EPSILON:
                    G_flux = P_last * math.sqrt(term)
                else:
                    G_flux = 0.0
            else:
                pr = inputs.P_ambient / P_last
                ratio_term = pr ** (2.0 / inputs.cp_cv_ratio) - pr ** ((inputs.cp_cv_ratio + 1.0) / inputs.cp_cv_ratio)
                if ratio_term > EPSILON:
                    factor = (2.0 * inputs.cp_cv_ratio / (inputs.cp_cv_ratio - 1.0)) * (inputs.molecular_weight / (inputs.Z * R * T_last))
                    G_flux = P_last * math.sqrt(factor * ratio_term)
                else:
                    G_flux = 0.0

            mdot[i] = inputs.Cd * A_orifice * G_flux

    # Total mass released
    total_released = float(np.sum(m0) - np.sum(y_interp[0::3, -1]))

    messages = ["Multi-segment pipeline blowdown"]
    if inputs.dynamic_props and inputs.mole_fractions:
        messages.append("Dynamic EOS: CoolProp")

    return PipelineBlowdownResult(
        t=t_eval,
        P_segments=P_segments,
        T_segments=T_segments,
        T_wall_segments=T_wall_segments,
        mdot=mdot,
        total_released=total_released,
        t_final=float(t_eval[-1]),
        messages=messages,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Feature 2: Two-Phase Flashing Flow Implementation
# ══════════════════════════════════════════════════════════════════════════════

def calculate_flashing_pipe_flow(inputs: FlashingPipeInput) -> FlashingPipeResult:
    """Calculate two-phase flashing flow with phase tracking (Feature 2).

    Uses CoolProp to perform PT flash along the pipe length,
    tracking vapor quality and phase transitions.

    Args:
        inputs: FlashingPipeInput dataclass.

    Returns:
        FlashingPipeResult with quality profile and phase information.
    """
    L = inputs.L
    D = inputs.D
    A = math.pi * (D / 2.0) ** 2
    n_points = inputs.n_points

    x_profile = np.linspace(0.0, L, n_points + 1)
    P_profile = np.zeros(n_points + 1)
    T_profile = np.zeros(n_points + 1)
    quality_profile = np.zeros(n_points + 1)
    phase_profile = []

    # Initial conditions
    P_profile[0] = inputs.P_up
    T_profile[0] = inputs.T
    quality_profile[0] = inputs.x_mass

    # Fluid properties
    if inputs.rho_g is None:
        rho_g = (inputs.P_up * inputs.molecular_weight) / (R * inputs.T)
    else:
        rho_g = inputs.rho_g

    rho_l = inputs.rho_l
    mu_l = inputs.mu_l
    mu_g = inputs.mu_g

    messages = []
    flashing_occurred = False
    flashing_fraction = 0.0

    # Calculate flow along pipe
    # Simplified: assume constant mass flow rate, track phase change
    mdot = 0.0
    is_choked = False

    for i in range(n_points):
        # Pressure drop (simplified linear for now)
        # More sophisticated: use Lockhart-Martinelli for two-phase
        P_prev = P_profile[i]
        T_prev = T_profile[i]
        x_prev = quality_profile[i]

        # Determine phase at current conditions
        if inputs.dynamic_props and inputs.mole_fractions and _COOLPROP_AVAILABLE:
            Z, k, props = _coolprop_flash(inputs.mole_fractions, P_prev, T_prev)
            phase = props["phase"]
            quality = props["quality"] if props["quality"] >= 0 else x_prev
            rho_current = props["rho"] if props["rho"] > EPSILON else (rho_l if x_prev < 0.5 else rho_g)
            mu_current = props["mu"] if props["mu"] > EPSILON else (mu_l if x_prev < 0.5 else mu_g)
        else:
            # Simple phase determination
            if x_prev <= EPSILON:
                phase = "liquid"
                quality = 0.0
                rho_current = rho_l
                mu_current = mu_l
            elif x_prev >= 1.0 - EPSILON:
                phase = "gas"
                quality = 1.0
                rho_current = rho_g
                mu_current = mu_g
            else:
                phase = "two_phase"
                quality = x_prev
                # Homogeneous equilibrium model
                v_mix = quality * (1.0 / rho_g) + (1.0 - quality) * (1.0 / rho_l)
                rho_current = 1.0 / v_mix if v_mix > EPSILON else rho_l
                mu_current = mu_l * (1.0 - quality) + mu_g * quality  # Simple mixing

        phase_profile.append(phase)
        quality_profile[i] = quality

        # Check for flashing (liquid -> two-phase or gas)
        # For simplicity, use critical pressure ratio check
        # More sophisticated would use bubble point from CoolProp
        k = inputs.cp_cv_ratio
        r_crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
        P_choked = P_prev * r_crit

        if inputs.P_down < P_choked and phase == "liquid":
            # Flashing may occur
            flashing_occurred = True

        # Calculate pressure drop to next point
        dx = L / n_points
        eps_D = inputs.roughness / D if D > EPSILON else 0.0

        # Two-phase multiplier
        if quality > EPSILON and quality < 1.0 - EPSILON:
            X_tt, Phi_sq = lockhart_martinelli_multiplier(
                quality, rho_l, rho_g, mu_l, mu_g
            )
            Phi = math.sqrt(Phi_sq)
        else:
            Phi = 1.0

        # Estimate velocity (need mass flow rate)
        # For first iteration, guess based on inlet conditions
        if i == 0:
            v_mix = quality * (1.0 / rho_g) + (1.0 - quality) * (1.0 / rho_l)
            rho_mix = 1.0 / v_mix if v_mix > EPSILON else rho_l

            # Estimate from Bernoulli
            dp_max = inputs.P_up - inputs.P_down
            v_guess = math.sqrt(2.0 * dp_max / rho_mix)
            mdot = rho_mix * A * v_guess

        vel = mdot / (rho_current * A) if rho_current * A > EPSILON else 1.0
        Re = rho_current * vel * D / mu_current
        f = colebrook_friction_factor(Re, eps_D)

        # Pressure drop
        dP = Phi * f * (dx / D) * (rho_current * vel * vel / 2.0)
        P_next = max(inputs.P_down, P_prev - dP)

        # Estimate temperature change (isentropic expansion approximation)
        if phase == "gas":
            T_next = T_prev * (P_next / P_prev) ** ((k - 1.0) / k)
        else:
            T_next = T_prev  # Liquid T roughly constant

        P_profile[i + 1] = P_next
        T_profile[i + 1] = max(T_next, 150.0)  # Minimum T

        # Update quality if flashing (simplified)
        if phase == "liquid" and P_next < P_choked:
            # Estimate flash fraction from energy balance
            # For now, simple linear approximation
            x_new = (P_prev - P_next) / (P_prev - inputs.P_down) * 0.5  # Max 50% flash
            x_new = min(1.0, x_new)
            quality_profile[i + 1] = x_new
            flashing_fraction = max(flashing_fraction, x_new)
        else:
            quality_profile[i + 1] = quality

    # Calculate final mdot
    phase_profile.append(phase_profile[-1] if phase_profile else "liquid")

    # Check choking at exit
    k = inputs.cp_cv_ratio
    r_crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
    P_choked = P_up = inputs.P_up * r_crit
    is_choked = inputs.P_down < P_choked

    if flashing_occurred:
        messages.append(f"Flashing detected: {flashing_fraction:.1%} of flow flashes to vapor")
    if is_choked:
        messages.append("Flow is choked at exit")
    if inputs.dynamic_props and inputs.mole_fractions:
        messages.append("Dynamic EOS: CoolProp")

    return FlashingPipeResult(
        mdot=mdot,
        P_profile=P_profile,
        T_profile=T_profile,
        quality_profile=quality_profile,
        phase_profile=phase_profile,
        flashing_fraction=flashing_fraction,
        is_choked=is_choked,
        messages=messages,
    )
