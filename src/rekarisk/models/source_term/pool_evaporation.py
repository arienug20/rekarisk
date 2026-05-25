"""
Rekarisk — Liquid Pool Spreading & Evaporation Models.

Calculates liquid pool behavior after a spill: spreading dynamics
(gravity-inertia → gravity-viscous transition), evaporation rate
(mass transfer with wind effect), cryogenic pool special cases,
and boiling vs evaporating regime determination.

Models:
  - Spreading: gravity-inertia regime (Briscoe & Shaw, 1980)
               gravity-viscous regime (Hoult, 1972)
  - Evaporation: mass transfer coefficient method (Sutton, Mackay-Matsugu)
  - Cryogenic pools: modified heat transfer (Kunsch, 1998)
  - Bunded pools: area-limited spreading

References:
  - CCPS Guidelines for Consequence Analysis (1999), Chapter 2
  - TNO Yellow Book (CPR 14E), Chapter 3
  - Briscoe & Shaw (1980), Prog. Energy Combust. Sci. 6(2), 127-133
  - Mackay & Matsugu (1973), Can. J. Chem. Eng. 51(6), 642-646
  - Kunsch (1998), J. Loss Prev. Proc. Ind. 11, 215-224
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from ...core.constants import R, P_ATM, G as GRAVITY, EPSILON, T_0C


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class PoolSurface(str, Enum):
    """Surface type for pool spreading."""
    LAND = "land"
    WATER = "water"
    CONCRETE = "concrete"


class PoolRegime(str, Enum):
    """Pool spreading regime."""
    GRAVITY_INERTIA = "gravity_inertia"
    GRAVITY_VISCOUS = "gravity_viscous"
    SURFACE_TENSION = "surface_tension"
    BUNDED = "bunded"            # area-limited by bund/dike
    EVAPORATING = "evaporating"
    BOILING = "boiling"


# ══════════════════════════════════════════════════════════════════════════════
# Input/Output Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PoolInput:
    """Input parameters for pool spreading and evaporation calculation.

    Attributes:
        substance: Substance identifier (name or formula).
        spill_mass: Total mass spilled [kg].
        T_ambient: Ambient temperature [K].
        wind_speed: Wind speed at 10 m height [m/s].
        surface: Surface type ('land', 'water', 'concrete').
        bunded_area: If bunded/diked, pool area [m²] (None for unconfined).
        duration: Spill duration [s] (0 for instantaneous).
        rho_l: Liquid density [kg/m³].
        mu_l: Liquid dynamic viscosity [Pa·s].
        surface_tension: Liquid surface tension [N/m].
        molecular_weight: [kg/mol].
        vapor_pressure: Liquid vapor pressure at T_ambient [Pa].
        heat_of_vaporization: Latent heat [J/kg].
        boiling_point: Normal boiling point [K].
        cp_liquid: Liquid specific heat [J/(kg·K)].
        diffusion_coeff: Diffusion coefficient in air [m²/s].
        T_ground: Ground/substrate temperature [K] (default T_ambient).
        substrate_thermal_cond: [W/(m·K)] for heat conduction.
        emissivity: Liquid surface emissivity (default 0.95).
    """
    substance: str = "generic"
    spill_mass: float = 1000.0       # [kg]
    T_ambient: float = 298.15         # [K]
    wind_speed: float = 3.0           # [m/s]
    surface: str = "land"
    bunded_area: float | None = None  # [m²]
    duration: float = 0.0             # [s] (instantaneous)

    # Physical properties
    rho_l: float = 1000.0            # [kg/m³]
    mu_l: float = 0.001              # [Pa·s]
    surface_tension: float = 0.073   # [N/m]
    molecular_weight: float = 0.018  # [kg/mol]
    vapor_pressure: float | None = None  # [Pa]
    heat_of_vaporization: float = 2.26e6  # [J/kg]
    boiling_point: float = 373.15    # [K]
    cp_liquid: float = 4184.0        # [J/(kg·K)]
    diffusion_coeff: float | None = None  # [m²/s]

    # Environmental
    T_ground: float | None = None
    substrate_thermal_cond: float = 1.0  # [W/(m·K)]
    emissivity: float = 0.95

    # Numerical
    n_time_steps: int = 100
    t_max: float = 600.0             # [s]


@dataclass
class PoolResult:
    """Results from pool spreading and evaporation calculation.

    Attributes:
        t: Time array [s].
        pool_radius: Pool radius array [m] (circular pool assumed).
        pool_area: Pool area array [m²].
        pool_thickness: Pool thickness array [m].
        evap_rate: Evaporation rate array [kg/s].
        total_evaporated: Total mass evaporated [kg].
        mass_remaining: Mass remaining in pool [kg].
        pool_regime: Final pool regime.
        avg_evap_rate: Average evaporation rate [kg/(m²·s)].
        messages: Info/warning strings.
    """
    t: np.ndarray
    pool_radius: np.ndarray
    pool_area: np.ndarray
    pool_thickness: np.ndarray
    evap_rate: np.ndarray
    total_evaporated: float
    mass_remaining: float
    pool_regime: str
    avg_evap_rate: float = 0.0
    messages: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Pool Spreading Models
# ══════════════════════════════════════════════════════════════════════════════

def gravity_inertia_radius(
    t: float,
    V0: float,
    g_prime: float,
) -> float:
    """Pool radius in gravity-inertia spreading regime.

    For an instantaneous spill on land:
        r(t) = 1.14 * (g' * V0)^(1/4) * t^(1/2)

    where g' = g * (ρ_l - ρ_air) / ρ_l ≈ g  (for liquid >> air density)

    Reference: Briscoe & Shaw (1980)

    Args:
        t: Time since spill [s].
        V0: Initial spill volume [m³].
        g_prime: Reduced gravity [m/s²] (≈ g for dense liquids).

    Returns:
        Pool radius [m].
    """
    if t <= 0 or V0 <= EPSILON:
        return 0.0
    return 1.14 * (g_prime * V0) ** 0.25 * math.sqrt(t)


def gravity_viscous_radius(
    t: float,
    V0: float,
    g_prime: float,
    nu: float,
) -> float:
    """Pool radius in gravity-viscous spreading regime.

    For an instantaneous spill on land:
        r(t) = 0.89 * (g' * V0² / ν)^(1/6) * t^(1/6)

    Reference: Hoult (1972), Fay (1971)

    Args:
        t: Time since spill [s].
        V0: Initial spill volume [m³].
        g_prime: Reduced gravity [m/s²].
        nu: Kinematic viscosity of liquid [m²/s].

    Returns:
        Pool radius [m].
    """
    if t <= 0 or V0 <= EPSILON or nu <= EPSILON:
        return 0.0
    return 0.89 * (g_prime * V0 * V0 / nu) ** (1.0 / 6.0) * t ** (1.0 / 6.0)


def transition_time(
    V0: float,
    g_prime: float,
    nu: float,
) -> float:
    """Estimate time of transition between spreading regimes.

    The transition from gravity-inertia to gravity-viscous occurs when
    the two spreading laws predict the same radius.

    Args:
        V0: Spill volume [m³].
        g_prime: Reduced gravity [m/s²].
        nu: Kinematic viscosity [m²/s].

    Returns:
        Transition time [s].
    """
    if V0 <= EPSILON or g_prime <= EPSILON or nu <= EPSILON:
        return 100.0  # default
    # Equal radius condition:
    # 1.14 * (g'*V0)^(1/4) * t^(1/2) = 0.89 * (g'*V0²/ν)^(1/6) * t^(1/6)
    # → t^(1/2 - 1/6) = t^(1/3) = 0.89/1.14 * (g'*V0²/ν)^(1/6) / (g'*V0)^(1/4)
    ratio_coeff = 0.89 / 1.14
    term1 = (g_prime * V0 * V0 / nu) ** (1.0 / 6.0)
    term2 = (g_prime * V0) ** 0.25
    t_cubic = ratio_coeff * term1 / term2
    return t_cubic ** 3 if t_cubic > 0 else 100.0


def minimum_pool_thickness(
    surface: str,
) -> float:
    """Minimum stable pool thickness based on surface type.

    Typical values:
      - Land/concrete: 0.005 m (5 mm) for rough surfaces
      - Water: 0.001 m (1 mm) — liquid spreads very thin on water
      - Smooth concrete: 0.001 m

    Args:
        surface: Surface type.

    Returns:
        Minimum pool thickness [m].
    """
    if surface == "water":
        return 0.001
    elif surface == "concrete":
        return 0.002
    else:  # land
        return 0.005


# ══════════════════════════════════════════════════════════════════════════════
# Evaporation Rate Models
# ══════════════════════════════════════════════════════════════════════════════

def mass_transfer_coefficient(
    wind_speed: float,
    pool_diameter: float,
    D_ab: float,
    nu_air: float = 1.568e-5,
) -> float:
    """Calculate mass transfer coefficient for pool evaporation.

    Uses the Mackay-Matsugu (1973) correlation:
        k_m = 0.00482 * u^0.78 * D^(-0.11) * Sc^(-0.67)

    or the simpler Sutton correlation:
        k_m = D_ab * Sh / D
        where Sh = 0.664 * Re^(1/2) * Sc^(1/3) (laminar)
              Sh = 0.037 * Re^(4/5) * Sc^(1/3) (turbulent)

    This implementation uses the CCPS recommended correlation:
        k_m = 0.002 * u_wind^(0.78) * (2*R_pool)^(-0.11)

    Args:
        wind_speed: Wind speed at 10 m [m/s].
        pool_diameter: Pool diameter (2 * radius) [m].
        D_ab: Diffusion coefficient [m²/s].
        nu_air: Kinematic viscosity of air [m²/s].

    Returns:
        Mass transfer coefficient [m/s].
    """
    if D_ab is None or D_ab <= EPSILON:
        D_ab = 1.5e-5  # default for air

    u_eff = max(wind_speed, 0.5)  # minimum wind speed
    D_eff = max(pool_diameter, 1.0)

    # Schmidt number
    Sc = nu_air / D_ab

    # Reynolds number based on pool diameter
    Re_pool = u_eff * D_eff / nu_air

    # Sherwood number (turbulent correlation for outdoor pools)
    if Re_pool < 1e5:
        Sh = 0.664 * math.sqrt(Re_pool) * Sc ** (1.0 / 3.0)
    else:
        Sh = 0.037 * Re_pool ** 0.8 * Sc ** (1.0 / 3.0)

    # Mass transfer coefficient
    k_m = Sh * D_ab / D_eff

    return k_m


def evaporation_rate(
    pool_area: float,
    k_m: float,
    P_vapor: float,
    T_pool: float,
    T_ambient: float,
    MW: float,
    P_ambient: float = P_ATM,
) -> float:
    """Calculate evaporation rate from a liquid pool.

    Evaporation rate [kg/s]:
        E = k_m * A * (P_vapor/(R*T_pool) * MW - P_vapor_ambient/(R*T_ambient) * MW)

    Simplified (small ambient vapor concentration):
        E = k_m * A * MW * P_vapor / (R * T_pool)

    Args:
        pool_area: Pool surface area [m²].
        k_m: Mass transfer coefficient [m/s].
        P_vapor: Vapor pressure of liquid at pool temperature [Pa].
        T_pool: Pool temperature [K].
        T_ambient: Ambient temperature [K].
        MW: Molecular weight [kg/mol].
        P_ambient: Ambient pressure [Pa].

    Returns:
        Evaporation rate [kg/s].
    """
    if pool_area <= EPSILON or k_m <= EPSILON or P_vapor <= EPSILON:
        return 0.0

    # Vapor concentration at pool surface (saturated)
    C_sat = MW * P_vapor / (R * T_pool)

    # Vapor concentration in ambient (assumed zero for outdoor spills)
    C_amb = 0.0

    return k_m * pool_area * (C_sat - C_amb)


def boiling_evaporation_rate(
    pool_area: float,
    T_pool: float,
    T_boil: float,
    T_ground: float,
    rho_l: float,
    h_fg: float,
    substrate_k: float = 1.0,
) -> float:
    """Calculate evaporation rate for a boiling/cryogenic pool.

    When the liquid is at or near its boiling point, the evaporation rate
    is controlled by heat transfer from the ground and atmosphere.

    Heat flux from ground: q_ground = k_sub * (T_ground - T_pool) / δ
    where δ is thermal penetration depth ~ sqrt(α * t)

    For boiling pools: E = q_total / h_fg

    Args:
        pool_area: Pool area [m²].
        T_pool: Pool temperature [K].
        T_boil: Boiling point [K].
        T_ground: Ground temperature [K].
        rho_l: Liquid density [kg/m³].
        h_fg: Latent heat of vaporization [J/kg].
        substrate_k: Substrate thermal conductivity [W/(m·K)].

    Returns:
        Evaporation rate [kg/s].
    """
    if pool_area <= EPSILON or h_fg <= EPSILON:
        return 0.0

    # Heat flux from ground (conduction)
    # Simplified: q ≈ k * ΔT / (characteristic depth)
    # Characteristic depth ~ 0.01 m (thermal boundary layer)
    delta_T = max(T_ground - T_pool, 0.0)
    if delta_T <= 0:
        return 0.0

    char_depth = 0.01  # [m] thermal boundary layer estimate
    q_ground = substrate_k * delta_T / char_depth  # [W/m²]

    # Heat flux from atmosphere (convection + radiation)
    # Convection: h ~ 10 W/(m²·K) for outdoor conditions
    h_conv = 10.0
    q_conv = h_conv * (T_ground - T_pool)  # using ambient ~ ground

    # Solar radiation (simplified: 500 W/m² daytime)
    q_solar = 500.0  # typical daytime insolation

    # Total heat flux
    q_total = (q_ground + q_conv + q_solar) * pool_area

    return q_total / h_fg


# ══════════════════════════════════════════════════════════════════════════════
# Pool Spreading + Evaporation Simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_pool(inputs: PoolInput) -> PoolResult:
    """Simulate pool spreading and evaporation over time.

    Combines spreading dynamics and evaporation rate calculations
    in a time-stepping Euler integration.

    Args:
        inputs: PoolInput with all required parameters.

    Returns:
        PoolResult with time histories.

    Example:
        >>> inp = PoolInput(
        ...     substance='ammonia', spill_mass=1000,
        ...     T_ambient=298, wind_speed=3.0,
        ...     rho_l=682, boiling_point=239.8, heat_of_vaporization=1.37e6,
        ...     vapor_pressure=8.5e5, molecular_weight=0.017
        ... )
        >>> result = simulate_pool(inp)
        >>> print(f"Evaporated: {result.total_evaporated:.1f} kg")
    """
    messages = []

    # Properties
    V0 = inputs.spill_mass / inputs.rho_l  # initial volume [m³]
    g_prime = GRAVITY * (inputs.rho_l - 1.2) / inputs.rho_l  # ~ GRAVITY for dense liquids
    nu = inputs.mu_l / inputs.rho_l if inputs.rho_l > EPSILON else 1e-6

    # Determine pool regime
    T_boil = inputs.boiling_point
    T_pool_init = min(inputs.T_ambient, T_boil)  # pool can't be hotter than boiling
    is_boiling = T_pool_init >= T_boil - EPSILON
    is_cryogenic = T_boil < 273.15  # boils below 0°C at 1 atm

    # Diffusion coefficient (if not provided, estimate)
    D_ab = inputs.diffusion_coeff
    if D_ab is None:
        # Estimate using Fuller-Schettler-Giddings simplified:
        # D_ab ~ 1e-7 * T^1.75 / P  (very rough)
        D_ab = 1.5e-5  # default for typical gases in air

    # Vapor pressure
    P_vap = inputs.vapor_pressure
    if P_vap is None:
        # Rough estimate: log10(P_vap) = A - B/(T+C) (Antoine, not available)
        # Fallback: 0.1 * P_ATM for non-volatile, P_ATM for boiling
        P_vap = 0.1 * P_ATM

    T_ground = inputs.T_ground if inputs.T_ground is not None else inputs.T_ambient
    h_min = minimum_pool_thickness(inputs.surface)

    # Time stepping
    n_steps = inputs.n_time_steps
    t_arr = np.linspace(0, inputs.t_max, n_steps)
    dt = t_arr[1] - t_arr[0]

    r_arr = np.zeros(n_steps)
    A_arr = np.zeros(n_steps)
    h_arr = np.zeros(n_steps)
    evap_arr = np.zeros(n_steps)
    mass_arr = np.zeros(n_steps)

    if inputs.bunded_area is not None and inputs.bunded_area > EPSILON:
        A_max = inputs.bunded_area
        r_max = math.sqrt(A_max / math.pi)
        bunded = True
    else:
        A_max = float("inf")
        r_max = float("inf")
        bunded = False

    # Transition time
    t_trans = transition_time(V0, g_prime, nu)

    # Initial
    m = inputs.spill_mass
    R_cur = 0.1  # start small, will rapidly expand

    for i in range(n_steps):
        t = t_arr[i]

        # Pool radius (spreading)
        if bunded:
            if R_cur < r_max:
                # Spreading until bund wall reached
                if t <= t_trans:
                    R_new = gravity_inertia_radius(t + dt if t < 1e-6 else t, V0, g_prime)
                else:
                    R_new = gravity_viscous_radius(t + dt, V0, g_prime, nu)
                R_cur = min(R_new, r_max)
            else:
                R_cur = r_max
        else:
            # Unconfined spreading
            if t <= t_trans:
                R_cur = gravity_inertia_radius(max(t, 1e-6), V0, g_prime)
            else:
                R_cur = gravity_viscous_radius(max(t, 1e-6), V0, g_prime)

        # Pool area
        A_pool = math.pi * R_cur * R_cur
        if bunded:
            A_pool = min(A_pool, A_max)

        # Pool thickness
        V_pool = m / inputs.rho_l if inputs.rho_l > EPSILON else 0.0
        h_pool = V_pool / A_pool if A_pool > EPSILON else 0.0

        # If thickness < minimum, pool stops spreading, area at minimum thickness
        if h_pool < h_min and not bunded:
            A_pool = V_pool / h_min
            R_cur = math.sqrt(A_pool / math.pi)

        # Evaporation rate
        if is_boiling or is_cryogenic:
            evap = boiling_evaporation_rate(
                A_pool, T_pool_init, T_boil, T_ground,
                inputs.rho_l, inputs.heat_of_vaporization, inputs.substrate_thermal_cond
            )
        else:
            k_m = mass_transfer_coefficient(
                inputs.wind_speed,
                2.0 * R_cur,
                D_ab,
            )
            evap = evaporation_rate(
                A_pool, k_m, P_vap, T_pool_init, inputs.T_ambient, inputs.molecular_weight
            )

        # Store
        r_arr[i] = R_cur
        A_arr[i] = A_pool
        h_arr[i] = h_pool
        evap_arr[i] = evap
        mass_arr[i] = m

        # Update mass
        m = max(m - evap * dt, 0.0)

    total_evap = inputs.spill_mass - mass_arr[-1]
    avg_evap_rate = total_evap / (inputs.t_max * A_arr.mean()) if A_arr.mean() > EPSILON else 0.0

    # Determine final regime
    if bunded:
        regime = PoolRegime.BUNDED.value
    elif is_boiling:
        regime = PoolRegime.BOILING.value
    elif t_arr[-1] <= t_trans:
        regime = PoolRegime.GRAVITY_INERTIA.value
    else:
        regime = PoolRegime.GRAVITY_VISCOUS.value

    if is_cryogenic:
        messages.append("Cryogenic pool — boiling heat transfer regime applied.")

    return PoolResult(
        t=t_arr,
        pool_radius=r_arr,
        pool_area=A_arr,
        pool_thickness=h_arr,
        evap_rate=evap_arr,
        total_evaporated=total_evap,
        mass_remaining=mass_arr[-1],
        pool_regime=regime,
        avg_evap_rate=avg_evap_rate,
        messages=messages,
    )
