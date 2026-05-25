"""
Rekarisk — Vessel Blowdown (Depressurization) Models.

Simulates the time-dependent depressurization of a vessel through an orifice
using mass and energy balance ODEs. Supports gas-only, two-phase, and
API 521 simplified modes.

References:
  - API 521 §5.15 — Pressure-relieving and Depressuring Systems
  - CCPS Guidelines for Consequence Analysis (1999), Chapter 2
  - TNO Yellow Book (CPR 14E), Chapter 2
  - Haque et al. (1992) — Blowdown of Pressure Vessels, Trans IChemE Vol 70
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ...core.constants import R, P_ATM, G, EPSILON, T_0C

# Conversion factor
PSI2PA = 6894.757293168  # 1 psia in Pascals

# Note: scipy.integrate.solve_ivp is imported at the function level
# to avoid mandatory SciPy import on module load (only needed for rigorous mode).


# ══════════════════════════════════════════════════════════════════════════════
# Input/Output Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VesselInput:
    """Input parameters for vessel blowdown calculation.

    Attributes:
        V: Vessel volume [m³].
        A_wall: Vessel wall surface area [m²] for heat transfer.
        P_initial: Initial vessel pressure [Pa abs].
        T_initial: Initial vessel temperature [K].
        composition: fluid identifier or Substance reference (used for props).
        orifice_d: Orifice/hole diameter [m].
        Cd: Discharge coefficient [-].
        t_max: Maximum simulation time [s].
        P_target: Target (final) pressure [Pa abs]. Simulation stops if reached.
        phase: 'gas', 'two_phase', or 'auto' (default: 'gas').
        rho_liquid: Liquid density [kg/m³] (for two-phase).
        molecular_weight: Molecular weight [kg/mol].
        cp_cv_ratio: Specific heat ratio k = Cp/Cv [-].
        Z: Real-gas compressibility factor [-] (default 1.0 = ideal gas).
        cv: Specific heat at constant volume [J/(kg·K)].
        cp: Specific heat at constant pressure [J/(kg·K)].
        heat_of_vaporization: Latent heat [J/kg] (for two-phase).
        T_boiling: Boiling point at P_initial [K] (for two-phase).
        U_htc: Wall heat transfer coefficient [W/(m²·K)] (default 0).
        T_ambient: Ambient temperature [K] for wall heat transfer.
        T_min: Minimum plausible temperature [K] (default 50 K = -370°F).
        mode: 'rigorous' (ODE) or 'api521' (simplified).
        n_time_steps: Number of output time points (default 100).
        mole_fractions: Dict[str, float] for CoolProp mixture (dynamic Z/k).
        dynamic_props: If True, compute Z and k dynamically via CoolProp PR EOS.
    """
    V: float
    A_wall: float
    P_initial: float
    T_initial: float
    composition: str = "air"
    orifice_d: float = 0.01
    Cd: float = 0.62
    t_max: float = 60.0
    P_target: float | None = None
    phase: str = "gas"

    # Fluid properties
    rho_liquid: float | None = None
    molecular_weight: float = 0.0289647  # air [kg/mol]
    cp_cv_ratio: float = 1.4
    Z: float = 1.0  # compressibility factor (initial, overridden if dynamic_props=True)
    cv: float | None = None
    cp: float | None = None
    heat_of_vaporization: float | None = None
    T_boiling: float | None = None

    # Heat transfer
    U_htc: float = 0.0
    T_ambient: float = 298.15
    T_min: float = 50.0

    # Solver
    mode: str = "rigorous"  # 'rigorous' or 'api521'
    n_time_steps: int = 100

    # Dynamic EOS (CoolProp Peng-Robinson)
    mole_fractions: dict[str, float] | None = None
    dynamic_props: bool = False


@dataclass
class VesselResult:
    """Results from vessel blowdown simulation.

    Attributes:
        t: Time array [s].
        P: Pressure array [Pa abs].
        T: Temperature array [K].
        m: Mass in vessel array [kg].
        mdot: Mass flow rate array [kg/s] (outflow, positive).
        m_remaining: Remaining mass in vessel array [kg] (same as m).
        phase_quality: Vapor mass fraction array [-] (two-phase only).
        total_mass_released: Total mass released during simulation [kg].
        t_final: Time when P_target was reached (or t_max) [s].
        events: Dict of event times.
        messages: Info/warning strings.
    """
    t: np.ndarray
    P: np.ndarray
    T: np.ndarray
    m: np.ndarray
    mdot: np.ndarray
    m_remaining: np.ndarray
    phase_quality: np.ndarray | None = None
    total_mass_released: float = 0.0
    t_final: float = 0.0
    events: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def _orifice_area(d: float) -> float:
    """Orifice area [m²]."""
    return math.pi * (d / 2.0) ** 2


def _gas_mass(V: float, P: float, T: float, MW: float, Z: float = 1.0) -> float:
    """Real gas mass in vessel [kg] using compressibility factor.

    Args:
        V: Volume [m³].
        P: Pressure [Pa].
        T: Temperature [K].
        MW: Molecular weight [kg/mol].
        Z: Compressibility factor [-].

    Returns:
        Mass [kg].
    """
    return (P * V * MW) / (Z * R * T)


def _gas_pressure(V: float, m: float, T: float, MW: float, Z: float = 1.0) -> float:
    """Pressure from real gas law [Pa]."""
    if V <= EPSILON:
        return 0.0
    return (m * Z * R * T) / (V * MW)


def _gas_choked_mass_flux(P: float, T: float, k: float, MW: float, Z: float = 1.0) -> float:
    """Choked mass flux for real gas [kg/(m²·s)].

    API 521 Eq. 25: G = P * sqrt( k*MW/(Z*R*T) * (2/(k+1))^((k+1)/(k-1)) )

    Args:
        P: Upstream pressure [Pa].
        T: Upstream temperature [K].
        k: Cp/Cv.
        MW: Molecular weight [kg/mol].
        Z: Compressibility factor [-].

    Returns:
        Mass flux [kg/(m²·s)].
    """
    if P <= EPSILON or T <= EPSILON or Z <= EPSILON:
        return 0.0
    exponent = (k + 1.0) / (k - 1.0)
    term = k * (MW / (Z * R * T)) * (2.0 / (k + 1.0)) ** exponent
    if term <= EPSILON:
        return 0.0
    return P * math.sqrt(term)


def _dynamic_eos_properties(
    mole_fracs: dict[str, float],
    P: float,
    T: float,
    fallback_k: float = 1.4,
    fallback_Z: float = 1.0,
) -> tuple[float, float, str]:
    """Compute Z and k dynamically via CoolProp HEOS (Helmholtz EOS).

    Uses CoolProp.AbstractState with HEOS backend and PT_INPUTS.
    HEOS uses set_mole_fractions() API (not REFPROP bracket syntax).
    Includes robust error handling for low-temperature / phase-boundary edge cases.
    Falls back gracefully if CoolProp is unavailable or HEOS fails.

    Args:
        mole_fracs: Dict of component name → mole fraction.
        P: Pressure [Pa].
        T: Temperature [K].
        fallback_k: Default k if computation fails.
        fallback_Z: Default Z if computation fails.

    Returns:
        Tuple of (Z, k, status_string).
    """
    try:
        import CoolProp
        from CoolProp.CoolProp import AbstractState

        # Build HEOS mixture: "COMP1&COMP2&..." + set_mole_fractions
        comp_names = []
        fracs = []
        for comp_name, frac in sorted(mole_fracs.items()):
            if frac > 0:
                comp_names.append(comp_name)
                fracs.append(frac)

        if len(comp_names) == 0:
            return fallback_Z, fallback_k, "no components"

        if len(comp_names) == 1:
            AS = AbstractState("HEOS", comp_names[0])
        else:
            AS = AbstractState("HEOS", "&".join(comp_names))
            AS.set_mole_fractions(fracs)

        # Force gas phase to avoid rhomolar < 0 at low T
        try:
            AS.specify_phase(CoolProp.iphase_gas)
        except Exception:
            pass  # some pure fluids may not support this

        AS.update(CoolProp.PT_INPUTS, P, T)

        Z_val = float(AS.compressibility_factor())

        # Validate — reject unphysical values
        if not (0.01 < Z_val < 10.0):
            return fallback_Z, fallback_k, "CoolProp: Z out of range"

        cp_molar = float(AS.cpmolar())
        cv_molar = float(AS.cvmolar())

        if cv_molar <= 0:
            return fallback_Z, fallback_k, "CoolProp: cv <= 0"

        k_val = cp_molar / cv_molar
        if not (1.001 < k_val < 5.0):
            k_val = fallback_k

        return Z_val, k_val, "CoolProp HEOS"

    except ImportError:
        return fallback_Z, fallback_k, "CoolProp unavailable"
    except Exception as exc:
        msg = str(exc)
        if "rhomolar" in msg or "density" in msg.lower():
            # EOS range violation — keep previous values
            return fallback_Z, fallback_k, "CoolProp: density out of range"
        return fallback_Z, fallback_k, f"CoolProp: {msg[:50]}"


# ══════════════════════════════════════════════════════════════════════════════
# Rigorous ODE Blowdown (Gas-Only)
# ══════════════════════════════════════════════════════════════════════════════

def _gas_blowdown_ode(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    """ODE right-hand side for gas blowdown.

    State vector y = [m, U] where:
        m = mass of gas in vessel [kg]
        U = total internal energy in vessel [J]

    ODEs:
        dm/dt = -mdot (outflow)
        dU/dt = -mdot * h_out + Q_wall

    Args:
        t: Time [s] (not used — system is autonomous).
        y: State vector [m, U].
        params: Dict with keys: V, k, MW, cv, Z, orifice_d, Cd, A_wall, U_htc,
                T_amb, P_atm.

    Returns:
        Derivatives [dm/dt, dU/dt].
    """
    m, U = y
    V = params["V"]
    k = params["k"]
    MW = params["MW"]
    cv = params["cv"]
    Z_c = params.get("Z", 1.0)
    A_hole = params["A_hole"]
    Cd = params["Cd"]
    A_wall = params["A_wall"]
    U_htc = params["U_htc"]
    T_amb = params["T_amb"]
    P_down = params["P_down"]

    if m <= EPSILON or V <= EPSILON:
        return np.array([0.0, 0.0])

    # Current state
    T = U / (m * cv) if m > EPSILON and cv > EPSILON else params["T_initial"]
    P = _gas_pressure(V, m, T, MW, Z_c)

    if P <= P_down + EPSILON:
        return np.array([0.0, 0.0])

    # Check if flow is choked
    r_crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
    P_choked = P * r_crit
    is_choked = P_down < P_choked

    # Mass flow rate
    if is_choked:
        G_flux = _gas_choked_mass_flux(P, T, k, MW, Z_c)
    else:
        # Subsonic flow
        pr = P_down / P
        ratio_term = pr ** (2.0 / k) - pr ** ((k + 1.0) / k)
        if ratio_term <= EPSILON or pr <= EPSILON:
            return np.array([0.0, 0.0])
        factor = (2.0 * k / (k - 1.0)) * (MW / (Z_c * R * T))
        G_flux = P * math.sqrt(factor * ratio_term)

    mdot = Cd * A_hole * G_flux

    # Enthalpy of exiting gas: h_out = cp * T
    cp_val = cv * k  # cp = k * cv for ideal gas
    h_out = cp_val * T

    # Heat transfer from wall
    Q_wall = U_htc * A_wall * (T_amb - T) if U_htc > EPSILON else 0.0

    # Derivatives
    dm_dt = -mdot  # mass leaving
    dU_dt = -mdot * h_out + Q_wall  # energy balance

    return np.array([dm_dt, dU_dt])


def _solve_gas_blowdown(inputs: VesselInput) -> dict:
    """Solve gas blowdown using ODE integration.

    Args:
        inputs: VesselInput dataclass.

    Returns:
        Dict with solution arrays.
    """
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        raise ImportError("scipy.integrate.solve_ivp is required for rigorous mode.")
    except Exception:
        raise

    # Compute cv from k and real-gas correction
    R_specific = inputs.Z * R / inputs.molecular_weight
    cv_val = inputs.cv if inputs.cv is not None else R_specific / (inputs.cp_cv_ratio - 1.0)

    # Initial conditions
    m0 = _gas_mass(inputs.V, inputs.P_initial, inputs.T_initial, inputs.molecular_weight, inputs.Z)
    U0 = m0 * cv_val * inputs.T_initial

    A_hole = _orifice_area(inputs.orifice_d)
    P_target = inputs.P_target if inputs.P_target is not None else P_ATM

    # Parameters for ODE
    params = {
        "V": inputs.V,
        "k": inputs.cp_cv_ratio,
        "MW": inputs.molecular_weight,
        "Z": inputs.Z,
        "cv": cv_val,
        "A_hole": A_hole,
        "Cd": inputs.Cd,
        "A_wall": inputs.A_wall,
        "U_htc": inputs.U_htc,
        "T_amb": inputs.T_ambient,
        "P_down": max(P_target, P_ATM),
        "T_initial": inputs.T_initial,
    }

    # Pressure threshold event
    def pressure_reached(t, y):
        m, U = y
        if m <= EPSILON:
            return 0.0
        T = U / (m * cv_val)
        P = _gas_pressure(inputs.V, m, T, inputs.molecular_weight)
        return P - P_target

    pressure_reached.terminal = True
    pressure_reached.direction = -1  # trigger when crossing zero from above

    # Solve ODE
    sol = solve_ivp(
        _gas_blowdown_ode,
        t_span=(0.0, inputs.t_max),
        y0=np.array([m0, U0]),
        args=(params,),
        method="RK45",
        events=pressure_reached,
        max_step=inputs.t_max / inputs.n_time_steps,
        rtol=1e-6,
        atol=1e-9,
    )

    # Extract results at uniform time points
    t_eval = np.linspace(0, sol.t[-1], inputs.n_time_steps)
    # Interpolate solution onto uniform grid
    m_interp = np.interp(t_eval, sol.t, sol.y[0])
    U_interp = np.interp(t_eval, sol.t, sol.y[1])

    # Compute derived quantities
    T_arr = np.where(m_interp > EPSILON,
                     U_interp / (m_interp * cv_val),
                     inputs.T_ambient)
    P_arr = np.array([_gas_pressure(inputs.V, m, T_in, inputs.molecular_weight)
                       for m, T_in in zip(m_interp, T_arr)])
    mdot_arr = np.zeros_like(t_eval)
    for i in range(len(t_eval)):
        if P_arr[i] <= P_target + EPSILON or m_interp[i] <= EPSILON:
            mdot_arr[i] = 0.0
            continue
        if P_arr[i] > EPSILON and T_arr[i] > EPSILON:
            mdot_arr[i] = inputs.Cd * A_hole * _gas_choked_mass_flux(
                P_arr[i], T_arr[i], inputs.cp_cv_ratio, inputs.molecular_weight
            )

    total_released = m0 - m_interp[-1]
    events_dict = {}
    if sol.t_events and len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
        events_dict["pressure_reached_at"] = float(sol.t_events[0][0])

    return {
        "t": t_eval,
        "P": P_arr,
        "T": T_arr,
        "m": m_interp,
        "mdot": mdot_arr,
        "m_remaining": m_interp,
        "total_mass_released": float(total_released),
        "t_final": float(t_eval[-1]),
        "events": events_dict,
        "phase_quality": None,
        "messages": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# API 521 Simplified Blowdown
# ══════════════════════════════════════════════════════════════════════════════

def _solve_api521_blowdown(inputs: VesselInput) -> dict:
    """Simplified blowdown using API 521 with dynamic real-gas correction.

    Physical model:
      1. Mass balance:  dm/dt = -Cd·A_orifice·G_choked_or_subsonic(P,T,Z,k)
      2. Energy:  T_new from isentropic relation + wall heat transfer
      3. Real gas P:  P = (m·Z·R·T)/(V·MW)

    If inputs.dynamic_props=True and mole_fractions provided:
      Z and k are recomputed at every timestep via CoolProp PT flash.

    Args:
        inputs: VesselInput dataclass.

    Returns:
        Dict with solution arrays (includes Z_arr, k_arr for diagnostics).
    """
    k_const = inputs.cp_cv_ratio
    Z_const = inputs.Z
    MW = inputs.molecular_weight
    A_hole = _orifice_area(inputs.orifice_d)
    Cd = inputs.Cd
    V = inputs.V
    P0 = inputs.P_initial
    T0 = inputs.T_initial
    P_target = inputs.P_target if inputs.P_target is not None else P_ATM

    use_dynamic = inputs.dynamic_props and inputs.mole_fractions is not None

    # ---- Initial Z and k (CoolProp or user-supplied) ----
    eos_status = "constant"
    if use_dynamic:
        Z_c, k_c, eos_status = _dynamic_eos_properties(
            inputs.mole_fractions, P0, T0, k_const, Z_const)
    else:
        Z_c, k_c = Z_const, k_const

    # Initial mass
    m0 = _gas_mass(V, P0, T0, MW, Z_c)

    # Time stepping
    n_steps = inputs.n_time_steps
    dt = inputs.t_max / n_steps

    max_steps = n_steps * 10
    t_list = [0.0]
    m_list = [m0]
    T_list = [T0]
    P_list = [P0]
    mdot_list = [0.0]
    Z_list = [Z_c]
    k_list = [k_c]

    m = m0
    T = T0
    P = P0

    t = 0.0
    step_count = 0
    reached_target = False
    cp_calls = 1

    while t < inputs.t_max and step_count < max_steps:
        step_count += 1

        if P <= P_target + EPSILON or m <= EPSILON:
            reached_target = True
            break

        # ---- Choking pressure ratio ----
        r_crit = (2.0 / (k_c + 1.0)) ** (k_c / (k_c - 1.0))

        # ---- Flow regime check ----
        P_choked_current = P * r_crit
        is_choked = P_target < P_choked_current

        # ---- Mass flux at current (Z,k) ----
        if is_choked:
            G_flux = _gas_choked_mass_flux(P, T, k_c, MW, Z_c)
        else:
            pr = max(P_target / P, EPSILON)
            if pr >= 1.0 - EPSILON:
                G_flux = 0.0
            else:
                ratio_term = pr ** (2.0 / k_c) - pr ** ((k_c + 1.0) / k_c)
                if ratio_term <= EPSILON:
                    G_flux = 0.0
                else:
                    factor = (2.0 * k_c / (k_c - 1.0)) * (MW / (Z_c * R * T))
                    G_flux = P * math.sqrt(max(factor * ratio_term, 0.0))

        mdot_i = Cd * A_hole * G_flux

        # ---- Adaptive sub-stepping ----
        max_dm_per_step = 0.05 * m
        max_dt = max_dm_per_step / max(mdot_i, EPSILON)
        dt_eff = min(dt, max_dt)
        dt_eff = max(dt_eff, dt * 0.005)

        # ---- Mass step ----
        dm = -mdot_i * dt_eff
        m_new = m + dm
        if m_new <= EPSILON:
            m_new = EPSILON

        # ---- Specific heats ----
        Z_R_MW = Z_c * R / MW
        cv_specific = Z_R_MW / (k_c - 1.0) if k_c > 1.001 else Z_R_MW / 0.4

        # ---- Isentropic temperature ----
        T_isentropic = T0 * (m_new / m0) ** (k_c - 1.0) if m0 > EPSILON and m_new > EPSILON else T0

        # ---- Wall heat transfer ----
        T_new = T_isentropic
        if inputs.U_htc > EPSILON and inputs.T_ambient > EPSILON and m_new > EPSILON:
            Q_wall = inputs.U_htc * inputs.A_wall * (inputs.T_ambient - T)
            dT_heat = Q_wall * dt_eff / (m_new * cv_specific) if cv_specific > EPSILON else 0.0
            T_new = T_isentropic + dT_heat

        T_new = max(T_new, inputs.T_min)

        # ---- Pressure (real gas law) ----
        P_new = _gas_pressure(V, m_new, T_new, MW, Z_c)
        P_new = max(P_new, P_ATM)

        # ---- Dynamic Z/k update from CoolProp PR EOS ----
        if use_dynamic:
            Z_new, k_new, eos_status = _dynamic_eos_properties(
                inputs.mole_fractions, P_new, T_new, k_const, Z_const)
            cp_calls += 1
            # Recompute P with updated Z for consistency
            P_new = max(_gas_pressure(V, m_new, T_new, MW, Z_new), P_ATM)
            Z_c = Z_new
            k_c = k_new

        # ---- Store state ----
        t += dt_eff
        m = m_new
        T = T_new
        P = P_new

        t_list.append(t)
        m_list.append(m)
        T_list.append(T)
        P_list.append(P)
        mdot_list.append(mdot_i)
        Z_list.append(Z_c)
        k_list.append(k_c)

    # ---- Arrays ----
    t_arr = np.array(t_list)
    m_arr = np.array(m_list)
    T_arr = np.array(T_list)
    P_arr = np.array(P_list)
    mdot_arr = np.array(mdot_list)
    Z_arr = np.array(Z_list)
    k_arr = np.array(k_list)

    total_released = m0 - m_arr[-1] if len(m_arr) > 0 else 0.0
    t_final = float(t_arr[-1]) if len(t_arr) > 0 else 0.0

    events = {}
    if reached_target:
        events["target_reached"] = True
        events["t_reached"] = t_final

    dm_lb = total_released * 2.20462
    m0_lb = m0 * 2.20462

    messages = [
        f"API 521 real-gas blowdown",
        f"EOS: {eos_status}",
        f"Z: {float(np.min(Z_arr)):.4f} → {float(np.max(Z_arr)):.4f} (avg {float(np.mean(Z_arr)):.3f})",
        f"k: {float(np.min(k_arr)):.3f} → {float(np.max(k_arr)):.3f} (avg {float(np.mean(k_arr)):.3f})",
        f"m0={m0_lb:.1f} lb, Δm={dm_lb:.1f} lb ({dm_lb/m0_lb*100:.0f}%)",
    ]
    if use_dynamic:
        messages.append(f"CoolProp calls: {cp_calls}")
    if inputs.U_htc > EPSILON:
        messages.append(f"Wall heat U={inputs.U_htc:.0f} W/m²·K")
    if reached_target:
        messages.append(f"P_target reached at t={t_final:.1f}s ({t_final/60:.1f} min)")
    else:
        messages.append(f"t_max reached, P_final={P_arr[-1]/PSI2PA:.1f} psia")

    return {
        "t": t_arr,
        "P": P_arr,
        "T": T_arr,
        "m": m_arr,
        "mdot": mdot_arr,
        "m_remaining": m_arr,
        "total_mass_released": float(total_released),
        "t_final": t_final,
        "events": events,
        "phase_quality": None,
        "messages": messages,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Two-Phase Blowdown (Liquid + Vapor)
# ══════════════════════════════════════════════════════════════════════════════

def _solve_two_phase_blowdown(inputs: VesselInput) -> dict:
    """Solve two-phase vessel blowdown.

    Simplified Euler integration with:
      - Flash calculation at each timestep
      - Liquid level tracking (constant cross-section assumed)
      - Vapor mass fraction tracking
      - HEM for orifice flow

    Args:
        inputs: VesselInput dataclass.

    Returns:
        Dict with solution arrays.
    """
    if inputs.rho_liquid is None:
        raise ValueError("rho_liquid is required for two-phase blowdown.")

    V = inputs.V
    A_hole = _orifice_area(inputs.orifice_d)
    Cd = inputs.Cd
    P_target = inputs.P_target if inputs.P_target is not None else P_ATM

    # Estimate k if not given
    k = inputs.cp_cv_ratio
    MW = inputs.molecular_weight

    # Initial mass
    rho_l = inputs.rho_liquid
    # Estimate initial vapor mass fraction (quality) from P vs Psat
    T_boil = inputs.T_boiling if inputs.T_boiling else inputs.T_initial
    # Simple: if T > T_boil, some fraction flashes
    x0 = 0.0
    if inputs.T_initial > T_boil and inputs.heat_of_vaporization:
        cp_l = inputs.cp if inputs.cp else 2000.0
        hfg = inputs.heat_of_vaporization
        x0 = cp_l * (inputs.T_initial - T_boil) / hfg
        x0 = max(0.0, min(1.0, x0))

    # Initial vapor and liquid densities
    rho_g0 = _gas_mass(1.0, inputs.P_initial, inputs.T_initial, MW)  # per m³
    v_g = 1.0 / rho_g0 if rho_g0 > EPSILON else 0.0
    v_l = 1.0 / rho_l

    # Initial total inventory mass
    v_mix = x0 * v_g + (1.0 - x0) * v_l
    rho_mix = 1.0 / v_mix if v_mix > EPSILON else rho_l
    m_total = rho_mix * V

    n_steps = inputs.n_time_steps
    t_arr = np.linspace(0, inputs.t_max, n_steps)
    dt = t_arr[1] - t_arr[0]

    P_arr = np.zeros(n_steps)
    T_arr = np.zeros(n_steps)
    m_arr = np.zeros(n_steps)
    mdot_arr = np.zeros(n_steps)
    x_arr = np.zeros(n_steps)  # vapor mass fraction
    rho_m_arr = np.zeros(n_steps)  # mixture density

    P = inputs.P_initial
    T = inputs.T_initial
    m = m_total
    x = x0

    for i in range(n_steps):
        P_arr[i] = P
        T_arr[i] = T
        m_arr[i] = m
        x_arr[i] = x

        if P <= P_target + EPSILON or m <= EPSILON:
            mdot_arr[i] = 0.0
            continue

        # Current fluid properties
        # Vapor density at current P, T
        rho_g = _gas_mass(1.0, P, T, MW)  # kg/m³ per m³
        v_g_cur = 1.0 / rho_g if rho_g > EPSILON else 0.0
        v_mix_cur = x * v_g_cur + (1.0 - x) * v_l
        rho_mix_cur = 1.0 / v_mix_cur if v_mix_cur > EPSILON else rho_l
        rho_m_arr[i] = rho_mix_cur

        # Flashing fraction from energy balance
        # Liquid that vaporizes: x = cp * (T - T_boil) / hfg
        if inputs.heat_of_vaporization and inputs.T_boiling:
            x_flash = 0.0
            if inputs.cp:
                x_flash = inputs.cp * (T - inputs.T_boiling) / inputs.heat_of_vaporization
            x = max(0.0, min(1.0, max(x, x_flash)))

        # Estimate omega for two-phase flow
        omega = 1.0
        if rho_g > EPSILON and rho_l > EPSILON:
            v_fg = v_g_cur - v_l
            if inputs.heat_of_vaporization and inputs.heat_of_vaporization > EPSILON:
                cp_val = inputs.cp if inputs.cp else 2000.0
                omega_s = cp_val * T * P * (v_fg / inputs.heat_of_vaporization) ** 2 / v_mix_cur
                omega = max(0.1, omega_s)

        # Orifice mass flow using two-phase HEM
        # For two-phase: G ~ omega-based or simple HEM
        # Simplified: use incompressible liquid Bernoulli with density correction
        # for low quality; or use gas choked for high quality
        if x < 0.01:
            # Essentially liquid — use Bernoulli with mixture density
            dp = P - P_target
            u = math.sqrt(2.0 * dp / rho_mix_cur) if dp > EPSILON else 0.0
            mdot = Cd * A_hole * rho_mix_cur * u
        elif x > 0.99:
            # Essentially vapor — use gas model
            G_flux = _gas_choked_mass_flux(P, T, k, MW)
            mdot = Cd * A_hole * G_flux
        else:
            # Two-phase mixture — use omega method
            # Critical pressure ratio
            eta_c = omega / (omega + 1.0)
            P_choked = P * eta_c
            is_choked = P_target < P_choked
            eta = eta_c if is_choked else P_target / P

            if eta <= EPSILON or eta >= 1.0:
                mdot = 0.0
            else:
                term = -2.0 * (omega * math.log(eta) + (omega - 1.0) * (1.0 - eta))
                denom = omega * (1.0 / eta - 1.0) + 1.0
                if term <= 0.0 or denom <= 0.0:
                    mdot = 0.0
                else:
                    G_star = math.sqrt(term) / denom
                    G_flux = G_star * math.sqrt(P * rho_mix_cur)
                    mdot = Cd * A_hole * G_flux

        mdot_arr[i] = mdot

        # Step forward (Euler)
        dm = -mdot * dt
        m_new = m + dm
        if m_new <= EPSILON:
            m_new = EPSILON

        # Estimate new pressure (constant volume)
        # P ~ m (ideal gas) or via isentropic relation
        # For two-phase, pressure drops faster initially
        # Simplified: P proportional to mass for vapor, more complex for two-phase
        if x > 0.1:
            P_new = P0 = inputs.P_initial
            P_new = P0 * (m_new / m_total) ** k if m_total > EPSILON else 0.0
        else:
            # Liquid-dominated — pressure proportional to mass remaining
            P_new = max(P_ATM, P * (m_new / m) if m > EPSILON else P_ATM)

        # Temperature estimate
        if x > 0.1:
            T_new = inputs.T_initial * (m_new / m_total) ** (k - 1.0) if m_total > EPSILON else inputs.T_ambient
        else:
            # Liquid stays near initial T (thermal inertia)
            T_new = T + (inputs.T_ambient - T) * 0.01  # slow heat transfer

        P = max(P_new, P_ATM)
        T = max(T_new, 50.0)  # minimum plausible temperature
        m = m_new

    total_released = m_total - m_arr[-1]

    return {
        "t": t_arr,
        "P": P_arr,
        "T": T_arr,
        "m": m_arr,
        "mdot": mdot_arr,
        "m_remaining": m_arr,
        "total_mass_released": float(total_released),
        "t_final": float(t_arr[-1]),
        "events": {},
        "phase_quality": x_arr,
        "messages": ["Two-phase blowdown using HEM + Euler integration"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def calculate_vessel_blowdown(inputs: VesselInput) -> VesselResult:
    """Run a vessel blowdown/depressurization simulation.

    Dispatches to the appropriate model based on phase and mode.

    Args:
        inputs: VesselInput with all required parameters.

    Returns:
        VesselResult with time histories of P, T, m, mdot.

    Raises:
        ValueError: If invalid phase or missing parameters.

    Example:
        >>> inp = VesselInput(
        ...     V=10.0, A_wall=25.0, P_initial=600000, T_initial=300,
        ...     orifice_d=0.025, Cd=0.62, t_max=120, P_target=101325,
        ...     mode='api521'
        ... )
        >>> result = calculate_vessel_blowdown(inp)
        >>> print(f"Released: {result.total_mass_released:.1f} kg")
    """
    phase = inputs.phase.lower()
    mode = inputs.mode.lower()

    if mode == "api521" or (mode == "rigorous" and phase == "gas"):
        # Try API 521 first; fall back to rigorous if SciPy available
        if mode == "api521":
            data = _solve_api521_blowdown(inputs)
        else:
            try:
                data = _solve_gas_blowdown(inputs)
            except (ImportError, Exception) as e:
                # Fall back to API 521
                data = _solve_api521_blowdown(inputs)
                data["messages"].append(f"Falling back to API 521: {e}")
    elif phase in ("two_phase",):
        data = _solve_two_phase_blowdown(inputs)
    else:
        data = _solve_api521_blowdown(inputs)

    return VesselResult(
        t=data["t"],
        P=data["P"],
        T=data["T"],
        m=data["m"],
        mdot=data["mdot"],
        m_remaining=data["m_remaining"],
        phase_quality=data.get("phase_quality"),
        total_mass_released=data["total_mass_released"],
        t_final=data["t_final"],
        events=data.get("events", {}),
        messages=data.get("messages", []),
    )
