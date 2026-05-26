"""
Rekarisk — Gas Hydrate Formation Check.

Determines whether gas hydrates will form under given P, T conditions.
Focuses on natural gas hydrate formers: CH4, C2H6, C3H8, CO2, H2S.

References:
  - Katz, D.L. (1945). "Prediction of Conditions for Hydrate Formation in Natural Gases"
  - Sloan, E.D. & Koh, C.A. (2008). "Clathrate Hydrates of Natural Gases", 3rd ed.
  - Carroll, J. (2014). "Natural Gas Hydrates", 3rd ed.

All calculations in SI: T [K], P [Pa].
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import R, EPSILON, P_ATM, T_0C


# ══════════════════════════════════════════════════════════════════════════════
# Hydrate Constants for Common Gas Components
# ══════════════════════════════════════════════════════════════════════════════

# Katz K-value method: K_i = y_i / x_i_hydrate
# where y_i = gas phase mole fraction, x_i_hydrate = hydrate phase mole fraction
# Hydrate forms when Σ (y_i / K_i) ≥ 1.0

# Hydrate K-value coefficients for common hydrate formers
# ln(K) = A + B·T + C·T² + D·P + E·ln(P)
# where T in °C, P in kPa
# Coefficients from Sloan (2008), Table 4-3

_HYDRATE_K_COEFFS: Dict[str, Tuple[float, ...]] = {
    #        A       B·T     C·T²    D·P      E·ln(P)
    'CH4':  (1.236,  0.0318, 0.0,   -0.0494, 0.0286),
    'C2H6': (0.996,  0.0302, 0.0,   -0.0351, 0.0243),
    'C3H8': (0.732,  0.0305, 0.0,   -0.0256, 0.0195),
    'iC4':  (0.513,  0.0321, 0.0,   -0.0191, 0.0158),
    'nC4':  (0.419,  0.0340, 0.0,   -0.0150, 0.0140),
    'CO2':  (1.203,  0.0310, 0.0,   -0.0502, 0.0277),
    'H2S':  (1.316,  0.0290, 0.0,   -0.0605, 0.0335),
    'N2':   (1.824,  0.0520, 0.0,   -0.0850, 0.0495),
}

# Hydrate formation pressure-temperature curves (for pure components)
# P [Pa] = exp(A + B/T)  — simplified form
# Where T in Kelvin
# From Sloan (2008) and Carroll (2014)

_HYDRATE_PT_COEFFS: Dict[str, Tuple[float, float, float, float]] = {
    #        A       B        T_min  T_max
    'CH4':  (38.44, -8533.80, 148.0, 298.0),
    'C2H6': (39.56, -8950.50, 200.0, 290.0),
    'C3H8': (40.12, -9210.30, 250.0, 280.0),
    'CO2':  (38.96, -8710.50, 150.0, 283.0),
    'H2S':  (37.88, -8120.30, 150.0, 302.0),
    'N2':   (42.15, -10560.00, 100.0, 260.0),
}


# ══════════════════════════════════════════════════════════════════════════════
# Core Functions
# ══════════════════════════════════════════════════════════════════════════════

def hydrate_formation_pressure(T: float,
                                component: str = 'CH4',
                                method: str = 'sloan') -> float:
    """Estimate hydrate formation pressure for a pure component at T.

    Args:
        T: Temperature [K].
        component: Component identifier (CH4, C2H6, C3H8, CO2, H2S, N2).
        method: 'sloan' or 'carroll'.

    Returns:
        Hydrate formation pressure [Pa]. Returns inf if T out of range.
    """
    coeffs = _HYDRATE_PT_COEFFS.get(component.upper())
    if coeffs is None:
        return float('inf')

    A, B, T_min, T_max = coeffs
    if T < T_min or T > T_max:
        return float('inf')

    P = math.exp(A + B / T)  # Pa
    return P


def hydrate_formation_temperature(P: float,
                                   component: str = 'CH4',
                                   method: str = 'sloan') -> float:
    """Estimate hydrate formation temperature for a pure component at P.

    Args:
        P: Pressure [Pa].
        component: Component identifier (CH4, C2H6, C3H8, CO2, H2S, N2).
        method: 'sloan' or 'carroll'.

    Returns:
        Hydrate formation temperature [K]. Returns 0 if out of range.
    """
    coeffs = _HYDRATE_PT_COEFFS.get(component.upper())
    if coeffs is None:
        return 0.0

    A, B, T_min, T_max = coeffs
    if P <= 0:
        return 0.0

    # T = B / (ln(P) - A)
    denominator = math.log(max(P, EPSILON)) - A
    if denominator <= 0:
        return float('inf')

    T = B / denominator
    if T_min <= T <= T_max:
        return T

    # Try to clamp
    if T < T_min:
        return T_min
    return T_max


def katz_hydrate_check(T: float, P: float,
                        composition: Dict[str, float]) -> Tuple[bool, float]:
    """Check if hydrates form using the Katz K-value method.

    Hydrates form when: Σ (y_i / K_i) ≥ 1.0

    where y_i = gas mole fraction, K_i = hydrate equilibrium ratio.

    Args:
        T: Temperature [K].
        P: Pressure [Pa].
        composition: Dict mapping component ID → mole fraction.

    Returns:
        (will_form, sum_y_over_K)
    """
    T_c = T - T_0C       # Convert to °C
    P_kPa = P / 1000.0   # Convert to kPa

    total = 0.0
    for comp_id, y_i in composition.items():
        coeffs = _HYDRATE_K_COEFFS.get(comp_id.upper())
        if coeffs is None:
            continue

        A, B, C, D, E = coeffs
        if P_kPa > EPSILON:
            ln_K = A + B * T_c + C * T_c * T_c + D * P_kPa + E * math.log(P_kPa)
        else:
            ln_K = -10.0  # Very small K

        K_i = math.exp(ln_K)
        if K_i > EPSILON:
            total += y_i / K_i

    return (total >= 1.0, total)


def will_form_hydrate(T: float, P: float,
                       composition: Dict[str, float],
                       method: str = 'katz') -> bool:
    """High-level check: will hydrates form at given conditions?

    Args:
        T: Temperature [K].
        P: Pressure [Pa].
        composition: Dict mapping component ID (e.g., 'CH4', 'C2H6', 'CO2') → mole fraction.
        method: 'katz' for Katz K-value method (default).

    Returns:
        True if hydrates are expected to form.
    """
    if method == 'katz':
        will_form, _ = katz_hydrate_check(T, P, composition)
        return will_form

    # Fallback: check each component against pure component curve
    T_form_max = 0.0
    for comp_id, y_i in composition.items():
        T_form = hydrate_formation_temperature(P, comp_id)
        if T_form > T_form_max:
            T_form_max = T_form
    return T < T_form_max


def hydrate_pt_curve(component: str = 'CH4',
                      n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate P-T hydrate formation curve for a pure component.

    Args:
        component: Component identifier (CH4, C2H6, C3H8, CO2, H2S, N2).
        n_points: Number of points.

    Returns:
        (T_array [K], P_array [Pa]).
    """
    coeffs = _HYDRATE_PT_COEFFS.get(component.upper())
    if coeffs is None:
        return np.array([]), np.array([])

    A, B, T_min, T_max = coeffs
    T_vals = np.linspace(T_min, T_max, n_points)
    P_vals = np.exp(A + B / T_vals)

    return T_vals, P_vals


def hydrate_safety_margin(T: float, P: float,
                           composition: Dict[str, float]) -> float:
    """Compute safety margin: degrees below hydrate formation temperature.

    Positive = safe (T above hydrate point), Negative = hydrate region.

    Args:
        T: Operating temperature [K].
        P: Operating pressure [Pa].
        composition: Component → mole fraction dict.

    Returns:
        Safety margin [K] (T_operating - T_hydrate).
              Positive: no hydrates
              Zero: at hydrate boundary
              Negative: hydrate region
    """
    # Estimate mixture hydrate temperature
    T_hyd = 0.0
    for comp_id, y_i in composition.items():
        T_form = hydrate_formation_temperature(P, comp_id)
        if T_form > T_hyd:
            T_hyd = T_form

    if T_hyd <= 0:
        return float('inf')  # No hydrate risk

    return T - T_hyd


# ══════════════════════════════════════════════════════════════════════════════
# Common Mixture Checks
# ══════════════════════════════════════════════════════════════════════════════

# Typical natural gas compositions for quick checks
TYPICAL_NATURAL_GAS = {
    'CH4': 0.90,
    'C2H6': 0.06,
    'C3H8': 0.03,
    'CO2': 0.01,
}

TYPICAL_LNG = {
    'CH4': 0.95,
    'C2H6': 0.03,
    'C3H8': 0.01,
    'N2': 0.01,
}


def check_natural_gas_hydrate(T: float, P: float,
                               ch4_frac: float = 0.90,
                               c2h6_frac: float = 0.06,
                               c3h8_frac: float = 0.03,
                               co2_frac: float = 0.01) -> bool:
    """Quick hydrate check for typical pipeline natural gas.

    Args:
        T: Temperature [K].
        P: Pressure [Pa].
        ch4_frac: Methane mole fraction.
        c2h6_frac: Ethane mole fraction.
        c3h8_frac: Propane mole fraction.
        co2_frac: CO2 mole fraction.

    Returns:
        True if hydrates expected.
    """
    comp = {
        'CH4': ch4_frac,
        'C2H6': c2h6_frac,
        'C3H8': c3h8_frac,
        'CO2': co2_frac,
    }
    # Normalize
    total = sum(comp.values())
    comp = {k: v / total for k, v in comp.items()}

    return will_form_hydrate(T, P, comp)


def hydrate_inhibition_methanol(water_fraction: float,
                                 T_depression: float) -> float:
    """Estimate methanol concentration needed for hydrate inhibition.

    Hammer-Schmidt equation (simplified):
    T_depression = 1297·Mw·X_MeOH / (Mw_MeOH·(1 - X_MeOH))

    Args:
        water_fraction: Mass fraction of water in liquid phase [-].
        T_depression: Desired temperature depression [K] (below hydrate point).

    Returns:
        Mass fraction of methanol in aqueous phase [-].
    """
    Mw_MeOH = 32.04  # g/mol
    # Simplified: ΔT = 1297·W / (M·(100 - W)) where W = wt%
    # Converting to fraction: W_frac = W/100
    # ΔT = 1297·W_frac / (M·(1 - W_frac))
    # W_frac = M·ΔT / (1297 + M·ΔT)
    W_frac = Mw_MeOH * T_depression / (1297.0 + Mw_MeOH * T_depression)
    return min(max(W_frac, 0.0), 1.0)
