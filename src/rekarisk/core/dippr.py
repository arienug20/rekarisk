"""
Rekarisk — DIPPR Correlation Engine.

Implements DIPPR 100-series temperature-dependent property correlations.
These correlations are the industry standard for computing temperature-dependent
thermophysical properties of pure compounds.

References:
  - DIPPR Project 801, Design Institute for Physical Properties
  - "The Properties of Gases and Liquids", Reid, Prausnitz & Poling, 5th ed.
  - Yaws' Handbook of Thermodynamic Properties

All correlations are in SI units: T [K], P [Pa], density [kmol/m³],
                                Cp [J/(kmol·K)], etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .constants import R, EPSILON


# ══════════════════════════════════════════════════════════════════════════════
# DIPPR Equation Type Registry
# ══════════════════════════════════════════════════════════════════════════════

# DIPPR 100-series equation forms.
# The equation number encodes both the property and the mathematical form.

DIPPR_100 = 100  # ln(VP) = A + B/T + C·ln(T) + D·T^E          (Vapor Pressure)
DIPPR_101 = 101  # ln(VP) = A + B/T + C·ln(T) + D·T^E           (Vapor Pressure, aliased)
DIPPR_102 = 102  # Y = A·T^B / (1 + C/T + D/T²)                 (Liquid Cp, Liquid Visc, etc.)
DIPPR_104 = 104  # Y = A + B·T + C·T² + D·T³ + E·T⁴            (Polynomial)
DIPPR_105 = 105  # Y = A / B^(1 + (1 - T/C)^D)                 (Liquid Density)
DIPPR_106 = 106  # Y = A·(1 - Tr)^(B + C·Tr + D·Tr² + E·Tr³)   (Sat. Liquid Density, Hvap, SurTen)
DIPPR_107 = 107  # Y = A + B·((C/T)/sinh(C/T))² + D·((E/T)/cosh(E/T))²  (Ideal Gas Cp)
DIPPR_114 = 114  # ln(Y) = A + B/(C + T) + D·ln(T)             (Liquid Therm. Cond.)
DIPPR_116 = 116  # Y = A + B·T + C·T²                           (Quadratic)

# Map names to equation types for convenience
DIPPR_FORM_TO_TYPE = {
    'vp': 101,                   # vapor pressure
    'liq_density': 105,          # liquid density (alternative: 106)
    'liq_cp': 102,               # liquid heat capacity
    'gas_cp': 107,               # ideal gas heat capacity
    'liq_visc': 102,             # liquid viscosity
    'gas_visc': 102,             # gas viscosity
    'liq_therm_cond': 114,       # liquid thermal conductivity
    'gas_therm_cond': 102,       # gas thermal conductivity (alternative: 110)
    'surf_tens': 106,            # surface tension
    'h_vap': 106,                # heat of vaporization
}


# ══════════════════════════════════════════════════════════════════════════════
# DIPPRParams Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DIPPRParams:
    """DIPPR correlation parameters for a single property.

    Attributes:
        eq_type: DIPPR equation type number (100-116).
        A, B, C, D, E: Correlation coefficients.
        t_min: Minimum temperature validity [K].
        t_max: Maximum temperature validity [K].
        units: Output units (e.g., 'Pa', 'kmol/m³', 'J/(kmol·K)').
    """
    eq_type: int
    A: float = 0.0
    B: float = 0.0
    C: float = 0.0
    D: float = 0.0
    E: float = 0.0
    t_min: float = 0.0
    t_max: float = 10000.0
    units: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DIPPRParams":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            eq_type=data.get('type', 0),
            A=data.get('A', 0.0),
            B=data.get('B', 0.0),
            C=data.get('C', 0.0),
            D=data.get('D', 0.0),
            E=data.get('E', 0.0),
            t_min=data.get('t_min', 0.0),
            t_max=data.get('t_max', 10000.0),
            units=data.get('units', ''),
        )

    @classmethod
    def from_tuple(cls, params: tuple) -> "DIPPRParams":
        """Create from tuple: (eq_type, A, B, C, D, E, t_min, t_max)."""
        eq_type = int(params[0])
        A, B, C, D, E = params[1:6]
        t_min = params[6] if len(params) > 6 else 0.0
        t_max = params[7] if len(params) > 7 else 10000.0
        return cls(eq_type=eq_type, A=A, B=B, C=C, D=D, E=E,
                   t_min=t_min, t_max=t_max)

    def to_tuple(self) -> tuple:
        """Convert to the tuple format used by Substance."""
        return (self.eq_type, self.A, self.B, self.C, self.D, self.E,
                self.t_min, self.t_max)


# ══════════════════════════════════════════════════════════════════════════════
# Core Evaluation Function
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(params: DIPPRParams, T: float,
             clamp: bool = True) -> float:
    """Evaluate a DIPPR correlation at temperature T [K].

    Args:
        params: DIPPRParams with equation type and coefficients.
        T: Temperature [K].
        clamp: If True, clamp T to validity range instead of raising.

    Returns:
        Computed property value in SI units (or as specified by params.units).

    Raises:
        ValueError: If equation type is unknown or T is outside range (if clamp=False).
    """
    if clamp and params.t_min > 0:
        if T < params.t_min:
            T = params.t_min
        if T > params.t_max:
            T = params.t_max

    eq_type = params.eq_type

    try:
        return _EQUATION_DISPATCH[eq_type](params, T)
    except KeyError:
        raise ValueError(f"Unknown DIPPR equation type: {eq_type}")


def _eq_100(params: DIPPRParams, T: float) -> float:
    """DIPPR 100/101: ln(VP) = A + B/T + C·ln(T) + D·T^E.

    Used for vapor pressure, liquid viscosity.
    Returns value in params.units (typically Pa for VP, Pa·s for viscosity).
    """
    A, B, C, D, E = params.A, params.B, params.C, params.D, params.E
    if T <= 0:
        return 0.0
    ln_y = A + B / T + C * math.log(T) + D * (T ** E)
    return math.exp(max(ln_y, -100.0))


def _eq_102(params: DIPPRParams, T: float) -> float:
    """DIPPR 102: Y = A·T^B / (1 + C/T + D/T²).

    Used for liquid Cp, liquid viscosity, gas viscosity,
    gas thermal conductivity.
    """
    A, B, C, D = params.A, params.B, params.C, params.D
    denom = 1.0 + C / T + D / (T * T)
    if abs(denom) < EPSILON:
        return 0.0
    return A * (T ** B) / denom


def _eq_104(params: DIPPRParams, T: float) -> float:
    """DIPPR 104: Y = A + B·T + C·T² + D·T³ + E·T⁴.

    Polynomial form for various properties.
    """
    A, B, C, D, E = params.A, params.B, params.C, params.D, params.E
    return A + B * T + C * T * T + D * (T ** 3) + E * (T ** 4)


def _eq_105(params: DIPPRParams, T: float) -> float:
    """DIPPR 105: Y = A / B^(1 + (1 - T/C)^D).

    Used for liquid density [kmol/m³].
    """
    A, B, C, D = params.A, params.B, params.C, params.D
    if C <= EPSILON:
        return 0.0
    inner = (1.0 - T / C) ** D if T < C else 0.0
    exponent = 1.0 + inner
    return A / (B ** exponent)


def _eq_106(params: DIPPRParams, T: float) -> float:
    """DIPPR 106: Y = A·(1 - Tr)^(B + C·Tr + D·Tr² + E·Tr³).

    Tr = T/Tc (C parameter = Tc).
    Used for saturated liquid density, surface tension, Hvap.
    """
    A, B, C, D, E = params.A, params.B, params.C, params.D, params.E
    # C = Tc in this form
    Tc = C
    if Tc <= EPSILON:
        return 0.0
    Tr = T / Tc
    if Tr >= 1.0:
        return 0.0  # Above critical point
    exponent = B + C * Tr + D * Tr * Tr + E * (Tr ** 3)
    return A * ((1.0 - Tr) ** exponent)


def _eq_107(params: DIPPRParams, T: float) -> float:
    """DIPPR 107: Y = A + B·((C/T)/sinh(C/T))² + D·((E/T)/cosh(E/T))².

    Used for ideal gas heat capacity [J/(kmol·K)].
    """
    A, B, C, D, E = params.A, params.B, params.C, params.D, params.E
    if T <= EPSILON:
        return A

    term1 = 1.0
    if abs(C) > EPSILON:
        ratio = C / T
        # Limit sinh ratio to avoid overflow
        if abs(ratio) > 50.0:
            term1 = 0.0
        else:
            term1 = min(ratio / math.sinh(ratio), 1.0)

    term2 = 1.0
    if abs(E) > EPSILON:
        ratio2 = E / T
        if abs(ratio2) > 50.0:
            term2 = 0.0
        else:
            term2 = min(ratio2 / math.cosh(ratio2), 1.0)

    return A + B * term1 * term1 + D * term2 * term2


def _eq_114(params: DIPPRParams, T: float) -> float:
    """DIPPR 114: ln(Y) = A + B/(C + T) + D·ln(T).

    Used for liquid thermal conductivity [W/(m·K)].
    """
    A, B, C, D = params.A, params.B, params.C, params.D
    if T <= EPSILON:
        return 0.0
    ln_y = A + B / (C + T) + D * math.log(T)
    return math.exp(max(ln_y, -100.0))


def _eq_116(params: DIPPRParams, T: float) -> float:
    """DIPPR 116: Y = A + B·T + C·T².

    Simple quadratic form.
    """
    A, B, C = params.A, params.B, params.C
    return A + B * T + C * T * T


# Equation dispatch table
_EQUATION_DISPATCH = {
    100: _eq_100,
    101: _eq_100,   # Same mathematical form as 100
    102: _eq_102,
    104: _eq_104,
    105: _eq_105,
    106: _eq_106,
    107: _eq_107,
    114: _eq_114,
    116: _eq_116,
}


# ══════════════════════════════════════════════════════════════════════════════
# High-Level Property Calculators
# ══════════════════════════════════════════════════════════════════════════════

def vapor_pressure(T: float, params: DIPPRParams) -> float:
    """Compute vapor pressure at T [K] → returns [Pa]."""
    return evaluate(params, T)


def liquid_density(T: float, params: DIPPRParams,
                   mw: float = 16.0) -> float:
    """Compute liquid density at T [K] → returns [kg/m³].

    DIPPR correlations return molar density [kmol/m³].
    Converted to mass density using molecular weight.

    Args:
        T: Temperature [K].
        params: DIPPRParams for liquid density.
        mw: Molecular weight [g/mol] (same as [kg/kmol]).

    Returns:
        Liquid density [kg/m³].
    """
    molar_density = evaluate(params, T)  # kmol/m³
    return molar_density * mw  # kg/m³


def liquid_heat_capacity(T: float, params: DIPPRParams,
                          mw: float = 16.0) -> float:
    """Compute liquid heat capacity at T [K] → returns [J/(kg·K)].

    DIPPR returns J/(kmol·K). Converted to mass basis.

    Args:
        T: Temperature [K].
        params: DIPPRParams for liquid Cp.
        mw: Molecular weight [kg/kmol].

    Returns:
        Liquid Cp [J/(kg·K)].
    """
    cp_molar = evaluate(params, T)  # J/(kmol·K)
    return cp_molar / mw if mw > EPSILON else 2000.0


def gas_heat_capacity(T: float, params: DIPPRParams,
                       mw: float = 16.0) -> float:
    """Compute ideal gas heat capacity at T [K] → returns [J/(kg·K)].

    Args:
        T: Temperature [K].
        params: DIPPRParams for gas Cp.
        mw: Molecular weight [kg/kmol].

    Returns:
        Gas Cp [J/(kg·K)].
    """
    cp_molar = evaluate(params, T)  # J/(kmol·K)
    return cp_molar / mw if mw > EPSILON else 1000.0


def liquid_viscosity(T: float, params: DIPPRParams) -> float:
    """Compute liquid dynamic viscosity at T [K] → returns [Pa·s]."""
    return evaluate(params, T)


def gas_viscosity(T: float, params: DIPPRParams) -> float:
    """Compute gas dynamic viscosity at T [K] → returns [Pa·s]."""
    return evaluate(params, T)


def liquid_thermal_cond(T: float, params: DIPPRParams) -> float:
    """Compute liquid thermal conductivity at T [K] → returns [W/(m·K)]."""
    return evaluate(params, T)


def gas_thermal_cond(T: float, params: DIPPRParams) -> float:
    """Compute gas thermal conductivity at T [K] → returns [W/(m·K)]."""
    return evaluate(params, T)


def surface_tension(T: float, params: DIPPRParams) -> float:
    """Compute surface tension at T [K] → returns [N/m]."""
    return evaluate(params, T)


def heat_of_vaporization(T: float, params: DIPPRParams,
                          mw: float = 16.0) -> float:
    """Compute heat of vaporization at T [K] → returns [J/kg].

    DIPPR returns J/kmol. Converted to J/kg.

    Args:
        T: Temperature [K].
        params: DIPPRParams for heat of vaporization.
        mw: Molecular weight [kg/kmol].

    Returns:
        Heat of vaporization [J/kg].
    """
    hv_molar = evaluate(params, T)  # J/kmol
    return hv_molar / mw if mw > EPSILON else 500000.0


# ══════════════════════════════════════════════════════════════════════════════
# Substance-Level Property Computer
# ══════════════════════════════════════════════════════════════════════════════

class DIPPRPropertyComputer:
    """Convenience class to compute all DIPPR properties for a substance.

    The substance's dippr_params dict maps property names to DIPPRParams.

    Usage:
        from rekarisk.core.dippr import DIPPRPropertyComputer
        from rekarisk.core.substance_db import get_database

        db = get_database()
        water = db.get('water')
        comp = DIPPRPropertyComputer(water)
        vp = comp.vapor_pressure(373.15)  # at 100°C
    """

    def __init__(self, substance):
        """Initialize with a Substance instance or dict.

        Args:
            substance: Substance object with molecular_weight and
                       optional 'dippr' attribute containing param dictionaries.
        """
        self._sub = substance
        self._mw = getattr(substance, 'molecular_weight', 16.0)
        # Handle dippr params
        dippr_dict = getattr(substance, 'dippr', None)
        if dippr_dict is None:
            self._params: Dict[str, DIPPRParams] = {}
        else:
            self._params = {}
            for key, val in dippr_dict.items():
                if isinstance(val, dict):
                    self._params[key] = DIPPRParams.from_dict(val)
                elif isinstance(val, DIPPRParams):
                    self._params[key] = val
                else:
                    self._params[key] = val  # Tuple form

    def _get_params(self, key: str) -> Optional[DIPPRParams]:
        """Get DIPPRParams for a property by key name."""
        params = self._params.get(key)
        if params is None:
            return None
        if isinstance(params, tuple):
            return DIPPRParams.from_tuple(params)
        return params

    def vapor_pressure(self, T: float) -> Optional[float]:
        """Vapor pressure [Pa] at T [K]."""
        params = self._get_params('vp')
        return vapor_pressure(T, params) if params else None

    def liquid_density(self, T: float) -> Optional[float]:
        """Liquid density [kg/m³] at T [K]."""
        params = self._get_params('liq_density')
        return liquid_density(T, params, self._mw) if params else None

    def liquid_heat_capacity(self, T: float) -> Optional[float]:
        """Liquid Cp [J/(kg·K)] at T [K]."""
        params = self._get_params('liq_cp')
        return liquid_heat_capacity(T, params, self._mw) if params else None

    def gas_heat_capacity(self, T: float) -> Optional[float]:
        """Ideal gas Cp [J/(kg·K)] at T [K]."""
        params = self._get_params('gas_cp')
        return gas_heat_capacity(T, params, self._mw) if params else None

    def liquid_viscosity(self, T: float) -> Optional[float]:
        """Liquid viscosity [Pa·s] at T [K]."""
        params = self._get_params('liq_visc')
        return liquid_viscosity(T, params) if params else None

    def gas_viscosity(self, T: float) -> Optional[float]:
        """Gas viscosity [Pa·s] at T [K]."""
        params = self._get_params('gas_visc')
        return gas_viscosity(T, params) if params else None

    def liquid_thermal_cond(self, T: float) -> Optional[float]:
        """Liquid thermal conductivity [W/(m·K)] at T [K]."""
        params = self._get_params('liq_therm_cond')
        return liquid_thermal_cond(T, params) if params else None

    def gas_thermal_cond(self, T: float) -> Optional[float]:
        """Gas thermal conductivity [W/(m·K)] at T [K]."""
        params = self._get_params('gas_therm_cond')
        return gas_thermal_cond(T, params) if params else None

    def surface_tension(self, T: float) -> Optional[float]:
        """Surface tension [N/m] at T [K]."""
        params = self._get_params('surf_tens')
        return surface_tension(T, params) if params else None

    def heat_of_vaporization(self, T: float) -> Optional[float]:
        """Heat of vaporization [J/kg] at T [K]."""
        params = self._get_params('h_vap')
        if params:
            return heat_of_vaporization(T, params, self._mw)
        vp_params = self._get_params('vp')
        if vp_params:
            # Clausius-Clapeyron from vapor pressure
            pass  # Not implemented here; use substance method
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers: Load DIPPR params from substance JSON dict
# ══════════════════════════════════════════════════════════════════════════════

def dippr_params_from_json(dippr_data: Optional[Dict[str, Any]]) -> Dict[str, DIPPRParams]:
    """Convert a substance JSON 'dippr' block to DIPPRParams objects.

    Args:
        dippr_data: The 'dippr' dict from JSON (e.g., {'vp': {'type': 100, ...}}).

    Returns:
        Dict mapping property key to DIPPRParams.
    """
    if not dippr_data:
        return {}
    result = {}
    for key, val in dippr_data.items():
        if isinstance(val, dict):
            result[key] = DIPPRParams.from_dict(val)
        elif isinstance(val, (list, tuple)):
            result[key] = DIPPRParams.from_tuple(tuple(val))
    return result
