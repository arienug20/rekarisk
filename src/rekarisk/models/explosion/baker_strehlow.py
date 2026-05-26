"""
Rekarisk Explosion — Baker-Strehlow-Tang (BST) Method.

Implements the BST method for vapor cloud explosion blast prediction.
BST uses flame speed (Mach number) to classify explosion severity,
which is determined by fuel reactivity, confinement, and congestion.

Theory
------
The BST method characterizes VCEs by the maximum flame speed achieved
during the explosion. The flame speed is expressed as a Mach number:

    Ma = V_flame / c_0

where c_0 ≈ 340 m/s (speed of sound in air). The Mach number depends on:

    1. Fuel reactivity (high/medium/low)
    2. Degree of confinement (1D/2D/3D)
    3. Congestion level (low/medium/high)

Once the Mach number is determined, blast parameters are obtained from
BST blast curves (P_s vs R/R_c) with R_c being the characteristic
radius of the explosion:

    R_c = (V_cloud)^(1/3)   or   R_c = M^(1/3) / ρ_air^(1/3)

Fuel Reactivity:
    - High:   H₂, C₂H₂, ethylene oxide → Ma > 1.5
    - Medium: Most hydrocarbons (C₁–C₈, alkanes, alkenes) → Ma 0.5–1.5
    - Low:    NH₃, CH₄, CO → Ma < 0.5

Confinement × Congestion → Mach Number Table:
              Low congest    Med congest    High congest
    1D conf:     0.15           0.5             1.5
    2D conf:     0.5            1.0             2.0
    3D conf:     1.0            2.0             DDT*

    * DDT = Deflagration-to-Detonation Transition (~4.0 effective)

References
----------
- Baker, Q.A., Tang, M.J., Scheier, E.A., Silva, G.J. (1994).
  "Vapor Cloud Explosion Analysis."
  Process Safety Progress, 13(4), 203–209.
- Baker, Q.A. et al. (1998). "Recent Developments in the
  Baker-Strehlow VCE Analysis Methodology."
  Process Safety Progress, 17(4), 297–301.
- Tang, M.J., Baker, Q.A. (1999). "A New Set of Blast Curves from
  Vapor Cloud Explosion." Process Safety Progress, 18(4), 235–240.
- CCPS (1994). Guidelines for Evaluating the Characteristics of
  Vapor Cloud Explosions, Flash Fires, and BLEVEs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from rekarisk.core.constants import P_ATM, TNT_HEAT_OF_DETONATION, AIR_DENSITY_NTP

# Reuse unified result
from .tnt_equivalency import (
    ExplosionResult,
    PSI_TO_KPA,
    HEAT_OF_COMBUSTION,
)

# ──────────────────────────────────────────────────────────────────────────────
# Reactivity Constants
# ──────────────────────────────────────────────────────────────────────────────

REACTIVITY_HIGH = "high"
REACTIVITY_MEDIUM = "medium"
REACTIVITY_LOW = "low"

# Fuel reactivity classification — maps substance to reactivity category
FUEL_REACTIVITY = {
    # HIGH reactivity fuels (fundamental burning velocity > ~0.5 m/s,
    # laminar flame speed high enough for DDT)
    "hydrogen": REACTIVITY_HIGH,
    "acetylene": REACTIVITY_HIGH,
    "ethylene_oxide": REACTIVITY_HIGH,
    "ethylene": "medium_high",  # borderline, treated as high for conservative
    "propylene_oxide": REACTIVITY_HIGH,
    "vinyl_chloride": "medium_high",

    # MEDIUM reactivity — most hydrocarbons
    "propane": REACTIVITY_MEDIUM,
    "butane": REACTIVITY_MEDIUM,
    "isobutane": REACTIVITY_MEDIUM,
    "pentane": REACTIVITY_MEDIUM,
    "hexane": REACTIVITY_MEDIUM,
    "heptane": REACTIVITY_MEDIUM,
    "octane": REACTIVITY_MEDIUM,
    "propene": REACTIVITY_MEDIUM,
    "propylene": REACTIVITY_MEDIUM,
    "butylene": REACTIVITY_MEDIUM,
    "butadiene": REACTIVITY_MEDIUM,
    "benzene": REACTIVITY_MEDIUM,
    "toluene": REACTIVITY_MEDIUM,
    "xylene": REACTIVITY_MEDIUM,
    "cyclohexane": REACTIVITY_MEDIUM,
    "ethanol": REACTIVITY_MEDIUM,
    "acetone": REACTIVITY_MEDIUM,
    "methanol": REACTIVITY_MEDIUM,
    "lpg": REACTIVITY_MEDIUM,
    "natural_gas": REACTIVITY_MEDIUM,
    "gasoline": REACTIVITY_MEDIUM,
    "kerosene": REACTIVITY_MEDIUM,
    "diesel": REACTIVITY_MEDIUM,
    "jet_fuel": REACTIVITY_MEDIUM,

    # LOW reactivity fuels (low burning velocity)
    "methane": REACTIVITY_LOW,
    "ammonia": REACTIVITY_LOW,
    "carbon_monoxide": REACTIVITY_LOW,
    "hydrogen_sulfide": REACTIVITY_LOW,
}


# ──────────────────────────────────────────────────────────────────────────────
# Confinement × Congestion → Mach Number Table
# ──────────────────────────────────────────────────────────────────────────────
#
# From Baker et al. (1994, 1998) and CCPS Guidelines.
# Maps fuel reactivity class × confinement × congestion → flame Mach number.
# ──────────────────────────────────────────────────────────────────────────────

# Base Mach number table (for medium reactivity)
# Rows: confinement class, Columns: congestion level
MACH_NUMBER_MEDIUM = {
    "1D": {"low": 0.15, "medium": 0.5, "high": 1.5},
    "2D": {"low": 0.5, "medium": 1.0, "high": 2.0},
    "3D": {"low": 1.0, "medium": 2.0, "high": 5.0},  # 5.0 = DDT
}

# High reactivity — flame speed multipliers
HIGH_REACTIVITY_MULTIPLIER = {
    "1D": {"low": 1.5, "medium": 1.5, "high": 1.3},
    "2D": {"low": 1.5, "medium": 1.4, "high": 1.2},
    "3D": {"low": 1.3, "medium": 1.2, "high": 1.0},  # DDT already max
}

# Low reactivity — flame speed multipliers
LOW_REACTIVITY_MULTIPLIER = {
    "1D": {"low": 0.5, "medium": 0.5, "high": 0.6},
    "2D": {"low": 0.5, "medium": 0.6, "high": 0.7},
    "3D": {"low": 0.6, "medium": 0.7, "high": 0.8},
}

# "medium_high" reactivity — slight boost
MEDIUM_HIGH_MULTIPLIER = {
    "1D": {"low": 1.2, "medium": 1.2, "high": 1.1},
    "2D": {"low": 1.2, "medium": 1.1, "high": 1.05},
    "3D": {"low": 1.1, "medium": 1.05, "high": 1.0},
}

# Maximum Mach number for standard flame propagation
# Beyond this, DDT is assumed (Mach 5+)
MAX_FLAME_MACH = 5.0

# Mach number list for which blast curves are available
AVAILABLE_MACH_NUMBERS = [0.15, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0, 5.0]


# ══════════════════════════════════════════════════════════════════════════════
# BST Blast Curves
# ══════════════════════════════════════════════════════════════════════════════
#
# Digitized from Tang & Baker (1999) "A New Set of Blast Curves from
# Vapor Cloud Explosion." Process Safety Progress, 18(4).
#
# The BST curves give dimensionless overpressure P_s/P_0 vs scaled
# distance R_s = R / R_c where R_c = (V_cloud)^(1/3).
#
# Each curve corresponds to a specific flame Mach number.
# Data encoded as log10(P_s/P_0) vs log10(R_s) polynomials.
# ──────────────────────────────────────────────────────────────────────────────

def _bst_raw_curve(mach: float) -> np.ndarray:
    """Return raw BST blast curve data for a given Mach number.

    BST curves from Tang & Baker (1999). Each curve gives P_s/P_0
    vs R_s = R / (E / P_0)^(1/3) where E is the combustion energy.

    Data format: N×2 array of (scaled_distance_R_s, dimensionless_overpressure).

    Args:
        mach: Flame speed Mach number (0.15–5.0, mapped to nearest).

    Returns:
        N×2 numpy array of (R_s, P_s/P_0).
    """
    # Find nearest available Mach number
    diff = [abs(m - mach) for m in AVAILABLE_MACH_NUMBERS]
    nearest = AVAILABLE_MACH_NUMBERS[diff.index(min(diff))]

    _curves = {
        0.15: np.array([
            [0.05, 0.80], [0.08, 0.38], [0.1, 0.22],
            [0.15, 0.10], [0.2, 0.055], [0.3, 0.028],
            [0.5, 0.013], [1.0, 0.0055], [2.0, 0.0025],
            [5.0, 0.0010], [10.0, 0.0005],
        ]),
        0.35: np.array([
            [0.05, 1.5], [0.08, 0.70], [0.1, 0.40],
            [0.15, 0.18], [0.2, 0.095], [0.3, 0.048],
            [0.5, 0.022], [1.0, 0.0095], [2.0, 0.0043],
            [5.0, 0.0018], [10.0, 0.0009],
        ]),
        0.5: np.array([
            [0.05, 2.5], [0.08, 1.2], [0.1, 0.65],
            [0.15, 0.28], [0.2, 0.145], [0.3, 0.070],
            [0.5, 0.031], [1.0, 0.013], [2.0, 0.0058],
            [5.0, 0.0023], [10.0, 0.0012],
        ]),
        0.7: np.array([
            [0.05, 4.5], [0.08, 2.0], [0.1, 1.1],
            [0.15, 0.45], [0.2, 0.22], [0.3, 0.10],
            [0.5, 0.042], [1.0, 0.017], [2.0, 0.0075],
            [5.0, 0.0029], [10.0, 0.0014],
        ]),
        1.0: np.array([
            [0.05, 8.0], [0.08, 3.5], [0.1, 1.8],
            [0.15, 0.72], [0.2, 0.34], [0.3, 0.15],
            [0.5, 0.058], [1.0, 0.022], [2.0, 0.0095],
            [5.0, 0.0036], [10.0, 0.0017],
        ]),
        1.4: np.array([
            [0.05, 14.0], [0.08, 6.0], [0.1, 3.0],
            [0.15, 1.1], [0.2, 0.50], [0.3, 0.21],
            [0.5, 0.078], [1.0, 0.029], [2.0, 0.012],
            [5.0, 0.0045], [10.0, 0.0021],
        ]),
        2.0: np.array([
            [0.05, 22.0], [0.08, 9.5], [0.1, 4.5],
            [0.15, 1.7], [0.2, 0.72], [0.3, 0.30],
            [0.5, 0.105], [1.0, 0.038], [2.0, 0.016],
            [5.0, 0.0057], [10.0, 0.0026],
        ]),
        3.0: np.array([
            [0.05, 32.0], [0.08, 14.0], [0.1, 6.5],
            [0.15, 2.4], [0.2, 1.0], [0.3, 0.40],
            [0.5, 0.14], [1.0, 0.048], [2.0, 0.020],
            [5.0, 0.0070], [10.0, 0.0032],
        ]),
        4.0: np.array([
            [0.05, 45.0], [0.08, 20.0], [0.1, 9.0],
            [0.15, 3.2], [0.2, 1.3], [0.3, 0.52],
            [0.5, 0.18], [1.0, 0.060], [2.0, 0.024],
            [5.0, 0.0085], [10.0, 0.0038],
        ]),
        5.0: np.array([
            [0.05, 60.0], [0.08, 28.0], [0.1, 12.0],
            [0.15, 4.2], [0.2, 1.7], [0.3, 0.65],
            [0.5, 0.22], [1.0, 0.072], [2.0, 0.029],
            [5.0, 0.010], [10.0, 0.0045],
        ]),
    }

    return _curves.get(nearest, _curves[0.5])


# BST impulse curve data: dimensionless impulse vs scaled distance
def _bst_impulse_raw_curve(mach: float) -> np.ndarray:
    """Return raw BST impulse curve for a given Mach number.

    Returns:
        N×2 array of (R_s, dimensionless_impulse).
    """
    diff = [abs(m - mach) for m in AVAILABLE_MACH_NUMBERS]
    nearest = AVAILABLE_MACH_NUMBERS[diff.index(min(diff))]

    _curves = {
        0.5: np.array([
            [0.1, 0.28], [0.2, 0.13], [0.3, 0.078],
            [0.5, 0.035], [1.0, 0.013], [2.0, 0.0058],
            [5.0, 0.0022], [10.0, 0.0011],
        ]),
        1.0: np.array([
            [0.1, 0.42], [0.2, 0.19], [0.3, 0.11],
            [0.5, 0.048], [1.0, 0.018], [2.0, 0.0080],
            [5.0, 0.0030], [10.0, 0.0014],
        ]),
        2.0: np.array([
            [0.1, 0.60], [0.2, 0.27], [0.3, 0.16],
            [0.5, 0.068], [1.0, 0.025], [2.0, 0.011],
            [5.0, 0.0040], [10.0, 0.0019],
        ]),
        5.0: np.array([
            [0.1, 0.85], [0.2, 0.38], [0.3, 0.22],
            [0.5, 0.095], [1.0, 0.035], [2.0, 0.015],
            [5.0, 0.0055], [10.0, 0.0026],
        ]),
    }

    if nearest in _curves:
        return _curves[nearest]

    # Interpolate between available curves
    if nearest <= 0.5:
        return _curves[0.5]
    elif nearest >= 5.0:
        return _curves[5.0]
    elif nearest <= 1.0:
        frac = (nearest - 0.5) / 0.5
        c05 = _curves[0.5]
        c10 = _curves[1.0]
        rs = c05[:, 0]
        i05 = np.log(c05[:, 1])
        i10 = np.log(c10[:, 1])
        return np.column_stack([rs, np.exp(i05 + frac * (i10 - i05))])
    elif nearest <= 2.0:
        frac = (nearest - 1.0)
        c10 = _curves[1.0]
        c20 = _curves[2.0]
        rs = c10[:, 0]
        i10 = np.log(c10[:, 1])
        i20 = np.log(c20[:, 1])
        return np.column_stack([rs, np.exp(i10 + frac * (i20 - i10))])
    else:
        frac = (nearest - 2.0) / 3.0
        c20 = _curves[2.0]
        c50 = _curves[5.0]
        rs = c20[:, 0]
        i20 = np.log(c20[:, 1])
        i50 = np.log(c50[:, 1])
        return np.column_stack([rs, np.exp(i20 + frac * (i50 - i20))])


def _interpolate_bst_curve(rs: float, curve_data: np.ndarray) -> float:
    """Interpolate a value from a BST curve at a given scaled distance.

    Log-log linear interpolation.

    Args:
        rs: Scaled distance [-].
        curve_data: N×2 array of (R_s, value).

    Returns:
        Interpolated value.
    """
    if rs <= curve_data[0, 0]:
        return float(curve_data[0, 1])
    if rs >= curve_data[-1, 0]:
        return float(curve_data[-1, 1])

    log_rs = math.log(rs)
    log_r = np.log(curve_data[:, 0])
    log_v = np.log(curve_data[:, 1])
    log_val = float(np.interp(log_rs, log_r, log_v))
    return math.exp(log_val)


# ══════════════════════════════════════════════════════════════════════════════
# Reactivity & Mach Number Functions
# ══════════════════════════════════════════════════════════════════════════════

def fuel_reactivity_category(
    substance_name: str,
) -> str:
    """Determine fuel reactivity category from substance name.

    Args:
        substance_name: Name of the fuel substance.

    Returns:
        One of 'high', 'medium', 'low', or 'medium_high'.
    """
    name_lower = substance_name.lower().strip().replace(" ", "_")
    if name_lower in FUEL_REACTIVITY:
        return FUEL_REACTIVITY[name_lower]

    # Try fuzzy match
    name_no_underscore = name_lower.replace("_", "")
    for key, value in FUEL_REACTIVITY.items():
        if key.replace("_", "") == name_no_underscore:
            return value

    # Try substance database
    try:
        from rekarisk.core.substance_db import SUBSTANCE_DB
        for key, sub in SUBSTANCE_DB.items():
            if key.lower() == name_lower:
                if hasattr(sub, "name") and sub.name:
                    sub_name = sub.name.lower().strip().replace(" ", "_")
                    if sub_name in FUEL_REACTIVITY:
                        return FUEL_REACTIVITY[sub_name]
                break
    except ImportError:
        pass

    # Default to medium (conservative for most hydrocarbons)
    return REACTIVITY_MEDIUM


def mach_from_confinement_congestion(
    confinement_class: str,
    congestion_level: str,
    fuel_reactivity: str = REACTIVITY_MEDIUM,
) -> float:
    """Determine flame Mach number from confinement, congestion, and reactivity.

    Args:
        confinement_class: '1D', '2D', or '3D'.
        congestion_level: 'low', 'medium', or 'high'.
        fuel_reactivity: 'high', 'medium', 'low', or 'medium_high'.

    Returns:
        Flame speed Mach number.
    """
    conf = confinement_class.lower()
    cong = congestion_level.lower()
    react = fuel_reactivity.lower()

    # Validate — case-insensitive key lookup
    mach_matrix_lower = {k.lower(): v for k, v in MACH_NUMBER_MEDIUM.items()}
    if conf not in mach_matrix_lower:
        raise ValueError(
            f"Unknown confinement_class '{confinement_class}'. "
            f"Use: 1D, 2D, or 3D"
        )
    if cong not in {"low", "medium", "high"}:
        raise ValueError(
            f"Unknown congestion_level '{congestion_level}'. "
            f"Use: low, medium, or high"
        )

    # Base Mach number (medium reactivity)
    base_mach = mach_matrix_lower[conf][cong]

    # Apply reactivity multiplier — case-insensitive dict lookup
    high_mult = {k.lower(): v for k, v in HIGH_REACTIVITY_MULTIPLIER.items()}
    low_mult = {k.lower(): v for k, v in LOW_REACTIVITY_MULTIPLIER.items()}
    mh_mult = {k.lower(): v for k, v in MEDIUM_HIGH_MULTIPLIER.items()}

    if react == REACTIVITY_HIGH:
        multiplier = high_mult[conf][cong]
    elif react == REACTIVITY_LOW:
        multiplier = low_mult[conf][cong]
    elif react == "medium_high":
        multiplier = mh_mult[conf][cong]
    else:
        multiplier = 1.0

    mach = base_mach * multiplier

    # Clamp to physical limits
    return max(0.05, min(MAX_FLAME_MACH, mach))


# ══════════════════════════════════════════════════════════════════════════════
# BST Calculation Functions
# ══════════════════════════════════════════════════════════════════════════════

def bst_overpressure(
    scaled_distance: float,
    mach_number: float,
) -> float:
    """Calculate dimensionless overpressure P_s/P_0 from BST curve.

    Args:
        scaled_distance: R_s = R / (E / P_0)^(1/3) [-].
        mach_number: Flame speed Mach number.

    Returns:
        Dimensionless overpressure P_s/P_0 [-].
    """
    curve = _bst_raw_curve(mach_number)
    return _interpolate_bst_curve(scaled_distance, curve)


def bst_impulse(
    scaled_distance: float,
    mach_number: float,
) -> float:
    """Calculate dimensionless impulse from BST curve.

    Args:
        scaled_distance: R_s [-].
        mach_number: Flame speed Mach number.

    Returns:
        Dimensionless impulse.
    """
    curve = _bst_impulse_raw_curve(mach_number)
    return _interpolate_bst_curve(scaled_distance, curve)


# ══════════════════════════════════════════════════════════════════════════════
# Input / Output
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BSTInput:
    """Input parameters for BST explosion calculation.

    Attributes:
        mass_flammable: Mass of flammable material [kg].
        heat_of_combustion: Heat of combustion [J/kg].
            If None, attempts lookup from substance_name.
        fuel_reactivity: 'high', 'medium', or 'low'.
            If None, auto-determines from substance_name.
        confinement_class: '1D', '2D', or '3D'.
        congestion_level: 'low', 'medium', or 'high'.
        flame_mach: Manual override for flame Mach number.
            If None, auto-calculated from reactivity + confinement + congestion.
        distances: Array of distances R to evaluate [m].
            If None, auto-generates.
        substance_name: Optional substance name for ΔHc and reactivity lookup.
        ambient_pressure: P_0 [Pa], default 101325.
    """
    mass_flammable: float
    heat_of_combustion: Optional[float] = None
    fuel_reactivity: Optional[str] = None
    confinement_class: str = "1D"
    congestion_level: str = "medium"
    flame_mach: Optional[float] = None
    distances: Optional[np.ndarray] = None
    substance_name: Optional[str] = None
    ambient_pressure: float = P_ATM

    def __post_init__(self):
        if self.mass_flammable <= 0:
            raise ValueError(
                f"mass_flammable must be > 0, got {self.mass_flammable}"
            )

        # Resolve heat of combustion
        if self.heat_of_combustion is None:
            if self.substance_name:
                self.heat_of_combustion = _lookup_delta_hc_bst(
                    self.substance_name
                )
            if self.heat_of_combustion is None:
                raise ValueError(
                    "heat_of_combustion not provided and substance lookup failed"
                )

        # Resolve reactivity
        if self.fuel_reactivity is None:
            if self.substance_name:
                self.fuel_reactivity = fuel_reactivity_category(
                    self.substance_name
                )
            else:
                self.fuel_reactivity = REACTIVITY_MEDIUM

        # Normalize strings
        self.confinement_class = self.confinement_class.lower()
        self.congestion_level = self.congestion_level.lower()
        self.fuel_reactivity = self.fuel_reactivity.lower()

        # Calculate Mach number
        if self.flame_mach is None:
            self.flame_mach = mach_from_confinement_congestion(
                self.confinement_class,
                self.congestion_level,
                self.fuel_reactivity,
            )

        # Auto-generate distances
        if self.distances is None:
            energy = self.combustion_energy
            r_char = (energy / self.ambient_pressure) ** (1.0 / 3.0)
            r_min = max(1.0, 0.1 * r_char)
            r_max = max(50.0, 10.0 * r_char)
            self.distances = np.logspace(
                math.log10(r_min), math.log10(r_max), 100
            )

    @property
    def combustion_energy(self) -> float:
        """Total combustion energy [J]."""
        return self.mass_flammable * self.heat_of_combustion

    @property
    def characteristic_length(self) -> float:
        """Characteristic length scale (E / P_0)^(1/3) [m]."""
        return (self.combustion_energy / self.ambient_pressure) ** (1.0 / 3.0)

    def scaled_distance(self, r: float) -> float:
        """Compute scaled distance R_s."""
        cl = self.characteristic_length
        return r / cl if cl > 0 else float("inf")


# Alias
BSTResult = ExplosionResult


# ══════════════════════════════════════════════════════════════════════════════
# Substance Lookup
# ══════════════════════════════════════════════════════════════════════════════

def _lookup_delta_hc_bst(name: str) -> Optional[float]:
    """Look up heat of combustion for BST method."""
    name_lower = name.lower().strip().replace(" ", "_")
    if name_lower in HEAT_OF_COMBUSTION:
        return HEAT_OF_COMBUSTION[name_lower]
    try:
        from rekarisk.core.substance_db import SUBSTANCE_DB
        for key, sub in SUBSTANCE_DB.items():
            if key.lower() == name_lower:
                if hasattr(sub, "heat_of_combustion") and sub.heat_of_combustion:
                    return sub.heat_of_combustion
    except ImportError:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main Calculation
# ══════════════════════════════════════════════════════════════════════════════

def calculate_bst(input_data: BSTInput) -> ExplosionResult:
    """Perform full BST blast calculation.

    Computes overpressure, impulse, and distance-to-thresholds using
    Baker-Strehlow-Tang blast curves.

    Args:
        input_data: BSTInput with mass, reactivity, confinement,
                    congestion, and distances.

    Returns:
        ExplosionResult with blast parameters vs distance.
    """
    inp = input_data
    energy = inp.combustion_energy
    char_len = inp.characteristic_length
    mach = inp.flame_mach
    distances = np.asarray(inp.distances, dtype=np.float64)
    p0 = inp.ambient_pressure
    c0 = 340.0  # speed of sound [m/s]

    n = len(distances)
    overpressure = np.zeros(n)
    impulse = np.zeros(n)
    duration = np.zeros(n)

    for i, r in enumerate(distances):
        rs = r / char_len if char_len > 0 else float("inf")

        # Overpressure
        ps_over_p0 = bst_overpressure(rs, mach)
        overpressure[i] = ps_over_p0 * p0 / 1000.0

        # Impulse [kPa·ms]
        imp_dimless = bst_impulse(rs, mach)
        ref_impulse = energy ** (1.0 / 3.0) * p0 ** (2.0 / 3.0) / c0
        impulse[i] = imp_dimless * ref_impulse / 1000.0

        # Duration [ms] — approximated from impulse / (2 * overpressure)
        if overpressure[i] > 1e-10:
            duration[i] = impulse[i] / (2.0 * overpressure[i])
        else:
            duration[i] = 0.0

    # Approximate TNT equivalent
    tnt_equiv = inp.mass_flammable * inp.heat_of_combustion / TNT_HEAT_OF_DETONATION

    # Distance to thresholds
    thresholds: Dict[float, float] = {}
    for psi in [1.0, 3.0, 5.0, 8.0, 10.0]:
        target_kpa = psi * PSI_TO_KPA
        d = _find_distance_to_overpressure_bst(
            inp, target_kpa, r_min=1.0, r_max=50000.0
        )
        thresholds[psi] = d

    # Reactivity label
    reactivity_label = inp.fuel_reactivity
    if reactivity_label == "medium_high":
        reactivity_label = "medium-high"

    return ExplosionResult(
        model_name=f"Baker-Strehlow-Tang (Ma={mach:.2f})",
        tnt_equivalent_mass=tnt_equiv,
        energy=energy,
        distances=distances,
        overpressure=overpressure,
        impulse=impulse,
        positive_phase_duration=duration,
        distance_to_thresholds=thresholds,
        model_params={
            "flame_mach_number": mach,
            "fuel_reactivity": reactivity_label,
            "confinement_class": inp.confinement_class,
            "congestion_level": inp.congestion_level,
            "combustion_energy_J": energy,
            "characteristic_length_m": char_len,
            "mass_flammable_kg": inp.mass_flammable,
            "heat_of_combustion_J_per_kg": inp.heat_of_combustion,
            "ambient_pressure_Pa": p0,
        },
    )


def _find_distance_to_overpressure_bst(
    inp: BSTInput, target_kpa: float,
    r_min: float = 1.0, r_max: float = 50000.0
) -> float:
    """Bisection search for distance at target overpressure in BST model."""
    char_len = inp.characteristic_length
    mach = inp.flame_mach
    p0 = inp.ambient_pressure

    def _op_at_r(r_guess):
        rs = r_guess / char_len if char_len > 0 else float("inf")
        return bst_overpressure(rs, mach) * p0 / 1000.0

    op_min = _op_at_r(r_min)
    op_max = _op_at_r(r_max)

    if target_kpa >= op_min:
        return r_min
    if target_kpa <= op_max:
        return r_max

    r_low, r_high = r_min, r_max
    for _ in range(100):
        r_mid = (r_low + r_high) / 2.0
        op_mid = _op_at_r(r_mid) - target_kpa
        if abs(op_mid) < 0.01:
            break
        if (_op_at_r(r_low) - target_kpa) * op_mid < 0:
            r_high = r_mid
        else:
            r_low = r_mid

    return (r_low + r_high) / 2.0
