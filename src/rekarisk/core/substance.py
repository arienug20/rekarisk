"""
Rekarisk — Substance Definition & Property Engine.

Defines the Substance dataclass and property calculation methods
using DIPPR 100-series correlations.

All calculations are performed in SI units (K, Pa, kg/m³, J/kg, etc.)
unless explicitly documented otherwise.

References:
  - DIPPR Project 801 (Design Institute for Physical Properties)
  - CCPS Guidelines for Consequence Analysis
  - TNO Yellow Book
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from .constants import R, T_0C, P_ATM, EPSILON


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class SubstancePhase:
    """Phase-of-matter identifiers."""
    GAS = "gas"
    LIQUID = "liquid"
    SOLID = "solid"
    TWO_PHASE = "two_phase"


class HazardClass:
    """Hazard classification for regulatory purposes."""
    FLAMMABLE = "flammable"
    TOXIC = "toxic"
    EXPLOSIVE = "explosive"
    CORROSIVE = "corrosive"
    OXIDIZING = "oxidizing"
    REACTIVE = "reactive"
    INERT = "inert"
    ASPHYXIANT = "asphyxiant"


class FireClass:
    """Fire hazard class (NFPA / API)."""
    IA = "IA"       # Flash point < 22.8 °C, BP < 37.8 °C
    IB = "IB"       # Flash point < 22.8 °C, BP ≥ 37.8 °C
    IC = "IC"       # Flash point ≥ 22.8 °C and < 37.8 °C
    II = "II"       # Flash point ≥ 37.8 °C and < 60 °C
    IIIA = "IIIA"   # Flash point ≥ 60 °C and < 93 °C
    IIIB = "IIIB"   # Flash point ≥ 93 °C


class DIPPRParam:
    """DIPPR correlation equation type identifiers.

    DIPPR equations have the form: Y = f(T) where T is in Kelvin.

    Common forms:
      101: ln(Y) = A + B/T + C·ln(T) + D·T^E        (vapor pressure)
      102: Y = A·T^B / (1 + C/T + D/T²)              (liquid heat capacity)
      105: Y = A / B^(1 + (1 - T/C)^D)               (liquid density)
      106: Y = A·(1 - Tr)^(B + C·Tr + D·Tr² + E·Tr³) (saturated liquid density)
      107: Y = A + B·((C/T)/sinh(C/T))² + D·((E/T)/cosh(E/T))² (ideal gas Cp)
      110: ln(Y) = A + B/(C + T)                      (vapor thermal conductivity)
    """
    EQ_101 = 101  # ln(Y) = A + B/T + C·ln(T) + D·T^E
    EQ_102 = 102  # Y = A·T^B / (1 + C/T + D/T²)
    EQ_104 = 104  # Y = A + B·T + C·T² + D·T³ + E·T⁴
    EQ_105 = 105  # Y = A / B^(1 + (1 - T/C)^D)
    EQ_106 = 106  # Y = A·(1 - Tr)^(B + C·Tr + D·Tr² + E·Tr³)  (Tr = T/Tc)
    EQ_107 = 107  # Y = A + B·((C/T)/sinh(C/T))² + D·((E/T)/cosh(E/T))²
    EQ_110 = 110  # ln(Y) = A + B/(C + T)
    EQ_114 = 114  # ln(Y) = A + B/(C + T) + D·ln(T)
    EQ_116 = 116  # Y = A + B·T + C·T²


# ══════════════════════════════════════════════════════════════════════════════
# Substance Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Substance:
    """Complete substance definition for consequence analysis.

    All scalar properties are in SI units unless otherwise indicated.
    Temperature-dependent properties are computed via DIPPR correlation
    parameters stored as coefficient tuples.

    Required fields (minimum to identify a substance):
      - id: Unique identifier
      - name: Display name
      - molecular_weight: [g/mol]

    Optional fields default to None — the caller should check before using.
    """

    # ── Identity ──
    id: str
    name: str

    # ── Basic Physical Properties ──
    molecular_weight: float          # [g/mol]
    cas_number: str | None = None    # CAS Registry Number
    un_number: str | None = None     # UN Number for transport
    formula: str | None = None       # Chemical formula

    # ── Phase Properties ──
    normal_boiling_point: float | None = None   # Tb [K] at 1 atm
    melting_point: float | None = None          # Tm [K]
    critical_temperature: float | None = None   # Tc [K]
    critical_pressure: float | None = None      # Pc [Pa]
    critical_volume: float | None = None        # Vc [m³/mol]
    acentric_factor: float | None = None        # ω [-]
    phase_at_ambient: str = "liquid"            # gas, liquid, solid

    # ── Flammability ──
    flash_point: float | None = None            # [K]
    auto_ignition_temp: float | None = None     # AIT [K]
    lower_flammability_limit: float | None = None   # LFL [vol fraction]
    upper_flammability_limit: float | None = None   # UFL [vol fraction]
    heat_of_combustion: float | None = None     # ΔHc [J/kg]
    fire_class: str | None = None               # IA, IB, IC, II, IIIA, IIIB

    # ── Toxicity ──
    erpg1: float | None = None                  # ERPG-1 [mg/m³] (60 min)
    erpg2: float | None = None                  # ERPG-2 [mg/m³] (60 min)
    erpg3: float | None = None                  # ERPG-3 [mg/m³] (60 min)
    aegl1_60min: float | None = None            # AEGL-1 [ppm]
    aegl2_60min: float | None = None            # AEGL-2 [ppm]
    aegl3_60min: float | None = None            # AEGL-3 [ppm]
    idlh: float | None = None                   # IDLH [ppm] (NIOSH)
    tlv_twa: float | None = None                # TLV-TWA [ppm] (ACGIH)
    tlv_stel: float | None = None               # TLV-STEL [ppm] (ACGIH)
    probit_a: float | None = None               # Probit constant a
    probit_b: float | None = None               # Probit constant b
    probit_n: float | None = None               # Probit exponent n

    # ── Hazard Classification ──
    hazard_classes: List[str] = field(default_factory=list)   # HazardClass values
    nfpa_health: int | None = None              # NFPA 704 Health (0-4)
    nfpa_flammability: int | None = None        # NFPA 704 Flammability (0-4)
    nfpa_reactivity: int | None = None          # NFPA 704 Reactivity (0-4)

    # ── DIPPR Correlation Parameters ──
    # Each is Tuple[eq_type, A, B, C, D, E, Tmin, Tmax]
    #   Tmin, Tmax: validity range [K]
    dippr_vapor_pressure: Tuple[float, ...] | None = None      # Eq 101
    dippr_liquid_density: Tuple[float, ...] | None = None      # Eq 105 or 106
    dippr_heat_capacity_liquid: Tuple[float, ...] | None = None  # Eq 102
    dippr_heat_capacity_gas: Tuple[float, ...] | None = None     # Eq 107
    dippr_viscosity_liquid: Tuple[float, ...] | None = None      # Eq 101 (ln)
    dippr_viscosity_gas: Tuple[float, ...] | None = None         # Eq 102
    dippr_thermal_cond_liquid: Tuple[float, ...] | None = None   # Eq 102
    dippr_thermal_cond_gas: Tuple[float, ...] | None = None      # Eq 102
    dippr_surface_tension: Tuple[float, ...] | None = None       # Eq 106
    dippr_heat_of_vaporization: Tuple[float, ...] | None = None  # Eq 106
    dippr_ideal_gas_enthalpy: Tuple[float, ...] | None = None    # Eq 107

    # ── Additional Properties ──
    heat_of_vaporization: float | None = None   # ΔHv at normal BP [J/kg]
    specific_heat_liquid: float | None = None   # Cp at 25 °C [J/(kg·K)]
    specific_heat_vapor: float | None = None    # Cp (ideal gas) [J/(kg·K)]
    cp_cv_ratio: float | None = None            # k = Cp/Cv [-]
    liquid_density: float | None = None         # ρ at NTP [kg/m³]
    vapor_density_ratio: float | None = None    # relative to air [-]
    diffusion_coefficient: float | None = None  # in air at STP [m²/s]
    viscosity_liquid: float | None = None       # at NTP [Pa·s]
    viscosity_vapor: float | None = None        # at NTP [Pa·s]
    thermal_cond_liquid: float | None = None    # at NTP [W/(m·K)]
    thermal_cond_vapor: float | None = None     # at NTP [W/(m·K)]
    surface_tension: float | None = None        # at NTP [N/m]

    # ── Source Term Parameters ──
    liquid_specific_heat: float | None = None   # Cp liquid [J/(kg·K)] (alias)
    enthalpy_of_formation: float | None = None  # ΔHf° [J/mol]
    heat_of_vaporization_at_bp: float | None = None  # [J/kg]
    saturated_vapor_pressure_20c: float | None = None  # [Pa] at 293.15 K

    # ── Metadata ──
    description: str | None = None
    source: str | None = None       # data source reference
    tags: List[str] = field(default_factory=list)
    is_mixture: bool = False
    mixture_components: List[Dict[str, Any]] = field(default_factory=list)

    # ── Post-init normalization ──

    def __post_init__(self):
        """Ensure lists are proper types after deserialization."""
        if not isinstance(self.hazard_classes, list):
            self.hazard_classes = list(self.hazard_classes or [])
        if not isinstance(self.tags, list):
            self.tags = list(self.tags or [])
        if not isinstance(self.mixture_components, list):
            self.mixture_components = list(self.mixture_components or [])

    # ──────────────────────────────────────────────────────────────────────
    # Phase & State
    # ──────────────────────────────────────────────────────────────────────

    @property
    def is_gas_at_ambient(self) -> bool:
        """True if substance is gaseous at NTP (20 °C, 1 atm)."""
        if self.normal_boiling_point is None:
            return self.phase_at_ambient == "gas"
        return self.normal_boiling_point <= 293.15

    @property
    def is_flammable(self) -> bool:
        """Check if substance has any flammable classification."""
        return any(c in self.hazard_classes for c in
                   [HazardClass.FLAMMABLE, HazardClass.EXPLOSIVE])

    @property
    def is_toxic(self) -> bool:
        """Check if substance is classified as toxic."""
        return HazardClass.TOXIC in self.hazard_classes

    @property
    def vapor_density(self) -> float | None:
        """Vapor density relative to air [dimensionless]."""
        if self.vapor_density_ratio is not None:
            return self.vapor_density_ratio
        return self.molecular_weight / 28.9647

    # ──────────────────────────────────────────────────────────────────────
    # DIPPR Correlation Evaluators
    # ──────────────────────────────────────────────────────────────────────

    def _eval_dippr(self, params: Tuple[float, ...], T: float) -> float:
        """Evaluate a DIPPR correlation at temperature T [K].

        Args:
            params: (eq_type, A, B, C, D, E, Tmin, Tmax) tuple.
            T: Temperature in Kelvin.

        Returns:
            Computed property value.
        """
        if params is None:
            raise ValueError(f"DIPPR parameters not available for requested property")

        eq_type = int(params[0])
        A, B, C, D, E = params[1:6]
        Tmin = params[6] if len(params) > 6 else 0.0
        Tmax = params[7] if len(params) > 7 else float("inf")

        if T < Tmin - EPSILON or T > Tmax + EPSILON:
            if T < Tmin:
                T = Tmin  # clamp to valid range
            else:
                T = Tmax

        try:
            if eq_type == DIPPRParam.EQ_101:
                # ln(Y) = A + B/T + C·ln(T) + D·T^E
                ln_y = A + B / T + C * math.log(T) + D * (T ** E)
                return math.exp(max(ln_y, -100.0))

            elif eq_type == DIPPRParam.EQ_102:
                # Y = A·T^B / (1 + C/T + D/T²)
                denom = 1.0 + C / T + D / (T * T)
                if abs(denom) < EPSILON:
                    return 0.0
                return A * (T ** B) / denom

            elif eq_type == DIPPRParam.EQ_104:
                # Y = A + B·T + C·T² + D·T³ + E·T⁴
                return A + B * T + C * T * T + D * T ** 3 + E * T ** 4

            elif eq_type == DIPPRParam.EQ_105:
                # Y = A / B^(1 + (1 - T/C)^D)
                inner = (1.0 - T / C) ** D
                return A / (B ** (1.0 + inner))

            elif eq_type == DIPPRParam.EQ_106:
                # Y = A·(1 - Tr)^(B + C·Tr + D·Tr² + E·Tr³)
                Tr = T / C  # C = Tc in this form
                if Tr >= 1.0 - EPSILON:
                    Tr = 1.0 - EPSILON
                exponent = B + C * Tr + D * Tr * Tr + E * Tr ** 3
                return A * ((1.0 - Tr) ** exponent)

            elif eq_type == DIPPRParam.EQ_107:
                # Y = A + B·((C/T)/sinh(C/T))² + D·((E/T)/cosh(E/T))²
                term1 = (C / T) / math.sinh(C / T) if abs(C / T) > EPSILON else 1.0
                term2 = (E / T) / math.cosh(E / T) if abs(E / T) > EPSILON else 1.0
                return A + B * term1 * term1 + D * term2 * term2

            elif eq_type == DIPPRParam.EQ_110:
                # ln(Y) = A + B/(C + T)
                denom = C + T
                if abs(denom) < EPSILON:
                    return 0.0
                return math.exp(A + B / denom)

            elif eq_type == DIPPRParam.EQ_114:
                # ln(Y) = A + B/(C + T) + D·ln(T)
                y = A + B / (C + T) + D * math.log(T)
                return math.exp(max(y, -100.0))

            elif eq_type == DIPPRParam.EQ_116:
                # Y = A + B·T + C·T²
                return A + B * T + C * T * T

            else:
                raise ValueError(f"Unknown DIPPR equation type: {eq_type}")

        except (OverflowError, ValueError):
            return 0.0

    # ── Property Getters ──

    def vapor_pressure(self, T: float) -> float:
        """Vapor pressure at temperature T [K] → returns [Pa]."""
        if self.dippr_vapor_pressure is not None:
            return self._eval_dippr(self.dippr_vapor_pressure, T)
        if self.normal_boiling_point is not None and T > 10:
            # Clausius-Clapeyron approximation: ln(P2/P1) = -ΔHv/R * (1/T2 - 1/T1)
            Tr = T / self.normal_boiling_point
            omega = self.acentric_factor or 0.0
            ln_pr = (math.log(P_ATM) + (1.0 - omega) * math.log(Tr) +
                     (3.0 - 3.0 * omega) * (1.0 - 1.0 / Tr))
            return min(math.exp(max(ln_pr, -50.0)), self.critical_pressure or 1e8)
        raise ValueError(f"Vapor pressure not available for {self.name}")

    def liquid_density_at_T(self, T: float) -> float:
        """Liquid density at temperature T [K] → returns [kg/m³]."""
        if self.dippr_liquid_density is not None:
            params = self.dippr_liquid_density
            eq_type = int(params[0])
            molar_density = self._eval_dippr(params, T)  # [kmol/m³] from DIPPR
            if eq_type in (DIPPRParam.EQ_105, DIPPRParam.EQ_106):
                # DIPPR 105/106 return molar density [kmol/m³]; convert to [kg/m³]
                return molar_density * self.molecular_weight
            return molar_density
        if self.liquid_density is not None:
            return self.liquid_density
        raise ValueError(f"Liquid density not available for {self.name}")

    def heat_capacity_liquid(self, T: float) -> float:
        """Liquid heat capacity at T [K] → returns [J/(kg·K)]."""
        if self.dippr_heat_capacity_liquid is not None:
            cp_j_kmol_k = self._eval_dippr(self.dippr_heat_capacity_liquid, T)
            # DIPPR returns J/(kmol·K); MW is kg/kmol → J/(kg·K)
            return cp_j_kmol_k / self.molecular_weight
        if self.specific_heat_liquid is not None:
            return self.specific_heat_liquid
        return 2000.0  # conservative default [J/(kg·K)]

    def heat_capacity_gas(self, T: float) -> float:
        """Ideal gas heat capacity at T [K] → returns [J/(kg·K)]."""
        if self.dippr_heat_capacity_gas is not None:
            cp_j_kmol_k = self._eval_dippr(self.dippr_heat_capacity_gas, T)
            # DIPPR returns J/(kmol·K); MW is kg/kmol → J/(kg·K)
            return cp_j_kmol_k / self.molecular_weight
        if self.specific_heat_vapor is not None:
            return self.specific_heat_vapor
        return 1000.0  # conservative default

    def liquid_viscosity(self, T: float) -> float:
        """Liquid dynamic viscosity at T [K] → returns [Pa·s]."""
        if self.dippr_viscosity_liquid is not None:
            return self._eval_dippr(self.dippr_viscosity_liquid, T)
        if self.viscosity_liquid is not None:
            return self.viscosity_liquid
        return 0.001  # ~water default

    def gas_viscosity(self, T: float) -> float:
        """Gas dynamic viscosity at T [K] → returns [Pa·s]."""
        if self.dippr_viscosity_gas is not None:
            return self._eval_dippr(self.dippr_viscosity_gas, T)
        if self.viscosity_vapor is not None:
            return self.viscosity_vapor
        return 1.8e-5

    def liquid_thermal_conductivity(self, T: float) -> float:
        """Liquid thermal conductivity at T [K] → returns [W/(m·K)]."""
        if self.dippr_thermal_cond_liquid is not None:
            return self._eval_dippr(self.dippr_thermal_cond_liquid, T)
        if self.thermal_cond_liquid is not None:
            return self.thermal_cond_liquid
        return 0.15

    def gas_thermal_conductivity(self, T: float) -> float:
        """Gas thermal conductivity at T [K] → returns [W/(m·K)]."""
        if self.dippr_thermal_cond_gas is not None:
            return self._eval_dippr(self.dippr_thermal_cond_gas, T)
        if self.thermal_cond_vapor is not None:
            return self.thermal_cond_vapor
        return 0.025

    def surface_tension_at_T(self, T: float) -> float:
        """Surface tension at T [K] → returns [N/m]."""
        if self.dippr_surface_tension is not None:
            return self._eval_dippr(self.dippr_surface_tension, T)
        if self.surface_tension is not None:
            return self.surface_tension
        return 0.02

    def heat_of_vaporization_at_T(self, T: float) -> float:
        """Heat of vaporization at T [K] → returns [J/kg]."""
        if self.dippr_heat_of_vaporization is not None:
            dhv_j_kmol = self._eval_dippr(self.dippr_heat_of_vaporization, T)
            return dhv_j_kmol / (self.molecular_weight)  # J/kmol → J/kg
        if self.heat_of_vaporization is not None:
            return self.heat_of_vaporization
        return 5e5  # conservative default

    # ──────────────────────────────────────────────────────────────────────
    # Derived / Convenience Properties
    # ──────────────────────────────────────────────────────────────────────

    def gas_density_at(self, T: float, P: float) -> float:
        """Ideal gas density at T [K], P [Pa] → returns [kg/m³]."""
        mw_kg_per_mol = self.molecular_weight * 0.001
        return (P * mw_kg_per_mol) / (R * T)

    def vapor_pressure_at_20c(self) -> float | None:
        """Saturated vapor pressure at 20 °C [Pa]."""
        try:
            return self.vapor_pressure(293.15)
        except (ValueError, AttributeError):
            return self.saturated_vapor_pressure_20c

    def boiling_point_at_pressure(self, P: float) -> float:
        """Estimate boiling point at pressure P [Pa] using Clausius-Clapeyron.

        Args:
            P: Absolute pressure [Pa].

        Returns:
            Boiling temperature [K].
        """
        if self.normal_boiling_point is None:
            raise ValueError(f"Boiling point not available for {self.name}")

        Tb = self.normal_boiling_point
        dhv = self.heat_of_vaporization or 300000.0
        # Clapeyron: ln(P2/P1) = (-ΔHv/R)·(1/T2 - 1/T1)
        # → 1/T2 = 1/T1 - (R/ΔHv)·ln(P2/P1)
        ln_ratio = math.log(max(P, EPSILON) / P_ATM)
        one_over_T = 1.0 / Tb - (R / dhv) * ln_ratio
        if one_over_T <= EPSILON:
            return 1e6  # absurdly high
        return 1.0 / one_over_T

    def mole_fraction_to_mass_concentration(self, ppm_vol: float,
                                             T: float = 298.15,
                                             P: float = P_ATM) -> float:
        """Convert ppm(v) to mass concentration [kg/m³] at given T, P.

        Args:
            ppm_vol: Concentration in parts per million (volume/volume).
            T: Temperature [K] (default: 298.15).
            P: Absolute pressure [Pa] (default: 101325).

        Returns:
            Mass concentration [kg/m³].
        """
        mw = self.molecular_weight * 0.001  # kg/mol
        vol_frac = ppm_vol * 1e-6
        return (P * mw * vol_frac) / (R * T)

    def mass_concentration_to_ppm(self, conc: float,
                                   T: float = 298.15,
                                   P: float = P_ATM) -> float:
        """Convert mass concentration [kg/m³] to ppm(v) at given T, P."""
        mw = self.molecular_weight * 0.001
        if mw < EPSILON:
            return float("inf")
        vol_frac = (conc * R * T) / (P * mw)
        return vol_frac * 1e6

    # ──────────────────────────────────────────────────────────────────────
    # Serialization Helpers
    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON export)."""
        import dataclasses
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None and f.default is None:
                continue
            if (f.default_factory is not dataclasses.MISSING
                    and f.default_factory is not None):
                try:
                    if val == f.default_factory():
                        continue
                except TypeError:
                    pass
            result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Substance":
        """Create a Substance from a dictionary (from JSON import).

        Supports both long names (cas_number) and short aliases (cas).
        """
        # Alias map: short JSON key → dataclass field name
        ALIASES = {
            'cas': 'cas_number',
            'un': 'un_number',
            'mw': 'molecular_weight',
            'nbp': 'normal_boiling_point',
            'tc': 'critical_temperature',
            'pc': 'critical_pressure',
            'vc': 'critical_volume',
            'omega': 'acentric_factor',
            'rho_liq': 'liquid_density',
            'lfl': 'lower_flammability_limit',
            'ufl': 'upper_flammability_limit',
            'flash_pt': 'flash_point',
            'ait': 'auto_ignition_temp',
            'dhc': 'heat_of_combustion',
            'dhv': 'heat_of_vaporization',
        }
        expanded = {}
        for k, v in data.items():
            key = ALIASES.get(k, k)
            expanded[key] = v
        # Handle dippr sub-dict → individual dippr_* fields
        dippr_data = expanded.pop('dippr', None)
        if dippr_data and isinstance(dippr_data, dict):
            DIPPR_MAP = {
                'vp': 'dippr_vapor_pressure',
                'vp_params': 'dippr_vapor_pressure',
                'liq_density': 'dippr_liquid_density',
                'rho_liq_params': 'dippr_liquid_density',
                'liq_cp': 'dippr_heat_capacity_liquid',
                'gas_cp': 'dippr_heat_capacity_gas',
                'liq_visc': 'dippr_viscosity_liquid',
                'gas_visc': 'dippr_viscosity_gas',
                'liq_tcond': 'dippr_thermal_cond_liquid',
                'gas_tcond': 'dippr_thermal_cond_gas',
                'surf_tens': 'dippr_surface_tension',
                'hvap': 'dippr_heat_of_vaporization',
                'h_ideal': 'dippr_ideal_gas_enthalpy',
            }
            for dk, dv in dippr_data.items():
                target = DIPPR_MAP.get(dk)
                if target:
                    # Convert dict to tuple for tuple-typed fields
                    if isinstance(dv, dict):
                        from .dippr import DIPPRParams
                        try:
                            expanded[target] = DIPPRParams.from_dict(dv)
                        except Exception:
                            expanded[target] = dv
                    else:
                        expanded[target] = dv
        # Handle toxic fields
        for toxic_key in ['aegl1_60min', 'aegl2_60min', 'aegl3_60min',
                          'idlh', 'probit_a', 'probit_b', 'probit_n',
                          'toxic_n']:
            if toxic_key in expanded and expanded[toxic_key] is None:
                expanded[toxic_key] = None
        # Filter to valid dataclass keys only
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in expanded.items() if k in valid_keys}
        # Auto-generate id from name if missing
        if 'id' not in filtered and 'name' in filtered:
            filtered['id'] = filtered['name']
        elif 'id' not in filtered:
            filtered['id'] = 'unknown'
        return cls(**filtered)

    def __repr__(self) -> str:
        props = []
        if self.molecular_weight:
            props.append(f"MW={self.molecular_weight:.1f}")
        if self.normal_boiling_point:
            bp_c = self.normal_boiling_point - T_0C
            props.append(f"Tb={bp_c:.1f}°C")
        if self.flash_point:
            fp_c = self.flash_point - T_0C
            props.append(f"FP={fp_c:.1f}°C")
        phase = "G" if self.is_gas_at_ambient else "L"
        props.append(phase)
        return f"Substance({self.name!r}, {', '.join(props)})"


# ══════════════════════════════════════════════════════════════════════════════
# Mixture Helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_mixture_molecular_weight(
    components: List[Tuple[Substance, float]]
) -> float:
    """Compute weight-averaged molecular weight for a mixture.

    Args:
        components: List of (Substance, mass_fraction) tuples.
            Mass fractions must sum to 1.0.

    Returns:
        Mixture molecular weight [g/mol].
    """
    if not components:
        return 0.0
    total = 0.0
    for sub, frac in components:
        total += sub.molecular_weight * frac
    return total


def compute_mixture_vapor_pressure(
    components: List[Tuple[Substance, float]],
    T: float,
) -> float:
    """Raoult's law vapor pressure for liquid mixture.

    Args:
        components: List of (Substance, mole_fraction) tuples.
        T: Temperature [K].

    Returns:
        Mixture vapor pressure [Pa].
    """
    total = 0.0
    for sub, x in components:
        total += x * sub.vapor_pressure(T)
    return total


def is_mixture_flammable(
    components: List[Tuple[Substance, float]],
    T: float,
) -> bool:
    """Check if a vapor mixture composition is flammable using Le Chatelier's rule.

    1 / LFL_mix = Σ (xi / LFL_i)

    Args:
        components: List of (Substance, mole_fraction in vapor) tuples.
        T: Temperature [K] (not used for LFL — constants assumed).

    Returns:
        True if the vapor mixture is in the flammable range.
    """
    inv_lfl = 0.0
    total_x = 0.0
    for sub, x in components:
        if x <= EPSILON:
            continue
        if sub.lower_flammability_limit is None or sub.lower_flammability_limit <= 0:
            continue
        inv_lfl += x / sub.lower_flammability_limit
        total_x += x
    if inv_lfl <= EPSILON or total_x <= EPSILON:
        return False
    lfl_mix = 1.0 / inv_lfl
    return total_x >= lfl_mix
