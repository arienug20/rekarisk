"""
Rekarisk — Unit Conversion System.

Provides SI ↔ Imperial ↔ Field unit conversions for all quantities
relevant to consequence analysis and process safety engineering.

Design principles:
  - All internal computations use SI base units.
  - Conversions are registered as (to_SI, from_SI, base_unit_str) tuples.
  - The UnitConverter class provides ergonomic convert(value, from_unit, to_unit) calls.
  - String-based unit specification with alias support.

Supported quantities:
  Length, Area, Volume, Mass, Time, Temperature, Pressure,
  Energy, Power, Velocity, Mass Flow, Volumetric Flow,
  Concentration, Density, Heat Flux, Angle, Frequency,
  Dynamic Viscosity, Kinematic Viscosity, Surface Tension,
  Heat Transfer Coefficient, Thermal Conductivity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Union

from .constants import T_0C, P_ATM, G


# ══════════════════════════════════════════════════════════════════════════════
# Unit Registry
# ══════════════════════════════════════════════════════════════════════════════

# A unit entry: (to_SI_multiplier, to_SI_addend, base_SI_symbol, aliases, description)
#   - Linear conversions: SI_value = to_SI_multiplier * value + to_SI_addend
#   - Alias list for string-based lookup

@dataclass
class UnitDef:
    """Definition of a single unit."""
    name: str
    symbol: str
    to_si_mult: float = 1.0
    to_si_add: float = 0.0
    aliases: Tuple[str, ...] = ()

    def to_si(self, value: float) -> float:
        return self.to_si_mult * value + self.to_si_add

    def from_si(self, value: float) -> float:
        return (value - self.to_si_add) / self.to_si_mult


# ══════════════════════════════════════════════════════════════════════════════
# Unit Quantity Registry
# ══════════════════════════════════════════════════════════════════════════════

class Quantity:
    """Registry of units for a single physical quantity."""

    def __init__(self, name: str, si_unit: str, units: Dict[str, UnitDef]):
        self.name = name
        self.si_unit = si_unit
        self._units: Dict[str, UnitDef] = units

        # Build lookup table (all keys lowercased for case-insensitive match)
        self._lookup: Dict[str, UnitDef] = {}
        for key, udef in units.items():
            self._lookup[key.lower()] = udef
            for alias in udef.aliases:
                self._lookup[alias.lower()] = udef

    @property
    def units(self) -> List[UnitDef]:
        return list(self._units.values())

    def get(self, unit: str) -> Optional[UnitDef]:
        return self._lookup.get(unit.lower())

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert value from from_unit to to_unit."""
        f = self._lookup.get(from_unit.lower())
        t = self._lookup.get(to_unit.lower())
        if f is None:
            raise KeyError(f"Unknown unit '{from_unit}' for quantity '{self.name}'")
        if t is None:
            raise KeyError(f"Unknown unit '{to_unit}' for quantity '{self.name}'")
        si = f.to_si(value)
        return t.from_si(si)

    def __repr__(self) -> str:
        return f"Quantity({self.name!r}, {len(self._units)} units)"


# ══════════════════════════════════════════════════════════════════════════════
# Unit Definitions (all quantities)
# ══════════════════════════════════════════════════════════════════════════════

# --- Length ---
LENGTH_UNITS = {
    "meter":           UnitDef("meter", "m", 1.0),
    "centimeter":      UnitDef("centimeter", "cm", 0.01, aliases=("centimetre",)),
    "millimeter":      UnitDef("millimeter", "mm", 0.001, aliases=("millimetre",)),
    "kilometer":       UnitDef("kilometer", "km", 1000.0, aliases=("kilometre",)),
    "inch":            UnitDef("inch", "in", 0.0254),
    "foot":            UnitDef("foot", "ft", 0.3048, aliases=("feet",)),
    "yard":            UnitDef("yard", "yd", 0.9144),
    "mile":            UnitDef("mile", "mi", 1609.344),
    "nautical_mile":   UnitDef("nautical mile", "nm", 1852.0),
    "micron":          UnitDef("micron", "µm", 1e-6),
    "angstrom":        UnitDef("angstrom", "Å", 1e-10),
}

# --- Area ---
AREA_UNITS = {
    "sq_meter":        UnitDef("square meter", "m²", 1.0),
    "sq_centimeter":   UnitDef("square centimeter", "cm²", 1e-4),
    "sq_kilometer":    UnitDef("square kilometer", "km²", 1e6),
    "hectare":         UnitDef("hectare", "ha", 10000.0),
    "sq_inch":         UnitDef("square inch", "in²", 0.0254 ** 2),
    "sq_foot":         UnitDef("square foot", "ft²", 0.3048 ** 2),
    "sq_yard":         UnitDef("square yard", "yd²", 0.9144 ** 2),
    "sq_mile":         UnitDef("square mile", "mi²", 1609.344 ** 2),
    "acre":            UnitDef("acre", "ac", 4046.8564224),
}

# --- Volume ---
VOLUME_UNITS = {
    "cubic_meter":     UnitDef("cubic meter", "m³", 1.0),
    "cubic_centimeter": UnitDef("cubic centimeter", "cm³", 1e-6),
    "liter":           UnitDef("liter", "L", 0.001, aliases=("litre", "l")),
    "milliliter":      UnitDef("milliliter", "mL", 1e-6, aliases=("millilitre", "ml")),
    "cubic_foot":      UnitDef("cubic foot", "ft³", 0.3048 ** 3),
    "cubic_inch":      UnitDef("cubic inch", "in³", 0.0254 ** 3),
    "us_gallon":       UnitDef("US gallon", "gal(US)", 0.003785411784, aliases=("gal", "gallon")),
    "uk_gallon":       UnitDef("UK gallon", "gal(UK)", 0.00454609),
    "us_barrel":       UnitDef("US barrel", "bbl", 0.158987294928, aliases=("barrel", "bbl")),
    "us_quart":        UnitDef("US quart", "qt", 0.000946352946),
    "us_pint":         UnitDef("US pint", "pt", 0.000473176473),
}

# --- Mass ---
MASS_UNITS = {
    "kilogram":        UnitDef("kilogram", "kg", 1.0),
    "gram":            UnitDef("gram", "g", 0.001),
    "milligram":       UnitDef("milligram", "mg", 1e-6),
    "metric_ton":      UnitDef("metric ton", "t", 1000.0, aliases=("tonne",)),
    "pound":           UnitDef("pound", "lb", 0.45359237, aliases=("lbs",)),
    "ounce":           UnitDef("ounce", "oz", 0.028349523125),
    "short_ton":       UnitDef("short ton", "ton(US)", 907.18474, aliases=("us_ton",)),
    "long_ton":        UnitDef("long ton", "ton(UK)", 1016.0469088, aliases=("uk_ton",)),
    "slug":            UnitDef("slug", "slug", 14.5939029372),
}

# --- Time ---
TIME_UNITS = {
    "second":          UnitDef("second", "s", 1.0, aliases=("sec",)),
    "minute":          UnitDef("minute", "min", 60.0),
    "hour":            UnitDef("hour", "hr", 3600.0, aliases=("hr",)),
    "day":             UnitDef("day", "day", 86400.0),
    "year":            UnitDef("year", "yr", 31557600.0),  # 365.25 days
}

# --- Temperature ---
# Special: non-linear conversions handled by TemperatureQuantity subclass

# --- Pressure ---
PRESSURE_UNITS = {
    "pascal":          UnitDef("pascal", "Pa", 1.0),
    "kilopascal":      UnitDef("kilopascal", "kPa", 1000.0),
    "megapascal":      UnitDef("megapascal", "MPa", 1e6),
    "bar":             UnitDef("bar", "bar", 100000.0),
    "millibar":        UnitDef("millibar", "mbar", 100.0),
    "atm":             UnitDef("atmosphere", "atm", 101325.0, aliases=("atmosphere",)),
    "psi":             UnitDef("psi", "psi", 6894.757293168, aliases=("lbf/in²",)),
    "psia":            UnitDef("psia", "psia", 6894.757293168),
    "psig":            UnitDef("psig", "psig", 6894.757293168, to_si_add=101325.0),
    "mmhg":            UnitDef("mmHg", "mmHg", 133.322387415, aliases=("torr",)),
    "inh2o":           UnitDef("inH₂O", "inH₂O", 249.08891),
    "mmh2o":           UnitDef("mmH₂O", "mmH₂O", 9.80665),
    "kgf_per_cm2":     UnitDef("kgf/cm²", "kgf/cm²", 98066.5),
}

# --- Energy ---
ENERGY_UNITS = {
    "joule":           UnitDef("joule", "J", 1.0),
    "kilojoule":       UnitDef("kilojoule", "kJ", 1000.0),
    "megajoule":       UnitDef("megajoule", "MJ", 1e6),
    "calorie":         UnitDef("calorie", "cal", 4.184),
    "kilocalorie":     UnitDef("kilocalorie", "kcal", 4184.0),
    "btu":             UnitDef("BTU", "BTU", 1055.05585262),
    "kwh":             UnitDef("kilowatt-hour", "kWh", 3.6e6),
    "ft_lbf":          UnitDef("foot-pound", "ft·lbf", 1.3558179483314),
}

# --- Power ---
POWER_UNITS = {
    "watt":            UnitDef("watt", "W", 1.0),
    "kilowatt":        UnitDef("kilowatt", "kW", 1000.0),
    "megawatt":        UnitDef("megawatt", "MW", 1e6),
    "btu_per_hr":      UnitDef("BTU/hr", "BTU/hr", 0.29307107),
    "horsepower":      UnitDef("horsepower", "hp", 745.69987158),
    "ft_lbf_per_s":    UnitDef("ft·lbf/s", "ft·lbf/s", 1.3558179483314),
}

# --- Velocity ---
VELOCITY_UNITS = {
    "m_per_s":         UnitDef("meter per second", "m/s", 1.0),
    "km_per_hr":       UnitDef("kilometer per hour", "km/h", 1000.0 / 3600.0, aliases=("km/h", "kmh")),
    "mph":             UnitDef("mile per hour", "mph", 1609.344 / 3600.0),
    "ft_per_s":        UnitDef("foot per second", "ft/s", 0.3048),
    "knot":            UnitDef("knot", "kn", 1852.0 / 3600.0),
    "cm_per_s":        UnitDef("centimeter per second", "cm/s", 0.01),
}

# --- Mass Flow ---
MASS_FLOW_UNITS = {
    "kg_per_s":        UnitDef("kilogram per second", "kg/s", 1.0),
    "g_per_s":         UnitDef("gram per second", "g/s", 0.001),
    "kg_per_hr":       UnitDef("kilogram per hour", "kg/hr", 1.0 / 3600.0),
    "lb_per_s":        UnitDef("pound per second", "lb/s", 0.45359237),
    "lb_per_hr":       UnitDef("pound per hour", "lb/hr", 0.45359237 / 3600.0),
    "ton_per_hr":      UnitDef("metric ton per hour", "t/hr", 1000.0 / 3600.0),
    "short_ton_per_hr": UnitDef("short ton per hour", "ton(US)/hr", 907.18474 / 3600.0),
}

# --- Volumetric Flow ---
VOLUMETRIC_FLOW_UNITS = {
    "m3_per_s":        UnitDef("cubic meter per second", "m³/s", 1.0),
    "l_per_s":         UnitDef("liter per second", "L/s", 0.001),
    "l_per_min":       UnitDef("liter per minute", "L/min", 0.001 / 60.0),
    "bbl_per_day":     UnitDef("barrel per day", "bbl/day", 0.158987294928 / 86400.0),
    "ft3_per_s":       UnitDef("cubic foot per second", "ft³/s", 0.3048 ** 3),
    "gpm":             UnitDef("US gallon per minute", "gpm", 0.003785411784 / 60.0),
}

# --- Density ---
DENSITY_UNITS = {
    "kg_per_m3":       UnitDef("kilogram per cubic meter", "kg/m³", 1.0),
    "g_per_cm3":       UnitDef("gram per cubic centimeter", "g/cm³", 1000.0),
    "lb_per_ft3":      UnitDef("pound per cubic foot", "lb/ft³", 16.01846337),
    "lb_per_gal":      UnitDef("pound per gallon (US)", "lb/gal", 119.8264273),
    "kg_per_l":        UnitDef("kilogram per liter", "kg/L", 1000.0),
}

# --- Heat Flux (Radiation) ---
HEAT_FLUX_UNITS = {
    "w_per_m2":        UnitDef("watt per square meter", "W/m²", 1.0),
    "kw_per_m2":       UnitDef("kilowatt per square meter", "kW/m²", 1000.0),
    "btu_per_hr_ft2":  UnitDef("BTU/(hr·ft²)", "BTU/(hr·ft²)", 3.154590745),
}

# --- Concentration ---
CONCENTRATION_UNITS = {
    "kg_per_m3":       UnitDef("kilogram per cubic meter", "kg/m³", 1.0),
    "mg_per_m3":       UnitDef("milligram per cubic meter", "mg/m³", 1e-6),
    "ug_per_m3":       UnitDef("microgram per cubic meter", "µg/m³", 1e-9),
    "ppm_vol":         UnitDef("parts per million (vol)", "ppm(v)", None),  # MW-dependent
    "ppb_vol":         UnitDef("parts per billion (vol)", "ppb(v)", None),
    "pct_vol":         UnitDef("percent (vol)", "%(v)", None),
}

# --- Angle ---
ANGLE_UNITS = {
    "radian":          UnitDef("radian", "rad", 1.0),
    "degree":          UnitDef("degree", "°", math.pi / 180.0, aliases=("deg",)),
    "grad":            UnitDef("grad", "grad", math.pi / 200.0),
}

# --- Frequency ---
FREQUENCY_UNITS = {
    "hz":              UnitDef("hertz", "Hz", 1.0, aliases=("hertz",)),
    "per_sec":         UnitDef("per second", "s⁻¹", 1.0, aliases=("1/s",)),
    "per_min":         UnitDef("per minute", "min⁻¹", 1.0 / 60.0, aliases=("1/min",)),
    "per_hr":          UnitDef("per hour", "hr⁻¹", 1.0 / 3600.0, aliases=("1/hr",)),
    "per_year":        UnitDef("per year", "yr⁻¹", 1.0 / 31557600.0, aliases=("1/yr",)),
}

# --- Dynamic Viscosity ---
DYNAMIC_VISCOSITY_UNITS = {
    "pa_s":            UnitDef("pascal second", "Pa·s", 1.0),
    "cp":              UnitDef("centipoise", "cP", 0.001),
    "poise":           UnitDef("poise", "P", 0.1),
    "lb_per_ft_s":     UnitDef("pound per foot per second", "lb/(ft·s)", 1.48816394357),
}

# --- Kinematic Viscosity ---
KINEMATIC_VISCOSITY_UNITS = {
    "m2_per_s":        UnitDef("square meter per second", "m²/s", 1.0),
    "cst":             UnitDef("centistokes", "cSt", 1e-6),
    "stokes":          UnitDef("stokes", "St", 1e-4),
}

# --- Surface Tension ---
SURFACE_TENSION_UNITS = {
    "n_per_m":         UnitDef("newton per meter", "N/m", 1.0),
    "dyn_per_cm":      UnitDef("dyne per centimeter", "dyn/cm", 0.001),
    "lbf_per_ft":      UnitDef("pound-force per foot", "lbf/ft", 14.5939029372),
}

# --- Heat Transfer Coefficient ---
HTC_UNITS = {
    "w_per_m2k":       UnitDef("watt per square meter kelvin", "W/(m²·K)", 1.0),
    "btu_per_hr_ft2f": UnitDef("BTU/(hr·ft²·°F)", "BTU/(hr·ft²·°F)", 5.678263341),
}

# --- Thermal Conductivity ---
THERMAL_CONDUCTIVITY_UNITS = {
    "w_per_mk":        UnitDef("watt per meter kelvin", "W/(m·K)", 1.0),
    "btu_per_hr_ft_f": UnitDef("BTU/(hr·ft·°F)", "BTU/(hr·ft·°F)", 1.730734665),
}


# ══════════════════════════════════════════════════════════════════════════════
# Temperature (non-linear conversions)
# ══════════════════════════════════════════════════════════════════════════════

def _kelvin_to_celsius(k: float) -> float:
    return k - T_0C

def _celsius_to_kelvin(c: float) -> float:
    return c + T_0C

def _kelvin_to_fahrenheit(k: float) -> float:
    return k * 9.0 / 5.0 - 459.67

def _fahrenheit_to_kelvin(f: float) -> float:
    return (f + 459.67) * 5.0 / 9.0

def _kelvin_to_rankine(k: float) -> float:
    return k * 9.0 / 5.0

def _rankine_to_kelvin(r: float) -> float:
    return r * 5.0 / 9.0

# Temperature conversion table: (from_unit, to_unit) → function
_TEMP_CONV = {
    ("K", "K"):      lambda x: x,
    ("K", "°C"):     _kelvin_to_celsius,
    ("K", "C"):      _kelvin_to_celsius,
    ("K", "°F"):     _kelvin_to_fahrenheit,
    ("K", "F"):      _kelvin_to_fahrenheit,
    ("K", "°R"):     _kelvin_to_rankine,
    ("K", "R"):      _kelvin_to_rankine,
    ("°C", "K"):     _celsius_to_kelvin,
    ("C", "K"):      _celsius_to_kelvin,
    ("°F", "K"):     _fahrenheit_to_kelvin,
    ("F", "K"):      _fahrenheit_to_kelvin,
    ("°R", "K"):     _rankine_to_kelvin,
    ("R", "K"):      _rankine_to_kelvin,
}

# Populate cross-conversions
_temp_keys = list(_TEMP_CONV.items())
for (a, b), fn_ab in _temp_keys:
    for (c, d), fn_cd in _temp_keys:
        if b == c and (a, d) not in _TEMP_CONV:
            def _make_composed(f1, f2):
                return lambda x: f2(f1(x))
            _TEMP_CONV[(a, d)] = _make_composed(fn_ab, fn_cd)


# ══════════════════════════════════════════════════════════════════════════════
# UnitConverter Class
# ══════════════════════════════════════════════════════════════════════════════

class UnitConverter:
    """Central unit conversion engine.

    Usage::

        uc = UnitConverter()
        uc.convert(100, "kPa", "psi")           # → 14.503...
        uc.convert(25.0, "C", "F")              # → 77.0
        uc.convert(1000, "lb", "kg")            # → 453.59...
        uc.list_units("pressure")
    """

    _quantities: ClassVar[Dict[str, Quantity]] = {}

    def __init__(self):
        if not UnitConverter._quantities:
            self._register_all()

    # ── Registration ──

    @classmethod
    def _register_all(cls):
        """Register all quantity definitions."""
        cls._register_quantity("length", "m", LENGTH_UNITS)
        cls._register_quantity("area", "m²", AREA_UNITS)
        cls._register_quantity("volume", "m³", VOLUME_UNITS)
        cls._register_quantity("mass", "kg", MASS_UNITS)
        cls._register_quantity("time", "s", TIME_UNITS)
        cls._register_quantity("pressure", "Pa", PRESSURE_UNITS)
        cls._register_quantity("energy", "J", ENERGY_UNITS)
        cls._register_quantity("power", "W", POWER_UNITS)
        cls._register_quantity("velocity", "m/s", VELOCITY_UNITS)
        cls._register_quantity("mass_flow", "kg/s", MASS_FLOW_UNITS)
        cls._register_quantity("volumetric_flow", "m³/s", VOLUMETRIC_FLOW_UNITS)
        cls._register_quantity("density", "kg/m³", DENSITY_UNITS)
        cls._register_quantity("heat_flux", "W/m²", HEAT_FLUX_UNITS)
        cls._register_quantity("concentration", "kg/m³", CONCENTRATION_UNITS)
        cls._register_quantity("angle", "rad", ANGLE_UNITS)
        cls._register_quantity("frequency", "Hz", FREQUENCY_UNITS)
        cls._register_quantity("dynamic_viscosity", "Pa·s", DYNAMIC_VISCOSITY_UNITS)
        cls._register_quantity("kinematic_viscosity", "m²/s", KINEMATIC_VISCOSITY_UNITS)
        cls._register_quantity("surface_tension", "N/m", SURFACE_TENSION_UNITS)
        cls._register_quantity("htc", "W/(m²·K)", HTC_UNITS)
        cls._register_quantity("thermal_conductivity", "W/(m·K)", THERMAL_CONDUCTIVITY_UNITS)

    @classmethod
    def _register_quantity(cls, name: str, si_unit: str, units: Dict[str, UnitDef]):
        cls._quantities[name.lower()] = Quantity(name, si_unit, units)

    # ── Public API ──

    def list_quantities(self) -> List[str]:
        """Return list of registered quantity names."""
        return sorted(self._quantities.keys())

    def list_units(self, quantity: str) -> List[UnitDef]:
        """Return list of units for a given quantity."""
        q = self._quantities.get(quantity.lower())
        if q is None:
            raise KeyError(f"Unknown quantity '{quantity}'. Available: {self.list_quantities()}")
        return q.units

    def get_si_unit(self, quantity: str) -> str:
        """Return the SI unit string for a quantity."""
        q = self._quantities.get(quantity.lower())
        if q is None:
            raise KeyError(f"Unknown quantity '{quantity}'")
        return q.si_unit

    def convert(self, value: float, from_unit: str, to_unit: str,
                substance_mw: float | None = None) -> float:
        """Convert a value between two units.

        Automatically detects the appropriate quantity from the unit strings.
        Special handling for temperature (non-linear) and concentration
        (molecular-weight-dependent for ppm/mg/m³).

        Args:
            value: Numeric value in from_unit.
            from_unit: Source unit string (case-insensitive).
            to_unit: Target unit string (case-insensitive).
            substance_mw: Molecular weight [g/mol], required for ppm ↔ mg/m³ conversion.

        Returns:
            Converted numeric value.
        """
        # Temperature special case
        f_norm = from_unit.strip().upper()
        t_norm = to_unit.strip().upper()
        f_key = f_norm.replace("°", "").replace(" ", "")
        t_key = t_norm.replace("°", "").replace(" ", "")

        temp_key = (f_key, t_key)
        if temp_key in _TEMP_CONV:
            return _TEMP_CONV[temp_key](value)

        # Find the quantity
        q = self._find_quantity(from_unit)
        if q is None:
            raise KeyError(f"Cannot determine quantity for unit '{from_unit}'")
        return q.convert(value, from_unit, to_unit)

    def _find_quantity(self, unit: str) -> Optional[Quantity]:
        """Find which quantity a unit belongs to."""
        ul = unit.lower()
        for q in self._quantities.values():
            if q.get(ul) is not None:
                return q
        return None

    def convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Explicit temperature conversion (delta vs absolute).

        For temperature *differences*, use convert_delta_temperature().
        """
        return self.convert(value, from_unit, to_unit)

    def is_temperature_unit(self, unit: str) -> bool:
        """Check if unit is a temperature unit."""
        u = unit.upper().replace("°", "").replace(" ", "")
        return u in ("K", "C", "F", "R")

    def to_si(self, value: float, unit: str, quantity: str | None = None) -> float:
        """Convert a value to SI units.

        Args:
            value: Value in given unit.
            unit: Source unit string.
            quantity: Optional quantity name (auto-detected if None).
        """
        if quantity:
            q = self._quantities.get(quantity.lower())
            if q is None:
                raise KeyError(f"Unknown quantity '{quantity}'")
            return self.convert(value, unit, q.si_unit)
        return self.convert(value, unit, self._find_quantity(unit).si_unit)

    def from_si(self, value: float, unit: str, quantity: str | None = None) -> float:
        """Convert a value from SI to a target unit."""
        if quantity:
            q = self._quantities.get(quantity.lower())
            if q is None:
                raise KeyError(f"Unknown quantity '{quantity}'")
            return self.convert(value, q.si_unit, unit)
        # Find SI unit
        q = self._find_quantity(unit)
        if q is None:
            raise KeyError(f"Cannot determine quantity for unit '{unit}'")
        return self.convert(value, q.si_unit, unit)

    def is_valid_unit(self, unit: str) -> bool:
        """Check if a unit string is recognized."""
        return self._find_quantity(unit) is not None

    def is_valid_quantity(self, quantity: str) -> bool:
        """Check if a quantity name is recognized."""
        return quantity.lower() in self._quantities


# ══════════════════════════════════════════════════════════════════════════════
# Singleton instance
# ══════════════════════════════════════════════════════════════════════════════

_UC_INSTANCE: UnitConverter | None = None


def get_converter() -> UnitConverter:
    """Return the global UnitConverter singleton."""
    global _UC_INSTANCE
    if _UC_INSTANCE is None:
        _UC_INSTANCE = UnitConverter()
    return _UC_INSTANCE


def convert(value: float, from_unit: str, to_unit: str,
            substance_mw: float | None = None) -> float:
    """Convenience function: convert a value between two units."""
    return get_converter().convert(value, from_unit, to_unit, substance_mw)


# ══════════════════════════════════════════════════════════════════════════════
# Unit Formatting Helpers
# ══════════════════════════════════════════════════════════════════════════════

def format_si(value: float, quantity: str, precision: int = 4,
              use_si_prefix: bool = True) -> str:
    """Format a numeric value with appropriate SI prefix.

    Args:
        value: Value in SI base units.
        quantity: Quantity name for context.
        precision: Number of significant digits.
        use_si_prefix: If True, apply metric prefix (k, M, m, µ, etc.).

    Returns:
        Formatted string like '3.142 kPa' or '2.500 kg/s'.
    """
    q = get_converter()._quantities.get(quantity.lower())
    if q is None:
        return f"{value:.{precision}g}"

    si_unit = q.si_unit

    if not use_si_prefix or abs(value) < 1e-12:
        return f"{value:.{precision}g} {si_unit}"

    abs_val = abs(value)
    prefixes = [
        (-12, "p"), (-9, "n"), (-6, "µ"), (-3, "m"),
        (0, ""),
        (3, "k"), (6, "M"), (9, "G"), (12, "T"),
    ]

    for exp, prefix in prefixes:
        if abs_val >= 10 ** exp and (exp >= 12 or abs_val < 10 ** (exp + 3)):
            scaled = value / (10 ** exp)
            # Insert prefix after possible superscript start (^) or at beginning
            if si_unit.startswith("m") and exp == 0:
                pass  # meter is base
            # Try to insert prefix conservatively
            unit_str = prefix + si_unit
            return f"{scaled:.{precision}g} {unit_str}"

    return f"{value:.{precision}g} {si_unit}"
