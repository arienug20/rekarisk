"""
Rekarisk — Input Validation & Sanity Checks.

Provides a comprehensive validation framework for consequence analysis
inputs. Each validator returns a ValidationResult with pass/fail status
and descriptive error messages.

Capabilities:
  - Range checks (min/max)
  - Unit-aware validation
  - Cross-field consistency checks
  - Sanity checks for physically impossible combinations
  - Substance-specific validation (e.g., toxic endpoint availability)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .constants import (
    P_ATM, MIN_WIND_SPEED, MAX_WIND_SPEED, T_0C,
    PG_STABILITY_CLASSES, SURFACE_ROUGHNESS, EPSILON,
)


# ══════════════════════════════════════════════════════════════════════════════
# Validation Result
# ══════════════════════════════════════════════════════════════════════════════

class Severity(Enum):
    """Validation message severity."""
    ERROR = "error"       # Invalid — must be fixed
    WARNING = "warning"   # Suspicious — proceed with caution
    INFO = "info"         # Informational — FYI


@dataclass
class ValidationMessage:
    """Single validation message."""
    severity: Severity
    field: str
    message: str


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    valid: bool = True
    messages: List[ValidationMessage] = field(default_factory=list)
    _warnings: List[ValidationMessage] = field(default_factory=list)
    _infos: List[ValidationMessage] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.severity == Severity.WARNING]

    @property
    def infos(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.severity == Severity.INFO]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def add_error(self, field: str, message: str) -> None:
        self.valid = False
        self.messages.append(ValidationMessage(Severity.ERROR, field, message))

    def add_warning(self, field: str, message: str) -> None:
        self.messages.append(ValidationMessage(Severity.WARNING, field, message))

    def add_info(self, field: str, message: str) -> None:
        self.messages.append(ValidationMessage(Severity.INFO, field, message))

    def merge(self, other: ValidationResult) -> "ValidationResult":
        """Merge another result into this one."""
        self.valid = self.valid and other.valid
        self.messages.extend(other.messages)
        return self

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        errors = self.error_count
        warns = self.warning_count
        infos = len(self.infos)
        status = "PASS" if self.valid else "FAIL"
        return f"ValidationResult({status}, E={errors}, W={warns}, I={infos})"


# ══════════════════════════════════════════════════════════════════════════════
# Validator Functions
# ══════════════════════════════════════════════════════════════════════════════

def validate_required(value: Any, field: str, label: str = "") -> ValidationResult:
    """Validate that a value is not None or empty.

    Args:
        value: Value to check.
        field: Field name for error messages.
        label: Human-readable field label.

    Returns:
        ValidationResult.
    """
    result = ValidationResult()
    lbl = label or field
    if value is None:
        result.add_error(field, f"{lbl} is required.")
    elif isinstance(value, str) and not value.strip():
        result.add_error(field, f"{lbl} must not be empty.")
    elif isinstance(value, (list, tuple, dict)) and len(value) == 0:
        result.add_error(field, f"{lbl} must not be empty.")
    return result


def validate_positive(value: float | None, field: str, label: str = "",
                      include_zero: bool = False) -> ValidationResult:
    """Validate that a numeric value is positive.

    Args:
        value: Numeric value.
        field: Field name.
        label: Human-readable label.
        include_zero: If True, zero is allowed.

    Returns:
        ValidationResult.
    """
    result = ValidationResult()
    lbl = label or field
    if value is None:
        result.add_error(field, f"{lbl} is required.")
        return result
    if not include_zero and value <= EPSILON:
        result.add_error(field, f"{lbl} must be greater than zero (got {value:.4g}).")
    elif include_zero and value < -EPSILON:
        result.add_error(field, f"{lbl} must be non-negative (got {value:.4g}).")
    return result


def validate_range(value: float | None, field: str, vmin: float, vmax: float,
                   label: str = "", units: str = "") -> ValidationResult:
    """Validate that a numeric value is within [vmin, vmax].

    Args:
        value: Numeric value.
        field: Field name.
        vmin: Minimum allowed value.
        vmax: Maximum allowed value.
        label: Human-readable label.
        units: Unit string for error message.

    Returns:
        ValidationResult.
    """
    result = ValidationResult()
    lbl = label or field
    unit_str = f" {units}" if units else ""
    if value is None:
        result.add_error(field, f"{lbl} is required.")
        return result
    if value < vmin - EPSILON:
        result.add_error(field,
                         f"{lbl} must be ≥ {vmin}{unit_str} (got {value:.4g}{unit_str}).")
    if value > vmax + EPSILON:
        result.add_error(field,
                         f"{lbl} must be ≤ {vmax}{unit_str} (got {value:.4g}{unit_str}).")
    return result


def validate_percentage(value: float | None, field: str, label: str = "") -> ValidationResult:
    """Validate a percentage value (0-100) or fraction (0-1)."""
    result = ValidationResult()
    lbl = label or field
    if value is None:
        result.add_error(field, f"{lbl} is required.")
        return result
    if value < 0.0 or value > 100.0:
        result.add_error(field, f"{lbl} must be between 0 and 100 (got {value:.4g}).")
    return result


def validate_temperature(value: float | None, field: str, label: str = "",
                         vmin: float = 10.0, vmax: float = 3000.0) -> ValidationResult:
    """Validate a temperature value [K].

    Default range: 10 K to 3000 K (covers all plausible scenarios).

    Args:
        value: Temperature [K].
        field: Field name.
        label: Human-readable label.
        vmin: Minimum temperature [K].
        vmax: Maximum temperature [K].

    Returns:
        ValidationResult.
    """
    return validate_range(value, field, vmin, vmax, label, "K")


def validate_pressure(value: float | None, field: str, label: str = "",
                      vmin: float = 100.0, vmax: float = 300e5) -> ValidationResult:
    """Validate absolute pressure [Pa]. Range: 100 Pa to 300 bar."""
    return validate_range(value, field, vmin, vmax, label, "Pa")


def validate_gauge_pressure(value: float | None, field: str, label: str = "",
                             vmin: float = -P_ATM, vmax: float = 299e5) -> ValidationResult:
    """Validate gauge pressure [Pa]. Range: -1 atm to ~299 bar(g)."""
    return validate_range(value, field, vmin, vmax, label, "Pa(g)")


def validate_wind_speed(value: float | None, field: str, label: str = "") -> ValidationResult:
    """Validate wind speed [m/s]. Range: 0.5 to 25 m/s (dispersion models)."""
    return validate_range(value, field, MIN_WIND_SPEED, MAX_WIND_SPEED, label, "m/s")


def validate_stability_class(value: str | None, field: str, label: str = "") -> ValidationResult:
    """Validate Pasquill-Gifford stability class."""
    result = ValidationResult()
    lbl = label or field
    if value is None:
        result.add_error(field, f"{lbl} is required.")
        return result
    if value.upper() not in PG_STABILITY_CLASSES:
        result.add_error(field,
                         f"{lbl} must be one of {PG_STABILITY_CLASSES} (got {value!r}).")
    return result


def validate_surface_roughness(value: float | None, field: str, label: str = "") -> ValidationResult:
    """Validate surface roughness length [m]."""
    return validate_range(value, field, 0.0001, 10.0, label, "m")


def validate_release_rate(value: float | None, field: str, label: str = "",
                          max_rate: float = 1e6) -> ValidationResult:
    """Validate release rate [kg/s]."""
    return validate_range(value, field, 1e-9, max_rate, label, "kg/s")


def validate_release_duration(value: float | None, field: str, label: str = "") -> ValidationResult:
    """Validate release duration [s]."""
    return validate_range(value, field, 1.0, 86400.0 * 365, label, "s")


def validate_hole_size(value: float | None, field: str, label: str = "") -> ValidationResult:
    """Validate hole/orifice diameter [m]."""
    return validate_range(value, field, 0.001, 2.0, label, "m")


def validate_volume(value: float | None, field: str, label: str = "",
                    vmax: float = 1e6) -> ValidationResult:
    """Validate volume [m³]."""
    return validate_range(value, field, 0.001, vmax, label, "m³")


# ══════════════════════════════════════════════════════════════════════════════
# Cross-Field Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_temperature_pressure_consistency(
    T: float | None, P: float | None,
    boiling_point: float | None = None,
) -> ValidationResult:
    """Check if temperature and pressure are physically consistent.

    Warns if liquid is above boiling point at given pressure,
    or if temperature is below absolute zero.

    Args:
        T: Temperature [K].
        P: Pressure [Pa].
        boiling_point: Optional boiling point at 1 atm [K].

    Returns:
        ValidationResult.
    """
    result = ValidationResult()
    if T is not None and T < 0.0:
        result.add_error("temperature", f"Temperature cannot be negative (got {T:.2f} K).")
    if T is not None and P is not None and boiling_point is not None:
        # Simple check: at P_atm, if T >> Tb, substance should be vapor
        if P < 1.5 * P_ATM and T > boiling_point + 50:
            result.add_warning("temperature",
                               f"At {P / P_ATM:.2f} atm, T={T - T_0C:.1f}°C "
                               f"is well above the boiling point ({boiling_point - T_0C:.1f}°C). "
                               f"Ensure the phase is correct.")
    return result


def validate_dispersion_inputs(
    wind_speed: float | None,
    stability: str | None,
    release_rate: float | None,
    release_duration: float | None,
) -> ValidationResult:
    """Validate a complete set of dispersion model inputs.

    Checks for common mistakes and physically impossible combinations.

    Args:
        wind_speed: Wind speed [m/s].
        stability: Pasquill-Gifford class.
        release_rate: Release rate [kg/s].
        release_duration: Release duration [s] (None for instantaneous).

    Returns:
        ValidationResult.
    """
    result = ValidationResult()
    result.merge(validate_wind_speed(wind_speed, "wind_speed", "Wind speed"))
    result.merge(validate_stability_class(stability, "stability", "Stability class"))
    result.merge(validate_release_rate(release_rate, "release_rate", "Release rate"))
    if release_duration is not None:
        result.merge(validate_release_duration(release_duration, "release_duration",
                                                "Release duration"))

    # Consistency: strong stability (E, F) + high wind speed is unlikely
    if stability and wind_speed:
        st = stability.upper()
        if st in ("E", "F") and wind_speed > 5.0:
            result.add_warning("stability",
                               f"Stability class {st} with wind speed {wind_speed:.1f} m/s "
                               f"is unusual. Class {st} typically occurs with light winds (< 3 m/s).")

    # Consistency: very unstable (A) + high wind speed is unusual
    if st == "A" and wind_speed > 6.0:
        result.add_warning("stability",
                           f"Stability class A (very unstable) with wind speed {wind_speed:.1f} m/s "
                           f"is unusual and may produce unreliable dispersion results.")

    return result


def validate_fire_inputs(
    pool_diameter: float | None = None,
    release_rate: float | None = None,
    flame_length_estimate: float | None = None,
) -> ValidationResult:
    """Validate fire model inputs.

    Args:
        pool_diameter: Pool diameter [m].
        release_rate: Mass release rate [kg/s] (for jet fire).
        flame_length_estimate: Estimated flame length [m] for sanity check.

    Returns:
        ValidationResult.
    """
    result = ValidationResult()

    if pool_diameter is not None:
        result.merge(validate_positive(pool_diameter, "pool_diameter", "Pool diameter"))

    if release_rate is not None:
        result.merge(validate_release_rate(release_rate, "release_rate", "Release rate"))

    # Sanity: flame longer than 10x pool diameter is unusual
    if pool_diameter and flame_length_estimate:
        if flame_length_estimate > 10 * pool_diameter:
            result.add_warning("flame_length",
                               f"Estimated flame length ({flame_length_estimate:.1f} m) "
                               f"is > 10× pool diameter ({pool_diameter:.1f} m). "
                               f"Check input parameters.")

    return result


def validate_explosion_inputs(
    flammable_mass: float | None,
    heat_of_combustion: float | None,
) -> ValidationResult:
    """Validate explosion model inputs.

    Args:
        flammable_mass: Mass of flammable material [kg].
        heat_of_combustion: Heat of combustion [J/kg].

    Returns:
        ValidationResult.
    """
    result = ValidationResult()

    if flammable_mass is not None:
        result.merge(validate_positive(flammable_mass, "flammable_mass",
                                        "Flammable mass"))
        if flammable_mass > 1e7:  # 10,000 tons — unusually large
            result.add_warning("flammable_mass",
                               f"Flammable mass is very large ({flammable_mass:.0f} kg). "
                               f"Confirm this is intentional.")

    if heat_of_combustion is not None:
        result.merge(validate_positive(heat_of_combustion, "heat_of_combustion",
                                        "Heat of combustion"))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Sanity Checks
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check_release(
    orifice_diameter: float | None,
    pressure: float | None,
    temperature: float | None,
    molecular_weight: float | None,
    release_rate: float | None = None,
) -> ValidationResult:
    """Sanity check: does release rate roughly match orifice flow estimate?

    Uses simplified orifice equation (liquid): m_dot ≈ Cd·A·√(2·ρ·ΔP)
    to see if the user-entered release rate is in the right ballpark.

    Args:
        orifice_diameter: Orifice/hole diameter [m].
        pressure: Upstream (stagnation) pressure [Pa].
        temperature: Upstream temperature [K].
        molecular_weight: Molecular weight [g/mol].
        release_rate: User-entered release rate [kg/s].

    Returns:
        ValidationResult.
    """
    result = ValidationResult()

    if not all(x is not None for x in [orifice_diameter, pressure, temperature,
                                        molecular_weight, release_rate]):
        return result

    # Rough liquid orifice estimate
    Cd = 0.62
    A = math.pi * (orifice_diameter / 2.0) ** 2
    rho = (molecular_weight * 0.001 * pressure) / (8.314 * temperature)  # ideal gas
    dp = max(pressure - P_ATM, 1000.0)
    mdot_est = Cd * A * math.sqrt(2.0 * rho * dp)

    ratio = release_rate / mdot_est if mdot_est > EPSILON else float("inf")

    if ratio > 100 and release_rate > 1e-3:
        result.add_warning("release_rate",
                           f"Release rate ({release_rate:.4g} kg/s) is >> estimated "
                           f"orifice flow ({mdot_est:.4g} kg/s). "
                           f"Check hole size, pressure, or release rate.")

    if ratio < 0.01 and release_rate > 1e-3:
        result.add_warning("release_rate",
                           f"Release rate ({release_rate:.4g} kg/s) is << estimated "
                           f"orifice flow ({mdot_est:.4g} kg/s). "
                           f"Check inputs — release may be too small for given hole size.")

    return result


def sanity_check_concentration(
    concentration: float | None,
    erpg3: float | None = None,
    lfl: float | None = None,
) -> ValidationResult:
    """Sanity check: is concentration in a meaningful range?

    Args:
        concentration: Computed concentration [kg/m³ or ppm].
        erpg3: ERPG-3 value for toxic assessment.
        lfl: Lower flammability limit.

    Returns:
        ValidationResult.
    """
    result = ValidationResult()

    if concentration is not None and concentration > 1.0:  # 1 kg/m³ ≈ 1e6 ppm
        result.add_warning("concentration",
                           f"Concentration ({concentration:.4g}) is very high "
                           f"and likely not physically realistic. Check inputs.")

    if concentration is not None and erpg3 is not None:
        if concentration > 100 * erpg3:
            result.add_info("concentration",
                            f"Concentration exceeds 100× ERPG-3 ({erpg3:.4g}). "
                            f"Immediate life-threatening conditions.")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Batch Validator
# ══════════════════════════════════════════════════════════════════════════════

ValidatorFunc = Callable[..., ValidationResult]

# Registry of named validators for each field
_FIELD_VALIDATORS: Dict[str, List[Tuple[ValidatorFunc, tuple, dict]]] = {}


def register_validator(field: str, validator: ValidatorFunc,
                       args: tuple = (), kwargs: dict | None = None) -> None:
    """Register a validator for a specific field.

    Args:
        field: Field name.
        validator: Validator function.
        args: Positional arguments (after the value).
        kwargs: Keyword arguments.
    """
    if field not in _FIELD_VALIDATORS:
        _FIELD_VALIDATORS[field] = []
    _FIELD_VALIDATORS[field].append((validator, args, kwargs or {}))


def validate_field(field: str, value: Any) -> ValidationResult:
    """Run all registered validators for a field.

    Args:
        field: Field name.
        value: Field value.

    Returns:
        Merged ValidationResult.
    """
    result = ValidationResult()
    if field in _FIELD_VALIDATORS:
        for validator, args, kwargs in _FIELD_VALIDATORS[field]:
            result.merge(validator(value, *args, **kwargs))
    return result


def validate_all(fields: Dict[str, Any]) -> ValidationResult:
    """Run validators for all fields in a dictionary.

    Args:
        fields: Dict of {field_name: value}.

    Returns:
        Merged ValidationResult.
    """
    result = ValidationResult()
    for field, value in fields.items():
        result.merge(validate_field(field, value))
    return result
