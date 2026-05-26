"""
Rekarisk QRA — Failure Frequency Database.

Equipment failure frequency data and lookup functions for quantitative
risk assessment. Frequencies are expressed as events per year.

Data sources:
  - HSE UK Hydrocarbon Release Database (HCRD)
  - OGP Risk Assessment Data Directory (No. 434)
  - OREDA Offshore Reliability Data Handbook
  - TNO Purple Book (CPR 18E) — Guidelines for QRA
  - CCPS Guidelines for Chemical Process Quantitative Risk Analysis
  - API RP 581 Risk-Based Inspection Technology
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional, Union

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────

class FrequencyClass(str, Enum):
    """Qualitative classification of failure frequencies (per year)."""
    VERY_LOW = "very_low"     # < 1e-6
    LOW = "low"               # 1e-6 to < 1e-4
    MEDIUM = "medium"         # 1e-4 to < 1e-2
    HIGH = "high"             # ≥ 1e-2

    @classmethod
    def from_frequency(cls, freq: float) -> "FrequencyClass":
        if freq < 1e-6:
            return cls.VERY_LOW
        elif freq < 1e-4:
            return cls.LOW
        elif freq < 1e-2:
            return cls.MEDIUM
        else:
            return cls.HIGH


class LeakSize(str, Enum):
    """Standard leak size categories per HSE/OGP convention."""
    FULL_BORE = "full_bore"       # Catastrophic rupture (>150 mm / >6 inch)
    LARGE = "large"               # 50-150 mm (2-6 inch)
    MEDIUM = "medium"             # 25-50 mm (1-2 inch)
    SMALL = "small"               # <25 mm (<1 inch) / pinhole


class ComponentType(str, Enum):
    """Equipment component types for frequency lookup."""
    VESSEL = "vessel"
    PIPE = "pipe"
    FLANGE = "flange"
    VALVE = "valve"
    HOSE = "hose"
    LOADING_ARM = "loading_arm"
    GASKET = "gasket"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    HEAT_EXCHANGER = "heat_exchanger"
    FILTER = "filter"
    INSTRUMENT = "instrument"
    STORAGE_TANK = "storage_tank"
    SMALL_BORE = "small_bore"


class DataSource(str, Enum):
    """Reference data source for frequency values."""
    HSE_UK = "HSE UK HCRD"
    OGP = "OGP RADD 434"
    OREDA = "OREDA (SINTEF)"
    TNO = "TNO Purple Book CPR 18E"
    CCPS = "CCPS CPQRA"
    API_581 = "API RP 581"
    CUSTOM = "custom"


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FrequencyEntry:
    """Single frequency data point with metadata."""
    component_type: ComponentType
    leak_size: LeakSize
    frequency: float               # events per year
    source: DataSource = DataSource.HSE_UK
    notes: str = ""
    uncertainty_factor: float = 3.0  # 95% confidence range factor

    @property
    def lower_bound(self) -> float:
        return self.frequency / self.uncertainty_factor

    @property
    def upper_bound(self) -> float:
        return self.frequency * self.uncertainty_factor


@dataclass
class ModificationFactor:
    """Frequency modification factor for equipment-specific conditions."""
    name: str
    value: float                     # 1.0 = no change
    description: str = ""
    source: Optional[DataSource] = None


# ──────────────────────────────────────────────────────────────────────
# Default Modification Factors (API 581 / TNO / CCPS)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MODIFICATION_FACTORS: dict[str, ModificationFactor] = {
    "good_inspection": ModificationFactor(
        "good_inspection", 0.5,
        "Effective inspection & maintenance programme",
        DataSource.API_581,
    ),
    "poor_inspection": ModificationFactor(
        "poor_inspection", 2.0,
        "Inadequate inspection & maintenance",
        DataSource.API_581,
    ),
    "lined_equipment": ModificationFactor(
        "lined_equipment", 0.1,
        "Equipment with corrosion-resistant lining",
        DataSource.TNO,
    ),
    "corrosive_service": ModificationFactor(
        "corrosive_service", 5.0,
        "Highly corrosive process fluid",
        DataSource.OGP,
    ),
    "high_cycle": ModificationFactor(
        "high_cycle", 3.0,
        "Frequent pressure/temperature cycling (fatigue)",
        DataSource.OGP,
    ),
    "vibration_environment": ModificationFactor(
        "vibration_environment", 4.0,
        "Continuous vibration (e.g., near compressors)",
        DataSource.OGP,
    ),
    "buried_pipe": ModificationFactor(
        "buried_pipe", 0.1,
        "Buried pipe — no external impact, limited corrosion",
        DataSource.TNO,
    ),
    "aboveground_exposed": ModificationFactor(
        "aboveground_exposed", 1.2,
        "Aboveground pipe exposed to external impact",
        DataSource.TNO,
    ),
    "congested_area": ModificationFactor(
        "congested_area", 1.5,
        "Equipment in congested process area",
        DataSource.TNO,
    ),
    "remote_location": ModificationFactor(
        "remote_location", 0.5,
        "Equipment in remote, low-activity area",
        DataSource.TNO,
    ),
    "new_equipment": ModificationFactor(
        "new_equipment", 0.2,
        "Equipment < 5 years old with modern standards",
        DataSource.CCPS,
    ),
    "aged_equipment": ModificationFactor(
        "aged_equipment", 3.0,
        "Equipment > 20 years old, potential degradation",
        DataSource.OGP,
    ),
    "seamless_pipe": ModificationFactor(
        "seamless_pipe", 0.5,
        "Seamless pipe vs. seamed/ERW",
        DataSource.OGP,
    ),
    "double_containment": ModificationFactor(
        "double_containment", 0.01,
        "Double-walled or double-containment vessel/pipe",
        DataSource.TNO,
    ),
    "external_impact_barrier": ModificationFactor(
        "external_impact_barrier", 0.1,
        "Physical protection barrier (bollards, crash barriers)",
        DataSource.TNO,
    ),
    "cathodic_protection": ModificationFactor(
        "cathodic_protection", 0.5,
        "Effective cathodic protection for buried pipe",
        DataSource.API_581,
    ),
    "high_temperature": ModificationFactor(
        "high_temperature", 2.0,
        "Operation > 200°C, increased creep/fatigue",
        DataSource.API_581,
    ),
    "low_temperature": ModificationFactor(
        "low_temperature", 1.5,
        "Operation < -20°C, brittle fracture risk",
        DataSource.API_581,
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Default Frequency Database
# ──────────────────────────────────────────────────────────────────────

# Generic failure frequencies per year.
# Keyed by (ComponentType, LeakSize).
# "generic" leak size applies when no specific size is provided.
GENERIC_FREQUENCIES: dict[tuple[ComponentType, LeakSize], FrequencyEntry] = {
    # ── Vessels ──────────────────────────────────────────────────────
    (ComponentType.VESSEL, LeakSize.FULL_BORE): FrequencyEntry(
        ComponentType.VESSEL, LeakSize.FULL_BORE, 5e-6, DataSource.TNO,
        "Vessel catastrophic rupture (per vessel-year)",
    ),
    (ComponentType.VESSEL, LeakSize.LARGE): FrequencyEntry(
        ComponentType.VESSEL, LeakSize.LARGE, 1e-5, DataSource.TNO,
        "Vessel large leak 50-150 mm",
    ),
    (ComponentType.VESSEL, LeakSize.MEDIUM): FrequencyEntry(
        ComponentType.VESSEL, LeakSize.MEDIUM, 3e-5, DataSource.TNO,
        "Vessel medium leak 25-50 mm",
    ),
    (ComponentType.VESSEL, LeakSize.SMALL): FrequencyEntry(
        ComponentType.VESSEL, LeakSize.SMALL, 5e-5, DataSource.TNO,
        "Vessel small leak <25 mm",
    ),

    # ── Pipes (per km-year) ──────────────────────────────────────────
    (ComponentType.PIPE, LeakSize.FULL_BORE): FrequencyEntry(
        ComponentType.PIPE, LeakSize.FULL_BORE, 1e-5, DataSource.TNO,
        "Pipe full-bore rupture per km-year",
    ),
    (ComponentType.PIPE, LeakSize.LARGE): FrequencyEntry(
        ComponentType.PIPE, LeakSize.LARGE, 2e-5, DataSource.TNO,
        "Pipe large leak 50-150 mm per km-year",
    ),
    (ComponentType.PIPE, LeakSize.MEDIUM): FrequencyEntry(
        ComponentType.PIPE, LeakSize.MEDIUM, 5e-5, DataSource.TNO,
        "Pipe medium leak 25-50 mm per km-year",
    ),
    (ComponentType.PIPE, LeakSize.SMALL): FrequencyEntry(
        ComponentType.PIPE, LeakSize.SMALL, 5e-5, DataSource.TNO,
        "Pipe small leak <25 mm per km-year",
    ),

    # ── Flanges (per flange-year) ────────────────────────────────────
    (ComponentType.FLANGE, LeakSize.SMALL): FrequencyEntry(
        ComponentType.FLANGE, LeakSize.SMALL, 1e-4, DataSource.OGP,
        "Flange/gasket leak per flange-year",
    ),
    (ComponentType.FLANGE, LeakSize.FULL_BORE): FrequencyEntry(
        ComponentType.FLANGE, LeakSize.FULL_BORE, 1e-5, DataSource.OGP,
        "Flange catastrophic failure (guillotine)",
    ),

    # ── Valves (per valve-year) ──────────────────────────────────────
    (ComponentType.VALVE, LeakSize.SMALL): FrequencyEntry(
        ComponentType.VALVE, LeakSize.SMALL, 5e-5, DataSource.OGP,
        "Valve external leak (gland/stem) per valve-year",
    ),
    (ComponentType.VALVE, LeakSize.FULL_BORE): FrequencyEntry(
        ComponentType.VALVE, LeakSize.FULL_BORE, 1e-6, DataSource.OGP,
        "Valve body rupture per valve-year",
    ),

    # ── Hoses (per hose-year) ────────────────────────────────────────
    (ComponentType.HOSE, LeakSize.FULL_BORE): FrequencyEntry(
        ComponentType.HOSE, LeakSize.FULL_BORE, 1e-3, DataSource.OGP,
        "Loading/unloading hose catastrophic failure",
    ),
    (ComponentType.HOSE, LeakSize.SMALL): FrequencyEntry(
        ComponentType.HOSE, LeakSize.SMALL, 1e-3, DataSource.OGP,
        "Hose small leak per hose-year",
    ),

    # ── Loading Arms ─────────────────────────────────────────────────
    (ComponentType.LOADING_ARM, LeakSize.FULL_BORE): FrequencyEntry(
        ComponentType.LOADING_ARM, LeakSize.FULL_BORE, 5e-4, DataSource.OGP,
        "Loading arm catastrophic failure per arm-year",
    ),
    (ComponentType.LOADING_ARM, LeakSize.SMALL): FrequencyEntry(
        ComponentType.LOADING_ARM, LeakSize.SMALL, 5e-4, DataSource.OGP,
        "Loading arm leak per arm-year",
    ),

    # ── Gaskets (per gasket-year) ────────────────────────────────────
    (ComponentType.GASKET, LeakSize.SMALL): FrequencyEntry(
        ComponentType.GASKET, LeakSize.SMALL, 1e-4, DataSource.OGP,
        "Gasket leak per gasket-year (non-flange)",
    ),

    # ── Pumps (per pump-year) ────────────────────────────────────────
    (ComponentType.PUMP, LeakSize.FULL_BORE): FrequencyEntry(
        ComponentType.PUMP, LeakSize.FULL_BORE, 1e-4, DataSource.HSE_UK,
        "Pump catastrophic failure per pump-year",
    ),
    (ComponentType.PUMP, LeakSize.SMALL): FrequencyEntry(
        ComponentType.PUMP, LeakSize.SMALL, 1e-3, DataSource.HSE_UK,
        "Pump seal leak per pump-year",
    ),

    # ── Compressors (per compressor-year) ────────────────────────────
    (ComponentType.COMPRESSOR, LeakSize.FULL_BORE): FrequencyEntry(
        ComponentType.COMPRESSOR, LeakSize.FULL_BORE, 1e-4, DataSource.HSE_UK,
        "Compressor catastrophic failure per unit-year",
    ),
    (ComponentType.COMPRESSOR, LeakSize.SMALL): FrequencyEntry(
        ComponentType.COMPRESSOR, LeakSize.SMALL, 5e-4, DataSource.HSE_UK,
        "Compressor seal/gland leak per unit-year",
    ),

    # ── Heat Exchangers (per unit-year) ──────────────────────────────
    (ComponentType.HEAT_EXCHANGER, LeakSize.FULL_BORE): FrequencyEntry(
        ComponentType.HEAT_EXCHANGER, LeakSize.FULL_BORE, 1e-5, DataSource.OGP,
        "Heat exchanger catastrophic failure",
    ),
    (ComponentType.HEAT_EXCHANGER, LeakSize.SMALL): FrequencyEntry(
        ComponentType.HEAT_EXCHANGER, LeakSize.SMALL, 1e-4, DataSource.OGP,
        "Heat exchanger tube leak",
    ),

    # ── Filters (per unit-year) ──────────────────────────────────────
    (ComponentType.FILTER, LeakSize.SMALL): FrequencyEntry(
        ComponentType.FILTER, LeakSize.SMALL, 1e-4, DataSource.OGP,
        "Filter/strainer leak per unit-year",
    ),

    # ── Instruments (per instrument-year) ────────────────────────────
    (ComponentType.INSTRUMENT, LeakSize.SMALL): FrequencyEntry(
        ComponentType.INSTRUMENT, LeakSize.SMALL, 1e-4, DataSource.OGP,
        "Instrument connection leak (small bore)",
    ),

    # ── Storage Tanks (per tank-year) ────────────────────────────────
    (ComponentType.STORAGE_TANK, LeakSize.FULL_BORE): FrequencyEntry(
        ComponentType.STORAGE_TANK, LeakSize.FULL_BORE, 1e-5, DataSource.TNO,
        "Atmospheric storage tank catastrophic failure",
    ),
    (ComponentType.STORAGE_TANK, LeakSize.SMALL): FrequencyEntry(
        ComponentType.STORAGE_TANK, LeakSize.SMALL, 1e-4, DataSource.TNO,
        "Storage tank small leak per tank-year",
    ),

    # ── Small Bore Tubing (per fitting-year) ─────────────────────────
    (ComponentType.SMALL_BORE, LeakSize.SMALL): FrequencyEntry(
        ComponentType.SMALL_BORE, LeakSize.SMALL, 5e-4, DataSource.HSE_UK,
        "Small bore tubing/fitting failure (vibration fatigue)",
    ),
}


# HSE UK-specific release frequencies (alternative source)
# Based on HCRD data — events per year, onshore/offshore average
HSE_UK_ADDITIONAL: dict[str, dict[str, float]] = {
    "external_impact": {
        "onshore": 1e-5,
        "offshore": 5e-6,
        "description": "External impact (vehicle, dropped object)",
    },
    "corrosion": {
        "generic": 5e-5,
        "description": "Internal/external corrosion failure rate",
    },
    "overpressure": {
        "generic": 1e-5,
        "description": "Overpressure protection failure leading to loss of containment",
    },
    "operator_error": {
        "generic": 5e-5,
        "description": "Operator error causing loss of containment",
    },
    "fatigue": {
        "generic": 1e-5,
        "description": "Thermal/mechanical fatigue failure",
    },
    "erosion": {
        "generic": 1e-5,
        "description": "Erosion failure (sand, high velocity)",
    },
    "vibration_induced": {
        "generic": 5e-5,
        "description": "Vibration-induced fatigue (small bore)",
    },
}


# ──────────────────────────────────────────────────────────────────────
# Database class
# ──────────────────────────────────────────────────────────────────────

class FailureFrequencyDB:
    """Maintainable failure frequency database.

    Loads default generic frequencies and allows custom entries to
    be added or overridden for project-specific use.

    Examples
    --------
    >>> db = FailureFrequencyDB()
    >>> freq = db.lookup("vessel", "full_bore")
    >>> freq
    5e-06
    >>> db.add_custom("custom_component", "small", 1e-3, "User")
    """

    _instance: ClassVar[Optional["FailureFrequencyDB"]] = None

    def __init__(self) -> None:
        self._entries: dict[tuple[ComponentType, LeakSize], FrequencyEntry] = {}
        self._additional: dict[str, dict] = {}
        self._custom_modifiers: dict[str, ModificationFactor] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default generic frequencies and modifiers."""
        self._entries.update(copy.deepcopy(GENERIC_FREQUENCIES))
        self._additional.update(copy.deepcopy(HSE_UK_ADDITIONAL))
        self._custom_modifiers.update(copy.deepcopy(DEFAULT_MODIFICATION_FACTORS))

    # ── Lookup ───────────────────────────────────────────────────────

    def lookup(
        self,
        component_type: Union[str, ComponentType],
        leak_size: Union[str, LeakSize] = "small",
    ) -> float:
        """Look up generic failure frequency for a component type.

        Parameters
        ----------
        component_type : str or ComponentType
            Equipment type (e.g., "vessel", "pipe", "valve").
        leak_size : str or LeakSize
            Leak size category. Default is "small".

        Returns
        -------
        float
            Failure frequency in events per year.
            Returns 1e-7 if no entry found (conservative minimal default).
        """
        ct = ComponentType(component_type) if isinstance(component_type, str) else component_type
        ls = LeakSize(leak_size) if isinstance(leak_size, str) else leak_size

        # Exact match first
        key = (ct, ls)
        if key in self._entries:
            return self._entries[key].frequency

        # Try matching with SMALL as fallback
        fallback = (ct, LeakSize.SMALL)
        if fallback in self._entries:
            return self._entries[fallback].frequency

        return 1e-7  # Minimal default

    def lookup_entry(
        self,
        component_type: Union[str, ComponentType],
        leak_size: Union[str, LeakSize] = "small",
    ) -> Optional[FrequencyEntry]:
        """Look up full frequency entry (includes metadata)."""
        ct = ComponentType(component_type) if isinstance(component_type, str) else component_type
        ls = LeakSize(leak_size) if isinstance(leak_size, str) else leak_size

        key = (ct, ls)
        if key in self._entries:
            return self._entries[key]
        fallback = (ct, LeakSize.SMALL)
        if fallback in self._entries:
            return self._entries[fallback]
        return None

    def lookup_additional(self, failure_mode: str, context: str = "generic") -> float:
        """Look up additional/contextual failure frequencies.

        Parameters
        ----------
        failure_mode : str
            Mode name (e.g., "external_impact", "corrosion", "overpressure").
        context : str
            Context qualifier (e.g., "onshore", "offshore", "generic").

        Returns
        -------
        float
            Frequency in events per year.
        """
        entry = self._additional.get(failure_mode, {})
        if isinstance(entry, dict):
            return float(entry.get(context, entry.get("generic", 1e-7)))
        return float(entry) if isinstance(entry, (int, float)) else 1e-7

    # ── Modification ─────────────────────────────────────────────────

    def add_custom(
        self,
        component_type: str,
        leak_size: str,
        frequency: float,
        source: str = "custom",
        notes: str = "",
        uncertainty_factor: float = 3.0,
    ) -> None:
        """Add or override a custom frequency entry."""
        ct = ComponentType(component_type)
        ls = LeakSize(leak_size)
        entry = FrequencyEntry(
            ct, ls, frequency,
            source=DataSource.CUSTOM,
            notes=f"{source}: {notes}" if notes else f"{source}",
            uncertainty_factor=uncertainty_factor,
        )
        self._entries[(ct, ls)] = entry

    def add_modifier(
        self,
        name: str,
        value: float,
        description: str = "",
        source: Optional[DataSource] = None,
    ) -> None:
        """Add a custom modification factor."""
        self._custom_modifiers[name] = ModificationFactor(
            name, value, description, source,
        )

    def get_modifier(self, name: str) -> Optional[ModificationFactor]:
        """Retrieve a modification factor by name."""
        return self._custom_modifiers.get(name)

    def list_modifiers(self) -> dict[str, ModificationFactor]:
        """Return all known modification factors."""
        return dict(self._custom_modifiers)

    def list_components(self) -> list[ComponentType]:
        """Return list of registered component types."""
        return sorted(set(ct for ct, _ in self._entries.keys()),
                      key=lambda x: x.value)

    def list_entries(self) -> list[FrequencyEntry]:
        """Return all frequency entries."""
        return list(self._entries.values())

    # ── Singleton ────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "FailureFrequencyDB":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        """Reset to default data (clear custom entries)."""
        self._entries.clear()
        self._additional.clear()
        self._custom_modifiers.clear()
        self._load_defaults()


# ──────────────────────────────────────────────────────────────────────
# Module-level convenience functions
# ──────────────────────────────────────────────────────────────────────

def get_default_db() -> FailureFrequencyDB:
    """Return the singleton failure frequency database."""
    return FailureFrequencyDB.instance()


def lookup_frequency(
    component_type: Union[str, ComponentType],
    leak_size: Union[str, LeakSize] = "small",
    db: Optional[FailureFrequencyDB] = None,
) -> float:
    """Look up generic failure frequency.

    Parameters
    ----------
    component_type : str or ComponentType
        Equipment component type.
    leak_size : str or LeakSize
        Leak size category. Default "small".
    db : FailureFrequencyDB, optional
        Database instance. Uses singleton if None.

    Returns
    -------
    float
        Failure frequency (per year).

    Examples
    --------
    >>> lookup_frequency("vessel", "full_bore")
    5e-06
    >>> lookup_frequency("pipe", "small")
    5e-05
    >>> lookup_frequency("hose", "full_bore")
    0.001
    """
    database = db or get_default_db()
    return database.lookup(component_type, leak_size)


def combine_frequencies(frequencies: list[float]) -> float:
    """Combine multiple independent failure frequencies.

    For independent events (OR logic), frequencies sum directly:
        F_total = Σ F_i

    This is valid when frequencies are small (< 0.1/yr) and events
    are mutually exclusive.

    Parameters
    ----------
    frequencies : list of float
        Individual failure frequencies (per year).

    Returns
    -------
    float
        Combined frequency (per year).
    """
    return float(np.sum(frequencies))


def adjust_frequency(
    base_frequency: float,
    factors: Union[list[float], list[str]],
    db: Optional[FailureFrequencyDB] = None,
) -> float:
    """Apply modification factors to a base frequency.

    Parameters
    ----------
    base_frequency : float
        Base failure frequency (per year).
    factors : list of float or list of str
        Either numeric factor values, or names of predefined
        modification factors to look up.
    db : FailureFrequencyDB, optional
        Database for factor name lookup.

    Returns
    -------
    float
        Adjusted frequency (base × Π factors).

    Notes
    -----
    Per TNO Purple Book and API 581, modification factors are
    multiplicative: F_adj = F_base × FE_mod × FM_mod × ...

    Examples
    --------
    >>> adjust_frequency(5e-6, [0.5, 2.0])
    5e-06
    >>> adjust_frequency(5e-6, ["good_inspection", "lined_equipment"])
    2.5e-07
    """
    database = db or get_default_db()
    result = base_frequency
    for factor in factors:
        if isinstance(factor, str):
            mod = database.get_modifier(factor)
            if mod is not None:
                result *= mod.value
            else:
                # Unknown factor; treat as no-op
                result *= 1.0
        else:
            result *= float(factor)
    return result


def classify_frequency(frequency: float) -> FrequencyClass:
    """Classify a failure frequency into qualitative bins.

    Parameters
    ----------
    frequency : float
        Failure frequency (per year).

    Returns
    -------
    FrequencyClass
        VERY_LOW, LOW, MEDIUM, or HIGH.
    """
    return FrequencyClass.from_frequency(frequency)
