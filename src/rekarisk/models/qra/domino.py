"""
Rekarisk — Escalation & Domino Effect Analysis.

Models the propagation of fire and explosion effects between equipment
items, determining the likelihood and consequences of cascading failures
(domino effects) in process facilities.

Escalation vectors:
    1. Thermal radiation (pool fire, jet fire, BLEVE fireball → adjacent vessels)
    2. Overpressure blast wave (VCE → structural damage, vessel rupture)
    3. Fire impingement (direct flame contact → localised heating)

Escalation criteria based on:
    - CCPS (2000). Guidelines for Chemical Process Quantitative Risk Analysis.
    - Cozzani et al. (2005). Journal of Hazardous Materials, A107(3).
    - Reniers & Cozzani (2013). Domino Effects in the Process Industries.
    - NORSOK Z-013 — Risk and Emergency Preparedness Assessment.
    - API RP 752/753 — Management of Hazards Associated with Location of Buildings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
    from matplotlib.collections import PatchCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class EquipmentType(str, Enum):
    """Process equipment categories for escalation analysis."""
    ATMOSPHERIC_TANK = "atmospheric_tank"
    PRESSURE_VESSEL = "pressure_vessel"
    REACTOR = "reactor"
    HEAT_EXCHANGER = "heat_exchanger"
    PIPELINE = "pipeline"
    COLUMN = "column"
    SEPARATOR = "separator"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    FIN_FAN_COOLER = "fin_fan_cooler"
    STRUCTURE = "structure"
    BUND = "bund"


class EscalationVector(str, Enum):
    """Mechanism by which an event propagates to secondary equipment."""
    THERMAL_RADIATION = "thermal_radiation"
    OVERPRESSURE = "overpressure"
    FIRE_IMPINGEMENT = "fire_impingement"
    FRAGMENT = "fragment_projection"
    DEBRIS = "debris"


class DamageLevel(str, Enum):
    """Severity of damage to secondary equipment."""
    NONE = "none"
    MINOR = "minor"           # Cosmetic damage, no release
    MODERATE = "moderate"     # Small leak, localised damage
    MAJOR = "major"           # Significant release, structural damage
    CATASTROPHIC = "catastrophic"  # Complete failure, total inventory release


class SubstanceCategory(str, Enum):
    """Hazard category of contained substance."""
    FLAMMABLE_LIQUID = "flammable_liquid"
    FLAMMABLE_GAS = "flammable_gas"
    FLAMMABLE_LPG = "flammable_lpg"
    TOXIC = "toxic"
    REACTIVE = "reactive"
    INERT = "inert"


# ══════════════════════════════════════════════════════════════════════════════
# Equipment Data
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Equipment:
    """Process equipment item for escalation analysis.

    Attributes
    ----------
    id : str
        Unique identifier (e.g. "V-101", "TK-201").
    name : str
        Human-readable name.
    equipment_type : EquipmentType
        Category of equipment.
    substance : str
        Name of contained substance.
    substance_category : SubstanceCategory
        Hazard classification.
    inventory_kg : float
        Total mass of hazardous contents [kg].
    x : float
        X-coordinate on facility layout [m].
    y : float
        Y-coordinate on facility layout [m].
    elevation : float
        Height above grade [m].
    diameter : float
        Equipment diameter [m].
    height : float
        Equipment height/length [m].
    operating_pressure : float
        Normal operating pressure [bar gauge].
    design_pressure : float
        Maximum design pressure [bar gauge].
    is_insulated : bool
        Whether equipment has fireproof insulation.
    has_deluge : bool
        Whether equipment is protected by water deluge/spray.
    has_bund : bool
        Whether equipment is within a bunded area.
    bund_radius : float
        Bund radius [m] (if applicable).
    ttf_base : float
        Base time to failure under fire exposure [min].
        Default values by equipment type per Cozzani et al.
    """
    id: str
    name: str
    equipment_type: EquipmentType
    substance: str
    substance_category: SubstanceCategory
    inventory_kg: float
    x: float = 0.0
    y: float = 0.0
    elevation: float = 0.0
    diameter: float = 2.0
    height: float = 5.0
    operating_pressure: float = 1.0
    design_pressure: float = 2.0
    is_insulated: bool = False
    has_deluge: bool = False
    has_bund: bool = False
    bund_radius: float = 0.0
    ttf_base: float = 0.0

    def __post_init__(self):
        if self.ttf_base <= 0:
            self.ttf_base = _default_ttf(self.equipment_type, self.is_insulated)

    @property
    def surface_area(self) -> float:
        """Estimated wetted surface area [m²]."""
        if self.equipment_type == EquipmentType.ATMOSPHERIC_TANK:
            # Cylindrical tank: π×D×H + 2×π×D²/4
            return math.pi * self.diameter * self.height + math.pi * self.diameter**2 / 2
        elif self.equipment_type == EquipmentType.PIPELINE:
            return math.pi * self.diameter * self.height  # length = height field
        else:
            # Vessel: cylinder + 2 hemispherical heads
            return math.pi * self.diameter * self.height + math.pi * self.diameter**2

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


def _default_ttf(eq_type: EquipmentType, insulated: bool) -> float:
    """Default time to failure [min] based on equipment type.

    Based on Cozzani et al. (2005) and API RP 521.
    """
    base = {
        EquipmentType.ATMOSPHERIC_TANK: 15.0,
        EquipmentType.PRESSURE_VESSEL: 20.0,
        EquipmentType.REACTOR: 25.0,
        EquipmentType.HEAT_EXCHANGER: 10.0,
        EquipmentType.PIPELINE: 8.0,
        EquipmentType.COLUMN: 25.0,
        EquipmentType.SEPARATOR: 15.0,
        EquipmentType.PUMP: 5.0,
        EquipmentType.COMPRESSOR: 5.0,
        EquipmentType.FIN_FAN_COOLER: 8.0,
        EquipmentType.STRUCTURE: 30.0,
    }.get(eq_type, 15.0)

    if insulated:
        base *= 3.0  # Insulation significantly delays failure

    return base


# ══════════════════════════════════════════════════════════════════════════════
# Escalation Thresholds
# ══════════════════════════════════════════════════════════════════════════════

# Thermal radiation escalation thresholds [kW/m²] per Cozzani et al. (2005)
# and CCPS Guidelines for CPQRA
THERMAL_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # Equipment type → {minor, major, catastrophic} thresholds in kW/m²
    "atmospheric_tank": {"minor": 8.0, "major": 15.0, "catastrophic": 25.0},
    "pressure_vessel": {"minor": 15.0, "major": 25.0, "catastrophic": 45.0},
    "reactor": {"minor": 15.0, "major": 25.0, "catastrophic": 45.0},
    "pipeline": {"minor": 10.0, "major": 20.0, "catastrophic": 30.0},
    "column": {"minor": 15.0, "major": 25.0, "catastrophic": 45.0},
    "separator": {"minor": 12.0, "major": 20.0, "catastrophic": 35.0},
    "heat_exchanger": {"minor": 12.0, "major": 20.0, "catastrophic": 35.0},
    "pump": {"minor": 8.0, "major": 15.0, "catastrophic": 25.0},
    "compressor": {"minor": 8.0, "major": 15.0, "catastrophic": 25.0},
    "fin_fan_cooler": {"minor": 8.0, "major": 15.0, "catastrophic": 25.0},
    "structure": {"minor": 12.0, "major": 25.0, "catastrophic": 50.0},
    "default": {"minor": 12.0, "major": 20.0, "catastrophic": 35.0},
}

# Overpressure escalation thresholds [kPa] per CCPS and API RP 752
OVERPRESSURE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "atmospheric_tank": {"minor": 3.5, "major": 14.0, "catastrophic": 35.0},
    "pressure_vessel": {"minor": 14.0, "major": 35.0, "catastrophic": 70.0},
    "reactor": {"minor": 14.0, "major": 35.0, "catastrophic": 70.0},
    "pipeline": {"minor": 7.0, "major": 20.0, "catastrophic": 50.0},
    "column": {"minor": 14.0, "major": 35.0, "catastrophic": 70.0},
    "separator": {"minor": 10.0, "major": 25.0, "catastrophic": 55.0},
    "structure": {"minor": 3.5, "major": 14.0, "catastrophic": 35.0},
    "default": {"minor": 7.0, "major": 20.0, "catastrophic": 45.0},
}

# Fire impingement time to failure thresholds [min]
IMPINGEMENT_TTF: Dict[str, float] = {
    "atmospheric_tank": 5.0,
    "pressure_vessel": 8.0,
    "pipeline": 3.0,
    "default": 5.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# Escalation Result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EscalationLink:
    """Single escalation link from source to target equipment.

    Attributes
    ----------
    source_id : str
        ID of the primary (failed) equipment.
    target_id : str
        ID of the secondary (potentially affected) equipment.
    distance_m : float
        Distance between equipment centres [m].
    vector : EscalationVector
        Mechanism of escalation.
    intensity : float
        Incident intensity at target (kW/m² or kPa).
    damage_level : DamageLevel
        Predicted damage severity.
    escalation_prob : float
        Probability of escalation (0–1).
    ttf_minutes : float
        Estimated time to failure of target [min].
    mitigation_factor : float
        Reduction factor from insulation/deluge (0–1).
    inventory_released_kg : float
        Mass released if target fails (fraction based on damage level).
    """
    source_id: str
    target_id: str
    distance_m: float
    vector: EscalationVector
    intensity: float
    damage_level: DamageLevel
    escalation_prob: float
    ttf_minutes: float
    mitigation_factor: float = 1.0
    inventory_released_kg: float = 0.0


@dataclass
class DominoScenario:
    """A complete domino scenario — cascade of escalating failures.

    Attributes
    ----------
    chain : list[str]
        Ordered sequence of equipment IDs that fail.
    links : list[EscalationLink]
        Escalation links between successive failures.
    total_frequency : float
        Frequency of the complete domino scenario [/yr].
    total_inventory_released : float
        Sum of all released inventories [kg].
    total_fatalities : int
        Estimated fatalities (if population data available).
    max_order : int
        Maximum escalation order (1 = primary only, 2 = 1 domino, etc.).
    """
    chain: List[str]
    links: List[EscalationLink]
    total_frequency: float
    total_inventory_released: float
    total_fatalities: int = 0
    max_order: int = 1


@dataclass
class DominoAnalysisResult:
    """Complete result of a domino effect analysis.

    Attributes
    ----------
    primary_event : str
        Equipment ID of the initiating event.
    primary_scenario : str
        Description of the primary event.
    primary_frequency : float
        Frequency of the primary event [/yr].
    escalation_links : list[EscalationLink]
        All identified escalation links from the primary event.
    domino_scenarios : list[DominoScenario]
        Complete domino scenarios (cascading chains).
    equipment_list : list[Equipment]
        All equipment in the analysis.
    max_escalation_distance_m : float
        Maximum distance at which escalation is possible.
    summary : dict
        Summary statistics.
    """
    primary_event: str
    primary_scenario: str
    primary_frequency: float
    escalation_links: List[EscalationLink]
    domino_scenarios: List[DominoScenario]
    equipment_list: List[Equipment]
    max_escalation_distance_m: float
    summary: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Core Calculation Functions
# ══════════════════════════════════════════════════════════════════════════════

def calculate_distance(eq1: Equipment, eq2: Equipment) -> float:
    """Calculate distance between two equipment items [m]."""
    return math.sqrt((eq1.x - eq2.x)**2 + (eq1.y - eq2.y)**2)


def thermal_radiation_at_distance(
    source_power_kw: float,
    distance_m: float,
    *,
    source_height: float = 0.0,
    target_height: float = 0.0,
    atmospheric_transmissivity: float = None,
) -> float:
    """Calculate thermal radiation at a given distance using point source model.

    Parameters
    ----------
    source_power_kw : float
        Total heat release rate of the fire [kW].
    distance_m : float
        Distance from source centre to target [m].
    source_height : float
        Height of the flame centre [m].
    target_height : float
        Height of the target [m].
    atmospheric_transmissivity : float, optional
        Atmospheric transmissivity (0–1). If None, calculated from distance.

    Returns
    -------
    float
        Incident thermal radiation [kW/m²].
    """
    # Effective distance including height difference
    dh = abs(source_height - target_height)
    r_eff = math.sqrt(distance_m**2 + dh**2)

    if r_eff < 1.0:
        r_eff = 1.0

    # Atmospheric transmissivity (Wayne's correlation, simplified)
    if atmospheric_transmissivity is None:
        # τ ≈ 1 - 0.058 * d^(0.9) for typical humidity
        atmospheric_transmissivity = max(0.3, 1.0 - 0.0008 * r_eff)

    # Point source model: Q = τ × Q_total / (4π × r²)
    q = atmospheric_transmissivity * source_power_kw / (4 * math.pi * r_eff**2)

    return q


def overpressure_at_distance_tnt(
    tnt_mass_kg: float,
    distance_m: float,
) -> float:
    """Calculate peak overpressure using TNT scaled distance model.

    Uses the Kingery-Bulmash relationship (simplified Hopkinson-Cranz).

    Parameters
    ----------
    tnt_mass_kg : float
        TNT equivalent mass [kg].
    distance_m : float
        Distance from blast centre [m].

    Returns
    -------
    float
        Peak side-on overpressure [kPa].
    """
    if tnt_mass_kg <= 0 or distance_m <= 0:
        return 0.0

    # Scaled distance Z = R / W^(1/3)
    Z = distance_m / (tnt_mass_kg ** (1.0 / 3.0))

    if Z < 0.1:
        Z = 0.1  # Near-field limit

    # Kingery-Bulmash simplified (overpressure in kPa)
    # ΔP/P0 = 0.84/Z + 2.7/Z² + 7.1/Z³ (for Z > 0.5)
    P0 = 101.325  # kPa
    delta_P = P0 * (0.84 / Z + 2.7 / Z**2 + 7.1 / Z**3)

    return max(delta_P, 0.0)


def assess_damage_level(
    intensity: float,
    vector: EscalationVector,
    equipment_type: EquipmentType,
) -> DamageLevel:
    """Assess damage level based on intensity, vector, and equipment type.

    Parameters
    ----------
    intensity : float
        Incident intensity (kW/m² for thermal, kPa for overpressure).
    vector : EscalationVector
        Escalation mechanism.
    equipment_type : EquipmentType
        Type of target equipment.

    Returns
    -------
    DamageLevel
        Predicted damage severity.
    """
    type_key = equipment_type.value

    if vector == EscalationVector.THERMAL_RADIATION:
        thresholds = THERMAL_THRESHOLDS.get(type_key, THERMAL_THRESHOLDS["default"])
    elif vector == EscalationVector.OVERPRESSURE:
        thresholds = OVERPRESSURE_THRESHOLDS.get(type_key, OVERPRESSURE_THRESHOLDS["default"])
    elif vector == EscalationVector.FIRE_IMPINGEMENT:
        # Fire impingement is always at least moderate
        if intensity > 0:
            return DamageLevel.MAJOR
        return DamageLevel.NONE
    else:
        thresholds = THERMAL_THRESHOLDS["default"]

    if intensity >= thresholds["catastrophic"]:
        return DamageLevel.CATASTROPHIC
    elif intensity >= thresholds["major"]:
        return DamageLevel.MAJOR
    elif intensity >= thresholds["minor"]:
        return DamageLevel.MODERATE
    else:
        return DamageLevel.NONE


def calculate_escalation_probability(
    damage_level: DamageLevel,
    equipment_type: EquipmentType,
    ttf_minutes: float,
    *,
    has_deluge: bool = False,
    has_insulation: bool = False,
    response_time_min: float = 10.0,
) -> float:
    """Calculate probability of escalation given damage level and mitigations.

    Based on probit approach from Cozzani et al. (2005).

    Parameters
    ----------
    damage_level : DamageLevel
        Assessed damage level.
    equipment_type : EquipmentType
        Type of target equipment.
    ttf_minutes : float
        Time to failure under current conditions [min].
    has_deluge : bool
        Whether water deluge is active.
    has_insulation : bool
        Whether equipment has fireproof insulation.
    response_time_min : float
        Time for emergency response [min].

    Returns
    -------
    float
        Probability of escalation (0–1).
    """
    if damage_level == DamageLevel.NONE:
        return 0.0

    # Base probabilities by damage level
    base_prob = {
        DamageLevel.MINOR: 0.05,
        DamageLevel.MODERATE: 0.20,
        DamageLevel.MAJOR: 0.50,
        DamageLevel.CATASTROPHIC: 0.90,
    }.get(damage_level, 0.0)

    # Time factor: if TTF > response time, lower probability (successful intervention)
    if ttf_minutes > response_time_min:
        time_factor = max(0.1, 1.0 - 0.5 * math.log(ttf_minutes / response_time_min))
    else:
        time_factor = 1.0

    # Mitigation factors
    deluge_factor = 0.4 if has_deluge else 1.0  # Deluge reduces by 60%
    insulation_factor = 0.3 if has_insulation else 1.0  # Insulation reduces by 70%

    prob = base_prob * time_factor * deluge_factor * insulation_factor
    return min(max(prob, 0.0), 1.0)


def calculate_ttf(
    equipment: Equipment,
    intensity: float,
    vector: EscalationVector,
) -> float:
    """Calculate time to failure for equipment under given conditions.

    Parameters
    ----------
    equipment : Equipment
        Target equipment.
    intensity : float
        Incident intensity (kW/m² or kPa).
    vector : EscalationVector
        Escalation mechanism.

    Returns
    -------
    float
        Estimated time to failure [min]. Returns infinity if no failure expected.
    """
    if vector == EscalationVector.THERMAL_RADIATION:
        # API 521 / Cozzani approach:
        # TTF inversely proportional to heat flux
        # ttf = C / Q^1.15 (simplified)
        # where C depends on equipment type
        if intensity <= 0:
            return float('inf')

        Q_ref = 25.0  # Reference heat flux [kW/m²]
        C = equipment.ttf_base * (Q_ref ** 1.15)

        ttf = C / (intensity ** 1.15)

    elif vector == EscalationVector.OVERPRESSURE:
        # If overpressure exceeds catastrophic threshold → immediate failure
        type_key = equipment.equipment_type.value
        thresholds = OVERPRESSURE_THRESHOLDS.get(type_key, OVERPRESSURE_THRESHOLDS["default"])

        if intensity >= thresholds["catastrophic"]:
            ttf = 0.5  # Nearly instantaneous
        elif intensity >= thresholds["major"]:
            ttf = 2.0  # Very fast
        elif intensity >= thresholds["minor"]:
            ttf = 10.0  # Minutes
        else:
            return float('inf')

    elif vector == EscalationVector.FIRE_IMPINGEMENT:
        # Direct flame contact → much shorter TTF
        type_key = equipment.equipment_type.value
        base_ttf = IMPINGEMENT_TTF.get(type_key, IMPINGEMENT_TTF["default"])
        ttf = base_ttf

    else:
        return float('inf')

    # Adjust for insulation
    if equipment.is_insulated:
        ttf *= 3.0

    # Adjust for deluge
    if equipment.has_deluge:
        ttf *= 2.0

    return ttf


# ══════════════════════════════════════════════════════════════════════════════
# Main Domino Analysis
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PrimaryEvent:
    """Definition of the initiating event for escalation analysis.

    Attributes
    ----------
    equipment_id : str
        ID of the equipment where the event originates.
    event_type : str
        Type of event: "pool_fire", "jet_fire", "bleve", "vce", "flash_fire".
    frequency : float
        Frequency of the event [/yr].
    thermal_power_kw : float
        Total heat release rate [kW] (for fire events).
    tnt_mass_kg : float
        TNT equivalent mass [kg] (for explosion events).
    fireball_radius_m : float
        BLEVE fireball radius [m].
    source_height_m : float
        Height of the flame centre [m].
    pool_radius_m : float
        Pool fire radius [m] (for fire impingement check).
    """
    equipment_id: str
    event_type: str
    frequency: float
    thermal_power_kw: float = 0.0
    tnt_mass_kg: float = 0.0
    fireball_radius_m: float = 0.0
    source_height_m: float = 0.0
    pool_radius_m: float = 0.0


def run_domino_analysis(
    primary_event: PrimaryEvent,
    equipment_list: List[Equipment],
    *,
    max_escalation_order: int = 3,
    response_time_min: float = 10.0,
    include_thermal: bool = True,
    include_overpressure: bool = True,
    include_impingement: bool = True,
) -> DominoAnalysisResult:
    """Run a complete domino effect escalation analysis.

    Evaluates all possible escalation paths from a primary event through
    the equipment layout, up to the specified escalation order.

    Parameters
    ----------
    primary_event : PrimaryEvent
        The initiating event.
    equipment_list : list[Equipment]
        All equipment items in the facility layout.
    max_escalation_order : int
        Maximum number of cascade levels (1 = primary only, 2 = one domino, etc.).
    response_time_min : float
        Emergency response time [min].
    include_thermal : bool
        Evaluate thermal radiation escalation.
    include_overpressure : bool
        Evaluate overpressure escalation.
    include_impingement : bool
        Evaluate fire impingement escalation.

    Returns
    -------
    DominoAnalysisResult
        Complete analysis results.
    """
    eq_map = {eq.id: eq for eq in equipment_list}
    all_links: List[EscalationLink] = []
    domino_scenarios: List[DominoScenario] = []

    # Find primary equipment
    primary_eq = eq_map.get(primary_event.equipment_id)
    if primary_eq is None:
        raise ValueError(f"Primary equipment '{primary_event.equipment_id}' not found in equipment list")

    # ── Level 1: Primary → All others ──
    level1_links = _evaluate_escalation(
        primary_event, equipment_list, eq_map,
        include_thermal=include_thermal,
        include_overpressure=include_overpressure,
        include_impingement=include_impingement,
        response_time_min=response_time_min,
    )
    all_links.extend(level1_links)

    # Build domino scenarios from level 1
    for link in level1_links:
        if link.damage_level != DamageLevel.NONE:
            target_eq = eq_map[link.target_id]
            released = _inventory_released(target_eq, link.damage_level)
            scenario = DominoScenario(
                chain=[primary_event.equipment_id, link.target_id],
                links=[link],
                total_frequency=primary_event.frequency * link.escalation_prob,
                total_inventory_released=primary_eq.inventory_kg + released,
                max_order=2,
            )
            domino_scenarios.append(scenario)

    # ── Level 2+: Cascading escalation ──
    if max_escalation_order >= 3:
        for scenario in list(domino_scenarios):  # Copy list since we'll modify
            last_eq_id = scenario.chain[-1]
            last_eq = eq_map[last_eq_id]

            # Assume secondary event is similar type (conservative)
            # For fire → use thermal power from released inventory
            secondary_thermal = _estimate_thermal_power(last_eq)
            secondary_tnt = _estimate_tnt_equivalent(last_eq)

            secondary_event = PrimaryEvent(
                equipment_id=last_eq_id,
                event_type=primary_event.event_type,
                frequency=scenario.total_frequency,
                thermal_power_kw=secondary_thermal,
                tnt_mass_kg=secondary_tnt,
                source_height_m=last_eq.elevation + last_eq.height / 2,
                pool_radius_m=last_eq.diameter * 2,
            )

            level2_links = _evaluate_escalation(
                secondary_event, equipment_list, eq_map,
                include_thermal=include_thermal,
                include_overpressure=include_overpressure,
                include_impingement=include_impingement,
                response_time_min=response_time_min,
                exclude_ids=set(scenario.chain),
            )

            for link in level2_links:
                if link.damage_level != DamageLevel.NONE:
                    target_eq = eq_map[link.target_id]
                    released = _inventory_released(target_eq, link.damage_level)
                    new_chain = scenario.chain + [link.target_id]
                    new_freq = scenario.total_frequency * link.escalation_prob

                    domino_scenarios.append(DominoScenario(
                        chain=new_chain,
                        links=scenario.links + [link],
                        total_frequency=new_freq,
                        total_inventory_released=scenario.total_inventory_released + released,
                        max_order=len(new_chain),
                    ))
                    all_links.append(link)

    # Calculate max escalation distance
    max_dist = 0.0
    for link in all_links:
        if link.damage_level != DamageLevel.NONE:
            max_dist = max(max_dist, link.distance_m)

    # Summary statistics
    significant_links = [l for l in all_links if l.damage_level != DamageLevel.NONE]
    summary = {
        "total_equipment": len(equipment_list),
        "total_escalation_links": len(all_links),
        "significant_links": len(significant_links),
        "domino_scenarios": len(domino_scenarios),
        "max_escalation_distance_m": max_dist,
        "max_cascade_order": max((s.max_order for s in domino_scenarios), default=1),
        "total_dominofrequency": sum(s.total_frequency for s in domino_scenarios),
        "worst_case_inventory_kg": max(
            (s.total_inventory_released for s in domino_scenarios), default=0
        ),
        "equipment_at_risk": list(set(
            l.target_id for l in significant_links
        )),
    }

    return DominoAnalysisResult(
        primary_event=primary_event.equipment_id,
        primary_scenario=f"{primary_event.event_type} at {primary_event.equipment_id}",
        primary_frequency=primary_event.frequency,
        escalation_links=all_links,
        domino_scenarios=domino_scenarios,
        equipment_list=equipment_list,
        max_escalation_distance_m=max_dist,
        summary=summary,
    )


def _evaluate_escalation(
    primary_event: PrimaryEvent,
    equipment_list: List[Equipment],
    eq_map: Dict[str, Equipment],
    *,
    include_thermal: bool,
    include_overpressure: bool,
    include_impingement: bool,
    response_time_min: float,
    exclude_ids: set = None,
) -> List[EscalationLink]:
    """Evaluate escalation from a single event to all equipment."""
    links = []
    exclude_ids = exclude_ids or {primary_event.equipment_id}

    for eq in equipment_list:
        if eq.id in exclude_ids:
            continue

        source_eq = eq_map[primary_event.equipment_id]
        dist = calculate_distance(source_eq, eq)

        # Skip if very far away
        if dist > 1000:
            continue

        mitigation = 1.0
        if eq.is_insulated:
            mitigation *= 0.3
        if eq.has_deluge:
            mitigation *= 0.4

        best_link = None

        # Thermal radiation check
        if include_thermal and primary_event.thermal_power_kw > 0:
            q = thermal_radiation_at_distance(
                primary_event.thermal_power_kw,
                dist,
                source_height=primary_event.source_height_m,
                target_height=eq.elevation,
            )
            damage = assess_damage_level(q, EscalationVector.THERMAL_RADIATION, eq.equipment_type)
            if damage != DamageLevel.NONE:
                ttf = calculate_ttf(eq, q, EscalationVector.THERMAL_RADIATION)
                prob = calculate_escalation_probability(
                    damage, eq.equipment_type, ttf,
                    has_deluge=eq.has_deluge, has_insulation=eq.is_insulated,
                    response_time_min=response_time_min,
                )
                released = _inventory_released(eq, damage)
                link = EscalationLink(
                    source_id=primary_event.equipment_id,
                    target_id=eq.id,
                    distance_m=dist,
                    vector=EscalationVector.THERMAL_RADIATION,
                    intensity=q,
                    damage_level=damage,
                    escalation_prob=prob,
                    ttf_minutes=ttf,
                    mitigation_factor=mitigation,
                    inventory_released_kg=released,
                )
                if best_link is None or link.escalation_prob > best_link.escalation_prob:
                    best_link = link

        # Overpressure check
        if include_overpressure and primary_event.tnt_mass_kg > 0:
            dP = overpressure_at_distance_tnt(primary_event.tnt_mass_kg, dist)
            damage = assess_damage_level(dP, EscalationVector.OVERPRESSURE, eq.equipment_type)
            if damage != DamageLevel.NONE:
                ttf = calculate_ttf(eq, dP, EscalationVector.OVERPRESSURE)
                prob = calculate_escalation_probability(
                    damage, eq.equipment_type, ttf,
                    has_deluge=eq.has_deluge, has_insulation=eq.is_insulated,
                    response_time_min=response_time_min,
                )
                released = _inventory_released(eq, damage)
                link = EscalationLink(
                    source_id=primary_event.equipment_id,
                    target_id=eq.id,
                    distance_m=dist,
                    vector=EscalationVector.OVERPRESSURE,
                    intensity=dP,
                    damage_level=damage,
                    escalation_prob=prob,
                    ttf_minutes=ttf,
                    mitigation_factor=mitigation,
                    inventory_released_kg=released,
                )
                if best_link is None or link.escalation_prob > best_link.escalation_prob:
                    best_link = link

        # Fire impingement check
        if include_impingement and primary_event.pool_radius_m > 0:
            if dist <= primary_event.pool_radius_m + eq.diameter / 2:
                damage = DamageLevel.MAJOR
                ttf = calculate_ttf(eq, 1.0, EscalationVector.FIRE_IMPINGEMENT)
                prob = calculate_escalation_probability(
                    damage, eq.equipment_type, ttf,
                    has_deluge=eq.has_deluge, has_insulation=eq.is_insulated,
                    response_time_min=response_time_min,
                )
                released = _inventory_released(eq, damage)
                link = EscalationLink(
                    source_id=primary_event.equipment_id,
                    target_id=eq.id,
                    distance_m=dist,
                    vector=EscalationVector.FIRE_IMPINGEMENT,
                    intensity=1.0,  # Impingement = direct contact
                    damage_level=damage,
                    escalation_prob=prob,
                    ttf_minutes=ttf,
                    mitigation_factor=mitigation,
                    inventory_released_kg=released,
                )
                if best_link is None or link.escalation_prob > best_link.escalation_prob:
                    best_link = link

        if best_link is not None:
            links.append(best_link)

    return links


def _inventory_released(equipment: Equipment, damage: DamageLevel) -> float:
    """Estimate fraction of inventory released based on damage level."""
    fraction = {
        DamageLevel.NONE: 0.0,
        DamageLevel.MINOR: 0.02,
        DamageLevel.MODERATE: 0.10,
        DamageLevel.MAJOR: 0.30,
        DamageLevel.CATASTROPHIC: 1.00,
    }.get(damage, 0.0)
    return equipment.inventory_kg * fraction


def _estimate_thermal_power(equipment: Equipment) -> float:
    """Estimate thermal power from equipment failure [kW]."""
    # Conservative: 30% of inventory burns in 10 minutes
    if equipment.substance_category in (
        SubstanceCategory.FLAMMABLE_LIQUID,
        SubstanceCategory.FLAMMABLE_GAS,
        SubstanceCategory.FLAMMABLE_LPG,
    ):
        hc = 44e6  # J/kg generic hydrocarbon
        mass_burned = equipment.inventory_kg * 0.3
        duration = 600  # 10 min in seconds
        return mass_burned * hc / duration / 1000  # kW
    return 0.0


def _estimate_tnt_equivalent(equipment: Equipment) -> float:
    """Estimate TNT equivalent mass for explosion from equipment failure [kg]."""
    if equipment.substance_category in (
        SubstanceCategory.FLAMMABLE_GAS,
        SubstanceCategory.FLAMMABLE_LPG,
    ):
        hc = 50e6  # J/kg
        mass_cloud = equipment.inventory_kg * 0.1  # 10% in cloud
        efficiency = 0.05
        tnt_eq = mass_cloud * hc * efficiency / 4.184e6  # kg TNT
        return tnt_eq
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_domino_map(
    result: DominoAnalysisResult,
    *,
    save_path: str = None,
    dpi: int = 150,
    show_labels: bool = True,
) -> Optional["plt.Figure"]:
    """Generate a facility layout map showing domino escalation paths.

    Parameters
    ----------
    result : DominoAnalysisResult
        Analysis results to visualize.
    save_path : str, optional
        File path to save the figure.
    dpi : int
        Image resolution.
    show_labels : bool
        Whether to show equipment ID labels.

    Returns
    -------
    matplotlib Figure or None
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required for plotting")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Color palette
    eq_colors = {
        EquipmentType.ATMOSPHERIC_TANK: "#3498db",
        EquipmentType.PRESSURE_VESSEL: "#e74c3c",
        EquipmentType.REACTOR: "#9b59b6",
        EquipmentType.COLUMN: "#1abc9c",
        EquipmentType.SEPARATOR: "#f39c12",
        EquipmentType.HEAT_EXCHANGER: "#2ecc71",
        EquipmentType.PIPELINE: "#95a5a6",
        EquipmentType.PUMP: "#e67e22",
        EquipmentType.COMPRESSOR: "#e67e22",
        EquipmentType.FIN_FAN_COOLER: "#16a085",
        EquipmentType.STRUCTURE: "#7f8c8d",
    }

    damage_colors = {
        DamageLevel.NONE: "#bdc3c7",
        DamageLevel.MINOR: "#f1c40f",
        DamageLevel.MODERATE: "#e67e22",
        DamageLevel.MAJOR: "#e74c3c",
        DamageLevel.CATASTROPHIC: "#8e44ad",
    }

    vector_styles = {
        EscalationVector.THERMAL_RADIATION: ("--", "🔥"),
        EscalationVector.OVERPRESSURE: (":", "💥"),
        EscalationVector.FIRE_IMPINGEMENT: ("-", "🔥"),
    }

    # Draw equipment
    eq_map = {eq.id: eq for eq in result.equipment_list}
    at_risk = set(result.summary.get("equipment_at_risk", []))

    for eq in result.equipment_list:
        color = eq_colors.get(eq.equipment_type, "#3498db")
        is_primary = eq.id == result.primary_event
        is_affected = eq.id in at_risk

        if is_primary:
            marker_size = 200
            edge_color = "red"
            edge_width = 3
            zorder = 10
        elif is_affected:
            marker_size = 150
            edge_color = "orange"
            edge_width = 2
            zorder = 8
        else:
            marker_size = 100
            edge_color = "gray"
            edge_width = 1
            zorder = 5

        ax.scatter(eq.x, eq.y, s=marker_size, c=color, marker="s",
                   edgecolors=edge_color, linewidths=edge_width, zorder=zorder)

        if show_labels:
            ax.annotate(
                eq.id,
                (eq.x, eq.y),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8,
                fontweight="bold" if is_primary else "normal",
                color="red" if is_primary else "black",
            )

    # Draw escalation links
    significant_links = [l for l in result.escalation_links if l.damage_level != DamageLevel.NONE]
    for link in significant_links:
        src = eq_map.get(link.source_id)
        tgt = eq_map.get(link.target_id)
        if src is None or tgt is None:
            continue

        linestyle, emoji = vector_styles.get(link.vector, ("-", ""))
        color = damage_colors.get(link.damage_level, "#e74c3c")
        linewidth = 1 + link.escalation_prob * 3

        ax.annotate(
            "", xy=(tgt.x, tgt.y), xytext=(src.x, src.y),
            arrowprops=dict(
                arrowstyle="->,head_width=0.4,head_length=0.3",
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                connectionstyle="arc3,rad=0.1",
            ),
            zorder=3,
        )

        # Intensity label on arrow
        mid_x = (src.x + tgt.x) / 2
        mid_y = (src.y + tgt.y) / 2
        if link.vector == EscalationVector.THERMAL_RADIATION:
            label = f"{link.intensity:.1f} kW/m²"
        elif link.vector == EscalationVector.OVERPRESSURE:
            label = f"{link.intensity:.1f} kPa"
        else:
            label = "Impingement"

        ax.annotate(
            f"{label}\nP={link.escalation_prob:.0%}",
            (mid_x, mid_y),
            fontsize=7,
            ha="center",
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor=color),
            zorder=15,
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
               markeredgecolor="red", markersize=12, label="Primary Event"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db",
               markeredgecolor="orange", markersize=10, label="At Risk"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db",
               markeredgecolor="gray", markersize=8, label="Equipment"),
        Line2D([0], [0], color="#e74c3c", linestyle="--", linewidth=2, label="Thermal"),
        Line2D([0], [0], color="#e74c3c", linestyle=":", linewidth=2, label="Overpressure"),
        Line2D([0], [0], color="#e74c3c", linestyle="-", linewidth=2, label="Impingement"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(
        f"Domino Effect Escalation Map\n"
        f"Primary: {result.primary_scenario} (f = {result.primary_frequency:.1e}/yr)",
        fontsize=13, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_escalation_summary(
    result: DominoAnalysisResult,
    *,
    save_path: str = None,
    dpi: int = 150,
) -> Optional["plt.Figure"]:
    """Generate a summary bar chart of escalation risks by target equipment.

    Parameters
    ----------
    result : DominoAnalysisResult
        Analysis results.
    save_path : str, optional
        File path.
    dpi : int
        Resolution.

    Returns
    -------
    matplotlib Figure or None
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    significant = [l for l in result.escalation_links if l.damage_level != DamageLevel.NONE]
    if not significant:
        ax1.text(0.5, 0.5, "No escalation risks identified", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=14)
        ax2.text(0.5, 0.5, "No escalation risks identified", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14)
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    # Sort by escalation probability
    sorted_links = sorted(significant, key=lambda l: -l.escalation_prob)

    targets = [l.target_id for l in sorted_links]
    probs = [l.escalation_prob for l in sorted_links]
    intensities = [l.intensity for l in sorted_links]
    vectors = [l.vector.value for l in sorted_links]

    # Colors by vector type
    colors = []
    for v in vectors:
        if "thermal" in v:
            colors.append("#e74c3c")
        elif "overpressure" in v:
            colors.append("#2980b9")
        else:
            colors.append("#e67e22")

    # Plot 1: Escalation probability by target
    bars = ax1.barh(range(len(targets)), probs, color=colors, edgecolor="white")
    ax1.set_yticks(range(len(targets)))
    ax1.set_yticklabels(targets, fontsize=9)
    ax1.set_xlabel("Escalation Probability")
    ax1.set_title("Escalation Probability by Target Equipment")
    ax1.invert_yaxis()

    # Add probability labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{prob:.1%}", va="center", fontsize=8)

    # Plot 2: Intensity at target
    ax2.barh(range(len(targets)), intensities, color=colors, edgecolor="white")
    ax2.set_yticks(range(len(targets)))
    ax2.set_yticklabels(targets, fontsize=9)
    ax2.set_xlabel("Intensity (kW/m² or kPa)")
    ax2.set_title("Incident Intensity at Target Equipment")
    ax2.invert_yaxis()

    for i, (bar, intensity) in enumerate(zip(
        ax2.containers[0] if ax2.containers else [], intensities
    )):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{intensity:.1f}", va="center", fontsize=8)

    # Vector legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Thermal Radiation"),
        Patch(facecolor="#2980b9", label="Overpressure"),
        Patch(facecolor="#e67e22", label="Fire Impingement"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.suptitle(
        f"Domino Escalation Summary — {result.primary_scenario}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_domino_chain(
    result: DominoAnalysisResult,
    *,
    save_path: str = None,
    dpi: int = 150,
) -> Optional["plt.Figure"]:
    """Generate a cascade chain diagram showing domino escalation paths.

    Parameters
    ----------
    result : DominoAnalysisResult
        Analysis results.
    save_path : str, optional
        File path.
    dpi : int
        Resolution.

    Returns
    -------
    matplotlib Figure or None
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required")

    fig, ax = plt.subplots(1, 1, figsize=(14, max(6, len(result.domino_scenarios) * 1.5)))

    scenarios = sorted(result.domino_scenarios, key=lambda s: -s.total_frequency)

    if not scenarios:
        ax.text(0.5, 0.5, "No domino scenarios identified", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    y_positions = {}
    y = 0
    for scenario in scenarios:
        chain_str = " → ".join(scenario.chain)
        y_positions[chain_str] = y
        y += 1

    max_chain_len = max(len(s.chain) for s in scenarios)

    for scenario in scenarios:
        y = y_positions[" → ".join(scenario.chain)]
        chain = scenario.chain

        for i, eq_id in enumerate(chain):
            x = i * 3
            is_primary = (i == 0)

            if is_primary:
                box_color = "#e74c3c"
                text_color = "white"
            else:
                # Color by damage level
                link = scenario.links[i-1] if i-1 < len(scenario.links) else None
                if link:
                    if link.damage_level == DamageLevel.CATASTROPHIC:
                        box_color = "#8e44ad"
                    elif link.damage_level == DamageLevel.MAJOR:
                        box_color = "#e67e22"
                    elif link.damage_level == DamageLevel.MODERATE:
                        box_color = "#f1c40f"
                    else:
                        box_color = "#bdc3c7"
                else:
                    box_color = "#3498db"
                text_color = "white"

            rect = FancyBboxPatch(
                (x - 1.0, y - 0.35), 2.0, 0.7,
                boxstyle="round,pad=0.1",
                facecolor=box_color, edgecolor="white", linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(x, y, eq_id, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

            # Arrow to next
            if i < len(chain) - 1:
                ax.annotate(
                    "", xy=(x + 2.0, y), xytext=(x + 1.0, y),
                    arrowprops=dict(arrowstyle="->", color="gray", linewidth=1.5),
                )

        # Frequency label
        ax.text(
            max_chain_len * 3 + 0.5, y,
            f"f = {scenario.total_frequency:.1e}/yr | "
            f"m = {scenario.total_inventory_released:.0f} kg",
            va="center", fontsize=8, color="#2c3e50",
        )

    ax.set_xlim(-1.5, max_chain_len * 3 + 8)
    ax.set_ylim(-1, len(scenarios))
    ax.set_xlabel("Escalation Order →")
    ax.set_title(
        "Domino Effect Chain Diagram\n"
        f"Primary: {result.primary_scenario}",
        fontsize=13, fontweight="bold",
    )
    ax.set_yticks([])
    ax.set_xticks([i * 3 for i in range(max_chain_len)])
    ax.set_xticklabels([f"Order {i+1}" for i in range(max_chain_len)])
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig
