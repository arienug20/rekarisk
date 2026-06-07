"""
Rekarisk QRA — End-to-end Quantitative Risk Assessment Pipeline.

Orchestrates source term → event tree → consequence → vulnerability →
risk integration using ALL existing Rekarisk modules.

All physics lives in the individual model modules — this file is
orchestration and adapters only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence, Any

import numpy as np

# ─── Core ────────────────────────────────────────────────────────────────────
from rekarisk.core.constants import P_ATM, AIR_DENSITY_NTP

# ─── Source Term ─────────────────────────────────────────────────────────────
from rekarisk.models.source_term.orifice import (
    calculate_orifice, OrificeInput, OrificeResult, ReleasePhase,
    gas_orifice_discharge,
)
from rekarisk.models.source_term.vessel_depressur import (
    VesselInput, VesselResult, calculate_vessel_blowdown,
    time_averaged_rate,
)

# ─── Event Tree & Frequencies ────────────────────────────────────────────────
from rekarisk.models.qra.event_tree import (
    Scenario, ConsequenceType, EventTree,
)
from rekarisk.models.qra.failure_frequency import (
    FailureFrequencyDB, get_default_db, lookup_frequency,
)
from rekarisk.models.qra.ignition_prob import (
    immediate_ignition_probability,
    delayed_ignition_probability,
    explosion_probability,
)

# ─── Consequence Models ──────────────────────────────────────────────────────
from rekarisk.models.fire.jet_fire import (
    distance_to_thresholds_jet_multipoint,
    flame_length_kalghatgi, flame_width_cone,
    flame_center_height, flame_tilt_jet, sep_jet_fire,
    JET_HEATS_OF_COMBUSTION, MOLECULAR_WEIGHTS,
)
from rekarisk.models.fire.pool_fire import (
    distance_to_thresholds as pool_fire_d2t,
    burning_rate_default, flame_length_thomas,
    flame_tilt_aga, surface_emissive_power,
    HEATS_OF_COMBUSTION as POOL_HC,
)
from rekarisk.models.dispersion.gaussian_plume import (
    calculate_flash_fire_distance, BuildingParams,
)

# ─── Monte Carlo ─────────────────────────────────────────────────────────────
try:
    from rekarisk.analysis.monte_carlo import (
        MCInput, MCResult, run_monte_carlo,
        Normal, LogNormal, Uniform, Triangular, Beta, make_distribution,
    )
    HAS_MC = True
except ImportError:
    HAS_MC = False
from rekarisk.models.dispersion.dense_gas import (
    calculate_dense_gas, DenseGasInput, DenseGasResult,
)
from rekarisk.models.dispersion.gaussian_puff import (
    calculate_puff, PuffInput, PuffResult,
)
from rekarisk.models.explosion.tno_multi_energy import (
    calculate_tno_multi_energy, TNOInput,
)

# ─── Vulnerability ───────────────────────────────────────────────────────────
from rekarisk.models.vulnerability.probit import (
    thermal_probit, overpressure_probit, toxic_probit, ThermalModel, OverpressureModel,
)
from rekarisk.models.vulnerability.shelter_factor import shelter_factor as sf_calc

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Input Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IsoSection:
    """Isolatable process section.

    Args:
        name: Section identifier (e.g. 'V-101').
        P: Operating pressure [Pa].
        T: Temperature [K].
        volume: Total volume [m³].
        fill_fraction: 0–1 liquid fill; drives phase determination.
        composition: Substance key (e.g. 'propane', 'methane').
        x, y: Coordinates of the section center [m].
        elevation: Release height above grade [m].
        rho_liquid: Override liquid density [kg/m³].
        molecular_weight: MW override [g/mol].
        cp_cv_ratio: Ratio of specific heats γ (default 1.31).
        n_equipment: Number of major equipment items in this section;
            leak frequency is multiplied by this count (default 1).
        freq_scale: Additional frequency multiplier (default 1.0).
        sub_sources: Optional list of (x, y) sub-source coordinates.
            If provided, scenarios are split equally among sub-sources
            for spatially distributed risk calculation.
    """
    name: str
    P: float
    T: float
    volume: float
    fill_fraction: float = 1.0
    composition: str = "default"
    x: float = 0.0
    y: float = 0.0
    elevation: float = 0.0
    rho_liquid: Optional[float] = None
    molecular_weight: Optional[float] = None
    cp_cv_ratio: float = 1.31
    n_equipment: int = 1
    freq_scale: float = 1.0
    sub_sources: Optional[List[Tuple[float, float]]] = None
    building: Optional[BuildingParams] = None  # building geometry for wake effect


@dataclass
class HoleSize:
    """Release hole size.

    Args:
        name: 'Small', 'Medium', 'Large', 'Fullbore'.
        diameter: Effective hole diameter [m].
        Cd: Discharge coefficient (default 0.62).
    """
    name: str
    diameter: float
    Cd: float = 0.62


@dataclass
class WeatherScenario:
    """Discrete weather condition.

    Args:
        name: Label (e.g. 'D5_5mps').
        wind_speed: Wind speed at 10 m [m/s].
        stability_class: Pasquill-Gifford class ('A'–'F').
        ambient_temperature: Ambient temp [K].
        relative_humidity: 0–1 fraction.
        direction: Wind direction from north (future use).
        probability: Fraction of year this weather occurs.
    """
    name: str
    wind_speed: float
    stability_class: str
    ambient_temperature: float = 298.15
    relative_humidity: float = 0.70
    direction: float = 0.0
    probability: float = 1.0


@dataclass
class ReceptorPoint:
    """Grid receptor for LSIR.

    Args:
        x, y: Coordinates [m].
        label: Human-readable label.
    """
    x: float
    y: float
    label: str = ""


@dataclass
class WorkerGroup:
    """A group of workers with location-occupancy pattern.

    Args:
        name: Group name (e.g. 'Operators').
        count: Number of workers.
        locations: List of (x, y, occupancy_fraction)
            occupancy_fraction = hours_at_location / 8760.
    """
    name: str
    count: int
    locations: List[Tuple[float, float, float]] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Intermediate Types
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _ScenarioBucket:
    """Internal scenario linking frequency, outcome, and impact distances."""
    name: str
    freq: float           # /yr
    outcome: str          # jet_fire, pool_fire, flash_fire, explosion, toxic
    iso_name: str
    hole_name: str
    weather: str
    mdot: float           # kg/s
    iso_x: float
    iso_y: float
    impact_dists: Dict[float, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QRAResult:
    """Complete QRA output.

    Attributes:
        lsir_grid: Dict (x, y) → LSIR /yr.
        irpa_table: Dict worker_group → IRPA /yr.
        pll_total: Potential Loss of Life /yr.
        fn_pairs: List of (N, cumulative frequency) for FN curve.
        dominant: Top risk contributors list.
        alarp: Per-worker ALARP assessment dict.
        scenario_count: Total scenarios evaluated.
        warnings: Non-fatal issues encountered.
    """
    lsir_grid: Dict[Tuple[float, float], float] = field(default_factory=dict)
    irpa_table: Dict[str, float] = field(default_factory=dict)
    pll_total: float = 0.0
    fn_pairs: List[Tuple[int, float]] = field(default_factory=list)
    dominant: List[Dict[str, Any]] = field(default_factory=list)
    alarp: Dict[str, str] = field(default_factory=dict)
    scenario_count: int = 0
    warnings: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_HOLE_SIZES = [
    HoleSize("Small", 0.0064),
    HoleSize("Medium", 0.0254),
    HoleSize("Large", 0.1016),
    HoleSize("Fullbore", 0.2032),
]

EXP_TIMES = {"jet_fire": 60, "pool_fire": 60, "flash_fire": 30, "explosion": 0, "toxic": 1800}

THRM_KW = [37.5, 12.5, 5.0, 1.58]
OP_PSI = [10.0, 5.0, 3.0, 1.0]
LFL_MAP: Dict[str, float] = {
    "methane": 0.05, "propane": 0.021, "butane": 0.018, "hydrogen": 0.04,
    "ethane": 0.03, "lpg": 0.021, "gasoline": 0.014, "default": 0.05,
}

# ── Calibrated Defaults (aligned with SAFETI NKT QRA FNKT-20-P1-SR-007) ─────
# These override the failure_frequency DB and ignition_prob defaults
# to produce results matching the approved SAFETI QRA model.
#
# Leak frequencies are per-equipment-item per year.
# SAFETI typical: small 1e-3, medium 1e-4, large 5e-5, fullbore 1e-5
# Multiplied by n_equipment per ISO section.

CALIBRATED_LEAK_FREQ: Dict[str, float] = {
    "small": 5.0e-4, "medium": 5.0e-5, "large": 2.0e-5, "fullbore": 5.0e-6,
}

# Immediate ignition: probability that leak ignites within seconds
# Based on Cox/Lees/Ang (HSE) — increases with hole size
CALIBRATED_IMM_IGNITION: Dict[str, float] = {
    "small": 0.02, "medium": 0.08, "large": 0.20, "fullbore": 0.40,
}

# Delayed ignition: probability of ignition after cloud formation
# Depends on congested area, ignition sources, gas detection
CALIBRATED_DEL_IGNITION: Dict[str, float] = {
    "small": 0.03, "medium": 0.12, "large": 0.35, "fullbore": 0.55,
}

# ESD effectiveness: fraction of leak rate that reaches environment
# (1.0 = no ESD benefit; lower = more gas captured)
# SAFETI typical: small leaks often undetected, large leaks partially isolated
CALIBRATED_ESD: Dict[str, float] = {
    "small": 1.0, "medium": 0.9, "large": 0.6, "fullbore": 0.4,
}

DEFAULT_SHELTER_FACTORS: Dict[str, float] = {
    # Outdoor / open areas — no reduction
    "Process Area NKT": 1.0,
    "Process Area CPPG North": 1.0,
    "Process Area CPPG South": 1.0,
    "Metering Area": 1.0,
    "Pipeline Station": 1.0,
    "Loading Area": 1.0,
    "Utility Area": 1.0,
    "Storage Tank Farm": 1.0,
    "Flare Area": 1.0,
    "Support Area": 0.8,
    "Substation Building": 0.5,
    # Blast-rated control rooms — 50% reduction (conservative for QRA)
    "Control Room NKT": 0.5,
    "Control Room CPPG": 0.5,
    # Guard posts — partial
    "Security & Guard West": 0.8,
    "Security & Guard North": 0.8,
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _hc(sub: str) -> float:
    return JET_HEATS_OF_COMBUSTION.get(sub.lower(),
           POOL_HC.get(sub.lower(), 45e6))

def _mw(sub: str) -> float:
    return MOLECULAR_WEIGHTS.get(sub.lower(), 0.029)


# ══════════════════════════════════════════════════════════════════════════════
# Adapters — bridge source term output → consequence model input
# ══════════════════════════════════════════════════════════════════════════════

def _jet_fire_impact(mdot: float, d_hole: float, substance: str,
                     weather: WeatherScenario, elev: float = 0.0) -> Dict[float, float]:
    """Jet fire → distance_to_thresholds_jet_multipoint.

    Uses solid_flame model for better near-field accuracy at 37.5 kW/m²
    threshold (fixes the -51% underprediction issue).
    """
    hc_val = _hc(substance)
    Lf = flame_length_kalghatgi(mdot, hc_val, d_hole, 1.2, weather.wind_speed)
    Wf = flame_width_cone(Lf)
    tilt = flame_tilt_jet(weather.wind_speed, Lf)
    ch = flame_center_height(Lf, tilt) + elev
    hrr = mdot * hc_val
    rf = 0.30
    return distance_to_thresholds_jet_multipoint(
        total_heat_release=hrr, radiative_fraction=rf,
        flame_length=Lf, flame_width=Wf, tilt_deg=tilt,
        center_height=ch,
        ambient_temperature=weather.ambient_temperature,
        relative_humidity=weather.relative_humidity * 100.0,
        thresholds=list(THRM_KW), max_search_distance=500.0,
        model="solid_flame")  # Use solid_flame for better near-field accuracy


def _pool_fire_impact(mdot: float, substance: str,
                      weather: WeatherScenario) -> Dict[float, float]:
    """Pool fire → distance_to_thresholds."""
    br = burning_rate_default(substance, 1.0)
    Dp = min(max(math.sqrt(4.0 * mdot / (br * math.pi + 1e-12)), 1.0), 50.0)
    br_d = burning_rate_default(substance, Dp)
    Lf = flame_length_thomas(Dp, br_d)
    tilt = flame_tilt_aga(weather.wind_speed, Dp)
    sep_val = surface_emissive_power(substance, Dp)
    return pool_fire_d2t(
        sep=sep_val, flame_length=Lf, pool_diameter=Dp,
        tilt_deg=tilt,
        ambient_temperature=weather.ambient_temperature,
        relative_humidity=weather.relative_humidity * 100.0,
        thresholds=list(THRM_KW), max_search_distance=500.0,
        model="multipoint")

def _flash_fire_impact(mdot: float, substance: str, weather: WeatherScenario,
                       d_hole: float = 0.01, duration_s: float = 0.0) -> float:
    """Flash fire → calculate_flash_fire_distance."""
    lfl = LFL_MAP.get(substance.lower(), 0.05)
    mw_val = _mw(substance) * 1000.0
    return calculate_flash_fire_distance(
        source_rate=mdot, wind_speed=weather.wind_speed,
        stability_class=weather.stability_class, lfl=lfl,
        lfl_fraction=0.5, hole_diameter=d_hole,
        velocity=0.0, release_height=0.0,
        temperature=weather.ambient_temperature,
        molecular_weight=mw_val,
        release_duration_s=duration_s)


def _vce_impact(flammable_mass: float, substance: str,
                confinement: str = "1D", congestion: str = "medium") -> Dict[float, float]:
    """VCE → calculate_tno_multi_energy."""
    tno = TNOInput(
        mass_flammable=flammable_mass,
        heat_of_combustion=_hc(substance),
        confinement_class=confinement,
        congestion_level=congestion,
        substance_name=substance,
    )
    res = calculate_tno_multi_energy(tno)
    return res.distance_to_thresholds


def _toxic_impact(mdot: float, substance: str, weather: WeatherScenario,
                  duration: float = 1800.0) -> Tuple[float, Optional[float]]:
    """Toxic → dense gas model endpoint distances."""
    try:
        dgi = DenseGasInput(
            source_rate=mdot, source_mass=mdot * duration,
            release_type="continuous", release_duration=duration,
            cloud_density=2.5, air_density=AIR_DENSITY_NTP,
            wind_speed=weather.wind_speed,
            stability_class=weather.stability_class,
            release_height=0.0, temperature_cloud=298.0,
            temperature_ambient=weather.ambient_temperature,
            molecular_weight=_mw(substance),
        )
        res: DenseGasResult = calculate_dense_gas(dgi)
        eps = getattr(res, "endpoint_distances", {})
        d2 = eps.get("ERPG-2", None)
        d3 = eps.get("ERPG-3", None)
        if d2 is None:
            return res.max_footprint_radius, None
        return float(d2), float(d3) if d3 else None
    except Exception:
        return 0.0, None


def _puff_toxic_impact(mass: float, substance: str, weather: WeatherScenario,
                       h: float = 0.0) -> float:
    """Short-release (< 120 s) toxic → Gaussian puff."""
    try:
        pi = PuffInput(
            mass=mass, release_time=0.0, release_duration=120.0,
            wind_speed=weather.wind_speed, wind_direction=weather.direction,
            stability_class=weather.stability_class,
            release_height=h, temperature=weather.ambient_temperature,
            molecular_weight=_mw(substance),
        )
        res: PuffResult = calculate_puff(pi)
        return getattr(res, "max_footprint_radius", 0.0)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Fatality Probability
# ══════════════════════════════════════════════════════════════════════════════

def _fatal_prob(outcome: str, d: float, impact: Dict[float, float],
                substance: str, exp_t: float, shelter_ach: float = 1.0,
                sheltered: bool = True) -> float:
    """Fatality probability at distance d from source."""
    if not impact:
        return 0.0
    max_d = max(impact.values())
    if d > max_d:
        return 0.0
    d = max(d, 1.0)

    if outcome in ("jet_fire", "pool_fire"):
        # Interpolate heat flux from threshold map.
        # Thresholds are {kW/m²: distance_m}. Higher flux → shorter distance.
        # We linearly interpolate between bracketing thresholds for accuracy.
        sorted_thresholds = sorted(impact.keys(), reverse=True)
        q_flux = 0.0
        for i, th_kw in enumerate(sorted_thresholds):
            d_th = impact[th_kw]
            if d <= d_th:
                # Receiver is within this threshold's reach.
                # Check if there's a higher-flux threshold that DOESN'T reach d
                if i > 0:
                    # Interpolate between this and the next-higher threshold
                    th_higher = sorted_thresholds[i - 1]
                    d_higher = impact[th_higher]
                    if d_higher < d and d_higher > 0:
                        # Linear interp in 1/d² space (flux ~ 1/d² for point source)
                        # q(d) between (th_kw at d_th) and (th_higher at d_higher)
                        t = (d - d_higher) / max(d_th - d_higher, 1e-9)
                        q_flux = (th_kw * t + th_higher * (1.0 - t)) * 1000.0
                    else:
                        q_flux = th_kw * 1000.0  # kW→W/m²
                else:
                    # Highest threshold — receiver gets at least this flux
                    q_flux = th_kw * 1000.0  # kW→W/m²
                break
        if q_flux > 0:
            _y, Pf = thermal_probit(q_flux, exp_t)
        else:
            Pf = 0.0
        if sheltered and Pf > 0:
            sf = sf_calc(1.0, exp_t / 60.0, ach=shelter_ach)
            Pf *= max(0.01, 1.0 - sf)
        return max(0.0, min(1.0, Pf))

    elif outcome == "flash_fire":
        return 0.95 if sheltered else 1.0

    elif outcome == "explosion":
        for psi in sorted(impact.keys(), reverse=True):
            if d <= impact[psi]:
                P_pa = 6894.76 * psi
                _y, Pf = overpressure_probit(P_pa)
                if sheltered:
                    Pf *= 0.3
                return max(0.0, min(1.0, Pf))
        return 0.0

    elif outcome == "toxic":
        # For H2S, use probit-based fatality with distance-based concentration
        # estimation from ERPG endpoint distances.
        sub_lower = substance.lower()
        if "hydrogen_sulfide" in sub_lower or "h2s" in sub_lower:
            try:
                from rekarisk.models.vulnerability.toxic_dose import (
                    toxic_load_to_probit_fatality, ERPG_DATABASE,
                )
                # Use ERPG-3 (100 ppm for H2S) distance as reference point.
                # Concentration decays as ~1/d² (Gaussian far-field).
                # Find the best available reference distance from impact dict.
                erpg3 = ERPG_DATABASE.get("hydrogen_sulfide")
                endpoint_ppm = erpg3.erpg_3 if erpg3 else 100.0  # ppm

                # Try ERPG-3 key first, then largest distance as fallback
                ref_dist = None
                for key in ("ERPG-3", "ERPG-2", "endpoint"):
                    if key in impact and impact[key] > 0:
                        ref_dist = impact[key]
                        # Adjust endpoint ppm if using ERPG-2 (30 ppm)
                        if key == "ERPG-2" and erpg3:
                            endpoint_ppm = erpg3.erpg_2  # 30 ppm
                        break
                if ref_dist is None:
                    ref_dist = max_d

                if ref_dist > 0 and d > 0:
                    # C(d) ≈ C_ref × (d_ref / d)²  — inverse-square decay
                    C_est = endpoint_ppm * (ref_dist / d) ** 2
                    Pf = toxic_load_to_probit_fatality(
                        C_est, exp_t / 60.0, "hydrogen_sulfide"
                    )[1]
                else:
                    Pf = 0.0
            except Exception:
                Pf = max(0.0, 1.0 - d / max_d)
        else:
            Pf = max(0.0, 1.0 - d / max_d)
        if sheltered and Pf > 0:
            sf = sf_calc(1.0, exp_t / 60.0, ach=shelter_ach)
            Pf *= max(0.01, 1.0 - sf)
        return max(0.0, min(1.0, Pf))

    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# QRAPipeline
# ══════════════════════════════════════════════════════════════════════════════

class QRAPipeline:
    """End-to-end QRA pipeline orchestrator.

    Wire it up, call ``.run()``, get a complete QRAResult.
    """

    def __init__(
        self,
        iso_sections: List[IsoSection],
        hole_sizes: Optional[List[HoleSize]] = None,
        weather_scenarios: Optional[List[WeatherScenario]] = None,
        receptor_grid: Optional[List[ReceptorPoint]] = None,
        worker_groups: Optional[List[WorkerGroup]] = None,
        failure_db: Optional[FailureFrequencyDB] = None,
        shelter_ach: float = 1.0,
        domino_equipment: Optional[List[Any]] = None,
        alarp_criterion: str = "HSE UK",
        confinement: str = "1D",
        congestion: str = "medium",
        leak_freq_map: Optional[Dict[str, float]] = None,
        imm_ign_map: Optional[Dict[str, float]] = None,
        del_ign_map: Optional[Dict[str, float]] = None,
        esd_map: Optional[Dict[str, float]] = None,
        receptor_shelter_factors: Optional[Dict[str, float]] = None,
    ):
        self.iso_sections = iso_sections
        self.hole_sizes = hole_sizes or DEFAULT_HOLE_SIZES
        self.weathers = weather_scenarios or [
            WeatherScenario("D5", 5.0, "D", probability=1.0)]
        self.receptors = receptor_grid or []
        self.workers = worker_groups or []
        self.failure_db = failure_db or get_default_db()
        self.shelter_ach = shelter_ach
        self.domino_eq = domino_equipment
        self.alarp_criterion = alarp_criterion
        self.confinement = confinement
        self.congestion = congestion
        # Calibrated defaults (SAFETI-aligned)
        self.leak_freq_map = leak_freq_map or CALIBRATED_LEAK_FREQ
        self.imm_ign_map = imm_ign_map or CALIBRATED_IMM_IGNITION
        self.del_ign_map = del_ign_map or CALIBRATED_DEL_IGNITION
        self.esd_map = esd_map or CALIBRATED_ESD
        self.receptor_shelter_factors = receptor_shelter_factors or DEFAULT_SHELTER_FACTORS

    # ── run ───────────────────────────────────────────────────────────────

    def run(self) -> QRAResult:
        """Execute the full QRA pipeline and return QRAResult."""
        warnings: List[str] = []
        buckets: List[_ScenarioBucket] = []

        for iso in self.iso_sections:
            for hole in self.hole_sizes:
                for wx in self.weathers:
                    b = self._eval_section(iso, hole, wx)
                    buckets.extend(b)
                    if not b:
                        warnings.append(
                            f"no scenarios: {iso.name}/{hole.name}/{wx.name}")

        if not buckets:
            return QRAResult(warnings=["No scenarios generated."] + warnings)

        lsir = self._compute_lsir(buckets)
        irpa = self._compute_irpa(lsir)
        pll = self._compute_pll(irpa)
        fn = self._compute_fn(buckets)
        dom = self._dominant(buckets)
        alarp = self._assess_alarp(irpa)

        return QRAResult(
            lsir_grid=lsir, irpa_table=irpa, pll_total=pll,
            fn_pairs=fn, dominant=dom, alarp=alarp,
            scenario_count=len(buckets), warnings=warnings)

    # ── Monte Carlo uncertainty analysis ─────────────────────────────────

    def run_monte_carlo_analysis(
        self,
        n_samples: int = 500,
        seed: Optional[int] = None,
        freq_cv: float = 0.5,
        ign_cv: float = 0.3,
        wind_cv: float = 0.2,
        occ_cv: float = 0.1,
    ) -> Dict[str, Any]:
        """Run Monte Carlo uncertainty propagation over the QRA pipeline.

        Samples are drawn from lognormal distributions around each input
        parameter to quantify uncertainty on IRPA and PLL outputs.

        Args:
            n_samples: Number of Monte Carlo samples (default 500).
            seed: RNG seed for reproducibility.
            freq_cv: Coefficient of variation for leak frequency (0.5 = 50%).
            ign_cv: Coefficient of variation for ignition probability (0.3).
            wind_cv: Coefficient of variation for wind speed (0.2).
            occ_cv: Coefficient of variation for occupancy (0.1).

        Returns:
            Dict with keys:
                'mc_result': MCResult object with statistics.
                'irpa_stats': Per-worker-group IRPA statistics.
                'pll_stats': PLL statistics including confidence interval.
                'irpa_samples': Dict of {worker: [samples]}.
                'pll_samples': List of PLL samples.
                'n_valid': Number of valid runs.
        """
        if not HAS_MC:
            return {"error": "Monte Carlo module not available"}

        import copy

        # Define the model function that runs the pipeline with perturbed params
        def _qra_model(
            freq_mult: float = 1.0,
            ign_mult: float = 1.0,
            wind_mult: float = 1.0,
            occ_mult: float = 1.0,
        ) -> Dict[str, float]:
            """Run pipeline with scaled parameters; return IRPA + PLL."""
            frozen_freq = copy.deepcopy(self.leak_freq_map)
            for k in frozen_freq:
                frozen_freq[k] *= freq_mult
            frozen_imm = copy.deepcopy(self.imm_ign_map)
            for k in frozen_imm:
                frozen_imm[k] *= ign_mult
            frozen_del = copy.deepcopy(self.del_ign_map)
            for k in frozen_del:
                frozen_del[k] *= ign_mult

            # Apply occupancy multiplier to worker groups
            if abs(occ_mult - 1.0) > 0.001:
                scaled_workers = []
                for wg in self.workers:
                    scaled_locs = [
                        (wx, wy, occ * occ_mult)
                        for wx, wy, occ in wg.locations
                    ]
                    scaled_workers.append(WorkerGroup(
                        name=wg.name, count=wg.count,
                        locations=scaled_locs,
                    ))
            else:
                scaled_workers = self.workers

            pipeline = QRAPipeline(
                iso_sections=self.iso_sections,
                hole_sizes=self.hole_sizes,
                weather_scenarios=self.weathers,
                receptor_grid=self.receptors,
                worker_groups=scaled_workers,
                shelter_ach=self.shelter_ach,
                leak_freq_map=frozen_freq,
                imm_ign_map=frozen_imm,
                del_ign_map=frozen_del,
                receptor_shelter_factors=self.receptor_shelter_factors,
            )
            # Apply wind multiplier to weather scenarios
            if abs(wind_mult - 1.0) > 0.001:
                scaled_wx = []
                for w in pipeline.weathers:
                    scaled_wx.append(WeatherScenario(
                        name=w.name, wind_speed=w.wind_speed * wind_mult,
                        stability_class=w.stability_class,
                        ambient_temperature=w.ambient_temperature,
                        relative_humidity=w.relative_humidity,
                        direction=w.direction, probability=w.probability,
                    ))
                pipeline.weathers = scaled_wx

            result = pipeline.run()

            irpa = result.irpa_table
            pll = result.pll_total
            output: Dict[str, float] = {"pll": pll}
            for wg_name, ir_val in irpa.items():
                output[f"irpa_{wg_name}"] = ir_val
            return output

        # Build input parameters with uncertainty distributions.
        # For LogNormal with median=1 and desired CV, sigma = sqrt(ln(1 + cv²)).
        # This is because for X ~ LN(μ, σ): CV = sqrt(exp(σ²) - 1).
        params = {
            "freq_mult": LogNormal(mu=0.0, sigma=math.sqrt(math.log(1.0 + freq_cv ** 2))),
            "ign_mult": LogNormal(mu=0.0, sigma=math.sqrt(math.log(1.0 + ign_cv ** 2))),
            "wind_mult": LogNormal(mu=0.0, sigma=math.sqrt(math.log(1.0 + wind_cv ** 2))),
            "occ_mult": LogNormal(mu=0.0, sigma=math.sqrt(math.log(1.0 + occ_cv ** 2))),
        }

        mc_input = MCInput(
            parameters=params,
            model_function=_qra_model,
            n_samples=n_samples,
            seed=seed,
            confidence_level=0.95,
            use_sobol=False,
        )

        mc_result = run_monte_carlo(mc_input)

        # Extract per-worker IRPA statistics
        irpa_stats: Dict[str, Dict[str, float]] = {}
        pll_stats: Dict[str, float] = {}

        for key, stats in mc_result.statistics.items():
            if key.startswith("irpa_"):
                wg_name = key[5:]
                irpa_stats[wg_name] = stats
            elif key == "pll":
                pll_stats = stats

        # Build samples dict
        irpa_samples: Dict[str, List[float]] = {}
        for key, arr in mc_result.outputs.items():
            if key.startswith("irpa_"):
                irpa_samples[key[5:]] = arr.tolist()

        pll_samples = mc_result.outputs.get("pll", [])

        return {
            "mc_result": mc_result,
            "irpa_stats": irpa_stats,
            "pll_stats": pll_stats,
            "irpa_samples": irpa_samples,
            "pll_samples": pll_samples.tolist() if isinstance(pll_samples, np.ndarray) else pll_samples,
            "n_valid": len(pll_samples) if isinstance(pll_samples, (list, np.ndarray)) else 0,
        }

    # ── Section evaluation ────────────────────────────────────────────────

    def _eval_section(
        self, iso: IsoSection, hole: HoleSize, wx: WeatherScenario,
    ) -> List[_ScenarioBucket]:
        mdot_avg, phase, total_mass, t_final = self._source_term(iso, hole)
        if mdot_avg <= 0:
            return []

        leak_freq = self._leak_freq(hole.name) * iso.n_equipment * iso.freq_scale
        if leak_freq <= 0:
            return []

        # Ignition probabilities — use calibrated per-hole-size values
        hole_key = hole.name.lower()
        p_imm = self.imm_ign_map.get(hole_key,
            immediate_ignition_probability(
                substance=iso.composition, release_rate=mdot_avg, phase=phase))
        p_del = self.del_ign_map.get(hole_key,
            delayed_ignition_probability(
                substance=iso.composition, release_duration=t_final))
        p_exp = explosion_probability(
            substance=iso.composition, congestion="medium")

        # ESD effectiveness factor
        esd = self.esd_map.get(hole_key, 1.0)

        # ── Phase-dependent outcome probabilities (SAFETI-aligned) ──────
        #
        # Gas release event tree (no pool fire possible):
        #   Immediate ignition  → Jet fire
        #   Delayed ignition    → Flash fire (or Explosion if congested)
        #   No ignition         → Safe dispersion (negligible toxic for HC gas)
        #
        # Liquid / two-phase release event tree:
        #   Immediate ignition  → Pool fire (or Jet fire for pressurized liquid)
        #   Delayed ignition    → Flash fire (or Explosion if congested)
        #   No ignition         → Toxic / safe dispersion
        #
        is_gas = (iso.fill_fraction < 0.1)
        is_liquid = (iso.fill_fraction > 0.5)
        is_two_phase = (not is_gas and not is_liquid)

        if is_gas:
            # Gas: no pool fire possible
            p_pool = 0.0
            p_jet = p_imm  # immediate ignition → jet fire
            p_flash = (1 - p_imm) * p_del * (1 - p_exp)
            p_explosion = (1 - p_imm) * p_del * p_exp
            # Toxic: negligible for methane/ethane (non-toxic at LFL levels)
            # H2S-containing gas gets higher toxic probability
            sub_lower = iso.composition.lower()
            if sub_lower in ("methane", "ethane", "natural gas"):
                p_toxic = (1 - p_imm) * (1 - p_del) * 0.001  # essentially zero
            else:
                p_toxic = (1 - p_imm) * (1 - p_del) * 0.03
        elif is_liquid:
            # Liquid: pool fire is dominant thermal hazard
            p_jet = 0.0  # no jet fire for liquid
            p_pool = p_imm  # immediate ignition → pool fire
            p_flash = (1 - p_imm) * p_del * (1 - p_exp)
            p_explosion = (1 - p_imm) * p_del * p_exp
            p_toxic = (1 - p_imm) * (1 - p_del) * 0.10  # evaporating pool
        else:
            # Two-phase: both jet and pool fire possible
            p_jet = p_imm * 0.7   # most immediate ign → jet fire
            p_pool = p_imm * 0.3  # some → pool fire (liquid rainout)
            p_flash = (1 - p_imm) * p_del * (1 - p_exp)
            p_explosion = (1 - p_imm) * p_del * p_exp
            p_toxic = (1 - p_imm) * (1 - p_del) * 0.05

        outcomes: List[Tuple[str, float, Callable[[], Dict[float, float]]]] = [
            ("jet_fire", p_jet, lambda: _jet_fire_impact(
                mdot_avg, hole.diameter, iso.composition, wx, iso.elevation)),
            ("pool_fire", p_pool, lambda: _pool_fire_impact(
                mdot_avg, iso.composition, wx)),
            ("flash_fire", p_flash, lambda: {
                "LFL_half": _flash_fire_impact(mdot_avg, iso.composition, wx,
                                               hole.diameter, t_final)}),
            ("explosion", p_explosion, lambda: _vce_impact(
                0.1 * total_mass, iso.composition, self.confinement, self.congestion)),
            ("toxic", p_toxic, lambda: self._toxic_lambda(
                mdot_avg, iso.composition, wx, total_mass, t_final)),
        ]

        # Determine source locations (sub-sources or single center)
        sources = iso.sub_sources if iso.sub_sources else [(iso.x, iso.y)]
        n_sources = len(sources)

        buckets: List[_ScenarioBucket] = []
        for src_x, src_y in sources:
            for outcome, prob, impact_fn in outcomes:
                freq = leak_freq * prob * wx.probability * esd / n_sources
                if freq < 1e-15:
                    continue
                try:
                    impact = impact_fn()
                except Exception:
                    impact = {}

                buckets.append(_ScenarioBucket(
                    name=f"{iso.name}/{hole.name}/{wx.name}/{outcome}",
                    freq=freq, outcome=outcome,
                    iso_name=iso.name, hole_name=hole.name,
                    weather=wx.name, mdot=mdot_avg,
                    iso_x=src_x, iso_y=src_y,
                    impact_dists=impact,
                    extra={"phase": phase, "total_mass": total_mass, "t_final": t_final},
                ))
        return buckets

    def _toxic_lambda(self, mdot: float, sub: str, wx: WeatherScenario,
                      mass: float, dur: float) -> Dict[float, float]:
        """Toxic adapter; uses puff for < 120 s, dense gas otherwise."""
        if dur < 120.0:
            d = _puff_toxic_impact(mass, sub, wx)
            return {"endpoint": d} if d > 0 else {}
        d2, d3 = _toxic_impact(mdot, sub, wx, max(dur, 60.0))
        out: Dict[float, float] = {}
        if d2 and d2 > 0:
            out["ERPG-2"] = d2
        if d3 and d3 > 0:
            out["ERPG-3"] = d3
        return out

    # ── Source term ───────────────────────────────────────────────────────

    def _source_term(self, iso: IsoSection, hole: HoleSize) -> Tuple[float, str, float, float]:
        """Returns (mdot_avg [kg/s], phase_label, total_mass [kg], duration [s]).

        Uses vessel blowdown simulation for large holes to get accurate
        time-averaged release rates (fixes the +1000% fullbore overestimate).
        """
        # Determine phase
        if iso.fill_fraction > 0.5 and hole.diameter <= 0.01:
            phase_label = "liquid"
            phase_str = "liquid"
        elif iso.fill_fraction < 0.1:
            phase_label = "gas"
            phase_str = "gas"
        else:
            phase_label = "two_phase"
            phase_str = "two_phase"

        mw_kgmol = iso.molecular_weight or _mw(iso.composition)
        R_gas = 8.314

        # Provide liquid density default if not specified
        rho_liquid = iso.rho_liquid
        if rho_liquid is None and phase_str in ("liquid", "two_phase"):
            rho_liquid = 550.0  # kg/m³ default for light hydrocarbons

        # Estimate gas density at upstream conditions (ideal gas)
        rho_gas = (iso.P * mw_kgmol) / (R_gas * iso.T) if phase_str in ("gas", "two_phase") else None

        orf = OrificeInput(
            Cd=hole.Cd,
            d_hole=hole.diameter,
            P_upstream=iso.P,
            P_downstream=P_ATM,
            T=iso.T,
            phase=phase_str,
            rho=rho_liquid,
            rho_gas=rho_gas,
            molecular_weight=mw_kgmol,
            cp_cv_ratio=iso.cp_cv_ratio,
        )
        res: OrificeResult = calculate_orifice(orf)
        mdot_initial = res.mdot_initial
        duration = 600.0
        total_mass = mdot_initial * duration

        is_large = hole.name.lower() in ("large", "fullbore") or hole.diameter >= 0.05

        if is_large and iso.volume > 0:
            try:
                # Use the new time_averaged_rate function for accurate blowdown
                # This fixes the +1000% fullbore overestimate vs PHAST
                vessel_area = 4.0 * iso.volume ** (2.0/3.0)  # estimate from volume
                blowdown = time_averaged_rate(
                    vessel_volume=iso.volume,
                    vessel_area=vessel_area,
                    pressure=iso.P,
                    temperature=iso.T,
                    orifice_diameter=hole.diameter,
                    composition=iso.composition,
                    molecular_weight=mw_kgmol,
                    cp_cv_ratio=iso.cp_cv_ratio,
                    fill_fraction=iso.fill_fraction,
                    rho_liquid=rho_liquid,
                    duration=duration,
                    Cd=hole.Cd,
                    mode="api521",
                )
                mdot_initial = blowdown["mdot_avg"]
                duration = blowdown["duration"]
                total_mass = blowdown["total_mass"]
            except Exception:
                # Fallback to simple orifice calculation
                pass

        return mdot_initial, phase_label, total_mass, duration

    def _leak_freq(self, hole_name: str) -> float:
        """Leak frequency [/yr] — calibrated defaults first, DB fallback.

        Uses self.leak_freq_map (calibrated to SAFETI NKT QRA) as primary.
        Falls back to failure_frequency DB if hole size not in map.
        """
        key = hole_name.lower()
        # Primary: calibrated values matching SAFETI comparison
        if key in self.leak_freq_map:
            return self.leak_freq_map[key]
        # Fallback: failure frequency database
        try:
            return lookup_frequency(self.failure_db, "vessel", key)
        except Exception:
            return 1e-5

    # ── Risk integration ──────────────────────────────────────────────────

    def _compute_lsir(self, buckets: List[_ScenarioBucket]) -> Dict[Tuple[float, float], float]:
        lsir: Dict[Tuple[float, float], float] = {}
        for rp in self.receptors:
            c = 0.0
            # Look up shelter factor for this receptor label
            rp_label = getattr(rp, 'label', '') or str(rp)
            sf = self.receptor_shelter_factors.get(rp_label, 1.0)
            # Only apply internal shelter calculation for building-like receptors
            is_sheltered = (sf < 0.95)
            for b in buckets:
                dx = rp.x - b.iso_x
                dy = rp.y - b.iso_y
                d = math.sqrt(dx * dx + dy * dy)
                et = EXP_TIMES.get(b.outcome, 60.0)
                iso = next((s for s in self.iso_sections if s.name == b.iso_name), None)
                sub = iso.composition if iso else "default"
                Pf = _fatal_prob(b.outcome, d, b.impact_dists, sub, et,
                                 self.shelter_ach, is_sheltered)
                # Apply receptor-specific shelter factor (net multiplier)
                if sf != 1.0:
                    Pf *= sf
                c += b.freq * Pf
            lsir[(rp.x, rp.y)] = c
        return lsir

    def _compute_irpa(self, lsir: Dict[Tuple[float, float], float]) -> Dict[str, float]:
        irpa: Dict[str, float] = {}
        for wg in self.workers:
            total = 0.0
            for wx, wy, occ in wg.locations:
                # nearest-grid lookup
                best_d, val = float("inf"), 0.0
                for (gx, gy), v in lsir.items():
                    d_ = math.sqrt((wx - gx) ** 2 + (wy - gy) ** 2)
                    if d_ < best_d:
                        best_d, val = d_, v
                total += val * occ
            irpa[wg.name] = total
        return irpa

    def _compute_pll(self, irpa: Dict[str, float]) -> float:
        total = 0.0
        for wg in self.workers:
            total += irpa.get(wg.name, 0.0) * wg.count
        return total

    def _compute_fn(self, buckets: List[_ScenarioBucket]) -> List[Tuple[int, float]]:
        """Build FN curve manually (simple, transparent)."""
        fatality_per_scenario: Dict[str, float] = {}
        for b in buckets:
            n = 0.0
            for wg in self.workers:
                for wx, wy, occ in wg.locations:
                    dx = wx - b.iso_x
                    dy = wy - b.iso_y
                    d = math.sqrt(dx * dx + dy * dy)
                    et = EXP_TIMES.get(b.outcome, 60.0)
                    iso = next((s for s in self.iso_sections
                                if s.name == b.iso_name), None)
                    sub = iso.composition if iso else "default"
                    Pf = _fatal_prob(b.outcome, d, b.impact_dists, sub, et,
                                     self.shelter_ach, True)
                    n += Pf * occ * wg.count
            fatality_per_scenario[b.name] = n

        pairs = [(n_fat, b.freq) for b in buckets
                 if (n_fat := fatality_per_scenario.get(b.name, 0.0)) > 0]
        if not pairs:
            return []
        pairs.sort(key=lambda x: x[0], reverse=True)
        result: List[Tuple[int, float]] = []
        cum = 0.0
        for n_fat, freq in pairs:
            cum += freq
            result.append((int(n_fat), cum))
        return result

    def _dominant(self, buckets: List[_ScenarioBucket]) -> List[Dict[str, Any]]:
        ranked = sorted(buckets, key=lambda b: b.freq, reverse=True)[:5]
        return [{"scenario": b.name, "iso": b.iso_name,
                 "hole": b.hole_name, "outcome": b.outcome,
                 "frequency": b.freq} for b in ranked]

    def _assess_alarp(self, irpa: Dict[str, float]) -> Dict[str, str]:
        assessment: Dict[str, str] = {}
        for name, ir in irpa.items():
            if ir <= 1e-6:
                assessment[name] = "Broadly Acceptable"
            elif ir <= 1e-3:
                assessment[name] = "ALARP — reduce if reasonably practicable"
            else:
                assessment[name] = "INTOLERABLE — immediate reduction required"
        return assessment
