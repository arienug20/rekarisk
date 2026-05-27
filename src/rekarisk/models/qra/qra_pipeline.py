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
from rekarisk.models.dispersion.gaussian_plume import calculate_flash_fire_distance
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
        x, y: Coordinates of the section [m].
        elevation: Release height above grade [m].
        rho_liquid: Override liquid density [kg/m³].
        molecular_weight: MW override [g/mol].
        cp_cv_ratio: Ratio of specific heats γ (default 1.31).
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
    """Jet fire → distance_to_thresholds_jet_multipoint."""
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
        model="multipoint")


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
        # Interpolate heat flux from threshold map
        for th_kw in sorted(impact.keys(), reverse=True):
            if d <= impact[th_kw]:
                q_flux = th_kw * 1000.0  # kW→W/m²
                _y, Pf = thermal_probit(q_flux, exp_t)
                break
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

    # ── Section evaluation ────────────────────────────────────────────────

    def _eval_section(
        self, iso: IsoSection, hole: HoleSize, wx: WeatherScenario,
    ) -> List[_ScenarioBucket]:
        mdot_avg, phase, total_mass, t_final = self._source_term(iso, hole)
        if mdot_avg <= 0:
            return []

        leak_freq = self._leak_freq(hole.name)
        if leak_freq <= 0:
            return []

        # Ignition probabilities
        p_imm = immediate_ignition_probability(
            substance=iso.composition, release_rate=mdot_avg, phase=phase)
        p_del = delayed_ignition_probability(
            substance=iso.composition, release_duration=t_final)
        p_exp = explosion_probability(
            substance=iso.composition, congestion="medium")

        outcomes: List[Tuple[str, float, Callable[[], Dict[float, float]]]] = [
            ("jet_fire", p_imm, lambda: _jet_fire_impact(
                mdot_avg, hole.diameter, iso.composition, wx, iso.elevation)),
            ("pool_fire", (1 - p_imm) * 0.15, lambda: _pool_fire_impact(
                mdot_avg, iso.composition, wx)),
            ("flash_fire", (1 - p_imm) * p_del * (1 - p_exp), lambda: {
                "LFL_half": _flash_fire_impact(mdot_avg, iso.composition, wx,
                                               hole.diameter, t_final)}),
            ("explosion", (1 - p_imm) * p_del * p_exp, lambda: _vce_impact(
                0.1 * total_mass, iso.composition, self.confinement, self.congestion)),
            ("toxic", (1 - p_imm) * (1 - p_del) * 0.05, lambda: self._toxic_lambda(
                mdot_avg, iso.composition, wx, total_mass, t_final)),
        ]

        buckets: List[_ScenarioBucket] = []
        for outcome, prob, impact_fn in outcomes:
            freq = leak_freq * prob * wx.probability
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
                iso_x=iso.x, iso_y=iso.y,
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
        """Returns (mdot_avg [kg/s], phase_label, total_mass [kg], duration [s])."""
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
                vi = VesselInput(
                    V=iso.volume, P0=iso.P, T0=iso.T,
                    d_hole=hole.diameter, Cd=hole.Cd,
                    composition=iso.composition,
                    liquid=(iso.fill_fraction >= 0.5),
                    cp_cv_ratio=iso.cp_cv_ratio,
                    molecular_weight=mw_kgmol,
                    rho_liquid=iso.rho_liquid,
                )
                br: VesselResult = calculate_vessel_blowdown(vi)
                duration = br.t_final
                total_mass = br.total_mass_released
                if duration > 0:
                    mdot_initial = total_mass / duration
            except Exception:
                pass

        return mdot_initial, phase_label, total_mass, duration

    def _leak_freq(self, hole_name: str) -> float:
        """Leak frequency [/yr] from failure DB or fallback."""
        try:
            return lookup_frequency(self.failure_db, "vessel", hole_name.lower())
        except Exception:
            fb: Dict[str, float] = {
                "small": 1e-4, "medium": 5e-5, "large": 1e-5, "fullbore": 5e-6}
            return fb.get(hole_name.lower(), 1e-5)

    # ── Risk integration ──────────────────────────────────────────────────

    def _compute_lsir(self, buckets: List[_ScenarioBucket]) -> Dict[Tuple[float, float], float]:
        lsir: Dict[Tuple[float, float], float] = {}
        for rp in self.receptors:
            c = 0.0
            for b in buckets:
                dx = rp.x - b.iso_x
                dy = rp.y - b.iso_y
                d = math.sqrt(dx * dx + dy * dy)
                et = EXP_TIMES.get(b.outcome, 60.0)
                iso = next((s for s in self.iso_sections if s.name == b.iso_name), None)
                sub = iso.composition if iso else "default"
                Pf = _fatal_prob(b.outcome, d, b.impact_dists, sub, et,
                                 self.shelter_ach, True)
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
