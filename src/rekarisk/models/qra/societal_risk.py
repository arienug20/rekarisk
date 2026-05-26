"""
Rekarisk QRA — Societal Risk (FN Curves).

Societal risk represents the relationship between the frequency and
the number of people suffering a specified level of harm in a given
population from the realisation of specified hazards.

The FN curve plots:
  F(N) = cumulative frequency of accidents causing N or more fatalities
  vs N = number of fatalities

References:
  - TNO Purple Book CPR 18E — Guidelines for QRA
  - CCPS/AIChE — Guidelines for CPQRA
  - HSE UK — Reducing Risks, Protecting People (R2P2)
  - Dutch External Safety Decree (BEVI)
  - NORSOK Z-013
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────

class FNStatus(str, Enum):
    """ALARP status from FN curve criterion comparison."""
    ACCEPTABLE = "acceptable"          # Below acceptable line
    ALARP = "alarp"                    # In ALARP region
    INTOLERABLE = "intolerable"        # Above intolerable line


# ──────────────────────────────────────────────────────────────────────
# FN Criteria definitions
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FNCriterion:
    """FN criterion curve definition.

    Attributes
    ----------
    name : str
        Criterion name (e.g., "Dutch Existing", "HSE UK Intolerable").
    standard : str
        Reference standard (e.g., "TNO/BEVI", "HSE UK", "CSC").
    description : str
        Human-readable description.
    formula_type : str
        Function type: "power_law", "constant", "custom".
    alpha : float
        Power law multiplier: F(N) = alpha / N^beta.
    beta : float
        Power law exponent.
    n_min : int
        Minimum N for which criterion applies (typically 1 or 10).
    is_intolerable_line : bool
        True if this criterion represents the intolerable boundary.
    """
    name: str
    standard: str = "TNO/BEVI"
    description: str = ""
    formula_type: str = "power_law"
    alpha: float = 1e-3
    beta: float = 2.0
    n_min: int = 1
    is_intolerable_line: bool = True

    def evaluate(self, n: float) -> float:
        """Evaluate the FN criterion at N fatalities.

        Returns
        -------
        float
            Maximum tolerable frequency F(N) for the given N.
        """
        if n < 1:
            n = 1

        if self.formula_type == "power_law":
            return self.alpha / (n ** self.beta)
        elif self.formula_type == "constant":
            return self.alpha  # Constant F threshold regardless of N
        else:
            return self.alpha / (n ** self.beta)

    def evaluate_array(self, n_array: np.ndarray) -> np.ndarray:
        """Evaluate FN criterion for an array of N values."""
        n = np.maximum(n_array, 1.0)
        f = np.zeros_like(n, dtype=np.float64)

        if self.formula_type == "power_law":
            f = self.alpha / (n ** self.beta)
        elif self.formula_type == "constant":
            f = np.full_like(n, self.alpha)
        else:
            f = self.alpha / (n ** self.beta)

        return np.maximum(f, 1e-12)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "standard": self.standard,
            "description": self.description,
            "formula_type": self.formula_type,
            "alpha": self.alpha,
            "beta": self.beta,
            "n_min": self.n_min,
            "is_intolerable_line": self.is_intolerable_line,
        }


# Standard FN criteria
FN_CRITERIA: dict[str, FNCriterion] = {
    "dutch_existing": FNCriterion(
        name="Dutch — Existing Installations",
        standard="TNO/BEVI",
        description="Dutch External Safety Decree: existing installations (BEVI art. 12). "
                    "F(N) < 1×10⁻³ / N² for N ≥ 10.",
        alpha=1e-3, beta=2.0, n_min=10,
    ),
    "dutch_new": FNCriterion(
        name="Dutch — New Installations",
        standard="TNO/BEVI",
        description="Dutch External Safety Decree: new installations. "
                    "F(N) < 1×10⁻⁴ / N² for N ≥ 10.",
        alpha=1e-4, beta=2.0, n_min=10,
    ),
    "hse_uk_intolerable": FNCriterion(
        name="HSE UK — Intolerable Line",
        standard="HSE UK",
        description="HSE R2P2: F(N) < 1×10⁻⁴ for N > 1 (intolerable boundary). "
                    "Accidents with societal concern.",
        alpha=1e-4, beta=0.0, n_min=1,
        formula_type="constant",
    ),
    "hse_uk_broadly_acceptable": FNCriterion(
        name="HSE UK — Broadly Acceptable",
        standard="HSE UK",
        description="HSE R2P2: broadly acceptable region. "
                    "Negligible societal risk.",
        alpha=1e-6, beta=0.0, n_min=1,
        formula_type="constant",
    ),
    "csc_canvey": FNCriterion(
        name="CSC — Canvey Island",
        standard="CSC (Advisory Committee on Major Hazards)",
        description="Canvey Island criterion: F(N) < 1×10⁻² / N.",
        alpha=1e-2, beta=1.0, n_min=1,
    ),
    "vrom_hong_kong": FNCriterion(
        name="Hong Kong — VROM Standard",
        standard="HK EPD (VROM-derived)",
        description="Hong Kong risk guidelines: F(N) < 1×10⁻³ / N² for existing, "
                    "< 1×10⁻⁴ / N² for new.",
        alpha=1e-3, beta=2.0, n_min=1,
    ),
    "norsok": FNCriterion(
        name="NORSOK Z-013 — Intolerable",
        standard="NORSOK",
        description="Norway offshore: F(N) exceeding 5×10⁻⁴ / N for N > 1 "
                    "is generally considered intolerable.",
        alpha=5e-4, beta=1.0, n_min=1,
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FNData:
    """FN curve data.

    Attributes
    ----------
    n_values : np.ndarray
        Array of N (fatality counts). Sorted ascending.
    f_values : np.ndarray
        Cumulative frequency F(N) for each N.
    max_n : int
        Maximum fatality count considered.
    total_frequency : float
        Sum of all scenario frequencies (total accident frequency).
    expected_fatalities : float
        Expected number of fatalities per year.
    potential_loss_of_life : float
        PLL — Expected fatalities per year (same as expected_fatalities).
    scenario_fatalities : dict[str, float]
        Expected fatalities per scenario.
    alarp_status : dict[str, FNStatus]
        ALARP status for each criterion evaluated.
    """
    n_values: np.ndarray = field(default_factory=lambda: np.array([]))
    f_values: np.ndarray = field(default_factory=lambda: np.array([]))
    max_n: int = 0
    total_frequency: float = 0.0
    expected_fatalities: float = 0.0
    potential_loss_of_life: float = 0.0
    scenario_fatalities: dict[str, float] = field(default_factory=dict)
    alarp_status: dict[str, FNStatus] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_values": self.n_values.tolist() if len(self.n_values) > 0 else [],
            "f_values": self.f_values.tolist() if len(self.f_values) > 0 else [],
            "max_n": self.max_n,
            "total_frequency": self.total_frequency,
            "expected_fatalities": self.expected_fatalities,
            "potential_loss_of_life": self.potential_loss_of_life,
            "scenario_fatalities": self.scenario_fatalities,
            "alarp_status": {k: v.value for k, v in self.alarp_status.items()},
        }


# ──────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────

def calculate_fn_curve(
    scenarios: list[Any],
    population_grid: Optional[np.ndarray] = None,
    vulnerability_results: Any = None,
    max_n: int = 1000,
    source_x: float = 0.0,
    source_y: float = 0.0,
    population_total: float = 1.0,
    grid_spacing: float = 10.0,
) -> FNData:
    """Calculate the FN (frequency-number) curve from scenarios.

    FN curve construction:
    1. For each scenario i: compute expected fatalities N_i
       N_i = f_i × P_fatality_i × population_exposed_i
    2. For each scenario i: pair (f_i, N_i)
    3. Sort by N descending
    4. F(N_k) = cumulative sum of f_i for N_i ≥ N_k

    Parameters
    ----------
    scenarios : list
        List of Scenario objects with 'probability' (frequency)
        and other attributes.
    population_grid : np.ndarray, optional
        2D array of population density (persons per grid cell).
    vulnerability_results : optional
        Pre-computed vulnerability data.
    max_n : int
        Upper bound for N values to consider.
    source_x, source_y : float
        Source coordinates (m).
    population_total : float
        Total exposed population (used when no grid provided).
    grid_spacing : float
        Grid spacing in meters (for area calculations).

    Returns
    -------
    FNData
        FN curve data with cumulative frequencies.

    Examples
    --------
    >>> from .event_tree import Scenario, ConsequenceType
    >>> s1 = Scenario("Fire", probability=1e-4, consequence_type="pool_fire")
    >>> s2 = Scenario("Burst", probability=1e-5, consequence_type="explosion")
    >>> fn = calculate_fn_curve([s1, s2], population_total=100)
    >>> fn.max_n > 0
    True
    >>> len(fn.n_values) > 0
    True
    >>> fn.f_values[0] >= fn.f_values[-1]  # F(N) decreasing
    True
    """
    if not scenarios:
        return FNData()

    # Step 1: Calculate expected fatalities for each scenario
    scenario_n: list[tuple[str, float, float]] = []  # (name, frequency, N_fatalities)

    total_freq = 0.0
    total_expected_fatalities = 0.0

    for scenario in scenarios:
        freq = getattr(scenario, "probability", 0.0)
        name = getattr(scenario, "name", "Unknown")
        total_freq += freq

        # Calculate expected fatalities for this scenario
        n_fatalities = _calculate_scenario_fatalities(
            scenario, population_grid, vulnerability_results,
            source_x, source_y, population_total, grid_spacing,
        )

        scenario_n.append((name, freq, n_fatalities))
        total_expected_fatalities += freq * n_fatalities

    # Step 2: Build FN curve
    # Create list of (N_i, f_i) pairs, only for scenarios with N > 0
    event_pairs = [(n_fat, freq) for _, freq, n_fat in scenario_n if n_fat > 0]

    if not event_pairs:
        return FNData(
            n_values=np.array([0]),
            f_values=np.array([total_freq]),
            max_n=0,
            total_frequency=total_freq,
            expected_fatalities=0.0,
            potential_loss_of_life=0.0,
            scenario_fatalities={name: n_fat for name, _, n_fat in scenario_n},
        )

    # Sort by N descending
    event_pairs.sort(key=lambda x: x[0], reverse=True)

    n_array = np.array([n for n, _ in event_pairs])
    f_array = np.array([f for _, f in event_pairs])

    # Cumulative sum: F(N_k) = sum of f_i for all N_i >= N_k
    f_cumulative = np.cumsum(f_array)

    # Clip max_n
    mask = n_array <= max_n
    if not np.any(mask) and len(n_array) > 0:
        # If all N > max_n, still include them but flag
        mask = np.ones_like(n_array, dtype=bool)

    n_values = n_array[mask]
    f_values = f_cumulative[mask]

    # Ensure decreasing behavior: F(1) >= F(2) >= ... (already satisfied by
    # construction since cumulative sum is non-decreasing and we sort descending)

    # Scenario fatality dictionary
    scenario_fatalities = {name: n_fat for name, _, n_fat in scenario_n}

    # Step 3: Evaluate against criteria
    alarp_status = _evaluate_alarp(n_values, f_values)

    return FNData(
        n_values=n_values,
        f_values=f_values,
        max_n=int(np.max(n_values)) if len(n_values) > 0 else 0,
        total_frequency=total_freq,
        expected_fatalities=total_expected_fatalities,
        potential_loss_of_life=total_expected_fatalities,
        scenario_fatalities=scenario_fatalities,
        alarp_status=alarp_status,
    )


def fn_data_to_plot(fn_data: FNData) -> tuple[np.ndarray, np.ndarray]:
    """Extract (N, F(N)) arrays for log-log plotting.

    Parameters
    ----------
    fn_data : FNData
        FN curve data.

    Returns
    -------
    tuple of (N_array, F_array)
        Arrays suitable for matplotlib log-log plot.

    Examples
    --------
    >>> s = Scenario("test", probability=1e-4, consequence_type="explosion")
    >>> fn = calculate_fn_curve([s], population_total=10)
    >>> n, f = fn_data_to_plot(fn)
    >>> len(n) == len(f)
    True
    """
    return fn_data.n_values, fn_data.f_values


def compare_to_criterion(
    fn_data: FNData,
    criterion: FNCriterion,
) -> FNStatus:
    """Compare FN curve against a risk criterion.

    The FN curve is intolerable if ANY point on the curve exceeds the
    criterion line: F(N) > F_criterion(N).

    Parameters
    ----------
    fn_data : FNData
        FN curve data.
    criterion : FNCriterion
        FN criterion to compare against.

    Returns
    -------
    FNStatus
        ACCEPTABLE, ALARP, or INTOLERABLE.

    Examples
    --------
    >>> criterion = FN_CRITERIA["hse_uk_intolerable"]
    >>> s = Scenario("safe", probability=1e-10, consequence_type="dispersion")
    >>> fn = calculate_fn_curve([s], population_total=1)
    >>> compare_to_criterion(fn, criterion)
    <FNStatus.ACCEPTABLE: 'acceptable'>
    """
    if len(fn_data.n_values) == 0:
        return FNStatus.ACCEPTABLE

    # Check each point on the FN curve
    for n, f in zip(fn_data.n_values, fn_data.f_values):
        # Only check N >= criterion.n_min
        if n < criterion.n_min:
            continue

        f_crit = criterion.evaluate(float(n))

        # Intolerable if F(N) exceeds criterion
        if f > f_crit:
            return FNStatus.INTOLERABLE

    return FNStatus.ACCEPTABLE


def get_pll(fn_data: FNData) -> float:
    """Get Potential Loss of Life (PLL).

    PLL = Σ (f_i × N_i) = expected number of fatalities per year.

    Parameters
    ----------
    fn_data : FNData
        FN curve data.

    Returns
    -------
    float
        Potential Loss of Life (fatalities per year).
    """
    return fn_data.potential_loss_of_life


def get_far(fn_data: FNData, exposed_hours_per_year: float = 2000.0) -> float:
    """Calculate Fatal Accident Rate (FAR).

    FAR = PLL × 10⁸ / (population × exposed_hours_per_year)

    Parameters
    ----------
    fn_data : FNData
        FN curve data.
    exposed_hours_per_year : float
        Number of hours per year a person is exposed. Default 2000
        (approximately 40 hrs/week × 50 weeks).

    Returns
    -------
    float
        FAR value.
    """
    # Estimate population from scenarios
    total_population = sum(
        fn_data.scenario_fatalities.values()
    ) / max(fn_data.total_frequency, 1e-20)

    if total_population <= 0:
        return 0.0

    far = fn_data.potential_loss_of_life * 1e8 / (total_population * exposed_hours_per_year)
    return far


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _calculate_scenario_fatalities(
    scenario: Any,
    population_grid: Optional[np.ndarray],
    vulnerability_results: Any,
    source_x: float,
    source_y: float,
    population_total: float,
    grid_spacing: float,
) -> float:
    """Calculate expected fatalities for a single scenario.

    N_i = P_fatality_i × population_exposed_i

    Parameters
    ----------
    scenario : Scenario
        The scenario being evaluated.
    population_grid : np.ndarray, optional
        2D population density grid.
    vulnerability_results : optional
        Vulnerability model results.
    source_x, source_y : float
        Source coordinates.
    population_total : float
        Total exposed population.
    grid_spacing : float
        Grid spacing in meters.

    Returns
    -------
    float
        Expected number of fatalities.
    """
    freq = getattr(scenario, "probability", 0.0)

    if freq <= 0:
        return 0.0

    if population_grid is not None and hasattr(population_grid, 'shape'):
        return _calculate_fatalities_with_grid(
            scenario, population_grid, vulnerability_results,
            source_x, source_y, grid_spacing,
        )
    else:
        return _calculate_fatalities_simplified(
            scenario, population_total, source_x, source_y,
        )


def _calculate_fatalities_simplified(
    scenario: Any,
    population_total: float,
    source_x: float,
    source_y: float,
) -> float:
    """Simplified fatality calculation without population grid.

    Uses representative fatality probability at source and nearby
    distances, multiplied by population fraction.

    Parameters
    ----------
    scenario : Scenario
        Scenario object.
    population_total : float
        Total exposed population.
    source_x, source_y : float
        Source coordinates (currently unused; reserved for future use).

    Returns
    -------
    float
        Expected fatalities.
    """
    ct = str(getattr(scenario, "consequence_type", "dispersion"))
    params = getattr(scenario, "consequence_params", {})

    # Default fatality probabilities per consequence type
    # These represent the average probability of fatality for a person
    # within the hazard zone
    fatality_factors: dict[str, float] = {
        "explosion": 0.3,   # Average within explosion zone
        "bleve": 0.5,       # High lethality within fireball zone
        "pool_fire": 0.15,  # Average within pool fire zone
        "jet_fire": 0.2,    # Average within jet fire zone
        "flash_fire": 0.1,  # Average within flash fire zone
        "toxic": 0.1,       # Average within toxic zone
        "dispersion": 0.01, # Low lethality for dispersion without ignition
        "safe_dispersal": 0.0,
    }

    p_fat_avg = fatality_factors.get(ct, 0.05)

    # Fraction of population within the hazard zone (simplified)
    # Consequence models would provide the exact hazard zone extent
    hazard_fraction = params.get("population_fraction", 0.1)

    # Expected fatalities = P_fatality_avg × population_exposed
    n_fatalities = p_fat_avg * population_total * hazard_fraction

    return n_fatalities


def _calculate_fatalities_with_grid(
    scenario: Any,
    population_grid: np.ndarray,
    vulnerability_results: Any,
    source_x: float,
    source_y: float,
    grid_spacing: float,
) -> float:
    """Calculate fatalities using detailed population grid.

    For each grid cell:
    N_cell = P_fatality(x_cell, y_cell) × population_cell

    Total N = Σ N_cell
    """
    ny, nx = population_grid.shape
    total_fatalities = 0.0

    # Calculate grid coordinates
    x_center = source_x  # Assume source is at center if not specified
    y_center = source_y

    for iy in range(ny):
        for ix in range(nx):
            pop = population_grid[iy, ix]
            if pop <= 0:
                continue

            # Cell center coordinates
            cx = x_center + (ix - nx / 2) * grid_spacing
            cy = y_center + (iy - ny / 2) * grid_spacing
            distance = math.sqrt((cx - source_x) ** 2 + (cy - source_y) ** 2)

            # Get fatality probability for this cell
            if vulnerability_results is not None:
                try:
                    ct = getattr(scenario, "consequence_type", "dispersion")
                    if hasattr(vulnerability_results, "get_fatality_at_point"):
                        p_fat = vulnerability_results.get_fatality_at_point(cx, cy, ct)
                    elif hasattr(vulnerability_results, "get_fatality"):
                        raw = vulnerability_results.get_fatality(distance, ct)
                        p_fat = float(raw) if raw is not None else 0.0
                    else:
                        p_fat = _simplified_fatality_for_grid(distance, scenario)
                except Exception:
                    p_fat = _simplified_fatality_for_grid(distance, scenario)
            else:
                p_fat = _simplified_fatality_for_grid(distance, scenario)

            total_fatalities += p_fat * pop

    return total_fatalities


def _simplified_fatality_for_grid(distance: float, scenario: Any) -> float:
    """Simplified fatality probability for grid-based calculation."""
    ct = str(getattr(scenario, "consequence_type", "dispersion"))
    params = getattr(scenario, "consequence_params", {})

    # Hazard radii by type
    radii: dict[str, float] = {
        "explosion": params.get("explosion_radius", 150.0),
        "bleve": params.get("bleve_radius", 200.0),
        "pool_fire": params.get("pool_fire_radius", 50.0),
        "jet_fire": params.get("jet_fire_radius", 30.0),
        "flash_fire": params.get("flash_fire_radius", 100.0),
        "toxic": params.get("toxic_radius", 300.0),
        "dispersion": params.get("dispersion_radius", 100.0),
        "safe_dispersal": 0.0,
    }

    r_char = radii.get(ct, 50.0)

    if r_char <= 0:
        return 0.0
    if distance <= 0:
        return 1.0

    # Sigmoid decay
    k = 10.0 / r_char
    p = 1.0 / (1.0 + math.exp(k * (distance - r_char)))

    return p


def _evaluate_alarp(
    n_values: np.ndarray,
    f_values: np.ndarray,
) -> dict[str, FNStatus]:
    """Evaluate FN curve against all standard criteria.

    Returns
    -------
    dict
        Mapping of criterion name to FNStatus.
    """
    status: dict[str, FNStatus] = {}

    for name, criterion in FN_CRITERIA.items():
        # Build a temporary FNData for comparison
        temp_fn = FNData(n_values=n_values, f_values=f_values)
        status[name] = compare_to_criterion(temp_fn, criterion)

    return status


# Needed for doctest
from .event_tree import Scenario, ConsequenceType
