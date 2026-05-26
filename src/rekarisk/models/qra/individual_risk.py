"""
Rekarisk QRA — Individual Risk (IRPA) Calculation.

Individual Risk Per Annum (IRPA) — the probability that an average
unprotected person, permanently present at a certain location, is
killed due to an accident resulting from a hazardous activity.

IR(x,y) = Σ (f_i × P_fatality,i(x,y))

where:
  f_i = frequency of scenario i (per year)
  P_fatality,i(x,y) = probability of fatality at receptor (x,y)
                       given scenario i occurs

References:
  - TNO Purple Book CPR 18E — Guidelines for QRA
  - CCPS/AIChE — Guidelines for CPQRA (2nd ed.)
  - HSE UK — Risk Criteria for Land-Use Planning
  - NORSOK Z-013 — Risk and Emergency Preparedness Assessment
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import interpolate


# ──────────────────────────────────────────────────────────────────────
# Risk thresholds (standard references)
# ──────────────────────────────────────────────────────────────────────

RISK_THRESHOLDS: dict[str, dict[str, float]] = {
    "hse_uk": {
        "negligible": 1e-6,       # Broadly acceptable (public)
        "acceptable": 1e-5,       # Acceptable for public
        "tolerable_worker": 1e-4, # Tolerable for workers (ALARP upper)
        "tolerable_public": 1e-5, # Tolerable for public (ALARP upper)
        "intolerable_worker": 1e-3, # Intolerable even for workers
        "intolerable_public": 1e-4, # Intolerable for public
    },
    "tno_dutch": {
        "negligible": 1e-8,       # Negligible risk
        "acceptable_existing": 1e-5,  # Acceptable for existing plants
        "acceptable_new": 1e-6,       # Acceptable for new plants
        "unacceptable": 1e-5,         # Unacceptable (new plants)
    },
    "norsok": {
        "negligible": 1e-6,
        "acceptable": 1e-5,
        "tolerable": 1e-4,
        "intolerable": 1e-3,
    },
    "ccps": {
        "negligible": 1e-6,
        "acceptable": 1e-5,
        "tolerable": 1e-4,
        "intolerable": 1e-3,
    },
}

# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class IndividualRiskResult:
    """Individual risk calculation result.

    Attributes
    ----------
    ir_grid : np.ndarray
        2D array of IR values (per year).
    x_coords : np.ndarray
        X coordinates of grid points (m).
    y_coords : np.ndarray
        Y coordinates of grid points (m).
    max_ir : float
        Maximum IR value in the grid.
    threshold_distances : dict[str, float]
        Distance to each IR threshold contour from source.
    scenarios_contribution : dict[str, float]
        Fractional contribution of each scenario to total IR.
    """
    ir_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    x_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    y_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    max_ir: float = 0.0
    threshold_distances: dict[str, float] = field(default_factory=dict)
    scenarios_contribution: dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────

def calculate_ir_at_point(
    x: float,
    y: float,
    scenarios: list[Any],
    vulnerability_results: Any = None,
    source_x: float = 0.0,
    source_y: float = 0.0,
) -> float:
    """Calculate Individual Risk at a specific receptor point.

    IR(x,y) = Σ f_i × P_fatality,i(d)

    where d = distance from source to receptor (x,y).

    Parameters
    ----------
    x : float
        Receptor X coordinate (m).
    y : float
        Receptor Y coordinate (m).
    scenarios : list
        List of Scenario objects with 'probability' (frequency) and
        'consequence_type' attributes.
    vulnerability_results : optional
        Pre-computed vulnerability data. If None, a simplified
        distance-based fatality probability model is used.
    source_x : float
        Source X coordinate (m).
    source_y : float
        Source Y coordinate (m).

    Returns
    -------
    float
        Individual Risk at (x,y) in fatalities per year.

    Examples
    --------
    >>> from .event_tree import Scenario, ConsequenceType
    >>> from dataclasses import dataclass
    >>> s = Scenario("test", probability=1e-5, consequence_type="explosion")
    >>> ir = calculate_ir_at_point(0, 0, [s])
    >>> ir > 0
    True
    """
    total_ir = 0.0
    distance = math.sqrt((x - source_x) ** 2 + (y - source_y) ** 2)

    for scenario in scenarios:
        freq = getattr(scenario, "probability", 0.0)

        if vulnerability_results is not None:
            # Use vulnerability model results if available
            p_fatality = _lookup_fatality_probability(
                x, y, scenario, vulnerability_results, source_x, source_y,
            )
        else:
            # Simplified distance-based model
            p_fatality = _simplified_fatality_probability(
                distance, getattr(scenario, "consequence_type", "dispersion"),
                getattr(scenario, "consequence_params", {}),
            )

        total_ir += freq * p_fatality

    return total_ir


def calculate_ir_grid(
    scenarios: list[Any],
    vulnerability_results: Any = None,
    x_range: Optional[tuple[float, float, int]] = None,
    y_range: Optional[tuple[float, float, int]] = None,
    source_x: float = 0.0,
    source_y: float = 0.0,
    grid_spacing: float = 10.0,
) -> IndividualRiskResult:
    """Calculate Individual Risk over a 2D grid.

    Parameters
    ----------
    scenarios : list
        List of Scenario objects.
    vulnerability_results : optional
        Pre-computed vulnerability data.
    x_range : (min, max, n_points), optional
        X-axis grid specification. Auto-computed if None.
    y_range : (min, max, n_points), optional
        Y-axis grid specification. Auto-computed if None.
    source_x : float
        Source X coordinate (m).
    source_y : float
        Source Y coordinate (m).
    grid_spacing : float
        Grid cell spacing in meters (when ranges not specified).

    Returns
    -------
    IndividualRiskResult
        Grid of IR values with metadata.
    """
    # Auto-determine grid bounds
    if x_range is None:
        # Default: ±500 m from source with 10 m spacing
        x_min, x_max, nx = source_x - 500, source_x + 500, 101
    else:
        x_min, x_max, nx = x_range

    if y_range is None:
        y_min, y_max, ny = source_y - 500, source_y + 500, 101
    else:
        y_min, y_max, ny = y_range

    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)

    ir_grid = np.zeros((ny, nx))
    max_ir = 0.0

    # Calculate IR at each grid point
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            ir = calculate_ir_at_point(
                x, y, scenarios, vulnerability_results, source_x, source_y,
            )
            ir_grid[i, j] = ir
            if ir > max_ir:
                max_ir = ir

    # Calculate threshold distances
    threshold_distances = _calculate_threshold_distances(
        max_ir, ir_grid, x_coords, y_coords, source_x, source_y,
    )

    # Calculate scenario contributions (at source point for max IR)
    scenarios_contribution = _calculate_scenario_contributions(
        scenarios, vulnerability_results, source_x, source_y,
    )

    return IndividualRiskResult(
        ir_grid=ir_grid,
        x_coords=x_coords,
        y_coords=y_coords,
        max_ir=max_ir,
        threshold_distances=threshold_distances,
        scenarios_contribution=scenarios_contribution,
    )


def ir_contour(
    result: IndividualRiskResult,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract contour data at a specific IR threshold.

    Uses matplotlib contour algorithm via scipy interpolation.

    Parameters
    ----------
    result : IndividualRiskResult
        IR grid result.
    threshold : float
        IR threshold value (per year).

    Returns
    -------
    tuple of (X, Y, Z) arrays ready for plotting.
        X, Y: meshgrid arrays, Z: binary mask (1.0 where IR >= threshold).
    """
    X, Y = np.meshgrid(result.x_coords, result.y_coords)
    Z = result.ir_grid

    try:
        # Find the contour at the threshold level
        cs = _find_contour_level(result.x_coords, result.y_coords, Z, threshold)
        return X, Y, cs
    except Exception:
        # Fallback: binary mask
        cs = (Z >= threshold).astype(np.float64)
        return X, Y, cs


def get_ir_at_source(result: IndividualRiskResult, source_x: float = 0.0, source_y: float = 0.0) -> float:
    """Extract IR value at the source location from grid result."""
    # Find nearest grid point to source
    ix = np.argmin(np.abs(result.x_coords - source_x))
    iy = np.argmin(np.abs(result.y_coords - source_y))
    return float(result.ir_grid[iy, ix])


def get_ir_at_distance(
    result: IndividualRiskResult,
    distance: float,
    source_x: float = 0.0,
    source_y: float = 0.0,
    direction: str = "east",
) -> float:
    """Extract IR value at a specified distance from source.

    Parameters
    ----------
    result : IndividualRiskResult
        IR grid result.
    distance : float
        Distance from source (m).
    source_x, source_y : float
        Source coordinates.
    direction : str
        Direction from source: 'east', 'west', 'north', 'south'.

    Returns
    -------
    float
        IR value at the specified distance.
    """
    offsets = {"east": (distance, 0), "west": (-distance, 0),
                "north": (0, distance), "south": (0, -distance)}
    dx, dy = offsets.get(direction, (distance, 0))

    x = source_x + dx
    y = source_y + dy

    ix = np.argmin(np.abs(result.x_coords - x))
    iy = np.argmin(np.abs(result.y_coords - y))

    return float(result.ir_grid[iy, ix])


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _simplified_fatality_probability(
    distance: float,
    consequence_type: str,
    params: dict,
) -> float:
    """Simplified fatality probability based on distance.

    Uses generic distance-decay functions for different hazard types.
    This is a placeholder for when detailed vulnerability results
    are not available.

    Parameters
    ----------
    distance : float
        Distance from source (m).
    consequence_type : str
        Type of consequence.
    params : dict
        Additional parameters.

    Returns
    -------
    float
        Fatality probability (0 to 1).
    """
    ct = str(consequence_type).lower()

    # Characteristic hazard radii (m) for different consequence types
    # These are generic; real values come from consequence models
    hazard_radii: dict[str, float] = {
        "explosion": params.get("explosion_radius", 150.0),
        "bleve": params.get("bleve_radius", 200.0),
        "pool_fire": params.get("pool_fire_radius", 50.0),
        "jet_fire": params.get("jet_fire_radius", 30.0),
        "flash_fire": params.get("flash_fire_radius", 100.0),
        "toxic": params.get("toxic_radius", 300.0),
        "dispersion": params.get("dispersion_radius", 100.0),
    }

    r_char = hazard_radii.get(ct, 50.0)

    # Sigmoid decay function: P_fatality = 1 / (1 + exp(k * (d - r_char)))
    # k controls steepness; steeper = closer to step function
    k = params.get("probit_k", 10.0 / r_char)

    if distance <= 0:
        return 1.0

    # Logistic function
    p = 1.0 / (1.0 + math.exp(k * (distance - r_char)))

    # Low-probability tail correction
    # Ensure negligible at very large distances
    if distance > 3 * r_char:
        p *= math.exp(-(distance - 3 * r_char) / r_char)

    return max(0.0, min(1.0, p))


def _lookup_fatality_probability(
    x: float,
    y: float,
    scenario: Any,
    vulnerability_results: Any,
    source_x: float,
    source_y: float,
) -> float:
    """Look up fatality probability from vulnerability model results.

    Parameters
    ----------
    x, y : float
        Receptor coordinates.
    scenario : Scenario
        The scenario being evaluated.
    vulnerability_results : VulnerabilityResults-like object
        Must support a method to query fatality probability at a point.
    source_x, source_y : float
        Source coordinates for distance calculation.

    Returns
    -------
    float
        Fatality probability (0 to 1).
    """
    distance = math.sqrt((x - source_x) ** 2 + (y - source_y) ** 2)
    ct = getattr(scenario, "consequence_type", "dispersion")

    # Try to use vulnerability results method
    try:
        if hasattr(vulnerability_results, "get_fatality_at_point"):
            return vulnerability_results.get_fatality_at_point(x, y, ct)
        elif hasattr(vulnerability_results, "get_fatality"):
            # 1D fatality vs distance
            p = vulnerability_results.get_fatality(distance, ct)
            if isinstance(p, (float, int)):
                return float(p)
    except Exception:
        pass

    # Fallback to simplified model
    return _simplified_fatality_probability(
        distance, ct, getattr(scenario, "consequence_params", {}),
    )


def _calculate_threshold_distances(
    max_ir: float,
    ir_grid: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    source_x: float,
    source_y: float,
) -> dict[str, float]:
    """Calculate distance to each IR threshold along cardinal directions.

    Returns
    -------
    dict
        Mapping of threshold label to distance in meters.
        e.g., {"1e-4/yr": 150.0, "1e-5/yr": 300.0, "1e-6/yr": 500.0}
    """
    thresholds = [1e-3, 1e-4, 1e-5, 1e-6]
    threshold_labels = ["1e-3/yr", "1e-4/yr", "1e-5/yr", "1e-6/yr"]

    result: dict[str, float] = {}

    # Find distances eastward from source
    src_ix = np.argmin(np.abs(x_coords - source_x))
    src_iy = np.argmin(np.abs(y_coords - source_y))

    for thresh, label in zip(thresholds, threshold_labels):
        if max_ir < thresh:
            result[label] = 0.0
            continue

        # Search eastward
        found = False
        for j in range(src_ix, len(x_coords)):
            if ir_grid[src_iy, j] < thresh:
                # Interpolate for more precise distance
                if j > src_ix and ir_grid[src_iy, j - 1] >= thresh:
                    d = _interpolate_distance(
                        x_coords[j - 1], ir_grid[src_iy, j - 1],
                        x_coords[j], ir_grid[src_iy, j],
                        thresh,
                    )
                    result[label] = d - source_x
                else:
                    result[label] = x_coords[j] - source_x
                found = True
                break
        if not found:
            result[label] = x_coords[-1] - source_x

    return result


def _interpolate_distance(
    x1: float, v1: float,
    x2: float, v2: float,
    target: float,
) -> float:
    """Linear interpolation to find distance at threshold crossing."""
    if abs(v1 - v2) < 1e-20:
        return x1
    return x1 + (target - v1) * (x2 - x1) / (v2 - v1)


def _calculate_scenario_contributions(
    scenarios: list[Any],
    vulnerability_results: Any,
    source_x: float,
    source_y: float,
) -> dict[str, float]:
    """Calculate each scenario's fractional contribution to IR at source.

    Returns
    -------
    dict
        Mapping of scenario name to fractional contribution (0 to 1).
    """
    contributions: dict[str, float] = {}
    total = 0.0

    for scenario in scenarios:
        freq = getattr(scenario, "probability", 0.0)
        name = getattr(scenario, "name", "Unknown")
        p_fat = _lookup_fatality_probability(
            source_x, source_y, scenario, vulnerability_results,
            source_x, source_y,
        )
        ir_contrib = freq * p_fat
        contributions[name] = ir_contrib
        total += ir_contrib

    if total > 0:
        contributions = {k: v / total for k, v in contributions.items()}

    return contributions


def _find_contour_level(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z: np.ndarray,
    level: float,
) -> np.ndarray:
    """Find contour for a specific level using marching squares approach.

    Returns a binary mask where z >= level.
    """
    return (z >= level).astype(np.float64)
