"""
Rekarisk — Flash Fire Model.

Determines the flammable cloud envelope from dispersion results and
calculates flash fire consequences: LFL/UFL contours, flammable area,
and simplified thermal radiation for personnel within/adjacent to
the flammable cloud.

References:
  - CCPS Guidelines for Consequence Analysis of Chemical Releases (1999)
  - TNO Yellow Book (CPR 14E), Chapter 5 — Flash Fires
  - AIChE/CCPS (1994) — Guidelines for Evaluating the Characteristics of
    Vapor Cloud Explosions, Flash Fires, and BLEVEs
  - HSE UK — Methods of Approximation and Determination of Human
    Vulnerability for Offshore Major Accident Hazard Assessment
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...core.constants import T_0C, EPSILON, AIR_MOLECULAR_WEIGHT


# ══════════════════════════════════════════════════════════════════════════════
# Input/Output Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FlashFireInput:
    """Input parameters for flash fire calculation.

    Attributes:
        dispersion_result: Result from a dispersion model. Expected to
            have concentration data (2D array or contour data).
            Built to work with PlumeResult / PuffResult.
        lfl: Lower Flammable Limit [vol%]. Default for generic HC: 1.4%.
            If None, auto-select from substance database.
        ufl: Upper Flammable Limit [vol%]. Default for generic HC: 7.4%.
            If None, auto-select from substance database.
        substance: Substance name. Used for radiation properties.
        heat_of_combustion: Lower heating value [J/kg]. Auto if None.
        sep_flash: Surface emissive power for flash fire [kW/m²].
            Typical: 150-200 kW/m². Default 173 kW/m² (CCPS).
        mode: "lfl_contour" or "from_grid". "lfl_contour" expects
            2D grid data; "from_grid" converts from concentration field.
    """
    dispersion_result: Optional[object] = None
    lfl: Optional[float] = None
    ufl: Optional[float] = None
    substance: str = "default"
    heat_of_combustion: Optional[float] = None
    sep_flash: float = 173.0  # kW/m²
    mode: str = "lfl_contour"

    def __post_init__(self):
        # Default LFL/UFL for common substances
        DEFAULT_FLAMMABILITY: Dict[str, Tuple[float, float]] = {
            "methane": (5.0, 15.0),
            "ethane": (3.0, 12.5),
            "propane": (2.1, 9.5),
            "butane": (1.8, 8.4),
            "pentane": (1.4, 7.8),
            "hexane": (1.2, 7.4),
            "heptane": (1.1, 6.7),
            "octane": (0.95, 6.5),
            "gasoline": (1.4, 7.6),
            "petrol": (1.4, 7.6),
            "kerosene": (0.7, 5.0),
            "diesel": (0.6, 5.0),
            "benzene": (1.3, 7.9),
            "toluene": (1.2, 7.1),
            "xylene": (1.1, 7.0),
            "methanol": (6.7, 36.0),
            "ethanol": (3.3, 19.0),
            "hydrogen": (4.0, 75.0),
            "ethylene": (2.7, 36.0),
            "propylene": (2.4, 10.3),
            "ammonia": (15.0, 28.0),
            "lpg": (2.1, 9.5),
            "lng": (5.0, 15.0),
            "default": (1.4, 7.4),
        }

        if self.lfl is None or self.ufl is None:
            lfl_d, ufl_d = DEFAULT_FLAMMABILITY.get(
                self.substance.lower(), DEFAULT_FLAMMABILITY["default"]
            )
            if self.lfl is None:
                self.lfl = lfl_d
            if self.ufl is None:
                self.ufl = ufl_d

        if self.heat_of_combustion is None:
            FLASH_HEATS = {
                "methane": 50.0e6, "propane": 46.3e6, "butane": 45.7e6,
                "pentane": 45.4e6, "hexane": 45.1e6, "heptane": 44.9e6,
                "octane": 44.4e6, "gasoline": 44.0e6, "kerosene": 43.5e6,
                "diesel": 43.0e6, "benzene": 40.1e6, "toluene": 40.6e6,
                "xylene": 40.9e6, "methanol": 19.9e6, "ethanol": 26.8e6,
                "hydrogen": 120.0e6, "ethylene": 47.2e6,
                "propylene": 45.8e6, "ammonia": 18.6e6,
                "lpg": 46.0e6, "lng": 50.0e6,
                "default": 45.0e6,
            }
            self.heat_of_combustion = FLASH_HEATS.get(
                self.substance.lower(), FLASH_HEATS["default"]
            )


@dataclass
class FlashFireResult:
    """Results from flash fire calculation.

    Attributes:
        lfl_contour: (N,2) array of (x,y) points defining the LFL boundary
            [m]. Points are ordered around the contour.
        ufl_contour: (N,2) array of (x,y) points defining the UFL boundary
            [m]. May be empty if UFL is very close to source.
        area_within_lfl: Area inside the LFL contour [m²].
        area_within_ufl: Area inside the UFL contour [m²].
        max_distance_to_lfl: Maximum radial distance to LFL contour [m].
        thermal_radiation_vs_distance: (N,2) array for simplified radiation
            if cloud is assumed to burn.
        status_messages: List of info/warning strings.
    """
    lfl_contour: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2))
    )
    ufl_contour: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2))
    )
    area_within_lfl: float = 0.0
    area_within_ufl: float = 0.0
    max_distance_to_lfl: float = 0.0
    thermal_radiation_vs_distance: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2))
    )
    status_messages: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Atmospheric Transmissivity (shared utility)
# ══════════════════════════════════════════════════════════════════════════════

def atmospheric_transmissivity(
    distance: float,
    ambient_temperature: float = 298.15,
    relative_humidity: float = 50.0,
) -> float:
    """Atmospheric transmissivity for thermal radiation.

    Args:
        distance: Path length [m].
        ambient_temperature: Ambient temperature [K].
        relative_humidity: Relative humidity [%].

    Returns:
        Transmissivity [-] (0 to 1).
    """
    if distance < EPSILON:
        return 1.0

    T_C = ambient_temperature - T_0C
    p_sat = 610.78 * math.exp(17.2694 * T_C / (T_C + 237.3))
    p_w = (relative_humidity / 100.0) * p_sat
    kappa = 2.02e-5 * (max(p_w, 1.0) ** 0.09)
    tau = math.exp(-kappa * distance)

    return min(max(tau, 0.01), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# LFL/UFL Contour Extraction
# ══════════════════════════════════════════════════════════════════════════════

def find_lfl_contour(
    concentration_field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    lfl: float,
) -> np.ndarray:
    """Extract LFL isopleth contour from a 2D concentration field.

    Uses marching squares (via matplotlib) to find the contour line
    at the LFL concentration level. Falls back to a simple radial
    search if the grid is radial.

    Args:
        concentration_field: 2D array of concentration [vol% or kg/m³].
        x_grid: 1D or 2D array of x-coordinates [m].
        y_grid: 1D or 2D array of y-coordinates [m].
        lfl: LFL concentration in same units as concentration_field.

    Returns:
        (N, 2) array of (x, y) contour points [m], or empty if no contour.
    """
    if concentration_field.size == 0:
        return np.zeros((0, 2))

    # Ensure 2D arrays
    if concentration_field.ndim == 1:
        concentration_field = concentration_field.reshape(1, -1)

    # Check if any value is above LFL
    if np.max(concentration_field) <= lfl:
        return np.zeros((0, 2))

    try:
        from matplotlib import pyplot as plt
        # Use matplotlib's contour algorithm
        # Ensure x_grid and y_grid are 2D
        if x_grid.ndim == 1:
            X, Y = np.meshgrid(x_grid, y_grid)
        else:
            X, Y = x_grid, y_grid

        # Find contour at LFL level
        contour = plt.contour(X, Y, concentration_field, levels=[lfl])
        plt.close('all')

        # Extract path vertices
        paths = contour.collections[0].get_paths() if contour.collections else []
        if not paths:
            return np.zeros((0, 2))

        all_points = []
        for path in paths:
            vertices = path.vertices
            if len(vertices) > 2:
                all_points.append(vertices)

        if all_points:
            return np.vstack(all_points)
        return np.zeros((0, 2))

    except (ImportError, Exception):
        # Fallback: radial approximation
        return _find_lfl_contour_radial(concentration_field, x_grid, y_grid, lfl)


def find_ufl_contour(
    concentration_field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    ufl: float,
) -> np.ndarray:
    """Extract UFL isopleth contour from a 2D concentration field.

    Identical algorithm to LFL contour but at UFL threshold.
    The UFL contour encloses the inner rich zone near the source.

    Args:
        concentration_field: 2D array of concentration [vol% or kg/m³].
        x_grid: 1D or 2D array of x-coordinates [m].
        y_grid: 1D or 2D array of y-coordinates [m].
        ufl: UFL concentration in same units as concentration_field.

    Returns:
        (N, 2) array of (x, y) contour points [m], or empty if no contour.
    """
    if concentration_field.size == 0:
        return np.zeros((0, 2))

    if np.max(concentration_field) <= ufl:
        return np.zeros((0, 2))

    try:
        from matplotlib import pyplot as plt
        if x_grid.ndim == 1:
            X, Y = np.meshgrid(x_grid, y_grid)
        else:
            X, Y = x_grid, y_grid

        contour = plt.contour(X, Y, concentration_field, levels=[ufl])
        plt.close('all')

        paths = contour.collections[0].get_paths() if contour.collections else []
        if not paths:
            return np.zeros((0, 2))

        all_points = []
        for path in paths:
            vertices = path.vertices
            if len(vertices) > 2:
                all_points.append(vertices)

        if all_points:
            return np.vstack(all_points)
        return np.zeros((0, 2))

    except (ImportError, Exception):
        return _find_lfl_contour_radial(concentration_field, x_grid, y_grid, ufl)


def _find_lfl_contour_radial(
    concentration_field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Radial fallback: find where concentration crosses threshold.

    For centerline-dominated dispersion, the LFL contour is approximately
    an ellipse centered on the plume centerline.

    Args:
        concentration_field: 2D concentration array.
        x_grid: 1D x-coordinates.
        y_grid: 1D y-coordinates.
        threshold: Concentration threshold.

    Returns:
        (N, 2) array of contour points.
    """
    if x_grid.ndim == 1:
        # Find max concentration at each downwind distance
        nx = len(x_grid)
        ny = len(y_grid)

        contour_pts = []
        for i in range(nx - 1):
            row = concentration_field[i, :] if concentration_field.ndim == 2 else concentration_field
            if concentration_field.ndim == 2:
                # Find crosswind extent where C > threshold
                above = row > threshold
                if np.any(above):
                    indices = np.where(above)[0]
                    y_left = y_grid[indices[0]]
                    y_right = y_grid[indices[-1]]
                    x_val = x_grid[i]

                    contour_pts.append([x_val, y_left])
                    contour_pts.append([x_val, y_right])

        if contour_pts:
            return np.array(contour_pts)

    return np.zeros((0, 2))


# ══════════════════════════════════════════════════════════════════════════════
# LFL Area and Max Distance
# ══════════════════════════════════════════════════════════════════════════════

def lfl_area(lfl_contour: np.ndarray) -> float:
    """Calculate area enclosed by the LFL contour using the shoelace formula.

    Args:
        lfl_contour: (N, 2) array of (x, y) contour points.

    Returns:
        Area [m²].
    """
    if lfl_contour.shape[0] < 3:
        return 0.0

    x = lfl_contour[:, 0]
    y = lfl_contour[:, 1]

    # Shoelace formula
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return float(area)


def max_lfl_distance(lfl_contour: np.ndarray, source_x: float = 0.0, source_y: float = 0.0) -> float:
    """Calculate the maximum radial distance from source to LFL contour.

    Args:
        lfl_contour: (N, 2) array of (x, y) contour points.
        source_x: X-coordinate of release source [m].
        source_y: Y-coordinate of release source [m].

    Returns:
        Maximum distance to LFL contour [m].
    """
    if lfl_contour.shape[0] < 1:
        return 0.0

    x = lfl_contour[:, 0]
    y = lfl_contour[:, 1]

    distances = np.sqrt((x - source_x) ** 2 + (y - source_y) ** 2)

    return float(np.max(distances))


# ══════════════════════════════════════════════════════════════════════════════
# Flash Fire Thermal Radiation (Simplified)
# ══════════════════════════════════════════════════════════════════════════════

def flash_fire_thermal_radiation(
    sep_flash: float,
    lfl_contour: np.ndarray,
    source_x: float = 0.0,
    source_y: float = 0.0,
    ambient_temperature: float = 298.15,
    relative_humidity: float = 50.0,
    min_distance: float = 1.0,
    max_distance: float = 500.0,
    n_points: int = 100,
) -> np.ndarray:
    """Calculate thermal radiation from flash fire (simplified).

    Assumes uniform burning of the entire flammable cloud at ground level.
    The SEP for flash fire is typically 150-200 kW/m² for short durations
    (~10-20 seconds).

    Simplified model: treat as a finite-area planar source at ground
    level using the LFL contour as the radiating area boundary.

    For radiation at distance d:
        q = τ · SEP · F

    where F is the view factor from the LFL-contour-bounded area to a
    vertical receiver at distance d.

    Args:
        sep_flash: Surface emissive power [kW/m²].
        lfl_contour: (N, 2) array — boundary of flammable area.
        source_x: Source x [m].
        source_y: Source y [m].
        ambient_temperature: Ambient temperature [K].
        relative_humidity: Relative humidity [%].
        min_distance: Min receiver distance [m].
        max_distance: Max receiver distance [m].
        n_points: Number of evaluation points.

    Returns:
        (N, 2) array — [distance_m, flux_kW_per_m2].
    """
    # If no contour, use a simplified model with a virtual cloud
    if lfl_contour.shape[0] < 3:
        # Fallback: point source approximation
        # Assume cloud has effective diameter of 10 m
        distances = np.linspace(min_distance, max_distance, n_points)
        fluxes = np.zeros(n_points)

        for i, d in enumerate(distances):
            tau = atmospheric_transmissivity(d, ambient_temperature, relative_humidity)
            # Point source with virtual area
            virtual_radius = 5.0  # m
            virtual_height = 2.0  # m
            area = math.pi * virtual_radius ** 2
            F = area / (4.0 * math.pi * d ** 2)
            fluxes[i] = tau * sep_flash * F

        return np.column_stack((distances, fluxes))

    # With contour: approximate as an equivalent circular area
    x = lfl_contour[:, 0]
    y = lfl_contour[:, 1]

    # Find bounding box
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Equivalent radius
    area = lfl_area(lfl_contour)
    equiv_radius = math.sqrt(area / math.pi) if area > 0 else 5.0

    # Center of the flammable area
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    distances = np.linspace(min_distance, max_distance, n_points)
    fluxes = np.zeros(n_points)

    for i, d in enumerate(distances):
        tau = atmospheric_transmissivity(d, ambient_temperature, relative_humidity)

        # View factor from circular area to vertical receiver at distance d
        # F = (1 + (d/R)²)^(-1) * cos(θ)
        # where θ is angle from horizontal plane
        if area <= 0:
            F = 0.0
        else:
            # Disk source to receiver
            # Distance from edge of cloud (simplified)
            effective_distance = max(d - equiv_radius, EPSILON)

            # View factor for a finite disk
            # F ≈ A / (4π · d²) for distant; for near: F → 1
            h_ratio = equiv_radius / max(effective_distance, EPSILON)
            F = 1.0 - 1.0 / math.sqrt(1.0 + h_ratio ** 2)

            # Small cloud correction: F cannot exceed ~0.5 for ground-level
            F = min(F, 0.5)

        fluxes[i] = tau * sep_flash * F

    return np.column_stack((distances, fluxes))


# ══════════════════════════════════════════════════════════════════════════════
# Main Calculation
# ══════════════════════════════════════════════════════════════════════════════

def calculate_flash_fire(
    input_data: FlashFireInput,
    concentration_field: Optional[np.ndarray] = None,
    x_grid: Optional[np.ndarray] = None,
    y_grid: Optional[np.ndarray] = None,
    min_distance: float = 1.0,
    max_distance: float = 500.0,
    n_points: int = 100,
) -> FlashFireResult:
    """Calculate flash fire extent and effects.

    Determines the LFL and UFL contours from dispersion results,
    computes the flammable area, and estimates simplified thermal
    radiation.

    The function accepts two input modes:
    1. From dispersion_result object (via FlashFireInput.dispersion_result):
       Extract concentration grid from the result object.
    2. From explicit arrays: concentration_field, x_grid, y_grid.

    Args:
        input_data: FlashFireInput with LFL/UFL and substance properties.
        concentration_field: 2D concentration array [vol% or kg/m³].
            Optional if dispersion_result is provided.
        x_grid: 1D/2D x-coordinates [m].
        y_grid: 1D/2D y-coordinates [m].
        min_distance: Min distance for radiation curve [m].
        max_distance: Max distance for radiation curve [m].
        n_points: Number of evaluation points.

    Returns:
        FlashFireResult with contour data, area, distance, and radiation.
    """
    messages = []
    lfl = input_data.lfl or 1.4
    ufl = input_data.ufl or 7.4

    # Try to extract concentration field from dispersion result
    if concentration_field is None and input_data.dispersion_result is not None:
        disp_result = input_data.dispersion_result
        # Extract from PlumeResult / PuffResult
        if hasattr(disp_result, 'concentration_field') and disp_result.concentration_field is not None:
            concentration_field = np.array(disp_result.concentration_field)
        elif hasattr(disp_result, 'concentration_2d'):
            concentration_field = np.array(disp_result.concentration_2d)

        if hasattr(disp_result, 'x_grid'):
            x_grid = np.array(disp_result.x_grid)
        elif hasattr(disp_result, 'x'):
            x_grid = np.array(disp_result.x)

        if hasattr(disp_result, 'y_grid'):
            y_grid = np.array(disp_result.y_grid)
        elif hasattr(disp_result, 'y'):
            y_grid = np.array(disp_result.y)

    # Validate we have data
    if concentration_field is None or x_grid is None or y_grid is None:
        return FlashFireResult(
            status_messages=["No concentration data provided. "
                            "Pass concentration_field, x_grid, y_grid arrays or "
                            "a dispersion result with concentration information."],
        )

    concentration_field = np.asarray(concentration_field, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    if concentration_field.size == 0:
        return FlashFireResult(
            status_messages=["Empty concentration field."],
        )

    # Find LFL contour
    lfl_contour = find_lfl_contour(concentration_field, x_grid, y_grid, lfl)

    if lfl_contour.shape[0] < 3:
        messages.append("LFL contour not found — concentrations may be below LFL everywhere")
    else:
        messages.append(f"LFL contour found: {lfl_contour.shape[0]} points")

    # Find UFL contour
    ufl_contour = find_ufl_contour(concentration_field, x_grid, y_grid, ufl)

    if ufl_contour.shape[0] < 3:
        messages.append(f"UFL contour not found — concentration may not reach UFL ({ufl}%)")
    else:
        messages.append(f"UFL contour found: {ufl_contour.shape[0]} points")

    # Compute area within LFL
    area_lfl = lfl_area(lfl_contour)
    area_ufl = lfl_area(ufl_contour)

    # Max distance to LFL
    max_dist = max_lfl_distance(lfl_contour)

    # Thermal radiation
    rad_vs_dist = flash_fire_thermal_radiation(
        sep_flash=input_data.sep_flash,
        lfl_contour=lfl_contour,
        ambient_temperature=298.15,
        relative_humidity=50.0,
        min_distance=min_distance,
        max_distance=max_distance,
        n_points=n_points,
    )

    if area_lfl > 10000:
        messages.append(f"Large flammable area ({area_lfl:.0f} m²) — consider detailed modeling")

    return FlashFireResult(
        lfl_contour=lfl_contour,
        ufl_contour=ufl_contour,
        area_within_lfl=round(area_lfl, 2),
        area_within_ufl=round(area_ufl, 2),
        max_distance_to_lfl=round(max_dist, 2),
        thermal_radiation_vs_distance=rad_vs_dist,
        status_messages=messages,
    )
