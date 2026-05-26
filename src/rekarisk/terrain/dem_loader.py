"""
Rekarisk Terrain — Digital Elevation Model Loader.

Foundation module for loading, querying, and analyzing digital elevation
models. Phase 8 supports CSV grid format (x, y, z columns). GeoTIFF and
HGT/DTED support deferred to a later phase.

Features:
    - Load DEM from CSV (x, y, z format)
    - Bilinear interpolation for off-grid elevation queries
    - Slope and aspect calculation
    - Contour line generation
    - Basic terrain statistics

References:
    - USGS Digital Elevation Model Standards
    - Map Algebra for terrain analysis (Tomlin, 1990)
    - Zevenbergen & Thorne (1987) — Quantitative analysis of land surface
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Lazy-import scipy to avoid binary incompatibility issues on import.
# SciPy functions are only needed when actually loading/interpolating DEM data.
_RegularGridInterpolator = None


def _get_interpolator_class():
    """Lazily import RegularGridInterpolator from scipy."""
    global _RegularGridInterpolator
    if _RegularGridInterpolator is None:
        from scipy.interpolate import RegularGridInterpolator as RGI
        _RegularGridInterpolator = RGI
    return _RegularGridInterpolator


# ══════════════════════════════════════════════════════════════════════════════
# Contour Line Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContourLine:
    """A single contour line (iso-elevation polygon).

    Attributes:
        elevation: Elevation value [m] of this contour.
        points: List of (x, y) coordinates tracing the contour.
        is_closed: Whether the contour forms a closed loop.
    """
    elevation: float
    points: List[Tuple[float, float]] = field(default_factory=list)
    is_closed: bool = True

    def to_polygon(self) -> List[List[float]]:
        """Return points as a polygon suitable for plotting/GeoJSON."""
        return [[p[0], p[1]] for p in self.points]

    def length(self) -> float:
        """Total length of the contour line [m]."""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self.points) - 1):
            dx = self.points[i + 1][0] - self.points[i][0]
            dy = self.points[i + 1][1] - self.points[i][1]
            total += np.sqrt(dx * dx + dy * dy)
        return total

    def area(self) -> float:
        """Approximate area enclosed by closed contour (Shoelace formula)."""
        if not self.is_closed or len(self.points) < 3:
            return 0.0
        pts = self.points
        area = 0.0
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# DEMData
# ══════════════════════════════════════════════════════════════════════════════

class DEMData:
    """Digital Elevation Model data loaded from CSV.

    Stores a 2D grid of elevation values and provides interpolation,
    slope calculation, and contour generation.

    The grid is assumed to be regular (uniform spacing) after loading.
    If the input CSV has irregular (scattered) points, the grid is
    constructed via interpolation during loading.

    Attributes:
        x: Array of x-coordinates [m] (length = n_cols).
        y: Array of y-coordinates [m] (length = n_rows).
        z: (n_rows, n_cols) array of elevations [m].
        cell_size: Grid spacing [m] (if uniform).
        unit: Elevation unit (always 'm').
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        name: str = "Untitled DEM",
    ):
        """Initialize DEM data directly from arrays.

        Args:
            x: (n_cols,) x-coordinate array.
            y: (n_rows,) y-coordinate array.
            z: (n_rows, n_cols) elevation array.
            name: Human-readable name for this DEM.
        """
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.name = name
        self.unit = "m"

        # Detect cell size
        self._detect_grid_properties()

        # Build interpolator
        self._interpolator = None  # : RegularGridInterpolator | None (lazy import)
        self._build_interpolator()

    @classmethod
    def from_csv(
        cls,
        filepath: str | Path,
        name: str | None = None,
        x_col: int | str = 0,
        y_col: int | str = 1,
        z_col: int | str = 2,
        delimiter: str = ",",
        skip_rows: int = 0,
        has_header: bool = False,
        grid_shape: Tuple[int, int] | None = None,
    ) -> DEMData:
        """Load a DEM from a CSV file.

        Accepts both regular grid data (with grid_shape) and scattered
        points (will build grid via interpolation).

        Args:
            filepath: Path to CSV file.
            name: DEM name (default: filename stem).
            x_col: Column index or name for x-coordinate.
            y_col: Column index or name for y-coordinate.
            z_col: Column index or name for elevation.
            delimiter: CSV delimiter character.
            skip_rows: Number of header rows to skip.
            has_header: Whether the first row contains column names.
            grid_shape: (n_rows, n_cols) for regular grid. If None,
                inferred from unique x/y values (must be regular).

        Returns:
            DEMData instance.

        Raises:
            ValueError: If CSV format is invalid or non-regular without grid_shape.
            FileNotFoundError: If filepath does not exist.
        """
        filepath = Path(filepath).expanduser().resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"DEM file not found: {filepath}")

        if name is None:
            name = filepath.stem

        # Read CSV
        all_rows: List[List[str]] = []
        with open(filepath, "r", encoding="utf-8") as f:
            # Skip header if needed
            if has_header:
                header_line = f.readline()
                header_names = [
                    h.strip().strip('"').strip("'")
                    for h in header_line.strip().split(delimiter)
                ]
                # Map column names to indices
                if isinstance(x_col, str):
                    try:
                        x_col = header_names.index(x_col)
                    except ValueError:
                        raise ValueError(f"Column '{x_col}' not found in header: {header_names}")
                if isinstance(y_col, str):
                    try:
                        y_col = header_names.index(y_col)
                    except ValueError:
                        raise ValueError(f"Column '{y_col}' not found in header: {header_names}")
                if isinstance(z_col, str):
                    try:
                        z_col = header_names.index(z_col)
                    except ValueError:
                        raise ValueError(f"Column '{z_col}' not found in header: {header_names}")

            # Skip extra rows
            for _ in range(skip_rows):
                f.readline()

            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if row:  # skip empty lines
                    all_rows.append(row)

        # Parse data
        xs = []
        ys = []
        zs = []
        for row in all_rows:
            try:
                xv = float(row[x_col])
                yv = float(row[y_col])
                zv = float(row[z_col])
                xs.append(xv)
                ys.append(yv)
                zs.append(zv)
            except (ValueError, IndexError):
                continue  # skip malformed rows

        if not xs:
            raise ValueError(f"No valid data rows found in {filepath}")

        xs_arr = np.array(xs)
        ys_arr = np.array(ys)
        zs_arr = np.array(zs)

        # Determine grid
        if grid_shape is not None:
            n_rows, n_cols = grid_shape
            expected = n_rows * n_cols
            if len(xs) < expected:
                raise ValueError(
                    f"Not enough data points: got {len(xs)}, "
                    f"expected at least {expected} for {n_rows}×{n_cols} grid"
                )
            xs_arr = xs_arr[:expected]
            ys_arr = ys_arr[:expected]
            zs_arr = zs_arr[:expected]

            x_unique = np.unique(xs_arr)
            y_unique = np.unique(ys_arr)

            if len(x_unique) != n_cols:
                raise ValueError(
                    f"Expected {n_cols} unique x-values, got {len(x_unique)}"
                )
            if len(y_unique) != n_rows:
                raise ValueError(
                    f"Expected {n_rows} unique y-values, got {len(y_unique)}"
                )

            z_grid = zs_arr.reshape(n_rows, n_cols)
            return cls(x=x_unique, y=y_unique, z=z_grid, name=name)
        else:
            # Auto-detect regular grid from unique values
            x_unique = np.unique(xs_arr)
            y_unique = np.unique(ys_arr)

            n_cols = len(x_unique)
            n_rows = len(y_unique)

            if n_rows * n_cols != len(xs) or n_cols < 2 or n_rows < 2:
                # Scattered data — build regular grid via interpolation
                return cls._from_scattered(
                    xs_arr, ys_arr, zs_arr, name,
                )

            # Regular grid — build the z matrix
            z_grid = np.zeros((n_rows, n_cols))
            for i, yv in enumerate(y_unique):
                for j, xv in enumerate(x_unique):
                    # Find matching data point
                    mask = (np.abs(xs_arr - xv) < 1e-8) & (np.abs(ys_arr - yv) < 1e-8)
                    indices = np.where(mask)[0]
                    if len(indices) > 0:
                        z_grid[i, j] = zs_arr[indices[0]]

            return cls(x=x_unique, y=y_unique, z=z_grid, name=name)

    @classmethod
    def _from_scattered(
        cls,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        name: str,
    ) -> DEMData:
        """Build a regular grid from scattered (irregular) data points.

        Uses linear interpolation on a fine grid, then optionally
        downsamples.
        """
        from scipy.interpolate import griddata

        # Determine grid bounds and resolution
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Aim for ~100 points per axis
        n_points = max(50, min(200, int(np.sqrt(len(xs)) * 3)))
        xi = np.linspace(x_min, x_max, n_points)
        yi = np.linspace(y_min, y_max, n_points)

        XX, YY = np.meshgrid(xi, yi)
        points = np.column_stack((xs, ys))

        try:
            zi = griddata(points, zs, (XX, YY), method="linear", fill_value=np.nan)
        except Exception:
            # Fallback: nearest-neighbor interpolation
            zi = griddata(points, zs, (XX, YY), method="nearest")

        return cls(x=xi, y=yi, z=zi, name=f"{name} (gridded)")

    # -- Grid properties ------------------------------------------------------

    def _detect_grid_properties(self) -> None:
        """Detect grid spacing, uniformity, and extents."""
        self.n_cols = len(self.x)
        self.n_rows = len(self.y)
        self.x_min, self.x_max = float(self.x.min()), float(self.x.max())
        self.y_min, self.y_max = float(self.y.min()), float(self.y.max())
        self.z_min, self.z_max = float(np.nanmin(self.z)), float(np.nanmax(self.z))

        if self.n_cols > 1:
            dx = np.diff(self.x)
            if np.allclose(dx, dx[0], rtol=0.01):
                self.cell_size = float(dx[0])
            else:
                self.cell_size = None  # non-uniform spacing
        else:
            self.cell_size = None

    def _build_interpolator(self) -> None:
        """Build a bilinear interpolation function from the grid."""
        try:
            RGI = _get_interpolator_class()
            self._interpolator = RGI(
                (self.y, self.x),
                self.z,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        except Exception:
            self._interpolator = None

    # -- Elevation queries ----------------------------------------------------

    def elevation(self, x: float, y: float) -> float:
        """Get elevation at (x, y) using bilinear interpolation.

        Args:
            x: X-coordinate [m].
            y: Y-coordinate [m].

        Returns:
            Elevation [m] or NaN if outside grid bounds.
        """
        # Try scipy interpolator first
        if self._interpolator is not None:
            result = self._interpolator([[y, x]])
            val = float(result[0])
            if not np.isnan(val):
                return val

        # Fallback: pure-numpy bilinear interpolation
        return self._bilinear_elevation(x, y)

    def _bilinear_elevation(self, x: float, y: float) -> float:
        """Pure-numpy bilinear interpolation fallback."""
        if not self.is_within_bounds(x, y):
            return float("nan")

        # Find surrounding grid cell indices
        col = np.searchsorted(self.x, x)
        row = np.searchsorted(self.y, y)

        # Clamp to valid range
        col = max(1, min(col, self.n_cols - 1))
        row = max(1, min(row, self.n_rows - 1))

        x0, x1 = self.x[col - 1], self.x[col]
        y0, y1 = self.y[row - 1], self.y[row]

        z00 = self.z[row - 1, col - 1]
        z10 = self.z[row - 1, col]
        z01 = self.z[row, col - 1]
        z11 = self.z[row, col]

        # Check for NaN in corner values
        if np.isnan(z00) or np.isnan(z10) or np.isnan(z01) or np.isnan(z11):
            return float("nan")

        # Bilinear interpolation
        if abs(x1 - x0) < 1e-12 or abs(y1 - y0) < 1e-12:
            return float(z00)

        tx = (x - x0) / (x1 - x0)
        ty = (y - y0) / (y1 - y0)

        z = (z00 * (1 - tx) * (1 - ty) +
             z10 * tx * (1 - ty) +
             z01 * (1 - tx) * ty +
             z11 * tx * ty)

        return float(z)

    def elevations(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Get elevations for an array of points.

        Args:
            xs: (N,) array of x-coordinates [m].
            ys: (N,) array of y-coordinates [m].

        Returns:
            (N,) array of elevations [m], NaN for out-of-bounds points.
        """
        # Try scipy interpolator first
        if self._interpolator is not None:
            points = np.column_stack((ys, xs))
            result = self._interpolator(points)
            return result

        # Fallback: per-point bilinear
        result = np.full(len(xs), np.nan)
        for i in range(len(xs)):
            result[i] = self._bilinear_elevation(xs[i], ys[i])
        return result

    def elevation_grid(self) -> np.ndarray:
        """Return the full elevation grid (n_rows, n_cols)."""
        return self.z.copy()

    def is_within_bounds(self, x: float, y: float) -> bool:
        """Check if (x, y) is within the DEM bounds."""
        return (
            self.x_min <= x <= self.x_max and
            self.y_min <= y <= self.y_max
        )

    # -- Slope and aspect -----------------------------------------------------

    def slope(
        self, x: float, y: float, method: str = "finite_difference"
    ) -> float:
        """Calculate slope at a point (tangent of slope angle).

        Uses the Zevenbergen & Thorne (1987) finite difference method
        on the 3×3 neighborhood around the query point.

        Args:
            x: X-coordinate [m].
            y: Y-coordinate [m].
            method: "finite_difference" (default) or "gradient".

        Returns:
            Slope as rise/run ratio (tan of slope angle).
            Higher values = steeper.
        """
        if not self.is_within_bounds(x, y):
            return 0.0

        # Find nearest grid cell
        col = np.searchsorted(self.x, x)
        row = np.searchsorted(self.y, y)

        if method == "gradient":
            # Use numpy gradient if cell_size is uniform
            if self.cell_size is None:
                return 0.0

            gy, gx = np.gradient(self.z, self.cell_size)
            if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
                gx_v = gx[row, col] if col < self.n_cols else gx[row, col - 1]
                gy_v = gy[row, col] if row < self.n_rows else gy[row - 1, col]
                return float(np.sqrt(gx_v ** 2 + gy_v ** 2))
            return 0.0

        else:
            # Zevenbergen & Thorne method using 3×3 neighborhood
            r_start = max(0, row - 1)
            r_end = min(self.n_rows, row + 2)
            c_start = max(0, col - 1)
            c_end = min(self.n_cols, col + 2)

            window = self.z[r_start:r_end, c_start:c_end]
            if window.size < 4:
                return 0.0

            # Compute finite differences
            if self.cell_size is not None:
                dx = self.cell_size
                dy = self.cell_size
            else:
                # Use actual coordinate differences
                dx_vals = np.diff(self.x[c_start:c_end]).mean() if c_end - c_start > 1 else 1.0
                dy_vals = np.diff(self.y[r_start:r_end]).mean() if r_end - r_start > 1 else 1.0
                dx = dx_vals if not np.isnan(dx_vals) else 1.0
                dy = dy_vals if not np.isnan(dy_vals) else 1.0

            # Central differences for slope
            center_row = (window.shape[0] - 1) // 2
            center_col = (window.shape[1] - 1) // 2

            if window.shape[0] >= 2 and window.shape[1] >= 2:
                if center_row > 0 and center_row < window.shape[0] - 1:
                    dz_dy = (window[center_row + 1, center_col] - window[center_row - 1, center_col]) / (2.0 * dy)
                else:
                    dz_dy = 0.0

                if center_col > 0 and center_col < window.shape[1] - 1:
                    dz_dx = (window[center_row, center_col + 1] - window[center_row, center_col - 1]) / (2.0 * dx)
                else:
                    dz_dx = 0.0
            else:
                dz_dx, dz_dy = 0.0, 0.0

            return float(np.sqrt(dz_dx ** 2 + dz_dy ** 2))

    def slope_degrees(self, x: float, y: float) -> float:
        """Slope in degrees [°] at (x, y)."""
        return float(np.degrees(np.arctan(self.slope(x, y))))

    def aspect(self, x: float, y: float) -> float:
        """Calculate aspect (direction of steepest descent) at (x, y).

        Returns:
            Aspect in degrees (0 = north, 90 = east, 180 = south, 270 = west).
        """
        if not self.is_within_bounds(x, y):
            return 0.0

        col = np.searchsorted(self.x, x)
        row = np.searchsorted(self.y, y)

        r_start = max(0, row - 1)
        r_end = min(self.n_rows, row + 2)
        c_start = max(0, col - 1)
        c_end = min(self.n_cols, col + 2)

        window = self.z[r_start:r_end, c_start:c_end]
        if window.size < 4:
            return 0.0

        center_row = (window.shape[0] - 1) // 2
        center_col = (window.shape[1] - 1) // 2

        if window.shape[0] >= 2 and window.shape[1] >= 2:
            if center_row > 0 and center_row < window.shape[0] - 1:
                dz_dy = (window[center_row + 1, center_col] - window[center_row - 1, center_col]) / 2.0
            else:
                dz_dy = 0.0

            if center_col > 0 and center_col < window.shape[1] - 1:
                dz_dx = (window[center_row, center_col + 1] - window[center_row, center_col - 1]) / 2.0
            else:
                dz_dx = 0.0
        else:
            dz_dx, dz_dy = 0.0, 0.0

        # Aspect: direction of steepest descent
        aspect_rad = np.arctan2(-dz_dx, -dz_dy)  # negative for descent
        aspect_deg = np.degrees(aspect_rad) % 360.0

        return float(aspect_deg)

    def slope_map(self) -> np.ndarray:
        """Compute slope (rise/run) for every grid cell.

        Returns:
            (n_rows, n_cols) array of slope values.
        """
        if self.cell_size is None:
            # Non-uniform grid — compute per-cell
            result = np.zeros_like(self.z)
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    result[i, j] = self.slope(self.x[j], self.y[i])
            return result

        gy, gx = np.gradient(self.z, self.cell_size)
        return np.sqrt(gx ** 2 + gy ** 2)

    # -- Contour generation ---------------------------------------------------

    def generate_contours(
        self, n_levels: int = 10, levels: List[float] | None = None,
    ) -> List[ContourLine]:
        """Generate contour lines at specified elevation levels.

        Uses matplotlib's contour algorithm (if available) or a
        simple marching-squares fallback.

        Args:
            n_levels: Number of contour levels (if levels not specified).
            levels: Explicit list of elevation levels for contours.

        Returns:
            List of ContourLine objects.
        """
        try:
            import matplotlib.pyplot as plt
            contour_set = plt.contour(
                self.x, self.y, self.z,
                levels=levels or n_levels,
            )
            plt.close()

            result = []
            for level_idx, segs in enumerate(contour_set.allsegs):
                elev = contour_set.levels[level_idx]
                for seg in segs:
                    points = [(float(p[0]), float(p[1])) for p in seg]
                    # Check if closed (first == last)
                    is_closed = (
                        len(points) >= 3 and
                        np.hypot(
                            points[0][0] - points[-1][0],
                            points[0][1] - points[-1][1],
                        ) < 1e-6
                    )
                    contour = ContourLine(
                        elevation=float(elev),
                        points=points,
                        is_closed=is_closed,
                    )
                    result.append(contour)

            return result

        except ImportError:
            # Fallback: simple marching squares
            return self._marching_squares_contours(n_levels, levels)

    def _marching_squares_contours(
        self, n_levels: int = 10, levels: List[float] | None = None,
    ) -> List[ContourLine]:
        """Simple marching squares contour extraction (no matplotlib)."""
        if levels is None:
            z_range = self.z_max - self.z_min
            if z_range < 1e-9:
                return []
            levels = np.linspace(
                self.z_min + z_range * 0.05,
                self.z_max - z_range * 0.05,
                n_levels,
            ).tolist()

        contours: List[ContourLine] = []

        for elev in levels:
            contour_points = []
            visited = set()

            for i in range(self.n_rows - 1):
                for j in range(self.n_cols - 1):
                    if (i, j) in visited:
                        continue

                    # Cell corners
                    z00 = self.z[i, j]
                    z10 = self.z[i, j + 1]
                    z11 = self.z[i + 1, j + 1]
                    z01 = self.z[i + 1, j]

                    # Build case index (4-bit)
                    case = 0
                    if z00 >= elev:
                        case |= 1
                    if z10 >= elev:
                        case |= 2
                    if z11 >= elev:
                        case |= 4
                    if z01 >= elev:
                        case |= 8

                    if case == 0 or case == 15:
                        continue  # no contour crossing

                    # Compute intersection points on edges
                    def interp(p1, p2, v1, v2):
                        """Linear interpolation of edge crossing."""
                        if abs(v2 - v1) < 1e-9:
                            return (p1 + p2) / 2.0
                        t = (elev - v1) / (v2 - v1)
                        return p1 + t * (p2 - p1)

                    x0 = self.x[j]
                    x1 = self.x[j + 1]
                    y0 = self.y[i]
                    y1 = self.y[i + 1]

                    # Bottom edge
                    if (z00 >= elev) != (z10 >= elev):
                        px = interp(x0, x1, z00, z10)
                        contour_points.append((px, y0))
                    # Right edge
                    if (z10 >= elev) != (z11 >= elev):
                        py = interp(y0, y1, z10, z11)
                        contour_points.append((x1, py))
                    # Top edge
                    if (z01 >= elev) != (z11 >= elev):
                        px = interp(x0, x1, z01, z11)
                        contour_points.append((px, y1))
                    # Left edge
                    if (z00 >= elev) != (z01 >= elev):
                        py = interp(y0, y1, z00, z01)
                        contour_points.append((x0, py))

                    visited.add((i, j))

            if contour_points:
                contours.append(ContourLine(
                    elevation=float(elev),
                    points=contour_points,
                    is_closed=False,
                ))

        return contours

    # -- Statistics -----------------------------------------------------------

    def stats(self) -> dict:
        """Compute summary statistics for the DEM.

        Returns:
            Dict with min, max, mean, std, range, and cell count.
        """
        valid_z = self.z[~np.isnan(self.z)]
        return {
            "name": self.name,
            "n_cells": f"{self.n_rows} × {self.n_cols}",
            "total_cells": int(self.n_rows * self.n_cols),
            "cell_size_m": self.cell_size,
            "bounds_x": [self.x_min, self.x_max],
            "bounds_y": [self.y_min, self.y_max],
            "z_min_m": float(np.nanmin(valid_z)) if len(valid_z) > 0 else None,
            "z_max_m": float(np.nanmax(valid_z)) if len(valid_z) > 0 else None,
            "z_mean_m": float(np.nanmean(valid_z)) if len(valid_z) > 0 else None,
            "z_std_m": float(np.nanstd(valid_z)) if len(valid_z) > 0 else None,
            "z_range_m": float(np.nanmax(valid_z) - np.nanmin(valid_z)) if len(valid_z) > 0 else None,
        }

    def to_dict(self) -> dict:
        """Serialize DEM metadata (not full grid data) to dict."""
        return {
            "name": self.name,
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "z_min": self.z_min,
            "z_max": self.z_max,
            "cell_size": self.cell_size,
            "unit": self.unit,
        }

    def to_json(self, path: str | Path) -> None:
        """Export DEM metadata to JSON."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    # -- Dunder ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DEMData(name='{self.name}', "
            f"shape=({self.n_rows},{self.n_cols}), "
            f"z_range=[{self.z_min:.1f}, {self.z_max:.1f}] m)"
        )

    def __len__(self) -> int:
        return int(self.n_rows * self.n_cols)
