"""
Rekarisk Meteorology — Wind Rose Data.

Wind rose data is a binned joint probability distribution of wind speed
and direction, typically presented as a polar frequency plot.

Default binning: 16 directions (N, NNE, NE, ENE, E, ...) × 6 speed bins.

References:
    - U.S. EPA AERMOD Implementation Guide
    - WMO Guide to Meteorological Instruments and Methods of Observation
"""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard 16 cardinal directions
DIRECTION_NAMES: Tuple[str, ...] = (
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
)

# Number of direction sectors
N_DIRECTIONS = 16

# Direction sector width (degrees)
SECTOR_WIDTH_DEG = 360.0 / N_DIRECTIONS  # 22.5°

# Default speed class upper bounds (m/s)
DEFAULT_SPEED_BINS: Tuple[float, ...] = (
    1.0,    # Calm / very light
    2.5,    # Light
    5.0,    # Moderate
    8.0,    # Fresh
    12.0,   # Strong
    20.0,   # Very strong / gale+
)

# Labels for speed classes
DEFAULT_SPEED_LABELS: Tuple[str, ...] = (
    "<1 m/s",
    "1-2.5 m/s",
    "2.5-5 m/s",
    "5-8 m/s",
    "8-12 m/s",
    ">12 m/s",
)

# Calm wind threshold [m/s]
CALM_THRESHOLD_MS = 0.5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def direction_index(angle_deg: float) -> int:
    """Convert a wind direction angle (from north, clockwise) to sector index.

    Sector 0 = N (centered at 0°/360°), Sector 1 = NNE, etc.

    Args:
        angle_deg: Wind direction in degrees (0-360, meteorological convention).

    Returns:
        Sector index 0-15.
    """
    half_width = SECTOR_WIDTH_DEG / 2.0
    return int((angle_deg + half_width) / SECTOR_WIDTH_DEG) % N_DIRECTIONS


def speed_class_index(speed_ms: float, bins: Sequence[float]) -> int:
    """Convert a wind speed to speed class index.

    Args:
        speed_ms: Wind speed [m/s].
        bins: Upper bounds of each speed class [m/s].

    Returns:
        Speed class index (0 to len(bins)-1). Values exceeding all bins
        go to the last class.
    """
    for i, upper in enumerate(bins):
        if speed_ms <= upper:
            return i
    return len(bins) - 1


def direction_angle_from_index(index: int) -> float:
    """Return the center angle (degrees from north) for a direction sector.

    Args:
        index: Sector index 0-15 (0 = N).

    Returns:
        Center angle in degrees.
    """
    return (index * SECTOR_WIDTH_DEG) % 360.0


def direction_name_from_angle(angle_deg: float) -> str:
    """Convert a wind direction angle to the nearest cardinal direction name.

    Args:
        angle_deg: Wind direction in degrees (0-360).

    Returns:
        Direction name string such as "N", "NNE", "NW".
    """
    idx = direction_index(angle_deg)
    return DIRECTION_NAMES[idx]


# ---------------------------------------------------------------------------
# WindRoseData
# ---------------------------------------------------------------------------


class WindRoseData:
    """Binned wind speed × wind direction frequency data.

    Represents a complete wind rose: a matrix of probabilities (or counts)
    for each combination of direction sector and speed class.

    Attributes:
        n_directions: Number of direction sectors (default 16).
        speed_bins: Upper bounds of speed classes [m/s].
        speed_labels: Labels for speed classes.
        calm_threshold: Wind speed threshold for "calm" [m/s].
    """

    def __init__(
        self,
        n_directions: int = N_DIRECTIONS,
        speed_bins: Optional[Sequence[float]] = None,
        speed_labels: Optional[Sequence[str]] = None,
        calm_threshold: float = CALM_THRESHOLD_MS,
    ):
        """Initialize WindRoseData with empty frequency matrix.

        Args:
            n_directions: Number of direction sectors.
            speed_bins: Upper bounds of speed bins [m/s].
            speed_labels: Labels for speed bins.
            calm_threshold: Calm wind threshold [m/s].
        """
        self.n_directions = n_directions
        self.speed_bins = tuple(speed_bins) if speed_bins else DEFAULT_SPEED_BINS
        self.speed_labels = tuple(speed_labels) if speed_labels else DEFAULT_SPEED_LABELS[:len(self.speed_bins)]
        self.calm_threshold = calm_threshold

        # Count matrix: rows = speed classes, columns = directions
        self._counts: np.ndarray = np.zeros(
            (len(self.speed_bins), self.n_directions), dtype=float
        )

        # Additional calm fraction (not assigned to any direction)
        self._calm_count: float = 0.0

        # Total observations
        self._total: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def counts(self) -> np.ndarray:
        """Raw count matrix (speed × direction)."""
        return self._counts.copy()

    @property
    def total(self) -> float:
        """Total number of observations (including calm)."""
        return self._total

    @property
    def calm_count(self) -> float:
        """Number of calm observations."""
        return self._calm_count

    # ------------------------------------------------------------------
    # Data entry
    # ------------------------------------------------------------------

    def set_count(self, direction_index: int, speed_class_index: int, count: float) -> None:
        """Set the count for a specific direction and speed class.

        Args:
            direction_index: Direction sector index (0 to n_directions-1).
            speed_class_index: Speed class index (0 to len(speed_bins)-1).
            count: Number of observations.
        """
        self._counts[speed_class_index, direction_index] = max(0, count)

    def set_frequencies(
        self,
        matrix: np.ndarray,
        total: float = 1.0,
        calm_fraction: float = 0.0,
    ) -> None:
        """Set frequencies from a probability matrix.

        Args:
            matrix: (n_speed, n_directions) array of probabilities.
            total: Total number of observations represented.
            calm_fraction: Fraction of calm winds [0-1].
        """
        if matrix.shape != (len(self.speed_bins), self.n_directions):
            raise ValueError(
                f"Matrix shape {matrix.shape} doesn't match "
                f"({len(self.speed_bins)}, {self.n_directions})"
            )
        self._total = total
        self._calm_count = calm_fraction * total
        self._counts = np.maximum(matrix * total * (1 - calm_fraction), 0.0)

    def set_probabilities(
        self,
        prob_matrix: np.ndarray,
        calm_probability: float = 0.0,
    ) -> None:
        """Set directly from a probability matrix (no scaling).

        Args:
            prob_matrix: (n_speed, n_directions) array of probabilities.
                Must sum to 1 - calm_probability.
            calm_probability: Probability of calm winds.
        """
        total_prob = prob_matrix.sum()
        if not (0.99 < total_prob + calm_probability < 1.01):
            # Normalize
            if total_prob > 0:
                prob_matrix = prob_matrix / total_prob * (1.0 - calm_probability)

        self._total = 1.0
        self._calm_count = calm_probability
        self._counts = prob_matrix

    def add_observation(self, speed_ms: float, direction_deg: float) -> None:
        """Add a single wind observation.

        Args:
            speed_ms: Wind speed [m/s].
            direction_deg: Wind direction [degrees from north].
        """
        self._total += 1.0
        if speed_ms < self.calm_threshold:
            self._calm_count += 1.0
        else:
            di = direction_index(direction_deg)
            si = speed_class_index(speed_ms, self.speed_bins)
            self._counts[si, di] += 1.0

    def add_observations(
        self,
        speeds_ms: Sequence[float],
        directions_deg: Sequence[float],
    ) -> None:
        """Add multiple observations from arrays.

        Args:
            speeds_ms: Array of wind speeds [m/s].
            directions_deg: Array of wind directions [degrees].
        """
        if len(speeds_ms) != len(directions_deg):
            raise ValueError(
                f"Length mismatch: {len(speeds_ms)} speeds vs "
                f"{len(directions_deg)} directions"
            )

        for speed, direction in zip(speeds_ms, directions_deg):
            self.add_observation(speed, direction)

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def probability(self, direction: int, speed_class: int) -> float:
        """Get probability for a given direction and speed class.

        Args:
            direction: Direction index (0 to n_directions-1).
            speed_class: Speed class index.

        Returns:
            Probability [0-1].
        """
        if self._total <= 0:
            return 0.0
        return self._counts[speed_class, direction] / self._total

    def direction_probability(self, direction: int) -> float:
        """Get total probability for a given direction (all speeds).

        Args:
            direction: Direction index.

        Returns:
            Probability [0-1].
        """
        if self._total <= 0:
            return 0.0
        return self._counts[:, direction].sum() / self._total

    def speed_class_probability(self, speed_class: int) -> float:
        """Get total probability for a given speed class (all directions).

        Args:
            speed_class: Speed class index.

        Returns:
            Probability [0-1].
        """
        if self._total <= 0:
            return 0.0
        return self._counts[speed_class, :].sum() / self._total

    def calm_fraction(self) -> float:
        """Return fraction of calm wind observations [0-1]."""
        if self._total <= 0:
            return 0.0
        return self._calm_count / self._total

    def dominant_direction(self) -> int:
        """Return the direction index with the highest total probability.

        Returns:
            Direction index (0-15). Returns 0 if no data.
        """
        if self._total <= 0:
            return 0
        dir_counts = self._counts.sum(axis=0)
        return int(np.argmax(dir_counts))

    def dominant_direction_name(self) -> str:
        """Return the name of the dominant wind direction."""
        idx = self.dominant_direction()
        return DIRECTION_NAMES[idx % N_DIRECTIONS]

    def mean_wind_speed(self) -> float:
        """Estimate mean wind speed from binned data [m/s].

        Uses bin midpoints weighted by frequency.
        """
        if self._total <= 0:
            return 0.0

        # Bin midpoints
        bin_edges = [0.0] + list(self.speed_bins)
        midpoints = [
            (bin_edges[i] + bin_edges[i + 1]) / 2
            for i in range(len(self.speed_bins))
        ]
        # For last bin, use 1.2x upper bound as midpoint estimate
        midpoints[-1] = self.speed_bins[-1] * 1.2

        total_count = self._counts.sum()
        if total_count <= 0:
            return 0.0

        weighted_sum = sum(
            midpoints[i] * self._counts[i, :].sum()
            for i in range(len(self.speed_bins))
        )
        return weighted_sum / total_count

    def conditional_probability(
        self,
        direction: Optional[int] = None,
        speed_class: Optional[int] = None,
    ) -> np.ndarray:
        """Calculate conditional probability distribution.

        If direction is given: P(speed_class | direction) — all speed classes for that direction.
        If speed_class is given: P(direction | speed_class) — all directions for that speed class.

        Args:
            direction: Condition on direction index (optional).
            speed_class: Condition on speed class index (optional).

        Returns:
            1D array of conditional probabilities.

        Raises:
            ValueError: If both or neither condition is given.
        """
        if direction is not None and speed_class is not None:
            raise ValueError("Provide only ONE condition, not both")
        if direction is None and speed_class is None:
            raise ValueError("Provide exactly one condition (direction or speed_class)")

        if self._total <= 0:
            return np.zeros(
                self.n_directions if speed_class is not None else len(self.speed_bins)
            )

        if direction is not None:
            row = self._counts[:, direction]
            total = row.sum()
            if total <= 0:
                return np.zeros(len(self.speed_bins))
            return row / total
        else:
            col = self._counts[speed_class, :]
            total = col.sum()
            if total <= 0:
                return np.zeros(self.n_directions)
            return col / total

    def joint_probability_distribution(self) -> np.ndarray:
        """Return the full joint probability matrix (speed × direction).

        Returns:
            (n_speed, n_direction) array of joint probabilities.
        """
        if self._total <= 0:
            return np.zeros((len(self.speed_bins), self.n_directions))
        return self._counts / self._total

    def direction_frequencies(self) -> np.ndarray:
        """Return probability for each direction sector (summing over speeds).

        Returns:
            1D array of length n_directions with probability per sector.
        """
        if self._total <= 0:
            return np.zeros(self.n_directions)
        return self._counts.sum(axis=0) / self._total

    def speed_frequencies(self) -> np.ndarray:
        """Return probability for each speed class (summing over directions).

        Returns:
            1D array of length len(speed_bins).
        """
        if self._total <= 0:
            return np.zeros(len(self.speed_bins))
        return self._counts.sum(axis=1) / self._total

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize wind rose to dictionary for JSON save."""
        return {
            "n_directions": self.n_directions,
            "speed_bins": list(self.speed_bins),
            "speed_labels": list(self.speed_labels),
            "calm_threshold": self.calm_threshold,
            "counts": self._counts.tolist(),
            "calm_count": self._calm_count,
            "total": self._total,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WindRoseData":
        """Deserialize from dictionary."""
        wr = cls(
            n_directions=d.get("n_directions", N_DIRECTIONS),
            speed_bins=tuple(d.get("speed_bins", DEFAULT_SPEED_BINS)),
            speed_labels=tuple(d.get("speed_labels", DEFAULT_SPEED_LABELS)),
            calm_threshold=d.get("calm_threshold", CALM_THRESHOLD_MS),
        )
        wr._counts = np.array(d.get("counts", [[0]]), dtype=float)
        wr._calm_count = d.get("calm_count", 0.0)
        wr._total = d.get("total", 0.0)
        return wr

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "WindRoseData":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_csv(self) -> str:
        """Export wind rose data to CSV string.

        Format: rows = speed classes, columns = directions.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["Speed Class"] + list(DIRECTION_NAMES[:self.n_directions]))

        # Data rows
        for i in range(len(self.speed_bins)):
            label = self.speed_labels[i] if i < len(self.speed_labels) else f"Bin {i}"
            row = [label] + [self._counts[i, j] for j in range(self.n_directions)]
            writer.writerow(row)

        # Summary row
        writer.writerow([])
        writer.writerow(["Calm count", self._calm_count])
        writer.writerow(["Total", self._total])

        return output.getvalue()

    def from_csv(
        self,
        csv_text: str,
        has_header: bool = True,
    ) -> None:
        """Load wind rose data from CSV string.

        Expected format: rows = speed classes, columns = directions.

        Args:
            csv_text: CSV-formatted string.
            has_header: Whether first row is a header.
        """
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)

        if has_header:
            # Skip header row
            start_row = 1
        else:
            start_row = 0

        # Read data rows
        speed_row_idx = 0
        for row in rows[start_row:]:
            # Skip empty or blank rows
            if not row or all(c.strip() == "" for c in row):
                continue

            first = row[0].strip() if row[0] else ""

            # Check for summary rows
            if first.lower().startswith("calm"):
                if len(row) > 1:
                    try:
                        self._calm_count = float(row[1])
                    except (ValueError, IndexError):
                        pass
                continue
            elif first.lower().startswith("total"):
                if len(row) > 1:
                    try:
                        self._total = float(row[1])
                    except (ValueError, IndexError):
                        pass
                continue

            if speed_row_idx >= len(self.speed_bins):
                break

            for j, val in enumerate(row[1:]):
                if j >= self.n_directions:
                    break
                try:
                    self._counts[speed_row_idx, j] = float(val)
                except (ValueError, IndexError):
                    pass

            speed_row_idx += 1

        # Recalculate total if not explicitly provided
        if self._total <= 0:
            self._total = self._counts.sum() + self._calm_count

    def summary(self) -> str:
        """Return a human-readable summary of the wind rose."""
        dom_idx = self.dominant_direction()
        dom_name = DIRECTION_NAMES[dom_idx % N_DIRECTIONS]
        lines = [
            f"Wind Rose Summary",
            f"{'─' * 40}",
            f"Total observations: {self._total:.0f}",
            f"Calm fraction:      {self.calm_fraction():.3f} ({self._calm_count:.0f})",
            f"Dominant direction: {dom_name} ({direction_angle_from_index(dom_idx):.0f}°)",
            f"Mean wind speed:    {self.mean_wind_speed():.2f} m/s",
            f"",
            f"Direction probabilities:",
        ]
        for i in range(self.n_directions):
            p = self.direction_probability(i)
            bar = "█" * int(p * 50)
            lines.append(
                f"  {DIRECTION_NAMES[i]:>4s}: {p:.4f} {bar}"
            )
        lines.append("")
        lines.append("Speed class probabilities:")
        for i in range(len(self.speed_bins)):
            p = self.speed_class_probability(i)
            label = self.speed_labels[i] if i < len(self.speed_labels) else f"Bin {i}"
            lines.append(f"  {label:>12s}: {p:.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"WindRoseData(n_directions={self.n_directions}, "
            f"speed_bins={self.speed_bins}, total={self._total:.0f})"
        )
