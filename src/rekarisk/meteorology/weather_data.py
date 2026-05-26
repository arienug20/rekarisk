"""
Rekarisk Meteorology — Weather Observation Data.

Provides data structures for working with meteorological time-series data,
including import from CSV, filtering, statistics, and joint probability
distribution computation.

Typical use case: loading a year of hourly met station data and extracting
statistics or wind rose probabilities for dispersion modeling.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .stability import StabilityClass, classify_stability
from .wind_rose import (
    DEFAULT_SPEED_BINS,
    DIRECTION_NAMES,
    CALM_THRESHOLD_MS,
    WindRoseData,
)


# ---------------------------------------------------------------------------
# WeatherObservation
# ---------------------------------------------------------------------------


@dataclass
class WeatherObservation:
    """A single weather observation at a point in time.

    All fields are optional except timestamp. If solar_radiation and
    cloud_cover are both None, stability_class must be explicitly provided.

    Attributes:
        timestamp: Observation date/time.
        wind_speed_ms: Wind speed at 10m [m/s].
        wind_direction_deg: Wind direction from north [degrees].
        temperature_k: Air temperature [K].
        pressure_pa: Atmospheric pressure [Pa].
        cloud_cover_oktas: Cloud cover [oktas, 0-8].
        solar_radiation_wm2: Solar radiation [W/m²].
        relative_humidity_pct: Relative humidity [%].
        precipitation_mm: Precipitation [mm].
        stability_class: Computed or manually set PG stability class.
        station_id: Weather station identifier.
        is_daytime: Whether observation is during daytime.
    """

    timestamp: datetime
    wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    temperature_k: Optional[float] = None
    pressure_pa: Optional[float] = None
    cloud_cover_oktas: Optional[float] = None
    solar_radiation_wm2: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    precipitation_mm: Optional[float] = None
    stability_class: Optional[StabilityClass] = None
    station_id: Optional[str] = None
    is_daytime: bool = True

    def __post_init__(self) -> None:
        """Auto-classify stability if not explicitly set and data is available."""
        if self.stability_class is None:
            if self.wind_speed_ms is not None:
                self.stability_class = classify_stability(
                    wind_speed_ms=self.wind_speed_ms,
                    solar_radiation=self.solar_radiation_wm2 if self.is_daytime else None,
                    cloud_cover_oktas=self.cloud_cover_oktas if not self.is_daytime else None,
                    is_daytime=self.is_daytime,
                )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "wind_speed_ms": self.wind_speed_ms,
            "wind_direction_deg": self.wind_direction_deg,
            "temperature_k": self.temperature_k,
            "pressure_pa": self.pressure_pa,
            "cloud_cover_oktas": self.cloud_cover_oktas,
            "solar_radiation_wm2": self.solar_radiation_wm2,
            "relative_humidity_pct": self.relative_humidity_pct,
            "precipitation_mm": self.precipitation_mm,
            "stability_class": self.stability_class,
            "station_id": self.station_id,
            "is_daytime": self.is_daytime,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeatherObservation":
        """Deserialize from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(d["timestamp"]),
            wind_speed_ms=d.get("wind_speed_ms"),
            wind_direction_deg=d.get("wind_direction_deg"),
            temperature_k=d.get("temperature_k"),
            pressure_pa=d.get("pressure_pa"),
            cloud_cover_oktas=d.get("cloud_cover_oktas"),
            solar_radiation_wm2=d.get("solar_radiation_wm2"),
            relative_humidity_pct=d.get("relative_humidity_pct"),
            precipitation_mm=d.get("precipitation_mm"),
            stability_class=d.get("stability_class"),
            station_id=d.get("station_id"),
            is_daytime=d.get("is_daytime", True),
        )

    def __repr__(self) -> str:
        return (
            f"WeatherObservation({self.timestamp.isoformat()}, "
            f"ws={self.wind_speed_ms} m/s, wd={self.wind_direction_deg}°, "
            f"stab={self.stability_class})"
        )


# ---------------------------------------------------------------------------
# WeatherDataset
# ---------------------------------------------------------------------------


class WeatherDataset:
    """Collection of weather observations as a time series.

    Provides statistical analysis, filtering, and conversion to wind rose
    data for use in dispersion calculations.
    """

    def __init__(self, observations: Optional[List[WeatherObservation]] = None):
        """Initialize dataset with optional list of observations.

        Args:
            observations: Initial list of observations.
        """
        self._observations: List[WeatherObservation] = list(observations) if observations else []
        # Sort by timestamp
        self._observations.sort(key=lambda o: o.timestamp)

    def __len__(self) -> int:
        return len(self._observations)

    def __iter__(self) -> Iterator[WeatherObservation]:
        return iter(self._observations)

    def __getitem__(self, index: int) -> WeatherObservation:
        return self._observations[index]

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    def add(self, obs: WeatherObservation) -> None:
        """Add a single observation, maintaining time order."""
        self._observations.append(obs)
        self._observations.sort(key=lambda o: o.timestamp)

    def add_all(self, observations: List[WeatherObservation]) -> None:
        """Add multiple observations."""
        self._observations.extend(observations)
        self._observations.sort(key=lambda o: o.timestamp)

    @property
    def observations(self) -> List[WeatherObservation]:
        """Return copy of observation list."""
        return list(self._observations)

    def clear(self) -> None:
        """Remove all observations."""
        self._observations.clear()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_date(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> "WeatherDataset":
        """Filter observations within a date range.

        Args:
            start: Start datetime (inclusive). None = unbounded start.
            end: End datetime (inclusive). None = unbounded end.

        Returns:
            New WeatherDataset with filtered observations.
        """
        filtered = []
        for obs in self._observations:
            if start is not None and obs.timestamp < start:
                continue
            if end is not None and obs.timestamp > end:
                continue
            filtered.append(obs)
        return WeatherDataset(filtered)

    def filter_by_time_of_day(
        self,
        start_time: Optional[time] = None,
        end_time: Optional[time] = None,
    ) -> "WeatherDataset":
        """Filter observations by time of day.

        Args:
            start_time: Start time (inclusive). None = 00:00.
            end_time: End time (inclusive). None = 23:59.

        Returns:
            New WeatherDataset.
        """
        filtered = []
        for obs in self._observations:
            t = obs.timestamp.time()
            if start_time is not None and t < start_time:
                continue
            if end_time is not None and t > end_time:
                continue
            filtered.append(obs)
        return WeatherDataset(filtered)

    def filter_by_daytime(self, is_daytime: bool = True) -> "WeatherDataset":
        """Filter observations by daytime/nighttime.

        Args:
            is_daytime: True for daytime, False for nighttime.

        Returns:
            New WeatherDataset.
        """
        return WeatherDataset([
            obs for obs in self._observations
            if obs.is_daytime == is_daytime
        ])

    def filter_by_stability(
        self,
        stability: StabilityClass,
    ) -> "WeatherDataset":
        """Filter observations by stability class.

        Args:
            stability: PG stability class (A-F).

        Returns:
            New WeatherDataset.
        """
        return WeatherDataset([
            obs for obs in self._observations
            if obs.stability_class == stability
        ])

    def filter_by_season(
        self,
        month_start: int = 1,
        month_end: int = 12,
    ) -> "WeatherDataset":
        """Filter observations by month range (seasonal).

        Args:
            month_start: Start month (1-12, inclusive).
            month_end: End month (1-12, inclusive).

        Returns:
            New WeatherDataset.
        """
        filtered = []
        for obs in self._observations:
            m = obs.timestamp.month
            if month_start <= month_end:
                if month_start <= m <= month_end:
                    filtered.append(obs)
            else:
                # Wrap-around season (e.g., Nov-Feb)
                if m >= month_start or m <= month_end:
                    filtered.append(obs)
        return WeatherDataset(filtered)

    def filter_by_wind_speed(
        self,
        min_speed: float = 0.0,
        max_speed: float = float("inf"),
    ) -> "WeatherDataset":
        """Filter by wind speed range.

        Args:
            min_speed: Minimum wind speed [m/s].
            max_speed: Maximum wind speed [m/s].

        Returns:
            New WeatherDataset.
        """
        return WeatherDataset([
            obs for obs in self._observations
            if obs.wind_speed_ms is not None
            and min_speed <= obs.wind_speed_ms <= max_speed
        ])

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _extract_field(self, field_name: str) -> np.ndarray:
        """Extract a field as numpy array, skipping None values."""
        values = [getattr(obs, field_name) for obs in self._observations]
        valid = [v for v in values if v is not None]
        return np.array(valid, dtype=float)

    def stats(self) -> Dict[str, Dict[str, float]]:
        """Compute basic statistics for all numeric fields.

        Returns:
            Dict of field_name → {mean, min, max, std, count}.
        """
        fields = [
            "wind_speed_ms",
            "wind_direction_deg",
            "temperature_k",
            "pressure_pa",
            "cloud_cover_oktas",
            "solar_radiation_wm2",
            "relative_humidity_pct",
            "precipitation_mm",
        ]

        result = {}
        for field in fields:
            arr = self._extract_field(field)
            if len(arr) > 0:
                result[field] = {
                    "mean": float(np.mean(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "count": len(arr),
                }
            else:
                result[field] = {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "std": 0.0,
                    "count": 0,
                }

        # Stability distribution
        stab_counts: Dict[str, int] = {}
        for obs in self._observations:
            if obs.stability_class:
                sc = obs.stability_class
                stab_counts[sc] = stab_counts.get(sc, 0) + 1
        total = len(self._observations)
        result["stability_distribution"] = {
            sc: count / total if total > 0 else 0.0
            for sc, count in stab_counts.items()
        }

        return result

    def mean_wind_speed(self) -> float:
        """Compute mean wind speed [m/s]."""
        arr = self._extract_field("wind_speed_ms")
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    def mean_temperature(self) -> float:
        """Compute mean temperature [K]."""
        arr = self._extract_field("temperature_k")
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    def stability_distribution(self) -> Dict[StabilityClass, float]:
        """Return probability distribution of stability classes.

        Returns:
            Dict mapping stability class to probability.
        """
        counts: Dict[str, int] = {}
        for obs in self._observations:
            if obs.stability_class:
                sc = obs.stability_class
                counts[sc] = counts.get(sc, 0) + 1

        total = max(sum(counts.values()), 1)
        return {sc: cnt / total for sc, cnt in counts.items()}  # type: ignore[misc]

    def dominant_stability(self) -> Optional[StabilityClass]:
        """Return the most frequent stability class."""
        dist = self.stability_distribution()
        if not dist:
            return None
        return max(dist, key=lambda k: dist[k])  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Wind Rose
    # ------------------------------------------------------------------

    def to_wind_rose(
        self,
        speed_bins: Optional[Sequence[float]] = None,
        calm_threshold: float = CALM_THRESHOLD_MS,
    ) -> WindRoseData:
        """Convert observations to a WindRoseData object.

        Args:
            speed_bins: Upper bounds of speed classes [m/s].
            calm_threshold: Calm wind threshold [m/s].

        Returns:
            WindRoseData populated from observations.
        """
        if speed_bins is None:
            speed_bins = DEFAULT_SPEED_BINS

        wr = WindRoseData(
            speed_bins=speed_bins,
            calm_threshold=calm_threshold,
        )

        valid = 0
        for obs in self._observations:
            if obs.wind_speed_ms is not None and obs.wind_direction_deg is not None:
                wr.add_observation(obs.wind_speed_ms, obs.wind_direction_deg)
                valid += 1

        return wr

    def joint_probability_distribution(
        self,
        n_directions: int = 16,
        speed_bins: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, float]:
        """Compute joint wind speed/direction probability distribution.

        Args:
            n_directions: Number of direction sectors.
            speed_bins: Speed class upper bounds [m/s].

        Returns:
            Tuple of (probability matrix (len(speed_bins) x n_directions),
            calm_probability).
        """
        wr = self.to_wind_rose(speed_bins=speed_bins)
        return wr.joint_probability_distribution(), wr.calm_fraction()

    # ------------------------------------------------------------------
    # Diurnal / Seasonal breakdown
    # ------------------------------------------------------------------

    def diurnal_breakdown(self) -> Dict[str, "WeatherDataset"]:
        """Split dataset into daytime and nighttime subsets.

        Returns:
            Dict with keys 'daytime' and 'nighttime'.
        """
        return {
            "daytime": self.filter_by_daytime(True),
            "nighttime": self.filter_by_daytime(False),
        }

    def seasonal_breakdown(self) -> Dict[str, "WeatherDataset"]:
        """Split dataset into four seasons (meteorological).

        Returns:
            Dict with keys 'DJF' (Dec-Feb), 'MAM' (Mar-May),
            'JJA' (Jun-Aug), 'SON' (Sep-Nov).
        """
        return {
            "DJF": self.filter_by_season(12, 2),
            "MAM": self.filter_by_season(3, 5),
            "JJA": self.filter_by_season(6, 8),
            "SON": self.filter_by_season(9, 11),
        }

    def hourly_breakdown(self) -> Dict[int, "WeatherDataset"]:
        """Split dataset by hour of day (0-23).

        Returns:
            Dict mapping hour to WeatherDataset.
        """
        result: Dict[int, List[WeatherObservation]] = {h: [] for h in range(24)}
        for obs in self._observations:
            result[obs.timestamp.hour].append(obs)
        return {h: WeatherDataset(obs_list) for h, obs_list in result.items()}

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        csv_text: str,
        column_mapping: Optional[Dict[str, str]] = None,
        has_header: bool = True,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
        timestamp_columns: Optional[List[str]] = None,
    ) -> "WeatherDataset":
        """Load weather observations from CSV.

        Supports automatic column detection for common formats:
            - 'timestamp' or 'date' / 'datetime'
            - 'wind_speed', 'ws', 'wind_speed_ms'
            - 'wind_direction', 'wd', 'wind_dir', 'wind_direction_deg'
            - 'temperature', 'temp', 'T', 'temperature_k'
            - 'pressure', 'P', 'pressure_pa'
            - 'cloud_cover', 'cloud', 'cloud_cover_oktas'
            - 'solar_radiation', 'solar', 'GHI', 'solar_radiation_wm2'
            - 'humidity', 'RH', 'relative_humidity_pct'
            - 'precipitation', 'precip', 'rain', 'precipitation_mm'

        Args:
            csv_text: CSV-formatted string.
            column_mapping: Optional manual mapping of column names to fields.
                e.g., {'WindSpeed': 'wind_speed_ms', 'Temp': 'temperature_k'}.
            has_header: Whether CSV has a header row.
            timestamp_format: strptime format for parsing timestamps.
            timestamp_columns: List of column names to concatenate for timestamp
                (e.g., ['date', 'time']). If not given, looks for a single
                timestamp column.

        Returns:
            WeatherDataset with parsed observations.
        """
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)

        if not rows:
            return cls()

        # Parse header
        header: List[str] = []
        data_start = 0
        if has_header:
            header = [h.strip().lower() for h in rows[0]]
            data_start = 1

        # Auto-detect column mapping if not provided
        if column_mapping is None:
            column_mapping = cls._auto_detect_columns(header)

        # Find timestamp column(s)
        ts_cols = timestamp_columns or cls._find_timestamp_columns(header)

        observations = []
        for row in rows[data_start:]:
            if not row or all(c.strip() == "" for c in row):
                continue

            try:
                obs = cls._parse_row(
                    row=row,
                    header=header,
                    column_mapping=column_mapping,
                    ts_cols=ts_cols,
                    timestamp_format=timestamp_format,
                )
                if obs is not None:
                    observations.append(obs)
            except (ValueError, IndexError):
                continue

        return cls(observations)

    @staticmethod
    def _auto_detect_columns(header: List[str]) -> Dict[str, str]:
        """Auto-detect column mapping from header names."""
        # Field name → possible CSV column names (lowercase)
        patterns: Dict[str, List[str]] = {
            "wind_speed_ms": [
                "wind_speed", "ws", "wind_speed_ms", "windspeed",
                "wind speed", "speed", "ws_ms", "windspd",
            ],
            "wind_direction_deg": [
                "wind_direction", "wd", "wind_direction_deg", "winddir",
                "wind direction", "direction", "wd_deg", "dir",
            ],
            "temperature_k": [
                "temperature", "temp", "t", "temperature_k",
                "temp_k", "tk", "dry_bulb", "drybulb",
            ],
            "pressure_pa": [
                "pressure", "p", "pressure_pa", "pres", "barometric",
                "atm_pressure", "p_pa", "station_pressure",
            ],
            "cloud_cover_oktas": [
                "cloud_cover", "cloud", "cloud_cover_oktas", "cloudcover",
                "cloud_oktas", "clouds", "oktas",
            ],
            "solar_radiation_wm2": [
                "solar_radiation", "solar", "ghi", "solar_radiation_wm2",
                "radiation", "insolation", "solar_wm2", "sw_down",
            ],
            "relative_humidity_pct": [
                "humidity", "rh", "relative_humidity_pct", "relhum",
                "relative humidity", "rh_pct",
            ],
            "precipitation_mm": [
                "precipitation", "precip", "rain", "precipitation_mm",
                "rain_mm", "prcp", "rainfall",
            ],
            "station_id": [
                "station", "station_id", "stn", "wban",
            ],
            "stability_class": [
                "stability_class", "stability", "pasquill_gifford",
                "pg_class", "pg",
            ],
        }

        mapping: Dict[str, str] = {}
        for i, col in enumerate(header):
            col_lower = col.strip().lower()
            for field, candidates in patterns.items():
                if col_lower in candidates:
                    mapping[field] = col_lower
                    break

        return mapping

    @staticmethod
    def _find_timestamp_columns(header: List[str]) -> List[str]:
        """Find timestamp-related columns in header (exact matching)."""
        ts_patterns = [
            "timestamp", "datetime", "date_time", "date time",
            "time", "obs_time", "observation_time",
            "local_time", "utc", "gmt",
        ]
        result = []
        for col in header:
            col_lower = col.strip().lower()
            for pat in ts_patterns:
                if col_lower == pat:
                    result.append(col_lower)
                    break
        # If only 'time' was found, also look for 'date' to pair with it
        if result == ["time"]:
            for col in header:
                col_lower = col.strip().lower()
                if col_lower in ("date", "obs_date"):
                    result.insert(0, col_lower)
                    break
        return result

    @staticmethod
    def _parse_row(
        row: List[str],
        header: List[str],
        column_mapping: Dict[str, str],
        ts_cols: List[str],
        timestamp_format: str,
    ) -> Optional[WeatherObservation]:
        """Parse a single CSV row into a WeatherObservation."""
        # Build value dict keyed by header names
        values: Dict[str, str] = {}
        for i, val in enumerate(row):
            if i < len(header):
                values[header[i]] = val.strip()

        # Parse timestamp
        timestamp = None
        if ts_cols:
            # Concatenate timestamp columns
            parts = []
            for col in ts_cols:
                if col in values:
                    parts.append(values[col])
            if parts:
                ts_str = " ".join(p.strip() for p in parts).strip()
                try:
                    timestamp = datetime.strptime(ts_str, timestamp_format)
                except ValueError:
                    # Try ISO format
                    try:
                        timestamp = datetime.fromisoformat(ts_str)
                    except ValueError:
                        return None
        else:
            # Try ISO format on any plausible column
            for col_name, val in values.items():
                try:
                    timestamp = datetime.fromisoformat(val)
                    break
                except ValueError:
                    pass

        if timestamp is None:
            return None

        # Parse fields using mapping
        def get_float(field: str) -> Optional[float]:
            col_name = column_mapping.get(field)
            if col_name and col_name in values:
                try:
                    return float(values[col_name])
                except (ValueError, TypeError):
                    return None
            return None

        # Helper to get string value from mapping
        def get_str(field: str) -> Optional[str]:
            col_name = column_mapping.get(field)
            if col_name:
                return values.get(col_name, None) or None
            return None

        obs = WeatherObservation(
            timestamp=timestamp,
            wind_speed_ms=get_float("wind_speed_ms"),
            wind_direction_deg=get_float("wind_direction_deg"),
            temperature_k=get_float("temperature_k"),
            pressure_pa=get_float("pressure_pa"),
            cloud_cover_oktas=get_float("cloud_cover_oktas"),
            solar_radiation_wm2=get_float("solar_radiation_wm2"),
            relative_humidity_pct=get_float("relative_humidity_pct"),
            precipitation_mm=get_float("precipitation_mm"),
            stability_class=get_str("stability_class"),  # type: ignore[arg-type]
            station_id=get_str("station_id"),
        )

        return obs

    def to_dict(self) -> dict:
        """Serialize to dictionary (list of observation dicts)."""
        return {
            "observations": [obs.to_dict() for obs in self._observations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeatherDataset":
        """Deserialize from dictionary."""
        obs_list = [WeatherObservation.from_dict(od) for od in d.get("observations", [])]
        return cls(obs_list)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "WeatherDataset":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_csv(self) -> str:
        """Export dataset to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "timestamp",
            "wind_speed_ms",
            "wind_direction_deg",
            "temperature_k",
            "pressure_pa",
            "cloud_cover_oktas",
            "solar_radiation_wm2",
            "relative_humidity_pct",
            "precipitation_mm",
            "stability_class",
            "station_id",
            "is_daytime",
        ])

        for obs in self._observations:
            writer.writerow([
                obs.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                obs.wind_speed_ms if obs.wind_speed_ms is not None else "",
                obs.wind_direction_deg if obs.wind_direction_deg is not None else "",
                obs.temperature_k if obs.temperature_k is not None else "",
                obs.pressure_pa if obs.pressure_pa is not None else "",
                obs.cloud_cover_oktas if obs.cloud_cover_oktas is not None else "",
                obs.solar_radiation_wm2 if obs.solar_radiation_wm2 is not None else "",
                obs.relative_humidity_pct if obs.relative_humidity_pct is not None else "",
                obs.precipitation_mm if obs.precipitation_mm is not None else "",
                obs.stability_class or "",
                obs.station_id or "",
                obs.is_daytime,
            ])

        return output.getvalue()

    def __repr__(self) -> str:
        return f"WeatherDataset({len(self._observations)} observations)"
