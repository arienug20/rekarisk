"""
Rekarisk Terrain — Obstacle Modeling.

Defines obstacle dataclasses for buildings, tanks, pipe racks, walls,
and custom geometries. Supports JSON import/export, spatial queries,
and bounding box calculations for line-of-sight and explosion analysis.

References:
    - CCPS Guidelines for Consequence Analysis of Chemical Releases (1999)
    - TNO Yellow Book (CPR 14E), Chapter 6 — Dispersion in Built-up Areas
    - Defra/HSE — Obstacle modelling for CFD and integral models
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Obstacle Types
# ══════════════════════════════════════════════════════════════════════════════

OBSTACLE_TYPES = ("building", "tank", "pipe_rack", "wall", "custom")

OBSTACLE_TYPE_LABELS = {
    "building": "Building",
    "tank": "Storage Tank",
    "pipe_rack": "Pipe Rack",
    "wall": "Wall / Blast Barrier",
    "custom": "Custom",
}

# Typical material acentric factors for explosion confinement
# (higher = more confinement / less venting)
TYPE_CONFINEMENT = {
    "building": 1.0,
    "tank": 0.6,
    "pipe_rack": 0.2,
    "wall": 0.8,
    "custom": 0.5,
}


# ══════════════════════════════════════════════════════════════════════════════
# Obstacle Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Obstacle:
    """A single obstacle in the terrain.

    Attributes:
        id: Unique identifier (UUID string).
        name: Human-readable name.
        type: Type — 'building', 'tank', 'pipe_rack', 'wall', 'custom'.
        position: (x, y) center coordinates [m].
        dimensions: (length, width, height) [m]. Length is along x-axis
            when orientation=0, width along y-axis.
        orientation: Rotation angle from north (y-axis) [deg], clockwise.
        porosity: 0.0 (solid) to 1.0 (fully porous). Affects explosion
            confinement calculations.
        elevation: Base elevation above local grade [m]. Default 0.0.
        metadata: Optional dict for user-defined properties.
    """
    name: str
    type: str = "building"
    position: Tuple[float, float] = (0.0, 0.0)
    dimensions: Tuple[float, float, float] = (10.0, 10.0, 5.0)
    orientation: float = 0.0
    porosity: float = 0.0
    elevation: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in OBSTACLE_TYPES:
            raise ValueError(
                f"Invalid obstacle type '{self.type}'. "
                f"Must be one of {OBSTACLE_TYPES}"
            )
        p0, p1 = self.position
        self.position = (float(p0), float(p1))
        d0, d1, d2 = self.dimensions
        self.dimensions = (float(d0), float(d1), float(d2))
        self.porosity = max(0.0, min(1.0, float(self.porosity)))

    # -- Convenience properties -----------------------------------------------

    @property
    def x(self) -> float:
        """Center x-coordinate [m]."""
        return self.position[0]

    @property
    def y(self) -> float:
        """Center y-coordinate [m]."""
        return self.position[1]

    @property
    def length(self) -> float:
        """Length (along local x when orientation=0) [m]."""
        return self.dimensions[0]

    @property
    def width(self) -> float:
        """Width (along local y when orientation=0) [m]."""
        return self.dimensions[1]

    @property
    def height(self) -> float:
        """Height above base elevation [m]."""
        return self.dimensions[2]

    @property
    def top_elevation(self) -> float:
        """Top elevation [m] (base elevation + height)."""
        return self.elevation + self.height

    @property
    def footprint_area(self) -> float:
        """Footprint area [m²]."""
        return self.length * self.width

    @property
    def volume(self) -> float:
        """Volume [m³]."""
        return self.length * self.width * self.height

    @property
    def confinement_factor(self) -> float:
        """Effective confinement factor (type factor × (1 - porosity))."""
        base = TYPE_CONFINEMENT.get(self.type, 0.5)
        return base * (1.0 - self.porosity)

    @property
    def label(self) -> str:
        """Human-readable type label."""
        return OBSTACLE_TYPE_LABELS.get(self.type, "Unknown")

    # -- Bounding box ---------------------------------------------------------

    def bounding_box(self) -> np.ndarray:
        """Return the 8 corner points of the oriented bounding box.

        Returns:
            (8, 3) numpy array of (x, y, z) coordinates.
            z ranges from elevation to top_elevation.
        """
        l2 = self.length / 2.0
        w2 = self.width / 2.0
        z_bot = self.elevation
        z_top = self.top_elevation

        # Local corners (centered at origin, before rotation)
        local_corners = np.array([
            [-l2, -w2],
            [ l2, -w2],
            [ l2,  w2],
            [-l2,  w2],
        ])

        # Rotation matrix (clockwise from north = positive y-axis)
        theta = np.radians(self.orientation)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Rotate and translate
        rotated = local_corners @ R.T
        translated = rotated + np.array([self.x, self.y])

        # Add z layers
        corners = np.zeros((8, 3))
        corners[0:4, 0:2] = translated
        corners[0:4, 2] = z_bot
        corners[4:8, 0:2] = translated
        corners[4:8, 2] = z_top

        return corners

    def aabb(self) -> Tuple[float, float, float, float, float, float]:
        """Get the axis-aligned bounding box (AABB).

        Returns:
            (x_min, y_min, z_min, x_max, y_max, z_max).
        """
        corners = self.bounding_box()
        return (
            float(corners[:, 0].min()),
            float(corners[:, 1].min()),
            float(corners[:, 2].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].max()),
            float(corners[:, 2].max()),
        )

    def corners_2d(self) -> np.ndarray:
        """Return the 4 ground-level corners (x, y) of the obstacle footprint.

        Returns:
            (4, 2) numpy array.
        """
        corners = self.bounding_box()
        return corners[0:4, 0:2]

    # -- Point containment ----------------------------------------------------

    def contains_point(
        self, x: float, y: float, z: float | None = None
    ) -> bool:
        """Check if a 2D or 3D point is inside the obstacle.

        For 2D check (z=None): only checks the footprint.
        For 3D check: checks footprint AND vertical extent.

        Args:
            x: X-coordinate [m].
            y: Y-coordinate [m].
            z: Z-coordinate [m] (optional).

        Returns:
            True if point is inside the obstacle.
        """
        # Translate to obstacle-local coordinates
        dx = x - self.x
        dy = y - self.y

        # Rotate to obstacle-local axes (counter-clockwise to undo orientation)
        theta = np.radians(-self.orientation)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        local_x = dx * cos_t - dy * sin_t
        local_y = dx * sin_t + dy * cos_t

        # Check within half-dimensions
        l2 = self.length / 2.0
        w2 = self.width / 2.0
        in_2d = abs(local_x) <= l2 and abs(local_y) <= w2

        if z is None:
            return in_2d
        else:
            return in_2d and self.elevation <= z <= self.top_elevation

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "position": list(self.position),
            "dimensions": list(self.dimensions),
            "orientation": self.orientation,
            "porosity": self.porosity,
            "elevation": self.elevation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> Obstacle:
        """Deserialize from dictionary."""
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            name=d["name"],
            type=d.get("type", "building"),
            position=tuple(d.get("position", [0.0, 0.0])),
            dimensions=tuple(d.get("dimensions", [10.0, 10.0, 5.0])),
            orientation=d.get("orientation", 0.0),
            porosity=d.get("porosity", 0.0),
            elevation=d.get("elevation", 0.0),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"Obstacle(id={self.id[:8]}..., name='{self.name}', "
            f"type={self.type}, pos=({self.x:.1f},{self.y:.1f}), "
            f"dim=({self.length:.1f},{self.width:.1f},{self.height:.1f}))"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Obstacle Collection
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ObstacleCollection:
    """A collection of obstacles with spatial query methods.

    Manages a list of obstacles and provides methods for:
    - Adding, removing, and querying obstacles
    - Finding obstacles by type, area, or spatial region
    - Checking point containment against all obstacles
    - Export/import to/from JSON
    """
    obstacles: List[Obstacle] = field(default_factory=list)

    # -- Basic collection operations ------------------------------------------

    def add(self, obstacle: Obstacle) -> None:
        """Add an obstacle to the collection."""
        # Check for duplicate ID
        if any(o.id == obstacle.id for o in self.obstacles):
            raise ValueError(f"Obstacle with id '{obstacle.id}' already exists")
        self.obstacles.append(obstacle)

    def remove(self, obstacle_id: str) -> bool:
        """Remove an obstacle by ID. Returns True if found and removed."""
        for i, o in enumerate(self.obstacles):
            if o.id == obstacle_id:
                self.obstacles.pop(i)
                return True
        return False

    def remove_by_name(self, name: str) -> int:
        """Remove all obstacles with the given name. Returns number removed."""
        before = len(self.obstacles)
        self.obstacles = [o for o in self.obstacles if o.name != name]
        return before - len(self.obstacles)

    def get(self, obstacle_id: str) -> Obstacle | None:
        """Get an obstacle by ID, or None if not found."""
        for o in self.obstacles:
            if o.id == obstacle_id:
                return o
        return None

    def clear(self) -> None:
        """Remove all obstacles."""
        self.obstacles.clear()

    # -- Query operations -----------------------------------------------------

    def by_type(self, obstacle_type: str) -> List[Obstacle]:
        """Return all obstacles of a given type."""
        return [o for o in self.obstacles if o.type == obstacle_type]

    def by_name(self, name: str) -> List[Obstacle]:
        """Return all obstacles with a given name (case-insensitive)."""
        n = name.lower()
        return [o for o in self.obstacles if o.name.lower() == n]

    def within_bounds(
        self,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
    ) -> List[Obstacle]:
        """Return obstacles whose center falls within the given bounds."""
        return [
            o for o in self.obstacles
            if x_min <= o.x <= x_max and y_min <= o.y <= y_max
        ]

    def intersecting_bounds(
        self,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
    ) -> List[Obstacle]:
        """Return obstacles whose bounding box intersects the given bounds."""
        result = []
        for o in self.obstacles:
            aabb = o.aabb()
            # AABB-AABB intersection test
            if (
                aabb[0] < x_max and aabb[3] > x_min and
                aabb[1] < y_max and aabb[4] > y_min
            ):
                result.append(o)
        return result

    def obstacle_at_point(
        self, x: float, y: float, z: float | None = None
    ) -> Obstacle | None:
        """Return the first obstacle containing the point, or None.

        If multiple obstacles overlap, returns the first found.
        """
        for o in self.obstacles:
            if o.contains_point(x, y, z):
                return o
        return None

    def all_obstacles_at_point(
        self, x: float, y: float, z: float | None = None
    ) -> List[Obstacle]:
        """Return all obstacles containing the point."""
        return [o for o in self.obstacles if o.contains_point(x, y, z)]

    # -- Aggregate properties -------------------------------------------------

    def total_bounding_box(
        self,
    ) -> Tuple[float, float, float, float, float, float] | None:
        """Compute the combined AABB of all obstacles.

        Returns:
            (x_min, y_min, z_min, x_max, y_max, z_max) or None if empty.
        """
        if not self.obstacles:
            return None

        x_min = float("inf")
        y_min = float("inf")
        z_min = float("inf")
        x_max = float("-inf")
        y_max = float("-inf")
        z_max = float("-inf")

        for o in self.obstacles:
            aabb = o.aabb()
            x_min = min(x_min, aabb[0])
            y_min = min(y_min, aabb[1])
            z_min = min(z_min, aabb[2])
            x_max = max(x_max, aabb[3])
            y_max = max(y_max, aabb[4])
            z_max = max(z_max, aabb[5])

        return (x_min, y_min, z_min, x_max, y_max, z_max)

    def type_summary(self) -> Dict[str, int]:
        """Return a count summary by obstacle type."""
        from collections import Counter
        return dict(Counter(o.type for o in self.obstacles))

    def total_footprint_area(self) -> float:
        """Sum of all obstacle footprint areas [m²]."""
        return sum(o.footprint_area for o in self.obstacles)

    def average_height(self) -> float:
        """Average obstacle height [m]. Returns 0 if empty."""
        if not self.obstacles:
            return 0.0
        return sum(o.height for o in self.obstacles) / len(self.obstacles)

    def max_height(self) -> float:
        """Maximum obstacle height [m]. Returns 0 if empty."""
        if not self.obstacles:
            return 0.0
        return max(o.height for o in self.obstacles)

    # -- Iteration ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.obstacles)

    def __iter__(self) -> Iterator[Obstacle]:
        return iter(self.obstacles)

    def __getitem__(self, idx: int) -> Obstacle:
        return self.obstacles[idx]

    def __contains__(self, obstacle: Obstacle) -> bool:
        return any(o.id == obstacle.id for o in self.obstacles)

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "obstacles": [o.to_dict() for o in self.obstacles],
            "version": "1.0",
        }

    def to_json(self, path: str | Path) -> None:
        """Export collection to a JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict) -> ObstacleCollection:
        """Deserialize from dictionary."""
        obstacles = [Obstacle.from_dict(od) for od in d.get("obstacles", [])]
        return cls(obstacles=obstacles)

    @classmethod
    def from_json(cls, path: str | Path) -> ObstacleCollection:
        """Import collection from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def __repr__(self) -> str:
        return (
            f"ObstacleCollection(n={len(self.obstacles)}, "
            f"types={list(self.type_summary().keys())})"
        )
