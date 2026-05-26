"""
Rekarisk Terrain — Line-of-Sight Engine.

Ray-casting based line-of-sight calculator for thermal radiation blocking
and explosion pressure obstruction. Determines what fraction of a fire's
view factor is blocked by terrain obstacles.

Supports three source geometries:
    - Vertical cylinder (pool fire)
    - Tilted cylinder (jet fire)
    - Sphere (BLEVE fireball)

Ray intersection algorithms:
    - Ray-AABB (axis-aligned bounding box) — fast pre-filter
    - Ray-OBB (oriented bounding box) — exact intersection

References:
    - Glassner, A. (1989) — An Introduction to Ray Tracing
    - CCPS Guidelines for Evaluating Process Plant Buildings (2012)
    - TNO Yellow Book (CPR 14E), Chapter 5
    - HSE — GAMES model for building-affected thermal radiation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .obstacle import Obstacle


# ══════════════════════════════════════════════════════════════════════════════
# Enums & Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

class LOSStatus(Enum):
    """Line-of-sight status between a source and receptor."""
    CLEAR = "clear"          # No blocking — full view
    PARTIAL = "partial"      # Some rays blocked (source partially obscured)
    BLOCKED = "blocked"      # All rays blocked — no direct view
    SINGLE_BLOCKED = "single_blocked"  # Center ray blocked (fast single-ray check)


class SourceGeometry(Enum):
    """Geometry type for a thermal radiation source."""
    VERTICAL_CYLINDER = "vertical_cylinder"  # Pool fire
    TILTED_CYLINDER = "tilted_cylinder"      # Jet fire
    SPHERE = "sphere"                         # BLEVE fireball


@dataclass
class LOSResult:
    """Result of a line-of-sight check.

    Attributes:
        status: Clear, partial, or blocked.
        blocked_fraction: 0.0 (fully clear) to 1.0 (fully blocked)
            of the source's projected area/view factor.
        blocking_obstacles: List of obstacle IDs that block the ray.
        hit_distances: Distances along the ray to obstacle intersections.
        source_point: (x, y, z) of the source centre.
        target_point: (x, y, z) of the receptor.
    """
    status: LOSStatus = LOSStatus.CLEAR
    blocked_fraction: float = 0.0
    blocking_obstacles: List[str] = field(default_factory=list)
    hit_distances: List[float] = field(default_factory=list)
    source_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def is_clear(self) -> bool:
        return self.status == LOSStatus.CLEAR

    @property
    def is_blocked(self) -> bool:
        return self.status == LOSStatus.BLOCKED


# ══════════════════════════════════════════════════════════════════════════════
# Ray Intersection Primitives
# ══════════════════════════════════════════════════════════════════════════════

def ray_aabb_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    aabb: Tuple[float, float, float, float, float, float],
) -> Tuple[bool, float, float]:
    """Ray-AABB intersection test (Slab method).

    Tests whether a ray intersects an axis-aligned bounding box.
    Returns (hit, t_min, t_max) where t parameters are along the ray
    from origin.

    Args:
        origin: (3,) ray origin (x, y, z).
        direction: (3,) ray direction (should be normalized).
        aabb: (x_min, y_min, z_min, x_max, y_max, z_max).

    Returns:
        (hit, t_min, t_max):
            hit: True if ray intersects the AABB.
            t_min: Entry distance along ray (≥0 for forward hits).
            t_max: Exit distance along ray.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = aabb

    # Compute intersection t-values for each axis.
    # Handle rays parallel to an axis: if direction[i] ≈ 0, check if origin
    # is within the slab bounds; if not, no intersection (early exit).
    t_entries = []
    t_exits = []

    for i in range(3):
        origin_i = origin[i]
        dir_i = direction[i]
        slab_min = aabb[i]
        slab_max = aabb[3 + i]

        if abs(dir_i) > 1e-12:
            inv = 1.0 / dir_i
            t0 = (slab_min - origin_i) * inv
            t1 = (slab_max - origin_i) * inv
            t_entries.append(min(t0, t1))
            t_exits.append(max(t0, t1))
        else:
            # Ray is parallel to this axis slab.
            # If origin is outside the slab, no intersection possible.
            if origin_i < slab_min or origin_i > slab_max:
                return False, 0.0, 0.0
            # Otherwise the ray stays within this slab for all t.
            # Contribute -inf as entry and +inf as exit.
            t_entries.append(float('-inf'))
            t_exits.append(float('inf'))

    t_min = max(t_entries)
    t_max = min(t_exits)

    # Intersection if t_min <= t_max and t_max >= 0
    if t_min <= t_max and t_max >= 0.0:
        # Clamp t_min to valid range
        t_entry = max(t_min, 0.0)
        return True, t_entry, t_max
    return False, 0.0, 0.0


def ray_obb_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    obb_center: np.ndarray,
    obb_half_extents: np.ndarray,
    obb_axes: np.ndarray,
) -> Tuple[bool, float, float]:
    """Ray-OBB (oriented bounding box) intersection test.

    Uses the separating-axis theorem (SAT) for ray-box intersection.
    Transforms the ray into the OBB's local coordinate system.

    Args:
        origin: (3,) ray origin in world coords.
        direction: (3,) ray direction (normalized) in world coords.
        obb_center: (3,) OBB center in world coords.
        obb_half_extents: (3,) half-extents along each local axis.
        obb_axes: (3,3) orthonormal basis; rows are local axes in world coords.

    Returns:
        (hit, t_min, t_max):
            hit: True if ray intersects the OBB.
            t_min: Entry distance.
            t_max: Exit distance.
    """
    # Transform ray into OBB local space
    delta = origin - obb_center

    # Ray origin in local coords: dot with each axis
    local_origin = np.zeros(3)
    local_dir = np.zeros(3)
    for i in range(3):
        local_origin[i] = np.dot(delta, obb_axes[i])
        local_dir[i] = np.dot(direction, obb_axes[i])

    # Now do standard AABB test in local coords
    aabb = (
        -obb_half_extents[0], -obb_half_extents[1], -obb_half_extents[2],
         obb_half_extents[0],  obb_half_extents[1],  obb_half_extents[2],
    )

    return ray_aabb_intersection(local_origin, local_dir, aabb)


# ══════════════════════════════════════════════════════════════════════════════
# Obstacle OBB Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _obstacle_obb(
    obstacle: Obstacle,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute OBB parameters for an obstacle.

    Returns:
        (center, half_extents, axes) where:
            center: (3,) OBB center.
            half_extents: (3,) half-dimensions.
            axes: (3,3) rows = local axes in world coords.
    """
    center = np.array([obstacle.x, obstacle.y, obstacle.elevation + obstacle.height / 2.0])
    half_extents = np.array([
        obstacle.length / 2.0,
        obstacle.width / 2.0,
        obstacle.height / 2.0,
    ])

    theta = np.radians(obstacle.orientation)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Local axes in world coords
    # x-axis (length direction) — rotated from world x by theta
    axes = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,   0.0,  1.0],
    ])

    return center, half_extents, axes


# ══════════════════════════════════════════════════════════════════════════════
# LOSEngine
# ══════════════════════════════════════════════════════════════════════════════

class LOSEngine:
    """Line-of-sight calculator for thermal radiation blocking.

    Uses ray-casting from the source to determine which portion of
    the source's projected area is visible to a receptor, accounting
    for obstacles in the terrain.

    The engine samples multiple rays from the source geometry surface
    toward the receptor to compute the blocked fraction.
    """

    def __init__(self, obstacles: List[Obstacle] | None = None):
        """Initialize the LOS engine.

        Args:
            obstacles: Optional list of obstacles for the scene.
        """
        self._obstacles: List[Obstacle] = list(obstacles) if obstacles else []

    @property
    def obstacles(self) -> List[Obstacle]:
        return self._obstacles

    @obstacles.setter
    def obstacles(self, new_obstacles: List[Obstacle]) -> None:
        self._obstacles = list(new_obstacles)

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self._obstacles.append(obstacle)

    def remove_obstacle(self, obstacle_id: str) -> bool:
        for i, o in enumerate(self._obstacles):
            if o.id == obstacle_id:
                self._obstacles.pop(i)
                return True
        return False

    # -- Single ray check -----------------------------------------------------

    def check_single_ray(
        self,
        source_xyz: Tuple[float, float, float],
        target_xyz: Tuple[float, float, float],
    ) -> Tuple[bool, List[str], List[float]]:
        """Cast a single ray from source to target.

        Args:
            source_xyz: (x, y, z) source point [m].
            target_xyz: (x, y, z) receptor point [m].

        Returns:
            (blocked, obstacle_ids, hit_distances):
                blocked: True if any obstacle intersects the ray.
                obstacle_ids: List of blocking obstacle IDs.
                hit_distances: Distance from source to each hit.
        """
        origin = np.array(source_xyz, dtype=float)
        target = np.array(target_xyz, dtype=float)
        ray_vec = target - origin
        ray_length = np.linalg.norm(ray_vec)

        if ray_length < 1e-9:
            return False, [], []

        direction = ray_vec / ray_length

        blocked = False
        blocking_ids: List[str] = []
        hit_distances: List[float] = []

        for obstacle in self._obstacles:
            # Fast AABB check first
            aabb = obstacle.aabb()
            hit_aabb, t_entry, _ = ray_aabb_intersection(origin, direction, aabb)

            if not hit_aabb or t_entry > ray_length:
                continue

            # OBB check for exact intersection
            center, half_ext, axes = _obstacle_obb(obstacle)
            hit_obb, t_entry_obb, _ = ray_obb_intersection(
                origin, direction, center, half_ext, axes,
            )

            if hit_obb and 0.0 <= t_entry_obb <= ray_length:
                blocked = True
                blocking_ids.append(obstacle.id)
                hit_distances.append(t_entry_obb)

        # Sort hits by distance
        if blocking_ids:
            sorted_pairs = sorted(zip(hit_distances, blocking_ids))
            hit_distances = [p[0] for p in sorted_pairs]
            blocking_ids = [p[1] for p in sorted_pairs]

        return blocked, blocking_ids, hit_distances

    # -- Full LOS check -------------------------------------------------------

    def check_los(
        self,
        source_xyz: Tuple[float, float, float],
        target_xyz: Tuple[float, float, float],
        source_geometry: SourceGeometry = SourceGeometry.VERTICAL_CYLINDER,
        source_dims: Tuple[float, float, float] = (1.0, 1.0, 5.0),
        num_samples: int = 16,
    ) -> LOSResult:
        """Check line-of-sight from a source to a receptor.

        Samples multiple rays across the source geometry to determine
        what fraction is blocked by obstacles.

        Args:
            source_xyz: (x, y, z) source base center [m].
            target_xyz: (x, y, z) receptor point [m].
            source_geometry: Geometry type of the source.
            source_dims: Source dimensions:
                - Vertical cylinder: (diameter, _, height) [m]
                - Tilted cylinder: (diameter, _, length) [m]
                - Sphere: (diameter, _, _) [m]
            num_samples: Number of ray samples across the source surface.

        Returns:
            LOSResult with status, blocked fraction, and blocking obstacles.
        """
        if not self._obstacles:
            return LOSResult(
                status=LOSStatus.CLEAR,
                source_point=source_xyz,
                target_point=target_xyz,
            )

        # Generate sample points on the source geometry surface
        sample_points = self._generate_source_samples(
            source_xyz, source_geometry, source_dims, num_samples,
        )

        num_blocked = 0
        all_blocking_ids: List[str] = []
        all_hit_distances: List[float] = []

        for sp in sample_points:
            blocked, ids, dists = self.check_single_ray(
                (sp[0], sp[1], sp[2]), target_xyz,
            )
            if blocked:
                num_blocked += 1
                for oid in ids:
                    if oid not in all_blocking_ids:
                        all_blocking_ids.append(oid)
                all_hit_distances.extend(dists)

        fraction = num_blocked / max(num_samples, 1)

        if fraction < 1e-6:
            status = LOSStatus.CLEAR
        elif fraction >= 0.999:
            status = LOSStatus.BLOCKED
        elif fraction >= 1.0 / max(num_samples, 1):
            status = LOSStatus.PARTIAL
        else:
            status = LOSStatus.CLEAR

        return LOSResult(
            status=status,
            blocked_fraction=fraction,
            blocking_obstacles=all_blocking_ids,
            hit_distances=all_hit_distances,
            source_point=source_xyz,
            target_point=target_xyz,
        )

    # -- Blocked fraction -----------------------------------------------------

    def blocked_fraction(
        self,
        source_xyz: Tuple[float, float, float],
        target_xyz: Tuple[float, float, float],
        source_geometry: SourceGeometry = SourceGeometry.VERTICAL_CYLINDER,
        source_dims: Tuple[float, float, float] = (1.0, 1.0, 5.0),
        num_samples: int = 32,
    ) -> float:
        """Calculate the blocked fraction of the view factor.

        Args:
            source_xyz: Source base center (x, y, z) [m].
            target_xyz: Receptor point (x, y, z) [m].
            source_geometry: Source geometry type.
            source_dims: Source dimensions [m].
            num_samples: Number of ray samples.

        Returns:
            Blocked fraction: 0.0 (fully visible) to 1.0 (fully blocked).
        """
        result = self.check_los(
            source_xyz, target_xyz, source_geometry, source_dims, num_samples,
        )
        return result.blocked_fraction

    # -- Shadow zone computation ----------------------------------------------

    def shadow_zone(
        self,
        source_xyz: Tuple[float, float, float],
        source_geometry: SourceGeometry = SourceGeometry.VERTICAL_CYLINDER,
        source_dims: Tuple[float, float, float] = (1.0, 1.0, 5.0),
        grid_bounds: Tuple[float, float, float, float] = (-100, -100, 100, 100),
        grid_resolution: int = 50,
        receptor_height: float = 1.5,
    ) -> np.ndarray:
        """Compute a 2D shadow map showing blocked fraction at each grid cell.

        Args:
            source_xyz: Source base center (x, y, z) [m].
            source_geometry: Source geometry type.
            source_dims: Source dimensions [m].
            grid_bounds: (x_min, y_min, x_max, y_max) of the grid.
            grid_resolution: Grid cells per axis.
            receptor_height: Height of receptor above ground [m].

        Returns:
            (grid_resolution, grid_resolution) float array:
                blocked fraction at each cell, 0.0 = clear, 1.0 = blocked.
        """
        x_min, y_min, x_max, y_max = grid_bounds
        xs = np.linspace(x_min, x_max, grid_resolution)
        ys = np.linspace(y_min, y_max, grid_resolution)
        shadow_map = np.zeros((grid_resolution, grid_resolution))

        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                target = (x, y, receptor_height)
                fraction = self.blocked_fraction(
                    source_xyz, target, source_geometry, source_dims,
                    num_samples=8,  # lower samples for speed in grid computation
                )
                shadow_map[i, j] = fraction

        return shadow_map

    # -- Source sample generation ---------------------------------------------

    def _generate_source_samples(
        self,
        base_center: Tuple[float, float, float],
        geometry: SourceGeometry,
        dims: Tuple[float, float, float],
        num_samples: int,
    ) -> np.ndarray:
        """Generate sample points distributed on the source geometry surface.

        For a cylinder, we sample points on the visible surface area
        facing the receptor direction. This gives a good approximation
        of the projected area blocking.

        Args:
            base_center: (x, y, z) of the base center.
            geometry: Source geometry type.
            dims: Dimensions [m].
            num_samples: Minimum number of samples.

        Returns:
            (N, 3) numpy array of sample points.
        """
        bx, by, bz = base_center

        if geometry == SourceGeometry.VERTICAL_CYLINDER:
            return self._samples_vertical_cylinder(
                bx, by, bz, dims, num_samples,
            )
        elif geometry == SourceGeometry.TILTED_CYLINDER:
            return self._samples_tilted_cylinder(
                bx, by, bz, dims, num_samples,
            )
        elif geometry == SourceGeometry.SPHERE:
            return self._samples_sphere(
                bx, by, bz, dims, num_samples,
            )
        else:
            # Default: point source
            return np.array([[bx, by, bz]])

    def _samples_vertical_cylinder(
        self,
        bx: float, by: float, bz: float,
        dims: Tuple[float, float, float],
        num_samples: int,
    ) -> np.ndarray:
        """Sample points on a vertical cylinder surface.

        Distributes samples evenly around the circumference and
        along the height.

        Args:
            bx, by, bz: Base center (bottom center).
            dims: (diameter, _, height).
            num_samples: Approximately this many samples.

        Returns:
            (N, 3) array of sample points.
        """
        diameter, _, height = dims
        radius = diameter / 2.0

        # Balance angular and vertical samples
        n_angular = max(4, int(math.sqrt(num_samples)))
        n_height = max(2, num_samples // n_angular)
        n_height = max(2, n_height)

        angles = np.linspace(0, 2 * math.pi, n_angular, endpoint=False)
        heights = np.linspace(
            bz + height * 0.1,  # slightly above base
            bz + height * 0.9,  # slightly below top
            n_height,
        )

        samples = []
        for h in heights:
            for a in angles:
                sx = bx + radius * math.cos(a)
                sy = by + radius * math.sin(a)
                sz = h
                samples.append([sx, sy, sz])

        return np.array(samples)

    def _samples_tilted_cylinder(
        self,
        bx: float, by: float, bz: float,
        dims: Tuple[float, float, float],
        num_samples: int,
    ) -> np.ndarray:
        """Sample points on a tilted cylinder (jet fire).

        The cylinder is assumed tilted at ~45° from horizontal.
        dims: (diameter, _, length).

        We sample along the length and around the circumference.
        """
        diameter, _, length = dims
        radius = diameter / 2.0

        n_length = max(2, int(math.sqrt(num_samples)))
        n_angular = max(4, num_samples // n_length)
        n_length = max(2, n_length)

        # Assume tilt away from receptor (worst case — in real use,
        # the caller should rotate the source appropriately)
        # Default: tilted 45° from vertical, away from the -y direction
        tilt_angle = math.radians(45.0)  # from vertical

        fracs = np.linspace(0.1, 0.9, n_length)
        angles_around = np.linspace(0, 2 * math.pi, n_angular, endpoint=False)

        samples = []
        for f in fracs:
            # Center of this cross-section along the cylinder axis
            dist_along = f * length
            # Horizontal component (length along ground)
            horiz = dist_along * math.sin(tilt_angle)
            # Height above base
            h = bz + dist_along * math.cos(tilt_angle)

            # Center at this segment (offset in -y direction for tilt)
            cx = bx
            cy = by - horiz  # tilted toward -y

            for a in angles_around:
                sx = cx + radius * math.cos(a)
                sy = cy + radius * math.sin(a)
                sz = h
                samples.append([sx, sy, sz])

        return np.array(samples)

    def _samples_sphere(
        self,
        bx: float, by: float, bz: float,
        dims: Tuple[float, float, float],
        num_samples: int,
    ) -> np.ndarray:
        """Sample points on a sphere surface (BLEVE fireball).

        Uses Fibonacci sphere for uniform distribution.

        Args:
            bx, by, bz: Sphere center.
            dims: (diameter, _, _).
            num_samples: Approximately this many samples.
        """
        diameter, _, _ = dims
        radius = diameter / 2.0

        # Fibonacci sphere — uniform distribution
        phi = math.pi * (3.0 - math.sqrt(5.0))

        samples = []
        for i in range(num_samples):
            y = 1.0 - (i / float(num_samples - 1)) * 2.0  # -1 to 1
            radius_at_y = math.sqrt(1.0 - y * y)
            theta = phi * i

            sx = bx + radius * math.cos(theta) * radius_at_y
            sy = by + radius * math.sin(theta) * radius_at_y
            sz = bz + radius * y
            samples.append([sx, sy, sz])

        return np.array(samples)
